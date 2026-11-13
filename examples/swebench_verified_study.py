"""Item response analysis of public SWE-bench Verified submissions.

Build the matrix first (nothing from the source repository is stored in this one):

    python scripts/fetch_swebench_verified.py DATA_DIR
    python examples/swebench_verified_study.py DATA_DIR/swebench_verified.csv --json study.json

Prints the numbers quoted in the README and writes them as JSON. Takes a few minutes.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from irtbench import fit_mml, read_matrix
from irtbench.compare import cross_validate, information_criteria, likelihood_ratio
from irtbench.diagnostics import (
    NEGATIVE_DISCRIMINATION,
    NEGATIVE_REST_CORRELATION,
    diagnose_items,
)
from irtbench.subset import evaluate_subsets

QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("matrix", help="swebench_verified.csv written by the fetch script")
    parser.add_argument("--json", help="write the numbers to this file")
    parser.add_argument("--splits", type=int, default=20, help="train/test splits of the models")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = time.time()

    data = read_matrix(args.matrix)
    shares = data.item_means()
    scores = data.model_means()
    out: dict[str, object] = {
        "n_models": data.n_models,
        "n_items": data.n_items,
        "n_missing": data.n_missing,
        "solved_by_all": int((shares == 1.0).sum()),
        "solved_by_none": int((shares == 0.0).sum()),
        "score_min": float(scores.min()),
        "score_median": float(np.median(scores)),
        "score_max": float(scores.max()),
    }
    print(f"{data.n_models} submissions x {data.n_items} items, {data.n_missing} missing cells")
    print(f"solved by all: {out['solved_by_all']}, solved by none: {out['solved_by_none']}")

    one = fit_mml(data, "1pl")
    two = fit_mml(data, "2pl")
    slopes = two.discrimination[two.estimated]
    report = diagnose_items(data, two, rasch=one)
    negative = report.flagged(NEGATIVE_DISCRIMINATION)
    backwards = report.flagged(NEGATIVE_REST_CORRELATION)
    common = float(one.discrimination[one.estimated][0])
    out["fit_1pl"] = {**information_criteria(one), "slope": common}
    out["fit_2pl"] = {
        **information_criteria(two),
        "slope_prior_sd": two.slope_prior_sd,
        "common_slope": two.slope_prior_mean,
        "n_nodes": two.n_nodes,
        "converged": two.converged,
    }
    out["discrimination"] = {
        "mean": float(slopes.mean()),
        "sd": float(slopes.std()),
        "quantiles": {str(q): float(np.quantile(slopes, q)) for q in QUANTILES},
        "below_half_of_median": int((slopes < 0.5 * np.median(slopes)).sum()),
    }
    out["negative_discrimination"] = [
        {
            "item": item,
            "discrimination": float(two.discrimination[data.items.index(item)]),
            "solved": float(shares[data.items.index(item)]),
            "rest_correlation": float(report.rest_correlation[data.items.index(item)]),
        }
        for item in negative
    ]
    out["negative_rest_correlation"] = list(backwards)
    out["prior_sensitivity"] = {}
    for sd in (0.25, 1.0):
        other = fit_mml(data, "2pl", slope_prior_sd=sd)
        out["prior_sensitivity"][str(sd)] = {
            "negative": int((other.discrimination[other.estimated] < 0).sum()),
            "log_likelihood": other.log_likelihood,
            "max_slope": float(np.nanmax(other.discrimination)),
        }
    out["likelihood_ratio"] = likelihood_ratio(one, two)
    out["misfit"] = {
        "infit_above_1.3": int((report.infit > 1.3).sum()),
        "outfit_above_2": int((report.outfit > 2.0).sum()),
    }
    print(f"1PL logL {one.log_likelihood:.1f}, 2PL logL {two.log_likelihood:.1f}")
    print("discrimination quantiles:", out["discrimination"]["quantiles"])
    print(f"negative discrimination: {len(negative)} items: {', '.join(negative[:8])}")
    print(f"negative rest correlation: {len(backwards)} items")

    out["held_out_cells"] = cross_validate(data, folds=5, seed=args.seed)
    for kind, row in out["held_out_cells"].items():
        print(f"held-out {kind}: logL/cell {row['log_likelihood']:.4f}, AUC {row['auc']:.4f}")

    out["subsets"] = evaluate_subsets(data, (25, 50, 100), splits=args.splits, seed=args.seed)
    for size, row in out["subsets"]["sizes"].items():
        cells = ", ".join(f"{name} {value['mae']:.4f}" for name, value in row.items())
        print(f"k={size}: MAE {cells}")

    out["seconds"] = round(time.time() - started, 1)
    print(f"done in {out['seconds']} s")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
