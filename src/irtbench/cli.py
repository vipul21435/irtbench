"""Command line interface: ``irtbench fit``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from irtbench import __version__
from irtbench.data import ResponseMatrix, read_matrix
from irtbench.mml import (
    ALL_CORRECT,
    ALL_INCORRECT,
    KINDS,
    NO_DATA,
    AbilityEstimates,
    IRTFit,
    estimate_abilities,
    fit_mml,
)


def _number(value: float, digits: int = 2) -> str:
    if np.isnan(value):
        return ""
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}f}"


def _table(header: list[str], rows: list[list[str]], text_columns: int = 1) -> list[str]:
    """A Markdown table whose first ``text_columns`` columns are left-aligned, the rest right."""
    rule = ["---"] * text_columns + ["---:"] * (len(header) - text_columns)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(rule) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _shown(total: int, limit: int) -> int:
    return total if limit <= 0 else min(limit, total)


def render_fit(
    name: str, data: ResponseMatrix, fit: IRTFit, abilities: AbilityEstimates, rows: int = 10
) -> str:
    """Markdown summary of a fit: counts, the hardest items and the strongest models."""
    cells = data.n_models * data.n_items
    counts = {state: fit.status.count(state) for state in (ALL_CORRECT, ALL_INCORRECT, NO_DATA)}
    ending = (
        f"EM converged in {fit.iterations} iterations"
        if fit.converged
        else f"EM stopped at the limit of {fit.iterations} iterations without converging"
    )
    lines = [
        f"# irtbench fit: {name}",
        "",
        f"- data: {data.n_models} models x {data.n_items} items, "
        f"{data.n_missing} of {cells} cells missing",
        f"- model: {fit.kind} by marginal maximum likelihood "
        f"(EM on a {fit.quadrature} grid of {fit.n_nodes} nodes)",
        f"- log-likelihood: {fit.log_likelihood:.3f} with {fit.n_parameters} parameters; {ending}",
        f"- items left out of the fit: {counts[ALL_CORRECT]} solved by all, "
        f"{counts[ALL_INCORRECT]} solved by none, {counts[NO_DATA]} without data",
    ]

    difficulty = fit.difficulty
    solved = data.item_means()
    # hardest first; items without data (NaN difficulty) go last
    without_data = np.isnan(difficulty)
    order = sorted(
        range(data.n_items),
        key=lambda j: (bool(without_data[j]), 0.0 if without_data[j] else -difficulty[j]),
    )
    count = _shown(data.n_items, rows)
    lines += ["", f"## Items (hardest first, {count} of {data.n_items})", ""]
    lines += _table(
        ["item", "status", "solved", "discrimination", "difficulty"],
        [
            [
                data.items[j],
                fit.status[j],
                _number(solved[j]),
                _number(fit.discrimination[j]),
                _number(difficulty[j]),
            ]
            for j in order[:count]
        ],
        text_columns=2,
    )

    share = data.model_means()
    ranking = sorted(range(data.n_models), key=lambda i: -abilities.theta[i])
    count = _shown(data.n_models, rows)
    lines += ["", f"## Models (strongest first, {count} of {data.n_models})", ""]
    lines += _table(
        ["model", "answered", "solved", "ability", "sd"],
        [
            [
                data.models[i],
                str(int(data.observed[i].sum())),
                _number(share[i]),
                _number(abilities.theta[i]),
                _number(abilities.sd[i]),
            ]
            for i in ranking[:count]
        ],
    )
    return "\n".join(lines) + "\n"


def _cmd_fit(args: argparse.Namespace) -> int:
    data = read_matrix(
        args.input,
        args.format,
        model_col=args.model_col,
        item_col=args.item_col,
        score_col=args.score_col,
    )
    fit = fit_mml(
        data,
        args.irt_model,
        n_quad=args.quad,
        resolution=args.resolution,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    abilities = estimate_abilities(fit, data)
    sys.stdout.write(render_fit(Path(args.input).name, data, fit, abilities, rows=args.rows))
    if fit.resolution > 1.0:
        print(
            f"irtbench: warning: the quadrature grid is {fit.resolution:.1f} times too coarse "
            "for the narrowest posterior or the steepest item",
            file=sys.stderr,
        )
    if not fit.converged:
        print(
            f"irtbench: warning: EM did not converge in {fit.iterations} iterations; "
            "the estimates may still be moving",
            file=sys.stderr,
        )
    if args.json:
        share = data.model_means()
        models = abilities.to_dict()
        for index, row in enumerate(models):
            row["solved"] = None if np.isnan(share[index]) else float(share[index])
        payload = {
            "irtbench": __version__,
            "data": {
                "n_models": data.n_models,
                "n_items": data.n_items,
                "n_missing": data.n_missing,
            },
            "fit": fit.to_dict(),
            "models": models,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="irtbench",
        description="Item response theory for a models-by-items matrix of benchmark results.",
    )
    parser.add_argument("--version", action="version", version=f"irtbench {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    fit = commands.add_parser(
        "fit",
        help="fit a 1PL or 2PL model and print item parameters and model abilities",
        description="Fit a 1PL (Rasch) or 2PL model by marginal maximum likelihood.",
    )
    fit.add_argument("input", help="results file: long CSV/TSV/JSONL or a wide CSV/TSV matrix")
    fit.add_argument(
        "--format",
        choices=("auto", "long", "wide"),
        default="auto",
        help="layout of the input (default: auto)",
    )
    fit.add_argument("--model-col", default="model", help="long format: model column")
    fit.add_argument("--item-col", default="item", help="long format: item column")
    fit.add_argument("--score-col", default="score", help="long format: 0/1 score column")
    fit.add_argument("--irt-model", choices=KINDS, default="2pl", help="default: 2pl")
    fit.add_argument(
        "--quad", type=int, default=61, help="nodes of the starting Gauss-Hermite rule"
    )
    fit.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="largest node gap in units of the narrowest posterior width (default: 1.0)",
    )
    fit.add_argument("--max-iter", type=int, default=2000, help="EM iteration limit")
    fit.add_argument("--tol", type=float, default=1e-6, help="EM stops below this parameter change")
    fit.add_argument("--rows", type=int, default=10, help="rows per printed table; 0 prints all")
    fit.add_argument("--json", metavar="PATH", help="also write the full result as JSON")
    fit.set_defaults(handler=_cmd_fit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, OSError) as error:
        print(f"irtbench: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
