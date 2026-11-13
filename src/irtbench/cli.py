"""Command line interface: ``irtbench fit | items | subset | predict | report``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from irtbench import __version__
from irtbench.compare import cross_validate, information_criteria, likelihood_ratio
from irtbench.data import ResponseMatrix, read_matrix
from irtbench.diagnostics import (
    NEGATIVE_DISCRIMINATION,
    NEGATIVE_REST_CORRELATION,
    SOLVED_BY_ALL,
    SOLVED_BY_NONE,
    ItemDiagnostics,
    diagnose_items,
)
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
from irtbench.se import bootstrap_standard_errors, standard_errors
from irtbench.subset import METHODS, evaluate_subsets, predict_scores, select_items


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


def _load(args: argparse.Namespace) -> ResponseMatrix:
    return read_matrix(
        args.input,
        args.format,
        model_col=args.model_col,
        item_col=args.item_col,
        score_col=args.score_col,
    )


def _fit(data: ResponseMatrix, args: argparse.Namespace, kind: str | None = None) -> IRTFit:
    fit = fit_mml(
        data,
        kind or args.irt_model,
        slope_prior_sd=args.slope_prior_sd if args.slope_prior_sd > 0 else None,
        n_quad=args.quad,
        resolution=args.resolution,
        max_iter=args.max_iter,
        tol=args.tol,
    )
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
    return fit


def _fit_lines(data: ResponseMatrix, fit: IRTFit) -> list[str]:
    cells = data.n_models * data.n_items
    counts = {state: fit.status.count(state) for state in (ALL_CORRECT, ALL_INCORRECT, NO_DATA)}
    ending = (
        f"EM converged in {fit.iterations} iterations"
        if fit.converged
        else f"EM stopped at the limit of {fit.iterations} iterations without converging"
    )
    prior = (
        ""
        if fit.slope_prior_sd is None
        else f", slope prior sd {fit.slope_prior_sd:g} around a common slope of "
        f"{fit.slope_prior_mean:.2f}"
    )
    return [
        f"- data: {data.n_models} models x {data.n_items} items, "
        f"{data.n_missing} of {cells} cells missing",
        f"- model: {fit.kind} by marginal maximum likelihood "
        f"(EM on a {fit.quadrature} grid of {fit.n_nodes} nodes{prior})",
        f"- log-likelihood: {fit.log_likelihood:.3f} with {fit.n_parameters} parameters; {ending}",
        f"- items left out of the fit: {counts[ALL_CORRECT]} solved by all, "
        f"{counts[ALL_INCORRECT]} solved by none, {counts[NO_DATA]} without data",
    ]


def render_fit(
    name: str, data: ResponseMatrix, fit: IRTFit, abilities: AbilityEstimates, rows: int = 10
) -> str:
    """Markdown summary of a fit: counts, the hardest items and the strongest models."""
    lines = [f"# irtbench fit: {name}", "", *_fit_lines(data, fit)]
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
    lines += ["", *_model_table(data, abilities, None, rows)]
    return "\n".join(lines) + "\n"


def _model_table(
    data: ResponseMatrix, abilities: AbilityEstimates, predicted: np.ndarray | None, rows: int
) -> list[str]:
    share = data.model_means()
    ranking = sorted(range(data.n_models), key=lambda i: -abilities.theta[i])
    count = _shown(data.n_models, rows)
    header = ["model", "answered", "solved", "ability", "sd"]
    if predicted is not None:
        header.append("predicted score")
    body = []
    for i in ranking[:count]:
        row = [
            data.models[i],
            str(int(data.observed[i].sum())),
            _number(share[i]),
            _number(abilities.theta[i]),
            _number(abilities.sd[i]),
        ]
        if predicted is not None:
            row.append(_number(predicted[i], 3))
        body.append(row)
    return [f"## Models (strongest first, {count} of {data.n_models})", "", *_table(header, body)]


def _models_json(
    data: ResponseMatrix, abilities: AbilityEstimates, predicted: np.ndarray | None = None
) -> list[dict[str, object]]:
    share = data.model_means()
    rows = abilities.to_dict()
    for index, row in enumerate(rows):
        row["solved"] = None if np.isnan(share[index]) else float(share[index])
        if predicted is not None:
            row["predicted_score"] = float(predicted[index])
    return rows


def _json_ready(value: object) -> object:
    """Replace numbers JSON cannot hold (NaN, infinities) by ``None``, at any depth."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, float | np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_json(path: str | None, payload: dict[str, object]) -> None:
    if path:
        Path(path).write_text(json.dumps(_json_ready(payload), indent=2, allow_nan=False) + "\n")


def _data_json(data: ResponseMatrix) -> dict[str, int]:
    return {"n_models": data.n_models, "n_items": data.n_items, "n_missing": data.n_missing}


def _cmd_fit(args: argparse.Namespace) -> int:
    data = _load(args)
    fit = _fit(data, args)
    abilities = estimate_abilities(fit, data)
    sys.stdout.write(render_fit(Path(args.input).name, data, fit, abilities, rows=args.rows))
    _write_json(
        args.json,
        {
            "irtbench": __version__,
            "data": _data_json(data),
            "fit": fit.to_dict(),
            "models": _models_json(data, abilities),
        },
    )
    return 0


def _item_lines(report: ItemDiagnostics, rows: int) -> list[str]:
    counts = {
        flag: len(report.flagged(flag))
        for flag in (
            SOLVED_BY_ALL,
            SOLVED_BY_NONE,
            NEGATIVE_DISCRIMINATION,
            NEGATIVE_REST_CORRELATION,
        )
    }
    lines = [
        f"- solved by all: {counts[SOLVED_BY_ALL]}; solved by none: {counts[SOLVED_BY_NONE]}",
        f"- negative discrimination: {counts[NEGATIVE_DISCRIMINATION]}; "
        f"negative correlation with the rest score: {counts[NEGATIVE_REST_CORRELATION]}",
    ]
    suspects = [
        j
        for j, flags in enumerate(report.flags)
        if NEGATIVE_DISCRIMINATION in flags or NEGATIVE_REST_CORRELATION in flags
    ]
    suspects.sort(key=lambda j: (report.discrimination[j], report.rest_correlation[j]))
    count = _shown(len(suspects), rows)
    lines += ["", f"## Items that behave backwards ({count} of {len(suspects)})", ""]
    lines += _table(
        ["item", "flags", "solved", "discrimination", "rest correlation", "infit", "outfit"],
        [
            [
                report.items[j],
                ", ".join(report.flags[j]),
                _number(report.solved[j]),
                _number(report.discrimination[j]),
                _number(report.rest_correlation[j]),
                _number(report.infit[j]),
                _number(report.outfit[j]),
            ]
            for j in suspects[:count]
        ],
        text_columns=2,
    )
    return lines


def _cmd_items(args: argparse.Namespace) -> int:
    data = _load(args)
    fit = _fit(data, args)
    rasch = fit if fit.kind == "1pl" else _fit(data, args, "1pl")
    report = diagnose_items(data, fit, rasch=rasch)
    lines = [f"# irtbench items: {Path(args.input).name}", "", *_fit_lines(data, fit)]
    lines += _item_lines(report, args.rows)
    rows = report.to_dict()
    if args.se != "none":
        if args.se == "information":
            errors = standard_errors(fit, data)
        else:
            errors = bootstrap_standard_errors(
                data,
                fit.kind,
                n_boot=args.boot,
                seed=args.seed,
                slope_prior_sd=fit.slope_prior_sd,
                n_quad=args.quad,
                resolution=args.resolution,
                max_iter=args.max_iter,
                tol=args.tol,
            )
        for index, row in enumerate(rows):
            for name, values in (
                ("discrimination_se", errors.discrimination),
                ("difficulty_se", errors.difficulty),
            ):
                row[name] = float(values[index]) if np.isfinite(values[index]) else None
        order = np.argsort(np.where(np.isfinite(fit.discrimination), fit.discrimination, np.inf))
        count = _shown(int(fit.estimated.sum()), args.rows)
        lines += ["", f"## Lowest discriminations with {args.se} standard errors", ""]
        lines += _table(
            ["item", "discrimination", "se", "difficulty", "se"],
            [
                [
                    data.items[j],
                    _number(fit.discrimination[j]),
                    _number(errors.discrimination[j]),
                    _number(fit.difficulty[j]),
                    _number(errors.difficulty[j]),
                ]
                for j in order[:count]
            ],
        )
    sys.stdout.write("\n".join(lines) + "\n")
    _write_json(
        args.json,
        {
            "irtbench": __version__,
            "data": _data_json(data),
            "fit": fit.to_dict(),
            "items": rows,
        },
    )
    return 0


def _cmd_subset(args: argparse.Namespace) -> int:
    data = _load(args)
    fit = _fit(data, args)
    abilities = estimate_abilities(fit, data)
    chosen = select_items(fit, args.k, args.method, abilities=abilities.theta, seed=args.seed)
    sys.stdout.write("\n".join(chosen) + "\n")
    _write_json(
        args.json,
        {
            "irtbench": __version__,
            "data": _data_json(data),
            "method": args.method,
            "k": args.k,
            "seed": args.seed,
            "items": list(chosen),
            "fit": fit.to_dict(),
        },
    )
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    try:
        payload = json.loads(Path(args.fit).read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{args.fit}: not valid JSON ({error.msg})") from None
    block = payload.get("fit", payload) if isinstance(payload, dict) else payload
    if not isinstance(block, dict):
        raise ValueError(f"{args.fit}: not a fit written by irtbench")
    fit = IRTFit.from_dict(block)
    data = _load(args)
    abilities = estimate_abilities(fit, data)
    predicted = predict_scores(fit, data)
    lines = [
        f"# irtbench predict: {Path(args.input).name}",
        "",
        f"- fit: {fit.kind} on {len(fit.items)} items; answers given on {data.n_items} of them",
        "- predicted score: estimated share of the fitted items the model would solve",
        "",
        *_model_table(data, abilities, predicted, args.rows),
    ]
    sys.stdout.write("\n".join(lines) + "\n")
    _write_json(
        args.json,
        {
            "irtbench": __version__,
            "data": _data_json(data),
            "n_fit_items": len(fit.items),
            "models": _models_json(data, abilities, predicted),
        },
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    data = _load(args)
    try:
        sizes = [int(part) for part in args.sizes.split(",") if part.strip()]
    except ValueError:
        raise ValueError("--sizes takes comma-separated integers, such as 25,50,100") from None
    one = _fit(data, args, "1pl")
    two = _fit(data, args, "2pl")
    abilities = estimate_abilities(two, data)
    report = diagnose_items(data, two, rasch=one)
    criteria = {"1pl": information_criteria(one), "2pl": information_criteria(two)}
    ratio = likelihood_ratio(one, two)
    options = {
        "slope_prior_sd": args.slope_prior_sd if args.slope_prior_sd > 0 else None,
        "n_quad": args.quad,
        "resolution": args.resolution,
        "max_iter": args.max_iter,
        "tol": args.tol,
    }
    lines = [f"# irtbench report: {Path(args.input).name}", "", *_fit_lines(data, two)]
    lines += ["", "## 1PL against 2PL", ""]
    lines += _table(
        ["model", "parameters", "log-likelihood", "AIC", "BIC"],
        [
            [
                kind,
                str(int(row["n_parameters"])),
                _number(row["log_likelihood"], 1),
                _number(row["aic"], 1),
                _number(row["bic"], 1),
            ]
            for kind, row in criteria.items()
        ],
    )
    lines += [
        "",
        f"Likelihood ratio: {ratio['statistic']:.1f} on {int(ratio['df'])} degrees of freedom, "
        f"p = {ratio['p_value']:.3g}.",
    ]
    payload: dict[str, object] = {
        "irtbench": __version__,
        "data": _data_json(data),
        "fit": two.to_dict(),
        "fit_1pl": one.to_dict(),
        "criteria": criteria,
        "likelihood_ratio": ratio,
        "items": report.to_dict(),
        "models": _models_json(data, abilities),
    }
    if args.folds > 0:
        held_out = cross_validate(data, folds=args.folds, seed=args.seed, **options)
        payload["held_out_cells"] = held_out
        lines += ["", f"Held-out cells ({args.folds}-fold):", ""]
        lines += _table(
            ["predictor", "log-likelihood per cell", "AUC", "accuracy"],
            [
                [
                    kind,
                    _number(row["log_likelihood"], 4),
                    _number(row["auc"], 4),
                    _number(row["accuracy"], 4),
                ]
                for kind, row in held_out.items()
            ],
        )
    lines += ["", "## Items", "", *_item_lines(report, args.rows)]
    sizes = [k for k in sizes if 1 <= k <= int(two.estimated.sum())]
    subsets = None
    if args.splits > 0 and sizes:
        subsets = evaluate_subsets(data, sizes, splits=args.splits, seed=args.seed, **options)
        payload["subsets"] = subsets
        names = list(next(iter(subsets["sizes"].values())))  # type: ignore[union-attr]
        lines += [
            "",
            "## Predicting the full score of unseen models from k items",
            "",
            f"Mean absolute error in score points over {subsets['splits']} splits "
            f"({subsets['n_train']} models to fit, {subsets['n_test']} to predict).",
            "",
        ]
        lines += _table(
            ["k", *names],
            [
                [size, *[_number(row[name]["mae"], 4) for name in names]]
                for size, row in subsets["sizes"].items()  # type: ignore[union-attr]
            ],
        )
    lines += ["", *_model_table(data, abilities, None, args.rows)]
    if args.plots:
        from irtbench.plots import save_plots

        written = save_plots(Path(args.plots), two, abilities.theta, subsets)
        lines += ["", "## Charts", "", *[f"- {path}" for path in written]]
        payload["charts"] = [str(path) for path in written]
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    _write_json(args.json, payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="irtbench",
        description="Item response theory for a models-by-items matrix of benchmark results.",
    )
    parser.add_argument("--version", action="version", version=f"irtbench {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    source = argparse.ArgumentParser(add_help=False)
    source.add_argument("input", help="results file: long CSV/TSV/JSONL or a wide CSV/TSV matrix")
    source.add_argument(
        "--format",
        choices=("auto", "long", "wide"),
        default="auto",
        help="layout of the input (default: auto)",
    )
    source.add_argument("--model-col", default="model", help="long format: model column")
    source.add_argument("--item-col", default="item", help="long format: item column")
    source.add_argument("--score-col", default="score", help="long format: 0/1 score column")
    source.add_argument("--rows", type=int, default=10, help="rows per printed table; 0 prints all")
    source.add_argument("--json", metavar="PATH", help="also write the full result as JSON")

    fitting = argparse.ArgumentParser(add_help=False)
    fitting.add_argument("--irt-model", choices=KINDS, default="2pl", help="default: 2pl")
    fitting.add_argument(
        "--slope-prior-sd",
        type=float,
        default=0.5,
        help="2PL: prior sd of each slope relative to the common slope; 0 for no prior "
        "(default: 0.5)",
    )
    fitting.add_argument(
        "--quad", type=int, default=61, help="nodes of the starting Gauss-Hermite rule"
    )
    fitting.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="largest node gap in units of the narrowest posterior width (default: 1.0)",
    )
    fitting.add_argument("--max-iter", type=int, default=2000, help="EM iteration limit")
    fitting.add_argument(
        "--tol", type=float, default=1e-6, help="EM stops below this parameter change"
    )

    fit = commands.add_parser(
        "fit",
        parents=[source, fitting],
        help="fit a 1PL or 2PL model and print item parameters and model abilities",
        description="Fit a 1PL (Rasch) or 2PL model by marginal maximum likelihood.",
    )
    fit.set_defaults(handler=_cmd_fit)

    items = commands.add_parser(
        "items",
        parents=[source, fitting],
        help="item diagnostics: uninformative items and items that behave backwards",
        description="Flag items solved by all or by none and items that weaker models solve "
        "more often than stronger ones.",
    )
    items.add_argument(
        "--se",
        choices=("none", "information", "bootstrap"),
        default="none",
        help="standard errors: observed information, or bootstrap over models (default: none)",
    )
    items.add_argument("--boot", type=int, default=200, help="bootstrap replicates")
    items.add_argument("--seed", type=int, default=0)
    items.set_defaults(handler=_cmd_items)

    subset = commands.add_parser(
        "subset",
        parents=[source, fitting],
        help="choose k items whose answers predict the full score",
        description="Choose k items; prints one item id per line. With --json the fit is "
        "saved too, for `irtbench predict`.",
    )
    subset.add_argument("-k", type=int, required=True, help="number of items to choose")
    subset.add_argument("--method", choices=METHODS, default="information")
    subset.add_argument("--seed", type=int, default=0)
    subset.set_defaults(handler=_cmd_subset)

    predict = commands.add_parser(
        "predict",
        parents=[source],
        help="predict full scores of models from their answers on a subset of items",
        description="Estimate each model's ability and full score from whatever items it has "
        "results for, using a fit saved by `fit --json` or `subset --json`.",
    )
    predict.add_argument("--fit", required=True, metavar="PATH", help="JSON with a saved fit")
    predict.set_defaults(handler=_cmd_predict)

    report = commands.add_parser(
        "report",
        parents=[source, fitting],
        help="full report: fits, 1PL against 2PL, item diagnostics, subset evaluation",
        description="Markdown report on standard output (or --out) and JSON with --json.",
    )
    report.add_argument("--out", metavar="PATH", help="write the Markdown here")
    report.add_argument("--folds", type=int, default=5, help="held-out cell folds; 0 skips")
    report.add_argument("--splits", type=int, default=10, help="model splits; 0 skips subsets")
    report.add_argument("--sizes", default="25,50,100", help="subset sizes (default: 25,50,100)")
    report.add_argument("--seed", type=int, default=0)
    report.add_argument("--plots", metavar="DIR", help="write charts here (needs matplotlib)")
    report.set_defaults(handler=_cmd_report)
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
