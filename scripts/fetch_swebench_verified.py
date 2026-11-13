"""Build a models-by-items results matrix from public SWE-bench Verified submissions.

The per-submission results live in the public repository https://github.com/SWE-bench/experiments
under ``evaluation/verified/<submission>/``. That repository carries no licence file, so nothing
from it is redistributed here: this script downloads the files it needs into a directory of
your choice and writes the matrix next to them.

    python scripts/fetch_swebench_verified.py DATA_DIR

It makes a shallow, blob-less, sparse clone (a few MB: only the results and metadata files are
downloaded), then writes

    DATA_DIR/swebench_verified.csv         long format: model, item, score
    DATA_DIR/swebench_verified_meta.json   commit hash, counts, per-submission source file

A submission is read from ``results/results.json`` when present (the instance ids listed under
``resolved`` score 1, every other instance scores 0) and otherwise from
``per_instance_details.json`` (one ``resolved`` flag per instance; an instance the file does not
mention is left missing). The item universe is the union of all instance ids that any of these
files mentions. A submission whose file marks no instance at all as resolved is taken to be an
incomplete record and is dropped unless ``--keep-empty`` is given; the dropped names are
printed and stored in the metadata file.

Requires git 2.27 or newer. Uses only the standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/SWE-bench/experiments"
PATTERNS = (
    "/evaluation/verified/*/results/results.json",
    "/evaluation/verified/*/per_instance_details.json",
)


def git(*arguments: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def clone(target: Path) -> str:
    """Sparse-clone the results files into ``target`` and return the commit hash."""
    if not (target / ".git").exists():
        git("clone", "--quiet", "--depth", "1", "--filter=blob:none", "--sparse", REPOSITORY,
            str(target))
    else:
        git("fetch", "--quiet", "--depth", "1", "origin", cwd=target)
        git("reset", "--quiet", "--hard", "FETCH_HEAD", cwd=target)
    git("sparse-checkout", "set", "--no-cone", *PATTERNS, cwd=target)
    return git("rev-parse", "HEAD", cwd=target)


def instance_ids(value: object) -> list[str]:
    return [entry for entry in value if isinstance(entry, str)] if isinstance(value, list) else []


def read_submissions(root: Path) -> tuple[dict[str, dict[str, int]], dict[str, str], set[str]]:
    """Scores per submission, the file each came from, and every instance id seen."""
    scores: dict[str, dict[str, int]] = {}
    source: dict[str, str] = {}
    universe: set[str] = set()
    listed: dict[str, set[str]] = {}
    for folder in sorted(path for path in (root / "evaluation" / "verified").iterdir()):
        results = folder / "results" / "results.json"
        details = folder / "per_instance_details.json"
        if results.is_file():
            payload = json.loads(results.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or "resolved" not in payload:
                continue
            for value in payload.values():
                universe.update(instance_ids(value))
            listed[folder.name] = set(instance_ids(payload["resolved"]))
            source[folder.name] = "results/results.json"
        elif details.is_file():
            payload = json.loads(details.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            rows = {
                str(item): int(bool(row.get("resolved")))
                for item, row in payload.items()
                if isinstance(row, dict) and "resolved" in row
            }
            universe.update(rows)
            scores[folder.name] = rows
            source[folder.name] = "per_instance_details.json"
    for name, resolved in listed.items():
        scores[name] = {item: int(item in resolved) for item in universe}
    return scores, source, universe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", type=Path, help="directory for the clone and the matrix")
    parser.add_argument(
        "--keep-empty", action="store_true", help="keep submissions with no resolved instance"
    )
    args = parser.parse_args(argv)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    commit = clone(args.data_dir / "experiments")
    scores, source, universe = read_submissions(args.data_dir / "experiments")
    items = sorted(universe)
    empty = sorted(name for name, row in scores.items() if not any(row.values()))
    if not args.keep_empty:
        for name in empty:
            del scores[name]
    models = sorted(scores)
    output = args.data_dir / "swebench_verified.csv"
    missing = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["model", "item", "score"])
        for model in models:
            for item in items:
                if item in scores[model]:
                    writer.writerow([model, item, scores[model][item]])
                else:
                    missing += 1
    meta = {
        "repository": REPOSITORY,
        "commit": commit,
        "n_models": len(models),
        "n_items": len(items),
        "n_missing": missing,
        "without_any_resolved": empty,
        "kept_empty": bool(args.keep_empty),
        "sources": {name: source[name] for name in models},
    }
    (args.data_dir / "swebench_verified_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"commit {commit}")
    if empty:
        verb = "kept" if args.keep_empty else "dropped"
        print(f"{verb} {len(empty)} submissions with no resolved instance: {', '.join(empty)}")
    print(f"{len(models)} submissions x {len(items)} items, {missing} missing cells -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
