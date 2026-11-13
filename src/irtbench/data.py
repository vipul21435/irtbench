"""Models-by-items matrices of binary benchmark results and the file formats they come in.

Two layouts are read:

* long: one record per cell with a model name, an item id and a score (CSV, TSV or JSON lines);
* wide: a CSV or TSV whose first column is the model name and whose other columns are items.

A score is 1 (solved) or 0 (not solved). A cell that is absent, empty or one of ``NA``, ``NaN``,
``null``, ``None`` is missing and is stored as NaN.

Files are read as UTF-8; a byte-order mark at the start, which spreadsheet programs add when
they save CSV, is skipped.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MISSING = frozenset({"", "na", "n/a", "nan", "none", "null"})
_ONE = frozenset({"1", "1.0", "true"})
_ZERO = frozenset({"0", "0.0", "false"})
_JSONL_SUFFIXES = frozenset({".jsonl", ".ndjson"})
# UTF-8 that also accepts a leading byte-order mark instead of gluing it to the first field
_ENCODING = "utf-8-sig"


def parse_score(value: object) -> float:
    """Map one cell to 1.0, 0.0 or NaN (missing).

    Accepts the numbers 0 and 1, booleans, and the strings ``0``, ``1``, ``true``, ``false``
    (any case). Anything else, including fractional scores, raises ``ValueError``.
    """
    if value is None:
        return float("nan")
    if isinstance(value, bool | np.bool_):
        return 1.0 if value else 0.0
    if isinstance(value, int | float | np.integer | np.floating):
        number = float(value)
        if number != number:
            return float("nan")
        if number in (0.0, 1.0):
            return number
        raise ValueError(f"score must be 0 or 1, got {value!r}")
    text = str(value).strip().lower()
    if text in _MISSING:
        return float("nan")
    if text in _ONE:
        return 1.0
    if text in _ZERO:
        return 0.0
    raise ValueError(f"score must be 0 or 1, got {value!r}")


@dataclass(frozen=True)
class ResponseMatrix:
    """Binary results of models (rows) on items (columns).

    ``scores`` has shape ``(n_models, n_items)`` and holds 1.0, 0.0 or NaN for a missing cell.
    The array is copied on construction and made read-only.
    """

    models: tuple[str, ...]
    items: tuple[str, ...]
    scores: np.ndarray

    def __post_init__(self) -> None:
        models = tuple(str(name) for name in self.models)
        items = tuple(str(name) for name in self.items)
        scores = np.array(self.scores, dtype=float)
        if scores.ndim != 2 or scores.shape != (len(models), len(items)):
            raise ValueError(
                f"scores has shape {scores.shape}, expected ({len(models)}, {len(items)})"
            )
        present = scores[~np.isnan(scores)]
        if not np.isin(present, (0.0, 1.0)).all():
            raise ValueError("scores must be 0, 1 or NaN")
        for label, names in (("model", models), ("item", items)):
            if len(set(names)) != len(names):
                raise ValueError(f"duplicate {label} names")
        scores.setflags(write=False)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "scores", scores)

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def observed(self) -> np.ndarray:
        """Boolean mask of the cells that are not missing."""
        return ~np.isnan(self.scores)

    @property
    def n_missing(self) -> int:
        return int(np.isnan(self.scores).sum())

    def item_means(self) -> np.ndarray:
        """Share of observed cells solved, per item; NaN for an item with no observed cell."""
        return _mean_over(self.scores, axis=0)

    def model_means(self) -> np.ndarray:
        """Share of observed cells solved, per model; NaN for a model with no observed cell."""
        return _mean_over(self.scores, axis=1)

    def take_models(self, index: Sequence[int] | np.ndarray) -> ResponseMatrix:
        """A new matrix with the given rows, in the given order."""
        rows = np.asarray(index, dtype=int)
        return ResponseMatrix(
            tuple(self.models[i] for i in rows), self.items, self.scores[rows, :]
        )

    def take_items(self, index: Sequence[int] | np.ndarray) -> ResponseMatrix:
        """A new matrix with the given columns, in the given order."""
        cols = np.asarray(index, dtype=int)
        return ResponseMatrix(self.models, tuple(self.items[j] for j in cols), self.scores[:, cols])


def _mean_over(scores: np.ndarray, axis: int) -> np.ndarray:
    seen = ~np.isnan(scores)
    count = seen.sum(axis=axis)
    total = np.where(seen, scores, 0.0).sum(axis=axis)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def from_long(records: Iterable[tuple[object, object, object]]) -> ResponseMatrix:
    """Build a matrix from ``(model, item, score)`` records.

    Models and items keep the order in which they first appear. A pair that never appears is a
    missing cell. Repeating a pair with the same score is harmless; repeating it with a different
    score raises ``ValueError``.
    """
    model_index: dict[str, int] = {}
    item_index: dict[str, int] = {}
    cells: dict[tuple[int, int], float] = {}
    for position, (model, item, score) in enumerate(records, start=1):
        model_name, item_name = str(model), str(item)
        try:
            value = parse_score(score)
        except ValueError as error:
            raise ValueError(f"record {position} ({model_name}, {item_name}): {error}") from None
        row = model_index.setdefault(model_name, len(model_index))
        col = item_index.setdefault(item_name, len(item_index))
        if value != value:
            continue
        previous = cells.setdefault((row, col), value)
        if previous != value:
            raise ValueError(
                f"record {position}: conflicting scores for model {model_name!r}, "
                f"item {item_name!r}"
            )
    scores = np.full((len(model_index), len(item_index)), np.nan)
    for (row, col), value in cells.items():
        scores[row, col] = value
    return ResponseMatrix(tuple(model_index), tuple(item_index), scores)


def _delimiter(path: Path) -> str:
    return "\t" if path.suffix.lower() == ".tsv" else ","


def read_long(
    path: str | Path,
    *,
    model_col: str = "model",
    item_col: str = "item",
    score_col: str = "score",
) -> ResponseMatrix:
    """Read long-format results from CSV, TSV or JSON lines (chosen by file suffix)."""
    path = Path(path)
    columns = (model_col, item_col, score_col)
    if path.suffix.lower() in _JSONL_SUFFIXES:
        return from_long(_jsonl_records(path, columns))
    return from_long(_csv_records(path, columns))


def _jsonl_records(
    path: Path, columns: tuple[str, str, str]
) -> Iterable[tuple[object, object, object]]:
    model_col, item_col, score_col = columns
    with path.open(encoding=_ENCODING) as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path.name} line {number}: not valid JSON ({error.msg})"
                ) from None
            if not isinstance(record, dict):
                raise ValueError(f"{path.name} line {number}: expected a JSON object")
            for column in (model_col, item_col):
                if record.get(column) is None:
                    raise ValueError(f"{path.name} line {number}: no {column!r} field")
            yield record[model_col], record[item_col], record.get(score_col)


def _csv_records(
    path: Path, columns: tuple[str, str, str]
) -> Iterable[tuple[object, object, object]]:
    model_col, item_col, score_col = columns
    with path.open(encoding=_ENCODING, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=_delimiter(path))
        header = reader.fieldnames or []
        absent = [column for column in columns if column not in header]
        if absent:
            raise ValueError(f"{path.name}: no column named {', '.join(absent)}")
        for row in reader:
            # a row with fewer fields than the header leaves the rest as None, which would
            # otherwise pass as a missing score or as a model or item called "None"
            fields = sum(value is not None for name, value in row.items() if name is not None)
            if fields < len(header):
                raise ValueError(
                    f"{path.name} line {reader.line_num}: {fields} fields for the "
                    f"{len(header)} columns of the header"
                )
            yield row[model_col], row[item_col], row[score_col]


def read_wide(path: str | Path) -> ResponseMatrix:
    """Read a wide matrix: a header of item ids, then one row per model led by its name."""
    path = Path(path)
    with path.open(encoding=_ENCODING, newline="") as handle:
        rows = [row for row in csv.reader(handle, delimiter=_delimiter(path)) if row]
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"{path.name}: expected a header row of items and at least one model row")
    items = tuple(rows[0][1:])
    models: list[str] = []
    scores = np.full((len(rows) - 1, len(items)), np.nan)
    for index, row in enumerate(rows[1:]):
        if len(row) != len(items) + 1:
            raise ValueError(
                f"{path.name} row {index + 2}: {len(row) - 1} cells for {len(items)} items"
            )
        models.append(row[0])
        for col, cell in enumerate(row[1:]):
            try:
                scores[index, col] = parse_score(cell)
            except ValueError as error:
                raise ValueError(
                    f"{path.name} row {index + 2}, item {items[col]!r}: {error}"
                ) from None
    return ResponseMatrix(tuple(models), items, scores)


def read_matrix(
    path: str | Path,
    fmt: str = "auto",
    *,
    model_col: str = "model",
    item_col: str = "item",
    score_col: str = "score",
) -> ResponseMatrix:
    """Read results in the long or the wide layout.

    With ``fmt="auto"`` a JSON lines file is long, and a CSV or TSV file is long when its header
    names the three long-format columns and wide otherwise.
    """
    path = Path(path)
    if fmt not in ("auto", "long", "wide"):
        raise ValueError(f"unknown format {fmt!r}; use auto, long or wide")
    if fmt == "auto":
        if path.suffix.lower() in _JSONL_SUFFIXES:
            fmt = "long"
        else:
            with path.open(encoding=_ENCODING, newline="") as handle:
                header = next(csv.reader(handle, delimiter=_delimiter(path)), [])
            fmt = "long" if {model_col, item_col, score_col} <= set(header) else "wide"
    if fmt == "long":
        return read_long(path, model_col=model_col, item_col=item_col, score_col=score_col)
    return read_wide(path)


def write_long(matrix: ResponseMatrix, path: str | Path) -> None:
    """Write the observed cells as long-format CSV with columns model, item, score."""
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["model", "item", "score"])
        for row, model in enumerate(matrix.models):
            for col, item in enumerate(matrix.items):
                value = matrix.scores[row, col]
                if value == value:
                    writer.writerow([model, item, int(value)])
