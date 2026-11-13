"""Item diagnostics: which items carry no information, which behave backwards, which misfit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from irtbench.data import ResponseMatrix
from irtbench.mml import ALL_CORRECT, ALL_INCORRECT, NO_DATA, IRTFit, estimate_abilities, sigmoid

SOLVED_BY_ALL = "solved_by_all"
SOLVED_BY_NONE = "solved_by_none"
NEGATIVE_DISCRIMINATION = "negative_discrimination"
NEGATIVE_REST_CORRELATION = "negative_rest_correlation"

_STATUS_FLAG = {ALL_CORRECT: SOLVED_BY_ALL, ALL_INCORRECT: SOLVED_BY_NONE, NO_DATA: NO_DATA}


@dataclass(frozen=True)
class ItemDiagnostics:
    """Per-item diagnostics, one entry per item of the data.

    ``rest_correlation`` is the point-biserial correlation between the item and the rest score.
    ``infit`` and ``outfit`` are Rasch mean squares (NaN unless a 1PL fit was available).
    ``flags`` lists what is wrong with each item, if anything.
    """

    items: tuple[str, ...]
    tried: np.ndarray
    solved: np.ndarray
    status: tuple[str, ...]
    discrimination: np.ndarray
    difficulty: np.ndarray
    rest_correlation: np.ndarray
    infit: np.ndarray
    outfit: np.ndarray
    flags: tuple[tuple[str, ...], ...]

    def flagged(self, flag: str) -> tuple[str, ...]:
        """Ids of the items that carry ``flag``."""
        pairs = zip(self.items, self.flags, strict=True)
        return tuple(item for item, flags in pairs if flag in flags)

    def to_dict(self) -> list[dict[str, object]]:
        def number(value: float) -> float | None:
            return float(value) if np.isfinite(value) else None

        return [
            {
                "item": item,
                "tried": int(self.tried[j]),
                "solved": number(self.solved[j]),
                "status": self.status[j],
                "discrimination": number(self.discrimination[j]),
                "difficulty": number(self.difficulty[j]),
                "rest_correlation": number(self.rest_correlation[j]),
                "infit": number(self.infit[j]),
                "outfit": number(self.outfit[j]),
                "flags": list(self.flags[j]),
            }
            for j, item in enumerate(self.items)
        ]


def rest_correlation(data: ResponseMatrix) -> np.ndarray:
    """Point-biserial correlation of each item with the rest score.

    The rest score of a model for an item is the share it solved among its other observed
    items. The correlation is taken over the models that have a result for the item and at
    least one other item. It is NaN when the item or the rest score does not vary there.
    """
    scores = data.scores
    seen = ~np.isnan(scores)
    filled = np.where(seen, scores, 0.0)
    total = filled.sum(axis=1, keepdims=True)
    count = seen.sum(axis=1, keepdims=True)
    usable = seen & (count > 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rest = np.where(usable, (total - filled) / (count - 1), 0.0)
    weight = usable.astype(float)
    n = weight.sum(axis=0)
    x = filled * weight
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_x = x.sum(axis=0) / n
        mean_r = rest.sum(axis=0) / n
        cov = (x * rest).sum(axis=0) / n - mean_x * mean_r
        var_x = mean_x * (1.0 - mean_x)
        var_r = (rest * rest).sum(axis=0) / n - mean_r * mean_r
        out = cov / np.sqrt(var_x * var_r)
    out[(n < 3) | ~(var_x > 1e-12) | ~(var_r > 1e-12)] = np.nan
    return out


def item_information(fit: IRTFit, theta: np.ndarray | list[float] | float) -> np.ndarray:
    """Fisher information of each item (columns) at each ability (rows): ``a^2 P (1 - P)``.

    Items left out of the fit carry no information and return 0; items without data return NaN.
    """
    probability = fit.probability(theta)
    slope = np.where(fit.estimated, fit.discrimination, 0.0)
    return slope[None, :] ** 2 * probability * (1.0 - probability)


def test_information(fit: IRTFit, theta: np.ndarray | list[float] | float) -> np.ndarray:
    """Information of the whole test at each ability: the sum of the item informations."""
    return np.nansum(item_information(fit, theta), axis=1)


test_information.__test__ = False  # not a pytest test, despite the name


def fit_statistics(fit: IRTFit, data: ResponseMatrix) -> tuple[np.ndarray, np.ndarray]:
    """Infit and outfit mean squares of each item under ``fit`` (meant for the 1PL).

    With ``P_ij`` the fitted probability at the model's EAP ability and ``W_ij = P_ij (1 -
    P_ij)``, outfit is the mean over models of ``(x_ij - P_ij)^2 / W_ij`` and infit is
    ``sum_i (x_ij - P_ij)^2 / sum_i W_ij``. Both are near 1 when the item behaves as the model
    says; values well above 1 mean more surprises than expected. Items left out of the fit are
    NaN.
    """
    column = {item: j for j, item in enumerate(fit.items)}
    missing = [item for item in data.items if item not in column]
    if missing:
        raise ValueError(f"{len(missing)} items are not in the fit: {', '.join(missing[:5])}")
    theta = estimate_abilities(fit, data).theta
    order = [column[item] for item in data.items]
    keep = fit.estimated[order]
    a, d = fit.discrimination[order], fit.intercept[order]
    infit = np.full(data.n_items, np.nan)
    outfit = np.full(data.n_items, np.nan)
    probability = sigmoid(np.outer(theta, a[keep]) + d[keep][None, :])
    cells = data.scores[:, keep]
    seen = ~np.isnan(cells)
    squared = np.where(seen, (np.where(seen, cells, 0.0) - probability) ** 2, 0.0)
    variance = np.where(seen, probability * (1.0 - probability), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        infit[keep] = squared.sum(axis=0) / variance.sum(axis=0)
        ratio = np.where(seen, squared / np.maximum(variance, 1e-300), 0.0)
        outfit[keep] = ratio.sum(axis=0) / seen.sum(axis=0)
    return infit, outfit


def diagnose_items(
    data: ResponseMatrix, fit: IRTFit, *, rasch: IRTFit | None = None
) -> ItemDiagnostics:
    """Diagnostics of every item of ``data`` under ``fit`` (fitted to the same items).

    Infit and outfit come from ``rasch`` when given, else from ``fit`` when it is a 1PL.
    Flags: ``solved_by_all``, ``solved_by_none``, ``no_data``; ``negative_discrimination`` when
    a 2PL slope is below zero; ``negative_rest_correlation`` when stronger models (by rest
    score) solve the item less often. The last two mark candidates for a broken or mislabelled
    item; neither proves it.
    """
    if fit.items != data.items:
        raise ValueError("the fit and the data must list the same items in the same order")
    correlation = rest_correlation(data)
    source = rasch if rasch is not None else (fit if fit.kind == "1pl" else None)
    if source is not None:
        infit, outfit = fit_statistics(source, data)
    else:
        infit = np.full(data.n_items, np.nan)
        outfit = np.full(data.n_items, np.nan)
    flags: list[tuple[str, ...]] = []
    for j in range(data.n_items):
        found = []
        if fit.status[j] in _STATUS_FLAG:
            found.append(_STATUS_FLAG[fit.status[j]])
        if fit.kind == "2pl" and fit.discrimination[j] < 0.0:
            found.append(NEGATIVE_DISCRIMINATION)
        if correlation[j] < 0.0:
            found.append(NEGATIVE_REST_CORRELATION)
        flags.append(tuple(found))
    return ItemDiagnostics(
        items=data.items,
        tried=data.observed.sum(axis=0).astype(int),
        solved=data.item_means(),
        status=fit.status,
        discrimination=fit.discrimination.copy(),
        difficulty=fit.difficulty,
        rest_correlation=correlation,
        infit=infit,
        outfit=outfit,
        flags=tuple(flags),
    )
