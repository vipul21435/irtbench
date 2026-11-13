"""Standard errors of item parameters: observed information, and bootstrap over models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from irtbench.data import ResponseMatrix
from irtbench.mml import IRTFit, _aligned_indicators, fit_mml, score_posterior, sigmoid


@dataclass(frozen=True)
class ItemStandardErrors:
    """Standard errors per item of ``fit.items``; NaN for items left out of the fit.

    ``replicates`` counts, per item, the bootstrap fits in which the item was estimated (0 for
    the information method).
    """

    items: tuple[str, ...]
    discrimination: np.ndarray
    intercept: np.ndarray
    difficulty: np.ndarray
    method: str
    replicates: np.ndarray


def observed_information(
    fit: IRTFit, data: ResponseMatrix, *, resolution: float = 1.0
) -> np.ndarray:
    """Observed information of the estimated item parameters.

    This is minus the Hessian of the marginal log-likelihood of ``data`` at ``fit``.

    Parameters are ordered slopes first, then intercepts (for the 1PL: the one slope, then the
    intercepts). The matrix is the complete-data information averaged over the ability
    posteriors, minus the posterior covariance of the complete-data score, summed over models
    (the missing-information identity of the EM literature).
    """
    solved, failed = _aligned_indicators(fit, data)
    seen = solved + failed
    keep = fit.estimated
    a, d = fit.discrimination[keep], fit.intercept[keep]
    grid, posterior, _, _ = score_posterior(fit, data, resolution=resolution)
    nodes = grid.nodes
    p = sigmoid(np.outer(a, nodes) + d[:, None])
    weight = (seen.T @ posterior) * p * (1.0 - p)
    count = a.size
    one_slope = fit.kind == "1pl"
    size = count + 1 if one_slope else 2 * count
    info = np.zeros((size, size))
    h_aa, h_ad, h_dd = weight @ (nodes * nodes), weight @ nodes, weight.sum(axis=1)
    slopes = np.zeros(count, dtype=int) if one_slope else np.arange(count)
    offset = 1 if one_slope else count
    np.add.at(info, (slopes, slopes), h_aa)
    np.add.at(info, (slopes, offset + np.arange(count)), h_ad)
    np.add.at(info, (offset + np.arange(count), slopes), h_ad)
    info[offset + np.arange(count), offset + np.arange(count)] += h_dd
    for row in range(data.n_models):
        mass = posterior[row]
        use = mass > 1e-14 * mass.max()
        mass = mass[use]
        residual = seen[row][:, None] * (solved[row][:, None] - p[:, use])
        by_slope = residual * nodes[use][None, :]
        if one_slope:
            score = np.vstack([by_slope.sum(axis=0, keepdims=True), residual])
        else:
            score = np.vstack([by_slope, residual])
        mean = score @ mass
        info -= (score * mass[None, :]) @ score.T - np.outer(mean, mean)
    return info


def standard_errors(
    fit: IRTFit, data: ResponseMatrix, *, resolution: float = 1.0
) -> ItemStandardErrors:
    """Standard errors from the inverse observed information.

    With a slope prior the prior's curvature (its precision ``1 / (s c)^2``, the common slope
    ``c`` held fixed) is added for the slopes, which makes these the spreads of a normal
    approximation to the posterior rather than sampling errors. The difficulty error comes
    from the delta method on ``b = -d / a``. Entries are NaN where the information matrix is
    not positive definite.
    """
    info = observed_information(fit, data, resolution=resolution)
    keep = fit.estimated
    count = int(keep.sum())
    a, d = fit.discrimination[keep], fit.intercept[keep]
    one_slope = fit.kind == "1pl"
    if fit.slope_prior_sd is not None and fit.slope_prior_mean:
        precision = 1.0 / (fit.slope_prior_sd * fit.slope_prior_mean) ** 2
        info[np.arange(count), np.arange(count)] += precision
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.full_like(info, np.nan)
    slopes = np.zeros(count, dtype=int) if one_slope else np.arange(count)
    intercepts = (1 if one_slope else count) + np.arange(count)
    var_a, var_d = cov[slopes, slopes], cov[intercepts, intercepts]
    cov_ad = cov[slopes, intercepts]
    b = -d / a
    var_b = (b * b * var_a + 2.0 * b * cov_ad + var_d) / (a * a)
    out = {name: np.full(len(fit.items), np.nan) for name in ("a", "d", "b")}
    with np.errstate(invalid="ignore"):
        out["a"][keep] = np.sqrt(var_a)
        out["d"][keep] = np.sqrt(var_d)
        out["b"][keep] = np.sqrt(var_b)
    return ItemStandardErrors(
        fit.items, out["a"], out["d"], out["b"], "information", np.zeros(len(fit.items), dtype=int)
    )


def bootstrap_standard_errors(
    data: ResponseMatrix,
    kind: str = "2pl",
    *,
    n_boot: int = 200,
    seed: int = 0,
    **fit_options: object,
) -> ItemStandardErrors:
    """Standard errors from refitting on models resampled with replacement.

    Each replicate draws as many models as the data has, refits (warm-started from the fit to
    all models) and records the parameters of the items it could estimate. The error of an
    item is the standard deviation over the replicates that estimated it. This is the spread
    over which models happened to be evaluated; it treats the items as fixed.
    """
    if n_boot < 2:
        raise ValueError("the bootstrap needs at least 2 replicates")
    base = fit_mml(data, kind, **fit_options)  # type: ignore[arg-type]
    start = (
        np.where(np.isfinite(base.discrimination), base.discrimination, 1.0),
        np.where(np.isfinite(base.intercept), base.intercept, 0.0),
    )
    rng = np.random.default_rng(seed)
    names = tuple(f"draw-{index}" for index in range(data.n_models))
    draws = np.full((3, n_boot, data.n_items), np.nan)
    for replicate in range(n_boot):
        rows = rng.integers(0, data.n_models, data.n_models)
        sample = ResponseMatrix(names, data.items, data.scores[rows])
        try:
            fit = fit_mml(sample, kind, start=start, **fit_options)  # type: ignore[arg-type]
        except ValueError:
            continue
        keep = fit.estimated
        draws[0, replicate, keep] = fit.discrimination[keep]
        draws[1, replicate, keep] = fit.intercept[keep]
        draws[2, replicate, keep] = fit.difficulty[keep]
    valid = np.isfinite(draws[0]).sum(axis=0)
    spread = np.full((3, data.n_items), np.nan)
    enough = valid >= 2
    with np.errstate(invalid="ignore"):
        spread[:, enough] = np.nanstd(draws[:, :, enough], axis=1, ddof=1)
    return ItemStandardErrors(data.items, spread[0], spread[1], spread[2], "bootstrap", valid)
