"""Joint maximum likelihood with a ridge penalty, for comparison with the marginal fit.

Joint maximum likelihood treats each model's ability as a parameter and maximises the
likelihood over abilities and item parameters together:

    logit P(x_ij = 1) = a_j * theta_i + d_j        (a_j = 1 in the 1PL, the Rasch model)

Without a penalty a model that solved everything (or nothing) has no finite ability and the
location of the scale is arbitrary. The ridge term ``ridge / 2 * (sum theta^2 + sum d^2 +
sum (a - 1)^2)`` fixes both. Joint estimates of item parameters are known to be biased when
the number of items is small, because every model brings a parameter of its own; the marginal
fit in :mod:`irtbench.mml` does not have that problem, which is why it is the default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from irtbench.data import ResponseMatrix
from irtbench.mml import ALL_CORRECT, ALL_INCORRECT, ESTIMATED, KINDS, _item_status, sigmoid


@dataclass(frozen=True)
class JMLFit:
    """Abilities and item parameters from a joint fit.

    Abilities are on the logit scale of the model (they are not standardised). Items left out
    (solved by all, by none, or without data) hold NaN parameters, as in the marginal fit.
    """

    kind: str
    models: tuple[str, ...]
    items: tuple[str, ...]
    ability: np.ndarray
    discrimination: np.ndarray
    intercept: np.ndarray
    status: tuple[str, ...]
    log_likelihood: float
    ridge: float
    iterations: int
    converged: bool

    @property
    def difficulty(self) -> np.ndarray:
        """``-d_j / a_j``; infinite for items solved by all or by none."""
        with np.errstate(divide="ignore", invalid="ignore"):
            out = -self.intercept / self.discrimination
        for index, state in enumerate(self.status):
            if state == ALL_CORRECT:
                out[index] = -np.inf
            elif state == ALL_INCORRECT:
                out[index] = np.inf
        return out


def fit_jml(
    data: ResponseMatrix,
    kind: str = "1pl",
    *,
    ridge: float = 0.1,
    max_iter: int = 2000,
    tol: float = 1e-8,
) -> JMLFit:
    """Fit a 1PL or 2PL model by joint maximum likelihood with a ridge penalty.

    Abilities and item parameters are updated in turn by damped Newton steps (each ability
    given the items, then each item given the abilities), followed by the shift of the whole
    scale that the penalty prefers, until no Newton step exceeds ``tol``. ``ridge`` must be
    positive.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown model {kind!r}; use one of {', '.join(KINDS)}")
    if not ridge > 0.0:
        raise ValueError("ridge must be positive")
    status = _item_status(data.scores)
    keep = status == ESTIMATED
    if int(keep.sum()) < 2:
        raise ValueError("joint ML needs at least 2 items that some models solve and some do not")
    cells = data.scores[:, keep]
    seen = (~np.isnan(cells)).astype(float)
    solved = np.where(seen > 0, cells, 0.0)
    theta = np.zeros(data.n_models)
    a = np.ones(int(keep.sum()))
    d = np.zeros(int(keep.sum()))
    converged = False
    iterations = 0
    while iterations < max_iter and not converged:
        iterations += 1
        p = sigmoid(np.outer(theta, a) + d[None, :])
        gradient = ((solved - seen * p) * a[None, :]).sum(axis=1) - ridge * theta
        curvature = (seen * p * (1.0 - p) * a[None, :] ** 2).sum(axis=1) + ridge
        step_theta = np.clip(gradient / curvature, -1.0, 1.0)
        theta = theta + step_theta

        p = sigmoid(np.outer(theta, a) + d[None, :])
        residual = solved - seen * p
        weight = seen * p * (1.0 - p)
        g_d = residual.sum(axis=0) - ridge * d
        h_dd = weight.sum(axis=0) + ridge
        if kind == "1pl":
            step_a = np.zeros_like(a)
            step_d = g_d / h_dd
        else:
            g_a = (residual * theta[:, None]).sum(axis=0) - ridge * (a - 1.0)
            h_aa = (weight * theta[:, None] ** 2).sum(axis=0) + ridge
            h_ad = (weight * theta[:, None]).sum(axis=0)
            det = h_aa * h_dd - h_ad * h_ad
            step_a = (h_dd * g_a - h_ad * g_d) / det
            step_d = (h_aa * g_d - h_ad * g_a) / det
        step_a, step_d = np.clip(step_a, -1.0, 1.0), np.clip(step_d, -1.0, 1.0)
        a, d = a + step_a, d + step_d
        # Moving every ability by c and every intercept by -a_j * c leaves the likelihood
        # unchanged, so the best c is the one that minimises the ridge term. Without this
        # exact step the alternating updates drift along that direction very slowly.
        shift = (float(a @ d) - float(theta.sum())) / (theta.size + float(a @ a))
        theta, d = theta + shift, d - a * shift
        moved = max(
            float(np.abs(step_theta).max()),
            float(np.abs(step_a).max()),
            float(np.abs(step_d).max()),
        )
        converged = moved < tol
    if a.sum() < 0.0:
        a, theta = -a, -theta
    z = np.outer(theta, a) + d[None, :]
    loglik = float((solved * z - seen * np.logaddexp(0.0, z)).sum())
    discrimination = np.full(data.n_items, np.nan)
    intercept = np.full(data.n_items, np.nan)
    discrimination[keep] = a
    intercept[keep] = d
    return JMLFit(
        kind=kind,
        models=data.models,
        items=data.items,
        ability=theta,
        discrimination=discrimination,
        intercept=intercept,
        status=tuple(str(state) for state in status),
        log_likelihood=loglik,
        ridge=ridge,
        iterations=iterations,
        converged=converged,
    )
