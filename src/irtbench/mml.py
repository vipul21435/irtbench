"""Marginal maximum likelihood for the Rasch (1PL) and 2PL models, fitted by EM.

Model. A model ``i`` with ability ``theta_i`` solves item ``j`` with probability

    P(x_ij = 1 | theta_i) = sigmoid(a_j * theta_i + d_j),        theta_i ~ N(0, 1).

``a_j`` is the discrimination and ``d_j`` the intercept. The difficulty ``b_j = -d_j / a_j`` is
the ability at which the item is solved half the time. In the 1PL every item shares one
discrimination ``a``. That is the Rasch model ``sigmoid(t_i - c_j)`` with ``t_i = a * theta_i``
distributed ``N(0, a^2)`` and ``c_j = -d_j``, written so that the 1PL is the 2PL with equal slopes
and the two can be compared by likelihood.

Estimation. Abilities are integrated out against the normal density on a fixed Gauss-Hermite grid
and the item parameters maximise the resulting marginal likelihood. EM alternates

* E-step: each model's posterior weight on every node, then for every item the expected number
  of models at each node that tried it (``n``) and that solved it (``r``);
* M-step: a weighted logistic regression of ``r`` out of ``n`` on the node values, per item for
  the 2PL and with one shared slope for the 1PL, solved by Newton steps with backtracking.

Items whose observed cells are all 1 or all 0 have no finite maximum-likelihood intercept. They
are flagged and left out of the fit; in the limit they contribute nothing to the likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from irtbench.data import ResponseMatrix
from irtbench.quadrature import gauss_hermite

KINDS = ("1pl", "2pl")

ESTIMATED = "estimated"
ALL_CORRECT = "all_correct"
ALL_INCORRECT = "all_incorrect"
NO_DATA = "no_data"

_RIDGE = 1e-10
_MIN_ITEMS = {"1pl": 2, "2pl": 3}


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    """Logistic function, stable for large arguments of either sign."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(z, dtype=float)))


@dataclass(frozen=True)
class IRTFit:
    """Item parameters of a fitted 1PL or 2PL model.

    ``discrimination`` and ``intercept`` have one entry per item. Items that were not estimated
    (see ``status``) hold NaN for the discrimination and ``+inf`` (solved by all), ``-inf``
    (solved by none) or NaN (no data) for the intercept.
    """

    kind: str
    items: tuple[str, ...]
    discrimination: np.ndarray
    intercept: np.ndarray
    status: tuple[str, ...]
    log_likelihood: float
    n_parameters: int
    n_models: int
    n_observations: int
    n_quad: int
    iterations: int
    converged: bool
    trace: tuple[float, ...] = ()

    @classmethod
    def from_parameters(
        cls,
        items: tuple[str, ...] | list[str],
        discrimination: np.ndarray | list[float],
        intercept: np.ndarray | list[float],
        *,
        kind: str = "2pl",
        n_quad: int = 61,
    ) -> IRTFit:
        """Wrap item parameters that were obtained elsewhere so models can be scored with them."""
        if kind not in KINDS:
            raise ValueError(f"unknown model {kind!r}; use one of {', '.join(KINDS)}")
        a = np.array(discrimination, dtype=float)
        d = np.array(intercept, dtype=float)
        if a.shape != (len(items),) or d.shape != (len(items),):
            raise ValueError("discrimination and intercept need one value per item")
        if not (np.isfinite(a).all() and np.isfinite(d).all()):
            raise ValueError("item parameters must be finite")
        return cls(
            kind=kind,
            items=tuple(str(item) for item in items),
            discrimination=a,
            intercept=d,
            status=(ESTIMATED,) * len(items),
            log_likelihood=float("nan"),
            n_parameters=len(items) + 1 if kind == "1pl" else 2 * len(items),
            n_models=0,
            n_observations=0,
            n_quad=n_quad,
            iterations=0,
            converged=True,
        )

    @property
    def estimated(self) -> np.ndarray:
        """Boolean mask of the items that have estimated parameters."""
        return np.array([state == ESTIMATED for state in self.status], dtype=bool)

    @property
    def difficulty(self) -> np.ndarray:
        """``b_j = -d_j / a_j``; ``-inf`` if solved by all, ``+inf`` if by none, NaN if no data."""
        out = np.full(len(self.items), np.nan)
        keep = self.estimated
        with np.errstate(divide="ignore", invalid="ignore"):
            out[keep] = -self.intercept[keep] / self.discrimination[keep]
        for index, state in enumerate(self.status):
            if state == ALL_CORRECT:
                out[index] = -np.inf
            elif state == ALL_INCORRECT:
                out[index] = np.inf
        return out

    def probability(self, theta: np.ndarray | list[float] | float) -> np.ndarray:
        """Probability of solving each item (columns) at each ability in ``theta`` (rows).

        Items solved by all or by none return 1 or 0; items without data return NaN.
        """
        ability = np.atleast_1d(np.asarray(theta, dtype=float))
        out = np.full((ability.size, len(self.items)), np.nan)
        keep = self.estimated
        out[:, keep] = sigmoid(
            ability[:, None] * self.discrimination[keep][None, :] + self.intercept[keep][None, :]
        )
        for index, state in enumerate(self.status):
            if state == ALL_CORRECT:
                out[:, index] = 1.0
            elif state == ALL_INCORRECT:
                out[:, index] = 0.0
        return out

    def to_dict(self) -> dict[str, object]:
        """A JSON-ready summary; values that are not finite become ``None``."""
        difficulty = self.difficulty
        return {
            "kind": self.kind,
            "method": "mml-em",
            "n_quad": self.n_quad,
            "log_likelihood": _finite(self.log_likelihood),
            "n_parameters": self.n_parameters,
            "n_models": self.n_models,
            "n_observations": self.n_observations,
            "iterations": self.iterations,
            "converged": self.converged,
            "items": [
                {
                    "item": item,
                    "status": self.status[index],
                    "discrimination": _finite(self.discrimination[index]),
                    "difficulty": _finite(difficulty[index]),
                    "intercept": _finite(self.intercept[index]),
                }
                for index, item in enumerate(self.items)
            ],
        }


@dataclass(frozen=True)
class AbilityEstimates:
    """Expected a posteriori (EAP) ability of each model with its posterior standard deviation."""

    models: tuple[str, ...]
    theta: np.ndarray
    sd: np.ndarray
    n_answered: np.ndarray

    def to_dict(self) -> list[dict[str, object]]:
        return [
            {
                "model": model,
                "theta": _finite(self.theta[index]),
                "sd": _finite(self.sd[index]),
                "n_answered": int(self.n_answered[index]),
            }
            for index, model in enumerate(self.models)
        ]


def _finite(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _posterior(
    a: np.ndarray,
    d: np.ndarray,
    solved: np.ndarray,
    failed: np.ndarray,
    nodes: np.ndarray,
    log_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior weight of every node for every model, and each model's marginal log-likelihood.

    ``solved`` and ``failed`` are 0/1 indicator matrices (models by items); a missing cell is 0 in
    both, so it drops out of the product over items.
    """
    z = np.outer(a, nodes) + d[:, None]
    log_p1 = -np.logaddexp(0.0, -z)
    log_joint = solved @ log_p1 + failed @ (log_p1 - z) + log_weights
    peak = log_joint.max(axis=1, keepdims=True)
    posterior = np.exp(log_joint - peak)
    total = posterior.sum(axis=1, keepdims=True)
    posterior /= total
    return posterior, (peak + np.log(total))[:, 0]


def _expected_loglik(
    a: np.ndarray, d: np.ndarray, r: np.ndarray, n0: np.ndarray, nodes: np.ndarray
) -> np.ndarray:
    """Expected complete-data log-likelihood per item from solved (r) and failed (n0) counts."""
    z = np.outer(a, nodes) + d[:, None]
    log_p1 = -np.logaddexp(0.0, -z)
    return (r * log_p1 + n0 * (log_p1 - z)).sum(axis=1)


def _m_step_2pl(
    a: np.ndarray,
    d: np.ndarray,
    r: np.ndarray,
    n0: np.ndarray,
    nodes: np.ndarray,
    *,
    max_newton: int = 30,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximise each item's expected log-likelihood over its slope and intercept."""
    n = r + n0
    nodes_sq = nodes * nodes
    q_now = _expected_loglik(a, d, r, n0, nodes)
    for _ in range(max_newton):
        p = sigmoid(np.outer(a, nodes) + d[:, None])
        residual = r - n * p
        weight = n * p * (1.0 - p)
        g_a = residual @ nodes
        g_d = residual.sum(axis=1)
        h_aa = weight @ nodes_sq + _RIDGE
        h_ad = weight @ nodes
        h_dd = weight.sum(axis=1) + _RIDGE
        det = h_aa * h_dd - h_ad * h_ad
        step_a = (h_dd * g_a - h_ad * g_d) / det
        step_d = (h_aa * g_d - h_ad * g_a) / det
        scale = np.ones_like(a)
        for _ in range(40):
            a_try = a + scale * step_a
            d_try = d + scale * step_d
            q_try = _expected_loglik(a_try, d_try, r, n0, nodes)
            worse = ~(q_try >= q_now - 1e-12 * (1.0 + np.abs(q_now)))
            if not worse.any():
                break
            scale[worse] *= 0.5
        else:
            a_try = np.where(worse, a, a_try)
            d_try = np.where(worse, d, d_try)
            q_try = np.where(worse, q_now, q_try)
        moved = max(float(np.abs(a_try - a).max()), float(np.abs(d_try - d).max()))
        a, d, q_now = a_try, d_try, q_try
        if moved < tol:
            break
    return a, d


def _m_step_1pl(
    a: np.ndarray,
    d: np.ndarray,
    r: np.ndarray,
    n0: np.ndarray,
    nodes: np.ndarray,
    *,
    max_newton: int = 30,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximise the expected log-likelihood over one shared slope and every intercept.

    The Hessian is an arrow matrix (the slope interacts with every intercept, the intercepts do
    not interact with each other), so the Newton system is solved through its Schur complement.
    """
    n = r + n0
    nodes_sq = nodes * nodes
    slope = float(a[0])
    q_now = float(_expected_loglik(np.full_like(d, slope), d, r, n0, nodes).sum())
    for _ in range(max_newton):
        p = sigmoid(slope * nodes[None, :] + d[:, None])
        residual = r - n * p
        weight = n * p * (1.0 - p)
        g_a = float((residual @ nodes).sum())
        g_d = residual.sum(axis=1)
        h_aa = float((weight @ nodes_sq).sum()) + _RIDGE
        h_ad = weight @ nodes
        h_dd = weight.sum(axis=1) + _RIDGE
        schur = max(h_aa - float((h_ad * h_ad / h_dd).sum()), _RIDGE)
        step_a = (g_a - float((h_ad * g_d / h_dd).sum())) / schur
        step_d = (g_d - h_ad * step_a) / h_dd
        scale = 1.0
        for _ in range(40):
            slope_try = slope + scale * step_a
            d_try = d + scale * step_d
            q_try = float(_expected_loglik(np.full_like(d, slope_try), d_try, r, n0, nodes).sum())
            if q_try >= q_now - 1e-12 * (1.0 + abs(q_now)):
                break
            scale *= 0.5
        else:
            slope_try, d_try, q_try = slope, d, q_now
        moved = max(abs(slope_try - slope), float(np.abs(d_try - d).max()))
        slope, d, q_now = slope_try, d_try, q_try
        if moved < tol:
            break
    return np.full_like(d, slope), d


def _item_status(scores: np.ndarray) -> np.ndarray:
    seen = ~np.isnan(scores)
    tried = seen.sum(axis=0)
    solved = np.where(seen, scores, 0.0).sum(axis=0)
    status = np.full(scores.shape[1], ESTIMATED, dtype=object)
    status[solved == tried] = ALL_CORRECT
    status[solved == 0] = ALL_INCORRECT
    status[tried == 0] = NO_DATA
    return status


def fit_mml(
    data: ResponseMatrix,
    kind: str = "2pl",
    *,
    n_quad: int = 61,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> IRTFit:
    """Fit a 1PL or 2PL model by marginal maximum likelihood.

    ``n_quad`` is the number of Gauss-Hermite nodes. EM stops when no item parameter moved by
    more than ``tol`` in an iteration, or after ``max_iter`` iterations (``converged`` is then
    False). The fit is oriented so that the discriminations sum to a positive number, which
    makes a higher ability mean a stronger model.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown model {kind!r}; use one of {', '.join(KINDS)}")
    scores = data.scores
    status = _item_status(scores)
    keep = status == ESTIMATED
    if int(keep.sum()) < _MIN_ITEMS[kind]:
        raise ValueError(
            f"the {kind} needs at least {_MIN_ITEMS[kind]} items that some models solve and "
            f"some do not; found {int(keep.sum())}"
        )
    solved = (scores[:, keep] == 1.0).astype(float)
    failed = (scores[:, keep] == 0.0).astype(float)
    answered = solved.sum(axis=1) + failed.sum(axis=1)

    nodes, log_weights = gauss_hermite(n_quad)
    share = solved.sum(axis=0) / (solved.sum(axis=0) + failed.sum(axis=0))
    a = np.ones(int(keep.sum()))
    # logit of the marginal share, widened for the spread of a standard normal ability
    d = np.log(share / (1.0 - share)) * np.sqrt(1.0 + 0.346)

    m_step = _m_step_1pl if kind == "1pl" else _m_step_2pl
    trace: list[float] = []
    converged = False
    iterations = 0
    while iterations < max_iter and not converged:
        iterations += 1
        posterior, loglik = _posterior(a, d, solved, failed, nodes, log_weights)
        trace.append(float(loglik.sum()))
        a_new, d_new = m_step(a, d, solved.T @ posterior, failed.T @ posterior, nodes)
        moved = max(float(np.abs(a_new - a).max()), float(np.abs(d_new - d).max()))
        a, d = a_new, d_new
        converged = moved < tol
    if a.sum() < 0.0:
        a = -a
    _, loglik = _posterior(a, d, solved, failed, nodes, log_weights)
    trace.append(float(loglik.sum()))

    discrimination = np.full(data.n_items, np.nan)
    intercept = np.full(data.n_items, np.nan)
    discrimination[keep] = a
    intercept[keep] = d
    intercept[status == ALL_CORRECT] = np.inf
    intercept[status == ALL_INCORRECT] = -np.inf
    n_estimated = int(keep.sum())
    return IRTFit(
        kind=kind,
        items=data.items,
        discrimination=discrimination,
        intercept=intercept,
        status=tuple(str(state) for state in status),
        log_likelihood=float(loglik.sum()),
        n_parameters=n_estimated + 1 if kind == "1pl" else 2 * n_estimated,
        n_models=int((answered > 0).sum()),
        n_observations=int(answered.sum()),
        n_quad=n_quad,
        iterations=iterations,
        converged=converged,
        trace=tuple(trace),
    )


def _aligned_indicators(fit: IRTFit, data: ResponseMatrix) -> tuple[np.ndarray, np.ndarray]:
    """Solved and failed indicators of ``data`` for the estimated items of ``fit``, in fit order.

    ``data`` may hold any subset of the fitted items in any order; fitted items it lacks count
    as missing. An item the fit does not know is an error.
    """
    column = {item: index for index, item in enumerate(data.items)}
    fitted = set(fit.items)
    unknown = [item for item in data.items if item not in fitted]
    if unknown:
        shown = ", ".join(unknown[:5])
        raise ValueError(f"{len(unknown)} items are not in the fit: {shown}")
    kept = [item for item, state in zip(fit.items, fit.status, strict=True) if state == ESTIMATED]
    solved = np.zeros((data.n_models, len(kept)))
    failed = np.zeros((data.n_models, len(kept)))
    for position, item in enumerate(kept):
        source = column.get(item)
        if source is None:
            continue
        cells = data.scores[:, source]
        solved[:, position] = cells == 1.0
        failed[:, position] = cells == 0.0
    return solved, failed


def estimate_abilities(
    fit: IRTFit, data: ResponseMatrix, *, n_quad: int | None = None
) -> AbilityEstimates:
    """EAP ability and posterior SD of every model in ``data`` under the item parameters of ``fit``.

    The posterior of a model is its likelihood on the estimated items it has results for, times
    the ``N(0, 1)`` density. A model without any such result gets the prior: ability 0, SD 1.
    """
    solved, failed = _aligned_indicators(fit, data)
    keep = fit.estimated
    nodes, log_weights = gauss_hermite(n_quad or fit.n_quad)
    posterior, _ = _posterior(
        fit.discrimination[keep], fit.intercept[keep], solved, failed, nodes, log_weights
    )
    theta = posterior @ nodes
    variance = posterior @ (nodes * nodes) - theta * theta
    return AbilityEstimates(
        models=data.models,
        theta=theta,
        sd=np.sqrt(np.maximum(variance, 0.0)),
        n_answered=(solved.sum(axis=1) + failed.sum(axis=1)).astype(int),
    )


def marginal_log_likelihood(
    fit: IRTFit, data: ResponseMatrix, *, n_quad: int | None = None
) -> float:
    """Marginal log-likelihood of ``data`` over the estimated items of ``fit``."""
    solved, failed = _aligned_indicators(fit, data)
    keep = fit.estimated
    nodes, log_weights = gauss_hermite(n_quad or fit.n_quad)
    _, loglik = _posterior(
        fit.discrimination[keep], fit.intercept[keep], solved, failed, nodes, log_weights
    )
    return float(loglik.sum())
