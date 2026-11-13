"""Marginal maximum likelihood for the Rasch (1PL) and 2PL models, fitted by EM.

Model. A model ``i`` with ability ``theta_i`` solves item ``j`` with probability

    P(x_ij = 1 | theta_i) = sigmoid(a_j * theta_i + d_j),        theta_i ~ N(0, 1).

``a_j`` is the discrimination and ``d_j`` the intercept. The difficulty ``b_j = -d_j / a_j`` is
the ability at which the item is solved half the time. In the 1PL every item shares one
discrimination ``a``. That is the Rasch model ``sigmoid(t_i - c_j)`` with ``t_i = a * theta_i``
distributed ``N(0, a^2)`` and ``c_j = -d_j``, written so that the 1PL is the 2PL with equal slopes
and the two can be compared by likelihood.

Estimation. Abilities are integrated out against the normal density on a quadrature grid and the
item parameters maximise the resulting marginal likelihood. EM alternates

* E-step: each model's posterior weight on every node, then for every item the expected number
  of models at each node that tried it (``n``) and that solved it (``r``);
* M-step: a weighted logistic regression of ``r`` out of ``n`` on the node values, per item for
  the 2PL and with one shared slope for the 1PL, solved by Newton steps with backtracking.

Quadrature. EM starts on a Gauss-Hermite rule. A fixed rule is only right while its nodes are
closer together than the posteriors are wide, and a model that answered hundreds of items has a
posterior standard deviation well below 0.1. So the node gap is compared with the width of every
posterior (the curvature of the log-posterior at its mode), and when the rule is too coarse EM
continues on an equally spaced grid whose step is below the narrowest posterior and below the
scale ``1 / max|a_j|`` on which the steepest item curve changes.

Slope prior. With few models the likelihood of the 2PL often has no maximum in some slopes: an
item solved by exactly the strongest models is fitted better and better as its slope grows. The
2PL is therefore fitted with a normal prior on the ratio of each slope to a common slope ``c``:
``a_j / c ~ N(1, s^2)``, which adds ``-sum_j (a_j / c - 1)^2 / (2 s^2)`` to the log-likelihood.
``c`` is not preset; it takes the value that maximises this term, ``sum(a^2) / sum(a)``. The
penalty therefore depends only on how much the slopes differ from each other relative to their
size and says nothing about their overall scale, which stays with the likelihood. (A penalty on
the raw slopes would be cheapened by shrinking all of them and spreading the abilities out to
compensate.) ``s`` is ``slope_prior_sd``, default 0.5: a slope of zero sits two prior standard
deviations below the common slope, so a negative slope needs evidence. As ``s`` goes to zero the
fit becomes the 1PL, and as it grows, the maximum-likelihood 2PL. The result is the mode of the
posterior of the item parameters (a Bayes modal estimate). ``slope_prior_sd=None`` removes it.

Items whose observed cells are all 1 or all 0 have no finite maximum-likelihood intercept. They
are flagged and left out of the fit; in the limit they contribute nothing to the likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from irtbench.data import ResponseMatrix
from irtbench.quadrature import Grid, gauss_hermite_grid, uniform_grid

KINDS = ("1pl", "2pl")

ESTIMATED = "estimated"
ALL_CORRECT = "all_correct"
ALL_INCORRECT = "all_incorrect"
NO_DATA = "no_data"

_RIDGE = 1e-10
_MIN_ITEMS = {"1pl": 2, "2pl": 3}
_WARMUP = 30  # EM iterations on the starting rule before its resolution is checked
_REACH = 8.0  # a grid spans every posterior out to at least this many of its widths
_DROP = 32.0  # ... and further, until the log-posterior is this far below its peak
_PRIOR_REACH = 8.5  # ... but never further than this from the mode (the prior bounds the tail)
_COVERED = 0.8  # a grid is rebuilt when a posterior needs more than this share of its span
_SAFETY = 0.8  # a rebuilt grid is this much finer than required, so it stays adequate
_SLOPE_RULE = 1.25  # node gap allowed per unit of 1 / max|a|
_MAX_REFINEMENTS = 8


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    """Logistic function, stable for large arguments of either sign."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(z, dtype=float)))


@dataclass(frozen=True)
class IRTFit:
    """Item parameters of a fitted 1PL or 2PL model.

    ``discrimination`` and ``intercept`` have one entry per item. Items that were not estimated
    (see ``status``) hold NaN for the discrimination and ``+inf`` (solved by all), ``-inf``
    (solved by none) or NaN (no data) for the intercept. ``item_tried`` and ``item_solved`` count
    the observed and the solved cells of each item in the data the fit was made on.

    ``quadrature``, ``n_nodes`` and ``resolution`` describe the grid EM finished on:
    ``resolution`` is the largest ratio of node gap to posterior width over the models (and the
    matching ratio for the steepest item), which the fit keeps at or below its target.
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
    quadrature: str = "gauss-hermite"
    n_nodes: int = 0
    resolution: float = float("nan")
    item_tried: np.ndarray | None = None
    item_solved: np.ndarray | None = None
    slope_prior_sd: float | None = None
    slope_prior_mean: float | None = None

    def __post_init__(self) -> None:
        for name in ("item_tried", "item_solved"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.zeros(len(self.items), dtype=int))

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

    @property
    def smoothed_rate(self) -> np.ndarray:
        """``(solved + 0.5) / (tried + 1)`` per item: a solve rate that is never exactly 0 or 1.

        It is what the predictive functions use for items left out of the fit, whose fitted
        probability would otherwise be exactly 1 or 0.
        """
        return (self.item_solved + 0.5) / (self.item_tried + 1.0)

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
            "quadrature": self.quadrature,
            "n_nodes": self.n_nodes,
            "resolution": _finite(self.resolution),
            "slope_prior_sd": self.slope_prior_sd,
            "slope_prior_mean": self.slope_prior_mean,
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
                    "tried": int(self.item_tried[index]),
                    "solved": int(self.item_solved[index]),
                }
                for index, item in enumerate(self.items)
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> IRTFit:
        """Rebuild a fit from :meth:`to_dict` output (the ``fit`` block of the CLI's JSON)."""
        try:
            rows = list(payload["items"])  # type: ignore[call-overload]
            status = tuple(str(row["status"]) for row in rows)
            a = np.array([_or_nan(row["discrimination"]) for row in rows], dtype=float)
            d = np.array([_or_nan(row["intercept"]) for row in rows], dtype=float)
            d[[state == ALL_CORRECT for state in status]] = np.inf
            d[[state == ALL_INCORRECT for state in status]] = -np.inf
            return cls(
                kind=str(payload["kind"]),
                items=tuple(str(row["item"]) for row in rows),
                discrimination=a,
                intercept=d,
                status=status,
                log_likelihood=_or_nan(payload.get("log_likelihood")),
                n_parameters=int(payload["n_parameters"]),  # type: ignore[call-overload]
                n_models=int(payload.get("n_models", 0)),  # type: ignore[call-overload]
                n_observations=int(payload.get("n_observations", 0)),  # type: ignore[call-overload]
                n_quad=int(payload.get("n_quad", 61)),  # type: ignore[call-overload]
                iterations=int(payload.get("iterations", 0)),  # type: ignore[call-overload]
                converged=bool(payload.get("converged", True)),
                quadrature=str(payload.get("quadrature", "gauss-hermite")),
                n_nodes=int(payload.get("n_nodes", 0)),  # type: ignore[call-overload]
                resolution=_or_nan(payload.get("resolution")),
                item_tried=np.array([int(row.get("tried", 0)) for row in rows]),
                item_solved=np.array([int(row.get("solved", 0)) for row in rows]),
                slope_prior_sd=_or_none(payload.get("slope_prior_sd")),
                slope_prior_mean=_or_none(payload.get("slope_prior_mean")),
            )
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"not a fit written by irtbench: {error!r}") from None


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


def _or_nan(value: object) -> float:
    return float("nan") if value is None else float(value)  # type: ignore[arg-type]


def _or_none(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


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


def posterior_width(
    a: np.ndarray,
    d: np.ndarray,
    solved: np.ndarray,
    failed: np.ndarray,
    start: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mode of each model's ability posterior and its width from the curvature at the mode.

    The log-posterior is ``sum_j [x_j log P_j + (1 - x_j) log(1 - P_j)] - theta^2 / 2``. It is
    strictly concave, so damped Newton steps find the mode; the width is
    ``1 / sqrt(1 + sum_j a_j^2 P_j (1 - P_j))`` there, summed over the items the model answered.
    """
    theta = np.zeros(solved.shape[0]) if start is None else np.array(start, dtype=float)
    seen = solved + failed
    a_squared = a * a
    for _ in range(60):
        p = sigmoid(np.outer(theta, a) + d[None, :])
        gradient = (solved - seen * p) @ a - theta
        curvature = (seen * p * (1.0 - p)) @ a_squared + 1.0
        step = np.clip(gradient / curvature, -1.0, 1.0)
        theta = theta + step
        if float(np.abs(step).max()) < 1e-9:
            break
    p = sigmoid(np.outer(theta, a) + d[None, :])
    curvature = (seen * p * (1.0 - p)) @ a_squared + 1.0
    return theta, 1.0 / np.sqrt(curvature)


def _extent(
    a: np.ndarray,
    d: np.ndarray,
    solved: np.ndarray,
    failed: np.ndarray,
    mode: np.ndarray,
    width: np.ndarray,
) -> np.ndarray:
    """Distance from each mode beyond which that posterior is negligible.

    A posterior built from few answers has tails close to the prior's, far heavier than its
    curvature at the mode suggests, so the distance starts at a multiple of the width and grows
    until the log-posterior has dropped enough on both sides. The log-posterior is concave with
    curvature at most -1, which bounds the search: ``_PRIOR_REACH`` is always enough.
    """

    def log_posterior(theta: np.ndarray) -> np.ndarray:
        z = np.outer(theta, a) + d[None, :]
        log_p1 = -np.logaddexp(0.0, -z)
        return (solved * log_p1 + failed * (log_p1 - z)).sum(axis=1) - 0.5 * theta * theta

    peak = log_posterior(mode)
    reach = np.minimum(_REACH * width, _PRIOR_REACH)
    for _ in range(12):
        drop = np.minimum(peak - log_posterior(mode - reach), peak - log_posterior(mode + reach))
        short = (drop < _DROP) & (reach < _PRIOR_REACH)
        if not short.any():
            break
        reach = np.where(short, np.minimum(1.5 * reach, _PRIOR_REACH), reach)
    return reach


def _required_gap(a: np.ndarray, width: np.ndarray, resolution: float) -> float:
    """Largest node gap that resolves every posterior and the steepest item curve."""
    gap = float(width.min())
    steepest = float(np.abs(a).max()) if a.size else 0.0
    if steepest > 0.0:
        gap = min(gap, _SLOPE_RULE / steepest)
    return resolution * gap


def _resolution_ratio(
    grid: Grid, a: np.ndarray, mode: np.ndarray, width: np.ndarray, reach: np.ndarray
) -> float:
    """How coarse ``grid`` is for these posteriors: at most 1 means fine enough.

    Infinite when a posterior reaches beyond the grid.
    """
    lower = float((mode - _COVERED * reach).min())
    upper = float((mode + _COVERED * reach).max())
    if not grid.covers(lower, upper):
        return float("inf")
    gap = grid.gap_at(mode)
    ratio = float((gap / width).max())
    steepest = float(np.abs(a).max()) if a.size else 0.0
    return max(ratio, float(gap.max()) * steepest / _SLOPE_RULE)


def covering_grid(
    a: np.ndarray,
    d: np.ndarray,
    solved: np.ndarray,
    failed: np.ndarray,
    *,
    resolution: float = 1.0,
    max_nodes: int = 2001,
    start: np.ndarray | None = None,
) -> tuple[Grid, float]:
    """An equally spaced grid fine and wide enough for the posteriors of these models.

    Returns the grid and its resolution ratio (above 1 only when ``max_nodes`` was too few).
    """
    mode, width = posterior_width(a, d, solved, failed, start)
    reach = _extent(a, d, solved, failed, mode, width)
    lower = float((mode - reach).min())
    upper = float((mode + reach).max())
    spacing = _SAFETY * _required_gap(a, width, resolution)
    spacing = max(spacing, (upper - lower) / (max_nodes - 1))
    grid = uniform_grid(spacing, lower, upper)
    return grid, _resolution_ratio(grid, a, mode, width, reach) / resolution


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
    precision: float = 0.0,
    centre: float = 0.0,
    *,
    max_newton: int = 30,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximise each item's expected log-likelihood over its slope and intercept.

    With ``precision > 0`` each slope also carries the log-density of a normal prior with that
    precision and mean ``centre``.
    """
    n = r + n0
    nodes_sq = nodes * nodes

    def objective(slope: np.ndarray, intercept: np.ndarray) -> np.ndarray:
        penalty = 0.5 * precision * (slope - centre) ** 2
        return _expected_loglik(slope, intercept, r, n0, nodes) - penalty

    q_now = objective(a, d)
    for _ in range(max_newton):
        p = sigmoid(np.outer(a, nodes) + d[:, None])
        residual = r - n * p
        weight = n * p * (1.0 - p)
        g_a = residual @ nodes - precision * (a - centre)
        g_d = residual.sum(axis=1)
        h_aa = weight @ nodes_sq + precision + _RIDGE
        h_ad = weight @ nodes
        h_dd = weight.sum(axis=1) + _RIDGE
        det = h_aa * h_dd - h_ad * h_ad
        step_a = (h_dd * g_a - h_ad * g_d) / det
        step_d = (h_aa * g_d - h_ad * g_a) / det
        scale = np.ones_like(a)
        for _ in range(40):
            a_try = a + scale * step_a
            d_try = d + scale * step_d
            q_try = objective(a_try, d_try)
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
    precision: float = 0.0,
    centre: float = 0.0,
    *,
    max_newton: int = 30,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximise the expected log-likelihood over one shared slope and every intercept.

    ``precision`` and ``centre`` are accepted for a common signature; the 1PL has no prior.

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


def _em(
    a: np.ndarray,
    d: np.ndarray,
    solved: np.ndarray,
    failed: np.ndarray,
    grid: Grid,
    kind: str,
    max_iter: int,
    tol: float,
    prior_sd: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int, bool, list[float]]:
    """Run EM on one grid; returns the parameters, iterations used, convergence and the trace.

    The trace holds the objective before each M-step: the marginal log-likelihood plus, with a
    slope prior, its log-density (up to a constant).

    With a slope prior the M-step treats the common slope ``c`` as fixed at its current best
    value, which makes the prior an independent normal on each slope; ``c`` is then updated.
    Both moves raise the objective.

    After each M-step the ability distribution is standardised again: the mean ``m`` and the
    variance ``v`` of the abilities under the current posteriors are what an ability
    distribution ``N(m, v)`` with free parameters would be set to, and moving to
    ``(theta - m) / sqrt(v)`` turns the item parameters into ``a * sqrt(v)`` and ``d + a * m``.
    This expanded step matters on long tests, where the answers pin down ``a * theta`` closely
    and plain EM, which can only change the scale through the prior on abilities, needs
    hundreds of iterations. The slope prior is unchanged by it, because it only sees ratios of
    slopes. On a coarse grid the rescaling is not exact; if it ever lowers the objective it is
    undone and switched off.
    """
    m_step = _m_step_1pl if kind == "1pl" else _m_step_2pl
    nodes = grid.nodes
    trace: list[float] = []
    converged = False
    expand = True
    plain: tuple[np.ndarray, np.ndarray] | None = None
    steps = 0
    while steps < max_iter and not converged:
        steps += 1
        posterior, loglik = _posterior(a, d, solved, failed, nodes, grid.log_weights)
        objective = float(loglik.sum()) + _log_prior(a, prior_sd)
        if expand and plain is not None and objective < trace[-1] - 1e-9 * (1.0 + abs(trace[-1])):
            a, d = plain
            expand = False
            posterior, loglik = _posterior(a, d, solved, failed, nodes, grid.log_weights)
            objective = float(loglik.sum()) + _log_prior(a, prior_sd)
        trace.append(objective)
        centre = _common_slope(a)
        precision = 0.0 if prior_sd is None or centre == 0.0 else 1.0 / (prior_sd * centre) ** 2
        a_new, d_new = m_step(
            a, d, solved.T @ posterior, failed.T @ posterior, nodes, precision, centre
        )
        if expand:
            plain = (a_new, d_new)
            mass = posterior.sum(axis=0) / posterior.shape[0]
            mean = float(mass @ nodes)
            variance = float(mass @ (nodes * nodes)) - mean * mean
            if variance > 0.0:
                a_new, d_new = a_new * np.sqrt(variance), d_new + a_new * mean
        moved = max(float(np.abs(a_new - a).max()), float(np.abs(d_new - d).max()))
        a, d = a_new, d_new
        converged = moved < tol
    return a, d, steps, converged, trace


def _common_slope(a: np.ndarray) -> float:
    """The common slope ``c`` that maximises the slope prior: ``sum(a^2) / sum(a)``."""
    total = float(a.sum())
    return float(a @ a) / total if abs(total) > 1e-12 else 0.0


def _log_prior(a: np.ndarray, prior_sd: float | None) -> float:
    """Log-density of the slope prior at its best common slope, up to a constant.

    ``-sum((a_j / c - 1)^2) / (2 s^2)`` with ``c = sum(a^2) / sum(a)`` equals
    ``-(J - sum(a)^2 / sum(a^2)) / (2 s^2)``.
    """
    if prior_sd is None:
        return 0.0
    squares = float(a @ a)
    if squares <= 0.0:
        return 0.0
    return -0.5 * (a.size - float(a.sum()) ** 2 / squares) / (prior_sd * prior_sd)


def _run(
    a: np.ndarray,
    d: np.ndarray,
    solved: np.ndarray,
    failed: np.ndarray,
    kind: str,
    grid: Grid,
    *,
    resolution: float | None,
    max_nodes: int,
    max_iter: int,
    tol: float,
    prior_sd: float | None = None,
) -> tuple[np.ndarray, np.ndarray, Grid, int, bool, list[float], float]:
    """EM with the grid checked against the posteriors and rebuilt when it is too coarse.

    Returns the parameters, the grid EM finished on, the iterations used, convergence, the
    objective trace on that grid and the resolution ratio relative to its target.
    """
    budget = max_iter
    iterations = 0
    refinements = 0
    warming = resolution is not None
    mode: np.ndarray | None = None
    ratio = float("nan")
    while True:
        limit = min(budget, _WARMUP) if warming else budget
        a, d, steps, converged, trace = _em(
            a, d, solved, failed, grid, kind, limit, tol, prior_sd
        )
        budget -= steps
        iterations += steps
        warming = False
        if resolution is None:
            break
        mode, width = posterior_width(a, d, solved, failed, mode)
        reach = _extent(a, d, solved, failed, mode, width)
        ratio = _resolution_ratio(grid, a, mode, width, reach) / resolution
        if ratio <= 1.0:
            if converged or budget <= 0:
                break
            continue
        if refinements >= _MAX_REFINEMENTS or budget <= 0:
            break
        finer, _ = covering_grid(
            a, d, solved, failed, resolution=resolution, max_nodes=max_nodes, start=mode
        )
        if finer.size >= max_nodes and grid.rule == "uniform" and grid.size >= max_nodes:
            break
        grid = finer
        refinements += 1
    return a, d, grid, iterations, converged, trace, ratio


def fit_mml(
    data: ResponseMatrix,
    kind: str = "2pl",
    *,
    slope_prior_sd: float | None = 0.5,
    n_quad: int = 61,
    resolution: float | None = 1.0,
    max_nodes: int = 2001,
    max_iter: int = 2000,
    tol: float = 1e-6,
    start: tuple[np.ndarray, np.ndarray] | None = None,
) -> IRTFit:
    """Fit a 1PL or 2PL model by marginal maximum likelihood.

    For the 2PL, ``slope_prior_sd`` is the standard deviation of the normal prior on the ratio
    of each slope to a common slope that is estimated with them (see the module docstring).
    ``slope_prior_sd=None`` gives the plain maximum-likelihood fit, which can fail to converge
    when there are few models. The 1PL has one slope and takes no prior. The fit records the
    common slope as ``slope_prior_mean``.

    EM starts on the ``n_quad``-point Gauss-Hermite rule. ``resolution`` is the largest node gap
    allowed, in units of the narrowest posterior width (and of ``1.25 / max|a_j|``); when the
    rule is coarser than that, EM continues on an equally spaced grid that meets it, with at
    most ``max_nodes`` nodes. ``resolution=None`` keeps the Gauss-Hermite rule whatever the
    posteriors look like, which is only right for short tests.

    EM stops when no item parameter moved by more than ``tol`` in an iteration, or after
    ``max_iter`` iterations (``converged`` is then False). The fit is oriented so that the
    discriminations sum to a positive number, which makes a higher ability mean a stronger
    model.

    ``start`` warm-starts EM from ``(discrimination, intercept)`` arrays with one value per item
    of ``data``, for example those of an earlier fit; entries of items that are not estimated
    are ignored. Without it every slope starts at 1 and every intercept at a value matched to
    the share of models that solved the item.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown model {kind!r}; use one of {', '.join(KINDS)}")
    if resolution is not None and not resolution > 0.0:
        raise ValueError("resolution must be positive, or None for a fixed Gauss-Hermite rule")
    if slope_prior_sd is not None and not slope_prior_sd > 0.0:
        raise ValueError("slope_prior_sd must be positive, or None for no prior")
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

    if start is None:
        share = solved.sum(axis=0) / (solved.sum(axis=0) + failed.sum(axis=0))
        a = np.ones(int(keep.sum()))
        # logit of the marginal share, widened for the spread of a standard normal ability
        d = np.log(share / (1.0 - share)) * np.sqrt(1.0 + 0.346)
    else:
        a_start = np.asarray(start[0], dtype=float)
        d_start = np.asarray(start[1], dtype=float)
        if a_start.shape != (data.n_items,) or d_start.shape != (data.n_items,):
            raise ValueError("start needs one discrimination and one intercept per item")
        a, d = a_start[keep].copy(), d_start[keep].copy()
        if not (np.isfinite(a).all() and np.isfinite(d).all()):
            raise ValueError("start must be finite for every item that is estimated")
    if kind == "1pl":
        a = np.full_like(a, a.mean())

    if kind == "1pl":
        slope_prior_sd = None
    a, d, grid, iterations, converged, trace, ratio = _run(
        a,
        d,
        solved,
        failed,
        kind,
        gauss_hermite_grid(n_quad),
        resolution=resolution,
        max_nodes=max_nodes,
        max_iter=max_iter,
        tol=tol,
        prior_sd=slope_prior_sd,
    )
    if a.sum() < 0.0:
        a = -a
    _, loglik = _posterior(a, d, solved, failed, grid.nodes, grid.log_weights)
    trace.append(float(loglik.sum()) + _log_prior(a, slope_prior_sd))

    discrimination = np.full(data.n_items, np.nan)
    intercept = np.full(data.n_items, np.nan)
    discrimination[keep] = a
    intercept[keep] = d
    intercept[status == ALL_CORRECT] = np.inf
    intercept[status == ALL_INCORRECT] = -np.inf
    n_estimated = int(keep.sum())
    seen = ~np.isnan(scores)
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
        quadrature=grid.rule,
        n_nodes=grid.size,
        resolution=ratio,
        item_tried=seen.sum(axis=0).astype(int),
        item_solved=np.where(seen, scores, 0.0).sum(axis=0).astype(int),
        slope_prior_sd=slope_prior_sd,
        slope_prior_mean=None if slope_prior_sd is None else _common_slope(a),
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


def score_posterior(
    fit: IRTFit, data: ResponseMatrix, *, resolution: float = 1.0, max_nodes: int = 2001
) -> tuple[Grid, np.ndarray, np.ndarray, np.ndarray]:
    """Posterior of every model in ``data`` over a grid made for those posteriors.

    Returns the grid, the posterior weights (models by nodes, rows summing to 1), each model's
    marginal log-likelihood, and the number of estimated items each model has results for.
    """
    if data.n_models == 0:
        raise ValueError("there are no models to score")
    solved, failed = _aligned_indicators(fit, data)
    keep = fit.estimated
    a, d = fit.discrimination[keep], fit.intercept[keep]
    grid, _ = covering_grid(a, d, solved, failed, resolution=resolution, max_nodes=max_nodes)
    posterior, loglik = _posterior(a, d, solved, failed, grid.nodes, grid.log_weights)
    return grid, posterior, loglik, (solved.sum(axis=1) + failed.sum(axis=1)).astype(int)


def estimate_abilities(
    fit: IRTFit, data: ResponseMatrix, *, resolution: float = 1.0
) -> AbilityEstimates:
    """EAP ability and posterior SD of every model in ``data`` under the item parameters of ``fit``.

    The posterior of a model is its likelihood on the estimated items it has results for, times
    the ``N(0, 1)`` density. A model without any such result gets the prior: ability 0, SD 1.
    """
    grid, posterior, _, answered = score_posterior(fit, data, resolution=resolution)
    theta = posterior @ grid.nodes
    variance = posterior @ (grid.nodes * grid.nodes) - theta * theta
    return AbilityEstimates(
        models=data.models,
        theta=theta,
        sd=np.sqrt(np.maximum(variance, 0.0)),
        n_answered=answered,
    )


def marginal_log_likelihood(
    fit: IRTFit, data: ResponseMatrix, *, resolution: float = 1.0
) -> float:
    """Marginal log-likelihood of ``data`` over the estimated items of ``fit``."""
    _, _, loglik, _ = score_posterior(fit, data, resolution=resolution)
    return float(loglik.sum())


def predict_probabilities(
    fit: IRTFit, data: ResponseMatrix, *, resolution: float = 1.0
) -> np.ndarray:
    """Probability that each model in ``data`` solves each item of ``fit``, given its results.

    For an estimated item this is the posterior predictive probability: the item curve averaged
    over the model's ability posterior, which is built from whatever cells ``data`` holds. For an
    item left out of the fit it is the item's smoothed solve rate, the same for every model; for
    an item without data it is NaN. The result has one row per model and one column per fitted
    item, in the order of ``fit.items``.
    """
    grid, posterior, _, _ = score_posterior(fit, data, resolution=resolution)
    keep = fit.estimated
    curves = sigmoid(np.outer(grid.nodes, fit.discrimination[keep]) + fit.intercept[keep][None, :])
    out = np.tile(fit.smoothed_rate, (data.n_models, 1))
    out[:, keep] = posterior @ curves
    out[:, [state == NO_DATA for state in fit.status]] = np.nan
    return out
