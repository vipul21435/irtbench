"""Seeded simulation of binary results from a 1PL or 2PL model, for tests and examples."""

from __future__ import annotations

import numpy as np

from irtbench.data import ResponseMatrix
from irtbench.mml import KINDS, sigmoid


def simulate_responses(
    n_models: int,
    n_items: int,
    *,
    kind: str = "2pl",
    seed: int = 0,
    missing: float = 0.0,
    ability: np.ndarray | list[float] | None = None,
    discrimination: np.ndarray | list[float] | None = None,
    difficulty: np.ndarray | list[float] | None = None,
) -> tuple[ResponseMatrix, dict[str, np.ndarray]]:
    """Draw a results matrix and return it with the parameters that generated it.

    A cell is solved with probability ``sigmoid(a_j * (theta_i - b_j))``. Parameters not passed
    are drawn: abilities and difficulties from ``N(0, 1)``, discriminations equal to 1 for the
    1PL and log-normal (log-scale SD 0.3) for the 2PL. ``missing`` is the share of cells removed
    completely at random.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown model {kind!r}; use one of {', '.join(KINDS)}")
    if not 0.0 <= missing < 1.0:
        raise ValueError("missing must be in [0, 1)")
    rng = np.random.default_rng(seed)
    theta = _given_or_drawn(ability, n_models, rng.standard_normal(n_models), "ability")
    b = _given_or_drawn(difficulty, n_items, rng.standard_normal(n_items), "difficulty")
    drawn = np.ones(n_items) if kind == "1pl" else np.exp(rng.normal(0.0, 0.3, n_items))
    a = _given_or_drawn(discrimination, n_items, drawn, "discrimination")
    probability = sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
    scores = (rng.random((n_models, n_items)) < probability).astype(float)
    if missing > 0.0:
        scores[rng.random((n_models, n_items)) < missing] = np.nan
    models = tuple(f"model-{index + 1:0{len(str(n_models))}d}" for index in range(n_models))
    items = tuple(f"item-{index + 1:0{len(str(n_items))}d}" for index in range(n_items))
    truth = {"ability": theta, "discrimination": a, "difficulty": b}
    return ResponseMatrix(models, items, scores), truth


def _given_or_drawn(
    given: np.ndarray | list[float] | None, size: int, drawn: np.ndarray, name: str
) -> np.ndarray:
    if given is None:
        return drawn
    values = np.asarray(given, dtype=float)
    if values.shape != (size,):
        raise ValueError(f"{name} needs {size} values, got shape {values.shape}")
    return values
