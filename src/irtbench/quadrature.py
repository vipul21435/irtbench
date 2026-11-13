"""Gauss-Hermite quadrature for expectations over a standard normal ability."""

from __future__ import annotations

import numpy as np

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def gauss_hermite(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and log-weights of the ``n``-point Gauss-Hermite rule for ``Z ~ N(0, 1)``.

    ``sum(exp(log_weights) * f(nodes))`` approximates ``E[f(Z)]`` and is exact when ``f`` is a
    polynomial of degree at most ``2n - 1``. The weights are returned as logarithms because the
    outer ones are far below anything that survives a product with a likelihood in linear space.
    """
    if n < 2:
        raise ValueError("a quadrature rule needs at least 2 nodes")
    nodes, weights = np.polynomial.hermite_e.hermegauss(n)
    if not (np.all(np.isfinite(weights)) and np.all(weights > 0.0)):
        raise ValueError(f"the {n}-point rule has weights too small to represent; use fewer nodes")
    return nodes, np.log(weights) - _LOG_SQRT_2PI
