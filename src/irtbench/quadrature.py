"""Quadrature grids for integrals over a standard normal ability.

Two rules are provided. The Gauss-Hermite rule is the classical choice and is very accurate
when the integrand is smooth on the scale of its node gaps. A model that answered hundreds of
items has a posterior far narrower than those gaps, so for long benchmarks the fit switches to an
equally spaced grid whose spacing is tied to the narrowest posterior. On a peaked, smooth
integrand the equally spaced (trapezoid) rule converges geometrically once the spacing is below
the width of the peak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


@dataclass(frozen=True)
class Grid:
    """Quadrature nodes with log-weights for expectations over ``Z ~ N(0, 1)``.

    ``sum(exp(log_weights) * f(nodes))`` approximates ``E[f(Z)]``. ``rule`` is
    ``"gauss-hermite"`` or ``"uniform"``.
    """

    nodes: np.ndarray
    log_weights: np.ndarray
    rule: str

    @property
    def size(self) -> int:
        return int(self.nodes.size)

    def gap_at(self, theta: np.ndarray) -> np.ndarray:
        """Distance between the two nodes around each value of ``theta``."""
        index = np.clip(np.searchsorted(self.nodes, theta), 1, self.size - 1)
        return self.nodes[index] - self.nodes[index - 1]

    def covers(self, lower: float, upper: float) -> bool:
        """Whether the nodes span ``[lower, upper]``."""
        return bool(self.nodes[0] <= lower and upper <= self.nodes[-1])


def gauss_hermite(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and log-weights of the ``n``-point Gauss-Hermite rule for ``Z ~ N(0, 1)``.

    ``sum(exp(log_weights) * f(nodes))`` approximates ``E[f(Z)]`` and is exact when ``f`` is a
    polynomial of degree at most ``2n - 1``. The weights are returned as logarithms because the
    outer ones are far below anything that survives a product with a likelihood in linear space.
    """
    if n < 2:
        raise ValueError("a quadrature rule needs at least 2 nodes")
    with np.errstate(all="ignore"):
        nodes, weights = np.polynomial.hermite_e.hermegauss(n)
    if not (np.all(np.isfinite(weights)) and np.all(weights > 0.0)):
        raise ValueError(f"the {n}-point rule has weights too small to represent; use fewer nodes")
    return nodes, np.log(weights) - _LOG_SQRT_2PI


def gauss_hermite_grid(n: int) -> Grid:
    """The ``n``-point Gauss-Hermite rule as a :class:`Grid`."""
    nodes, log_weights = gauss_hermite(n)
    return Grid(nodes, log_weights, "gauss-hermite")


def uniform_grid(spacing: float, lower: float, upper: float) -> Grid:
    """Equally spaced nodes on ``[lower, upper]`` weighted by the standard normal density.

    The weights are ``density(node) * step`` and are not renormalised, so the rule approximates
    the integral over the whole line as long as the integrand is negligible outside the range.
    The step is at most ``spacing``.
    """
    if not (spacing > 0.0 and upper > lower):
        raise ValueError("uniform_grid needs a positive spacing and upper > lower")
    count = max(int(np.ceil((upper - lower) / spacing)) + 1, 3)
    nodes = np.linspace(lower, upper, count)
    step = (upper - lower) / (count - 1)
    return Grid(nodes, -0.5 * nodes * nodes - _LOG_SQRT_2PI + np.log(step), "uniform")
