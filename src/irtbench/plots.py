"""Charts of a fit, written as PNG files. Needs matplotlib (``pip install 'irtbench[plot]'``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from irtbench.diagnostics import test_information
from irtbench.mml import IRTFit

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
# one fixed colour per series name, so a series keeps its colour in every chart
SERIES = {
    "information": "#2a78d6",
    "cluster": "#eb6834",
    "random": "#1baf7a",
    "random_mean": "#eda100",
}
LABELS = {
    "information": "information",
    "cluster": "cluster",
    "random": "random",
    "random_mean": "random, plain mean",
}


def _pyplot():
    try:
        import matplotlib
    except ImportError:
        raise ValueError("charts need matplotlib: pip install 'irtbench[plot]'") from None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _axes(plt, title: str, xlabel: str, ylabel: str):
    figure, axes = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    axes.set_title(title, loc="left", color=INK, fontsize=12)
    axes.set_xlabel(xlabel, color=SECONDARY)
    axes.set_ylabel(ylabel, color=SECONDARY)
    axes.grid(True, color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED, labelcolor=SECONDARY, length=3)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(MUTED)
    return figure, axes


def _save(plt, figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return path


def plot_items(fit: IRTFit, path: Path) -> Path:
    """Discrimination against difficulty of every estimated item; negative slopes marked."""
    plt = _pyplot()
    keep = fit.estimated
    slope, difficulty = fit.discrimination[keep], fit.difficulty[keep]
    inside = np.abs(difficulty) <= 6.0
    backwards = slope < 0.0
    figure, axes = _axes(
        plt,
        f"Item parameters ({int(keep.sum())} items, {fit.kind})",
        "difficulty (ability at which half the models solve the item)",
        "discrimination",
    )
    axes.axhline(0.0, color=MUTED, linewidth=1.0)
    usual = inside & ~backwards
    axes.scatter(
        difficulty[usual], slope[usual], s=22, color=SERIES["information"], alpha=0.75,
        edgecolors=SURFACE, linewidths=0.6, label="positive discrimination",
    )
    odd = inside & backwards
    if odd.any():
        axes.scatter(
            difficulty[odd], slope[odd], s=46, color=SERIES["cluster"], marker="D",
            edgecolors=SURFACE, linewidths=0.8, label="negative discrimination",
        )
        axes.legend(frameon=False, labelcolor=SECONDARY, loc="upper right")
    hidden = int((~inside).sum())
    if hidden:
        axes.annotate(
            f"{hidden} items with |difficulty| > 6 not shown", (0.99, 0.02),
            xycoords="axes fraction", ha="right", color=MUTED, fontsize=8,
        )
    return _save(plt, figure, path)


def plot_information(fit: IRTFit, abilities: np.ndarray, path: Path) -> Path:
    """Test information over the ability range, with the models' abilities as ticks."""
    plt = _pyplot()
    abilities = np.asarray(abilities, dtype=float)
    low = min(-3.0, float(abilities.min()) - 0.5)
    high = max(3.0, float(abilities.max()) + 0.5)
    theta = np.linspace(low, high, 400)
    information = test_information(fit, theta)
    figure, axes = _axes(
        plt,
        "Test information (ticks: abilities of the models)",
        "ability",
        "information (1 / variance of an ability estimate)",
    )
    axes.plot(theta, information, color=SERIES["information"], linewidth=2.0)
    axes.plot(
        abilities, np.zeros_like(abilities), linestyle="none", marker="|", markersize=12,
        color=SECONDARY, markeredgewidth=0.8,
    )
    axes.set_ylim(bottom=0.0)
    return _save(plt, figure, path)


def plot_subsets(subsets: dict[str, object], path: Path) -> Path:
    """Error of the predicted full score against the number of items, one line per method."""
    plt = _pyplot()
    sizes = subsets["sizes"]
    ks = [int(k) for k in sizes]  # type: ignore[union-attr]
    names = [name for name in SERIES if name in next(iter(sizes.values()))]  # type: ignore[union-attr]
    figure, axes = _axes(
        plt,
        f"Predicting the full score of unseen models ({subsets['n_test']} per split, "
        f"{subsets['splits']} splits)",
        "items the model is run on",
        "mean absolute error (percentage points of score)",
    )
    for name in names:
        errors = [100.0 * sizes[str(k)][name]["mae"] for k in ks]  # type: ignore[index]
        axes.plot(
            ks, errors, color=SERIES[name], linewidth=2.0, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label=LABELS[name],
        )
    axes.set_xticks(ks)
    axes.set_ylim(bottom=0.0)
    axes.legend(frameon=False, labelcolor=SECONDARY)
    return _save(plt, figure, path)


def save_plots(
    directory: Path, fit: IRTFit, abilities: np.ndarray, subsets: dict[str, object] | None
) -> list[Path]:
    """Write ``items.png``, ``information.png`` and, with a subset evaluation, ``subsets.png``."""
    directory.mkdir(parents=True, exist_ok=True)
    written = [
        plot_items(fit, directory / "items.png"),
        plot_information(fit, abilities, directory / "information.png"),
    ]
    if subsets:
        written.append(plot_subsets(subsets, directory / "subsets.png"))
    return written
