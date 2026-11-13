"""Choose a small set of items and estimate the full score of a model from its answers on it.

The estimator follows the idea of IRT-based score estimation used by tinyBenchmarks: the
answers on the subset locate the model's ability, and the items it was not run on are filled in
with the probability the fitted item curves give at that ability. Here the ability is not a
point estimate; each missing item gets its posterior predictive probability.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import special

from irtbench.data import ResponseMatrix
from irtbench.mml import (
    NO_DATA,
    IRTFit,
    estimate_abilities,
    fit_mml,
    predict_probabilities,
    sigmoid,
)

METHODS = ("information", "cluster", "random")


def spread_abilities(abilities: np.ndarray | None = None, count: int = 9) -> np.ndarray:
    """Abilities spread over the population: evenly spaced quantiles from 10 % to 90 %.

    The quantiles are those of ``abilities`` when given and of ``N(0, 1)`` otherwise.
    """
    levels = np.linspace(0.1, 0.9, count)
    if abilities is None:
        return special.ndtri(levels)
    return np.quantile(np.asarray(abilities, dtype=float), levels)


def _candidates(fit: IRTFit) -> np.ndarray:
    return np.flatnonzero(fit.estimated)


def _curves(fit: IRTFit, targets: np.ndarray) -> np.ndarray:
    keep = _candidates(fit)
    return sigmoid(np.outer(fit.discrimination[keep], targets) + fit.intercept[keep][:, None])


def select_information(fit: IRTFit, k: int, targets: np.ndarray) -> list[int]:
    """Items with the most Fisher information at abilities spread over the population.

    The targets are visited in turn; each visit takes the unused item with the largest
    information ``a^2 P (1 - P)`` at that ability, so the subset measures weak and strong
    models alike.
    """
    keep = _candidates(fit)
    curves = _curves(fit, targets)
    information = fit.discrimination[keep][:, None] ** 2 * curves * (1.0 - curves)
    free = np.ones(keep.size, dtype=bool)
    chosen: list[int] = []
    for slot in range(k):
        column = np.where(free, information[:, slot % targets.size], -np.inf)
        best = int(np.argmax(column))
        free[best] = False
        chosen.append(int(keep[best]))
    return chosen


def select_clusters(fit: IRTFit, k: int, targets: np.ndarray, seed: int = 0) -> list[int]:
    """One representative per cluster of items with similar fitted curves.

    Each item is described by its fitted probability at the target abilities, which is a
    bounded function of its two parameters. The items are grouped by k-means (seeded, with
    the usual distance-weighted initialisation) and the item nearest each centre is taken.
    """
    keep = _candidates(fit)
    points = _curves(fit, targets)
    rng = np.random.default_rng(seed)
    centres = points[[int(rng.integers(keep.size))]]
    while centres.shape[0] < k:
        distance = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        total = float(distance.sum())
        weights = distance / total if total > 0.0 else np.full(keep.size, 1.0 / keep.size)
        centres = np.vstack([centres, points[int(rng.choice(keep.size, p=weights))]])
    for _ in range(100):
        distance = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        label = distance.argmin(axis=1)
        updated = centres.copy()
        for cluster in range(k):
            members = points[label == cluster]
            if members.size:
                updated[cluster] = members.mean(axis=0)
        if np.allclose(updated, centres, atol=1e-10):
            break
        centres = updated
    distance = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    free = np.ones(keep.size, dtype=bool)
    chosen: list[int] = []
    for cluster in range(k):
        column = np.where(free, distance[:, cluster], np.inf)
        best = int(np.argmin(column))
        free[best] = False
        chosen.append(int(keep[best]))
    return chosen


def select_items(
    fit: IRTFit,
    k: int,
    method: str = "information",
    *,
    abilities: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[str, ...]:
    """Choose ``k`` items of ``fit`` by ``method``: ``information``, ``cluster`` or ``random``.

    Only items with estimated parameters are eligible, since an item solved by all or by none
    says nothing about a new model under the fit. ``abilities`` (for example those of the
    models the fit was made on) set the abilities the first two methods aim at.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {', '.join(METHODS)}")
    eligible = _candidates(fit)
    if not 1 <= k <= eligible.size:
        raise ValueError(f"k must be between 1 and {eligible.size}, the number of estimated items")
    if method == "random":
        picked = np.random.default_rng(seed).choice(eligible, size=k, replace=False)
        return tuple(fit.items[int(j)] for j in picked)
    targets = spread_abilities(abilities)
    if method == "information":
        return tuple(fit.items[j] for j in select_information(fit, k, targets))
    return tuple(fit.items[j] for j in select_clusters(fit, k, targets, seed))


def predict_scores(fit: IRTFit, data: ResponseMatrix) -> np.ndarray:
    """Estimated share of all fitted items each model in ``data`` would solve.

    The cells ``data`` holds count as they are. Every other fitted item counts with its
    predicted probability given those cells (see :func:`irtbench.mml.predict_probabilities`).
    Items the fit has no data for are left out of the average.
    """
    probability = predict_probabilities(fit, data)
    column = {item: j for j, item in enumerate(fit.items)}
    for source, item in enumerate(data.items):
        cells = data.scores[:, source]
        seen = ~np.isnan(cells)
        probability[seen, column[item]] = cells[seen]
    counted = np.array([state != NO_DATA for state in fit.status])
    return probability[:, counted].mean(axis=1)


def evaluate_subsets(
    data: ResponseMatrix,
    sizes: Sequence[int] = (25, 50, 100),
    *,
    methods: Sequence[str] = METHODS,
    splits: int = 20,
    test_share: float = 0.25,
    random_draws: int = 10,
    seed: int = 0,
    kind: str = "2pl",
    **fit_options: object,
) -> dict[str, object]:
    """Error of predicting the full score of unseen models from a subset of items.

    In each split the models are divided at random; the IRT model is fitted to the training
    models only, the items are chosen from that fit only, and each test model's full score is
    predicted from its answers on the chosen items. The error is the mean absolute difference
    from the test model's actual share of solved items (over its observed cells), in score
    points between 0 and 1.

    For ``random`` the error is averaged over ``random_draws`` subsets per split.
    ``random_mean`` is the baseline without any model: the plain mean of the answers on
    ``random_draws`` subsets of the same size per split, drawn uniformly from every item of
    ``data``. That includes the items the fit left out (solved by all or by none of the
    training models), which the true score counts too; a mean over the estimated items alone
    would be biased. The result gives, per size and method, the mean and standard deviation
    over splits.
    """
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; use one of {', '.join(METHODS)}")
    n_test = int(round(test_share * data.n_models))
    if not 1 <= n_test < data.n_models - 1:
        raise ValueError("test_share leaves too few models to train or to test on")
    names = [*methods, "random_mean"] if "random" in methods else list(methods)
    errors: dict[int, dict[str, list[float]]] = {
        int(k): {name: [] for name in names} for k in sizes
    }
    for split in range(splits):
        rng = np.random.default_rng([seed, split])
        # the baseline has its own stream, so the model-based figures do not depend on it
        baseline_rng = np.random.default_rng([seed, split, 1])
        order = rng.permutation(data.n_models)
        test, train = data.take_models(order[:n_test]), data.take_models(order[n_test:])
        fit = fit_mml(train, kind, **fit_options)  # type: ignore[arg-type]
        abilities = estimate_abilities(fit, train).theta
        truth = test.model_means()
        position = {item: j for j, item in enumerate(test.items)}
        for k in errors:
            for method in methods:
                draws = random_draws if method == "random" else 1
                model_based = []
                for draw in range(draws):
                    chosen = select_items(
                        fit, k, method, abilities=abilities, seed=int(rng.integers(2**31)) + draw
                    )
                    answers = test.take_items([position[item] for item in chosen])
                    model_based.append(float(np.abs(predict_scores(fit, answers) - truth).mean()))
                errors[k][method].append(float(np.mean(model_based)))
                if method == "random":
                    plain = []
                    for _ in range(random_draws):
                        drawn = baseline_rng.choice(data.n_items, size=k, replace=False)
                        means = test.take_items(drawn).model_means()
                        plain.append(float(np.abs(means - truth).mean()))
                    errors[k]["random_mean"].append(float(np.mean(plain)))
    return {
        "splits": splits,
        "n_train": data.n_models - n_test,
        "n_test": n_test,
        "kind": kind,
        "sizes": {
            str(k): {
                name: {"mae": float(np.mean(values)), "sd": float(np.std(values))}
                for name, values in by_method.items()
            }
            for k, by_method in errors.items()
        },
    }
