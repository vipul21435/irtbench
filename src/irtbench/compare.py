"""Compare the 1PL with the 2PL: information criteria, likelihood ratio, held-out cells."""

from __future__ import annotations

import numpy as np
from scipy import special

from irtbench.data import ResponseMatrix
from irtbench.mml import IRTFit, fit_mml, predict_probabilities


def information_criteria(fit: IRTFit) -> dict[str, float]:
    """AIC and BIC of a fit from its marginal log-likelihood.

    ``AIC = 2k - 2 logL`` and ``BIC = k log(n) - 2 logL`` with ``k`` the number of item
    parameters and ``n`` the number of models, which are the independent units of the marginal
    likelihood. With a slope prior the log-likelihood is taken at the penalised estimate and
    ``k`` still counts every slope, so both criteria are then harsher on the 2PL than they
    would be at its maximum-likelihood fit.
    """
    k, loglik = fit.n_parameters, fit.log_likelihood
    return {
        "log_likelihood": loglik,
        "n_parameters": float(k),
        "aic": 2.0 * k - 2.0 * loglik,
        "bic": k * float(np.log(max(fit.n_models, 1))) - 2.0 * loglik,
    }


def likelihood_ratio(smaller: IRTFit, larger: IRTFit) -> dict[str, float]:
    """Likelihood-ratio test of the 1PL (``smaller``) inside the 2PL (``larger``).

    The statistic is ``2 (logL_2PL - logL_1PL)`` on ``k_2PL - k_1PL`` degrees of freedom with a
    chi-square reference. The reference is an asymptotic result for many models and a fixed
    number of items, and for maximum-likelihood fits; with a slope prior on the 2PL the
    statistic is smaller than at the maximum, so the test is conservative.
    """
    if smaller.items != larger.items or smaller.status != larger.status:
        raise ValueError("the two fits must be of the same data")
    df = larger.n_parameters - smaller.n_parameters
    if df <= 0:
        raise ValueError("the second fit must have more parameters than the first")
    statistic = 2.0 * (larger.log_likelihood - smaller.log_likelihood)
    p_value = float(special.gammaincc(0.5 * df, 0.5 * max(statistic, 0.0)))
    return {"statistic": statistic, "df": float(df), "p_value": p_value}


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve: the chance a solved cell scores above an unsolved one.

    Ties count one half. NaN when only one class is present.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    ranks = (upper - 0.5 * (counts - 1))[inverse]  # average rank, 1-based
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def held_out_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Mean log-likelihood per cell, AUC and accuracy at 0.5 of predicted probabilities."""
    labels = np.asarray(labels, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    loglik = labels * np.log(probability) + (1.0 - labels) * np.log(1.0 - probability)
    return {
        "log_likelihood": float(loglik.mean()),
        "auc": auc(labels == 1.0, probability),
        "accuracy": float(((probability > 0.5) == (labels == 1.0)).mean()),
        "n_cells": float(labels.size),
    }


def cross_validate(
    data: ResponseMatrix,
    kinds: tuple[str, ...] = ("1pl", "2pl"),
    *,
    folds: int = 5,
    seed: int = 0,
    **fit_options: object,
) -> dict[str, dict[str, float]]:
    """Held-out fit of each model kind on masked cells.

    The observed cells are split at random into ``folds`` parts. Each part in turn is hidden,
    the model is fitted to the rest, and every hidden cell is predicted from its model's
    remaining cells (posterior predictive probability; an item left out of that fit gets its
    smoothed solve rate). The metrics pool all hidden cells. ``item_rate`` is the baseline that
    ignores the model and predicts each item's smoothed solve rate in the training cells.
    """
    if folds < 2:
        raise ValueError("cross-validation needs at least 2 folds")
    rows, cols = np.nonzero(data.observed)
    if rows.size < folds:
        raise ValueError("fewer observed cells than folds")
    fold = np.random.default_rng(seed).permutation(rows.size) % folds
    labels = data.scores[rows, cols]
    predicted = {kind: np.full(rows.size, np.nan) for kind in (*kinds, "item_rate")}
    for index in range(folds):
        hidden = fold == index
        scores = data.scores.copy()
        scores[rows[hidden], cols[hidden]] = np.nan
        training = ResponseMatrix(data.models, data.items, scores)
        seen = ~np.isnan(scores)
        rate = (np.where(seen, scores, 0.0).sum(axis=0) + 0.5) / (seen.sum(axis=0) + 1.0)
        predicted["item_rate"][hidden] = rate[cols[hidden]]
        for kind in kinds:
            fit = fit_mml(training, kind, **fit_options)  # type: ignore[arg-type]
            probability = predict_probabilities(fit, training)
            cells = probability[rows[hidden], cols[hidden]]
            predicted[kind][hidden] = np.where(np.isnan(cells), rate[cols[hidden]], cells)
    return {kind: held_out_metrics(labels, values) for kind, values in predicted.items()}
