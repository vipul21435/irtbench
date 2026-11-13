import numpy as np
import pytest
from scipy import stats

from irtbench.compare import (
    auc,
    cross_validate,
    held_out_metrics,
    information_criteria,
    likelihood_ratio,
)
from irtbench.mml import fit_mml
from irtbench.simulate import simulate_responses


def test_information_criteria_formulas():
    data, _ = simulate_responses(200, 6, kind="1pl", seed=51)
    fit = fit_mml(data, "1pl")
    criteria = information_criteria(fit)
    assert criteria["n_parameters"] == 7
    assert criteria["aic"] == pytest.approx(14 - 2 * fit.log_likelihood)
    assert criteria["bic"] == pytest.approx(7 * np.log(200) - 2 * fit.log_likelihood)


def test_likelihood_ratio_matches_scipy_chi_square():
    data, _ = simulate_responses(1500, 8, kind="2pl", seed=52)
    one = fit_mml(data, "1pl")
    two = fit_mml(data, "2pl", slope_prior_sd=None)
    test = likelihood_ratio(one, two)
    assert test["df"] == 7
    assert test["statistic"] == pytest.approx(2 * (two.log_likelihood - one.log_likelihood))
    assert test["p_value"] == pytest.approx(stats.chi2.sf(test["statistic"], 7), rel=1e-10)
    assert test["p_value"] < 0.01  # the slopes really differ in this simulation
    with pytest.raises(ValueError, match="more parameters"):
        likelihood_ratio(two, one)
    other, _ = simulate_responses(1500, 9, kind="2pl", seed=52)
    with pytest.raises(ValueError, match="same data"):
        likelihood_ratio(one, fit_mml(other, "2pl"))


def test_likelihood_ratio_keeps_the_1pl_when_it_is_true():
    data, _ = simulate_responses(1500, 8, kind="1pl", seed=53)
    test = likelihood_ratio(fit_mml(data, "1pl"), fit_mml(data, "2pl", slope_prior_sd=None))
    assert test["statistic"] >= -1e-6
    assert test["p_value"] > 0.01


def test_auc_matches_the_mann_whitney_statistic():
    rng = np.random.default_rng(54)
    labels = rng.random(300) < 0.4
    scores = np.round(rng.normal(labels * 0.8, 1.0), 1)  # rounding creates ties
    u = stats.mannwhitneyu(scores[labels], scores[~labels]).statistic
    assert auc(labels, scores) == pytest.approx(u / (labels.sum() * (~labels).sum()))
    assert auc(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2])) == 1.0
    assert auc(np.array([1, 0]), np.array([0.5, 0.5])) == 0.5
    assert np.isnan(auc(np.array([1, 1]), np.array([0.2, 0.3])))


def test_held_out_metrics_hand_worked():
    metrics = held_out_metrics(np.array([1.0, 0.0, 1.0]), np.array([0.8, 0.4, 0.3]))
    assert metrics["log_likelihood"] == pytest.approx(np.mean(np.log([0.8, 0.6, 0.3])))
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["auc"] == 0.5
    assert metrics["n_cells"] == 3


def test_cross_validation_prefers_the_model_that_generated_the_data():
    rng = np.random.default_rng(55)
    slopes = np.exp(rng.normal(0.0, 0.6, 25))
    data, _ = simulate_responses(400, 25, kind="2pl", seed=55, discrimination=slopes)
    result = cross_validate(data, folds=4, seed=1)
    assert set(result) == {"1pl", "2pl", "item_rate"}
    assert result["1pl"]["n_cells"] == 400 * 25
    assert result["2pl"]["log_likelihood"] > result["1pl"]["log_likelihood"]
    assert result["1pl"]["log_likelihood"] > result["item_rate"]["log_likelihood"]
    assert result["2pl"]["auc"] > result["1pl"]["auc"] > result["item_rate"]["auc"]
    again = cross_validate(data, folds=4, seed=1)
    assert again["2pl"]["log_likelihood"] == result["2pl"]["log_likelihood"]


def test_cross_validation_handles_missing_cells_and_validates():
    data, _ = simulate_responses(80, 8, kind="1pl", seed=56, missing=0.3)
    result = cross_validate(data, kinds=("1pl",), folds=3, seed=2)
    assert result["1pl"]["n_cells"] == 80 * 8 - data.n_missing
    assert np.isfinite(result["1pl"]["log_likelihood"])
    with pytest.raises(ValueError, match="at least 2 folds"):
        cross_validate(data, folds=1)
