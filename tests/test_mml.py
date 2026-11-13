import json

import numpy as np
import pytest
from scipy import integrate, stats

from irtbench.data import ResponseMatrix
from irtbench.mml import (
    ALL_CORRECT,
    ALL_INCORRECT,
    ESTIMATED,
    NO_DATA,
    IRTFit,
    estimate_abilities,
    fit_mml,
    marginal_log_likelihood,
    sigmoid,
)
from irtbench.quadrature import gauss_hermite
from irtbench.simulate import simulate_responses


def rmse(estimate, truth):
    return float(np.sqrt(np.mean((np.asarray(estimate) - np.asarray(truth)) ** 2)))


def one_model(cells):
    items = tuple(f"q{index + 1}" for index in range(len(cells)))
    return ResponseMatrix(("m",), items, np.array([cells], dtype=float))


def pattern_integral(a, d, cells, power=0):
    """Integral of theta**power * likelihood(theta) * normal density, by adaptive quadrature."""

    def integrand(theta):
        value = theta**power * stats.norm.pdf(theta)
        for slope, intercept, cell in zip(a, d, cells, strict=True):
            p = 1.0 / (1.0 + np.exp(-(slope * theta + intercept)))
            value *= p if cell == 1 else 1.0 - p
        return value

    return integrate.quad(integrand, -12, 12, epsabs=1e-13, epsrel=1e-13)[0]


# --- quadrature ---------------------------------------------------------------------------


def test_gauss_hermite_reproduces_normal_moments():
    nodes, log_weights = gauss_hermite(21)
    weights = np.exp(log_weights)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert (weights * nodes).sum() == pytest.approx(0.0, abs=1e-12)
    assert (weights * nodes**2).sum() == pytest.approx(1.0, abs=1e-12)
    assert (weights * nodes**4).sum() == pytest.approx(3.0, abs=1e-10)
    assert (weights * nodes**6).sum() == pytest.approx(15.0, abs=1e-9)


def test_gauss_hermite_rejects_one_node():
    with pytest.raises(ValueError, match="at least 2 nodes"):
        gauss_hermite(1)


def test_gauss_hermite_rejects_a_rule_whose_weights_underflow():
    with pytest.raises(ValueError, match="use fewer nodes"):
        gauss_hermite(600)


# --- likelihood and abilities at known parameters -----------------------------------------


def test_sigmoid_is_stable_in_the_tails():
    values = sigmoid(np.array([-800.0, 0.0, 800.0]))
    assert values.tolist() == [0.0, 0.5, 1.0]


def test_likelihood_with_flat_items_has_a_closed_form():
    # with zero discrimination the ability drops out: every cell is Bernoulli(sigmoid(d_j))
    intercepts = np.array([0.4, -1.1, 2.0])
    fit = IRTFit.from_parameters(("q1", "q2", "q3"), np.zeros(3), intercepts)
    scores = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, np.nan], [1.0, 1.0, 1.0]])
    data = ResponseMatrix(("m1", "m2", "m3"), ("q1", "q2", "q3"), scores)
    p = 1.0 / (1.0 + np.exp(-intercepts))
    expected = np.nansum(scores * np.log(p) + (1.0 - scores) * np.log(1.0 - p))
    assert marginal_log_likelihood(fit, data) == pytest.approx(expected, abs=1e-10)


def test_likelihood_of_a_symmetric_item_is_one_half():
    fit = IRTFit.from_parameters(("q1",), [1.7], [0.0])
    assert marginal_log_likelihood(fit, one_model([1])) == pytest.approx(np.log(0.5), abs=1e-7)


def test_likelihood_matches_adaptive_integration():
    a = [1.7, 0.6, 1.2, -0.4]
    d = [0.3, -0.8, 1.5, 0.2]
    cells = [1, 0, 1, 1]
    fit = IRTFit.from_parameters(("q1", "q2", "q3", "q4"), a, d)
    expected = np.log(pattern_integral(a, d, cells))
    assert marginal_log_likelihood(fit, one_model(cells)) == pytest.approx(expected, abs=1e-7)
    fine = marginal_log_likelihood(fit, one_model(cells), resolution=0.4)
    assert fine == pytest.approx(expected, abs=1e-12)


def test_eap_and_posterior_sd_match_adaptive_integration():
    a = [1.7, 0.6, 1.2, 0.9, 1.1]
    d = [0.3, -0.8, 1.5, 0.2, -0.5]
    cells = [1, 0, 1, 1, 0]
    fit = IRTFit.from_parameters(tuple(f"q{k + 1}" for k in range(5)), a, d)
    mass = pattern_integral(a, d, cells)
    mean = pattern_integral(a, d, cells, power=1) / mass
    second = pattern_integral(a, d, cells, power=2) / mass
    abilities = estimate_abilities(fit, one_model(cells))
    assert abilities.theta[0] == pytest.approx(mean, abs=1e-7)
    assert abilities.sd[0] == pytest.approx(np.sqrt(second - mean**2), abs=1e-7)
    assert abilities.n_answered.tolist() == [5]


def test_model_without_results_gets_the_prior():
    fit = IRTFit.from_parameters(("q1", "q2"), [1.0, 1.5], [0.0, 0.5])
    abilities = estimate_abilities(fit, one_model([np.nan, np.nan]))
    assert abilities.theta[0] == pytest.approx(0.0, abs=1e-10)
    assert abilities.sd[0] == pytest.approx(1.0, abs=1e-10)
    assert abilities.n_answered.tolist() == [0]


def test_abilities_accept_a_subset_of_items_in_any_order():
    fit = IRTFit.from_parameters(("q1", "q2", "q3"), [1.0, 1.5, 0.7], [0.0, 0.5, -0.3])
    full = ResponseMatrix(("m",), ("q1", "q2", "q3"), np.array([[1.0, np.nan, 0.0]]))
    subset = ResponseMatrix(("m",), ("q3", "q1"), np.array([[0.0, 1.0]]))
    assert estimate_abilities(fit, subset).theta[0] == pytest.approx(
        estimate_abilities(fit, full).theta[0], abs=1e-12
    )
    stranger = ResponseMatrix(("m",), ("q9",), np.array([[1.0]]))
    with pytest.raises(ValueError, match="1 items are not in the fit: q9"):
        estimate_abilities(fit, stranger)


def test_from_parameters_validates():
    with pytest.raises(ValueError, match="one value per item"):
        IRTFit.from_parameters(("q1", "q2"), [1.0], [0.0, 0.1])
    with pytest.raises(ValueError, match="finite"):
        IRTFit.from_parameters(("q1",), [np.nan], [0.0])
    with pytest.raises(ValueError, match="unknown model"):
        IRTFit.from_parameters(("q1",), [1.0], [0.0], kind="3pl")


# --- fitting ------------------------------------------------------------------------------


def test_two_item_rasch_reproduces_the_pattern_table():
    # Two items under the 1PL have three parameters for three free pattern probabilities, so
    # with a positive association the fit reproduces the observed 2x2 table exactly.
    counts = {(1, 1): 30, (1, 0): 15, (0, 1): 10, (0, 0): 25}
    rows = [pattern for pattern, count in counts.items() for _ in range(count)]
    data = ResponseMatrix(
        tuple(f"m{index}" for index in range(len(rows))), ("q1", "q2"), np.array(rows, dtype=float)
    )
    fit = fit_mml(data, "1pl", tol=1e-10, max_iter=20000)
    assert fit.converged
    assert fit.n_parameters == 3
    a, d = fit.discrimination, fit.intercept
    for pattern, count in counts.items():
        assert pattern_integral(a, d, pattern) == pytest.approx(count / 80, abs=1e-6)
    saturated = sum(count * np.log(count / 80) for count in counts.values())
    assert fit.log_likelihood == pytest.approx(saturated, abs=1e-6)


@pytest.mark.parametrize("kind", ["1pl", "2pl"])
def test_em_never_decreases_the_likelihood(kind):
    data, _ = simulate_responses(400, 12, kind=kind, seed=11, missing=0.1)
    fit = fit_mml(data, kind)
    assert fit.converged
    assert 2 <= len(fit.trace) <= fit.iterations + 1
    assert np.diff(fit.trace).min() > -1e-8
    assert fit.log_likelihood == pytest.approx(marginal_log_likelihood(fit, data), abs=1e-5)


def test_2pl_recovers_simulated_parameters():
    data, truth = simulate_responses(3000, 20, kind="2pl", seed=1)
    fit = fit_mml(data, "2pl")
    assert fit.converged
    assert fit.n_parameters == 40
    assert np.corrcoef(fit.discrimination, truth["discrimination"])[0, 1] > 0.95
    assert rmse(fit.discrimination, truth["discrimination"]) < 0.12
    assert np.corrcoef(fit.difficulty, truth["difficulty"])[0, 1] > 0.99
    assert rmse(fit.difficulty, truth["difficulty"]) < 0.12
    abilities = estimate_abilities(fit, data)
    assert np.corrcoef(abilities.theta, truth["ability"])[0, 1] > 0.85
    # the posterior SD is an honest error bar: about 68 % of abilities within one SD
    inside = np.abs(abilities.theta - truth["ability"]) < abilities.sd
    assert 0.63 < inside.mean() < 0.73


def test_1pl_recovers_simulated_parameters():
    data, truth = simulate_responses(2000, 20, kind="1pl", seed=3, discrimination=np.full(20, 1.3))
    fit = fit_mml(data, "1pl")
    assert fit.converged
    assert fit.n_parameters == 21
    assert np.ptp(fit.discrimination) == 0.0
    assert fit.discrimination[0] == pytest.approx(1.3, abs=0.08)
    assert rmse(fit.difficulty, truth["difficulty"]) < 0.1
    # Rasch scale: abilities a * theta, difficulties -d
    assert rmse(-fit.intercept, 1.3 * truth["difficulty"]) < 0.12


def test_recovery_with_missing_cells():
    data, truth = simulate_responses(3000, 20, kind="2pl", seed=4, missing=0.3)
    assert data.n_missing > 15000
    fit = fit_mml(data, "2pl")
    assert fit.n_observations == 3000 * 20 - data.n_missing
    assert np.corrcoef(fit.discrimination, truth["discrimination"])[0, 1] > 0.93
    assert rmse(fit.difficulty, truth["difficulty"]) < 0.15


def test_a_model_without_results_changes_nothing():
    data, _ = simulate_responses(300, 10, kind="2pl", seed=5)
    padded = ResponseMatrix(
        data.models + ("empty",), data.items, np.vstack([data.scores, np.full((1, 10), np.nan)])
    )
    base = fit_mml(data, "2pl", tol=1e-10)
    extra = fit_mml(padded, "2pl", tol=1e-10)
    np.testing.assert_allclose(extra.discrimination, base.discrimination, atol=1e-6)
    np.testing.assert_allclose(extra.intercept, base.intercept, atol=1e-6)
    assert extra.log_likelihood == pytest.approx(base.log_likelihood, abs=1e-6)
    assert extra.n_models == base.n_models == 300


@pytest.mark.parametrize("kind", ["1pl", "2pl"])
def test_fit_does_not_depend_on_model_or_item_order(kind):
    data, _ = simulate_responses(250, 9, kind=kind, seed=6, missing=0.05)
    rng = np.random.default_rng(0)
    rows, cols = rng.permutation(data.n_models), rng.permutation(data.n_items)
    base = fit_mml(data, kind, tol=1e-9)
    shuffled = fit_mml(data.take_models(rows).take_items(cols), kind, tol=1e-9)
    assert shuffled.items == tuple(data.items[j] for j in cols)
    np.testing.assert_allclose(shuffled.discrimination, base.discrimination[cols], atol=1e-7)
    np.testing.assert_allclose(shuffled.intercept, base.intercept[cols], atol=1e-7)
    assert shuffled.log_likelihood == pytest.approx(base.log_likelihood, abs=1e-8)
    theta = estimate_abilities(base, data).theta
    theta_shuffled = estimate_abilities(shuffled, data.take_models(rows)).theta
    np.testing.assert_allclose(theta_shuffled, theta[rows], atol=1e-7)


def test_items_without_variation_are_flagged_and_left_out():
    data, _ = simulate_responses(300, 8, kind="2pl", seed=7)
    n = data.n_models
    extra = np.column_stack([np.ones(n), np.zeros(n), np.full(n, np.nan)])
    extra[0, 0] = np.nan
    wider = ResponseMatrix(
        data.models, data.items + ("easy", "hard", "unseen"), np.hstack([data.scores, extra])
    )
    base = fit_mml(data, "2pl")
    fit = fit_mml(wider, "2pl")
    assert fit.status[:8] == (ESTIMATED,) * 8
    assert fit.status[8:] == (ALL_CORRECT, ALL_INCORRECT, NO_DATA)
    assert fit.estimated.tolist() == [True] * 8 + [False] * 3
    assert fit.n_parameters == base.n_parameters == 16
    assert fit.n_observations == base.n_observations
    np.testing.assert_allclose(fit.discrimination[:8], base.discrimination, atol=1e-12)
    np.testing.assert_allclose(fit.intercept[:8], base.intercept, atol=1e-12)
    assert np.isnan(fit.discrimination[8:]).all()
    assert fit.difficulty[8] == -np.inf
    assert fit.difficulty[9] == np.inf
    assert np.isnan(fit.difficulty[10])
    probability = fit.probability([-1.0, 2.0])
    assert probability.shape == (2, 11)
    assert probability[:, 8].tolist() == [1.0, 1.0]
    assert probability[:, 9].tolist() == [0.0, 0.0]
    assert np.isnan(probability[:, 10]).all()
    assert (np.diff(probability[:, :8], axis=0) > 0).all()


def test_probability_follows_the_model_equation():
    fit = IRTFit.from_parameters(("q1", "q2"), [1.5, 0.5], [-0.75, 1.0])
    assert fit.difficulty.tolist() == [0.5, -2.0]
    np.testing.assert_allclose(fit.probability(0.5), [[0.5, 1 / (1 + np.exp(-1.25))]])


def test_higher_ability_means_a_higher_score():
    data, _ = simulate_responses(200, 15, kind="2pl", seed=8)
    fit = fit_mml(data, "2pl")
    assert fit.discrimination.sum() > 0
    theta = estimate_abilities(fit, data).theta
    assert np.corrcoef(theta, data.model_means())[0, 1] > 0.9


def test_warm_start_stays_at_the_solution():
    data, _ = simulate_responses(300, 10, kind="2pl", seed=12)
    base = fit_mml(data, "2pl", tol=1e-9)
    again = fit_mml(data, "2pl", tol=1e-9, start=(base.discrimination, base.intercept))
    assert again.iterations <= 5
    np.testing.assert_allclose(again.discrimination, base.discrimination, atol=1e-7)
    assert again.log_likelihood == pytest.approx(base.log_likelihood, abs=1e-9)


def test_mirrored_start_is_flipped_back_to_positive_slopes():
    # (a, theta) and (-a, -theta) have the same likelihood; the fit reports the positive one
    data, _ = simulate_responses(300, 10, kind="2pl", seed=12)
    base = fit_mml(data, "2pl", tol=1e-9)
    mirrored = fit_mml(data, "2pl", tol=1e-9, start=(-base.discrimination, base.intercept))
    assert mirrored.discrimination.sum() > 0
    np.testing.assert_allclose(mirrored.discrimination, base.discrimination, atol=1e-6)
    np.testing.assert_allclose(mirrored.intercept, base.intercept, atol=1e-6)


def test_start_for_the_1pl_uses_the_mean_slope():
    data, _ = simulate_responses(300, 10, kind="1pl", seed=13)
    base = fit_mml(data, "1pl", tol=1e-9)
    uneven = np.linspace(0.5, 1.5, 10)
    fit = fit_mml(data, "1pl", tol=1e-9, start=(uneven, base.intercept))
    assert np.ptp(fit.discrimination) == 0.0
    np.testing.assert_allclose(fit.discrimination, base.discrimination, atol=1e-6)


def test_start_is_validated():
    data, _ = simulate_responses(100, 6, kind="2pl", seed=14)
    with pytest.raises(ValueError, match="one discrimination and one intercept per item"):
        fit_mml(data, "2pl", start=(np.ones(5), np.zeros(6)))
    with pytest.raises(ValueError, match="finite"):
        fit_mml(data, "2pl", start=(np.full(6, np.nan), np.zeros(6)))


def test_simulation_is_seeded_and_validated():
    first, truth = simulate_responses(30, 5, seed=3, missing=0.2)
    second, _ = simulate_responses(30, 5, seed=3, missing=0.2)
    np.testing.assert_array_equal(first.scores, second.scores)
    assert first.models[0] == "model-01"
    assert first.items[-1] == "item-5"
    assert truth["ability"].shape == (30,)
    assert 0 < first.n_missing < 150
    given, kept = simulate_responses(4, 2, difficulty=[0.5, -0.5], discrimination=[1.0, 2.0])
    assert kept["difficulty"].tolist() == [0.5, -0.5]
    assert kept["discrimination"].tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="unknown model"):
        simulate_responses(4, 2, kind="3pl")
    with pytest.raises(ValueError, match=r"missing must be in \[0, 1\)"):
        simulate_responses(4, 2, missing=1.0)
    with pytest.raises(ValueError, match="difficulty needs 2 values"):
        simulate_responses(4, 2, difficulty=[0.0])


def test_fit_rejects_unusable_input():
    data, _ = simulate_responses(50, 6, seed=9)
    with pytest.raises(ValueError, match="unknown model"):
        fit_mml(data, "3pl")
    constant = ResponseMatrix(("a", "b"), ("q1", "q2", "q3"), np.ones((2, 3)))
    with pytest.raises(ValueError, match="needs at least 3 items"):
        fit_mml(constant, "2pl")
    lonely = ResponseMatrix(
        ("a", "b", "c"),
        ("q1", "q2"),
        np.array([[1.0, 0.0], [np.nan, np.nan], [np.nan, np.nan]]),
    )
    with pytest.raises(ValueError, match="needs at least 2 items"):
        fit_mml(lonely, "1pl")


def test_fit_summary_is_json_ready():
    data, _ = simulate_responses(120, 6, kind="2pl", seed=10)
    wider = ResponseMatrix(
        data.models, data.items + ("easy",), np.hstack([data.scores, np.ones((120, 1))])
    )
    fit = fit_mml(wider, "2pl")
    summary = json.loads(json.dumps(fit.to_dict(), allow_nan=False))
    assert summary["kind"] == "2pl"
    assert len(summary["items"]) == 7
    assert summary["items"][6] == {
        "item": "easy",
        "status": "all_correct",
        "discrimination": None,
        "difficulty": None,
        "intercept": None,
        "tried": 120,
        "solved": 120,
    }
    rows = json.loads(json.dumps(estimate_abilities(fit, wider).to_dict(), allow_nan=False))
    assert rows[0].keys() == {"model", "theta", "sd", "n_answered"}
    assert rows[0]["n_answered"] == 6
