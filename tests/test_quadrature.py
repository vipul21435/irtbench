import json

import numpy as np
import pytest
from scipy import integrate, optimize

from irtbench.data import ResponseMatrix
from irtbench.mml import (
    IRTFit,
    estimate_abilities,
    fit_mml,
    marginal_log_likelihood,
    posterior_width,
    predict_probabilities,
)
from irtbench.quadrature import gauss_hermite_grid, uniform_grid
from irtbench.simulate import simulate_responses


def log_posterior(theta, a, d, cells):
    z = a * theta + d
    return float(np.sum(cells * z - np.logaddexp(0.0, z)) - 0.5 * theta**2)


def long_pattern(n_items, seed):
    rng = np.random.default_rng(seed)
    a = np.exp(rng.normal(0.2, 0.3, n_items))
    d = rng.normal(0.0, 1.5, n_items)
    cells = (rng.random(n_items) < 1.0 / (1.0 + np.exp(-(a * 0.7 + d)))).astype(float)
    items = tuple(f"q{index}" for index in range(n_items))
    return a, d, cells, IRTFit.from_parameters(items, a, d), ResponseMatrix(("m",), items, [cells])


def standardised_long_benchmark():
    rng = np.random.default_rng(5)
    ability = rng.standard_normal(100)
    ability = (ability - ability.mean()) / ability.std()
    data, _ = simulate_responses(
        100, 500, kind="1pl", seed=5, ability=ability, discrimination=np.full(500, 1.5)
    )
    return data


def test_uniform_grid_integrates_the_normal_density():
    grid = uniform_grid(0.5, -8.0, 8.0)
    weights = np.exp(grid.log_weights)
    assert grid.rule == "uniform"
    assert grid.size == 33
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert (weights * grid.nodes**2).sum() == pytest.approx(1.0, abs=1e-12)
    assert grid.gap_at(np.array([-20.0, 0.1, 20.0])).tolist() == [0.5, 0.5, 0.5]
    assert grid.covers(-8.0, 7.5)
    assert not grid.covers(-8.5, 0.0)
    with pytest.raises(ValueError, match="positive spacing"):
        uniform_grid(0.0, -1.0, 1.0)


def test_gauss_hermite_gaps_widen_away_from_zero():
    grid = gauss_hermite_grid(61)
    gaps = grid.gap_at(np.array([0.0, 8.0]))
    assert gaps[0] == pytest.approx(0.4006, abs=1e-3)
    assert gaps[1] > gaps[0]


def test_posterior_width_matches_a_brute_force_search():
    a, d, cells, fit, data = long_pattern(40, seed=1)
    mode, width = posterior_width(a, d, cells[None, :], 1.0 - cells[None, :])
    best = optimize.minimize_scalar(
        lambda theta: -log_posterior(theta, a, d, cells), bounds=(-6, 6), method="bounded",
        options={"xatol": 1e-10},
    )
    assert mode[0] == pytest.approx(best.x, abs=1e-6)
    step = 1e-4
    second = (
        log_posterior(best.x + step, a, d, cells)
        - 2 * log_posterior(best.x, a, d, cells)
        + log_posterior(best.x - step, a, d, cells)
    ) / step**2
    assert width[0] == pytest.approx(1.0 / np.sqrt(-second), rel=1e-4)


@pytest.mark.parametrize("n_items", [3, 40, 300, 2000])
def test_likelihood_is_accurate_from_short_to_very_long_tests(n_items):
    a, d, cells, fit, data = long_pattern(n_items, seed=n_items)
    best = optimize.minimize_scalar(
        lambda theta: -log_posterior(theta, a, d, cells), bounds=(-8, 8), method="bounded",
        options={"xatol": 1e-10},
    )
    peak = -best.fun
    mass = integrate.quad(
        lambda theta: np.exp(log_posterior(theta, a, d, cells) - peak),
        -10,
        10,
        points=[best.x - 0.5, best.x, best.x + 0.5],
        limit=400,
        epsabs=0,
        epsrel=1e-12,
    )[0]
    expected = peak + np.log(mass) - 0.5 * np.log(2 * np.pi)
    assert marginal_log_likelihood(fit, data) == pytest.approx(expected, abs=1e-7)


def test_long_benchmark_recovers_the_slope_only_on_a_fine_enough_grid():
    # 100 models x 500 items generated with slope 1.5 and abilities of mean 0 and SD 1
    data = standardised_long_benchmark()
    fit = fit_mml(data, "1pl")
    assert fit.converged
    assert fit.quadrature == "uniform"
    assert fit.resolution <= 1.0
    assert fit.discrimination[0] == pytest.approx(1.5, abs=0.05)
    assert np.median(estimate_abilities(fit, data).sd) < 0.1

    finer = fit_mml(data, "1pl", resolution=0.5)
    assert finer.n_nodes > 1.8 * fit.n_nodes
    assert finer.discrimination[0] == pytest.approx(fit.discrimination[0], abs=1e-4)
    assert finer.log_likelihood == pytest.approx(fit.log_likelihood, abs=1e-3)

    # the classical fixed rule puts every model on one or two nodes and gets the scale wrong
    fixed = fit_mml(data, "1pl", resolution=None)
    assert fixed.quadrature == "gauss-hermite"
    assert fixed.n_nodes == 61
    assert np.isnan(fixed.resolution)
    assert fixed.discrimination[0] < 1.3


def test_short_test_stays_on_the_gauss_hermite_rule():
    data, _ = simulate_responses(400, 8, kind="1pl", seed=2)
    fit = fit_mml(data, "1pl")
    assert fit.quadrature == "gauss-hermite"
    assert fit.n_nodes == 61
    assert fit.resolution <= 1.0
    fine = fit_mml(data, "1pl", resolution=0.3)
    assert fine.quadrature == "uniform"
    np.testing.assert_allclose(fine.discrimination, fit.discrimination, atol=1e-5)
    np.testing.assert_allclose(fine.intercept, fit.intercept, atol=1e-5)


def test_node_limit_is_reported_as_an_unresolved_grid():
    data, _ = simulate_responses(60, 150, kind="1pl", seed=3, discrimination=np.full(150, 1.5))
    fit = fit_mml(data, "1pl", max_nodes=25)
    assert fit.n_nodes <= 25
    assert fit.resolution > 1.0
    with pytest.raises(ValueError, match="resolution must be positive"):
        fit_mml(data, "1pl", resolution=0.0)


def test_predictive_probability_matches_adaptive_integration():
    a = np.array([1.4, 0.7, 1.1, 0.9])
    d = np.array([0.2, -0.6, 1.0, -0.3])
    cells = np.array([1.0, 0.0, 1.0])
    fit = IRTFit.from_parameters(("q1", "q2", "q3", "q4"), a, d)
    answered = ResponseMatrix(("m",), ("q1", "q2", "q3"), [cells])

    def weight(theta):
        return np.exp(log_posterior(theta, a[:3], d[:3], cells))

    mass = integrate.quad(weight, -12, 12, epsabs=1e-14, epsrel=1e-13)[0]
    expected = [
        integrate.quad(
            lambda theta, k=k: weight(theta) / (1.0 + np.exp(-(a[k] * theta + d[k]))),
            -12,
            12,
            epsabs=1e-14,
            epsrel=1e-13,
        )[0]
        / mass
        for k in range(4)
    ]
    np.testing.assert_allclose(predict_probabilities(fit, answered), [expected], atol=1e-7)


def test_predictive_probability_of_items_left_out_is_their_smoothed_rate():
    data, _ = simulate_responses(50, 6, kind="1pl", seed=4)
    extra = np.column_stack([np.ones(50), np.zeros(50), np.full(50, np.nan)])
    wider = ResponseMatrix(
        data.models, data.items + ("easy", "hard", "unseen"), np.hstack([data.scores, extra])
    )
    fit = fit_mml(wider, "1pl")
    probability = predict_probabilities(fit, wider)
    assert probability.shape == (50, 9)
    assert ((probability[:, :6] > 0) & (probability[:, :6] < 1)).all()
    np.testing.assert_allclose(probability[:, 6], 50.5 / 51)
    np.testing.assert_allclose(probability[:, 7], 0.5 / 51)
    assert np.isnan(probability[:, 8]).all()
    np.testing.assert_allclose(fit.smoothed_rate[6:8], [50.5 / 51, 0.5 / 51])


def test_fit_survives_a_json_round_trip():
    data, _ = simulate_responses(80, 7, kind="2pl", seed=6)
    wider = ResponseMatrix(
        data.models, data.items + ("easy",), np.hstack([data.scores, np.ones((80, 1))])
    )
    fit = fit_mml(wider, "2pl")
    again = IRTFit.from_dict(json.loads(json.dumps(fit.to_dict())))
    assert again.items == fit.items
    assert again.status == fit.status
    assert again.kind == "2pl"
    assert again.n_parameters == fit.n_parameters
    np.testing.assert_array_equal(again.discrimination, fit.discrimination)
    np.testing.assert_array_equal(again.intercept, fit.intercept)
    np.testing.assert_array_equal(again.item_solved, fit.item_solved)
    np.testing.assert_array_equal(
        estimate_abilities(again, wider).theta, estimate_abilities(fit, wider).theta
    )
    with pytest.raises(ValueError, match="not a fit written by irtbench"):
        IRTFit.from_dict({"kind": "2pl"})


def test_scoring_needs_at_least_one_model():
    fit = IRTFit.from_parameters(("q1",), [1.0], [0.0])
    empty = ResponseMatrix((), ("q1",), np.zeros((0, 1)))
    with pytest.raises(ValueError, match="no models to score"):
        estimate_abilities(fit, empty)
