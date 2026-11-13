import numpy as np
import pytest

from irtbench.mml import _common_slope, _log_prior, estimate_abilities, fit_mml
from irtbench.simulate import simulate_responses


def few_models():
    return simulate_responses(40, 24, kind="2pl", seed=2026, missing=0.03)[0]


def test_without_a_prior_the_2pl_runs_away_on_few_models():
    fit = fit_mml(few_models(), "2pl", slope_prior_sd=None, max_iter=150, max_nodes=301)
    assert not fit.converged
    assert fit.discrimination.max() > 10
    assert fit.slope_prior_sd is None
    assert fit.slope_prior_mean is None


def test_the_prior_makes_the_2pl_converge_on_few_models():
    data = few_models()
    fit = fit_mml(data, "2pl")
    assert fit.converged
    assert fit.slope_prior_sd == 0.5
    assert np.abs(fit.discrimination).max() < 5
    assert np.diff(fit.trace).min() > -1e-8
    common = fit_mml(data, "1pl")
    assert fit.log_likelihood > common.log_likelihood
    slopes = fit.discrimination
    assert fit.slope_prior_mean == pytest.approx(float(slopes @ slopes) / float(slopes.sum()))


def test_a_tight_prior_gives_the_1pl_and_a_wide_one_the_maximum_likelihood_fit():
    data, _ = simulate_responses(3000, 12, kind="2pl", seed=31)
    common = fit_mml(data, "1pl", tol=1e-9)
    tight = fit_mml(data, "2pl", slope_prior_sd=1e-4, tol=1e-9)
    np.testing.assert_allclose(tight.discrimination, common.discrimination, atol=1e-3)
    assert tight.log_likelihood == pytest.approx(common.log_likelihood, abs=1e-2)

    plain = fit_mml(data, "2pl", slope_prior_sd=None, tol=1e-9)
    wide = fit_mml(data, "2pl", slope_prior_sd=1e3, tol=1e-9)
    np.testing.assert_allclose(wide.discrimination, plain.discrimination, atol=1e-4)
    default = fit_mml(data, "2pl", tol=1e-9)
    assert plain.log_likelihood >= default.log_likelihood >= common.log_likelihood
    # with 3000 models the default prior barely moves the estimates
    np.testing.assert_allclose(default.discrimination, plain.discrimination, rtol=0.03)


def test_the_prior_shrinks_slopes_towards_the_common_slope():
    data, _ = simulate_responses(60, 40, kind="2pl", seed=32)
    spread = [
        float(np.std(fit_mml(data, "2pl", slope_prior_sd=sd).discrimination))
        for sd in (1.0, 0.5, 0.25)
    ]
    assert spread[0] > spread[1] > spread[2] > 0


def test_abilities_keep_unit_spread_under_the_prior():
    # a penalty that could be bought off by shrinking every slope would inflate the abilities
    rng = np.random.default_rng(100)
    slopes = np.exp(rng.normal(0.2, 0.4, 200))
    data, truth = simulate_responses(60, 200, kind="2pl", seed=0, discrimination=slopes)
    fit = fit_mml(data, "2pl")
    theta = estimate_abilities(fit, data).theta
    assert 0.9 < theta.std() < 1.1
    assert np.corrcoef(theta, truth["ability"])[0, 1] > 0.98
    keep = fit.estimated
    assert np.corrcoef(fit.discrimination[keep], slopes[keep])[0, 1] > 0.6


def test_prior_arguments_are_validated():
    data = few_models()
    with pytest.raises(ValueError, match="slope_prior_sd must be positive"):
        fit_mml(data, "2pl", slope_prior_sd=0.0)
    assert fit_mml(data, "1pl", slope_prior_sd=0.5).slope_prior_sd is None


def test_prior_density_only_sees_ratios_of_slopes():
    slopes = np.array([0.5, 1.0, 1.5, 2.5])
    centre = _common_slope(slopes)
    assert centre == pytest.approx(float(slopes @ slopes) / slopes.sum())
    direct = -float(((slopes / centre - 1.0) ** 2).sum()) / (2 * 0.5**2)
    assert _log_prior(slopes, 0.5) == pytest.approx(direct)
    assert _log_prior(3.0 * slopes, 0.5) == pytest.approx(direct)
    assert _log_prior(np.full(4, 1.7), 0.5) == pytest.approx(0.0, abs=1e-12)
    assert _log_prior(slopes, None) == 0.0
    # no other common slope gives a higher density
    for other in (0.8 * centre, 1.2 * centre):
        assert -float(((slopes / other - 1.0) ** 2).sum()) / (2 * 0.5**2) < direct
