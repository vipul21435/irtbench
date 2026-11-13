import numpy as np
import pytest

from irtbench.data import ResponseMatrix
from irtbench.mml import IRTFit, fit_mml, marginal_log_likelihood
from irtbench.se import bootstrap_standard_errors, observed_information, standard_errors
from irtbench.simulate import simulate_responses


def numerical_information(data, kind, a, d, step=1e-3):
    """Minus the Hessian of the marginal log-likelihood by central differences."""
    items = data.items

    def loglik(vector):
        if kind == "1pl":
            slopes, intercepts = np.full(len(items), vector[0]), vector[1:]
        else:
            slopes, intercepts = vector[: len(items)], vector[len(items) :]
        fit = IRTFit.from_parameters(items, slopes, intercepts, kind=kind)
        return marginal_log_likelihood(fit, data, resolution=0.3)

    centre = np.concatenate([[a[0]], d]) if kind == "1pl" else np.concatenate([a, d])
    size = centre.size
    hessian = np.zeros((size, size))
    for i in range(size):
        for j in range(i, size):
            total = 0.0
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                point = centre.copy()
                point[i] += si * step
                point[j] += sj * step
                total += si * sj * loglik(point)
            hessian[i, j] = hessian[j, i] = total / (4 * step * step)
    return -hessian


@pytest.mark.parametrize("kind", ["1pl", "2pl"])
def test_observed_information_matches_a_numerical_hessian(kind):
    data, _ = simulate_responses(250, 4, kind=kind, seed=91, missing=0.1)
    fit = fit_mml(data, kind, slope_prior_sd=None, tol=1e-9)
    info = observed_information(fit, data, resolution=0.3)
    expected = numerical_information(data, kind, fit.discrimination, fit.intercept)
    assert info.shape == expected.shape == ((5, 5) if kind == "1pl" else (8, 8))
    np.testing.assert_allclose(info, expected, rtol=2e-3, atol=2e-3)
    assert (np.linalg.eigvalsh(info) > 0).all()


def test_standard_errors_shrink_with_more_models_and_cover_the_truth():
    slopes = np.array([0.8, 1.0, 1.2, 1.5, 0.9, 1.1])
    difficulty = np.linspace(-1.0, 1.0, 6)
    small, _ = simulate_responses(500, 6, seed=92, discrimination=slopes, difficulty=difficulty)
    large, _ = simulate_responses(2000, 6, seed=92, discrimination=slopes, difficulty=difficulty)
    fit_small = fit_mml(small, "2pl", slope_prior_sd=None)
    fit_large = fit_mml(large, "2pl", slope_prior_sd=None)
    se_small = standard_errors(fit_small, small)
    se_large = standard_errors(fit_large, large)
    ratio = se_small.discrimination / se_large.discrimination
    assert ((ratio > 1.5) & (ratio < 2.7)).all()
    assert se_large.method == "information"
    z = (fit_large.discrimination - slopes) / se_large.discrimination
    assert np.abs(z).max() < 3.5
    zb = (fit_large.difficulty - difficulty) / se_large.difficulty
    assert np.abs(zb).max() < 3.5


def test_standard_errors_skip_items_left_out_and_use_the_prior():
    data, _ = simulate_responses(80, 6, kind="2pl", seed=93)
    wider = ResponseMatrix(
        data.models, data.items + ("free",), np.hstack([data.scores, np.ones((80, 1))])
    )
    with_prior = standard_errors(fit_mml(wider, "2pl"), wider)
    without = standard_errors(fit_mml(wider, "2pl", slope_prior_sd=None, max_iter=200), wider)
    assert np.isnan(with_prior.discrimination[6])
    assert np.isnan(with_prior.difficulty[6])
    assert np.isfinite(with_prior.discrimination[:6]).all()
    # the prior adds curvature, so on the whole its errors are smaller
    assert with_prior.discrimination[:6].mean() < without.discrimination[:6].mean()
    one = standard_errors(fit_mml(wider, "1pl"), wider)
    assert np.ptp(one.discrimination[:6]) == 0.0
    assert one.discrimination[0] > 0


def test_bootstrap_agrees_with_the_information_errors():
    data, _ = simulate_responses(400, 5, kind="1pl", seed=94)
    fit = fit_mml(data, "1pl")
    by_information = standard_errors(fit, data)
    by_bootstrap = bootstrap_standard_errors(data, "1pl", n_boot=30, seed=1)
    assert by_bootstrap.method == "bootstrap"
    assert (by_bootstrap.replicates == 30).all()
    ratio = by_bootstrap.intercept / by_information.intercept
    assert ((ratio > 0.5) & (ratio < 2.0)).all()
    assert 0.7 < np.median(ratio) < 1.4
    # the seed decides the resamples; a handful of replicates is enough to see that
    few = bootstrap_standard_errors(data, "1pl", n_boot=5, seed=1)
    again = bootstrap_standard_errors(data, "1pl", n_boot=5, seed=1)
    np.testing.assert_allclose(again.intercept, few.intercept, rtol=1e-12)
    other = bootstrap_standard_errors(data, "1pl", n_boot=5, seed=2)
    assert not np.allclose(other.intercept, few.intercept, rtol=1e-6)
    with pytest.raises(ValueError, match="at least 2 replicates"):
        bootstrap_standard_errors(data, n_boot=1)


def test_bootstrap_counts_replicates_where_an_item_could_not_be_estimated():
    data, _ = simulate_responses(30, 5, kind="1pl", seed=95)
    scores = data.scores.copy()
    scores[:, 4] = 0.0
    scores[0, 4] = 1.0  # solved by exactly one model: absent from about a third of resamples
    rare = ResponseMatrix(data.models, data.items, scores)
    errors = bootstrap_standard_errors(rare, "1pl", n_boot=12, seed=2)
    assert 0 < errors.replicates[4] < 12
    assert (errors.replicates[:4] == 12).all()
    assert np.isfinite(errors.intercept[:4]).all()
