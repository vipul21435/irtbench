"""The EM fit against an independent maximisation of the same marginal likelihood.

The reference shares no code with the package: it evaluates the marginal likelihood with
scipy's Gauss-Hermite nodes and hands it to a general-purpose optimiser.
"""

import numpy as np
import pytest
from scipy import optimize, special

from irtbench.mml import fit_mml
from irtbench.simulate import simulate_responses


def reference_fit(scores, kind):
    nodes, weights = special.roots_hermitenorm(81)
    log_weights = np.log(weights / weights.sum())
    n_items = scores.shape[1]

    def unpack(vector):
        if kind == "1pl":
            return np.full(n_items, vector[0]), vector[1:]
        return vector[:n_items], vector[n_items:]

    def negative_loglik(vector):
        a, d = unpack(vector)
        z = nodes[:, None] * a[None, :] + d[None, :]
        log_p = -np.logaddexp(0.0, -z)
        log_q = -np.logaddexp(0.0, z)
        seen = ~np.isnan(scores)
        x = np.where(seen, scores, 0.0)
        joint = x @ log_p.T + (seen - x) @ log_q.T + log_weights[None, :]
        return -special.logsumexp(joint, axis=1).sum()

    start = np.concatenate([[1.0], np.zeros(n_items)] if kind == "1pl" else
                           [np.ones(n_items), np.zeros(n_items)])
    result = optimize.minimize(negative_loglik, start, method="BFGS", options={"gtol": 1e-7})
    a, d = unpack(result.x)
    return a, d, -result.fun


@pytest.mark.parametrize("kind", ["1pl", "2pl"])
@pytest.mark.parametrize("missing", [0.0, 0.2])
def test_em_reaches_the_same_maximum_as_a_general_optimiser(kind, missing):
    data, _ = simulate_responses(800, 5, kind=kind, seed=101, missing=missing)
    fit = fit_mml(data, kind, slope_prior_sd=None, tol=1e-9)
    a, d, loglik = reference_fit(data.scores, kind)
    np.testing.assert_allclose(fit.discrimination, a, atol=2e-4)
    np.testing.assert_allclose(fit.intercept, d, atol=2e-4)
    assert fit.log_likelihood == pytest.approx(loglik, abs=1e-5)
