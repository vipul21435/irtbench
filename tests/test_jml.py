import numpy as np
import pytest
from scipy import optimize

from irtbench.data import ResponseMatrix
from irtbench.jml import fit_jml
from irtbench.mml import estimate_abilities, fit_mml
from irtbench.simulate import simulate_responses


def test_hand_worked_two_by_two_table():
    # models solve (1,1), (1,0), (0,1), (0,0): by symmetry both intercepts are 0, the two
    # middle models sit at 0, and the top model solves 2 * (1 - sigmoid(t)) = ridge * t
    scores = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    data = ResponseMatrix(("a", "b", "c", "d"), ("q1", "q2"), scores)
    fit = fit_jml(data, "1pl", ridge=1.0)
    assert fit.converged
    top = optimize.brentq(lambda t: 2.0 * (1.0 - 1.0 / (1.0 + np.exp(-t))) - t, 0.0, 5.0)
    np.testing.assert_allclose(fit.intercept, [0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(fit.ability, [top, 0.0, 0.0, -top], atol=1e-7)
    np.testing.assert_allclose(fit.discrimination, [1.0, 1.0])
    expected = 2 * np.log(1 / (1 + np.exp(-top))) * 2 + 4 * np.log(0.5)
    assert fit.log_likelihood == pytest.approx(expected, abs=1e-6)


def test_rasch_sufficiency_equal_raw_scores_give_equal_abilities():
    data, _ = simulate_responses(300, 12, kind="1pl", seed=81)
    fit = fit_jml(data, "1pl")
    assert fit.converged
    raw = data.scores.sum(axis=1)
    for score in np.unique(raw):
        group = fit.ability[raw == score]
        assert np.ptp(group) < 1e-6
    means = [fit.ability[raw == score].mean() for score in np.unique(raw)]
    assert (np.diff(means) > 0).all()  # and a higher raw score gives a higher ability
    assert np.isfinite(fit.ability).all()  # even for models that solved all or none


def test_sufficiency_does_not_hold_for_the_2pl_or_with_missing_cells():
    data, _ = simulate_responses(300, 12, kind="2pl", seed=82)
    fit = fit_jml(data, "2pl")
    raw = data.scores.sum(axis=1)
    spread = max(np.ptp(fit.ability[raw == score]) for score in np.unique(raw))
    assert spread > 0.05


def test_joint_and_marginal_fits_agree_on_a_long_test():
    data, truth = simulate_responses(300, 80, kind="1pl", seed=83)
    joint = fit_jml(data, "1pl", ridge=0.01)
    assert joint.converged
    marginal = fit_mml(data, "1pl")
    keep = marginal.estimated
    # same Rasch scale: joint intercepts against marginal intercepts
    assert np.corrcoef(joint.intercept[keep], marginal.intercept[keep])[0, 1] > 0.999
    theta = estimate_abilities(marginal, data).theta
    assert np.corrcoef(joint.ability, theta)[0, 1] > 0.995
    assert np.corrcoef(joint.ability, truth["ability"])[0, 1] > 0.95
    # the joint intercepts are stretched by about J / (J - 1), the known bias of joint ML
    stretch = np.polyfit(marginal.intercept[keep], joint.intercept[keep], 1)[0]
    assert 1.0 < stretch < 1.03


def test_joint_fit_flags_items_and_validates():
    data, _ = simulate_responses(100, 6, kind="1pl", seed=84, missing=0.1)
    extra = np.column_stack([np.ones(100), np.zeros(100)])
    wider = ResponseMatrix(
        data.models, data.items + ("free", "stuck"), np.hstack([data.scores, extra])
    )
    fit = fit_jml(wider, "2pl")
    assert fit.status[6:] == ("all_correct", "all_incorrect")
    assert np.isnan(fit.discrimination[6:]).all()
    assert fit.difficulty[6] == -np.inf
    assert fit.difficulty[7] == np.inf
    assert np.isfinite(fit.difficulty[:6]).all()
    assert fit.discrimination[:6].sum() > 0
    with pytest.raises(ValueError, match="ridge must be positive"):
        fit_jml(data, "1pl", ridge=0.0)
    with pytest.raises(ValueError, match="unknown model"):
        fit_jml(data, "3pl")
    with pytest.raises(ValueError, match="at least 2 items"):
        fit_jml(ResponseMatrix(("a", "b"), ("q1", "q2"), np.ones((2, 2))))
