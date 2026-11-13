import numpy as np
import pytest
from scipy import stats

from irtbench.data import ResponseMatrix
from irtbench.diagnostics import (
    NEGATIVE_DISCRIMINATION,
    NEGATIVE_REST_CORRELATION,
    diagnose_items,
    fit_statistics,
    item_information,
    rest_correlation,
    test_information,
)
from irtbench.mml import IRTFit, fit_mml
from irtbench.simulate import simulate_responses


def test_rest_correlation_matches_scipy_point_biserial():
    data, _ = simulate_responses(120, 7, kind="2pl", seed=41, missing=0.1)
    ours = rest_correlation(data)
    scores = data.scores
    for j in range(data.n_items):
        rows = [i for i in range(data.n_models) if not np.isnan(scores[i, j])]
        rest, cells = [], []
        for i in rows:
            others = np.delete(scores[i], j)
            others = others[~np.isnan(others)]
            if others.size:
                rest.append(others.mean())
                cells.append(scores[i, j])
        expected = stats.pointbiserialr(cells, rest)[0]
        assert ours[j] == pytest.approx(expected, abs=1e-12)


def test_rest_correlation_is_nan_without_variation():
    scores = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    data = ResponseMatrix(("a", "b", "c", "d"), ("q1", "q2", "q3"), scores)
    correlation = rest_correlation(data)
    assert np.isnan(correlation[0])
    assert np.isfinite(correlation[1:]).all()


def test_information_has_its_closed_form_and_peaks_at_the_difficulty():
    fit = IRTFit.from_parameters(("q1", "q2"), [2.0, 0.5], [-1.0, 0.0])
    theta = np.array([0.5, 0.0, 3.0])
    info = item_information(fit, theta)
    assert info[0, 0] == pytest.approx(2.0**2 * 0.25)  # at b = 0.5, P = 1/2
    p = 1 / (1 + np.exp(-(0.5 * 3.0)))
    assert info[2, 1] == pytest.approx(0.25 * p * (1 - p))
    grid = np.linspace(-4, 4, 801)
    assert grid[np.argmax(item_information(fit, grid)[:, 0])] == pytest.approx(0.5, abs=0.01)
    np.testing.assert_allclose(test_information(fit, theta), info.sum(axis=1))


def test_a_reversed_item_is_flagged():
    data, _ = simulate_responses(300, 12, kind="2pl", seed=42)
    scores = data.scores.copy()
    scores[:, 3] = 1.0 - scores[:, 3]  # stronger models now fail item 4
    extra = np.column_stack([np.ones(300), np.zeros(300)])
    broken = ResponseMatrix(
        data.models, data.items + ("free", "stuck"), np.hstack([scores, extra])
    )
    fit = fit_mml(broken, "2pl")
    report = diagnose_items(broken, fit, rasch=fit_mml(broken, "1pl"))
    assert report.flags[3] == (NEGATIVE_DISCRIMINATION, NEGATIVE_REST_CORRELATION)
    assert report.flagged(NEGATIVE_DISCRIMINATION) == (broken.items[3],)
    assert report.flagged("solved_by_all") == ("free",)
    assert report.flagged("solved_by_none") == ("stuck",)
    clean = [flags for j, flags in enumerate(report.flags) if j not in (3, 12, 13)]
    assert all(flags == () for flags in clean)
    # the reversed item misfits the Rasch model far more than any other
    assert np.nanargmax(report.outfit) == 3
    assert np.nanargmax(report.infit) == 3
    assert report.infit[3] > 1.1
    assert np.isnan(report.infit[12:]).all()
    rows = report.to_dict()
    assert rows[3]["flags"] == [NEGATIVE_DISCRIMINATION, NEGATIVE_REST_CORRELATION]
    assert rows[12]["discrimination"] is None


def test_fit_statistics_are_near_one_when_the_rasch_model_holds():
    data, _ = simulate_responses(1500, 60, kind="1pl", seed=43)
    fit = fit_mml(data, "1pl")
    infit, outfit = fit_statistics(fit, data)
    np.testing.assert_allclose(infit, 1.0, atol=0.08)
    np.testing.assert_allclose(outfit, 1.0, atol=0.25)
    report = diagnose_items(data, fit)
    np.testing.assert_allclose(report.infit, infit)


def test_fit_statistics_hand_worked():
    # one item, abilities fixed by a second item being absent: check the formula on 3 models
    fit = IRTFit.from_parameters(("q1",), [1.0], [0.0], kind="1pl")
    data = ResponseMatrix(("a", "b", "c"), ("q1",), np.array([[1.0], [0.0], [1.0]]))
    from irtbench.mml import estimate_abilities

    theta = estimate_abilities(fit, data).theta
    p = 1 / (1 + np.exp(-theta))
    x = np.array([1.0, 0.0, 1.0])
    infit, outfit = fit_statistics(fit, data)
    assert infit[0] == pytest.approx(((x - p) ** 2).sum() / (p * (1 - p)).sum())
    assert outfit[0] == pytest.approx(np.mean((x - p) ** 2 / (p * (1 - p))))


def test_diagnostics_need_matching_items():
    data, _ = simulate_responses(50, 5, kind="1pl", seed=44)
    fit = fit_mml(data, "1pl")
    with pytest.raises(ValueError, match="same items"):
        diagnose_items(data.take_items([1, 0, 2, 3, 4]), fit)
    stranger = ResponseMatrix(("m",), ("zz",), np.array([[1.0]]))
    with pytest.raises(ValueError, match="not in the fit"):
        fit_statistics(fit, stranger)
    assert np.isnan(diagnose_items(data, fit_mml(data, "2pl")).infit).all()
