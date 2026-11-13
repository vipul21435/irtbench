import numpy as np
import pytest
from scipy import stats

from irtbench.data import ResponseMatrix
from irtbench.mml import IRTFit, fit_mml, predict_probabilities
from irtbench.simulate import simulate_responses
from irtbench.subset import (
    evaluate_subsets,
    predict_scores,
    select_clusters,
    select_information,
    select_items,
    spread_abilities,
)


def two_groups():
    # ten easy flat items, ten hard steep items, plus one item nobody can be told apart by
    slopes = [0.6] * 10 + [2.0] * 10
    difficulty = [-1.5] * 10 + [1.5] * 10
    intercepts = [-a * b for a, b in zip(slopes, difficulty, strict=True)]
    items = tuple(f"q{index:02d}" for index in range(20))
    return IRTFit.from_parameters(items, slopes, intercepts)


def test_spread_abilities_are_population_quantiles():
    targets = spread_abilities()
    np.testing.assert_allclose(targets, stats.norm.ppf(np.linspace(0.1, 0.9, 9)))
    sample = np.arange(101.0)
    np.testing.assert_allclose(spread_abilities(sample, 3), [10.0, 50.0, 90.0])


def test_information_selection_takes_the_most_informative_item_per_target():
    fit = two_groups()
    low, high = np.array([-1.5]), np.array([1.5])
    assert set(select_information(fit, 4, low)) <= set(range(10))
    assert set(select_information(fit, 4, high)) <= set(range(10, 20))
    both = select_information(fit, 6, np.array([-1.5, 1.5]))
    assert sum(index < 10 for index in both) == 3
    assert len(set(both)) == 6


def test_cluster_selection_covers_every_kind_of_item():
    fit = two_groups()
    chosen = select_clusters(fit, 2, spread_abilities(), seed=3)
    assert sorted(index // 10 for index in chosen) == [0, 1]
    assert len(set(select_clusters(fit, 7, spread_abilities(), seed=3))) == 7


def test_select_items_is_seeded_and_skips_items_left_out():
    data, _ = simulate_responses(200, 12, kind="2pl", seed=61)
    wider = ResponseMatrix(
        data.models, data.items + ("free",), np.hstack([data.scores, np.ones((200, 1))])
    )
    fit = fit_mml(wider, "2pl")
    for method in ("information", "cluster", "random"):
        chosen = select_items(fit, 12, method, seed=5)
        assert sorted(chosen) == sorted(data.items)
        assert select_items(fit, 5, method, seed=5) == select_items(fit, 5, method, seed=5)
    assert select_items(fit, 5, "random", seed=5) != select_items(fit, 5, "random", seed=6)
    with pytest.raises(ValueError, match="k must be between 1 and 12"):
        select_items(fit, 13)
    with pytest.raises(ValueError, match="unknown method"):
        select_items(fit, 3, "greedy")


def test_predicted_score_counts_answers_as_given_and_fills_in_the_rest():
    data, _ = simulate_responses(150, 10, kind="2pl", seed=62)
    fit = fit_mml(data, "2pl")
    np.testing.assert_allclose(predict_scores(fit, data), data.model_means(), atol=1e-12)
    answers = data.take_items([0, 4, 7])
    probability = predict_probabilities(fit, answers)
    probability[:, [0, 4, 7]] = answers.scores
    np.testing.assert_allclose(predict_scores(fit, answers), probability.mean(axis=1))
    nothing = ResponseMatrix(("new",), data.items[:1], np.array([[np.nan]]))
    prior_only = predict_scores(fit, nothing)[0]
    assert prior_only == pytest.approx(data.scores.mean(), abs=0.03)


def test_subset_evaluation_beats_the_plain_mean_on_simulated_data():
    data, _ = simulate_responses(120, 150, kind="2pl", seed=63)
    result = evaluate_subsets(data, sizes=(10, 30), splits=4, random_draws=4, seed=1)
    assert result["n_test"] == 30
    assert result["n_train"] == 90
    for size in ("10", "30"):
        row = result["sizes"][size]
        assert set(row) == {"information", "cluster", "random", "random_mean"}
        assert row["random"]["mae"] < row["random_mean"]["mae"]
        assert row["information"]["mae"] < row["random_mean"]["mae"]
        assert 0 < row["cluster"]["mae"] < 0.1
    assert result["sizes"]["30"]["random"]["mae"] < result["sizes"]["10"]["random"]["mae"]
    again = evaluate_subsets(data, sizes=(10, 30), splits=4, random_draws=4, seed=1)
    assert again == result


def test_subset_evaluation_validates():
    data, _ = simulate_responses(20, 8, kind="1pl", seed=64)
    with pytest.raises(ValueError, match="too few models"):
        evaluate_subsets(data, sizes=(3,), test_share=0.99)
    with pytest.raises(ValueError, match="unknown method"):
        evaluate_subsets(data, sizes=(3,), methods=("best",))
