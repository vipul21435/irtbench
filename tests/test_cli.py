import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from irtbench import __version__
from irtbench.cli import main
from irtbench.data import ResponseMatrix, read_matrix, write_long
from irtbench.simulate import simulate_responses

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "toy_results.csv"
ANSWERS = EXAMPLE.with_name("answers.csv")


@pytest.fixture
def long_csv(tmp_path):
    data, _ = simulate_responses(80, 8, kind="1pl", seed=21, missing=0.05)
    path = tmp_path / "results.csv"
    write_long(data, path)
    return path


def test_fit_prints_a_summary_and_writes_json(long_csv, tmp_path, capsys):
    out = tmp_path / "fit.json"
    assert main(["fit", str(long_csv), "--irt-model", "1pl", "--json", str(out)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    text = captured.out
    assert text.startswith("# irtbench fit: results.csv\n")
    assert "- data: 80 models x 8 items, " in text
    assert "- model: 1pl by marginal maximum likelihood (EM on a " in text
    assert "with 9 parameters; EM converged in" in text
    assert "## Items (hardest first, 8 of 8)" in text
    assert "## Models (strongest first, 10 of 80)" in text
    assert "| item | status | solved | discrimination | difficulty |" in text

    payload = json.loads(out.read_text())
    assert payload["irtbench"] == __version__
    assert payload["data"]["n_models"] == 80
    assert payload["fit"]["kind"] == "1pl"
    assert payload["fit"]["converged"] is True
    assert len(payload["fit"]["items"]) == 8
    assert len(payload["models"]) == 80
    assert set(payload["models"][0]) == {"model", "theta", "sd", "n_answered", "solved"}


def test_fit_orders_items_by_difficulty_and_models_by_ability(long_csv, capsys):
    assert main(["fit", str(long_csv), "--irt-model", "1pl", "--rows", "0"]) == 0
    text = capsys.readouterr().out
    items, models = text.split("## Models")
    difficulties = [float(line.split("|")[5]) for line in items.splitlines() if "| item-" in line]
    abilities = [float(line.split("|")[4]) for line in models.splitlines() if "| model-" in line]
    assert len(difficulties) == 8
    assert len(abilities) == 80
    assert difficulties == sorted(difficulties, reverse=True)
    assert abilities == sorted(abilities, reverse=True)


def test_fit_reads_a_wide_matrix_with_flagged_items(tmp_path, capsys):
    data, _ = simulate_responses(60, 5, kind="1pl", seed=22)
    scores = np.hstack([data.scores, np.ones((60, 1)), np.zeros((60, 1))])
    header = ",".join(("model",) + data.items + ("free", "stuck"))
    rows = [
        ",".join([name] + [str(int(cell)) for cell in scores[index]])
        for index, name in enumerate(data.models)
    ]
    path = tmp_path / "wide.csv"
    path.write_text("\n".join([header] + rows) + "\n")
    assert main(["fit", str(path), "--irt-model", "1pl", "--rows", "0"]) == 0
    text = capsys.readouterr().out
    assert "- items left out of the fit: 1 solved by all, 1 solved by none, 0 without data" in text
    rows_of = ("| item-", "| free", "| stuck")
    table = [line for line in text.splitlines() if line.startswith(rows_of)]
    assert table[0] == "| stuck | all_incorrect | 0.00 |  | inf |"
    assert table[-1] == "| free | all_correct | 1.00 |  | -inf |"


def test_fit_accepts_renamed_columns_in_jsonl(tmp_path, capsys):
    data, _ = simulate_responses(50, 4, kind="1pl", seed=23)
    path = tmp_path / "runs.jsonl"
    with path.open("w") as handle:
        for i, model in enumerate(data.models):
            for j, item in enumerate(data.items):
                record = {"system": model, "instance": item, "resolved": bool(data.scores[i, j])}
                handle.write(json.dumps(record) + "\n")
    code = main(
        [
            "fit",
            str(path),
            "--irt-model",
            "1pl",
            "--model-col",
            "system",
            "--item-col",
            "instance",
            "--score-col",
            "resolved",
        ]
    )
    assert code == 0
    assert "- data: 50 models x 4 items, 0 of 200 cells missing" in capsys.readouterr().out


def test_fit_warns_when_em_stops_at_the_limit(long_csv, capsys):
    assert main(["fit", str(long_csv), "--max-iter", "2"]) == 0
    captured = capsys.readouterr()
    assert "EM stopped at the limit of 2 iterations without converging" in captured.out
    assert "warning: EM did not converge in 2 iterations" in captured.err


def test_bad_input_is_an_error_not_a_traceback(tmp_path, capsys):
    assert main(["fit", str(tmp_path / "absent.csv")]) == 2
    assert "irtbench: error:" in capsys.readouterr().err
    path = tmp_path / "bad.csv"
    path.write_text("model,item,score\nm1,q1,0.5\n")
    assert main(["fit", str(path)]) == 2
    assert "score must be 0 or 1" in capsys.readouterr().err
    constant = ResponseMatrix(("a", "b"), ("q1", "q2", "q3"), np.ones((2, 3)))
    write_long(constant, path)
    assert main(["fit", str(path)]) == 2
    assert "needs at least 3 items" in capsys.readouterr().err


def test_a_command_is_required(capsys):
    with pytest.raises(SystemExit) as raised:
        main([])
    assert raised.value.code == 2


def test_module_entry_point_reports_the_version():
    result = subprocess.run(
        [sys.executable, "-m", "irtbench", "--version"], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == f"irtbench {__version__}"


@pytest.mark.skipif(not EXAMPLE.exists(), reason="examples are not shipped with this install")
def test_bundled_example_fits(capsys):
    assert main(["fit", str(EXAMPLE), "--irt-model", "1pl"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "- data: 40 models x 24 items, 27 of 960 cells missing" in captured.out
    assert "EM converged" in captured.out


@pytest.fixture
def benchmark_csv(tmp_path):
    data, _ = simulate_responses(60, 30, kind="2pl", seed=71)
    scores = data.scores.copy()
    scores[:, 2] = 1.0 - scores[:, 2]  # one item that stronger models fail
    extra = np.column_stack([np.ones(60), np.zeros(60)])
    wider = ResponseMatrix(
        data.models, data.items + ("free", "stuck"), np.hstack([scores, extra])
    )
    path = tmp_path / "bench.csv"
    write_long(wider, path)
    return path


def test_items_lists_the_item_that_behaves_backwards(benchmark_csv, tmp_path, capsys):
    out = tmp_path / "items.json"
    assert main(["items", str(benchmark_csv), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.startswith("# irtbench items: bench.csv\n")
    assert "- solved by all: 1; solved by none: 1" in text
    assert "- negative discrimination: 1; negative correlation with the rest score: " in text
    table = [line for line in text.splitlines() if line.startswith("| item-")]
    assert table[0].startswith("| item-03 | negative_discrimination, negative_rest_correlation |")
    rows = json.loads(out.read_text())["items"]
    assert len(rows) == 32
    assert rows[2]["flags"] == ["negative_discrimination", "negative_rest_correlation"]
    assert rows[30]["flags"] == ["solved_by_all"]
    assert rows[0]["infit"] is not None


def test_subset_then_predict_round_trip(benchmark_csv, tmp_path, capsys):
    saved = tmp_path / "subset.json"
    assert main(["subset", str(benchmark_csv), "-k", "8", "--json", str(saved)]) == 0
    chosen = capsys.readouterr().out.split()
    assert len(chosen) == len(set(chosen)) == 8
    payload = json.loads(saved.read_text())
    assert payload["items"] == chosen
    assert payload["method"] == "information"

    # a new file holding only the answers on the chosen items
    full = {}
    for line in benchmark_csv.read_text().splitlines()[1:]:
        model, item, score = line.split(",")
        full.setdefault(model, []).append(int(score))
    answers = tmp_path / "answers.csv"
    kept = [
        line for line in benchmark_csv.read_text().splitlines()[1:] if line.split(",")[1] in chosen
    ]
    answers.write_text("model,item,score\n" + "\n".join(kept) + "\n")
    result = tmp_path / "predicted.json"
    code = main(
        ["predict", str(answers), "--fit", str(saved), "--rows", "0", "--json", str(result)]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "- fit: 2pl on 32 items; answers given on 8 of them" in text
    assert "| model | answered | solved | ability | sd | predicted score |" in text
    models = json.loads(result.read_text())["models"]
    assert len(models) == 60
    errors = [abs(row["predicted_score"] - np.mean(full[row["model"]])) for row in models]
    assert np.mean(errors) < 0.1
    assert all(row["n_answered"] == 8 for row in models)


def test_subset_methods_and_errors(benchmark_csv, tmp_path, capsys):
    assert main(["subset", str(benchmark_csv), "-k", "5", "--method", "random", "--seed", "3"]) == 0
    first = capsys.readouterr().out
    assert main(["subset", str(benchmark_csv), "-k", "5", "--method", "random", "--seed", "3"]) == 0
    assert capsys.readouterr().out == first
    assert main(["subset", str(benchmark_csv), "-k", "5", "--method", "cluster"]) == 0
    assert len(capsys.readouterr().out.split()) == 5
    assert main(["subset", str(benchmark_csv), "-k", "500"]) == 2
    assert "k must be between 1 and 30" in capsys.readouterr().err


@pytest.mark.skipif(not ANSWERS.exists(), reason="examples are not shipped with this install")
def test_readme_quickstart_predicts_from_the_bundled_answers(tmp_path, capsys):
    saved = tmp_path / "subset.json"
    assert main(["subset", str(EXAMPLE), "-k", "6", "--json", str(saved)]) == 0
    chosen = capsys.readouterr().out.split()
    assert len(chosen) == 6
    # answers.csv holds the cells of three models on exactly the six chosen items
    answers, full = read_matrix(ANSWERS), read_matrix(EXAMPLE)
    assert answers.models == ("model-01", "model-02", "model-03")
    assert sorted(answers.items) == sorted(chosen)
    assert answers.n_missing == 0
    rows = [full.models.index(model) for model in answers.models]
    cols = [full.items.index(item) for item in answers.items]
    np.testing.assert_array_equal(answers.scores, full.scores[np.ix_(rows, cols)])

    out = tmp_path / "predict.json"
    assert main(["predict", str(ANSWERS), "--fit", str(saved), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.startswith("# irtbench predict: answers.csv\n")
    assert "- fit: 2pl on 24 items; answers given on 6 of them" in text
    # the table printed in the README, to the three decimals it shows
    predicted = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0].startswith("model-"):
            predicted[cells[0]] = float(cells[-1])
    assert list(predicted) == ["model-02", "model-01", "model-03"]
    expected = {"model-02": 0.461, "model-01": 0.289, "model-03": 0.279}
    for model, value in expected.items():
        assert predicted[model] == pytest.approx(value, abs=2e-3)


def test_predict_rejects_a_file_that_is_not_a_fit(benchmark_csv, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert main(["predict", str(benchmark_csv), "--fit", str(bad)]) == 2
    assert "not valid JSON" in capsys.readouterr().err
    bad.write_text('{"kind": "2pl"}')
    assert main(["predict", str(benchmark_csv), "--fit", str(bad)]) == 2
    assert "not a fit written by irtbench" in capsys.readouterr().err
    bad.write_text("[1, 2]")
    assert main(["predict", str(benchmark_csv), "--fit", str(bad)]) == 2
    assert "not a fit written by irtbench" in capsys.readouterr().err


def test_report_covers_every_section_and_writes_charts(benchmark_csv, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out, saved, plots = tmp_path / "report.md", tmp_path / "report.json", tmp_path / "charts"
    code = main(
        [
            "report",
            str(benchmark_csv),
            "--folds",
            "2",
            "--splits",
            "2",
            "--sizes",
            "5,10,999",
            "--out",
            str(out),
            "--json",
            str(saved),
            "--plots",
            str(plots),
        ]
    )
    assert code == 0
    assert capsys.readouterr().out == ""
    text = out.read_text()
    for heading in (
        "# irtbench report: bench.csv",
        "## 1PL against 2PL",
        "Held-out cells (2-fold):",
        "## Items that behave backwards (",
        "## Predicting the full score of unseen models from k items",
        "## Models (strongest first, 10 of 60)",
        "## Charts",
    ):
        assert heading in text
    assert "| k | information | cluster | random | random_mean |" in text
    assert "Likelihood ratio: " in text
    payload = json.loads(saved.read_text())
    assert set(payload["subsets"]["sizes"]) == {"5", "10"}
    assert set(payload["held_out_cells"]) == {"1pl", "2pl", "item_rate"}
    assert payload["criteria"]["2pl"]["n_parameters"] == 60
    for name in ("items.png", "information.png", "subsets.png"):
        assert (plots / name).read_bytes().startswith(b"\x89PNG")


def test_report_can_skip_the_slow_parts(benchmark_csv, capsys):
    assert main(["report", str(benchmark_csv), "--folds", "0", "--splits", "0"]) == 0
    text = capsys.readouterr().out
    assert "Held-out cells" not in text
    assert "Predicting the full score" not in text
    assert main(["report", str(benchmark_csv), "--sizes", "ten"]) == 2
    assert "--sizes takes comma-separated integers" in capsys.readouterr().err


def test_no_prior_flag_reaches_the_fit(long_csv, tmp_path, capsys):
    out = tmp_path / "fit.json"
    assert main(["fit", str(long_csv), "--slope-prior-sd", "0", "--json", str(out)]) == 0
    assert json.loads(out.read_text())["fit"]["slope_prior_sd"] is None
    assert "slope prior" not in capsys.readouterr().out
    assert main(["fit", str(long_csv), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["fit"]["slope_prior_sd"] == 0.5
    assert "slope prior sd 0.5 around a common slope of" in capsys.readouterr().out


def test_items_can_add_standard_errors(benchmark_csv, tmp_path, capsys):
    out = tmp_path / "items.json"
    assert main(["items", str(benchmark_csv), "--se", "information", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "## Lowest discriminations with information standard errors" in text
    rows = json.loads(out.read_text())["items"]
    assert rows[0]["discrimination_se"] > 0
    assert rows[30]["discrimination_se"] is None
    code = main(["items", str(benchmark_csv), "--se", "bootstrap", "--boot", "5", "--rows", "3"])
    assert code == 0
    assert "## Lowest discriminations with bootstrap standard errors" in capsys.readouterr().out


def test_json_output_never_holds_nan(tmp_path):
    from irtbench.cli import _json_ready, _write_json

    payload = {"a": float("nan"), "b": [1.5, float("inf"), np.float64(2.0)], "c": np.int64(3)}
    assert _json_ready(payload) == {"a": None, "b": [1.5, None, 2.0], "c": 3}
    out = tmp_path / "x.json"
    _write_json(str(out), payload)
    assert json.loads(out.read_text()) == {"a": None, "b": [1.5, None, 2.0], "c": 3}
    _write_json(None, payload)
