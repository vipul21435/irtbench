import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from irtbench import __version__
from irtbench.cli import main
from irtbench.data import ResponseMatrix, write_long
from irtbench.simulate import simulate_responses

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "toy_results.csv"


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
