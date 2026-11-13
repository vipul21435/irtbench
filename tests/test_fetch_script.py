import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_swebench_verified.py"


def load_script():
    spec = importlib.util.spec_from_file_location("fetch_swebench_verified", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_submissions_are_read_from_either_file_layout(tmp_path):
    root = tmp_path / "evaluation" / "verified"
    write(
        root / "sub-a" / "results" / "results.json",
        {"resolved": ["i1", "i3"], "no_generation": ["i2"], "generated": ["i1", "i3", "i4"]},
    )
    write(
        root / "sub-b" / "per_instance_details.json",
        {
            "i1": {"resolved": True, "cost": 0.1},
            "i2": {"resolved": False},
            "i5": {"resolved": True},
            "i6": {"cost": 1.0},
        },
    )
    write(root / "sub-c" / "results" / "results.json", ["not", "a", "mapping"])
    write(root / "sub-d" / "results" / "results.json", {"no_logs": ["i1"]})
    write(root / "sub-e" / "per_instance_details.json", ["i1"])
    (root / "sub-f").mkdir()
    scores, source, universe = load_script().read_submissions(tmp_path)
    assert universe == {"i1", "i2", "i3", "i4", "i5"}
    assert set(scores) == {"sub-a", "sub-b"}
    assert scores["sub-a"] == {"i1": 1, "i2": 0, "i3": 1, "i4": 0, "i5": 0}
    assert scores["sub-b"] == {"i1": 1, "i2": 0, "i5": 1}
    assert source == {"sub-a": "results/results.json", "sub-b": "per_instance_details.json"}


def test_results_file_wins_when_a_submission_has_both(tmp_path):
    root = tmp_path / "evaluation" / "verified"
    write(root / "sub" / "results" / "results.json", {"resolved": ["i1"], "no_logs": ["i2"]})
    write(root / "sub" / "per_instance_details.json", {"i1": {"resolved": False}})
    scores, source, _ = load_script().read_submissions(tmp_path)
    assert scores["sub"] == {"i1": 1, "i2": 0}
    assert source["sub"] == "results/results.json"
