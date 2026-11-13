import json

import numpy as np
import pytest

from irtbench.data import (
    ResponseMatrix,
    from_long,
    parse_score,
    read_long,
    read_matrix,
    read_wide,
    write_long,
)


def test_parse_score_accepts_binary_spellings():
    assert parse_score(1) == 1.0
    assert parse_score(0.0) == 0.0
    assert parse_score(True) == 1.0
    assert parse_score(" False ") == 0.0
    assert parse_score("1") == 1.0
    assert parse_score(np.int64(1)) == 1.0


@pytest.mark.parametrize("cell", [None, "", "NA", "nan", "null", float("nan")])
def test_parse_score_missing(cell):
    assert np.isnan(parse_score(cell))


@pytest.mark.parametrize("cell", [0.5, 2, "yes", "0.3"])
def test_parse_score_rejects_non_binary(cell):
    with pytest.raises(ValueError, match="score must be 0 or 1"):
        parse_score(cell)


def test_from_long_keeps_first_appearance_order_and_marks_gaps():
    matrix = from_long([("b", "q2", 1), ("b", "q1", 0), ("a", "q2", 0)])
    assert matrix.models == ("b", "a")
    assert matrix.items == ("q2", "q1")
    assert matrix.scores[0].tolist() == [1.0, 0.0]
    assert matrix.scores[1, 0] == 0.0
    assert np.isnan(matrix.scores[1, 1])
    assert matrix.n_missing == 1
    assert matrix.observed.tolist() == [[True, True], [True, False]]


def test_from_long_explicit_missing_score_still_registers_the_labels():
    matrix = from_long([("a", "q1", 1), ("a", "q2", None), ("b", "q1", "")])
    assert matrix.models == ("a", "b")
    assert matrix.items == ("q1", "q2")
    assert matrix.n_missing == 3


def test_from_long_duplicates():
    same = from_long([("a", "q1", 1), ("a", "q1", "1")])
    assert same.scores.tolist() == [[1.0]]
    with pytest.raises(ValueError, match="conflicting scores"):
        from_long([("a", "q1", 1), ("a", "q1", 0)])


def test_from_long_reports_the_bad_record():
    with pytest.raises(ValueError, match=r"record 2 \(a, q2\)"):
        from_long([("a", "q1", 1), ("a", "q2", 0.5)])


def test_response_matrix_validates_and_is_read_only():
    with pytest.raises(ValueError, match="shape"):
        ResponseMatrix(("a",), ("q1", "q2"), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="0, 1 or NaN"):
        ResponseMatrix(("a",), ("q1",), np.array([[0.4]]))
    with pytest.raises(ValueError, match="duplicate item"):
        ResponseMatrix(("a",), ("q1", "q1"), np.zeros((1, 2)))
    matrix = ResponseMatrix(("a",), ("q1",), np.array([[1.0]]))
    with pytest.raises(ValueError):
        matrix.scores[0, 0] = 0.0


def test_means_ignore_missing_cells():
    matrix = ResponseMatrix(
        ("a", "b", "c"),
        ("q1", "q2"),
        np.array([[1.0, np.nan], [0.0, np.nan], [1.0, np.nan]]),
    )
    assert matrix.item_means()[0] == pytest.approx(2 / 3)
    assert np.isnan(matrix.item_means()[1])
    assert matrix.model_means().tolist() == [1.0, 0.0, 1.0]


def test_take_models_and_items():
    matrix = from_long([("a", "q1", 1), ("a", "q2", 0), ("b", "q1", 0), ("b", "q2", 1)])
    flipped = matrix.take_models([1, 0]).take_items([1, 0])
    assert flipped.models == ("b", "a")
    assert flipped.items == ("q2", "q1")
    assert flipped.scores.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_read_long_csv_and_tsv(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("model,item,score,extra\nm1,q1,1,x\nm1,q2,0,y\nm2,q1,,z\n")
    matrix = read_long(csv_path)
    assert matrix.models == ("m1", "m2")
    assert matrix.scores[0].tolist() == [1.0, 0.0]
    assert np.isnan(matrix.scores[1]).all()

    tsv_path = tmp_path / "results.tsv"
    tsv_path.write_text("system\tinstance\tresolved\nm1\tq1\ttrue\nm2\tq1\tfalse\n")
    renamed = read_long(tsv_path, model_col="system", item_col="instance", score_col="resolved")
    assert renamed.scores.tolist() == [[1.0], [0.0]]


def test_read_long_csv_needs_its_columns(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text("model,item\nm1,q1\n")
    with pytest.raises(ValueError, match="no column named score"):
        read_long(path)


def test_read_long_csv_rejects_a_row_shorter_than_the_header(tmp_path):
    path = tmp_path / "short.csv"
    # the second record lost its score field; an empty score needs its comma ("m2,q1,")
    path.write_text("model,item,score\nm1,q1,1\nm2,q1\nm2,q2,0\n")
    with pytest.raises(ValueError, match=r"short\.csv line 3: 2 fields for the 3 columns"):
        read_long(path)
    path.write_text("model,item,score\nm1,q1,1\nm2\n")
    with pytest.raises(ValueError, match=r"line 3: 1 fields for the 3 columns"):
        read_long(path)
    path.write_text("run,model,item,score\n7,m1,q1,1\n7,m2,q1\n")
    with pytest.raises(ValueError, match=r"line 3: 3 fields for the 4 columns"):
        read_matrix(path)
    path.write_text("model,item,score\nm1,q1,1\nm2,q1,\n")
    matrix = read_long(path)
    assert matrix.models == ("m1", "m2")
    assert matrix.n_missing == 1


def test_a_byte_order_mark_is_not_read_as_part_of_the_first_column(tmp_path):
    long_path = tmp_path / "long.csv"
    long_path.write_text("model,item,score\nm1,q1,1\nm2,q1,0\n", encoding="utf-8-sig")
    assert long_path.read_bytes().startswith(b"\xef\xbb\xbf")
    for matrix in (read_long(long_path), read_matrix(long_path)):
        assert matrix.models == ("m1", "m2")
        assert matrix.items == ("q1",)
        np.testing.assert_allclose(matrix.scores, [[1.0], [0.0]])
    wide_path = tmp_path / "wide.csv"
    wide_path.write_text("model,q1,q2\nm1,1,0\n", encoding="utf-8-sig")
    assert read_wide(wide_path).items == ("q1", "q2")
    lines_path = tmp_path / "long.jsonl"
    lines_path.write_text('{"model": "m1", "item": "q1", "score": 1}\n', encoding="utf-8-sig")
    np.testing.assert_allclose(read_long(lines_path).scores, [[1.0]])


def test_read_long_jsonl(tmp_path):
    path = tmp_path / "results.jsonl"
    rows = [
        {"model": "m1", "item": "q1", "score": 1},
        {"model": "m1", "item": "q2", "score": False},
        {"model": "m2", "item": "q1", "score": None},
        {"model": "m2", "item": "q2"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n")
    matrix = read_long(path)
    assert matrix.scores[0].tolist() == [1.0, 0.0]
    assert matrix.n_missing == 2


def test_read_long_jsonl_errors_name_the_line(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('{"model": "m1", "item": "q1", "score": 1}\nnot json\n')
    with pytest.raises(ValueError, match="line 2: not valid JSON"):
        read_long(path)
    path.write_text('{"model": "m1", "score": 1}\n')
    with pytest.raises(ValueError, match="line 1: no 'item' field"):
        read_long(path)
    path.write_text("[1, 2]\n")
    with pytest.raises(ValueError, match="expected a JSON object"):
        read_long(path)


def test_read_wide(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text("model,q1,q2,q3\nm1,1,0,\nm2,0,1,1\n\n")
    matrix = read_wide(path)
    assert matrix.models == ("m1", "m2")
    assert matrix.items == ("q1", "q2", "q3")
    assert matrix.n_missing == 1
    assert matrix.scores[1].tolist() == [0.0, 1.0, 1.0]


def test_read_wide_errors(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text("model,q1,q2\nm1,1\n")
    with pytest.raises(ValueError, match="row 2: 1 cells for 2 items"):
        read_wide(path)
    path.write_text("model,q1,q2\nm1,1,7\n")
    with pytest.raises(ValueError, match="row 2, item 'q2'"):
        read_wide(path)
    path.write_text("model,q1\n")
    with pytest.raises(ValueError, match="at least one model row"):
        read_wide(path)


def test_read_matrix_detects_the_layout(tmp_path):
    long_path = tmp_path / "long.csv"
    long_path.write_text("item,model,score\nq1,m1,1\nq1,m2,0\n")
    wide_path = tmp_path / "wide.csv"
    wide_path.write_text("name,q1\nm1,1\nm2,0\n")
    assert read_matrix(long_path).scores.tolist() == [[1.0], [0.0]]
    assert read_matrix(wide_path).scores.tolist() == [[1.0], [0.0]]
    assert read_matrix(wide_path, "wide").models == ("m1", "m2")
    with pytest.raises(ValueError, match="unknown format"):
        read_matrix(wide_path, "matrix")


def test_write_long_round_trip(tmp_path):
    matrix = ResponseMatrix(
        ("m1", "m2"), ("q1", "q2"), np.array([[1.0, np.nan], [0.0, 1.0]])
    )
    path = tmp_path / "out.csv"
    write_long(matrix, path)
    again = read_matrix(path)
    assert again.models == matrix.models
    assert again.items == matrix.items
    np.testing.assert_array_equal(again.scores, matrix.scores)
