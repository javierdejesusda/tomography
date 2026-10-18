from pathlib import Path

import pytest

from tomography.site.tables import _cells


def test_num_writes_fixed_decimals_and_a_plain_hyphen_minus() -> None:
    assert _cells.num(0.9143) == "0.914"
    assert _cells.num(-0.0354, 3) == "-0.035"
    assert _cells.num(1.0, 2) == "1.00"


def test_num_writes_n_a_for_a_missing_value() -> None:
    assert _cells.num(None) == "n/a"


def test_interval_puts_the_bounds_in_brackets() -> None:
    assert _cells.interval(0.914, 0.895, 0.931) == "0.914 [0.895, 0.931]"
    assert _cells.interval(0.5, None, None) == "0.500"
    assert _cells.interval(None, None, None) == "n/a"


def test_percent_rounds_to_whole_numbers() -> None:
    assert _cells.percent(0.5916) == "59"
    assert _cells.percent(None) == "n/a"


def test_count_writes_a_ratio_with_thousands_separators() -> None:
    assert _cells.count(1377, 21988) == "1,377 / 21,988"


def test_read_csv_parses_numbers_and_strips_the_bom(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("\ufeffa,b\n1,x\n,2.5\n", encoding="utf-8")
    assert _cells.read_csv(path) == [
        {"a": 1.0, "b": "x"},
        {"a": None, "b": 2.5},
    ]


def test_rows_of_a_json_table_zips_columns_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    path.write_text(
        '{"_meta": {}, "data": {"columns": ["a", "b"], "rows": [[1, 2]]}}',
        encoding="utf-8",
    )
    assert _cells.read_table(path) == [{"a": 1, "b": 2}]
    path.write_text('{"_meta": {}, "data": [{"a": 3}]}', encoding="utf-8")
    assert _cells.read_table(path) == [{"a": 3}]


def test_unknown_alignment_is_rejected() -> None:
    with pytest.raises(ValueError, match="alignment"):
        _cells.check_align("lx", 2)
    with pytest.raises(ValueError, match="alignment"):
        _cells.check_align("lr", 3)
    assert _cells.check_align("lr", 2) == "lr"
