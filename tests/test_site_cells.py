from pathlib import Path

import pytest

from tomography.site.tables import _cells

MINUS = chr(0x2212)


def test_num_writes_fixed_decimals_and_a_true_minus() -> None:
    assert _cells.num(0.9143) == "0.914"
    assert _cells.num(-0.0354, 3) == f"{MINUS}0.035"
    assert _cells.num(1.0, 2) == "1.00"


def test_num_rounds_first_so_a_negative_zero_has_no_sign() -> None:
    assert _cells.num(-0.0004) == "0.000"
    assert _cells.num(-1e-16) == "0.000"
    assert _cells.num(-0.0006) == f"{MINUS}0.001"


def test_pm_writes_a_symmetric_interval_with_the_plus_minus_sign() -> None:
    assert _cells.pm(0.109, 0.0204) == "0.109 ± 0.020"
    assert _cells.pm(-0.0004, 0.01) == "0.000 ± 0.010"
    assert _cells.pm(-0.25, 0.02) == f"{MINUS}0.250 ± 0.020"
    assert _cells.pm(None, 0.02) == "n/a"


def test_num_writes_n_a_for_a_missing_value() -> None:
    assert _cells.num(None) == "n/a"


def test_interval_puts_the_bounds_in_brackets() -> None:
    assert _cells.interval(0.914, 0.895, 0.931) == "0.914 [0.895, 0.931]"
    assert _cells.interval(0.5, None, None) == "0.500"
    assert _cells.interval(None, None, None) == "n/a"
    assert _cells.interval(0.011, -0.006, 0.032) == (
        f"0.011 [{MINUS}0.006, 0.032]"
    )


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
