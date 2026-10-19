import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tomography.site import runfiles


def _write(path: Path, data: object) -> None:
    path.write_text(json.dumps({"_meta": {"k": 1}, "data": data}), "utf-8")


@pytest.fixture
def run(tmp_path: Path) -> Path:
    _write(tmp_path / "summary.json", {"best_layer": 7})
    return tmp_path


def test_summary_returns_the_data_payload(run: Path) -> None:
    assert runfiles.summary(run) == {"best_layer": 7}


def test_data_returns_the_payload_of_any_json_file(run: Path) -> None:
    _write(run / "curves.json", {"series": ["reader", "random"]})
    assert runfiles.data(run, "curves.json")["series"] == ["reader", "random"]


def test_rows_reads_a_columns_and_rows_table(run: Path) -> None:
    _write(
        run / "table.json", {"columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
    )
    assert runfiles.rows(run, "table.json") == [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]


def test_rows_reads_a_list_of_records(run: Path) -> None:
    _write(run / "table.json", [{"a": 1}, {"a": 2}])
    assert runfiles.rows(run, "table.json") == [{"a": 1}, {"a": 2}]


def test_rows_reads_csv_with_numbers_parsed_and_blanks_as_none(
    run: Path,
) -> None:
    with (run / "t.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "x", "y"])
        writer.writerow(["a", "0.5", ""])
    assert runfiles.rows(run, "t.csv") == [{"name": "a", "x": 0.5, "y": None}]


def test_arrays_reads_every_array_of_an_npz(run: Path) -> None:
    np.savez(run / "curves.npz", x=np.arange(3), mean=np.ones((2, 3)))
    loaded = runfiles.arrays(run, "curves.npz")
    assert set(loaded) == {"x", "mean"}
    assert loaded["mean"].shape == (2, 3)


def test_a_missing_file_raises_file_not_found(run: Path) -> None:
    with pytest.raises(FileNotFoundError):
        runfiles.data(run, "absent.json")
