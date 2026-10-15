"""Tests for the shared labeled-matrix writer."""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.experiments import matrices
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.matrices import plot_matrix_file
from tomography.experiments.matrices import write_matrix

ROWS = ("a", "b", "c")
COLS = ("x", "y")


def _run_dir(tmp_path: Path) -> RunDir:
    return RunDir.create(
        ExperimentConfig(experiment="t1", model="tiny"), out_root=tmp_path
    )


def _values() -> NDArray[np.float64]:
    return np.arange(6, dtype=np.float64).reshape(3, 2) / 10


def test_write_matrix_round_trips_the_values_and_labels(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    write_matrix(
        run_dir, "grid", _values(), rows=ROWS, cols=COLS, title="Cosine at 3"
    )
    arrays = run_dir.read_arrays("grid")
    assert sorted(arrays) == ["values"]
    assert arrays["values"].dtype == np.float64
    np.testing.assert_array_equal(arrays["values"], _values())
    assert run_dir.read_json("grid") == {
        "rows": list(ROWS),
        "cols": list(COLS),
        "title": "Cosine at 3",
    }


def test_write_matrix_uses_the_stem_and_keeps_blank_cells(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    values = _values()
    values[1, 0] = np.nan
    write_matrix(run_dir, "m2", values, rows=ROWS, cols=COLS, title="t")
    assert (run_dir.path / "m2.npz").is_file()
    assert (run_dir.path / "m2.json").is_file()
    assert not (run_dir.path / "grid.npz").exists()
    np.testing.assert_array_equal(run_dir.read_arrays("m2")["values"], values)


@pytest.mark.parametrize(
    ("values", "rows", "cols", "message"),
    [
        (np.zeros(6), ROWS, COLS, "2-D"),
        (np.zeros((3, 2, 1)), ROWS, COLS, "2-D"),
        (np.zeros((3, 2)), ROWS[:2], COLS, "rows"),
        (np.zeros((3, 2)), ROWS, COLS[:1], "cols"),
    ],
    ids=["1d", "3d", "row-count", "col-count"],
)
def test_write_matrix_validates_the_shape_before_writing(
    tmp_path: Path,
    values: NDArray[np.float64],
    rows: tuple[str, ...],
    cols: tuple[str, ...],
    message: str,
) -> None:
    run_dir = _run_dir(tmp_path)
    with pytest.raises(ValueError, match=message):
        write_matrix(run_dir, "grid", values, rows=rows, cols=cols, title="t")
    assert not (run_dir.path / "grid.npz").exists()
    assert not (run_dir.path / "grid.json").exists()


def _written(tmp_path: Path, stem: str = "grid") -> RunDir:
    run_dir = _run_dir(tmp_path)
    write_matrix(
        run_dir, stem, _values(), rows=ROWS, cols=COLS, title="Cosine at 3"
    )
    return run_dir


def test_plot_matrix_file_writes_three_formats_named_after_the_stem(
    tmp_path: Path,
) -> None:
    run_dir = _written(tmp_path, "grid")
    write_matrix(
        run_dir, "other", _values(), rows=ROWS, cols=COLS, title="Other"
    )
    out = tmp_path / "figs"
    paths = plot_matrix_file(run_dir, out, stem="other")
    assert sorted(p.name for p in paths) == [
        "other.pdf",
        "other.png",
        "other.svg",
    ]
    assert all(p.parent == out and p.stat().st_size > 0 for p in paths)
    assert not (out / "grid.png").exists()


def test_plot_matrix_file_draws_the_stored_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[NDArray[np.float64], dict[str, Any]]] = []
    real = matrices.plot_matrix

    def spy(values: NDArray[np.float64], **kwargs: Any) -> Any:
        calls.append((values, kwargs))
        return real(values, **kwargs)

    monkeypatch.setattr(matrices, "plot_matrix", spy)
    plot_matrix_file(_written(tmp_path), tmp_path / "figs", stem="grid")
    ((values, kwargs),) = calls
    np.testing.assert_array_equal(values, _values())
    assert kwargs == {
        "rows": list(ROWS),
        "cols": list(COLS),
        "title": "Cosine at 3",
        "kind": "sequential",
        "vmin": None,
        "vmax": None,
    }


def test_plot_matrix_file_passes_the_kind_and_the_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs: list[dict[str, Any]] = []
    real = matrices.plot_matrix

    def spy(values: NDArray[np.float64], **options: Any) -> Any:
        kwargs.append(options)
        return real(values, **options)

    monkeypatch.setattr(matrices, "plot_matrix", spy)
    paths = plot_matrix_file(
        _written(tmp_path),
        tmp_path / "figs",
        stem="grid",
        kind="diverging",
        vmin=-1.0,
        vmax=1.0,
    )
    assert [(k["kind"], k["vmin"], k["vmax"]) for k in kwargs] == [
        ("diverging", -1.0, 1.0)
    ]
    assert all(p.stat().st_size > 0 for p in paths)


def test_plot_matrix_file_draws_blank_cells(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    values = _values()
    values[0, 1] = np.nan
    write_matrix(run_dir, "grid", values, rows=ROWS, cols=COLS, title="t")
    paths = plot_matrix_file(run_dir, tmp_path / "figs", stem="grid")
    assert len(paths) == 3
