"""Tests for the writer of curves with given interval bands."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.intervals import write_interval_curves
from tomography.experiments.sweep import plot_curves_file

SERIES = ("known", "unknown")
X = (2000.0, 2004.5, 2009.0, 2013.5)


@dataclass(frozen=True, kw_only=True)
class _Config(ExperimentConfig):
    experiment: str = "t1"
    model: str = "tiny"


def _run_dir(tmp_path: Path) -> RunDir:
    return RunDir.create(_Config(), out_root=tmp_path)


def _bands() -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    mean = np.array([[0.2, 0.4, 0.6, 0.8], [0.9, 0.7, 0.5, 0.3]])
    return mean, mean - 0.1, mean + 0.05


def test_writes_the_arrays_and_the_sidecar(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    mean, lo, hi = _bands()

    write_interval_curves(
        run_dir,
        X,
        mean,
        lo,
        hi,
        series=SERIES,
        xlabel="Year",
        ylabel="Known rate",
        chance=0.5,
        n_layers=12,
    )

    arrays = run_dir.read_arrays("curves")
    assert set(arrays) == {"x", "mean", "lo", "hi", "per_seed"}
    np.testing.assert_array_equal(arrays["x"], np.array(X))
    np.testing.assert_array_equal(arrays["mean"], mean)
    np.testing.assert_array_equal(arrays["lo"], lo)
    np.testing.assert_array_equal(arrays["hi"], hi)
    assert arrays["per_seed"].shape == (2, 1, 4)
    np.testing.assert_array_equal(arrays["per_seed"], mean[:, None, :])
    assert run_dir.read_json("curves") == {
        "series": ["known", "unknown"],
        "xlabel": "Year",
        "ylabel": "Known rate",
        "chance": 0.5,
        "n_layers": 12,
    }


def test_stores_float64_whatever_the_input_type(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    mean = np.array([[1, 2, 3], [4, 5, 6]])

    write_interval_curves(
        run_dir,
        [2000, 2001, 2002],
        mean,
        mean - 1,
        mean + 1,
        series=SERIES,
        xlabel="Year",
        ylabel="Count",
    )

    arrays = run_dir.read_arrays("curves")
    assert {key: value.dtype for key, value in arrays.items()} == {
        key: np.dtype(np.float64) for key in arrays
    }
    np.testing.assert_array_equal(arrays["x"], np.array([2000.0, 2001, 2002]))
    np.testing.assert_array_equal(arrays["hi"], mean + 1.0)


def test_chance_and_depth_default_to_none(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    mean, lo, hi = _bands()

    write_interval_curves(
        run_dir, X, mean, lo, hi, series=SERIES, xlabel="x", ylabel="y"
    )

    sidecar = run_dir.read_json("curves")
    assert isinstance(sidecar, dict)
    assert sidecar["chance"] is None
    assert sidecar["n_layers"] is None


def test_the_stem_names_the_pair(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    mean, lo, hi = _bands()

    write_interval_curves(
        run_dir,
        X,
        mean,
        lo,
        hi,
        series=SERIES,
        stem="risk_coverage",
        xlabel="Coverage",
        ylabel="Risk",
    )

    assert (run_dir.path / "risk_coverage.npz").exists()
    assert (run_dir.path / "risk_coverage.json").exists()
    assert not (run_dir.path / "curves.npz").exists()
    assert not (run_dir.path / "curves.json").exists()


def test_the_pair_is_drawn_by_plot_curves_file(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    mean, lo, hi = _bands()
    write_interval_curves(
        run_dir,
        X,
        mean,
        lo,
        hi,
        series=SERIES,
        xlabel="Year",
        ylabel="Known rate",
        chance=0.5,
    )
    out = tmp_path / "figs"

    paths = plot_curves_file(run_dir, out)

    assert sorted(p.name for p in paths) == [
        "curves.pdf",
        "curves.png",
        "curves.svg",
    ]
    assert all(p.parent == out and p.stat().st_size > 0 for p in paths)


def test_a_named_pair_is_drawn_under_its_name(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    mean, lo, hi = _bands()
    write_interval_curves(
        run_dir,
        X,
        mean,
        lo,
        hi,
        series=SERIES,
        stem="dose",
        xlabel="Coefficient",
        ylabel="Effect",
        n_layers=12,
    )

    paths = plot_curves_file(run_dir, tmp_path / "figs", stem="dose")

    assert sorted(p.name for p in paths) == ["dose.pdf", "dose.png", "dose.svg"]


@pytest.mark.parametrize(
    ("x", "mean", "lo", "hi", "series", "message"),
    [
        (
            X,
            np.zeros(4),
            np.zeros(4),
            np.zeros(4),
            SERIES,
            r"mean must have shape \(S, K\)",
        ),
        (
            X,
            np.zeros((2, 4)),
            np.zeros((2, 3)),
            np.zeros((2, 4)),
            SERIES,
            r"lo must have the shape of mean",
        ),
        (
            X,
            np.zeros((2, 4)),
            np.zeros((2, 4)),
            np.zeros((1, 4)),
            SERIES,
            r"hi must have the shape of mean",
        ),
        (
            X,
            np.zeros((2, 4)),
            np.zeros((2, 4)),
            np.zeros((2, 4)),
            ("a",),
            r"1 series names for 2 series",
        ),
        (
            X[:3],
            np.zeros((2, 4)),
            np.zeros((2, 4)),
            np.zeros((2, 4)),
            SERIES,
            r"3 x values for 4 columns",
        ),
    ],
    ids=["mean-not-2d", "lo-shape", "hi-shape", "series-count", "x-length"],
)
def test_validates_shapes_and_lengths(
    tmp_path: Path,
    x: Sequence[float],
    mean: NDArray[np.float64],
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    series: Sequence[str],
    message: str,
) -> None:
    run_dir = _run_dir(tmp_path)

    with pytest.raises(ValueError, match=message):
        write_interval_curves(
            run_dir, x, mean, lo, hi, series=series, xlabel="x", ylabel="y"
        )

    assert not (run_dir.path / "curves.npz").exists()
    assert not (run_dir.path / "curves.json").exists()
