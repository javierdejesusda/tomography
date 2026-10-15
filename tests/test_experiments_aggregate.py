"""Tests for the run-reading library shared by the study experiments."""

from collections.abc import Sequence
import dataclasses
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.cli import load_experiment
from tomography.experiments.aggregate import Curves
from tomography.experiments.aggregate import ReaderSet
from tomography.experiments.aggregate import best_point
from tomography.experiments.aggregate import common_layers
from tomography.experiments.aggregate import default_run
from tomography.experiments.aggregate import depth_layer
from tomography.experiments.aggregate import finished_run
from tomography.experiments.aggregate import gather_readers
from tomography.experiments.aggregate import label_stem
from tomography.experiments.aggregate import oriented
from tomography.experiments.aggregate import out_root_of
from tomography.experiments.aggregate import read_curves
from tomography.experiments.aggregate import read_readers
from tomography.experiments.aggregate import reading_point
from tomography.experiments.aggregate import relative_depth
from tomography.experiments.aggregate import resolve_runs
from tomography.experiments.aggregate import run_depth
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.sweep import write_curves

MODEL = "tiny"


def _config(
    experiment: str, model: str = MODEL, **overrides: object
) -> ExperimentConfig:
    """The default config of an experiment, optionally with changed fields."""
    config = load_experiment(experiment).config_cls.from_dict(
        {"experiment": experiment, "model": model}
    )
    return dataclasses.replace(config, **overrides)


def _run(
    out_root: Path,
    experiment: str,
    model: str = MODEL,
    *,
    finished: bool = True,
    **overrides: object,
) -> RunDir:
    run_dir = RunDir.create(
        _config(experiment, model, **overrides), out_root=out_root
    )
    if finished:
        run_dir.finalize()
    return run_dir


def test_out_root_of_is_the_folder_above_the_experiment_folder(
    tmp_path: Path,
) -> None:
    assert out_root_of(_run(tmp_path / "res", "r1")) == tmp_path / "res"


def test_finished_run_opens_a_finished_run(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "r1")
    assert finished_run(run_dir.path).path == run_dir.path


def test_finished_run_rejects_an_unfinished_run(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "r1", finished=False)
    with pytest.raises(ValueError, match="is not a finished run"):
        finished_run(run_dir.path)


def test_default_run_finds_the_default_config_run(tmp_path: Path) -> None:
    _run(tmp_path, "r1", n_seeds=4)
    expected = _run(tmp_path, "r1")
    _run(tmp_path, "r2")
    found = default_run(tmp_path, "r1", MODEL)
    assert found.path == expected.path
    assert found.config["experiment"] == "r1"


def test_default_run_names_the_run_that_is_missing(tmp_path: Path) -> None:
    _run(tmp_path, "r1", n_seeds=4)
    path = RunDir.path_for(_config("r1"), out_root=tmp_path)
    with pytest.raises(
        ValueError,
        match=f"no run of r1 on {MODEL} with the default config at",
    ) as caught:
        default_run(tmp_path, "r1", MODEL)
    assert str(path) in str(caught.value)


def test_default_run_does_not_fall_back_to_another_model(
    tmp_path: Path,
) -> None:
    _run(tmp_path, "r1", "other")
    with pytest.raises(ValueError, match="no run of r1"):
        default_run(tmp_path, "r1", MODEL)


def test_default_run_rejects_an_unfinished_default_run(tmp_path: Path) -> None:
    _run(tmp_path, "r1", finished=False)
    with pytest.raises(ValueError, match="is not a finished run"):
        default_run(tmp_path, "r1", MODEL)


def test_resolve_runs_maps_every_requested_pair_to_its_default_run(
    tmp_path: Path,
) -> None:
    runs = {
        (experiment, model): _run(tmp_path, experiment, model)
        for experiment in ("r1", "r2")
        for model in (MODEL, "other")
    }
    got = resolve_runs(tmp_path, ("r1", "r2"), (MODEL, "other"))
    assert list(got) == list(runs)
    assert {pair: run.path for pair, run in got.items()} == {
        pair: run.path for pair, run in runs.items()
    }


def test_an_extra_run_replaces_the_default_run_of_its_pair(
    tmp_path: Path,
) -> None:
    default = _run(tmp_path, "r2")
    extra = _run(tmp_path, "r1", n_seeds=4)
    got = resolve_runs(tmp_path, ("r1", "r2"), (MODEL,), [str(extra.path)])
    assert got[("r1", MODEL)].path == extra.path
    assert got[("r2", MODEL)].path == default.path


def test_resolve_runs_fails_for_a_missing_default_run(tmp_path: Path) -> None:
    _run(tmp_path, "r1")
    with pytest.raises(ValueError, match="no run of r2"):
        resolve_runs(tmp_path, ("r1", "r2"), (MODEL,))


def test_two_extra_runs_of_one_pair_are_refused(tmp_path: Path) -> None:
    first = _run(tmp_path, "r1", n_seeds=4)
    second = _run(tmp_path, "r1", n_seeds=5)
    with pytest.raises(ValueError, match="two extra runs of r1 on tiny"):
        resolve_runs(
            tmp_path, ("r1",), (MODEL,), [str(first.path), str(second.path)]
        )


def test_an_extra_run_of_a_pair_that_was_not_requested_is_refused(
    tmp_path: Path,
) -> None:
    _run(tmp_path, "r1")
    stray = _run(tmp_path, "r2", n_seeds=4)
    with pytest.raises(ValueError, match="not requested"):
        resolve_runs(tmp_path, ("r1",), (MODEL,), [str(stray.path)])


def test_an_extra_run_of_another_model_is_refused(tmp_path: Path) -> None:
    _run(tmp_path, "r1")
    stray = _run(tmp_path, "r1", "other", n_seeds=4)
    with pytest.raises(ValueError, match=r"r1 on other.*not requested"):
        resolve_runs(tmp_path, ("r1",), (MODEL,), [str(stray.path)])


def test_an_unfinished_extra_run_is_refused(tmp_path: Path) -> None:
    _run(tmp_path, "r1")
    extra = _run(tmp_path, "r1", finished=False, n_seeds=4)
    with pytest.raises(ValueError, match="is not a finished run"):
        resolve_runs(tmp_path, ("r1",), (MODEL,), [str(extra.path)])


def test_the_default_run_is_not_needed_when_an_extra_run_replaces_it(
    tmp_path: Path,
) -> None:
    extra = _run(tmp_path, "r1", n_seeds=4)
    got = resolve_runs(tmp_path, ("r1",), (MODEL,), [str(extra.path)])
    assert [run.path for run in got.values()] == [extra.path]


def _curves(
    run_dir: RunDir,
    stem: str = "curves",
    *,
    n_layers: int | None = 12,
    layers: tuple[int, ...] = (1, 2, 3),
    series: tuple[str, ...] = ("reader", "random"),
    values: NDArray[np.float64] | None = None,
    chance: float | None = None,
) -> None:
    per_seed = (
        np.random.default_rng(0).uniform(size=(len(series), 3, len(layers)))
        if values is None
        else values
    )
    write_curves(
        run_dir,
        per_seed,
        layers,
        series=series,
        stem=stem,
        n_layers=n_layers,
        chance=chance,
        n_boot=10,
    )


def _readers(
    run_dir: RunDir,
    stem: str = "readers",
    *,
    n: int = 3,
    layers: tuple[int, ...] = (1, 2, 3),
    d: int = 8,
    seed: int = 0,
    signs: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    directions = (3.0 * rng.normal(size=(n, len(layers), d))).astype(np.float32)
    run_dir.write_arrays(
        stem,
        layers=np.asarray(layers, dtype=np.int64),
        directions=directions,
        signs=np.ones((n, len(layers)), dtype=np.float32)
        if signs is None
        else signs,
    )
    return directions


def test_oriented_normalizes_and_applies_the_signs() -> None:
    directions = np.array([[[3.0, 4.0], [0.0, 2.0]]], dtype=np.float32)
    signs = np.array([[-1.0, 1.0]], dtype=np.float32)
    got = oriented(directions, signs)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, [[[-0.6, -0.8], [0.0, 1.0]]], atol=1e-7)


def test_oriented_rejects_a_zero_direction() -> None:
    directions = np.ones((2, 3, 4))
    directions[1, 2] = 0.0
    with pytest.raises(ValueError, match="zero"):
        oriented(directions, np.ones((2, 3)))


def test_oriented_rejects_signs_of_another_shape() -> None:
    with pytest.raises(ValueError, match="signs"):
        oriented(np.ones((2, 3, 4)), np.ones((2, 4)))


def test_run_depth_is_the_value_every_sidecar_agrees_on(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path, "r7")
    _curves(run_dir, "curves", n_layers=12)
    _curves(run_dir, "curves_val", n_layers=12)
    _curves(run_dir, "curves_fake", n_layers=None)
    assert run_depth(run_dir) == 12


def test_run_depth_refuses_sidecars_that_disagree(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "r7")
    _curves(run_dir, "curves", n_layers=12)
    _curves(run_dir, "curves_val", n_layers=13)
    with pytest.raises(ValueError, match="n_layers"):
        run_depth(run_dir)


def test_run_depth_needs_a_value(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "r7")
    with pytest.raises(ValueError, match="n_layers"):
        run_depth(run_dir)
    _curves(run_dir, "curves", n_layers=None)
    with pytest.raises(ValueError, match="n_layers"):
        run_depth(run_dir)


def test_read_readers_returns_one_oriented_set_for_a_readers_file(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path, "r1")
    _curves(run_dir, n_layers=12)
    signs = np.ones((3, 3), dtype=np.float32)
    signs[1, 2] = -1.0
    directions = _readers(run_dir, layers=(2, 4, 6), signs=signs)
    got = read_readers(run_dir)
    assert list(got) == ["r1"]
    readers = got["r1"]
    assert readers.layers == (2, 4, 6)
    assert readers.n_layers == 12
    assert readers.vectors.shape == (3, 3, 8)
    np.testing.assert_allclose(
        np.linalg.norm(readers.vectors, axis=-1), 1.0, atol=1e-12
    )
    expected = directions[1, 2] / np.linalg.norm(directions[1, 2])
    np.testing.assert_allclose(readers.vectors[1, 2], -expected, atol=1e-7)


def test_read_readers_labels_each_suffixed_file_and_skips_other_names(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path, "r6")
    _curves(run_dir, n_layers=12)
    for stem in ("readers_truthfulqa", "readers_arc", "readersx"):
        _readers(run_dir, stem)
    assert list(read_readers(run_dir)) == ["r6:arc", "r6:truthfulqa"]


def test_read_readers_puts_the_plain_file_before_the_suffixed_ones(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path, "r6")
    _curves(run_dir, n_layers=12)
    _readers(run_dir, "readers_anger")
    _readers(run_dir, "readers")
    assert list(read_readers(run_dir)) == ["r6", "r6:anger"]


def test_read_readers_needs_a_readers_file(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "r1")
    _curves(run_dir, n_layers=12)
    _readers(run_dir, "readersx")
    with pytest.raises(ValueError, match="no readers file"):
        read_readers(run_dir)


def test_read_readers_needs_a_depth(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "r1")
    _readers(run_dir)
    with pytest.raises(ValueError, match="n_layers"):
        read_readers(run_dir)


def test_read_readers_checks_the_layers_against_the_directions(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path, "r1")
    _curves(run_dir, n_layers=12)
    run_dir.write_arrays(
        "readers",
        layers=np.array([1, 2], dtype=np.int64),
        directions=np.ones((3, 3, 4), dtype=np.float32),
        signs=np.ones((3, 3), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="layers"):
        read_readers(run_dir)


def _set(
    n: int = 3, layers: tuple[int, ...] = (1, 2, 3), n_layers: int = 12
) -> ReaderSet:
    vectors = np.random.default_rng(n).normal(size=(n, len(layers), 4))
    return ReaderSet(layers, vectors, n_layers)


def test_common_layers_returns_layers_depth_and_the_smallest_count() -> None:
    readers = {"a": _set(5), "b": _set(3), "c": _set(4)}
    assert common_layers(readers) == ((1, 2, 3), 12, 3)


def test_common_layers_needs_a_set() -> None:
    with pytest.raises(ValueError, match="no reader"):
        common_layers({})


def test_common_layers_allows_the_series_limit_and_refuses_more() -> None:
    readers = {f"c{i}": _set() for i in range(9)}
    assert common_layers(dict(list(readers.items())[:8]))[2] == 3
    with pytest.raises(ValueError, match="at most 8"):
        common_layers(readers)
    assert common_layers(readers, max_series=9)[2] == 3


def test_common_layers_refuses_different_layers() -> None:
    readers = {"a": _set(), "b": _set(layers=(1, 2, 4))}
    with pytest.raises(ValueError, match="layers"):
        common_layers(readers)


def test_common_layers_refuses_different_depths() -> None:
    readers = {"a": _set(), "b": _set(n_layers=24)}
    with pytest.raises(ValueError, match="depth"):
        common_layers(readers)


def _finished_with_readers(
    out_root: Path, experiment: str, *stems: str, **overrides: object
) -> RunDir:
    run_dir = RunDir.create(_config(experiment, **overrides), out_root=out_root)
    _curves(run_dir, n_layers=12)
    for i, stem in enumerate(stems):
        _readers(run_dir, stem, seed=i)
    run_dir.finalize()
    return run_dir


def test_gather_readers_concatenates_the_sets_in_source_order(
    tmp_path: Path,
) -> None:
    _finished_with_readers(tmp_path, "r6", "readers_b", "readers_a")
    _finished_with_readers(tmp_path, "r1", "readers")
    got = gather_readers(tmp_path, ("r6", "r1"), MODEL)
    assert list(got) == ["r6:a", "r6:b", "r1"]
    assert got["r1"].layers == (1, 2, 3)


def test_gather_readers_reads_an_extra_run_instead_of_the_default(
    tmp_path: Path,
) -> None:
    _finished_with_readers(tmp_path, "r1", "readers")
    extra = _finished_with_readers(tmp_path, "r1", "readers", n_seeds=4)
    np.testing.assert_array_equal(
        gather_readers(tmp_path, ("r1",), MODEL, [str(extra.path)])[
            "r1"
        ].vectors,
        read_readers(extra)["r1"].vectors,
    )


def test_read_curves_returns_the_arrays_and_the_sidecar(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path, "r1")
    values = np.random.default_rng(1).uniform(size=(2, 3, 4))
    _curves(
        run_dir,
        "curves_x",
        layers=(2, 4, 6, 8),
        values=values,
        chance=0.5,
    )
    curves = read_curves(run_dir, "curves_x")
    np.testing.assert_array_equal(curves.x, [2.0, 4.0, 6.0, 8.0])
    np.testing.assert_allclose(curves.mean, values.mean(axis=1))
    np.testing.assert_array_equal(curves.per_seed, values)
    assert (curves.lo <= curves.mean).all()
    assert (curves.mean <= curves.hi).all()
    assert curves.series == ("reader", "random")
    assert curves.n_layers == 12
    assert curves.chance == 0.5
    _curves(run_dir, "curves_y", layers=(2, 4, 6, 8), n_layers=None)
    assert read_curves(run_dir, "curves_y").n_layers is None
    assert read_curves(run_dir, "curves_y").chance is None


def test_best_point_is_the_first_best_layer_with_its_seed_values(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path, "r1")
    values = np.zeros((2, 3, 4))
    values[0, :, 1] = [0.9, 0.7, 0.8]
    values[0, :, 2] = [0.8, 0.9, 0.7]
    values[1, :, 3] = [0.5, 0.6, 0.7]
    _curves(run_dir, layers=(2, 4, 6, 8), values=values)
    curves = read_curves(run_dir)
    layer, seeds = best_point(curves)
    assert layer == 4
    np.testing.assert_array_equal(seeds, [0.9, 0.7, 0.8])
    layer, seeds = best_point(curves, "random")
    assert layer == 8
    np.testing.assert_array_equal(seeds, [0.5, 0.6, 0.7])
    with pytest.raises(ValueError, match="shuffled"):
        best_point(curves, "shuffled")


def _two_curves(
    tmp_path: Path, *, select_layers: tuple[int, ...] = (2, 4, 6, 8)
) -> tuple[Curves, Curves]:
    run_dir = _run(tmp_path, "r1")
    reported = np.zeros((2, 3, 4))
    reported[0, :, 3] = [0.9, 0.8, 0.7]
    reported[0, :, 1] = [0.6, 0.5, 0.4]
    reported[1, :, 0] = [0.3, 0.2, 0.1]
    chosen = np.zeros((2, 3, 4))
    chosen[0, :, 1] = 0.8
    chosen[1, :, 2] = 0.7
    _curves(run_dir, "curves", layers=(2, 4, 6, 8), values=reported)
    _curves(run_dir, "curves_val", layers=select_layers, values=chosen)
    return read_curves(run_dir, "curves"), read_curves(run_dir, "curves_val")


def test_best_point_chooses_the_layer_on_the_select_curves(
    tmp_path: Path,
) -> None:
    curves, select = _two_curves(tmp_path)
    layer, seeds = best_point(curves, select=select)
    assert layer == 4
    np.testing.assert_array_equal(seeds, [0.6, 0.5, 0.4])
    assert best_point(curves)[0] == 8
    layer, seeds = best_point(curves, "random", select)
    assert layer == 6
    np.testing.assert_array_equal(seeds, [0.0, 0.0, 0.0])


def test_best_point_refuses_select_curves_at_other_layers(
    tmp_path: Path,
) -> None:
    curves, select = _two_curves(tmp_path, select_layers=(1, 2, 3, 4))
    with pytest.raises(ValueError, match="layers"):
        best_point(curves, select=select)


def test_best_point_refuses_select_curves_without_the_series(
    tmp_path: Path,
) -> None:
    curves, select = _two_curves(tmp_path)
    narrow = dataclasses.replace(select, series=("reader", "other"))
    with pytest.raises(ValueError, match="random"):
        best_point(curves, "random", narrow)


def _reading_run(
    tmp_path: Path,
    *,
    stems: tuple[str, ...] = ("curves", "curves_val"),
    n_layers: int | None = 12,
) -> RunDir:
    """A run whose reported curves peak at layer 8 and val curves at 4."""
    run_dir = _run(tmp_path, "r2")
    reported = np.zeros((2, 3, 4))
    reported[0, :, 3] = [0.9, 0.8, 0.7]
    reported[0, :, 1] = [0.6, 0.5, 0.4]
    reported[1, :, 2] = [0.3, 0.2, 0.1]
    chosen = np.zeros((2, 3, 4))
    chosen[0, :, 1] = 0.8
    chosen[1, :, 2] = 0.7
    for stem in stems:
        values = chosen if stem == "curves_val" else reported
        _curves(
            run_dir,
            stem,
            layers=(2, 4, 6, 8),
            values=values,
            n_layers=n_layers,
        )
    return run_dir


def test_reading_point_chooses_the_layer_on_the_validation_curves(
    tmp_path: Path,
) -> None:
    point = reading_point(_reading_run(tmp_path))
    assert point.layer == 4
    assert point.n_layers == 12
    assert point.selected_on == "curves_val"
    np.testing.assert_array_equal(point.values, [0.6, 0.5, 0.4])


def test_reading_point_chooses_on_the_reported_curves_without_validation(
    tmp_path: Path,
) -> None:
    point = reading_point(_reading_run(tmp_path, stems=("curves",)))
    assert point.layer == 8
    assert point.selected_on == "curves"
    np.testing.assert_array_equal(point.values, [0.9, 0.8, 0.7])


def test_reading_point_takes_the_series_and_the_stems_asked_for(
    tmp_path: Path,
) -> None:
    run_dir = _reading_run(tmp_path, stems=("curves", "curves_val", "curves_x"))
    point = reading_point(
        run_dir, "random", report="curves_x", select="curves_val"
    )
    assert point.layer == 6
    assert point.selected_on == "curves_val"
    np.testing.assert_array_equal(point.values, [0.3, 0.2, 0.1])
    other = reading_point(run_dir, report="curves_x", select="curves_none")
    assert other.selected_on == "curves_x"
    assert other.layer == 8


def test_reading_point_reads_the_depth_of_the_reported_curves(
    tmp_path: Path,
) -> None:
    run_dir = _reading_run(tmp_path, n_layers=None)
    with pytest.raises(ValueError, match="n_layers"):
        reading_point(run_dir)


def test_reading_point_refuses_an_unknown_series(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shuffled"):
        reading_point(_reading_run(tmp_path), "shuffled")


def test_relative_depth_is_the_layer_over_the_depth() -> None:
    np.testing.assert_allclose(relative_depth([0, 6, 12], 12), [0.0, 0.5, 1.0])


@pytest.mark.parametrize("layers", [[-1, 3], [3, 13]], ids=["below", "above"])
def test_relative_depth_refuses_a_layer_outside_the_model(
    layers: list[int],
) -> None:
    with pytest.raises(ValueError, match="layer"):
        relative_depth(layers, 12)


@pytest.mark.parametrize(
    ("layers", "depth", "n_layers", "expected"),
    [
        (range(1, 13), 0.5, 12, 6),
        ((2, 4, 6, 8, 10, 12), 0.5, 12, 6),
        ((1, 5, 12), 0.5, 12, 5),
        ((1, 2, 3), 0.5, 3, 1),
        ((3, 4), 0.14, 25, 3),
        ((1, 2, 3), 1.0, 3, 3),
        ((1, 2, 3), 0.01, 3, 1),
    ],
    ids=["all", "even", "uneven", "tie-lower", "float-tie", "last", "first"],
)
def test_depth_layer_is_the_layer_nearest_the_depth(
    layers: Sequence[int], depth: float, n_layers: int, expected: int
) -> None:
    assert depth_layer(layers, depth, n_layers) == expected


@pytest.mark.parametrize("depth", [0.0, -0.5, 1.5], ids=["zero", "neg", "big"])
def test_depth_layer_refuses_a_depth_outside_zero_to_one(depth: float) -> None:
    with pytest.raises(ValueError, match="depth"):
        depth_layer((1, 2, 3), depth, 3)


def test_depth_layer_needs_a_layer() -> None:
    with pytest.raises(ValueError, match="layers"):
        depth_layer((), 0.5, 3)


def test_label_stem_replaces_the_colon() -> None:
    assert label_stem("r6:truthfulqa") == "r6_truthfulqa"
    assert label_stem("r1") == "r1"
