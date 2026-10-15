"""Tests for the A1 direction geometry analysis."""

from collections.abc import Mapping
import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.analysis import agreement_pseudo_values
from tomography.analysis import layer_cosines
from tomography.analysis import mean_direction
from tomography.analysis import seed_agreement
from tomography.cli import load_experiment
from tomography.experiments import matrices
from tomography.experiments.a1_geometry import Config
from tomography.experiments.a1_geometry import execute
from tomography.experiments.a1_geometry import plot
from tomography.experiments.a1_geometry import run
from tomography.experiments.aggregate import ReaderSet
from tomography.experiments.base import RunDir
from tomography.experiments.sweep import write_curves
from tomography.reading.metrics import bootstrap_ci
from tomography.reading.metrics import cosine_matrix

MODEL = "tiny"
D = 64
LAYERS = (1, 2, 3)
N_LAYERS = 4
SOURCES = ("r1", "r2", "r3", "r7", "r6")


def _unit(seed: int) -> NDArray[np.float64]:
    vector = np.random.default_rng(seed).normal(size=D)
    return vector / np.linalg.norm(vector)


def _directions(
    n: int, base: NDArray[np.float64] | None, seed: int
) -> NDArray[np.float64]:
    """Seeds' directions: a planted base plus small noise, or pure noise."""
    noise = np.random.default_rng(seed).normal(size=(n, len(LAYERS), D))
    return noise if base is None else base + 0.02 * noise


def _sets() -> dict[str, ReaderSet]:
    shared = _unit(100)
    planted = {
        "r1": _directions(4, shared, 1),
        "r2": _directions(4, shared, 2),
        "r3": _directions(4, None, 3),
        "r7": _directions(4, _unit(101), 4),
        "r6:arc": _directions(6, _unit(102), 5),
        "r6:truthfulqa": _directions(6, _unit(103), 6),
    }
    return {
        label: ReaderSet(
            LAYERS,
            vectors / np.linalg.norm(vectors, axis=-1, keepdims=True),
            N_LAYERS,
        )
        for label, vectors in planted.items()
    }


def _config(**overrides: Any) -> Config:
    base = Config(model=MODEL, sources=SOURCES, n_boot=20)
    return dataclasses.replace(base, **overrides)


def _run_dir(tmp_path: Path, config: Config | None = None) -> RunDir:
    return RunDir.create(config or _config(), out_root=tmp_path / "res")


def _matrix(run_dir: RunDir, stem: str) -> NDArray[np.float64]:
    return run_dir.read_arrays(stem)["values"]


def _summary(run_dir: RunDir) -> dict[str, Any]:
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    return summary


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "a1"
    assert config.model == "gpt2"
    assert config.sources == ("r1", "r2", "r3", "r5", "r7")
    assert config.extra_runs == ()
    assert config.depth == 0.5
    assert config.n_boot == 2000
    assert Config.from_dict(config.to_dict()) == config


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("a1")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_execute_recovers_the_planted_structure(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    readers = _sets()
    execute(_config(), run_dir, readers)

    concepts = _matrix(run_dir, "matrix_concepts")
    labels = list(readers)
    assert concepts.shape == (6, 6)
    np.testing.assert_allclose(np.diag(concepts), 1.0, atol=1e-12)
    assert concepts[labels.index("r1"), labels.index("r2")] > 0.95
    assert abs(concepts[labels.index("r1"), labels.index("r3")]) < 0.7
    assert abs(concepts[labels.index("r1"), labels.index("r7")]) < 0.7

    layers = _matrix(run_dir, "matrix_layers_r1")
    assert layers.shape == (3, 3)
    np.testing.assert_allclose(np.diag(layers), 1.0, atol=1e-12)
    assert layers.min() > 0.9
    assert _matrix(run_dir, "matrix_layers_r3").min() < 0.7

    summary = _summary(run_dir)
    assert summary["agreement_at_layer"]["r1"] > 0.9
    assert abs(summary["agreement_at_layer"]["r3"]) < 0.7


def test_execute_labels_the_matrices_and_names_the_stems(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    readers = _sets()
    execute(_config(), run_dir, readers)
    assert run_dir.read_json("matrix_concepts") == {
        "rows": list(readers),
        "cols": list(readers),
        "title": "Cosine between concepts at layer 2",
    }
    sidecar = run_dir.read_json("matrix_layers_r6_truthfulqa")
    assert isinstance(sidecar, dict)
    assert sidecar["rows"] == sidecar["cols"] == ["1", "2", "3"]
    assert "r6:truthfulqa" in sidecar["title"]
    assert not (run_dir.path / "matrix_layers_r6:truthfulqa.npz").exists()


def test_execute_uses_the_first_seeds_of_the_longer_sets(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    readers = _sets()
    execute(_config(), run_dir, readers)
    longer = readers["r6:arc"].vectors
    assert longer.shape[0] == 6
    np.testing.assert_allclose(
        _matrix(run_dir, "matrix_layers_r6_arc"), layer_cosines(longer[:4])
    )
    curves = run_dir.read_arrays("curves_agreement")
    assert curves["per_seed"].shape == (6, 4, 3)
    np.testing.assert_allclose(
        curves["per_seed"][list(readers).index("r6:arc")],
        seed_agreement(longer[:4]),
    )
    assert _summary(run_dir)["n_seeds"] == 4


def test_execute_writes_the_agreement_curves(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path, _config(seed=3, n_boot=10))
    readers = _sets()
    execute(_config(seed=3, n_boot=10), run_dir, readers)
    curves = run_dir.read_arrays("curves_agreement")
    assert curves["x"].tolist() == [1.0, 2.0, 3.0]
    np.testing.assert_allclose(curves["mean"], curves["per_seed"].mean(axis=1))
    pseudo = np.stack(
        [agreement_pseudo_values(r.vectors[:4]) for r in readers.values()]
    )
    lo, hi = bootstrap_ci(np.moveaxis(pseudo, 1, 0), n_boot=10, seed=3)
    np.testing.assert_array_equal(curves["lo"], lo)
    np.testing.assert_array_equal(curves["hi"], hi)
    assert run_dir.read_json("curves_agreement") == {
        "series": list(readers),
        "xlabel": "Layer",
        "ylabel": "Cosine to the other seeds",
        "chance": 0.0,
        "n_layers": N_LAYERS,
    }


def test_execute_reads_the_concept_matrix_at_the_depth_layer(
    tmp_path: Path,
) -> None:
    readers = {
        label: ReaderSet((2, 4, 6), r.vectors, 6)
        for label, r in _sets().items()
    }
    for depth, layer, position in ((0.5, 2, 0), (0.75, 4, 1), (1.0, 6, 2)):
        run_dir = _run_dir(tmp_path / f"d{position}", _config(depth=depth))
        execute(_config(depth=depth), run_dir, readers)
        means = np.stack(
            [mean_direction(r.vectors[:4])[position] for r in readers.values()]
        )
        np.testing.assert_allclose(
            _matrix(run_dir, "matrix_concepts"), cosine_matrix(means)
        )
        summary = _summary(run_dir)
        assert summary["layer"] == layer
        assert summary["depth"] == depth


def test_the_summary_holds_the_reference_and_the_agreement(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    readers = _sets()
    execute(_config(), run_dir, readers)
    summary = _summary(run_dir)
    assert set(summary) == {
        "concepts",
        "layer",
        "depth",
        "n_seeds",
        "chance_abs_cosine",
        "agreement_at_layer",
    }
    assert summary["concepts"] == list(readers)
    assert summary["layer"] == 2
    assert summary["chance_abs_cosine"] == pytest.approx(
        math.sqrt(2.0 / (math.pi * D))
    )
    for label, r in readers.items():
        expected = seed_agreement(r.vectors[:4])[:, 1].mean()
        assert summary["agreement_at_layer"][label] == pytest.approx(expected)


def _single(seeds: int) -> Mapping[str, ReaderSet]:
    return {
        "r1": _sets()["r1"],
        "r2": ReaderSet(LAYERS, _sets()["r2"].vectors[:seeds], N_LAYERS),
    }


@pytest.mark.parametrize(
    ("readers", "config", "match"),
    [
        (_single(1), _config(), "at least 3 seeds"),
        (_single(2), _config(), "at least 3 seeds"),
        ({}, _config(), "no reader"),
        (
            {f"c{i}": _sets()["r1"] for i in range(9)},
            _config(),
            "at most 8",
        ),
        (
            {
                "a": _sets()["r1"],
                "b": ReaderSet((1, 2, 4), _sets()["r2"].vectors, N_LAYERS),
            },
            _config(),
            "layers",
        ),
        ({"a": _sets()["r1"]}, _config(depth=0.0), "depth"),
        ({"a": _sets()["r1"]}, _config(depth=1.5), "depth"),
        ({"a": _sets()["r1"]}, _config(n_boot=0), "n_boot"),
        ({"a": _sets()["r1"]}, _config(seed=-1), "seed"),
    ],
    ids=[
        "one-seed",
        "two-seeds",
        "none",
        "nine",
        "layers",
        "depth-zero",
        "depth-above",
        "n-boot",
        "seed",
    ],
)
def test_execute_refuses_unusable_input_before_writing(
    tmp_path: Path,
    readers: Mapping[str, ReaderSet],
    config: Config,
    match: str,
) -> None:
    run_dir = _run_dir(tmp_path, config)
    with pytest.raises(ValueError, match=match):
        execute(config, run_dir, readers)
    assert not list(run_dir.path.glob("*.npz"))
    assert not (run_dir.path / "summary.json").exists()


def _write_source(
    out_root: Path,
    experiment: str,
    sets: Mapping[str, ReaderSet],
    flip_seed: int | None = None,
    **overrides: Any,
) -> RunDir:
    """A finished default run holding the given sets as readers files.

    With `flip_seed` every direction is stored negated or not at random, with
    the sign that restores it, so the oriented vectors do not change.
    """
    config = load_experiment(experiment).config_cls.from_dict(
        {"experiment": experiment, "model": MODEL}
    )
    run_dir = RunDir.create(
        dataclasses.replace(config, **overrides), out_root=out_root
    )
    n_seeds = min(r.vectors.shape[0] for r in sets.values())
    write_curves(
        run_dir,
        np.zeros((1, n_seeds, len(LAYERS))),
        LAYERS,
        series=("reader",),
        n_layers=N_LAYERS,
        n_boot=10,
    )
    for label, readers in sets.items():
        suffix = label.partition(":")[2]
        directions = readers.vectors.astype(np.float32)
        signs = np.ones(directions.shape[:2], dtype=np.float32)
        if flip_seed is not None:
            signs = (
                np.random.default_rng(flip_seed)
                .choice([-1.0, 1.0], size=signs.shape)
                .astype(np.float32)
            )
            directions = directions * signs[..., None]
        run_dir.write_arrays(
            f"readers_{suffix}" if suffix else "readers",
            layers=np.asarray(readers.layers, dtype=np.int64),
            directions=directions,
            signs=signs,
        )
    run_dir.finalize()
    return run_dir


def _write_all_sources(out_root: Path, flip_seed: int | None = None) -> None:
    sets = _sets()
    for experiment in ("r1", "r2", "r3", "r7"):
        _write_source(
            out_root, experiment, {experiment: sets[experiment]}, flip_seed
        )
    _write_source(
        out_root,
        "r6",
        {label: r for label, r in sets.items() if label.startswith("r6")},
        flip_seed,
    )


def test_run_reads_the_default_runs_under_its_own_out_root(
    tmp_path: Path,
) -> None:
    _write_all_sources(tmp_path / "res")
    config = _config()
    run_dir = _run_dir(tmp_path, config)
    run(config, run_dir)
    summary = _summary(run_dir)
    assert summary["concepts"] == list(_sets())
    assert summary["n_seeds"] == 4
    assert (run_dir.path / "matrix_concepts.npz").is_file()


def test_negated_directions_with_negative_signs_change_nothing(
    tmp_path: Path,
) -> None:
    config = _config()
    runs = {}
    for name, flip_seed in (("plain", None), ("flipped", 5)):
        _write_all_sources(tmp_path / name / "res", flip_seed)
        runs[name] = _run_dir(tmp_path / name, config)
        run(config, runs[name])
    stems = sorted(p.stem for p in runs["plain"].path.glob("*.npz"))
    assert "curves_agreement" in stems
    assert len(stems) == 8
    for stem in stems:
        plain = runs["plain"].read_arrays(stem)
        flipped = runs["flipped"].read_arrays(stem)
        assert set(plain) == set(flipped)
        for key in plain:
            np.testing.assert_allclose(
                flipped[key], plain[key], atol=1e-12, err_msg=f"{stem}:{key}"
            )
    assert _summary(runs["flipped"]) == _summary(runs["plain"])


def test_run_names_the_source_run_that_is_missing(tmp_path: Path) -> None:
    config = _config(sources=("r1", "r2"))
    _write_source(tmp_path / "res", "r1", {"r1": _sets()["r1"]})
    with pytest.raises(ValueError, match="no run of r2"):
        run(config, _run_dir(tmp_path, config))


def test_run_reads_an_extra_run_instead_of_the_default(
    tmp_path: Path,
) -> None:
    sets = _sets()
    other = {"r1": sets["r2"]}
    _write_source(tmp_path / "res", "r1", {"r1": sets["r1"]})
    extra = _write_source(tmp_path / "other", "r1", other, n_seeds=4)
    config = _config(sources=("r1",), extra_runs=(str(extra.path),))
    run_dir = _run_dir(tmp_path, config)
    run(config, run_dir)
    np.testing.assert_allclose(
        _matrix(run_dir, "matrix_layers_r1"),
        layer_cosines(sets["r2"].vectors),
        atol=1e-6,
    )


def test_plot_draws_the_curves_and_every_matrix(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    execute(_config(), run_dir, _sets())
    paths = plot(run_dir, tmp_path / "figs")
    assert len(paths) == 3 * (1 + 6 + 1)
    names = {p.name for p in paths}
    assert {
        "curves_agreement.pdf",
        "matrix_concepts.png",
        "matrix_layers_r6_truthfulqa.svg",
    } <= names
    assert all(p.stat().st_size > 0 for p in paths)


def test_plot_draws_the_matrices_on_the_cosine_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _run_dir(tmp_path)
    execute(_config(), run_dir, _sets())
    options: list[dict[str, Any]] = []
    real = matrices.plot_matrix

    def spy(values: NDArray[np.float64], **kwargs: Any) -> Any:
        options.append(kwargs)
        return real(values, **kwargs)

    monkeypatch.setattr(matrices, "plot_matrix", spy)
    plot(run_dir, tmp_path / "figs")
    assert len(options) == 7
    assert {(o["kind"], o["vmin"], o["vmax"]) for o in options} == {
        ("diverging", -1.0, 1.0)
    }
