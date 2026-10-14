"""Tests for the R3 morality reading experiment."""

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tomography.cli import load_experiment
from tomography.datasets.ethics import load_ethics_morality
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.experiments import r3_morality
from tomography.experiments.base import RunDir
from tomography.experiments.r3_morality import Config
from tomography.experiments.r3_morality import execute
from tomography.experiments.r3_morality import plot
from tomography.experiments.r3_morality import run
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import SweepResult
from tomography.model import LoadedModel
from tomography.model import load_model


def _examples(per_cell: int = 10) -> list[Example]:
    good = ("kind", "fair", "calm", "warm")
    bad = ("cruel", "rude", "mean", "cold")
    examples = []
    for split in ("train", "val", "test"):
        for label, words in ((1, good), (0, bad)):
            for short in (0, 1):
                for i in range(per_cell):
                    examples.append(
                        Example(
                            uid=f"{split}-{label}-{short}-{i}",
                            text=f"I was {words[i % 4]} to {words[(i + 1) % 4]}"
                            f" {split} {label} {short} {i}",
                            label=label,
                            split=split,
                            meta={"is_short": short},
                        )
                    )
    return examples


def _config(**overrides: Any) -> Config:
    base = Config(model="tiny", n_seeds=3, n_pairs=8, n_eval_pairs=12)
    return dataclasses.replace(base, **overrides)


def _planted_sweep(
    loaded: LoadedModel, peaks: dict[str, tuple[int, float]]
) -> SweepResult:
    """Reader accuracy of 0.5 everywhere except one planted peak per set."""
    layers = tuple(range(1, loaded.n_layers + 1))
    accuracy = np.full((len(peaks), len(SERIES), 2, len(layers)), 0.5)
    for e, (at, value) in enumerate(peaks.values()):
        accuracy[e, 0, :, at] = value
    reader = SimpleNamespace(
        directions=np.zeros((len(layers), loaded.d_model), dtype=np.float32),
        signs=np.ones(len(layers), dtype=np.float32),
    )
    return SweepResult(layers, tuple(peaks), accuracy, (reader, reader))


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "r3"
    assert config.model == "gpt2"
    assert config.match_length is True
    assert config.n_seeds == 10
    assert config.n_pairs == 64
    assert config.n_eval_pairs is None
    assert Config.from_dict(config.to_dict()) == config
    assert Config.from_dict({"match_length": False}).match_length is False


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("r3")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_execute_writes_curves_readers_and_summary(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _examples())
    n_layers = tiny_arch.n_layers
    layers = np.arange(1, n_layers + 1)

    for stem in ("curves", "curves_val"):
        curves = run_dir.read_arrays(stem)
        assert curves["x"].tolist() == layers.tolist()
        assert curves["mean"].shape == (len(SERIES), n_layers)
        assert curves["per_seed"].shape == (len(SERIES), 3, n_layers)
        sidecar = run_dir.read_json(stem)
        assert isinstance(sidecar, dict)
        assert sidecar["series"] == list(SERIES)
        assert sidecar["chance"] == 0.5
        assert sidecar["n_layers"] == n_layers

    readers = run_dir.read_arrays("readers")
    assert readers["layers"].dtype == np.int64
    assert readers["layers"].tolist() == layers.tolist()
    assert readers["directions"].shape == (3, n_layers, tiny_arch.d_model)
    assert readers["directions"].dtype == np.float32
    assert readers["signs"].shape == (3, n_layers)
    assert readers["signs"].dtype == np.float32

    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert set(summary) == {
        "best_layer",
        "val_accuracy",
        "test_accuracy",
        "ci",
        "n_seeds",
        "n_pairs",
        "n_eval_pairs",
    }
    assert summary["best_layer"] in layers.tolist()
    assert summary["n_seeds"] == 3
    assert summary["n_pairs"] == 8
    assert summary["n_eval_pairs"] == {"val": 12, "test": 12}
    at = layers.tolist().index(summary["best_layer"])
    test = run_dir.read_arrays("curves")
    val = run_dir.read_arrays("curves_val")
    assert summary["val_accuracy"] == pytest.approx(val["mean"][0, at])
    assert summary["test_accuracy"] == pytest.approx(test["mean"][0, at])
    lo, hi = summary["ci"]
    assert lo == pytest.approx(test["lo"][0, at])
    assert hi == pytest.approx(test["hi"][0, at])
    assert lo <= summary["test_accuracy"] <= hi


def test_the_layer_is_chosen_on_val_and_reported_on_test(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planted = _planted_sweep(tiny_arch, {"val": (0, 0.9), "test": (2, 0.8)})
    monkeypatch.setattr(r3_morality, "reading_sweep", lambda *a, **k: planted)
    config = _config(n_seeds=2)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _examples())
    assert run_dir.read_arrays("curves")["mean"][0].argmax() == 2
    assert run_dir.read_arrays("curves_val")["mean"][0].argmax() == 0
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["best_layer"] == 1
    assert summary["val_accuracy"] == pytest.approx(0.9)
    assert summary["test_accuracy"] == pytest.approx(0.5)
    assert summary["ci"] == pytest.approx([0.5, 0.5])


def test_sets_come_from_the_right_split_and_length_class(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[StimulusSet] = []

    def recording(*args: Any, **kwargs: Any) -> StimulusSet:
        stimuli = build_stimulus_set(*args, **kwargs)
        built.append(stimuli)
        return stimuli

    monkeypatch.setattr(r3_morality, "build_stimulus_set", recording)
    examples = _examples()
    by_uid = {e.uid: e for e in examples}
    config = _config()
    execute(
        config, RunDir.create(config, out_root=tmp_path), tiny_arch, examples
    )

    assert len(built) == config.n_seeds + 2
    expected = ["train"] * config.n_seeds + ["val", "test"]
    for split, stimuli in zip(expected, built, strict=True):
        assert {by_uid[u].split for u in stimuli.uids} == {split}
        for first, second in stimuli.pairs.tolist():
            a, b = by_uid[stimuli.uids[first]], by_uid[stimuli.uids[second]]
            assert a.meta["is_short"] == b.meta["is_short"]
            assert a.label != b.label


def test_match_length_toggle_matters_when_length_tracks_the_label(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    aligned = [
        dataclasses.replace(e, meta={"is_short": e.label}) for e in _examples()
    ]
    strict = _config(match_length=True)
    with pytest.raises(ValueError, match="no pair can be formed"):
        execute(
            strict,
            RunDir.create(strict, out_root=tmp_path / "strict"),
            tiny_arch,
            aligned,
        )
    loose = _config(match_length=False)
    run_dir = RunDir.create(loose, out_root=tmp_path / "loose")
    execute(loose, run_dir, tiny_arch, aligned)
    assert (run_dir.path / "summary.json").is_file()


def test_execute_is_deterministic(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(n_seeds=2)
    first = RunDir.create(config, out_root=tmp_path / "a")
    second = RunDir.create(config, out_root=tmp_path / "b")
    execute(config, first, tiny_arch, _examples())
    execute(config, second, tiny_arch, _examples())
    for stem in ("curves", "curves_val", "readers"):
        a, b = first.read_arrays(stem), second.read_arrays(stem)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_plot_writes_three_formats_per_split(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(n_seeds=2)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _examples())
    paths = plot(run_dir, tmp_path / "figs")
    assert sorted(p.name for p in paths) == [
        f"{stem}.{ext}"
        for stem in ("curves", "curves_val")
        for ext in ("pdf", "png", "svg")
    ]
    assert all(p.stat().st_size > 0 for p in paths)


def test_run_loads_the_model_and_the_data(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []

    def fake_load_model(key: str) -> LoadedModel:
        asked.append(key)
        return tiny_arch

    monkeypatch.setattr(r3_morality, "load_model", fake_load_model)
    monkeypatch.setattr(r3_morality, "load_ethics_morality", _examples)
    config = _config(n_seeds=2)
    run_dir = RunDir.create(config, out_root=tmp_path)
    run(config, run_dir)
    assert asked == ["tiny"]
    assert (run_dir.path / "summary.json").is_file()
    assert "finished_at" in run_dir.meta
    assert run_dir.meta["finished_at"] is None


@pytest.mark.slow
def test_r3_on_gpt2_reads_morality_above_the_floor(tmp_path: Path) -> None:
    config = Config(n_seeds=3, n_pairs=32, n_eval_pairs=128)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(
        config,
        run_dir,
        load_model("gpt2"),
        load_ethics_morality(offline=True),
    )
    curves = run_dir.read_arrays("curves")
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    for name, row in zip(SERIES, curves["mean"], strict=True):
        print(name, np.round(row, 3).tolist())
    print(summary)
    assert summary["test_accuracy"] > 0.55
