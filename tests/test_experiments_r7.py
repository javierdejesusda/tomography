"""Tests for the R7 truthfulness reading experiment."""

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tomography.cli import load_experiment
from tomography.datasets.facts import load_facts
from tomography.datasets.geometry_of_truth import SUBSETS
from tomography.datasets.geometry_of_truth import load_geometry_of_truth
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.experiments import r7_truth
from tomography.experiments.base import RunDir
from tomography.experiments.r7_truth import Config
from tomography.experiments.r7_truth import execute
from tomography.experiments.r7_truth import plot
from tomography.experiments.r7_truth import run
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import SweepResult
from tomography.model import LoadedModel
from tomography.model import load_model

OOD = ("cities", "neg_cities")


def _facts(per_label: int = 14) -> list[Example]:
    true = ("rivers", "stars", "birds", "trees")
    false = ("clocks", "coins", "boats", "lamps")
    examples = []
    for split in ("train", "val", "test"):
        for label, words in ((1, true), (0, false)):
            for i in range(per_label):
                examples.append(
                    Example(
                        uid=f"facts-{split}-{label}-{i}",
                        text=f"The {words[i % 4]} and {words[(i + 1) % 4]}"
                        f" {split} {label} {i}",
                        label=label,
                        split=split,
                    )
                )
    return examples


def _truth_sets(
    subsets: tuple[str, ...] = OOD, per_label: int = 14
) -> list[Example]:
    examples = []
    for subset in subsets:
        for i in range(2 * per_label):
            label = i % 2
            examples.append(
                Example(
                    uid=f"got-{subset}-{i}",
                    text=f"{subset} town {i} has a {'big' if label else 'old'}"
                    f" wall {i}",
                    label=label,
                    split=("train", "val", "test")[i % 3],
                    meta={"subset": subset},
                )
            )
    return examples


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        n_seeds=3,
        n_pairs=8,
        n_eval_pairs=12,
        ood_subsets=OOD,
    )
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
    assert config.experiment == "r7"
    assert config.model == "gpt2"
    assert config.concept == "truthfulness"
    assert config.method == "logistic"
    assert config.ood_subsets == SUBSETS
    assert len(config.ood_subsets) == 9
    assert config.n_eval_pairs is None
    assert Config.from_dict(config.to_dict()) == config
    parsed = Config.from_dict({"ood_subsets": ["cities", "neg_cities"]})
    assert parsed.ood_subsets == ("cities", "neg_cities")


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("r7")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_execute_writes_curves_readers_and_summary(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _facts(), _truth_sets())
    n_layers = tiny_arch.n_layers
    layers = np.arange(1, n_layers + 1)
    names = ["val", "test", *OOD]
    stems = {
        name: "curves" if name == "test" else f"curves_{name}" for name in names
    }

    curves = {name: run_dir.read_arrays(stem) for name, stem in stems.items()}
    for name, stem in stems.items():
        assert curves[name]["x"].tolist() == layers.tolist()
        assert curves[name]["mean"].shape == (len(SERIES), n_layers)
        assert curves[name]["per_seed"].shape == (len(SERIES), 3, n_layers)
        sidecar = run_dir.read_json(stem)
        assert isinstance(sidecar, dict)
        assert sidecar["series"] == list(SERIES)
        assert sidecar["chance"] == 0.5
        assert sidecar["n_layers"] == n_layers

    readers = run_dir.read_arrays("readers")
    assert readers["layers"].dtype == np.int64
    assert readers["directions"].shape == (3, n_layers, tiny_arch.d_model)
    assert readers["signs"].shape == (3, n_layers)

    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert set(summary) == {"val_layer", "sets", "n_seeds", "n_pairs"}
    assert summary["n_seeds"] == 3
    assert summary["n_pairs"] == 8
    assert list(summary["sets"]) == names
    assert summary["val_layer"] == summary["sets"]["val"]["best_layer"]
    at_val = layers.tolist().index(summary["val_layer"])
    for name in names:
        entry = summary["sets"][name]
        assert set(entry) == {
            "best_layer",
            "best_accuracy",
            "ci",
            "accuracy_at_val_layer",
            "n_eval_pairs",
        }
        assert entry["best_layer"] in layers.tolist()
        at = layers.tolist().index(entry["best_layer"])
        mean = curves[name]["mean"][0]
        assert entry["best_accuracy"] == pytest.approx(mean[at])
        assert entry["accuracy_at_val_layer"] == pytest.approx(mean[at_val])
        lo, hi = entry["ci"]
        assert lo == pytest.approx(curves[name]["lo"][0, at])
        assert hi == pytest.approx(curves[name]["hi"][0, at])
        assert lo <= entry["best_accuracy"] <= hi
        assert entry["n_eval_pairs"] == 12


def test_each_set_reports_its_best_layer_and_the_val_layer_accuracy(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planted = _planted_sweep(
        tiny_arch,
        {
            "val": (0, 0.9),
            "test": (2, 0.8),
            "cities": (1, 0.7),
            "neg_cities": (2, 0.6),
        },
    )
    monkeypatch.setattr(r7_truth, "reading_sweep", lambda *a, **k: planted)
    config = _config(n_seeds=2)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _facts(), _truth_sets())
    assert run_dir.read_arrays("curves")["mean"][0].argmax() == 2
    assert run_dir.read_arrays("curves_cities")["mean"][0].argmax() == 1
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["val_layer"] == 1
    expected = {
        "val": (1, 0.9, 0.9),
        "test": (3, 0.8, 0.5),
        "cities": (2, 0.7, 0.5),
        "neg_cities": (3, 0.6, 0.5),
    }
    for name, (layer, best, at_val) in expected.items():
        entry = summary["sets"][name]
        assert entry["best_layer"] == layer
        assert entry["best_accuracy"] == pytest.approx(best)
        assert entry["accuracy_at_val_layer"] == pytest.approx(at_val)


def test_the_readers_are_fitted_once_and_scored_on_every_set(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[StimulusSet], list[str]]] = []
    original = r7_truth.reading_sweep

    def recording(
        loaded: LoadedModel,
        train_sets: list[StimulusSet],
        eval_sets: dict[str, StimulusSet],
        **kwargs: Any,
    ) -> SweepResult:
        calls.append((list(train_sets), list(eval_sets)))
        return original(loaded, train_sets, eval_sets, **kwargs)

    monkeypatch.setattr(r7_truth, "reading_sweep", recording)
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _facts(), _truth_sets())
    assert len(calls) == 1
    train_sets, names = calls[0]
    assert len(train_sets) == config.n_seeds
    assert names == ["val", "test", "cities", "neg_cities"]


def _record_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> list[StimulusSet]:
    built: list[StimulusSet] = []

    def recording(*args: Any, **kwargs: Any) -> StimulusSet:
        stimuli = build_stimulus_set(*args, **kwargs)
        built.append(stimuli)
        return stimuli

    monkeypatch.setattr(r7_truth, "build_stimulus_set", recording)
    return built


def test_sets_hold_the_right_examples_and_the_concept_word(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = _record_builds(monkeypatch)
    config = _config(n_seeds=2)
    execute(
        config,
        RunDir.create(config, out_root=tmp_path),
        tiny_arch,
        _facts(),
        _truth_sets(),
    )
    assert len(built) == 2 + 4
    train, val, test, cities, neg_cities = built[:2], *built[2:]
    for stimuli in train:
        assert all(u.startswith("facts-train-") for u in stimuli.uids)
    assert all(u.startswith("facts-val-") for u in val.uids)
    assert all(u.startswith("facts-test-") for u in test.uids)
    assert all(u.startswith("got-cities-") for u in cities.uids)
    assert all(u.startswith("got-neg_cities-") for u in neg_cities.uids)
    splits = {
        e.split
        for e in _truth_sets()
        if e.uid in set(cities.uids) | set(neg_cities.uids)
    }
    assert len(splits) > 1
    for stimuli in built:
        assert all("truthfulness" in text for text in stimuli.texts)


def test_the_concept_word_follows_the_config(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = _record_builds(monkeypatch)
    config = _config(n_seeds=2, concept="honesty")
    execute(
        config,
        RunDir.create(config, out_root=tmp_path),
        tiny_arch,
        _facts(),
        _truth_sets(),
    )
    for stimuli in built:
        assert all("honesty" in text for text in stimuli.texts)
        assert not any("truthfulness" in text for text in stimuli.texts)


def test_a_subset_without_examples_is_an_error(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(ood_subsets=("cities", "larger_than"))
    run_dir = RunDir.create(config, out_root=tmp_path)
    with pytest.raises(ValueError, match="larger_than"):
        execute(config, run_dir, tiny_arch, _facts(), _truth_sets())


def test_execute_is_deterministic(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(n_seeds=2)
    first = RunDir.create(config, out_root=tmp_path / "a")
    second = RunDir.create(config, out_root=tmp_path / "b")
    execute(config, first, tiny_arch, _facts(), _truth_sets())
    execute(config, second, tiny_arch, _facts(), _truth_sets())
    for stem in ("curves", "curves_cities", "readers"):
        a, b = first.read_arrays(stem), second.read_arrays(stem)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_plot_writes_three_formats_per_set(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(n_seeds=2)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _facts(), _truth_sets())
    paths = plot(run_dir, tmp_path / "figs")
    stems = ("curves", "curves_cities", "curves_neg_cities", "curves_val")
    assert sorted(p.name for p in paths) == [
        f"{stem}.{ext}" for stem in stems for ext in ("pdf", "png", "svg")
    ]
    assert all(p.stat().st_size > 0 for p in paths)


def test_run_loads_the_model_and_the_data(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []
    seen: list[tuple[str, ...]] = []

    def fake_load_model(key: str) -> LoadedModel:
        asked.append(key)
        return tiny_arch

    def fake_load_truth(*, subsets: tuple[str, ...]) -> list[Example]:
        seen.append(subsets)
        return _truth_sets()

    monkeypatch.setattr(r7_truth, "load_model", fake_load_model)
    monkeypatch.setattr(r7_truth, "load_facts", _facts)
    monkeypatch.setattr(r7_truth, "load_geometry_of_truth", fake_load_truth)
    config = _config(n_seeds=2)
    run_dir = RunDir.create(config, out_root=tmp_path)
    run(config, run_dir)
    assert asked == ["tiny"]
    assert seen == [config.ood_subsets]
    assert (run_dir.path / "summary.json").is_file()
    assert run_dir.meta["finished_at"] is None


@pytest.mark.slow
def test_r7_on_gpt2_reads_truthfulness_above_the_floor(tmp_path: Path) -> None:
    config = Config(
        n_seeds=3,
        n_pairs=128,
        ood_subsets=("cities", "neg_cities", "sp_en_trans"),
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(
        config,
        run_dir,
        load_model("gpt2"),
        load_facts(offline=True),
        load_geometry_of_truth(subsets=config.ood_subsets, offline=True),
    )
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    for name, entry in summary["sets"].items():
        print(
            name,
            entry["best_layer"],
            round(entry["best_accuracy"], 3),
            round(entry["accuracy_at_val_layer"], 3),
        )
    assert summary["sets"]["test"]["accuracy_at_val_layer"] > 0.55
