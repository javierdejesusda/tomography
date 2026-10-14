"""Tests for the R1 sentiment reading experiment."""

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tomography.activations import Activations
from tomography.cli import load_experiment
from tomography.datasets.records import Example
from tomography.datasets.sst2 import load_sst2
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import r1_sentiment
from tomography.experiments.base import RunDir
from tomography.experiments.r1_sentiment import Config
from tomography.experiments.r1_sentiment import execute
from tomography.experiments.r1_sentiment import plot
from tomography.experiments.r1_sentiment import run
from tomography.experiments.r1_sentiment import select_examples
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import Prepared
from tomography.experiments.sweep import sweep_from_activations
from tomography.model import LoadedModel
from tomography.model import load_model

DIM = 64
LAYERS = (1, 2)


def _examples(per_class: int = 40) -> list[Example]:
    positive = ("good", "fine", "great", "nice")
    negative = ("bad", "poor", "dull", "grim")
    examples = []
    for split in ("train", "val"):
        for i in range(per_class):
            for label, words in ((1, positive), (0, negative)):
                examples.append(
                    Example(
                        uid=f"{split}-{label}-{i}",
                        text=f"a {words[i % 4]} movie {words[(i + 1) % 4]} {i}",
                        label=label,
                        split="train" if split == "train" else "val",
                    )
                )
    return examples


def _config(**overrides: Any) -> Config:
    base = Config(model="tiny", n_seeds=3, n_pairs=8, n_eval_pairs=12)
    return dataclasses.replace(base, **overrides)


def test_select_examples_filters_split_and_length_keeping_order() -> None:
    examples = [
        Example(uid="a", text="one two three", label=1, split="train"),
        Example(uid="b", text="one two", label=0, split="train"),
        Example(uid="c", text="one two three four", label=0, split="val"),
        Example(uid="d", text="five six seven", label=0, split="train"),
    ]
    got = select_examples(examples, split="train", min_words=3)
    assert [e.uid for e in got] == ["a", "d"]
    assert [e.uid for e in select_examples(examples, split="train")] == [
        "a",
        "b",
        "d",
    ]
    assert select_examples(examples, split="test") == []


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "r1"
    assert config.model == "gpt2"
    assert config.n_eval_pairs == 256
    assert config.min_words == 3
    assert config.method == "logistic"
    assert config.n_seeds == 10
    assert config.n_pairs == 64
    assert Config.from_dict(config.to_dict()) == config


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("r1")
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

    curves = run_dir.read_arrays("curves")
    assert curves["x"].tolist() == layers.tolist()
    assert curves["mean"].shape == (len(SERIES), n_layers)
    assert curves["per_seed"].shape == (len(SERIES), 3, n_layers)
    sidecar = run_dir.read_json("curves")
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
        "best_accuracy",
        "ci",
        "n_seeds",
        "n_pairs",
        "n_eval_pairs",
    }
    assert summary["best_layer"] in layers.tolist()
    assert summary["n_seeds"] == 3
    assert summary["n_pairs"] == 8
    assert summary["n_eval_pairs"] == 12
    lo, hi = summary["ci"]
    best = layers.tolist().index(summary["best_layer"])
    assert summary["best_accuracy"] == pytest.approx(curves["mean"][0, best])
    assert lo == pytest.approx(curves["lo"][0, best])
    assert hi == pytest.approx(curves["hi"][0, best])
    assert lo <= summary["best_accuracy"] <= hi


def test_execute_reports_the_actual_number_of_eval_pairs(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(n_eval_pairs=None, n_seeds=2)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _examples(per_class=20))
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["n_eval_pairs"] == 20


def test_execute_is_deterministic(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(n_seeds=2)
    first = RunDir.create(config, out_root=tmp_path / "a")
    second = RunDir.create(config, out_root=tmp_path / "b")
    execute(config, first, tiny_arch, _examples())
    execute(config, second, tiny_arch, _examples())
    for stem in ("curves", "readers"):
        a, b = first.read_arrays(stem), second.read_arrays(stem)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_plot_writes_three_formats(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(n_seeds=2)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _examples())
    paths = plot(run_dir, tmp_path / "figs")
    assert sorted(p.name for p in paths) == [
        "curves.pdf",
        "curves.png",
        "curves.svg",
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

    monkeypatch.setattr(r1_sentiment, "load_model", fake_load_model)
    monkeypatch.setattr(r1_sentiment, "load_sst2", _examples)
    config = _config(n_seeds=2)
    run_dir = RunDir.create(config, out_root=tmp_path)
    run(config, run_dir)
    assert asked == ["tiny"]
    assert (run_dir.path / "summary.json").is_file()
    assert "finished_at" in run_dir.meta
    assert run_dir.meta["finished_at"] is None


def _unmatched_pairs(n_pairs: int, *, seed: int) -> Prepared:
    """Pairs of unrelated texts: loud content, quiet sentiment.

    Every row is independent content noise, large along one nuisance axis,
    plus a small offset along the sentiment axis signed by its label. As in
    the label-contrastive pairing, the two members of a pair share nothing
    but their opposite labels.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_pairs
    first_is_positive = rng.integers(0, 2, size=n_pairs)
    labels = np.zeros(n, dtype=np.int64)
    labels[2 * np.arange(n_pairs)] = first_is_positive
    labels[2 * np.arange(n_pairs) + 1] = 1 - first_is_positive
    axes = np.linalg.qr(np.random.default_rng(7).normal(size=(DIM, 2)))[0]
    sentiment, nuisance = axes[:, 0], axes[:, 1]
    data = rng.normal(size=(len(LAYERS), n, DIM))
    data += 10.0 * rng.normal(size=(len(LAYERS), n, 1)) * nuisance
    data += (2 * labels - 1)[None, :, None] * 2.0 * sentiment
    pairs = np.arange(n, dtype=np.int64).reshape(n_pairs, 2)
    stimuli = StimulusSet(
        texts=tuple(f"text {i}" for i in range(n)),
        response_spans=(None,) * n,
        concept_spans=(None,) * n,
        labels=labels,
        pairs=pairs,
        groups=pairs.copy(),
        uids=tuple(f"u{i // 2}" for i in range(n)),
    )
    act = Activations(
        data=data.astype(np.float32),
        layers=LAYERS,
        position="last",
        row_stimulus=np.arange(n, dtype=np.int64),
        row_rank=np.zeros(n, dtype=np.int64),
        row_token=np.zeros(n, dtype=np.int64),
        row_token_id=np.zeros(n, dtype=np.int64),
        n_stimuli=n,
    )
    return stimuli, act


def test_default_method_reads_sentiment_from_unmatched_pairs() -> None:
    train = [_unmatched_pairs(32, seed=i) for i in range(3)]
    evals = {"val": _unmatched_pairs(128, seed=99)}
    reader = sweep_from_activations(
        train, evals, method=Config().method, seed=0
    ).per_seed("val")[0]
    assert reader.mean(axis=0).min() > 0.85


def test_pca_diff_follows_the_loud_content_axis_not_sentiment() -> None:
    train = [_unmatched_pairs(32, seed=i) for i in range(3)]
    evals = {"val": _unmatched_pairs(128, seed=99)}
    reader = sweep_from_activations(
        train, evals, method="pca_diff", seed=0
    ).per_seed("val")[0]
    assert reader.mean(axis=0).max() < 0.65


@pytest.mark.slow
def test_r1_on_gpt2_reads_sentiment_above_the_floor(tmp_path: Path) -> None:
    config = Config(n_seeds=3, n_pairs=32, n_eval_pairs=128)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, load_model("gpt2"), load_sst2(offline=True))
    curves = run_dir.read_arrays("curves")
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    for name, row in zip(SERIES, curves["mean"], strict=True):
        print(name, np.round(row, 3).tolist())
    print(summary)
    assert summary["best_accuracy"] > 0.65
