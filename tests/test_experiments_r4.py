"""Tests for the R4 emotions reading experiment."""

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tomography.cli import load_experiment
from tomography.datasets import emotions as emotion_data
from tomography.datasets.emotions import EMOTIONS
from tomography.datasets.emotions import load_emotions
from tomography.datasets.records import Example
from tomography.experiments import r4_emotions
from tomography.experiments.base import RunDir
from tomography.experiments.r4_emotions import Config
from tomography.experiments.r4_emotions import execute
from tomography.experiments.r4_emotions import plot
from tomography.experiments.r4_emotions import run
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import SweepResult
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.metrics import cosine_matrix
import tomography.templates as template_data
from tomography.templates import PAPER_TEMPLATES

READER = SERIES.index("reader")
SUBSET = ("anger", "fear", "surprise")
VAL_PEAKS = (1, 0, 1)
TEST_PEAKS = (2, 1, 0)


def _scenarios(per_split: int = 14) -> list[Example]:
    examples = []
    for split in ("train", "val", "test"):
        for k, name in enumerate(EMOTIONS):
            for r in range(per_split):
                examples.append(
                    Example(
                        uid=f"emotion-{name}-{split}-{r}",
                        text=(
                            f"I feel {name} and {name} again, "
                            f"scene {r} of kind {k} in {split}"
                        ),
                        label=k,
                        split=split,
                        meta={"emotion": name},
                    )
                )
    return examples


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        n_seeds=2,
        n_pairs=6,
        n_eval_pairs=4,
        embed_points=24,
        perplexity=5.0,
    )
    return dataclasses.replace(base, **overrides)


def test_emotion_names_agree_between_data_and_templates() -> None:
    assert emotion_data.EMOTIONS == template_data.EMOTIONS


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "r4"
    assert config.model == "gpt2"
    assert config.method == "logistic"
    assert config.emotions == EMOTIONS
    assert config.embed_layers is None
    assert config.embed_points == 600
    assert config.perplexity == 30.0
    assert Config.from_dict(config.to_dict()) == config


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("r4")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_execute_writes_curves_matrix_embedding_and_summary(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _scenarios())
    n_layers = tiny_arch.n_layers
    layers = list(range(1, n_layers + 1))

    for emotion in EMOTIONS:
        curves = run_dir.read_arrays(f"curves_{emotion}")
        assert curves["x"].tolist() == layers
        assert curves["mean"].shape == (3, n_layers)
        assert curves["per_seed"].shape == (3, 2, n_layers)
        sidecar = run_dir.read_json(f"curves_{emotion}")
        assert isinstance(sidecar, dict)
        assert sidecar["series"] == list(SERIES)
        assert sidecar["chance"] == 0.5

    matrix = run_dir.read_arrays("matrix")["values"]
    assert matrix.shape == (6, 6)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(matrix), 1.0)
    assert np.abs(matrix).max() <= 1.0
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    sidecar = run_dir.read_json("matrix")
    assert isinstance(sidecar, dict)
    assert sidecar["rows"] == list(EMOTIONS)
    assert sidecar["cols"] == list(EMOTIONS)
    assert str(summary["matrix_layer"]) in sidecar["title"]

    embedding = run_dir.read_arrays("embedding")
    assert embedding["xy"].shape == (n_layers, 24, 2)
    assert embedding["xy"].dtype == np.float32
    assert embedding["labels"].shape == (24,)
    assert embedding["labels"].dtype == np.int64
    assert set(embedding["labels"].tolist()) <= set(range(6))
    assert embedding["layers"].tolist() == layers
    assert run_dir.read_json("embedding") == {"classes": list(EMOTIONS)}

    assert set(summary) == {
        "matrix_layer",
        "emotions",
        "n_seeds",
        "n_pairs",
        "n_eval_pairs",
    }
    assert summary["matrix_layer"] in layers
    assert summary["n_seeds"] == 2
    assert summary["n_pairs"] == 6
    assert set(summary["emotions"]) == set(EMOTIONS)
    for emotion in EMOTIONS:
        entry = summary["emotions"][emotion]
        assert set(entry) == {
            "best_layer",
            "val_accuracy",
            "test_accuracy",
            "ci",
        }
        assert entry["best_layer"] in layers
        at = layers.index(entry["best_layer"])
        test = run_dir.read_arrays(f"curves_{emotion}")
        assert entry["test_accuracy"] == pytest.approx(test["mean"][0, at])
        assert entry["ci"] == pytest.approx(
            [test["lo"][0, at], test["hi"][0, at]]
        )
        assert summary["n_eval_pairs"][emotion] == {"val": 4, "test": 4}


def _plant_sweeps(monkeypatch: pytest.MonkeyPatch) -> list[SweepResult]:
    planted: list[SweepResult] = []
    real = r4_emotions.reading_sweep

    def spy(*args: Any, **kwargs: Any) -> SweepResult:
        result = real(*args, **kwargs)
        k = len(planted)
        accuracy = np.zeros(result.accuracy.shape)
        for name, peak in (("val", VAL_PEAKS[k]), ("test", TEST_PEAKS[k])):
            e = result.eval_names.index(name)
            for s in range(accuracy.shape[1]):
                for seed in range(accuracy.shape[2]):
                    accuracy[e, s, seed] = 0.5 - 0.05 * s + 0.01 * seed
                    accuracy[e, s, seed, peak] += 0.3
        first = result.readers[0]
        readers = (
            dataclasses.replace(first, signs=-first.signs),
            *result.readers[1:],
        )
        planted.append(
            dataclasses.replace(result, accuracy=accuracy, readers=readers)
        )
        return planted[-1]

    monkeypatch.setattr(r4_emotions, "reading_sweep", spy)
    return planted


def test_each_layer_is_chosen_on_val_and_reported_on_test(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planted = _plant_sweeps(monkeypatch)
    config = _config(emotions=SUBSET)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _scenarios())
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    for emotion, result, peak in zip(SUBSET, planted, VAL_PEAKS, strict=True):
        curves = run_dir.read_arrays(f"curves_{emotion}")
        np.testing.assert_allclose(curves["per_seed"], result.per_seed("test"))
        val = result.per_seed("val")[READER].mean(axis=0)
        test = result.per_seed("test")[READER].mean(axis=0)
        entry = summary["emotions"][emotion]
        assert entry["best_layer"] == result.layers[peak]
        assert entry["val_accuracy"] == pytest.approx(val[peak])
        assert entry["test_accuracy"] == pytest.approx(test[peak])
        assert entry["ci"] == pytest.approx(
            [curves["lo"][READER, peak], curves["hi"][READER, peak]]
        )


def test_the_matrix_compares_the_signed_seed_means_at_the_best_val_layer(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planted = _plant_sweeps(monkeypatch)
    config = _config(emotions=SUBSET)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _scenarios())
    layers = planted[0].layers
    mean_val = np.mean(
        [r.per_seed("val")[READER].mean(axis=0) for r in planted], axis=0
    )
    at = int(np.argmax(mean_val))
    assert at != len(layers) - 1
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["matrix_layer"] == layers[at]
    directions = np.stack(
        [
            np.mean([r.signs[at] * r.directions[at] for r in p.readers], axis=0)
            for p in planted
        ]
    )
    np.testing.assert_allclose(
        run_dir.read_arrays("matrix")["values"],
        cosine_matrix(directions),
        atol=1e-6,
    )


def test_the_embedding_draws_test_scenarios_at_the_requested_layers(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[list[str], Any]] = []
    real = r4_emotions.collect_activations

    def spy(loaded: LoadedModel, texts: Any, **kwargs: Any) -> Any:
        seen.append((list(texts), kwargs["layers"]))
        return real(loaded, texts, **kwargs)

    monkeypatch.setattr(r4_emotions, "collect_activations", spy)
    config = _config(emotions=SUBSET, embed_layers=(2, 3))
    run_dir = RunDir.create(config, out_root=tmp_path)
    scenarios = _scenarios()
    execute(config, run_dir, tiny_arch, scenarios)
    [(texts, layers)] = seen
    assert tuple(layers) == (2, 3)
    assert run_dir.read_arrays("embedding")["layers"].tolist() == [2, 3]
    assert len(texts) == config.embed_points
    for text in texts:
        [owner] = [e for e in scenarios if e.text in text]
        assert owner.split == "test"
        assert owner.meta["emotion"] in SUBSET


def test_pairing_is_one_vs_rest_per_emotion_and_split_before_pairing(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[Example], dict[str, Any]]] = []
    real = r4_emotions.build_stimulus_set

    def spy(examples: Any, template: Any, chat: Any, **kwargs: Any) -> Any:
        calls.append((list(examples), {"template": template, **kwargs}))
        return real(examples, template, chat, **kwargs)

    monkeypatch.setattr(r4_emotions, "build_stimulus_set", spy)
    config = _config()
    execute(
        config,
        RunDir.create(config, out_root=tmp_path),
        tiny_arch,
        _scenarios(),
    )
    per_emotion = config.n_seeds + 2
    assert len(calls) == per_emotion * len(EMOTIONS)
    for k, emotion in enumerate(EMOTIONS):
        block = calls[k * per_emotion : (k + 1) * per_emotion]
        for _, kwargs in block:
            assert kwargs["pairing"] == "one_vs_rest"
            assert kwargs["target_class"] == k
            assert kwargs["concept"] == emotion
            assert kwargs["template"] is PAPER_TEMPLATES["emotion_concept"]
        splits = [{e.split for e in examples} for examples, _ in block]
        assert splits[: config.n_seeds] == [{"train"}] * config.n_seeds
        assert splits[config.n_seeds :] == [{"val"}, {"test"}]


def test_a_subset_keeps_the_class_ids_of_the_full_list(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(emotions=("anger", "fear"))
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _scenarios())
    written = {p.stem for p in run_dir.path.glob("curves_*.npz")}
    assert written == {"curves_anger", "curves_fear"}
    matrix = run_dir.read_arrays("matrix")["values"]
    assert matrix.shape == (2, 2)
    sidecar = run_dir.read_json("matrix")
    assert isinstance(sidecar, dict)
    assert sidecar["rows"] == ["anger", "fear"]
    labels = run_dir.read_arrays("embedding")["labels"]
    assert set(labels.tolist()) == {2, 3}


@pytest.mark.parametrize(
    "emotions",
    [(), ("anger", "joy"), ("anger", "anger")],
    ids=["empty", "unknown", "duplicate"],
)
def test_bad_emotions_fail_before_the_model_runs(
    emotions: tuple[str, ...],
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the model must not run")

    monkeypatch.setattr(r4_emotions, "reading_sweep", forbidden)
    config = _config(emotions=emotions)
    run_dir = RunDir.create(config, out_root=tmp_path)
    with pytest.raises(ValueError, match="emotions"):
        execute(config, run_dir, tiny_arch, _scenarios())


def test_execute_is_deterministic(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(emotions=("anger", "fear", "surprise"))
    first = RunDir.create(config, out_root=tmp_path / "a")
    second = RunDir.create(config, out_root=tmp_path / "b")
    execute(config, first, tiny_arch, _scenarios())
    execute(config, second, tiny_arch, _scenarios())
    stems = ["matrix", "embedding", *(f"curves_{e}" for e in config.emotions)]
    for stem in stems:
        a, b = first.read_arrays(stem), second.read_arrays(stem)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_the_default_direction_is_logistic_and_pca_diff_stays_selectable(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(emotions=("anger", "fear"))
    pca = dataclasses.replace(config, method="pca_diff")
    default_dir = RunDir.create(config, out_root=tmp_path / "default")
    pca_dir = RunDir.create(pca, out_root=tmp_path / "pca")
    execute(config, default_dir, tiny_arch, _scenarios())
    execute(pca, pca_dir, tiny_arch, _scenarios())
    default = default_dir.read_arrays("matrix")["values"]
    chosen = pca_dir.read_arrays("matrix")["values"]
    assert default.shape == chosen.shape
    assert not np.allclose(default, chosen)


def test_plot_writes_every_figure_in_three_formats(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(n_seeds=2)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _scenarios())
    paths = plot(run_dir, tmp_path / "figs")
    stems = [*(f"curves_{e}" for e in EMOTIONS), "matrix", "embedding"]
    assert [p.stem for p in paths if p.suffix == ".pdf"] == stems
    assert sorted(p.name for p in paths) == sorted(
        f"{stem}.{ext}" for stem in stems for ext in ("pdf", "png", "svg")
    )
    assert all(p.stat().st_size > 0 for p in paths)
    assert sorted(p.name for p in (tmp_path / "figs").iterdir()) == sorted(
        p.name for p in paths
    )


def test_run_loads_the_model_and_the_scenarios(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []

    def fake_load_model(key: str) -> LoadedModel:
        asked.append(key)
        return tiny_arch

    monkeypatch.setattr(r4_emotions, "load_model", fake_load_model)
    monkeypatch.setattr(r4_emotions, "load_emotions", _scenarios)
    config = _config(emotions=("anger", "fear"))
    run_dir = RunDir.create(config, out_root=tmp_path)
    run(config, run_dir)
    assert asked == ["tiny"]
    assert (run_dir.path / "summary.json").is_file()
    assert run_dir.meta["finished_at"] is None


@pytest.mark.slow
def test_r4_on_gpt2_reads_emotions_above_the_floor(tmp_path: Path) -> None:
    config = Config(n_seeds=3, n_pairs=32, embed_points=300)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(
        config,
        run_dir,
        load_model("gpt2"),
        load_emotions(offline=True),
    )
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    for emotion, entry in summary["emotions"].items():
        print(emotion, entry)
    mean = np.mean([e["test_accuracy"] for e in summary["emotions"].values()])
    assert mean > 0.6
    assert (run_dir.path / "embedding.npz").is_file()
    assert (run_dir.path / "matrix.npz").is_file()
