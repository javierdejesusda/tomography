"""Tests for the R5 honesty reading experiment."""

import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tomography.cli import load_experiment
from tomography.datasets.facts import load_facts
from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.experiments import r5_honesty
from tomography.experiments.base import RunDir
from tomography.experiments.r5_honesty import Config
from tomography.experiments.r5_honesty import execute
from tomography.experiments.r5_honesty import plot
from tomography.experiments.r5_honesty import run
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import SweepResult
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.templates import PAPER_TEMPLATES

PERSONAS = ("honest", "untruthful")
READER = SERIES.index("reader")
PEAKS = {"val": 1, "test": 2, "transfer_val": 0, "transfer_test": 1}


def _statements(
    count: int = 12, splits: tuple[Split, ...] = ("train", "val", "test")
) -> list[Example]:
    examples = []
    for split in splits:
        for r in range(count):
            body = "plain and long " * (r % 4 + 1)
            examples.append(
                Example(
                    uid=f"fact-{split}-{r}",
                    text=f"The fact number {r} is {body}",
                    label=1,
                    split=split,
                )
            )
            examples.append(
                Example(
                    uid=f"fact-{split}-{r}-false",
                    text=f"The claim number {r} is odd and {body}",
                    label=0,
                    split=split,
                )
            )
    return examples


def _true(statements: list[Example]) -> list[Example]:
    return [e for e in statements if e.label == 1]


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        n_seeds=2,
        n_pairs=6,
        n_eval_pairs=4,
        n_scan_statements=2,
    )
    return dataclasses.replace(base, **overrides)


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "r5"
    assert config.model == "gpt2"
    assert config.method == "logistic"
    assert config.position == "response"
    assert config.n_scan_statements == 4
    assert Config.from_dict(config.to_dict()) == config


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("r5")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def _scan(run_dir: RunDir, i: int, persona: str) -> tuple[Any, list[str], Any]:
    stem = f"scan_{i}_{persona}"
    arrays = run_dir.read_arrays(stem)
    sidecar = run_dir.read_json(stem)
    assert isinstance(sidecar, dict)
    assert sidecar["mask_first"] is False
    return arrays, sidecar["tokens"], sidecar


def _curves(run_dir: RunDir, stem: str, n_layers: int) -> dict[str, Any]:
    curves = run_dir.read_arrays(stem)
    assert curves["x"].tolist() == list(range(1, n_layers + 1))
    assert curves["mean"].shape == (3, n_layers)
    assert curves["per_seed"].shape == (3, 2, n_layers)
    sidecar = run_dir.read_json(stem)
    assert isinstance(sidecar, dict)
    assert sidecar["series"] == list(SERIES)
    assert sidecar["chance"] == 0.5
    return curves


def test_execute_writes_curves_scans_and_summary(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _statements())
    n_layers = tiny_arch.n_layers
    layers = list(range(1, n_layers + 1))

    sanity = _curves(run_dir, "curves", n_layers)
    transfer = _curves(run_dir, "curves_transfer", n_layers)

    for i in range(2):
        tokens = {}
        for persona in PERSONAS:
            arrays, tokens[persona], _ = _scan(run_dir, i, persona)
            assert arrays["layers"].dtype == np.int64
            assert arrays["layers"].tolist() == layers
            scores = arrays["scores"]
            assert scores.shape == (n_layers, len(tokens[persona]))
            assert np.isfinite(scores).all()
            assert all(isinstance(t, str) and t for t in tokens[persona])
        assert tokens["honest"] == tokens["untruthful"]
    assert not (run_dir.path / "scan_2_honest.npz").exists()

    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert set(summary) == {
        "best_layer",
        "val_accuracy",
        "test_accuracy",
        "ci",
        "in_template",
        "n_seeds",
        "n_pairs",
        "n_eval_pairs",
        "n_scan",
    }
    assert summary["best_layer"] in layers
    at = layers.index(summary["best_layer"])
    assert summary["test_accuracy"] == pytest.approx(transfer["mean"][0, at])
    assert summary["ci"] == pytest.approx(
        [transfer["lo"][0, at], transfer["hi"][0, at]]
    )
    check = summary["in_template"]
    assert set(check) == {"best_layer", "val_accuracy", "test_accuracy", "ci"}
    assert check["best_layer"] in layers
    at = layers.index(check["best_layer"])
    assert check["test_accuracy"] == pytest.approx(sanity["mean"][0, at])
    assert check["ci"] == pytest.approx(
        [sanity["lo"][0, at], sanity["hi"][0, at]]
    )
    assert summary["n_seeds"] == 2
    assert summary["n_pairs"] == 6
    assert summary["n_eval_pairs"] == {
        "transfer": {"val": 4, "test": 4},
        "in_template": {"val": 4, "test": 4},
    }
    assert summary["n_scan"] == 2


def _plant_sweep(monkeypatch: pytest.MonkeyPatch) -> list[SweepResult]:
    planted: list[SweepResult] = []
    real = r5_honesty.sweep_from_activations

    def spy(*args: Any, **kwargs: Any) -> SweepResult:
        result = real(*args, **kwargs)
        accuracy = np.zeros(result.accuracy.shape)
        for e, name in enumerate(result.eval_names):
            for s in range(accuracy.shape[1]):
                for seed in range(accuracy.shape[2]):
                    accuracy[e, s, seed] = (
                        0.5 - 0.05 * s + 0.01 * seed + 0.02 * e
                    )
                    accuracy[e, s, seed, PEAKS[name]] += 0.3
        planted.append(dataclasses.replace(result, accuracy=accuracy))
        return planted[-1]

    monkeypatch.setattr(r5_honesty, "sweep_from_activations", spy)
    return planted


def test_each_layer_is_chosen_on_val_and_reported_on_test(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planted = _plant_sweep(monkeypatch)
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _statements())
    [result] = planted
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    reports = (
        ("curves_transfer", "transfer_val", "transfer_test", summary),
        ("curves", "val", "test", summary["in_template"]),
    )
    for stem, val_name, test_name, report in reports:
        curves = run_dir.read_arrays(stem)
        np.testing.assert_allclose(
            curves["per_seed"], result.per_seed(test_name)
        )
        val = result.per_seed(val_name)[READER].mean(axis=0)
        test = result.per_seed(test_name)[READER].mean(axis=0)
        at = PEAKS[val_name]
        assert report["best_layer"] == result.layers[at]
        assert report["val_accuracy"] == pytest.approx(val[at])
        assert report["test_accuracy"] == pytest.approx(test[at])
        assert report["ci"] == pytest.approx(
            [curves["lo"][READER, at], curves["hi"][READER, at]]
        )


def test_the_scans_are_scored_by_the_first_reader(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planted = _plant_sweep(monkeypatch)
    activations: list[Any] = []
    real = r5_honesty.stimulus_activations

    def spy(*args: Any, **kwargs: Any) -> Any:
        activations.append(real(*args, **kwargs))
        return activations[-1]

    monkeypatch.setattr(r5_honesty, "stimulus_activations", spy)
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _statements())
    expected = planted[0].readers[0].score(activations[-1].data)
    stored = np.concatenate(
        [
            run_dir.read_arrays(f"scan_{i}_{persona}")["scores"]
            for i in range(config.n_scan_statements)
            for persona in PERSONAS
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        np.sort(stored, axis=1), np.sort(expected, axis=1), rtol=1e-5
    )


def test_the_honest_scan_scores_above_the_untruthful_one(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _statements())
    for i in range(config.n_scan_statements):
        honest = run_dir.read_arrays(f"scan_{i}_honest")["scores"]
        untruthful = run_dir.read_arrays(f"scan_{i}_untruthful")["scores"]
        assert (honest.mean(axis=1) > untruthful.mean(axis=1)).all()


def test_the_default_direction_is_logistic_and_pca_diff_stays_selectable(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config()
    pca = dataclasses.replace(config, method="pca_diff")
    default_dir = RunDir.create(config, out_root=tmp_path / "default")
    pca_dir = RunDir.create(pca, out_root=tmp_path / "pca")
    execute(config, default_dir, tiny_arch, _statements())
    execute(pca, pca_dir, tiny_arch, _statements())
    default = default_dir.read_arrays("scan_0_honest")["scores"]
    chosen = pca_dir.read_arrays("scan_0_honest")["scores"]
    assert default.shape == chosen.shape
    assert not np.allclose(default, chosen)


def test_persona_pairing_and_splits_before_pairing(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[Example], dict[str, Any]]] = []
    real = r5_honesty.build_stimulus_set

    def spy(examples: Any, template: Any, chat: Any, **kwargs: Any) -> Any:
        calls.append((list(examples), {"template": template, **kwargs}))
        return real(examples, template, chat, **kwargs)

    positions: list[str] = []
    real_activations = r5_honesty.stimulus_activations

    def spy_activations(*args: Any, **kwargs: Any) -> Any:
        positions.append(kwargs["position"])
        return real_activations(*args, **kwargs)

    monkeypatch.setattr(r5_honesty, "build_stimulus_set", spy)
    monkeypatch.setattr(r5_honesty, "stimulus_activations", spy_activations)
    config = _config()
    statements = _statements()
    execute(
        config,
        RunDir.create(config, out_root=tmp_path),
        tiny_arch,
        statements,
    )
    n = config.n_seeds
    persona = calls[: n + 2] + calls[-1:]
    assert len(calls) == n + 5
    for _, kwargs in persona:
        assert kwargs["pairing"] == "persona"
        assert kwargs["template"] is PAPER_TEMPLATES["honesty"]
    splits = [{e.split for e in examples} for examples, _ in persona]
    assert splits == [{"train"}] * n + [{"val"}, {"test"}, {"test"}]
    for examples, _ in persona:
        assert {e.label for e in examples} == {1}
    uids = [{e.uid for e in examples} for examples, _ in persona]
    assert uids[0] | uids[n] | uids[n + 1] == {e.uid for e in _true(statements)}
    assert not uids[0] & uids[n]
    assert not uids[n] & uids[n + 1]
    assert not uids[0] & uids[n + 1]
    test = [e for e in _true(statements) if e.split == "test"]
    assert [e.uid for e in calls[-1][0]] == [e.uid for e in test[:2]]
    assert calls[-1][1]["n_pairs"] is None
    assert positions[-1] == "response"


def test_the_transfer_sets_are_plain_statements_scored_by_the_same_readers(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[tuple[list[Example], dict[str, Any]]] = []
    real_build = r5_honesty.build_stimulus_set

    def spy_build(
        examples: Any, template: Any, chat: Any, **kwargs: Any
    ) -> Any:
        built.append((list(examples), {"template": template, **kwargs}))
        return real_build(examples, template, chat, **kwargs)

    collected: list[tuple[str, tuple[str, ...]]] = []
    real_activations = r5_honesty.stimulus_activations

    def spy_activations(loaded: Any, stimuli: Any, **kwargs: Any) -> Any:
        collected.append((kwargs["position"], stimuli.texts))
        return real_activations(loaded, stimuli, **kwargs)

    sweeps: list[tuple[int, list[str]]] = []
    real_sweep = r5_honesty.sweep_from_activations

    def spy_sweep(train: Any, evals: Any, **kwargs: Any) -> Any:
        sweeps.append((len(train), list(evals)))
        return real_sweep(train, evals, **kwargs)

    monkeypatch.setattr(r5_honesty, "build_stimulus_set", spy_build)
    monkeypatch.setattr(r5_honesty, "stimulus_activations", spy_activations)
    monkeypatch.setattr(r5_honesty, "sweep_from_activations", spy_sweep)
    config = _config()
    execute(
        config,
        RunDir.create(config, out_root=tmp_path),
        tiny_arch,
        _statements(),
    )
    n = config.n_seeds
    for (examples, kwargs), split in zip(
        built[n + 2 : n + 4], ("val", "test"), strict=True
    ):
        assert kwargs["pairing"] == "label_contrastive"
        assert kwargs["n_pairs"] == config.n_eval_pairs
        assert kwargs["template"].user == "{stimulus}"
        assert kwargs["template"].tagged is False
        assert {e.split for e in examples} == {split}
        assert {e.label for e in examples} == {0, 1}
    assert sweeps == [(n, ["val", "test", "transfer_val", "transfer_test"])]
    mean_texts = [texts for position, texts in collected if position == "mean"]
    assert len(mean_texts) == 2
    assert all("Pretend" not in t for texts in mean_texts for t in texts)
    assert all("USER" not in t for texts in mean_texts for t in texts)
    assert [position for position, _ in collected].count("response") == n + 3


def test_scans_follow_the_input_order_of_the_test_statements(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    statements = _statements()
    test = [e for e in _true(statements) if e.split == "test"]
    rest = [e for e in statements if e.split != "test"]
    falses = [e for e in statements if e.split == "test" and e.label == 0]
    shuffled = rest + test[::-1] + falses
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, shuffled)
    for i, statement in enumerate(test[::-1][:2]):
        for persona in PERSONAS:
            _, tokens, _ = _scan(run_dir, i, persona)
            assert "".join(tokens).strip() == statement.text.strip()


@pytest.mark.parametrize(
    "n_scan", [-1, 13], ids=["negative", "more-than-the-true-test-statements"]
)
def test_a_bad_scan_count_fails_before_the_model_runs(
    n_scan: int,
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the model must not run")

    monkeypatch.setattr(r5_honesty, "sweep_from_activations", forbidden)
    monkeypatch.setattr(r5_honesty, "stimulus_activations", forbidden)
    config = _config(n_scan_statements=n_scan)
    run_dir = RunDir.create(config, out_root=tmp_path)
    with pytest.raises(ValueError, match="n_scan_statements"):
        execute(config, run_dir, tiny_arch, _statements())


def test_as_many_scan_statements_as_true_test_statements_are_accepted(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(n_scan_statements=12)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _statements())
    assert (run_dir.path / "scan_11_honest.npz").is_file()
    assert (run_dir.path / "scan_11_untruthful.npz").is_file()
    assert not (run_dir.path / "scan_12_honest.npz").exists()


def test_zero_scan_statements_write_no_scans(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(n_scan_statements=0)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _statements())
    assert not list(run_dir.path.glob("scan_*"))
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["n_scan"] == 0
    paths = plot(run_dir, tmp_path / "figs")
    assert [p.stem for p in paths if p.suffix == ".pdf"] == [
        "curves_transfer",
        "curves",
    ]


def test_statements_without_both_labels_fail_before_the_model_runs(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the model must not run")

    monkeypatch.setattr(r5_honesty, "stimulus_activations", forbidden)
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    with pytest.raises(ValueError, match="pair"):
        execute(config, run_dir, tiny_arch, _true(_statements()))


def test_execute_is_deterministic(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config()
    first = RunDir.create(config, out_root=tmp_path / "a")
    second = RunDir.create(config, out_root=tmp_path / "b")
    execute(config, first, tiny_arch, _statements())
    execute(config, second, tiny_arch, _statements())
    stems = [
        "curves",
        "curves_transfer",
        *(f"scan_{i}_{p}" for i in range(2) for p in PERSONAS),
    ]
    for stem in stems:
        a, b = first.read_arrays(stem), second.read_arrays(stem)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_plot_writes_the_curves_and_scan_figures(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _statements())
    paths = plot(run_dir, tmp_path / "figs")
    stems = [
        "curves_transfer",
        "curves",
        *(f"scan_{i}_{p}" for i in range(2) for p in PERSONAS),
    ]
    assert [p.stem for p in paths if p.suffix == ".pdf"] == stems
    assert sorted(p.name for p in paths) == sorted(
        f"{stem}.{ext}" for stem in stems for ext in ("pdf", "png", "svg")
    )
    assert all(p.stat().st_size > 0 for p in paths)
    assert sorted(p.name for p in (tmp_path / "figs").iterdir()) == sorted(
        p.name for p in paths
    )


def test_run_passes_the_true_and_the_false_statements(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts = _statements(count=4)
    seen: list[list[Example]] = []

    def record(
        config: Config,
        run_dir: RunDir,
        loaded: LoadedModel,
        statements: Any,
    ) -> None:
        seen.append(list(statements))

    monkeypatch.setattr(r5_honesty, "load_facts", lambda: facts)
    monkeypatch.setattr(r5_honesty, "load_model", lambda key: tiny_arch)
    monkeypatch.setattr(r5_honesty, "execute", record)
    config = _config()
    run(config, RunDir.create(config, out_root=tmp_path))
    assert seen == [facts]
    assert {e.label for e in seen[0]} == {0, 1}


@pytest.mark.slow
def test_r5_on_gpt2_runs_both_evaluations(tmp_path: Path) -> None:
    config = Config(n_seeds=3, n_pairs=32)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(
        config,
        run_dir,
        load_model("gpt2"),
        load_facts(offline=True),
    )
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    for stem in ("curves", "curves_transfer"):
        curves = run_dir.read_arrays(stem)
        for name, row in zip(SERIES, curves["mean"], strict=True):
            print(stem, name, np.round(row, 3).tolist())
    print(summary)
    assert summary["in_template"]["test_accuracy"] > 0.9
    assert math.isfinite(summary["test_accuracy"])
    assert 0.0 <= summary["test_accuracy"] <= 1.0
    assert summary["n_eval_pairs"]["transfer"]["test"] > 0
    assert (run_dir.path / "curves_transfer.npz").is_file()
    assert (run_dir.path / "scan_0_honest.npz").is_file()
