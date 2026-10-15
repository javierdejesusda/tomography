"""Tests for the LR1 experiment: one LoRRA adapter trained on Alpaca pairs."""

from collections.abc import Sequence
import copy
import dataclasses
from pathlib import Path
import statistics
from typing import Any

import numpy as np
from peft.tuners.lora import LoraLayer
import pytest
import torch

from tomography.cli import load_experiment
from tomography.control.generate import Completion
from tomography.datasets.records import Example
from tomography.experiments import lr1_lorra
from tomography.experiments.base import RunDir
from tomography.experiments.lr1_lorra import Config
from tomography.experiments.lr1_lorra import execute
from tomography.experiments.lr1_lorra import gated_arm
from tomography.experiments.lr1_lorra import load_adapter
from tomography.experiments.lr1_lorra import plot
from tomography.experiments.lr1_lorra import run
from tomography.experiments.steering import Arm
from tomography.experiments.steering import GenSettings
from tomography.experiments.steering import LogprobFn
from tomography.experiments.steering import PerplexityFn
from tomography.experiments.steering import plain_arm
from tomography.lorra import LorraTrace
from tomography.lorra import default_layers
from tomography.model import LoadedModel

INSTRUCTIONS = (
    "Is the movie good?",
    "Is the dog lazy?",
    "Does she feel happy?",
    "Pick a number.",
    "Is the model right?",
    "Are the answers true?",
)
RESPONSES = (
    "Yes, the movie is very good.",
    "The quick brown fox jumps over the lazy dog.",
    "She felt happy, then sad, then angry.",
    "Pick a number between 0 and 9: 0123456789.",
    "The model reads the stream of the residual.",
    "Honest answers are true, and dishonest answers are false.",
)
TRAINABLE = {"gpt2": 2048, "qwen3": 1792}
GREEDY = GenSettings(
    max_new_tokens=5,
    temperature=None,
    num_samples=1,
    batch_size=2,
    score_max_length=8,
)
PROMPTS = ["Is the movie good?", "Pick a number."]
CONTINUATIONS = [" Yes, the movie is good.", " 0123456789"]
NEUTRAL = [
    "The quick brown fox jumps over the lazy dog.",
    "A short text, a longer text, and another text.",
]


def _examples(
    n_train: int = 24, n_val: int = 8, *, indexed: bool = True
) -> list[Example]:
    """Alpaca-like rows whose instruction names the split.

    Indexed rows are all distinct. The others cycle through the same two
    pairs in both splits, which a few training steps learn well enough to
    lower the held-out loss.
    """
    rows = []
    for split, count in (("train", n_train), ("val", n_val)):
        for i in range(count):
            kind = i % 6 if indexed else i % 2
            suffix = f" {split} {i}" if indexed else ""
            rows.append(
                Example(
                    uid=f"alpaca-{split}-{i}",
                    text=RESPONSES[(i + i // 6) % 6 if indexed else kind],
                    label=-1,
                    split="train" if split == "train" else "val",
                    meta={"instruction": f"{INSTRUCTIONS[kind]}{suffix}"},
                )
            )
    return rows


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        layers=(1, 2),
        lr=3e-2,
        batch_size=4,
        steps=12,
        n_train=20,
        n_heldout=6,
        max_prompt_tokens=64,
        max_response_tokens=8,
    )
    return dataclasses.replace(base, **overrides)


def _execute(
    loaded: LoadedModel,
    tmp_path: Path,
    examples: Sequence[Example] | None = None,
    **overrides: Any,
) -> RunDir:
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(
        config, run_dir, loaded, _examples() if examples is None else examples
    )
    return run_dir


def _finished(loaded: LoadedModel, tmp_path: Path, **overrides: Any) -> RunDir:
    run_dir = _execute(loaded, tmp_path, **overrides)
    run_dir.finalize()
    return run_dir


def _summary(run_dir: RunDir) -> dict[str, Any]:
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    return summary


def _table(run_dir: RunDir) -> list[dict[str, Any]]:
    table = run_dir.read_json("table")
    assert isinstance(table, list)
    return table


def _lora_layers(loaded: LoadedModel) -> list[torch.nn.Module]:
    return [m for m in loaded.model.modules() if isinstance(m, LoraLayer)]


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "lr1"
    assert config.model == "qwen3-1.7b"
    assert config.layers is None
    assert config.alpha == 5.0
    assert config.pos_control == "Give a truthful answer."
    assert config.neg_control == "Give an untruthful answer."
    assert (config.rank, config.lora_alpha, config.dropout) == (8, 16, 0.05)
    assert (config.lr, config.batch_size, config.steps) == (3e-4, 16, 70)
    assert (config.max_prompt_tokens, config.max_response_tokens) == (256, 64)
    assert config.loss == "mean_norm"
    assert config.gradient_checkpointing is False
    assert (config.n_train, config.n_heldout) == (2000, 64)
    assert config.bos is None
    assert config.seed == 0
    assert Config.from_dict(config.to_dict()) == config
    tuned = Config(layers=(2, 4), loss="sum_squared", bos="none", seed=3)
    assert Config.from_dict(tuned.to_dict()) == tuned


def test_settings_default_the_layers_to_the_reference_fraction() -> None:
    assert Config().settings(28).layers == default_layers(28)
    assert Config().settings(28).layers == (9, 11, 13, 15, 17)
    assert Config(layers=(2, 4)).settings(28).layers == (2, 4)


def test_settings_copy_every_training_field() -> None:
    config = Config(
        layers=(1, 3),
        alpha=2.5,
        pos_control="Be honest.",
        neg_control="Be dishonest.",
        rank=4,
        lora_alpha=8,
        dropout=0.1,
        lr=1e-3,
        batch_size=3,
        steps=5,
        max_prompt_tokens=100,
        max_response_tokens=10,
        loss="sum_squared",
        gradient_checkpointing=True,
        seed=7,
    )
    settings = config.settings(3)
    assert dataclasses.asdict(settings) == {
        "layers": (1, 3),
        "alpha": 2.5,
        "pos_control": "Be honest.",
        "neg_control": "Be dishonest.",
        "rank": 4,
        "lora_alpha": 8,
        "dropout": 0.1,
        "lr": 1e-3,
        "batch_size": 3,
        "steps": 5,
        "max_prompt_tokens": 100,
        "max_response_tokens": 10,
        "loss": "sum_squared",
        "gradient_checkpointing": True,
        "seed": 7,
    }


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("lr1")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_execute_writes_the_adapter_curves_table_and_summary(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(tiny_arch, tmp_path)
    for name in (
        "adapter.npz",
        "curves.npz",
        "curves.json",
        "table.json",
        "summary.json",
    ):
        assert (run_dir.path / name).is_file(), name


def test_execute_stores_the_adapter_weights(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(tiny_arch, tmp_path)
    arrays = run_dir.read_arrays("adapter")
    assert all(a.dtype == np.float32 for a in arrays.values())
    assert (
        sum(a.size for a in arrays.values())
        == TRAINABLE[tiny_arch.spec.model_type]
    )
    assert any(np.abs(a).sum() > 0 for k, a in arrays.items() if "lora_B" in k)


def test_execute_writes_the_loss_curve_of_one_run(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(tiny_arch, tmp_path, steps=9)
    curves = run_dir.read_arrays("curves")
    assert curves["x"].tolist() == list(range(9))
    assert curves["mean"].shape == (1, 9)
    assert curves["per_seed"].shape == (1, 1, 9)
    np.testing.assert_array_equal(curves["lo"], curves["mean"])
    np.testing.assert_array_equal(curves["hi"], curves["mean"])
    assert np.isfinite(curves["mean"]).all()
    sidecar = run_dir.read_json("curves")
    assert isinstance(sidecar, dict)
    assert sidecar["series"] == ["train loss"]
    assert sidecar["xlabel"] == "Step"
    assert sidecar["ylabel"] == "Loss"
    assert sidecar["chance"] is None


def test_the_curve_is_the_loss_of_every_step(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    traces: list[LorraTrace] = []
    real = lr1_lorra.train_lorra

    def spy(*args: Any, **kwargs: Any) -> LorraTrace:
        traces.append(real(*args, **kwargs))
        return traces[-1]

    monkeypatch.setattr(lr1_lorra, "train_lorra", spy)
    run_dir = _execute(tiny_arch, tmp_path, steps=7)
    curves = run_dir.read_arrays("curves")
    np.testing.assert_allclose(curves["mean"][0], traces[0].losses, rtol=1e-6)


def test_the_table_has_one_row_per_target_layer(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measured: list[dict[int, float]] = []
    real = lr1_lorra.heldout_losses

    def spy(*args: Any, **kwargs: Any) -> dict[int, float]:
        measured.append(real(*args, **kwargs))
        return measured[-1]

    monkeypatch.setattr(lr1_lorra, "heldout_losses", spy)
    examples = _examples(indexed=False)
    run_dir = _execute(tiny_arch, tmp_path, examples, steps=30)
    before, after = measured
    rows = _table(run_dir)
    assert [row["layer"] for row in rows] == [1, 2]
    for row in rows:
        layer = row["layer"]
        assert set(row) == {
            "layer",
            "heldout_before",
            "heldout_after",
            "reduction",
        }
        assert row["heldout_before"] == pytest.approx(before[layer])
        assert row["heldout_after"] == pytest.approx(after[layer])
        assert row["reduction"] == pytest.approx(
            1 - after[layer] / before[layer]
        )
        assert row["reduction"] > 0.0


def test_the_reduction_is_null_when_the_untrained_loss_is_zero(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def zero(*args: Any, **kwargs: Any) -> dict[int, float]:
        return dict.fromkeys((1, 2), 0.0)

    monkeypatch.setattr(lr1_lorra, "heldout_losses", zero)
    rows = _table(_execute(tiny_arch, tmp_path))
    assert [row["heldout_before"] for row in rows] == [0.0, 0.0]
    assert [row["reduction"] for row in rows] == [None, None]


def test_the_summary_reports_the_run(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    traces: list[LorraTrace] = []
    real = lr1_lorra.train_lorra

    def spy(*args: Any, **kwargs: Any) -> LorraTrace:
        traces.append(real(*args, **kwargs))
        return traces[-1]

    monkeypatch.setattr(lr1_lorra, "train_lorra", spy)
    run_dir = _execute(tiny_arch, tmp_path, steps=25)
    summary = _summary(run_dir)
    assert set(summary) == {
        "layers",
        "blocks",
        "n_trainable",
        "n_train",
        "n_heldout",
        "dropped_train",
        "dropped_heldout",
        "steps",
        "first_loss",
        "last_loss",
        "peak_memory_mb",
        "seconds",
    }
    losses = traces[0].losses
    assert summary["layers"] == [1, 2]
    assert summary["blocks"] == [0, 1]
    assert summary["n_trainable"] == TRAINABLE[tiny_arch.spec.model_type]
    assert (summary["n_train"], summary["n_heldout"]) == (20, 6)
    assert (summary["dropped_train"], summary["dropped_heldout"]) == (0, 0)
    assert summary["steps"] == 25
    assert summary["first_loss"] == pytest.approx(statistics.fmean(losses[:10]))
    assert summary["last_loss"] == pytest.approx(statistics.fmean(losses[-10:]))
    assert summary["peak_memory_mb"] is None
    assert summary["seconds"] > 0.0


def test_first_and_last_loss_average_all_steps_of_a_short_run(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(tiny_arch, tmp_path, steps=4)
    summary = _summary(run_dir)
    curve = run_dir.read_arrays("curves")["mean"][0]
    assert summary["first_loss"] == pytest.approx(float(curve.mean()))
    assert summary["last_loss"] == pytest.approx(float(curve.mean()))


def test_execute_draws_train_and_heldout_pairs_from_their_splits(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, list[Any]] = {"train": [], "heldout": []}
    real_train, real_heldout = lr1_lorra.train_lorra, lr1_lorra.heldout_losses

    def train_spy(
        loaded: Any, adapter: Any, pairs: Any, *a: Any, **k: Any
    ) -> Any:
        seen["train"] = list(pairs)
        return real_train(loaded, adapter, pairs, *a, **k)

    def heldout_spy(
        loaded: Any, adapter: Any, pairs: Any, *a: Any, **k: Any
    ) -> Any:
        seen["heldout"] = list(pairs)
        return real_heldout(loaded, adapter, pairs, *a, **k)

    monkeypatch.setattr(lr1_lorra, "train_lorra", train_spy)
    monkeypatch.setattr(lr1_lorra, "heldout_losses", heldout_spy)
    _execute(tiny_arch, tmp_path, n_train=10, n_heldout=5)
    assert len(seen["train"]) == 10
    assert len(seen["heldout"]) == 5
    assert all(" train " in instruction for instruction, _ in seen["train"])
    assert all(" val " in instruction for instruction, _ in seen["heldout"])
    assert len({pair[0] for pair in seen["train"]}) == 10
    assert len({pair[0] for pair in seen["heldout"]}) == 5


def test_execute_uses_every_row_when_the_bound_is_larger(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(tiny_arch, tmp_path, n_train=500, n_heldout=500)
    summary = _summary(run_dir)
    assert (summary["n_train"], summary["n_heldout"]) == (24, 8)


def test_the_draw_depends_on_the_seed_and_is_reproducible(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drawn: list[list[Any]] = []
    real = lr1_lorra.train_lorra

    def spy(loaded: Any, adapter: Any, pairs: Any, *a: Any, **k: Any) -> Any:
        drawn.append(list(pairs))
        return real(loaded, adapter, pairs, *a, **k)

    monkeypatch.setattr(lr1_lorra, "train_lorra", spy)
    twins = [copy.deepcopy(tiny_arch) for _ in range(3)]
    for index, (loaded, seed) in enumerate(zip(twins, (0, 0, 1), strict=True)):
        _execute(loaded, tmp_path / str(index), n_train=10, steps=2, seed=seed)
    assert drawn[0] == drawn[1]
    assert drawn[0] != drawn[2]


def test_execute_drops_the_pairs_that_do_not_fit(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    examples = _examples()
    examples.append(
        Example(
            uid="alpaca-train-empty",
            text="",
            label=-1,
            split="train",
            meta={"instruction": "Is it empty?"},
        )
    )
    examples.append(
        Example(
            uid="alpaca-val-long",
            text="Yes.",
            label=-1,
            split="val",
            meta={"instruction": " ".join(INSTRUCTIONS * 4)},
        )
    )
    run_dir = _execute(
        tiny_arch, tmp_path, examples, n_train=500, n_heldout=500
    )
    summary = _summary(run_dir)
    assert (summary["dropped_train"], summary["dropped_heldout"]) == (1, 1)
    assert (summary["n_train"], summary["n_heldout"]) == (24, 8)


def test_execute_needs_a_full_batch_of_training_pairs(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="training"):
        _execute(tiny_arch, tmp_path, _examples(n_train=3))
    with pytest.raises(ValueError, match="training"):
        _execute(tiny_arch, tmp_path / "small-bound", n_train=3)
    assert _lora_layers(tiny_arch) == []


def test_execute_needs_a_heldout_pair(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="held-out"):
        _execute(tiny_arch, tmp_path, _examples(n_val=0))
    assert _lora_layers(tiny_arch) == []


def test_execute_is_deterministic(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    twin = copy.deepcopy(tiny_arch)
    first = _execute(tiny_arch, tmp_path / "a")
    second = _execute(twin, tmp_path / "b")
    for stem in ("adapter", "curves"):
        a, b = first.read_arrays(stem), second.read_arrays(stem)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    assert _table(first) == _table(second)


def test_run_loads_the_model_and_the_alpaca_rows(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []

    def fake_load_model(key: str) -> LoadedModel:
        asked.append(key)
        return tiny_arch

    monkeypatch.setattr(lr1_lorra, "load_model", fake_load_model)
    monkeypatch.setattr(lr1_lorra, "load_alpaca", _examples)
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    run(config, run_dir)
    assert asked == ["tiny"]
    assert (run_dir.path / "adapter.npz").is_file()
    assert run_dir.meta["finished_at"] is None


def test_plot_writes_the_loss_curve_in_three_formats(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(tiny_arch, tmp_path)
    paths = plot(run_dir, tmp_path / "figs")
    assert sorted(p.name for p in paths) == [
        "curves.pdf",
        "curves.png",
        "curves.svg",
    ]
    assert all(p.stat().st_size > 0 for p in paths)


def _greedy(loaded: LoadedModel) -> list[list[Completion]]:
    return plain_arm(loaded, GREEDY).generate(PROMPTS)


def _scorers(arm: Arm) -> tuple[LogprobFn, PerplexityFn]:
    assert arm.logprobs is not None
    assert arm.perplexity is not None
    return arm.logprobs, arm.perplexity


def _scores(loaded: LoadedModel) -> np.ndarray[Any, Any]:
    logprobs, _ = _scorers(plain_arm(loaded, GREEDY))
    return logprobs(PROMPTS, CONTINUATIONS)


def test_load_adapter_reproduces_the_trained_model(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    fresh = copy.deepcopy(tiny_arch)
    run_dir = _finished(tiny_arch, tmp_path)
    trained, trained_scores = _greedy(tiny_arch), _scores(tiny_arch)
    assert not np.array_equal(trained_scores, _scores(fresh))
    adapter = load_adapter(fresh, run_dir.path)
    assert _greedy(fresh) == trained
    np.testing.assert_array_equal(_scores(fresh), trained_scores)
    assert adapter.settings.layers == (1, 2)
    assert adapter.blocks == (0, 1)
    assert adapter.n_trainable() == TRAINABLE[tiny_arch.spec.model_type]


def test_load_adapter_accepts_a_string_path(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    fresh = copy.deepcopy(tiny_arch)
    run_dir = _finished(tiny_arch, tmp_path)
    load_adapter(fresh, str(run_dir.path))
    assert _greedy(fresh) == _greedy(tiny_arch)


def test_load_adapter_defaults_the_layers_from_the_model_depth(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    fresh = copy.deepcopy(tiny_arch)
    run_dir = _finished(tiny_arch, tmp_path, layers=None)
    adapter = load_adapter(fresh, run_dir.path)
    assert adapter.settings.layers == default_layers(tiny_arch.n_layers)


def test_load_adapter_rejects_an_unfinished_run(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    fresh = copy.deepcopy(tiny_arch)
    run_dir = _execute(tiny_arch, tmp_path)
    with pytest.raises(ValueError, match="not a finished run"):
        load_adapter(fresh, run_dir.path)
    assert _lora_layers(fresh) == []


def test_load_adapter_rejects_another_architecture(
    tiny_gpt2: LoadedModel, tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _finished(tiny_gpt2, tmp_path)
    with pytest.raises(ValueError, match="keys"):
        load_adapter(tiny_qwen3, run_dir.path)


def test_gated_arm_off_is_the_model_that_never_had_an_adapter(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    pristine = copy.deepcopy(tiny_arch)
    fresh = copy.deepcopy(tiny_arch)
    run_dir = _finished(tiny_arch, tmp_path)
    adapter = load_adapter(fresh, run_dir.path)
    reference = plain_arm(pristine, GREEDY, name="none")
    off = gated_arm(
        adapter, plain_arm(fresh, GREEDY, name="none"), enabled=False
    )
    off_logprobs, off_perplexity = _scorers(off)
    ref_logprobs, ref_perplexity = _scorers(reference)
    assert off.generate(PROMPTS) == reference.generate(PROMPTS)
    np.testing.assert_array_equal(
        off_logprobs(PROMPTS, CONTINUATIONS),
        ref_logprobs(PROMPTS, CONTINUATIONS),
    )
    assert off_perplexity(NEUTRAL) == ref_perplexity(NEUTRAL)


def test_gated_arm_on_is_the_adapted_model(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    pristine = copy.deepcopy(tiny_arch)
    fresh = copy.deepcopy(tiny_arch)
    run_dir = _finished(tiny_arch, tmp_path)
    adapter = load_adapter(fresh, run_dir.path)
    reference = plain_arm(pristine, GREEDY, name="lorra")
    on = gated_arm(
        adapter, plain_arm(fresh, GREEDY, name="lorra"), enabled=True
    )
    trained = plain_arm(tiny_arch, GREEDY, name="lorra")
    on_logprobs, on_perplexity = _scorers(on)
    trained_logprobs, _ = _scorers(trained)
    ref_logprobs, ref_perplexity = _scorers(reference)
    assert on.generate(PROMPTS) == trained.generate(PROMPTS)
    on_scores = on_logprobs(PROMPTS, CONTINUATIONS)
    np.testing.assert_array_equal(
        on_scores, trained_logprobs(PROMPTS, CONTINUATIONS)
    )
    assert not np.array_equal(on_scores, ref_logprobs(PROMPTS, CONTINUATIONS))
    assert on_perplexity(NEUTRAL) != ref_perplexity(NEUTRAL)


def test_gated_arm_gates_every_callable_and_keeps_the_name(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    fresh = copy.deepcopy(tiny_arch)
    run_dir = _finished(tiny_arch, tmp_path)
    adapter = load_adapter(fresh, run_dir.path)
    probe = _lora_layers(fresh)[0]
    states: list[bool] = []

    def generate(prompts: Sequence[str]) -> list[list[Completion]]:
        states.append(probe.disable_adapters)
        return [[] for _ in prompts]

    def logprobs(
        prompts: Sequence[str], continuations: Sequence[str]
    ) -> np.ndarray[Any, Any]:
        states.append(probe.disable_adapters)
        return np.zeros(len(prompts), dtype=np.float32)

    def perplexity(texts: Sequence[str]) -> float:
        states.append(probe.disable_adapters)
        return 1.0

    arm = Arm("mine", generate, logprobs, perplexity)
    for enabled in (False, True):
        states.clear()
        gated = gated_arm(adapter, arm, enabled=enabled)
        assert gated.name == "mine"
        gated_logprobs, gated_perplexity = _scorers(gated)
        gated.generate(["x"])
        gated_logprobs(["x"], ["y"])
        gated_perplexity(["z"])
        assert states == [not enabled] * 3


def test_gated_arm_keeps_a_missing_callable_missing(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    fresh = copy.deepcopy(tiny_arch)
    run_dir = _finished(tiny_arch, tmp_path)
    adapter = load_adapter(fresh, run_dir.path)
    bare = Arm("bare", lambda prompts: [[] for _ in prompts])
    gated = gated_arm(adapter, bare, enabled=False)
    assert gated.name == "bare"
    assert gated.logprobs is None
    assert gated.perplexity is None
    assert gated.generate(["x"]) == [[]]
