"""Tests for the P9d steering experiment."""

from collections.abc import Callable
from collections.abc import Sequence
import csv
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.cli import load_experiment
from tomography.control.generate import Completion
from tomography.datasets.records import MetaValue
from tomography.datasets.records import QAExample
from tomography.datasets.records import Split
from tomography.experiments import p9d_steering
from tomography.experiments.base import RunDir
from tomography.experiments.p9a_labels import Config as LabelsConfig
from tomography.experiments.p9d_steering import Config
from tomography.experiments.p9d_steering import execute
from tomography.experiments.p9d_steering import plan_groups
from tomography.experiments.p9d_steering import plot
from tomography.experiments.p9d_steering import resolve_mode
from tomography.experiments.p9d_steering import run
from tomography.experiments.steering import Arm
from tomography.experiments.steering import GenSettings
from tomography.experiments.steering import reader_arm
from tomography.experiments.sweep import chat_format_for
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import ItemLabel
from tomography.knowledge import Label
from tomography.knowledge import few_shot_prompt
from tomography.model import MODELS
from tomography.model import LoadedModel
from tomography.p9.behavior import CHECK_COLUMNS
from tomography.p9.pool import LabeledItem
from tomography.p9.pool import pool_records
from tomography.p9.reading import collect_item_activations
from tomography.p9.reading import fit_type_reader
from tomography.p9.reading import p9_template
from tomography.p9.reading import type_auroc
from tomography.p9.signals import TokenStats
from tomography.p9.signals import first_answer_token
from tomography.p9.unknowns import type_pools
from tomography.reading.metrics import wilson_ci
from tomography.reading.reader import RepReader
from tomography.templates import ChatFormat

SPLITS: tuple[Split, ...] = ("train", "train", "train", "val", "test")
LABELS: tuple[Label, ...] = (
    "known",
    "known",
    "unknown",
    "unknown",
    "known",
    "uncertain",
)
RELATIONS = ("father", "mother", "occupation")


def _items(n_detail: int = 40) -> list[LabeledItem]:
    """A planted pool: three knowledge contrasts, none of them tiny."""
    items: list[LabeledItem] = []

    def add(
        prefix: str,
        n: int,
        source: str,
        kind: str,
        label_of: Any,
        meta_of: Any,
        *,
        gold: bool = True,
    ) -> None:
        for i in range(n):
            uid = f"{prefix}:{i:03d}"
            label: Label = label_of(i)
            correct = gold and label == "known"
            example = QAExample(
                uid=uid,
                question=f"Who is {prefix} {i}?",
                answers=("gold", "alias") if gold else (),
                kind=kind,
                split=SPLITS[i % 5],
                meta={"source": source, **meta_of(i)},
            )
            item_label = ItemLabel(
                uid=uid,
                label=label,
                basis="behavior" if gold else "construction",
                greedy=("gold" if correct else "other") if gold else "",
                greedy_correct=correct,
                samples=(("gold",) if correct else ("other",)) * 5
                if gold
                else (),
                n_correct=5 if correct else 0,
            )
            items.append(LabeledItem(example, item_label))

    def unknown(_: int) -> Label:
        return "unknown"

    def relation(i: int) -> dict[str, MetaValue]:
        return {"relation": RELATIONS[i % 3], "popularity": None}

    add(
        "popqa",
        120,
        "popqa",
        "real",
        lambda i: LABELS[i % 6],
        lambda i: {
            "relation": RELATIONS[i % 3],
            "popularity": 10 ** (1 + i % 7),
        },
    )
    add(
        "generated", 40, "generated", "fictional", unknown, relation, gold=False
    )
    add(
        "detail",
        n_detail,
        "wikidata",
        "unknowable_detail",
        unknown,
        lambda i: {"relation": "ordinary_date"},
        gold=False,
    )
    for source in ("triviaqa", "nq_open"):
        add(
            source,
            60,
            source,
            "open",
            lambda i: "known" if i % 2 == 0 else "unknown",
            lambda i: {},
        )
    return items


def _labels_run(
    tmp_path: Path,
    items: Sequence[LabeledItem],
    *,
    known_contrasts: bool = True,
    n_shots: int = 2,
) -> RunDir:
    run_dir = RunDir.create(
        LabelsConfig(model="tiny"), out_root=tmp_path / "labels"
    )
    run_dir.write_json("items", pool_records(items))
    run_dir.write_json(
        "summary",
        {"known_contrasts": known_contrasts, "recipe": {"n_shots": n_shots}},
    )
    run_dir.finalize()
    return run_dir


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        labels_run="unused",
        n_shots=2,
        n_prompts=4,
        n_calibration=10,
        n_pairs=16,
        min_pairs=8,
        min_eval=4,
        n_random=2,
        n_boot=50,
        n_hand_check=10,
        batch_size=4,
        max_new_tokens=4,
        coeffs=(-0.2, 0.0, 0.2),
        eval_types=("obscure_real", "fictional_generated"),
    )
    return dataclasses.replace(base, **overrides)


def _run_dir(tmp_path: Path, config: Config) -> RunDir:
    return RunDir.create(config, out_root=tmp_path)


def _uids(items: Sequence[LabeledItem]) -> list[str]:
    return [i.example.uid for i in items]


@pytest.mark.parametrize("n_shots", [0, 15])
def test_the_shot_count_accepts_its_bounds(n_shots: int) -> None:
    assert _config(n_shots=n_shots).n_shots == n_shots


def test_the_shot_count_range_is_left_to_the_labels_config() -> None:
    assert _config(n_shots=16).n_shots == 16


def test_config_defaults_and_round_trip() -> None:
    config = Config(labels_run="runs/p9a")

    assert config.experiment == "p9d"
    assert config.model == "qwen3-1.7b"
    assert config.template == "fewshot"
    assert config.n_shots == 15
    assert config.direction_type == "obscure_real"
    assert config.eval_types == (
        "obscure_real",
        "fictional_generated",
        "unknowable_detail",
        "false_premise",
    )
    assert config.method == "logistic"
    assert config.preprocess == "center"
    assert config.n_pairs == 64
    assert config.n_random == 3
    assert config.min_pairs == 16
    assert config.min_eval == 10
    assert config.behavior == "auto"
    assert config.instruction.startswith("Answer the question in a few words.")
    assert config.n_hand_check == 100
    assert config.n_boot == 1000
    assert config.n_prompts == 100
    assert config.max_new_tokens == 24
    assert config.temperature is None
    assert config.top_p == 1.0
    assert config.num_samples == 1
    assert config.coeffs == (-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4)
    assert config.mode == "relative"
    assert config.mask == "response"
    assert Config.from_dict(config.to_dict()) == config


def test_labels_run_is_required() -> None:
    with pytest.raises(ValueError, match="labels_run"):
        Config.from_dict({"experiment": "p9d"})


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"coeffs": (-0.2, 0.2)}, "0.0"),
        ({"coeffs": (0.0, float("nan"))}, "finite"),
        ({"coeffs": (0.0,)}, "distinct"),
        ({"coeffs": (0.0, 0.0)}, "distinct"),
        ({"n_random": 0}, "n_random"),
        ({"n_prompts": 0}, "n_prompts"),
        ({"behavior": "sometimes"}, "behavior"),
        ({"min_eval": 0}, "min_eval"),
        ({"min_pairs": 0}, "min_pairs"),
        ({"n_pairs": 0}, "n_pairs"),
        ({"num_samples": 2}, "needs a temperature"),
        ({"temperature": 0.0}, "temperature"),
        ({"temperature": -0.5}, "temperature"),
        ({"temperature": 0.8, "top_p": 0.0}, "top_p"),
        ({"temperature": 0.8, "top_p": 1.5}, "top_p"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"batch_size": 0}, "batch_size"),
        ({"mode": "fractional"}, "mode must be one of"),
        ({"op": "cubic"}, "op must be one of"),
        ({"mask": "first"}, "mask must be one of"),
    ],
    ids=[
        "no_zero",
        "nan_coeff",
        "single_coeff",
        "repeated_coeff",
        "no_random",
        "no_prompts",
        "bad_behavior",
        "no_eval_items",
        "no_pairs_minimum",
        "no_pairs",
        "samples_without_temperature",
        "zero_temperature",
        "negative_temperature",
        "zero_top_p",
        "large_top_p",
        "no_new_tokens",
        "no_batch",
        "bad_mode",
        "bad_operator",
        "bad_mask",
    ],
)
def test_config_rejects_settings_that_would_fail_late(
    overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _config(**overrides)


@pytest.mark.parametrize(
    "name", ["n_random", "n_prompts", "n_pairs", "min_pairs", "min_eval"]
)
def test_config_accepts_a_count_of_one(name: str) -> None:
    assert getattr(_config(**{name: 1}), name) == 1


@pytest.mark.parametrize("behavior", ["auto", "abstention", "shift"])
def test_every_behavior_rejects_the_same_generation_settings(
    behavior: str,
) -> None:
    with pytest.raises(ValueError, match="needs a temperature"):
        _config(behavior=behavior, num_samples=2)


def test_a_closed_gate_raises_before_any_model_work(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("model work before the gate")

    monkeypatch.setattr(p9d_steering, "collect_item_activations", forbidden)
    config = _config()

    with pytest.raises(ValueError, match="known_contrasts"):
        execute(
            config,
            _run_dir(tmp_path, config),
            tiny_arch,
            _items(),
            known_contrasts=False,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"direction_type": "real_vs_generated"},
        {"eval_types": ("obscure_real", "pre_vs_post_cutoff")},
    ],
    ids=["direction", "evaluation"],
)
def test_an_unknown_type_contrast_is_refused(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(
        ValueError, match="known versus unknown contrasts only, not"
    ):
        plan_groups(_config(**overrides), _items(), True)


def test_post_cutoff_is_selectable_though_not_a_default() -> None:
    config = _config(eval_types=("obscure_real", "post_cutoff"))

    plan = plan_groups(config, _items(), True)

    assert "post_cutoff" not in Config(labels_run="runs/p9a").eval_types
    assert plan.skipped == {"post_cutoff": "no such type"}


def test_the_direction_contrast_must_exist() -> None:
    with pytest.raises(ValueError, match=r"post_cutoff.*available"):
        plan_groups(_config(direction_type="post_cutoff"), _items(), True)


def test_the_direction_contrast_must_pass_the_pool_check() -> None:
    with pytest.raises(ValueError, match="fewer than 1000 train items"):
        plan_groups(_config(min_pairs=1000), _items(), True)


def test_the_evaluation_groups_follow_the_calibration_group() -> None:
    plan = plan_groups(_config(), _items(), True)

    assert plan.groups[0].name == "calibration"
    assert plan.evaluation == plan.groups[1:]
    assert plan.evaluation


def test_a_contrast_without_enough_test_items_is_skipped_and_named() -> None:
    config = _config(
        eval_types=("obscure_real", "missing", "unknowable_detail")
    )

    plan = plan_groups(config, _items(n_detail=6), True)

    assert [g.name for g in plan.groups] == ["calibration", "obscure_real"]
    assert plan.skipped == {
        "missing": "no such type",
        "unknowable_detail": "fewer than 4 test items on one side",
    }


def test_without_a_usable_evaluation_contrast_the_plan_raises() -> None:
    with pytest.raises(ValueError, match="no usable evaluation contrast"):
        plan_groups(_config(eval_types=("missing",)), _items(), True)


def test_the_groups_take_the_first_prompts_of_each_side_by_uid() -> None:
    items = _items()
    config = _config(
        n_prompts=3, eval_types=("obscure_real", "unknowable_detail")
    )
    pools = type_pools(items)

    plan = plan_groups(config, items, True)

    names = [g.name for g in plan.groups]
    assert names == ["calibration", "obscure_real", "unknowable_detail"]
    val_known, val_unknown = pools["obscure_real"].split("val")
    test_known, test_unknown = pools["obscure_real"].split("test")
    assert _uids(plan.groups[0].known) == sorted(_uids(val_known))[:3]
    assert _uids(plan.groups[0].unknown) == sorted(_uids(val_unknown))[:3]
    assert _uids(plan.groups[1].known) == sorted(_uids(test_known))[:3]
    assert _uids(plan.groups[1].unknown) == sorted(_uids(test_unknown))[:3]
    detail_known, detail_unknown = pools["unknowable_detail"].split("test")
    assert _uids(plan.groups[2].known) == sorted(_uids(detail_known))[:3]
    assert _uids(plan.groups[2].unknown) == sorted(_uids(detail_unknown))[:3]
    assert plan.direction.name == "obscure_real"
    assert plan.skipped == {}


def test_the_calibration_and_evaluation_items_never_overlap() -> None:
    plan = plan_groups(_config(n_prompts=50), _items(), True)

    calibration, evaluation = plan.groups[0], plan.groups[1]
    assert not set(_uids(calibration.known)) & set(_uids(evaluation.known))
    assert not set(_uids(calibration.unknown)) & set(_uids(evaluation.unknown))


def test_auto_is_abstention_only_for_a_post_trained_checkpoint(
    tiny_qwen3: LoadedModel,
) -> None:
    post_trained = dataclasses.replace(
        tiny_qwen3, checkpoint=MODELS["qwen3-1.7b"]
    )
    base = dataclasses.replace(tiny_qwen3, checkpoint=MODELS["qwen3-1.7b-base"])

    assert resolve_mode("auto", tiny_qwen3) == "shift"
    assert resolve_mode("auto", base) == "shift"
    assert resolve_mode("auto", post_trained) == "abstention"
    assert resolve_mode("shift", post_trained) == "shift"
    assert resolve_mode("abstention", base) == "abstention"


def test_abstention_needs_a_chat_template(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    config = _config(behavior="abstention")

    with pytest.raises(ValueError, match="needs a tokenizer with a chat templ"):
        execute(
            config,
            _run_dir(tmp_path, config),
            tiny_gpt2,
            _items(),
            known_contrasts=True,
        )


@pytest.mark.parametrize("layers", [(0,), (4,), (1, 9)])
def test_steered_layers_must_exist(
    tiny_arch: LoadedModel, tmp_path: Path, layers: tuple[int, ...]
) -> None:
    config = _config(layers=layers)

    with pytest.raises(ValueError, match=r"must lie in 1\.\.3"):
        execute(
            config,
            _run_dir(tmp_path, config),
            tiny_arch,
            _items(),
            known_contrasts=True,
        )


SHIFT_TYPES = ("obscure_real", "fictional_generated", "unknowable_detail")
METRICS = ("entropy", "answer_logprob")


def _execute(tiny: LoadedModel, out: Path, **overrides: Any) -> RunDir:
    config = _config(**overrides)
    run_dir = _run_dir(out, config)
    execute(config, run_dir, tiny, _items(), known_contrasts=True)
    return run_dir


_SHIFT_RUNS: dict[str, RunDir] = {}


@pytest.fixture
def shift_run(
    tiny_arch: LoadedModel, tmp_path_factory: pytest.TempPathFactory
) -> RunDir:
    """One shift run per architecture with three evaluation contrasts."""
    key = type(tiny_arch.model).__name__
    if key not in _SHIFT_RUNS:
        out = tmp_path_factory.mktemp(f"p9d-shift-{key}")
        _SHIFT_RUNS[key] = _execute(tiny_arch, out, eval_types=SHIFT_TYPES)
    return _SHIFT_RUNS[key]


@dataclasses.dataclass
class _Calls:
    stats: list[dict[str, Any]]
    norms: list[dict[str, Any]]


@pytest.fixture
def spies(monkeypatch: pytest.MonkeyPatch) -> _Calls:
    """Records the arguments of the model calls and runs them for real."""
    calls = _Calls([], [])
    real_stats = p9d_steering.next_token_stats
    real_norms = p9d_steering.mean_residual_norms

    def stats(
        loaded: LoadedModel, prompts: Sequence[str], **kwargs: Any
    ) -> Any:
        calls.stats.append({"prompts": list(prompts), **kwargs})
        return real_stats(loaded, prompts, **kwargs)

    def norms(
        loaded: LoadedModel,
        texts: Sequence[str],
        layers: Sequence[int],
        **kwargs: Any,
    ) -> dict[int, float]:
        result = real_norms(loaded, texts, layers, **kwargs)
        calls.norms.append(
            {"texts": list(texts), "layers": list(layers), "result": result}
            | kwargs
        )
        return result

    monkeypatch.setattr(p9d_steering, "next_token_stats", stats)
    monkeypatch.setattr(p9d_steering, "mean_residual_norms", norms)
    return calls


def _summary(run_dir: RunDir) -> dict[str, Any]:
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    return summary


def _table(run_dir: RunDir) -> tuple[list[str], list[list[Any]]]:
    table = run_dir.read_json("table")
    assert isinstance(table, dict)
    return table["columns"], table["rows"]


def _files(run_dir: RunDir) -> list[str]:
    return sorted(p.name for p in run_dir.path.iterdir() if p.is_file())


def _both(
    pool_split: tuple[list[LabeledItem], list[LabeledItem]],
) -> list[LabeledItem]:
    return [i for side in pool_split for i in side]


def test_shift_mode_writes_a_curve_pair_per_contrast_and_metric(
    shift_run: RunDir,
) -> None:
    stems = [f"curves_{t}_{m}" for t in SHIFT_TYPES for m in METRICS]

    assert _files(shift_run) == sorted(
        [
            "config.json",
            "meta.json",
            "summary.json",
            "table.json",
            "table_dose.json",
            "table_selectivity.json",
            *(f"{stem}.{ext}" for stem in stems for ext in ("json", "npz")),
        ]
    )


def test_shift_table_has_a_row_per_cell_with_a_finite_pair(
    shift_run: RunDir,
) -> None:
    columns, rows = _table(shift_run)

    assert columns == [
        "type",
        "side",
        "arm",
        "coeff",
        "metric",
        "shift",
        "lo",
        "hi",
    ]
    assert len(rows) == 126
    assert {r[0] for r in rows} == {"calibration", *SHIFT_TYPES}
    assert {r[1] for r in rows} == {"known", "unknown"}
    assert {r[2] for r in rows} == {"lat0", "random0", "random1"}
    assert {r[3] for r in rows} == {-0.2, 0.0, 0.2}
    cells = {(r[0], r[1], r[4]) for r in rows}
    assert ("fictional_generated", "known", "answer_logprob") in cells
    assert ("fictional_generated", "unknown", "answer_logprob") not in cells
    assert ("unknowable_detail", "unknown", "answer_logprob") not in cells
    assert ("unknowable_detail", "unknown", "entropy") in cells


def test_a_zero_coefficient_shifts_nothing_and_a_nonzero_one_does(
    shift_run: RunDir,
) -> None:
    _, rows = _table(shift_run)

    zero = [r for r in rows if r[3] == 0.0]
    assert zero
    assert max(abs(v) for r in zero for v in r[5:]) < 1e-4
    moved = [r for r in rows if r[3] != 0.0 and r[4] == "entropy"]
    assert max(abs(r[5]) for r in moved) > 0.01


def test_shift_curves_hold_four_series_and_drop_a_missing_one(
    shift_run: RunDir,
) -> None:
    full = shift_run.read_json("curves_obscure_real_entropy")
    arrays = shift_run.read_arrays("curves_obscure_real_entropy")
    partial = shift_run.read_json("curves_fictional_generated_answer_logprob")

    assert full["series"] == [
        "lat: known",
        "lat: unknown",
        "random: known",
        "random: unknown",
    ]
    assert (full["xlabel"], full["ylabel"]) == (
        "Relative coefficient",
        "Shift in entropy",
    )
    assert arrays["x"].tolist() == [-0.2, 0.0, 0.2]
    assert arrays["mean"].shape == (4, 3)
    assert np.all(arrays["lo"] <= arrays["hi"])
    assert partial["series"] == ["lat: known", "random: known"]
    assert partial["ylabel"] == "Shift in answer_logprob"


def test_shift_curves_are_the_table_cells(shift_run: RunDir) -> None:
    _, rows = _table(shift_run)
    arrays = shift_run.read_arrays("curves_obscure_real_entropy")

    def cell(side: str, arm: str, coeff: float) -> list[float]:
        (row,) = (
            r
            for r in rows
            if r[:5] == ["obscure_real", side, arm, coeff, "entropy"]
        )
        return row[5:]

    for k, coeff in enumerate((-0.2, 0.0, 0.2)):
        shift, lo, hi = cell("known", "lat0", coeff)
        assert arrays["mean"][0, k] == pytest.approx(shift)
        assert arrays["lo"][0, k] == pytest.approx(lo)
        assert arrays["hi"][0, k] == pytest.approx(hi)
        readers = [cell("unknown", f"random{r}", coeff)[0] for r in range(2)]
        assert arrays["mean"][3, k] == pytest.approx(np.mean(readers))
        assert arrays["lo"][3, k] == pytest.approx(min(readers))
        assert arrays["hi"][3, k] == pytest.approx(max(readers))


def test_shift_summary(shift_run: RunDir, tiny_arch: LoadedModel) -> None:
    summary = _summary(shift_run)
    plan = plan_groups(_config(eval_types=SHIFT_TYPES), _items(), True)

    val = summary["direction"]["val_auroc"]
    assert summary["mode"] == "shift"
    assert summary["layer_source"] == "validation"
    assert summary["direction"]["type"] == "obscure_real"
    assert len(val) == tiny_arch.n_layers
    assert summary["layers"] == [1 + int(np.argmax(val))]
    assert summary["coeffs"] == [-0.2, 0.0, 0.2]
    assert summary["types"] == list(SHIFT_TYPES)
    assert summary["skipped"] == {}
    assert summary["n_with_answer"] == {
        g.name: {
            "known": sum(bool(i.example.answers) for i in g.known),
            "unknown": sum(bool(i.example.answers) for i in g.unknown),
        }
        for g in plan.groups
    }
    assert summary["n_with_answer"]["fictional_generated"]["unknown"] == 0
    assert "chosen_coeff" not in summary
    assert "hand_check_rows" not in summary


def test_a_configured_layer_is_the_steered_layer(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(tiny_arch, tmp_path, layers=(2,))

    summary = _summary(run_dir)
    assert summary["layers"] == [2]
    assert summary["layer_source"] == "config"
    assert len(summary["direction"]["val_auroc"]) == tiny_arch.n_layers


def test_a_shift_is_the_steered_value_minus_the_unsteered_one(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = {"lat0": 1.0, "random0": 2.0, "random1": 3.0, "random2": 10.0}
    seen: dict[bytes, str] = {}

    def fake_stats(
        loaded: LoadedModel, prompts: Sequence[str], **kwargs: Any
    ) -> TokenStats:
        entropy, logprob = 1.0, -2.0
        controller = kwargs["edits"]
        if controller is not None:
            vector = next(iter(controller.vectors.values())).numpy()
            if vector.tobytes() not in seen:
                seen[vector.tobytes()] = list(weight)[len(seen)]
            arm = seen[vector.tobytes()]
            entropy += controller.coeff * weight[arm]
            logprob -= controller.coeff * weight[arm]
        n = len(prompts)
        return TokenStats(np.zeros(n), np.full(n, entropy), np.full(n, logprob))

    monkeypatch.setattr(p9d_steering, "next_token_stats", fake_stats)
    config = _config(eval_types=SHIFT_TYPES, n_random=3)
    run_dir = _run_dir(tmp_path, config)
    execute(config, run_dir, tiny_arch, _items(), known_contrasts=True)

    _, rows = _table(run_dir)
    assert len(rows) == 168
    for _, _, arm, coeff, metric, *cell in rows:
        sign = 1.0 if metric == "entropy" else -1.0
        assert cell == pytest.approx([sign * coeff * weight[arm]] * 3)
    arrays = run_dir.read_arrays("curves_obscure_real_entropy")
    scale = np.array([-0.2, 0.0, 0.2])
    assert arrays["mean"] == pytest.approx(
        np.stack([1 * scale, 1 * scale, 5 * scale, 5 * scale])
    )
    assert arrays["lo"][2] == pytest.approx(np.minimum(2 * scale, 10 * scale))
    assert arrays["hi"][2] == pytest.approx(np.maximum(2 * scale, 10 * scale))


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "mask": "all",
            "preserve_norm": True,
            "layers": (2,),
            "op": "piecewise",
        },
        {"mode": "absolute", "layers": (1, 3), "bos": "none"},
    ],
    ids=["all_preserve", "absolute"],
)
def test_the_controllers_carry_the_config_settings(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    spies: _Calls,
    overrides: dict[str, Any],
) -> None:
    config = _config(coeffs=(0.0, 0.3), **overrides)
    run_dir = _run_dir(tmp_path, config)
    execute(config, run_dir, tiny_arch, _items(), known_contrasts=True)

    base, *steered = spies.stats
    controllers = [c["edits"] for c in steered]
    (norms,) = spies.norms
    assert base.get("edits") is None
    assert len(controllers) == 6
    assert [c.coeff for c in controllers] == [0.0, 0.3] * 3
    for controller in controllers:
        assert list(controller.vectors) == list(config.layers)
        assert controller.op == config.op
        assert controller.mode == config.mode
        assert controller.mask == config.mask
        assert controller.preserve_norm == config.preserve_norm
        assert controller.norms == norms["result"]
    assert norms["layers"] == list(config.layers)
    assert all(c["batch_size"] == 4 for c in spies.stats)
    assert all(c["bos"] == config.bos for c in spies.stats)
    assert (norms["bos"], norms["batch_size"]) == (config.bos, 4)


def test_the_norms_come_from_held_out_train_prompts(
    tiny_arch: LoadedModel, tmp_path: Path, spies: _Calls
) -> None:
    train = _both(type_pools(_items())["obscure_real"].split("train"))
    _execute(tiny_arch, tmp_path, layers=(2,))

    first = sorted(train, key=lambda i: i.example.uid)[:10]
    (norms,) = spies.norms
    assert norms["texts"] == [
        few_shot_prompt(i.example.question, DEFAULT_SHOTS[:2]) for i in first
    ]
    assert norms["layers"] == [2]


def test_the_random_arms_are_unit_directions_drawn_from_the_seed(
    tiny_arch: LoadedModel, tmp_path: Path, spies: _Calls
) -> None:
    _execute(tiny_arch, tmp_path, seed=7, layers=(2,), coeffs=(0.0, 0.2))

    _, *steered = spies.stats
    vectors = [c["edits"].vectors[2].numpy() for c in steered]
    rng = np.random.default_rng(7)
    for k in range(2):
        draw = rng.standard_normal((tiny_arch.n_layers, tiny_arch.d_model))[1]
        expected = draw / np.linalg.norm(draw)
        assert vectors[2 * (k + 1)] == pytest.approx(expected, abs=1e-6)
        assert vectors[2 * (k + 1) + 1] == pytest.approx(expected, abs=1e-6)
    for vector in vectors:
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"method": "mean_diff", "preprocess": "unit", "seed": 7, "bos": "none"},
    ],
    ids=["logistic", "mean_diff"],
)
def test_the_steering_direction_is_fitted_on_the_train_items(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    spies: _Calls,
    overrides: dict[str, Any],
) -> None:
    config = _config(layers=(2,), coeffs=(0.0, 0.2), n_pairs=6, **overrides)
    pool = type_pools(_items())["obscure_real"]
    run_dir = _run_dir(tmp_path, config)
    execute(config, run_dir, tiny_arch, _items(), known_contrasts=True)
    acts = collect_item_activations(
        tiny_arch,
        [*_both(pool.split("train")), *_both(pool.split("val"))],
        p9_template("fewshot", DEFAULT_SHOTS[:2]),
        chat_format_for(tiny_arch),
        layers=tuple(range(1, tiny_arch.n_layers + 1)),
        bos=config.bos,
        batch_size=4,
    )
    reader = fit_type_reader(
        acts,
        pool,
        method=config.method,
        preprocess=config.preprocess,
        n_pairs=config.n_pairs,
        seed=config.seed,
    )

    steered = spies.stats[1]["edits"].vectors[2].numpy()
    assert steered == pytest.approx(
        reader.signs[1] * reader.directions[1], abs=1e-5
    )
    val = type_auroc(reader, acts, pool, "val")
    assert _summary(run_dir)["direction"]["val_auroc"] == pytest.approx(val)


def test_each_distinct_prompt_is_run_once_with_its_gold_token(
    tiny_arch: LoadedModel, tmp_path: Path, spies: _Calls
) -> None:
    config = _config(eval_types=SHIFT_TYPES)
    plan = plan_groups(config, _items(), True)
    _execute(tiny_arch, tmp_path, eval_types=SHIFT_TYPES)

    unique = {
        i.example.uid: i for g in plan.groups for i in (*g.known, *g.unknown)
    }
    expected = {
        few_shot_prompt(i.example.question, DEFAULT_SHOTS[:2]): (
            first_answer_token(tiny_arch, i.example.answers[0])
            if i.example.answers
            else 0
        )
        for i in unique.values()
    }
    for call in spies.stats:
        pairs = zip(call["prompts"], call["targets"], strict=True)
        assert len(call["prompts"]) == len(unique)
        assert dict(pairs) == expected


ABSTAIN = "I do not know."


@dataclasses.dataclass
class _ArmCalls:
    arms: list[dict[str, Any]]
    prompts: list[list[str]]


def _chat_prompt(loaded: LoadedModel, config: Config, item: LabeledItem) -> str:
    chat = ChatFormat.from_tokenizer(loaded.tokenizer)
    return chat.render_turn(
        f"{config.instruction}\n\nQuestion: {item.example.question}"
    )[0]


def _canned_run(
    loaded: LoadedModel,
    out: Path,
    monkeypatch: pytest.MonkeyPatch,
    rule: Callable[[LabeledItem, int, float], list[str]],
    **overrides: Any,
) -> tuple[RunDir, _ArmCalls]:
    """Runs abstention mode against arms that answer by `rule`.

    The rule gets the item, the arm (0 is the steering direction, then the
    random ones in order) and the coefficient, and returns the samples.
    """
    config = _config(behavior="abstention", **overrides)
    items = _items()
    by_prompt = {_chat_prompt(loaded, config, i): i for i in items}
    calls = _ArmCalls([], [])
    seen: dict[int, int] = {}

    def fake_arm(
        loaded: LoadedModel,
        settings: GenSettings,
        reader: Any,
        *,
        layers: Sequence[int],
        coeff: float,
        **kwargs: Any,
    ) -> Arm:
        index = seen.setdefault(id(reader), len(seen))
        calls.arms.append(
            {
                "index": index,
                "reader": reader,
                "settings": settings,
                "layers": layers,
                "coeff": coeff,
                **kwargs,
            }
        )

        def generate(prompts: Sequence[str]) -> list[list[Completion]]:
            calls.prompts.append(list(prompts))
            return [
                [
                    Completion(t, (), True)
                    for t in rule(by_prompt[p], index, coeff)
                ]
                for p in prompts
            ]

        return Arm(name=f"{coeff:g}", generate=generate)

    monkeypatch.setattr(p9d_steering, "reader_arm", fake_arm)
    run_dir = _run_dir(out, config)
    execute(config, run_dir, loaded, items, known_contrasts=True)
    return run_dir, calls


def _is_known(item: LabeledItem) -> bool:
    return item.label.label == "known"


def _answer(item: LabeledItem) -> list[str]:
    return ["gold" if _is_known(item) else "other"]


def _planted(item: LabeledItem, arm: int, coeff: float) -> list[str]:
    """Arm 0 refuses known items at -0.2 and unknown ones at 0.2.

    Arm 1, a random one, refuses known items at 0.2.
    """
    refusing = {(0, -0.2, True), (0, 0.2, False), (1, 0.2, True)}
    if (arm, coeff, _is_known(item)) in refusing:
        return [ABSTAIN]
    return _answer(item)


def _row(
    rows: list[list[Any]], columns: list[str], **match: Any
) -> dict[str, Any]:
    found = [
        dict(zip(columns, r, strict=True))
        for r in rows
        if all(
            dict(zip(columns, r, strict=True))[k] == v for k, v in match.items()
        )
    ]
    assert len(found) == 1, (match, len(found))
    return found[0]


def test_abstention_rates_per_group_arm_and_coefficient(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _canned_run(tiny_qwen3, tmp_path, monkeypatch, _planted)

    columns, rows = _table(run_dir)

    assert columns == [
        "type",
        "arm",
        "coeff",
        "false_refusal",
        "correct_refusal",
        "coverage",
        "wrong_among_answered",
        "balanced",
    ]
    assert len(rows) == 27
    assert {r[0] for r in rows} == {
        "calibration",
        "obscure_real",
        "fictional_generated",
    }
    assert {r[1] for r in rows} == {"lat0", "random0", "random1"}
    expected = {
        ("lat0", -0.2): (1.0, 0.0, 0.5, 1.0, 0.0),
        ("lat0", 0.2): (0.0, 1.0, 0.5, 0.0, 1.0),
        ("lat0", 0.0): (0.0, 0.0, 1.0, 0.5, 0.5),
        ("random0", 0.2): (1.0, 0.0, 0.5, 1.0, 0.0),
        ("random1", 0.2): (0.0, 0.0, 1.0, 0.5, 0.5),
    }
    for name in ("calibration", "obscure_real", "fictional_generated"):
        for (arm, coeff), values in expected.items():
            row = _row(rows, columns, type=name, arm=arm, coeff=coeff)
            assert tuple(row[c] for c in columns[3:]) == pytest.approx(values)


def test_the_chosen_coefficient_and_the_paired_effects(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _canned_run(tiny_qwen3, tmp_path, monkeypatch, _planted)

    summary = _summary(run_dir)
    table = run_dir.read_json("table_effect")
    assert isinstance(table, dict)
    columns, rows = table["columns"], table["rows"]
    assert (summary["chosen_coeff"], summary["chosen_index"]) == (0.2, 2)
    assert columns == [
        "type",
        "side",
        "arm",
        "coeff",
        "n",
        "abstention_shift",
        "lo",
        "hi",
    ]
    assert len(rows) == 12
    assert {r[0] for r in rows} == {"obscure_real", "fictional_generated"}
    assert {r[2] for r in rows} == {"lat0", "random0", "random1"}
    assert {r[3] for r in rows} == {0.2}
    shifts = {
        ("known", "lat0"): 0.0,
        ("unknown", "lat0"): 1.0,
        ("known", "random0"): 1.0,
        ("unknown", "random0"): 0.0,
        ("known", "random1"): 0.0,
        ("unknown", "random1"): 0.0,
    }
    for (side, arm), shift in shifts.items():
        row = _row(rows, columns, type="obscure_real", side=side, arm=arm)
        assert row["n"] == 4
        assert row["abstention_shift"] == pytest.approx(shift)
        assert row["lo"] <= shift <= row["hi"]


def test_the_coefficient_is_chosen_on_the_calibration_items_only(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def rule(item: LabeledItem, arm: int, coeff: float) -> list[str]:
        best = -0.2 if item.example.split == "val" else 0.2
        if arm == 0 and coeff == best and not _is_known(item):
            return [ABSTAIN]
        return _answer(item)

    run_dir, _ = _canned_run(tiny_qwen3, tmp_path, monkeypatch, rule)

    summary = _summary(run_dir)
    table = run_dir.read_json("table_effect")
    assert isinstance(table, dict)
    row = _row(
        table["rows"],
        table["columns"],
        type="obscure_real",
        side="unknown",
        arm="lat0",
    )
    assert (summary["chosen_coeff"], summary["chosen_index"]) == (-0.2, 0)
    assert row["coeff"] == -0.2
    assert row["abstention_shift"] == 0.0


def test_a_direction_that_changes_nothing_chooses_zero(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _canned_run(
        tiny_qwen3, tmp_path, monkeypatch, lambda i, a, c: _answer(i)
    )

    summary = _summary(run_dir)
    table = run_dir.read_json("table_effect")
    assert isinstance(table, dict)
    assert (summary["chosen_coeff"], summary["chosen_index"]) == (0.0, 1)
    assert {r[3] for r in table["rows"]} == {0.0}
    assert {r[5] for r in table["rows"]} == {0.0}


def test_abstention_curves_hold_rates_with_wilson_and_random_bands(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _canned_run(tiny_qwen3, tmp_path, monkeypatch, _planted)

    sidecar = run_dir.read_json("curves_obscure_real")
    arrays = run_dir.read_arrays("curves_obscure_real")
    assert isinstance(sidecar, dict)
    assert sidecar["series"] == [
        "lat: known",
        "lat: unknown",
        "random: known",
        "random: unknown",
    ]
    assert (sidecar["xlabel"], sidecar["ylabel"]) == (
        "Relative coefficient",
        "Abstention rate",
    )
    assert arrays["x"].tolist() == [-0.2, 0.0, 0.2]
    assert arrays["mean"].tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.0],
    ]
    assert arrays["lo"][0, 0] == pytest.approx(wilson_ci(4, 4)[0])
    assert arrays["lo"][1, 2] == pytest.approx(wilson_ci(4, 4)[0])
    assert arrays["hi"][1, 0] == pytest.approx(wilson_ci(0, 4)[1])
    assert arrays["lo"][2].tolist() == [0.0, 0.0, 0.0]
    assert arrays["hi"][2].tolist() == [0.0, 0.0, 1.0]
    assert sorted(p.name for p in run_dir.path.glob("curves_*.json")) == [
        "curves_fictional_generated.json",
        "curves_obscure_real.json",
    ]


def test_the_hand_check_alternates_between_abstained_and_answered(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _canned_run(tiny_qwen3, tmp_path, monkeypatch, _planted)

    with (run_dir.path / "hand_check.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    questions = {i.example.uid: i.example.question for i in _items()}
    known = {i.example.uid: _is_known(i) for i in _items()}
    assert list(rows[0]) == list(CHECK_COLUMNS)
    assert len(rows) == 10 == _summary(run_dir)["hand_check_rows"]
    assert [r["abstained"] for r in rows] == ["1", "0"] * 5
    assert {r["coeff"] for r in rows} <= {"-0.2", "0", "0.2"}
    assert {r["human_label"] for r in rows} == {""}
    for row in rows:
        assert row["question"] == questions[row["uid"]]
        assert row["completion"] in {ABSTAIN, "gold", "other"}
        assert row["abstained"] == str(int(row["completion"] == ABSTAIN))
        assert row["correct"] == str(int(row["completion"] == "gold"))
        if row["completion"] == ABSTAIN:
            assert (row["coeff"], known[row["uid"]]) in {
                ("-0.2", True),
                ("0.2", False),
            }


def test_the_abstention_x_label_names_the_coefficient_mode(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _canned_run(
        tiny_qwen3, tmp_path, monkeypatch, _planted, mode="absolute"
    )

    sidecar = run_dir.read_json("curves_obscure_real")
    assert isinstance(sidecar, dict)
    assert sidecar["xlabel"] == "Coefficient"


def test_the_shift_x_label_names_the_coefficient_mode(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(
        tiny_arch, tmp_path, eval_types=SHIFT_TYPES, mode="absolute"
    )

    sidecar = run_dir.read_json("curves_obscure_real_entropy")
    assert isinstance(sidecar, dict)
    assert sidecar["xlabel"] == "Coefficient"


def test_the_arms_are_built_with_the_config_settings(
    tiny_qwen3: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spies: _Calls,
) -> None:
    config = _config(behavior="abstention", mask="all", preserve_norm=True)
    plan = plan_groups(config, _items(), True)
    run_dir, calls = _canned_run(
        tiny_qwen3,
        tmp_path,
        monkeypatch,
        _planted,
        mask="all",
        preserve_norm=True,
    )

    unique = {
        i.example.uid: i for g in plan.groups for i in (*g.known, *g.unknown)
    }
    expected = sorted(
        _chat_prompt(tiny_qwen3, config, i) for i in unique.values()
    )
    (norms,) = spies.norms
    assert [c["index"] for c in calls.arms] == [0] * 3 + [1] * 2 + [2] * 2
    assert [c["coeff"] for c in calls.arms] == [-0.2, 0.0, 0.2] + [
        -0.2,
        0.2,
    ] * 2
    assert [c["reader"].method for c in calls.arms] == ["logistic"] * 3 + [
        "random"
    ] * 4
    for call in calls.arms:
        assert call["settings"] == GenSettings.from_config(config)
        assert tuple(call["layers"]) == tuple(_summary(run_dir)["layers"])
        assert call["norms"] == norms["result"]
        assert (call["op"], call["mode"], call["mask"]) == (
            config.op,
            config.mode,
            "all",
        )
        assert call["preserve_norm"] is True
    assert len(calls.prompts) == 7
    for prompts in calls.prompts:
        assert sorted(prompts) == expected


def test_the_random_arms_share_the_unsteered_rates(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _canned_run(tiny_qwen3, tmp_path, monkeypatch, _planted)

    columns, rows = _table(run_dir)
    for name in ("calibration", "obscure_real", "fictional_generated"):
        lat = _row(rows, columns, type=name, arm="lat0", coeff=0.0)
        for arm in ("random0", "random1"):
            row = _row(rows, columns, type=name, arm=arm, coeff=0.0)
            assert row == {**lat, "arm": arm}


@pytest.mark.parametrize("temperature", [None, 0.8], ids=["greedy", "sampled"])
def test_a_zero_coefficient_generates_the_same_text_for_any_direction(
    tiny_qwen3: LoadedModel, temperature: float | None
) -> None:
    rng = np.random.default_rng(0)
    layers = (1, 2)
    settings = GenSettings(
        max_new_tokens=6,
        temperature=temperature,
        num_samples=1 if temperature is None else 2,
        batch_size=2,
    )
    prompts = ["Q: Who wrote it?\nA:", "Q: Where is it?\nA:"]

    texts = []
    for _ in range(2):
        reader = RepReader.from_arrays(
            "random",
            layers,
            rng.standard_normal((2, tiny_qwen3.d_model)),
            np.zeros((2, tiny_qwen3.d_model)),
        )
        arm = reader_arm(
            tiny_qwen3,
            settings,
            reader,
            layers=layers,
            coeff=0.0,
            norms={1: 2.0, 2: 3.0},
        )
        texts.append([[c.text for c in row] for row in arm.generate(prompts)])

    assert texts[0] == texts[1]


def test_the_norms_are_measured_on_the_chat_prompts_of_the_train_items(
    tiny_qwen3: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spies: _Calls,
) -> None:
    config = _config(behavior="abstention")
    train = _both(type_pools(_items())["obscure_real"].split("train"))
    _canned_run(tiny_qwen3, tmp_path, monkeypatch, _planted)

    first = sorted(train, key=lambda i: i.example.uid)[:10]
    (norms,) = spies.norms
    assert norms["texts"] == [
        _chat_prompt(tiny_qwen3, config, i) for i in first
    ]


def test_only_the_first_sample_is_classified(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def rule(item: LabeledItem, arm: int, coeff: float) -> list[str]:
        return [ABSTAIN, *_answer(item)]

    run_dir, _ = _canned_run(
        tiny_qwen3,
        tmp_path,
        monkeypatch,
        rule,
        num_samples=2,
        temperature=0.8,
    )

    columns, rows = _table(run_dir)
    row = _row(rows, columns, type="calibration", arm="lat0", coeff=0.0)
    assert (row["false_refusal"], row["correct_refusal"]) == (1.0, 1.0)
    assert row["coverage"] == 0.0
    assert row["wrong_among_answered"] is None


def test_abstention_summary(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _canned_run(tiny_qwen3, tmp_path, monkeypatch, _planted)

    summary = _summary(run_dir)
    assert summary["mode"] == "abstention"
    assert summary["layer_source"] == "validation"
    assert summary["direction"]["type"] == "obscure_real"
    assert len(summary["direction"]["val_auroc"]) == tiny_qwen3.n_layers
    assert summary["layers"] == [
        1 + int(np.argmax(summary["direction"]["val_auroc"]))
    ]
    assert summary["coeffs"] == [-0.2, 0.0, 0.2]
    assert summary["types"] == ["obscure_real", "fictional_generated"]
    assert summary["skipped"] == {}
    assert "n_with_answer" not in summary
    assert summary["calibration_balanced_range"] == pytest.approx(
        {"lat0": 1.0, "random0": 0.5, "random1": 0.0}
    )


def test_abstention_runs_end_to_end_on_the_tiny_qwen3(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(
        tiny_qwen3,
        tmp_path,
        behavior="abstention",
        instruction="Say unknown if unsure.",
        eval_types=SHIFT_TYPES,
    )

    _, rows = _table(run_dir)
    table = run_dir.read_json("table_effect")
    assert isinstance(table, dict)
    assert len(rows) == 36
    assert len(table["rows"]) == 18
    for row in table["rows"]:
        assert row[6] <= row[5] <= row[7]
    assert _files(run_dir) == sorted(
        [
            "config.json",
            "meta.json",
            "summary.json",
            "table.json",
            "table_dose.json",
            "table_selectivity.json",
            "table_effect.json",
            "hand_check.csv",
            *(
                f"curves_{t}.{ext}"
                for t in SHIFT_TYPES
                for ext in ("json", "npz")
            ),
        ]
    )
    assert all(0.0 <= r[3] <= 1.0 and 0.0 <= r[4] <= 1.0 for r in rows)


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("p9d")

    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_run_reads_the_labels_and_the_gate(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = _labels_run(tmp_path, _items())
    asked: list[str] = []

    def fake_load(name: str) -> LoadedModel:
        asked.append(name)
        return tiny_arch

    monkeypatch.setattr(p9d_steering, "load_model", fake_load)
    config = _config(labels_run=str(labels.path))

    run_dir = _run_dir(tmp_path, config)
    run(config, run_dir)

    assert asked == ["tiny"]
    assert _summary(run_dir)["mode"] == "shift"


def test_run_with_a_closed_gate_raises_the_gate_error(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = _labels_run(tmp_path, _items(), known_contrasts=False)
    monkeypatch.setattr(p9d_steering, "load_model", lambda name: tiny_arch)
    config = _config(labels_run=str(labels.path))

    with pytest.raises(ValueError, match="known_contrasts"):
        run(config, _run_dir(tmp_path, config))


def test_run_refuses_labels_made_for_another_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = _labels_run(tmp_path, _items())

    def never(name: str) -> LoadedModel:
        raise AssertionError("the model must not load first")

    monkeypatch.setattr(p9d_steering, "load_model", never)
    config = _config(labels_run=str(labels.path), model="gpt2")

    with pytest.raises(ValueError, match="labeled 'tiny', not 'gpt2'"):
        run(config, _run_dir(tmp_path, config))


def test_run_refuses_labels_made_with_another_number_of_shots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = _labels_run(tmp_path, _items(), n_shots=15)

    def never(name: str) -> LoadedModel:
        raise AssertionError("the model must not load first")

    monkeypatch.setattr(p9d_steering, "load_model", never)
    config = _config(labels_run=str(labels.path), n_shots=2)

    with pytest.raises(
        ValueError, match=r"n_shots 15.*n_shots is 2.*direction is read on"
    ):
        run(config, _run_dir(tmp_path, config))


def test_run_refuses_an_unfinished_labeling_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = RunDir.create(
        LabelsConfig(model="tiny"), out_root=tmp_path / "labels"
    )
    labels.write_json("items", pool_records(_items()))
    labels.write_json("summary", {"known_contrasts": True})

    def never(name: str) -> LoadedModel:
        raise AssertionError("the model must not load first")

    monkeypatch.setattr(p9d_steering, "load_model", never)
    config = _config(labels_run=str(labels.path))

    with pytest.raises(ValueError, match="not a finished run"):
        run(config, _run_dir(tmp_path, config))


def test_plot_draws_every_curve_stem(shift_run: RunDir, tmp_path: Path) -> None:
    paths = plot(shift_run, tmp_path / "figures")

    assert len(paths) == 18
    assert all(p.is_file() for p in paths)
    assert {p.stem for p in paths} == {
        f"curves_{t}_{m}" for t in SHIFT_TYPES for m in METRICS
    }
    assert {p.suffix for p in paths} == {".pdf", ".png", ".svg"}


def test_plot_draws_the_curves_of_an_abstention_run(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(
        tiny_qwen3,
        tmp_path,
        behavior="abstention",
        instruction="Say unknown if unsure.",
    )

    paths = plot(run_dir, tmp_path / "figures")

    assert {p.stem for p in paths} == {
        "curves_obscure_real",
        "curves_fictional_generated",
    }
    assert len(paths) == 6


def test_the_bootstrap_settings_reach_every_shift_interval(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    real = p9d_steering.shift_ci

    def spy(base: Any, steered: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real(base, steered, **kwargs)

    monkeypatch.setattr(p9d_steering, "shift_ci", spy)
    slope_calls: list[dict[str, Any]] = []
    real_slope = p9d_steering.slope_ci

    def slope_spy(values: Any, reference: Any = None, **kwargs: Any) -> Any:
        slope_calls.append(kwargs)
        return real_slope(values, reference, **kwargs)

    monkeypatch.setattr(p9d_steering, "slope_ci", slope_spy)

    _execute(tiny_arch, tmp_path, seed=5, n_boot=77)

    assert calls
    assert all(c == {"n_boot": 77, "seed": 5} for c in calls)
    assert slope_calls
    assert all(c == {"n_boot": 77, "seed": 5} for c in slope_calls)


def test_the_seed_and_sizes_reach_the_helpers_of_abstention_mode(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intervals: list[dict[str, Any]] = []
    slopes: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    real_shift = p9d_steering.shift_ci
    real_slope = p9d_steering.slope_ci
    real_check = p9d_steering.write_completion_check

    def shift_spy(base: Any, steered: Any, **kwargs: Any) -> Any:
        intervals.append(kwargs)
        return real_shift(base, steered, **kwargs)

    def slope_spy(values: Any, reference: Any = None, **kwargs: Any) -> Any:
        slopes.append(kwargs)
        return real_slope(values, reference, **kwargs)

    def check_spy(path: Path, records: Any, **kwargs: Any) -> int:
        checks.append(kwargs)
        return real_check(path, records, **kwargs)

    monkeypatch.setattr(p9d_steering, "shift_ci", shift_spy)
    monkeypatch.setattr(p9d_steering, "slope_ci", slope_spy)
    monkeypatch.setattr(p9d_steering, "write_completion_check", check_spy)

    _canned_run(
        tiny_qwen3,
        tmp_path,
        monkeypatch,
        _planted,
        seed=5,
        n_boot=77,
        n_hand_check=6,
    )

    assert len(intervals) == 12
    assert all(c == {"n_boot": 77, "seed": 5} for c in intervals)
    assert len(slopes) == 20
    assert all(c == {"n_boot": 77, "seed": 5} for c in slopes)
    assert checks == [{"n": 6, "seed": 5}]


def _dose_pool(config: Config) -> set[str]:
    """The first unknown question of every group, by uid."""
    plan = plan_groups(config, _items(), True)
    return {g.unknown[0].example.uid for g in plan.groups}


DOSE_GRID = (-0.4, -0.2, 0.0, 0.2, 0.4)
DOSE_OVERRIDES: dict[str, Any] = {
    "n_prompts": 2,
    "n_random": 3,
    "n_boot": 200,
    "coeffs": DOSE_GRID,
}


def _dose_run(
    loaded: LoadedModel,
    out: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    random1_acts: bool = True,
    **overrides: Any,
) -> RunDir:
    """Runs abstention mode where the arms plant known per-item slopes.

    On the unknown items, the steering arm abstains below zero, the first
    random arm never, the second on the first item of each group below zero
    (when `random1_acts`), the third on both items above zero.
    """
    settings = {**DOSE_OVERRIDES, **overrides}
    first = _dose_pool(_config(**settings))

    def rule(item: LabeledItem, arm: int, coeff: float) -> list[str]:
        if _is_known(item):
            return _answer(item)
        refuse = {
            0: coeff < 0,
            1: False,
            2: random1_acts and item.example.uid in first and coeff < 0,
            3: coeff > 0,
        }
        return [ABSTAIN] if refuse.get(arm, False) else _answer(item)

    run_dir, _ = _canned_run(loaded, out, monkeypatch, rule, **settings)
    return run_dir


def _dose(run_dir: RunDir) -> tuple[list[str], list[list[Any]]]:
    table = run_dir.read_json("table_dose")
    assert isinstance(table, dict)
    return table["columns"], table["rows"]


def test_abstention_dose_slopes_per_arm_and_paired_contrast(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _dose_run(tiny_qwen3, tmp_path, monkeypatch)

    columns, rows = _dose(run_dir)

    assert columns == [
        "type",
        "side",
        "metric",
        "arm",
        "vs",
        "n",
        "slope",
        "lo",
        "hi",
        "span",
        "abs_gap",
        "abs_lo",
        "abs_hi",
    ]
    assert len(rows) == 28
    assert {r[2] for r in rows} == {"abstention"}
    expected = {
        ("lat0", ""): (-1.5, -1.5, -1.5),
        ("random0", ""): (0.0, 0.0, 0.0),
        ("random1", ""): (-0.75, -1.5, 0.0),
        ("random2", ""): (1.5, 1.5, 1.5),
        ("lat0", "random0"): (-1.5, -1.5, -1.5),
        ("lat0", "random1"): (-0.75, -1.5, 0.0),
        ("lat0", "random2"): (-3.0, -3.0, -3.0),
    }
    gaps = {
        "random0": (1.5, 1.5, 1.5),
        "random1": (0.75, 0.0, 1.5),
        "random2": (0.0, 0.0, 0.0),
    }
    for name in ("obscure_real", "fictional_generated"):
        for (arm, vs), (slope, lo, hi) in expected.items():
            row = _row(rows, columns, type=name, side="unknown", arm=arm, vs=vs)
            assert row["n"] == 2
            assert (row["slope"], row["lo"], row["hi"]) == pytest.approx(
                (slope, lo, hi)
            )
            assert row["span"] == pytest.approx(slope * 0.8)
            cells = (row["abs_gap"], row["abs_lo"], row["abs_hi"])
            if vs:
                assert cells == pytest.approx(gaps[vs])
            else:
                assert cells == (None, None, None)
        known = _row(rows, columns, type=name, side="known", arm="lat0", vs="")
        assert known["slope"] == 0.0


def test_the_headline_is_the_unknown_side_of_the_direction_contrast(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _dose_run(tiny_qwen3, tmp_path, monkeypatch)

    headline = _summary(run_dir)["headline"]

    assert headline["group"] == "obscure_real"
    assert headline["side"] == "unknown"
    assert headline["metric"] == "abstention"
    assert headline["lat_slope"] == pytest.approx(-1.5)
    assert (headline["lat_lo"], headline["lat_hi"]) == pytest.approx(
        (-1.5, -1.5)
    )
    assert headline["max_abs_random_slope"] == pytest.approx(1.5)
    gaps = {
        "random0": [1.5, 1.5, 1.5],
        "random1": [0.75, 0.0, 1.5],
        "random2": [0.0, 0.0, 0.0],
    }
    assert list(headline["abs_gaps"]) == list(gaps)
    for arm, cell in gaps.items():
        assert headline["abs_gaps"][arm] == pytest.approx(cell)
    assert headline["distinguishable"] is False


def test_the_claim_holds_when_steering_outmoves_every_random_arm(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _dose_run(
        tiny_qwen3, tmp_path, monkeypatch, random1_acts=False, n_random=2
    )

    headline = _summary(run_dir)["headline"]

    assert headline["max_abs_random_slope"] == 0.0
    assert headline["distinguishable"] is True


def test_the_headline_is_null_without_the_direction_contrast(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _dose_run(
        tiny_qwen3,
        tmp_path,
        monkeypatch,
        eval_types=("fictional_generated",),
    )

    assert _summary(run_dir)["headline"] is None


def test_every_arm_has_its_own_effect_row(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _dose_run(tiny_qwen3, tmp_path, monkeypatch)

    summary = _summary(run_dir)
    table = run_dir.read_json("table_effect")
    assert isinstance(table, dict)
    columns, rows = table["columns"], table["rows"]
    chosen = summary["chosen_coeff"]
    assert chosen == -0.2
    assert len(rows) == 16
    assert {r[2] for r in rows} == {"lat0", "random0", "random1", "random2"}
    shifts = {"lat0": 1.0, "random0": 0.0, "random1": 0.5, "random2": 0.0}
    for arm, shift in shifts.items():
        row = _row(rows, columns, type="obscure_real", side="unknown", arm=arm)
        assert row["coeff"] == chosen
        assert row["abstention_shift"] == pytest.approx(shift)


def test_the_summary_records_the_template_and_the_group_sizes(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _canned_run(
        tiny_qwen3, tmp_path, monkeypatch, _planted, n_prompts=3
    )

    summary = _summary(run_dir)
    plan = plan_groups(_config(n_prompts=3), _items(), True)
    assert summary["template"] == "fewshot"
    assert summary["n_items"] == {
        g.name: {"known": len(g.known), "unknown": len(g.unknown)}
        for g in plan.groups
    }
    assert summary["n_items"]["obscure_real"] == {"known": 3, "unknown": 3}


def test_shift_dose_slopes_follow_the_planted_entropy_dose(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = {"lat0": 2.0, "random0": 0.0, "random1": 0.0}
    seen: dict[bytes, str] = {}

    def fake_stats(
        loaded: LoadedModel, prompts: Sequence[str], **kwargs: Any
    ) -> TokenStats:
        entropy = 1.0 + 0.1 * np.arange(len(prompts))
        controller = kwargs["edits"]
        if controller is not None:
            vector = next(iter(controller.vectors.values())).numpy()
            if vector.tobytes() not in seen:
                seen[vector.tobytes()] = list(weight)[len(seen)]
            arm = seen[vector.tobytes()]
            entropy = entropy + controller.coeff * weight[arm]
        n = len(prompts)
        return TokenStats(np.zeros(n), entropy, np.full(n, -2.0))

    monkeypatch.setattr(p9d_steering, "next_token_stats", fake_stats)
    config = _config(coeffs=DOSE_GRID)
    run_dir = _run_dir(tmp_path, config)
    execute(config, run_dir, tiny_arch, _items(), known_contrasts=True)

    columns, rows = _dose(run_dir)

    assert len(rows) == 35
    assert {r[2] for r in rows} == {"entropy", "answer_logprob"}
    answer_rows = [r for r in rows if r[2] == "answer_logprob"]
    assert {(r[0], r[1]) for r in answer_rows} == {
        ("obscure_real", "known"),
        ("obscure_real", "unknown"),
        ("fictional_generated", "known"),
    }
    assert {r[5] for r in answer_rows} == {4}
    assert {r[6] for r in answer_rows} == {0.0}
    for side in ("known", "unknown"):
        match = {"type": "fictional_generated", "side": side}
        own = _row(rows, columns, metric="entropy", arm="lat0", vs="", **match)
        assert own["n"] == 4
        assert (own["slope"], own["lo"], own["hi"]) == pytest.approx(
            (2.0, 2.0, 2.0)
        )
        assert own["span"] == pytest.approx(1.6)
        flat = _row(
            rows, columns, metric="entropy", arm="random0", vs="", **match
        )
        assert flat["slope"] == pytest.approx(0.0)
        diff = _row(
            rows, columns, metric="entropy", arm="lat0", vs="random1", **match
        )
        assert diff["slope"] == pytest.approx(2.0)
    headline = _summary(run_dir)["headline"]
    assert headline["metric"] == "entropy"
    assert headline["lat_slope"] == pytest.approx(2.0)
    assert headline["max_abs_random_slope"] == pytest.approx(0.0)
    assert headline["distinguishable"] is True


def _planted_values(*columns: list[float]) -> NDArray[np.float64]:
    """Stacks per item columns into the (coefficients, items) layout."""
    return np.array(columns, dtype=np.float64).T


def _dose_row(arm: str, vs: str, slope: float | None, abs_lo: float | None):
    """A hand built row of the dose table for the headline tests."""
    cells = [None] * 4 if slope is None else [slope, slope, slope, 0.0]
    gap = [None, None, None] if not vs else [0.5, abs_lo, 1.0]
    return ["g", "unknown", "abstention", arm, vs, 2, *cells, *gap]


def test_the_seed_and_size_reach_every_absolute_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    real = p9d_steering.abs_slope_gap_ci

    def spy(slopes: Any, reference: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real(slopes, reference, **kwargs)

    monkeypatch.setattr(p9d_steering, "abs_slope_gap_ci", spy)
    step = [1.0, 1.0, 0.0, 0.0, 0.0]
    values = {
        "lat0": _planted_values(step, step[::-1]),
        "random0": _planted_values(step, step),
        "random1": _planted_values(step[::-1], step),
    }

    p9d_steering._dose_rows(
        "g", "unknown", "abstention", DOSE_GRID, values, n_boot=77, seed=5
    )

    assert calls == [{"n_boot": 77, "seed": 5}] * 2


def test_a_zero_lower_bound_of_the_gap_does_not_establish_the_claim() -> None:
    rows = [
        _dose_row("lat0", "", -1.0, None),
        _dose_row("random0", "", 0.0, None),
        _dose_row("lat0", "random0", -1.0, 0.0),
    ]
    assert (
        p9d_steering._headline(rows, "g", "abstention")["distinguishable"]
        is False
    )
    rows[-1] = _dose_row("lat0", "random0", -1.0, 0.25)
    assert (
        p9d_steering._headline(rows, "g", "abstention")["distinguishable"]
        is True
    )


def test_the_largest_random_slope_is_taken_in_magnitude() -> None:
    rows = [
        _dose_row("lat0", "", 1.0, None),
        _dose_row("random0", "", -2.0, None),
        _dose_row("random1", "", 0.5, None),
    ]
    headline = p9d_steering._headline(rows, "g", "abstention")
    assert headline["max_abs_random_slope"] == 2.0


def test_an_item_without_a_finite_slope_is_not_counted() -> None:
    step = [1.0, 1.0, 0.0, 0.0, 0.0]
    broken = [1.0, np.nan, 0.0, 0.0, 0.0]
    values = {
        "lat0": _planted_values(step, broken, step),
        "random0": _planted_values(step, broken, step[::-1]),
    }

    rows = p9d_steering._dose_rows(
        "g", "unknown", "abstention", DOSE_GRID, values, n_boot=20, seed=0
    )

    assert [r[5] for r in rows] == [2, 2, 2]


def test_a_random_arm_without_slopes_leaves_empty_rows() -> None:
    step = [1.0, 1.0, 0.0, 0.0, 0.0]
    nothing = [np.nan] * 5
    values = {
        "lat0": _planted_values(step, step),
        "random0": _planted_values(nothing, nothing),
        "random1": _planted_values(step[::-1], step[::-1]),
    }

    rows = p9d_steering._dose_rows(
        "g", "unknown", "abstention", DOSE_GRID, values, n_boot=20, seed=0
    )

    by_key = {(r[3], r[4]): r for r in rows}
    assert len(rows) == 5
    assert by_key[("random0", "")][5:] == [0] + [None] * 7
    assert by_key[("lat0", "random0")][5:] == [0] + [None] * 7
    assert by_key[("lat0", "random1")][5] == 2
    headline = p9d_steering._headline(rows, "g", "abstention")
    assert headline["distinguishable"] is False
    assert headline["max_abs_random_slope"] == pytest.approx(1.5)


def test_items_finite_in_only_one_arm_leave_the_contrast_empty() -> None:
    step = [1.0, 1.0, 0.0, 0.0, 0.0]
    nothing = [np.nan] * 5
    values = {
        "lat0": _planted_values(step, nothing),
        "random0": _planted_values(nothing, step),
    }

    rows = p9d_steering._dose_rows(
        "g", "unknown", "abstention", DOSE_GRID, values, n_boot=20, seed=0
    )

    by_key = {(r[3], r[4]): r for r in rows}
    assert by_key[("random0", "")][5] == 1
    assert by_key[("lat0", "random0")][5:] == [0] + [None] * 7


SELECTIVITY_COLUMNS = [
    "type",
    "metric",
    "n_known",
    "n_unknown",
    "known_slope",
    "unknown_slope",
    "gap",
    "lo",
    "hi",
    "selective",
]


def _selectivity(run_dir: RunDir) -> tuple[list[str], list[list[Any]]]:
    table = run_dir.read_json("table_selectivity")
    assert isinstance(table, dict)
    return table["columns"], table["rows"]


def _selective_run(
    loaded: LoadedModel,
    out: Path,
    monkeypatch: pytest.MonkeyPatch,
    known_below: float,
) -> RunDir:
    """Runs abstention mode with a planted slope on both sides.

    The steering arm abstains on the unknown items below -0.1 and on the known
    items below `known_below`, so the unknown slope is -1.5 and the known
    slope is -1.5 for -0.1, -1.0 for -0.3 and 0.0 for -0.5.
    """
    settings = {**DOSE_OVERRIDES}

    def rule(item: LabeledItem, arm: int, coeff: float) -> list[str]:
        edge = known_below if _is_known(item) else -0.1
        return [ABSTAIN] if arm == 0 and coeff < edge else _answer(item)

    run_dir, _ = _canned_run(loaded, out, monkeypatch, rule, **settings)
    return run_dir


def test_the_selectivity_table_has_the_agreed_columns_and_rows(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _dose_run(tiny_qwen3, tmp_path, monkeypatch)

    columns, rows = _selectivity(run_dir)

    assert columns == SELECTIVITY_COLUMNS
    assert [(r[0], r[1]) for r in rows] == [
        ("obscure_real", "abstention"),
        ("fictional_generated", "abstention"),
    ]


def test_a_larger_unknown_slope_is_selective(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _selective_run(tiny_qwen3, tmp_path, monkeypatch, -0.3)

    columns, rows = _selectivity(run_dir)

    for name in ("obscure_real", "fictional_generated"):
        row = _row(rows, columns, type=name)
        assert (row["n_known"], row["n_unknown"]) == (2, 2)
        assert row["known_slope"] == pytest.approx(-1.0)
        assert row["unknown_slope"] == pytest.approx(-1.5)
        assert (row["gap"], row["lo"], row["hi"]) == pytest.approx(
            (0.5, 0.5, 0.5)
        )
        assert row["selective"] is True
    assert _summary(run_dir)["headline"]["selective"] is True


def test_an_equal_slope_on_both_sides_is_not_selective(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _selective_run(tiny_qwen3, tmp_path, monkeypatch, -0.1)

    columns, rows = _selectivity(run_dir)

    row = _row(rows, columns, type="obscure_real")
    assert row["known_slope"] == pytest.approx(-1.5)
    assert row["gap"] == pytest.approx(0.0)
    assert row["selective"] is False
    assert _summary(run_dir)["headline"]["selective"] is False


def test_the_selectivity_gap_uses_magnitudes(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def rule(item: LabeledItem, arm: int, coeff: float) -> list[str]:
        if _is_known(item):
            moved = arm == 0 and coeff > 0
        else:
            moved = arm == 0 and coeff < 0
        return [ABSTAIN] if moved else _answer(item)

    run_dir, _ = _canned_run(
        tiny_qwen3, tmp_path, monkeypatch, rule, **DOSE_OVERRIDES
    )

    columns, rows = _selectivity(run_dir)

    row = _row(rows, columns, type="obscure_real")
    assert row["known_slope"] == pytest.approx(1.5)
    assert row["unknown_slope"] == pytest.approx(-1.5)
    assert row["gap"] == pytest.approx(0.0)
    assert row["selective"] is False


def test_the_selectivity_interval_gets_the_seed_and_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    real = p9d_steering.selectivity_gap

    def spy(unknown: Any, known: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real(unknown, known, **kwargs)

    monkeypatch.setattr(p9d_steering, "selectivity_gap", spy)
    slopes = {
        ("g", "known", "abstention"): {"lat0": np.array([0.0, 0.5])},
        ("g", "unknown", "abstention"): {"lat0": np.array([1.0, 1.5])},
    }

    rows = p9d_steering._selectivity_rows(slopes, n_boot=77, seed=5)

    assert calls == [{"n_boot": 77, "seed": 5}]
    assert rows[0][:4] == ["g", "abstention", 2, 2]


def test_a_side_without_finite_slopes_leaves_an_empty_row() -> None:
    slopes = {
        ("g", "known", "answer_logprob"): {"lat0": np.array([0.5, np.nan])},
        ("g", "unknown", "answer_logprob"): {
            "lat0": np.array([np.nan, np.nan])
        },
        ("h", "known", "answer_logprob"): {"lat0": np.array([np.nan])},
        ("h", "unknown", "answer_logprob"): {"lat0": np.array([np.nan])},
    }

    rows = p9d_steering._selectivity_rows(slopes, n_boot=20, seed=0)

    assert rows == [["g", "answer_logprob", 1, 0, *[None] * 6]]


def test_the_selective_flag_is_null_for_an_empty_row_or_no_headline() -> None:
    dose = [
        _dose_row("lat0", "", -1.0, None),
        _dose_row("random0", "", 0.0, None),
        _dose_row("lat0", "random0", -1.0, 0.25),
    ]
    empty = [["g", "abstention", 1, 0, *[None] * 6]]
    full = [["g", "abstention", 2, 2, 0.0, 1.0, 1.0, 0.5, 1.5, True]]
    headline = p9d_steering._headline(dose, "g", "abstention")

    assert p9d_steering._with_selective(headline, empty)["selective"] is None
    assert p9d_steering._with_selective(headline, full)["selective"] is True
    assert p9d_steering._with_selective(None, full) is None


def test_shift_mode_writes_the_selectivity_table(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = {"lat0": 2.0, "random0": 0.0, "random1": 0.0}
    seen: dict[bytes, str] = {}

    def fake_stats(
        loaded: LoadedModel, prompts: Sequence[str], **kwargs: Any
    ) -> TokenStats:
        entropy = 1.0 + 0.1 * np.arange(len(prompts))
        controller = kwargs["edits"]
        if controller is not None:
            vector = next(iter(controller.vectors.values())).numpy()
            if vector.tobytes() not in seen:
                seen[vector.tobytes()] = list(weight)[len(seen)]
            entropy = (
                entropy + controller.coeff * weight[seen[vector.tobytes()]]
            )
        n = len(prompts)
        return TokenStats(np.zeros(n), entropy, np.full(n, -2.0))

    monkeypatch.setattr(p9d_steering, "next_token_stats", fake_stats)
    config = _config(coeffs=DOSE_GRID)
    run_dir = _run_dir(tmp_path, config)
    execute(config, run_dir, tiny_arch, _items(), known_contrasts=True)

    columns, rows = _selectivity(run_dir)

    assert columns == SELECTIVITY_COLUMNS
    entropy = _row(rows, columns, type="fictional_generated", metric="entropy")
    assert (entropy["n_known"], entropy["n_unknown"]) == (4, 4)
    assert entropy["known_slope"] == pytest.approx(2.0)
    assert entropy["unknown_slope"] == pytest.approx(2.0)
    only_known = _row(
        rows, columns, type="fictional_generated", metric="answer_logprob"
    )
    assert only_known["n_known"] == 4
    assert only_known["n_unknown"] == 0
    assert only_known["selective"] is None
    assert only_known["gap"] is None
    headline = _summary(run_dir)["headline"]
    both = _row(rows, columns, type="obscure_real", metric="entropy")
    assert headline["selective"] == both["selective"]
