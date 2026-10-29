"""Tests for the b1 experiment: published error detectors beside P9c."""

from collections.abc import Sequence
import dataclasses
import json
import math
from pathlib import Path
import shutil
import types
from typing import Any
import zlib

import numpy as np
from numpy.typing import NDArray
import pytest
import torch

from tomography.cli import load_experiment
from tomography.datasets.records import MetaValue
from tomography.datasets.records import QAExample
from tomography.datasets.records import Split
from tomography.experiments import b1_detection_baselines
from tomography.experiments import p9c_reliability
from tomography.experiments.b1_detection_baselines import BASELINES
from tomography.experiments.b1_detection_baselines import PAIRED_COLUMNS
from tomography.experiments.b1_detection_baselines import REFERENCES
from tomography.experiments.b1_detection_baselines import Config
from tomography.experiments.b1_detection_baselines import execute
from tomography.experiments.b1_detection_baselines import plot
from tomography.experiments.b1_detection_baselines import run
from tomography.experiments.base import RunDir
from tomography.experiments.p9a_labels import Config as LabelsConfig
from tomography.experiments.p9c_reliability import TABLE_COLUMNS
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import ItemLabel
from tomography.knowledge import few_shot_prompt
from tomography.model import LoadedModel
from tomography.p9.answer_probe import answer_activations
from tomography.p9.answer_probe import fit_answer_probe
from tomography.p9.eigenscore import eigenscores
from tomography.p9.pool import LabeledItem
from tomography.p9.pool import pool_records
from tomography.p9.reading import ItemActivations
from tomography.p9.risk import aurc
from tomography.p9.risk import auroc_ci
from tomography.p9.risk import auroc_diff_ci
from tomography.p9.risk import grouped_auroc_ci
from tomography.p9.semantic_entropy import entailment_pairs
from tomography.reading.metrics import auroc

DEFAULT_HASH = "b3d52a528c68"
SPLITS: tuple[Split, ...] = ("train", "train", "train", "val", "test")
RELATIONS = ("father", "mother", "occupation")
DATASETS = ("popqa", "triviaqa")
LAYERS = (1, 2, 3)
N_SHOTS = 2
P9C_BOOT = 60
MIXED = ("other", "gold", "gold", "gold", "gold")
BLANK = ("", " ", "", "", "")
H_MIXED = -(0.2 * math.log(0.2) + 0.8 * math.log(0.8))


def _item(
    uid: str,
    split: Split,
    source: str,
    correct: bool,
    meta: dict[str, MetaValue],
) -> LabeledItem:
    i = int(uid.split(":")[1])
    if correct:
        greedy, samples = "gold", ("gold",) * 5
    elif i % 17 == 2:
        greedy, samples = "", BLANK if i % 34 == 19 else ()
    elif i % 3 == 0:
        greedy, samples = "other", ("other",) * 5
    else:
        greedy, samples = "other", MIXED
    example = QAExample(
        uid=uid,
        question=f"Who is {uid.replace(':', ' ')}?",
        answers=("gold",),
        kind="real" if source == "popqa" else "open",
        split=split,
        meta={"source": source, **meta},
    )
    label = ItemLabel(
        uid=uid,
        label="known" if correct else "unknown",
        basis="behavior",
        greedy=greedy,
        greedy_correct=correct,
        samples=samples,
        n_correct=sum(s == "gold" for s in samples),
    )
    return LabeledItem(example, label)


def _items() -> list[LabeledItem]:
    popqa = [
        _item(
            f"popqa:{i}",
            SPLITS[i % 5],
            "popqa",
            i % 6 in (0, 1, 4),
            {
                "relation": RELATIONS[i % 3],
                "popularity": 10 ** (1 + (0 if i % 6 in (0, 1, 4) else 3)),
                "subject": f"Person {i}",
            },
        )
        for i in range(150)
    ]
    trivia = [
        _item(f"triviaqa:{i}", SPLITS[i % 5], "triviaqa", i % 2 == 0, {})
        for i in range(60)
    ]
    return [*popqa, *trivia]


def _dataset(items: Sequence[LabeledItem], name: str) -> list[LabeledItem]:
    return [i for i in items if i.group == name]


def _test_items(name: str) -> list[LabeledItem]:
    return [i for i in _dataset(_items(), name) if i.example.split == "test"]


def _split(items: Sequence[LabeledItem]) -> NDArray[np.str_]:
    return np.array([i.example.split for i in items])


def _correct(items: Sequence[LabeledItem]) -> NDArray[np.int64]:
    return np.array([i.label.greedy_correct for i in items], dtype=np.int64)


def _prompts(items: Sequence[LabeledItem]) -> list[str]:
    shots = DEFAULT_SHOTS[:N_SHOTS]
    return [few_shot_prompt(i.example.question, shots) for i in items]


def _planted_collect(
    loaded: LoadedModel,
    items: Sequence[LabeledItem],
    template: Any,
    chat: Any,
    *,
    layers: Sequence[int],
    bos: Any = None,
    batch_size: int = 8,
) -> ItemActivations:
    data = np.zeros((len(layers), len(items), loaded.d_model), dtype=np.float32)
    for j, item in enumerate(items):
        rng = np.random.default_rng(zlib.crc32(item.example.uid.encode()))
        data[:, j] = 0.1 * rng.standard_normal((len(layers), loaded.d_model))
        if item.label.greedy_correct:
            data[:, j, 0] += 1.0
    return ItemActivations(
        tuple(i.example.uid for i in items), tuple(layers), data
    )


def _p9c_run(
    loaded: LoadedModel, root: Path, *, known_contrasts: bool
) -> RunDir:
    items = _items()
    labels = RunDir.create(LabelsConfig(model="tiny"), out_root=root / "p9a")
    labels.write_json("items", pool_records(items))
    labels.write_json(
        "summary",
        {"known_contrasts": known_contrasts, "recipe": {"n_shots": N_SHOTS}},
    )
    labels.finalize()
    config = p9c_reliability.Config(
        model="tiny",
        labels_run=str(labels.path),
        datasets=DATASETS,
        n_shots=N_SHOTS,
        n_seeds=2,
        n_pairs=16,
        min_pairs=8,
        min_probe=20,
        n_boot=P9C_BOOT,
        batch_size=16,
    )
    run_dir = RunDir.create(config, out_root=root)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            p9c_reliability, "collect_item_activations", _planted_collect
        )
        p9c_reliability.execute(
            config, run_dir, loaded, items, known_contrasts=known_contrasts
        )
    run_dir.finalize()
    return run_dir


_P9C_RUNS: dict[bool, RunDir] = {}


def _cached_p9c(
    loaded: LoadedModel, factory: pytest.TempPathFactory, gate: bool
) -> RunDir:
    if gate not in _P9C_RUNS:
        _P9C_RUNS[gate] = _p9c_run(
            loaded, factory.mktemp(f"p9c-{gate}"), known_contrasts=gate
        )
    return _P9C_RUNS[gate]


@pytest.fixture
def reliability(
    tiny_gpt2: LoadedModel, tmp_path_factory: pytest.TempPathFactory
) -> RunDir:
    """A finished P9c run whose `lat` reader exists."""
    return _cached_p9c(tiny_gpt2, tmp_path_factory, True)


@pytest.fixture
def closed_reliability(
    tiny_gpt2: LoadedModel, tmp_path_factory: pytest.TempPathFactory
) -> RunDir:
    """A finished P9c run whose gate gave no `lat` reader."""
    return _cached_p9c(tiny_gpt2, tmp_path_factory, False)


def _never(
    premises: Sequence[str], hypotheses: Sequence[str]
) -> NDArray[np.bool_]:
    return np.zeros(len(premises), dtype=bool)


def _always(
    premises: Sequence[str], hypotheses: Sequence[str]
) -> NDArray[np.bool_]:
    return np.ones(len(premises), dtype=bool)


def _no_forward(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("no forward pass may run")


@pytest.fixture
def no_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes every forward pass of a baseline fail the test."""
    monkeypatch.setattr(
        b1_detection_baselines, "answer_activations", _no_forward
    )
    monkeypatch.setattr(b1_detection_baselines, "eigenscores", _no_forward)


def _config(reliability: RunDir, **overrides: Any) -> Config:
    base = Config(
        model="tiny", reliability_run=str(reliability.path), batch_size=16
    )
    return dataclasses.replace(base, **overrides)


def _execute(
    loaded: LoadedModel,
    reliability: RunDir,
    out: Path,
    *,
    entails: Any = _never,
    **overrides: Any,
) -> RunDir:
    config = _config(reliability, **overrides)
    run_dir = RunDir.create(config, out_root=out)
    execute(config, run_dir, loaded, reliability, _items(), entails=entails)
    return run_dir


_DEFAULT_RUNS: dict[str, RunDir] = {}


@pytest.fixture
def default_run(
    tiny_gpt2: LoadedModel,
    reliability: RunDir,
    tmp_path_factory: pytest.TempPathFactory,
) -> RunDir:
    """One b1 run with the default settings on the P9c run with `lat`."""
    if "default" not in _DEFAULT_RUNS:
        _DEFAULT_RUNS["default"] = _execute(
            tiny_gpt2, reliability, tmp_path_factory.mktemp("b1")
        )
    return _DEFAULT_RUNS["default"]


def _summary(run_dir: RunDir) -> dict[str, Any]:
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    return summary


def _rows(
    run_dir: RunDir, name: str, columns: list[str]
) -> list[dict[str, Any]]:
    table = run_dir.read_json(name)
    assert isinstance(table, dict)
    assert table["columns"] == columns
    return [dict(zip(columns, row, strict=True)) for row in table["rows"]]


def _row(
    rows: list[dict[str, Any]], dataset: str, predictor: str
) -> dict[str, Any]:
    (row,) = [
        r
        for r in rows
        if r["dataset"] == dataset and r["predictor"] == predictor
    ]
    return row


def _copy(run_dir: RunDir, target: Path) -> RunDir:
    shutil.copytree(run_dir.path, target)
    return RunDir.open(target)


def test_the_constants_name_the_baselines_and_the_references() -> None:
    assert BASELINES == ("answer_probe", "semantic_entropy", "eigenscore")
    assert REFERENCES == ("lat", "p_true")
    assert PAIRED_COLUMNS == [
        "dataset",
        "predictor",
        "reference",
        "diff",
        "lo",
        "hi",
    ]


def test_config_defaults_and_round_trip() -> None:
    config = Config(reliability_run="runs/p9c")

    assert config.experiment == "b1"
    assert config.model == "gpt2"
    assert config.seed == 0
    assert config.baselines == BASELINES
    assert config.references == REFERENCES
    assert config.batch_size == 8
    assert Config.from_dict(config.to_dict()) == config


def test_the_default_config_hash_is_pinned() -> None:
    assert Config(reliability_run="runs/p9c").config_hash() == DEFAULT_HASH


def test_config_requires_the_reliability_run() -> None:
    with pytest.raises(ValueError, match="reliability_run"):
        Config.from_dict({})


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("b1")

    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"baselines": ()}, "at least one baseline"),
        ({"baselines": ("magic",)}, "unknown baselines: magic"),
        ({"baselines": ("eigenscore",) * 2}, "repeated baselines: eigenscore"),
        ({"references": ("magic",)}, "unknown references: magic"),
        ({"references": ("lat", "lat")}, "repeated references: lat"),
    ],
)
def test_an_invalid_config_fails_before_any_forward_pass(
    tiny_gpt2: LoadedModel,
    reliability: RunDir,
    tmp_path: Path,
    no_forward: None,
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _execute(tiny_gpt2, reliability, tmp_path, **overrides)


def test_semantic_entropy_without_an_entailment_model_is_an_error(
    tiny_gpt2: LoadedModel, reliability: RunDir, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="entailment model"):
        _execute(tiny_gpt2, reliability, tmp_path, entails=None)


def test_execute_writes_the_files_of_a_run(default_run: RunDir) -> None:
    files = sorted(p.name for p in default_run.path.iterdir() if p.is_file())

    assert files == [
        "config.json",
        "meta.json",
        "signals_popqa.npz",
        "signals_triviaqa.npz",
        "summary.json",
        "table.json",
        "table_paired.json",
    ]


def test_the_table_keeps_the_p9c_rows_and_appends_the_baselines(
    default_run: RunDir, reliability: RunDir
) -> None:
    ours = _rows(default_run, "table", TABLE_COLUMNS)
    theirs = _rows(reliability, "table", TABLE_COLUMNS)

    assert [r["dataset"] for r in ours] == sorted(
        (r["dataset"] for r in ours), key=DATASETS.index
    )
    for dataset in DATASETS:
        rows = [r for r in ours if r["dataset"] == dataset]
        p9c = [r for r in theirs if r["dataset"] == dataset]
        assert rows[: len(p9c)] == p9c
        assert [r["predictor"] for r in rows[len(p9c) :]] == list(BASELINES)


def test_baseline_rows_follow_the_saved_signals_and_the_p9c_bootstrap(
    default_run: RunDir,
) -> None:
    rows = _rows(default_run, "table", TABLE_COLUMNS)

    for dataset in DATASETS:
        signals = default_run.read_arrays(f"signals_{dataset}")
        correct = signals["correct"]
        for name in BASELINES:
            row = _row(rows, dataset, name)
            point, lo, hi = auroc_ci(
                signals[name], correct, n_boot=P9C_BOOT, seed=0
            )
            assert row["n"] == correct.size
            assert row["n_error"] == int((correct == 0).sum())
            assert [row["auroc"], row["lo"], row["hi"]] == pytest.approx(
                [point, lo, hi]
            )
            assert row["aurc"] == pytest.approx(
                aurc(signals[name], correct, seed=0)
            )


def test_popqa_baseline_rows_hold_the_within_relation_auroc(
    default_run: RunDir, reliability: RunDir
) -> None:
    rows = _rows(default_run, "table", TABLE_COLUMNS)
    p9c_rows = _rows(reliability, "table", TABLE_COLUMNS)
    relation = reliability.read_arrays("signals_popqa")["relation"]
    signals = default_run.read_arrays("signals_popqa")

    for name in BASELINES:
        row = _row(rows, "popqa", name)
        within = grouped_auroc_ci(
            signals[name],
            signals["correct"],
            relation,
            n_boot=P9C_BOOT,
            seed=0,
        )
        assert [
            row["auroc_within"],
            row["lo_within"],
            row["hi_within"],
        ] == pytest.approx(list(within))
        assert (
            row["n_pairs_within"]
            == _row(p9c_rows, "popqa", "p_true")["n_pairs_within"]
        )
        trivia = _row(rows, "triviaqa", name)
        assert [
            trivia[c]
            for c in (
                "auroc_within",
                "lo_within",
                "hi_within",
                "n_pairs_within",
            )
        ] == [None] * 4


def test_the_signals_hold_finite_test_scores_and_the_stored_outcomes(
    default_run: RunDir, reliability: RunDir
) -> None:
    for dataset in DATASETS:
        ours = default_run.read_arrays(f"signals_{dataset}")
        theirs = reliability.read_arrays(f"signals_{dataset}")

        assert set(ours) == {*BASELINES, "correct"}
        np.testing.assert_array_equal(ours["correct"], theirs["correct"])
        for name in BASELINES:
            assert ours[name].shape == theirs["correct"].shape
            assert np.isfinite(ours[name]).all()


def _tamper(run_dir: RunDir, how: str) -> None:
    arrays = run_dir.read_arrays("signals_popqa")
    if how == "flip":
        arrays["correct"][0] = 1 - arrays["correct"][0]
    elif how == "drop":
        arrays = {k: v[1:] for k, v in arrays.items()}
    else:
        arrays["relation"][0] = "zzz"
    run_dir.write_arrays("signals_popqa", **arrays)


@pytest.mark.parametrize("how", ["flip", "drop", "relation"])
def test_stored_outcomes_that_differ_from_the_labels_stop_the_run_first(
    tiny_gpt2: LoadedModel,
    reliability: RunDir,
    tmp_path: Path,
    no_forward: None,
    how: str,
) -> None:
    copy = _copy(reliability, tmp_path / "p9c")
    _tamper(copy, how)

    with pytest.raises(ValueError, match=r"popqa: the test .* differ"):
        _execute(tiny_gpt2, copy, tmp_path, entails=_no_forward)


def test_the_datasets_must_be_those_the_p9c_run_evaluated(
    tiny_gpt2: LoadedModel, reliability: RunDir, tmp_path: Path
) -> None:
    copy = _copy(reliability, tmp_path / "p9c")
    summary = _summary(copy)
    copy.write_json("summary", {**summary, "datasets": ["popqa"]})

    with pytest.raises(ValueError, match="datasets"):
        _execute(tiny_gpt2, copy, tmp_path)


def test_a_p9c_table_in_other_columns_is_refused(
    tiny_gpt2: LoadedModel, reliability: RunDir, tmp_path: Path
) -> None:
    copy = _copy(reliability, tmp_path / "p9c")
    table = copy.read_json("table")
    assert isinstance(table, dict)
    copy.write_json(
        "table",
        {
            "columns": table["columns"][:-1],
            "rows": [r[:-1] for r in table["rows"]],
        },
    )

    with pytest.raises(ValueError, match="columns"):
        _execute(tiny_gpt2, copy, tmp_path)


def test_the_answer_probe_is_fit_on_train_with_its_layer_chosen_on_val(
    default_run: RunDir, tiny_gpt2: LoadedModel
) -> None:
    rows = _rows(default_run, "table", TABLE_COLUMNS)
    summary = _summary(default_run)

    for dataset in DATASETS:
        items = _dataset(_items(), dataset)
        split = _split(items)
        h = answer_activations(
            tiny_gpt2,
            _prompts(items),
            [i.label.greedy for i in items],
            layers=LAYERS,
            batch_size=16,
            share_prefix=True,
        )
        probe = fit_answer_probe(
            h,
            LAYERS,
            _correct(items),
            train=split == "train",
            val=split == "val",
        )
        lowest = np.nanmin(probe.scores[split == "train"])
        filled = np.where(np.isnan(probe.scores), lowest, probe.scores)
        signals = default_run.read_arrays(f"signals_{dataset}")

        np.testing.assert_allclose(
            signals["answer_probe"], filled[split == "test"]
        )
        assert summary["answer_probe"][dataset]["layer"] == probe.layer
        assert summary["answer_probe"][dataset]["val_auroc"] == pytest.approx(
            probe.val_auroc.tolist()
        )
        n_empty = sum(not i.label.greedy for i in items)
        assert n_empty > 0
        assert _row(rows, dataset, "answer_probe")["n_filled"] == n_empty


def _planted_answers(
    loaded: LoadedModel,
    prompts: Sequence[str],
    answers: Sequence[str],
    **kwargs: Any,
) -> NDArray[np.float32]:
    """Answer activations that carry correctness at layer 2.

    The wrong answer to "popqa 14", a test item, sits far below every other
    item, so the lowest probe score is not a train score.
    """
    layers = list(kwargs["layers"])
    rng = np.random.default_rng(0)
    h = rng.standard_normal((len(layers), len(answers), 8))
    h[layers.index(2), :, 0] += 4.0 * np.array([a == "gold" for a in answers])
    h[:, ["popqa 14?" in p for p in prompts], 0] -= 50.0
    h[:, [not a for a in answers]] = np.nan
    return h.astype(np.float32)


_PLANTED_RUNS: dict[str, RunDir] = {}


@pytest.fixture
def planted_run(
    tiny_gpt2: LoadedModel,
    reliability: RunDir,
    tmp_path_factory: pytest.TempPathFactory,
) -> RunDir:
    """One probe-only b1 run on `_planted_answers`."""
    if "probe" not in _PLANTED_RUNS:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                b1_detection_baselines, "answer_activations", _planted_answers
            )
            _PLANTED_RUNS["probe"] = _execute(
                tiny_gpt2,
                reliability,
                tmp_path_factory.mktemp("b1-planted"),
                baselines=("answer_probe",),
            )
    return _PLANTED_RUNS["probe"]


def test_the_answer_probe_finds_a_planted_answer_signal(
    planted_run: RunDir,
) -> None:
    rows = _rows(planted_run, "table", TABLE_COLUMNS)
    for dataset in DATASETS:
        assert _summary(planted_run)["answer_probe"][dataset]["layer"] == 2
        assert _row(rows, dataset, "answer_probe")["auroc"] > 0.9


def test_an_empty_answer_takes_the_lowest_train_score_of_the_probe(
    tiny_gpt2: LoadedModel, planted_run: RunDir
) -> None:
    items = _dataset(_items(), "popqa")
    split = _split(items)
    probe = fit_answer_probe(
        _planted_answers(
            tiny_gpt2,
            _prompts(items),
            [i.label.greedy for i in items],
            layers=LAYERS,
        ),
        LAYERS,
        _correct(items),
        train=split == "train",
        val=split == "val",
    )
    lowest_train = np.nanmin(probe.scores[split == "train"])
    assert np.nanmin(probe.scores) < lowest_train
    test = _test_items("popqa")
    empty = [k for k, i in enumerate(test) if not i.label.greedy]
    scores = planted_run.read_arrays("signals_popqa")["answer_probe"]
    np.testing.assert_allclose(scores[empty], lowest_train)


def test_semantic_entropy_is_the_negated_entropy_of_the_answer_clusters(
    default_run: RunDir,
) -> None:
    rows = _rows(default_run, "table", TABLE_COLUMNS)

    for dataset in DATASETS:
        test = _test_items(dataset)
        missing = [i.label.samples in ((), BLANK) for i in test]
        expected = np.array(
            [-H_MIXED if i.label.samples == MIXED else 0.0 for i in test]
        )
        expected[missing] = -H_MIXED
        signals = default_run.read_arrays(f"signals_{dataset}")

        np.testing.assert_allclose(signals["semantic_entropy"], expected)
        assert any(missing)
        assert _row(rows, dataset, "semantic_entropy")["n_filled"] == sum(
            missing
        )
    assert any(i.label.samples == BLANK for i in _test_items("triviaqa"))


def test_the_entailment_verdicts_set_the_clusters(
    tiny_gpt2: LoadedModel, reliability: RunDir, tmp_path: Path
) -> None:
    run_dir = _execute(
        tiny_gpt2,
        reliability,
        tmp_path,
        entails=_always,
        baselines=("semantic_entropy",),
    )

    for dataset in DATASETS:
        signals = run_dir.read_arrays(f"signals_{dataset}")
        np.testing.assert_array_equal(signals["semantic_entropy"], 0.0)


def test_the_entailment_model_judges_every_pair_of_each_test_question(
    tiny_gpt2: LoadedModel, reliability: RunDir, tmp_path: Path
) -> None:
    calls: list[list[tuple[str, str]]] = []

    def recording(
        premises: Sequence[str], hypotheses: Sequence[str]
    ) -> NDArray[np.bool_]:
        calls.append(list(zip(premises, hypotheses, strict=True)))
        return _never(premises, hypotheses)

    run_dir = _execute(
        tiny_gpt2,
        reliability,
        tmp_path,
        entails=recording,
        baselines=("semantic_entropy",),
    )

    pairs = {
        dataset: [
            entailment_pairs(i.example.question, i.label.samples)
            for i in _test_items(dataset)
        ]
        for dataset in DATASETS
    }
    assert calls == [p for d in DATASETS for p in pairs[d] if p]
    assert _summary(run_dir)["nli_pairs"] == {
        d: sum(map(len, ps)) for d, ps in pairs.items()
    }


def test_eigenscore_is_the_negated_score_of_the_test_samples(
    default_run: RunDir, tiny_gpt2: LoadedModel
) -> None:
    rows = _rows(default_run, "table", TABLE_COLUMNS)

    for dataset in DATASETS:
        test = _test_items(dataset)
        raw = -eigenscores(
            tiny_gpt2,
            _prompts(test),
            [i.label.samples for i in test],
            batch_size=16,
            share_prefix=True,
        )
        expected = np.where(np.isnan(raw), np.nanmin(raw), raw)
        signals = default_run.read_arrays(f"signals_{dataset}")

        np.testing.assert_allclose(
            signals["eigenscore"], expected, rtol=1e-6, atol=1e-9
        )
        assert _row(rows, dataset, "eigenscore")["n_filled"] == int(
            np.isnan(raw).sum()
        )
    assert _summary(default_run)["eigen_layer"] == 2


def test_a_higher_baseline_score_means_a_more_likely_correct_answer(
    default_run: RunDir,
) -> None:
    signals = default_run.read_arrays("signals_popqa")

    assert auroc(signals["semantic_entropy"], signals["correct"]) > 0.5


def test_eigenscore_ranks_the_consistent_samples_of_correct_answers_higher(
    default_run: RunDir,
) -> None:
    for dataset in DATASETS:
        signals = default_run.read_arrays(f"signals_{dataset}")

        assert auroc(signals["eigenscore"], signals["correct"]) > 0.5


def test_the_paired_table_holds_every_baseline_against_every_reference(
    default_run: RunDir, reliability: RunDir
) -> None:
    rows = _rows(default_run, "table_paired", PAIRED_COLUMNS)

    assert [(r["dataset"], r["predictor"], r["reference"]) for r in rows] == [
        (d, b, r) for d in DATASETS for b in BASELINES for r in REFERENCES
    ]
    for row in rows:
        ours = default_run.read_arrays(f"signals_{row['dataset']}")
        theirs = reliability.read_arrays(f"signals_{row['dataset']}")
        expected = auroc_diff_ci(
            ours[row["predictor"]],
            theirs[row["reference"]],
            theirs["correct"],
            n_boot=P9C_BOOT,
            seed=0,
        )
        assert [row["diff"], row["lo"], row["hi"]] == pytest.approx(
            list(expected)
        )


def test_a_reference_the_p9c_run_lacks_is_named_in_the_summary(
    tiny_gpt2: LoadedModel, closed_reliability: RunDir, tmp_path: Path
) -> None:
    run_dir = _execute(
        tiny_gpt2,
        closed_reliability,
        tmp_path,
        baselines=("semantic_entropy",),
    )

    rows = _rows(run_dir, "table_paired", PAIRED_COLUMNS)
    assert {r["reference"] for r in rows} == {"p_true"}
    assert _summary(run_dir)["missing_references"] == {
        "popqa": ["lat"],
        "triviaqa": ["lat"],
    }


def test_a_subset_of_baselines_runs_alone_and_needs_no_entailment_model(
    tiny_gpt2: LoadedModel, reliability: RunDir, tmp_path: Path
) -> None:
    run_dir = _execute(
        tiny_gpt2,
        reliability,
        tmp_path,
        entails=None,
        baselines=("eigenscore",),
    )

    rows = _rows(run_dir, "table", TABLE_COLUMNS)
    p9c = {r["predictor"] for r in _rows(reliability, "table", TABLE_COLUMNS)}
    assert {r["predictor"] for r in rows} - p9c == {"eigenscore"}
    assert set(run_dir.read_arrays("signals_popqa")) == {
        "eigenscore",
        "correct",
    }
    summary = _summary(run_dir)
    assert "answer_probe" not in summary
    assert "nli_pairs" not in summary
    assert set(summary["seconds"]["popqa"]) == {"eigenscore"}


def test_the_summary_records_the_inputs_and_the_settings(
    default_run: RunDir, reliability: RunDir
) -> None:
    summary = _summary(default_run)

    assert summary["reliability_run"] == str(reliability.path)
    assert summary["labels_run"] == reliability.config["labels_run"]
    assert summary["datasets"] == list(DATASETS)
    assert summary["baselines"] == list(BASELINES)
    assert summary["references"] == list(REFERENCES)
    assert summary["missing_references"] == {}
    for dataset in DATASETS:
        assert set(summary["seconds"][dataset]) == set(BASELINES)
        assert all(s >= 0 for s in summary["seconds"][dataset].values())


def test_execute_is_deterministic_and_takes_the_seed_of_the_p9c_run(
    tiny_gpt2: LoadedModel,
    reliability: RunDir,
    default_run: RunDir,
    tmp_path: Path,
) -> None:
    other = _execute(tiny_gpt2, reliability, tmp_path, seed=7)

    for name, columns in (
        ("table", TABLE_COLUMNS),
        ("table_paired", PAIRED_COLUMNS),
    ):
        assert _rows(other, name, columns) == _rows(default_run, name, columns)
    for dataset in DATASETS:
        ours = other.read_arrays(f"signals_{dataset}")
        theirs = default_run.read_arrays(f"signals_{dataset}")
        for key in ours:
            np.testing.assert_array_equal(ours[key], theirs[key])


def test_run_reads_the_labels_of_the_p9c_run_and_loads_both_models(
    tiny_gpt2: LoadedModel,
    reliability: RunDir,
    default_run: RunDir,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[tuple[str, dict[str, Any]]] = []

    def fake_load(name: str, **kwargs: Any) -> LoadedModel:
        asked.append((name, kwargs))
        return tiny_gpt2

    def fake_nli(**kwargs: Any) -> types.SimpleNamespace:
        asked.append(("nli", kwargs))
        return types.SimpleNamespace(entails=_never)

    monkeypatch.setattr(b1_detection_baselines, "load_model", fake_load)
    monkeypatch.setattr(b1_detection_baselines, "load_nli", fake_nli)
    config = _config(reliability)
    run_dir = RunDir.create(config, out_root=tmp_path)

    run(config, run_dir)

    assert asked == [
        ("nli", {"device": "cpu"}),
        ("tiny", {"dtype": torch.float32}),
    ]
    assert _rows(run_dir, "table", TABLE_COLUMNS) == _rows(
        default_run, "table", TABLE_COLUMNS
    )


def test_run_loads_no_entailment_model_without_semantic_entropy(
    tiny_gpt2: LoadedModel,
    reliability: RunDir,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        b1_detection_baselines, "load_model", lambda name, **kwargs: tiny_gpt2
    )
    monkeypatch.setattr(b1_detection_baselines, "load_nli", _no_forward)
    config = _config(reliability, baselines=("eigenscore",))

    run(config, RunDir.create(config, out_root=tmp_path))


def test_run_refuses_an_unfinished_p9c_run(
    reliability: RunDir, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = _copy(reliability, tmp_path / "p9c")
    meta = json.loads((copy.path / "meta.json").read_text("utf-8"))
    (copy.path / "meta.json").write_text(
        json.dumps({**meta, "finished_at": None}), "utf-8"
    )
    monkeypatch.setattr(b1_detection_baselines, "load_model", _no_forward)
    config = _config(copy)

    with pytest.raises(ValueError, match="not a finished run"):
        run(config, RunDir.create(config, out_root=tmp_path))


def test_run_checks_the_config_before_loading_a_model(
    reliability: RunDir, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(b1_detection_baselines, "load_model", _no_forward)
    monkeypatch.setattr(b1_detection_baselines, "load_nli", _no_forward)
    config = _config(reliability, baselines=("magic",))

    with pytest.raises(ValueError, match="unknown baselines: magic"):
        run(config, RunDir.create(config, out_root=tmp_path))


def test_run_refuses_a_p9c_run_of_another_model(
    reliability: RunDir, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(b1_detection_baselines, "load_model", _no_forward)
    config = _config(reliability, model="gpt2")

    with pytest.raises(ValueError, match="'tiny', not of 'gpt2'"):
        run(config, RunDir.create(config, out_root=tmp_path))


def test_plot_draws_no_figure(default_run: RunDir, tmp_path: Path) -> None:
    assert plot(default_run, tmp_path) == []
