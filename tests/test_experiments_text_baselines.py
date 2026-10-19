"""Tests for the n-gram text baselines of the reading concepts."""

from collections.abc import Sequence
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from tomography.cli import load_experiment
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.experiments import p3_size
from tomography.experiments import r1_sentiment
from tomography.experiments import r2_utility
from tomography.experiments import r3_morality
from tomography.experiments import r5_honesty
from tomography.experiments import r7_truth
from tomography.experiments import tb1_text_baselines as text_baselines
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.tb1_text_baselines import CONCEPTS
from tomography.experiments.tb1_text_baselines import Config
from tomography.experiments.tb1_text_baselines import TextSet
from tomography.experiments.tb1_text_baselines import concept_items
from tomography.experiments.tb1_text_baselines import execute
from tomography.experiments.tb1_text_baselines import interval
from tomography.experiments.tb1_text_baselines import seed_accuracy
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import eval_pairs
from tomography.experiments.variants import eval_seed
from tomography.experiments.variants import task_set
from tomography.experiments.variants import task_splits
from tomography.experiments.variants import template_variants
from tomography.model import LoadedModel
from tomography.reading.metrics import wilson_ci
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat

PLAIN = ChatFormat("plain")
SPLITS = ("train", "val", "test")
GOOD = ("good", "fine", "great", "nice", "lovely", "solid")
BAD = ("bad", "poor", "dull", "grim", "awful", "weak")


def _loaded() -> LoadedModel:
    """A stand-in with no checkpoint, which renders the plain chat format."""
    return cast(LoadedModel, SimpleNamespace(checkpoint=None))


def _sentence(label: int, split: str, i: int) -> str:
    words = GOOD if label else BAD
    return (
        f"{split} one {words[i % 6]} movie {words[(i + 2) % 6]} scene {i % 9}"
    )


def _labelled(prefix: str, per_class: int, **meta: int) -> list[Example]:
    """Labelled examples in every split, with distinct uids and texts."""
    return [
        Example(
            uid=f"{prefix}-{split}-{label}-{i}",
            text=_sentence(label, split, i),
            label=label,
            split=split,  # type: ignore[arg-type]
            meta=dict(meta),
        )
        for split in SPLITS
        for label in (1, 0)
        for i in range(per_class)
    ]


def _utility(per_split: int) -> list[Example]:
    return [
        Example(
            uid=f"u-{split}-{label}-{i}",
            text=_sentence(label, split, i),
            label=label,
            split=split,  # type: ignore[arg-type]
            meta={"pair": f"{split}-{i}"},
        )
        for split in SPLITS
        for i in range(per_split)
        for label in (1, 0)
    ]


def _sources() -> dict[str, list[Example]]:
    return {
        "sst2": [e for e in _labelled("s", 300) if e.split in ("train", "val")],
        "facts": _labelled("f", 150),
        "ethics_utility": _utility(150),
        "ethics_morality": _labelled("m", 150, is_short=1),
    }


def _same(mine: TextSet, reader: StimulusSet) -> None:
    assert mine.uids == reader.uids
    assert np.array_equal(mine.labels, reader.labels)
    assert np.array_equal(mine.groups, reader.groups)


def test_seed_accuracy_gives_half_credit_to_a_tie() -> None:
    train = ["good movie", "good movie", "bad movie", "bad movie"] * 3
    labels = np.array([1, 1, 0, 0] * 3, dtype=np.int64)
    test = ["good movie", "bad movie", "so so", "so so"]
    got = seed_accuracy(
        train,
        labels,
        TextSet(
            texts=tuple(test),
            uids=("a", "b", "c", "d"),
            labels=np.array([1, 0, 1, 0], dtype=np.int64),
            groups=np.array([[0, 1], [2, 3]], dtype=np.int64),
        ),
        analyzer="word",
    )
    assert got == 0.75


def test_seed_accuracy_rejects_an_unknown_analyzer() -> None:
    labels = np.array([1, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="analyzer"):
        seed_accuracy(
            ["a", "b"],
            labels,
            TextSet(
                texts=("a", "b"),
                uids=("a", "b"),
                labels=labels,
                groups=np.array([[0, 1]], dtype=np.int64),
            ),
            analyzer="bytes",  # type: ignore[arg-type]
        )


def test_interval_is_the_wilson_interval_of_the_expected_wins() -> None:
    assert interval(0.75, 2) == pytest.approx(wilson_ci(1.5, 2))
    assert interval(0.75, 200) == pytest.approx(wilson_ci(150, 200))
    lo, hi = interval(0.75, 200)
    assert lo < 0.75 < hi


def test_the_reader_items_of_r1_are_the_items_of_the_baseline() -> None:
    sources = _sources()
    config = r1_sentiment.Config()
    items = concept_items("sentiment_r1", sources, Config())
    train = r1_sentiment.select_examples(
        sources["sst2"], split="train", min_words=config.min_words
    )
    val = r1_sentiment.select_examples(
        sources["sst2"], split="val", min_words=config.min_words
    )
    assert len(items.train) == config.n_seeds
    for i, mine in enumerate(items.train):
        _same(
            mine,
            r1_sentiment._stimuli(
                train,
                PLAIN,
                config.n_pairs,
                derive_seed(config.seed, "train", i),
            ),
        )
    _same(
        items.test,
        r1_sentiment._stimuli(
            val,
            PLAIN,
            config.n_eval_pairs,
            derive_seed(config.seed, "eval"),
        ),
    )


def test_the_reader_items_of_r2_are_the_items_of_the_baseline() -> None:
    sources = _sources()
    config = r2_utility.Config()
    items = concept_items("utility", sources, Config())
    rows = sources["ethics_utility"]

    def part(split: str) -> list[Example]:
        return [e for e in rows if e.split == split]

    for i, mine in enumerate(items.train):
        _same(
            mine,
            r2_utility._stimuli(
                part("train"),
                PLAIN,
                config.n_pairs,
                derive_seed(config.seed, "train", i),
            ),
        )
    _same(
        items.test,
        r2_utility._stimuli(
            part("test"),
            PLAIN,
            config.n_eval_pairs,
            derive_seed(config.seed, "eval", "test"),
        ),
    )


def test_the_reader_items_of_r3_are_the_items_of_the_baseline() -> None:
    sources = _sources()
    config = r3_morality.Config()
    items = concept_items("morality", sources, Config())
    rows = [e for e in sources["ethics_morality"] if e.meta["is_short"] == 1]

    def part(split: str) -> list[Example]:
        return [e for e in rows if e.split == split]

    for i, mine in enumerate(items.train):
        _same(
            mine,
            r3_morality._stimuli(
                part("train"),
                _loaded(),
                config.n_pairs,
                derive_seed(config.seed, "train", i),
                config.match_length,
            ),
        )
    _same(
        items.test,
        r3_morality._stimuli(
            part("test"),
            _loaded(),
            config.n_eval_pairs,
            derive_seed(config.seed, "eval", "test"),
            config.match_length,
        ),
    )


def test_the_reader_items_of_r7_are_the_items_of_the_baseline() -> None:
    sources = _sources()
    config = r7_truth.Config()
    items = concept_items("truth", sources, Config())
    rows = sources["facts"]

    def part(split: str) -> list[Example]:
        return [e for e in rows if e.split == split]

    for i, mine in enumerate(items.train):
        _same(
            mine,
            r7_truth._stimuli(
                part("train"),
                _loaded(),
                config.concept,
                config.n_pairs,
                derive_seed(config.seed, "train", i),
            ),
        )
    _same(
        items.test,
        r7_truth._stimuli(
            part("test"),
            _loaded(),
            config.concept,
            config.n_eval_pairs,
            derive_seed(config.seed, "eval", "test"),
        ),
    )


def test_the_transfer_items_of_r5_are_the_items_of_the_baseline() -> None:
    sources = _sources()
    config = r5_honesty.Config()
    items = concept_items("honesty", sources, Config())
    rows = sources["facts"]

    def part(split: str) -> list[Example]:
        return [e for e in rows if e.split == split]

    assert len(items.train) == config.n_seeds
    for i, mine in enumerate(items.train):
        _same(
            mine,
            r5_honesty._plain_stimuli(
                part("train"),
                _loaded(),
                config.n_pairs,
                derive_seed(config.seed, "train", i),
            ),
        )
    _same(
        items.test,
        r5_honesty._plain_stimuli(
            part("test"),
            _loaded(),
            config.n_eval_pairs,
            derive_seed(config.seed, "eval", "transfer", "test"),
        ),
    )


def test_the_held_out_sentiment_items_are_the_report_items_of_p3() -> None:
    sources = _sources()
    config = p3_size.Config()
    items = concept_items("sentiment", sources, Config())
    task = READING_TASKS["sentiment"]
    (variant,), _ = template_variants(task, PLAIN, kinds=("paper",))
    splits = task_splits(task, sources, seed=config.seed)
    assert len(items.train) == config.n_seeds
    for r, mine in enumerate(items.train):
        _same(
            mine,
            task_set(
                task,
                variant,
                splits.train,
                n_pairs=64,
                seed=derive_seed(config.seed, "size", "sentiment", 64, r),
            ),
        )
    _same(
        items.test,
        task_set(
            task,
            variant,
            splits.report,
            n_pairs=eval_pairs(task, splits.report, config.n_eval_pairs),
            seed=eval_seed(task, "report", config.seed),
        ),
    )


def test_the_pairs_do_not_depend_on_the_chat_format() -> None:
    rows = [e for e in _sources()["facts"] if e.split == "test"]

    def build(chat: ChatFormat) -> StimulusSet:
        return build_stimulus_set(
            rows,
            PAPER_TEMPLATES["concept"],
            chat,
            pairing="label_contrastive",
            concept="truthfulness",
            n_pairs=20,
            seed=3,
        )

    plain, bare = build(PLAIN), build(ChatFormat("none"))
    assert plain.texts != bare.texts
    assert plain.uids == bare.uids
    assert np.array_equal(plain.pairs, bare.pairs)


def test_the_text_of_an_item_is_the_raw_statement() -> None:
    sources = _sources()
    items = concept_items("truth", sources, Config())
    text_of = {e.uid: e.text for e in sources["facts"]}
    assert items.test.texts == tuple(text_of[u] for u in items.test.uids)
    assert all("USER:" not in t for t in items.test.texts)


def test_every_concept_has_items_with_one_positive_per_group() -> None:
    sources = _sources()
    for concept in CONCEPTS:
        items = concept_items(concept, sources, Config())
        labels = items.test.labels
        groups = items.test.groups
        assert items.train, concept
        assert (labels[groups].sum(axis=1) == 1).all(), concept


def test_an_unknown_concept_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown concept"):
        concept_items("emotion", _sources(), Config())


def test_an_empty_split_is_an_error_that_names_the_concept() -> None:
    sources = _sources()
    sources["facts"] = [e for e in sources["facts"] if e.split != "test"]
    with pytest.raises(ValueError, match=r"truth.*test"):
        concept_items("truth", sources, Config())


def test_a_missing_source_is_an_error() -> None:
    sources = _sources()
    del sources["ethics_morality"]
    with pytest.raises(ValueError, match="ethics_morality"):
        concept_items("morality", sources, Config())


def test_the_config_rejects_an_unknown_concept_and_an_empty_list() -> None:
    with pytest.raises(ValueError, match="unknown concept"):
        Config(concepts=("truth", "nope"))
    with pytest.raises(ValueError, match="at least one"):
        Config(concepts=())


def _run(tmp_path: Path, concepts: Sequence[str]) -> RunDir:
    config = Config(concepts=tuple(concepts))
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, _sources())
    return run_dir


def test_execute_writes_one_row_per_concept_and_analyzer(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path, ("truth", "utility"))
    table = run_dir.read_json("table")
    assert table["columns"] == list(text_baselines.COLUMNS)
    rows = [
        dict(zip(table["columns"], row, strict=True)) for row in table["rows"]
    ]
    assert [(r["concept"], r["analyzer"]) for r in rows] == [
        ("truth", "word"),
        ("truth", "char"),
        ("utility", "word"),
        ("utility", "char"),
    ]
    for row in rows:
        assert 0.0 <= row["lo"] <= row["accuracy"] <= row["hi"] <= 1.0
        assert row["seed_min"] <= row["accuracy"] <= row["seed_max"]
        assert row["n_seeds"] == 10
        assert row["n_train_pairs"] == 64
    assert {r["n_pairs"] for r in rows if r["concept"] == "truth"} == {150}
    assert {r["source"] for r in rows} == {"r7", "r2"}


def test_execute_reads_the_surface_form_of_a_separable_concept(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path, ("truth",))
    rows = run_dir.read_json("table")["rows"]
    assert all(row[3] > 0.95 for row in rows)


def test_execute_writes_a_summary_that_states_the_scope(
    tmp_path: Path,
) -> None:
    run_dir = _run(tmp_path, ("truth",))
    summary = run_dir.read_json("summary")
    assert summary["model_independent"] is True
    assert summary["concepts"] == ["truth"]
    assert summary["analyzers"] == ["word", "char"]
    assert summary["ties"] == "share the credit"
    assert summary["interval"] == "wilson 95 percent"
    json.dumps(summary)


def test_the_experiment_is_registered_with_the_command_line() -> None:
    experiment = load_experiment("tb1")
    assert experiment.config_cls is Config
    assert experiment.plot is text_baselines.plot


def test_plot_draws_nothing_for_a_table_only_run(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, ("truth",))
    assert text_baselines.plot(run_dir, tmp_path) == []
