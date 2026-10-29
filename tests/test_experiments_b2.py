"""Tests for the B2 finder baselines experiment."""

import dataclasses
from typing import Any

import numpy as np
import pytest

from tomography.datasets.records import Example
from tomography.experiments import r1_sentiment
from tomography.experiments.b2_finder_baselines import Config
from tomography.experiments.b2_finder_baselines import completion_texts
from tomography.experiments.b2_finder_baselines import dataset_items
from tomography.experiments.base import derive_seed
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import task_splits
from tomography.templates import ChatFormat

TINY_CCS: dict[str, Any] = {"ccs_restarts": 1, "ccs_epochs": 20}
PLAIN = ChatFormat("plain")
SPLITS = ("train", "val", "test")
LAYERS = (1, 2)


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        datasets=("sentiment", "cities"),
        n_seeds=2,
        n_pairs=4,
        n_boot=50,
        **TINY_CCS,
    )
    return dataclasses.replace(base, **overrides)


def _sentiment(per_label: int = 12) -> list[Example]:
    words = {
        1: ("good", "great", "fine", "warm"),
        0: ("bad", "dull", "poor", "cold"),
    }
    examples = [
        Example(uid="sst2-short", text="good film", label=1, split="train")
    ]
    for split in ("train", "val"):
        for label in (1, 0):
            for i in range(per_label):
                first, second = words[label][i % 4], words[label][(i + 1) % 4]
                examples.append(
                    Example(
                        uid=f"sst2-{split}-{label}-{i}",
                        text=f"a {first} and {second} film {i}",
                        label=label,
                        split=split,
                    )
                )
    return examples


def _truth_sets(
    families: tuple[tuple[str, str], ...] = (("cities", "neg_cities"),),
    n: int = 24,
) -> list[Example]:
    """Statement sets and their negations, aligned row by row.

    Statement i has label i % 2 and split `SPLITS[i % 3]`; its negation has
    the other label and the next split, so a pair spans two loader splits.
    """
    examples = []
    for first, second in families:
        for subset, negated in ((first, 0), (second, 1)):
            for i in range(n):
                examples.append(
                    Example(
                        uid=f"got-{subset}-{i}",
                        text=f"town {i} is {'not ' * negated}in land {i % 5}.",
                        label=(i + negated) % 2,
                        split=SPLITS[(i + negated) % 3],
                        meta={"subset": subset},
                    )
                )
    return examples


def test_config_defaults_round_trip_and_pinned_hash() -> None:
    config = Config()
    assert config.experiment == "b2"
    assert config.model == "gpt2"
    assert config.datasets == (
        "sentiment",
        "cities",
        "larger_than",
        "sp_en_trans",
    )
    assert (config.n_seeds, config.n_pairs, config.n_eval_pairs) == (
        10,
        64,
        None,
    )
    assert (config.ccs_restarts, config.ccs_epochs) == (10, 1000)
    assert config.ccs_lr == 0.01
    assert Config.from_dict(config.to_dict()) == config
    assert config.config_hash() == "317ba03fe78b"


@pytest.mark.parametrize(
    ("datasets", "message"),
    [
        ((), "at least one dataset"),
        (("sentiment", "facts"), "unknown dataset 'facts'"),
        (("cities", "cities"), "repeated dataset 'cities'"),
    ],
)
def test_the_dataset_list_is_checked(
    datasets: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(datasets=datasets)


def test_truth_pairs_join_each_statement_to_its_negation_in_its_split() -> None:
    rows = _truth_sets()
    statements = {e.uid: e for e in rows if e.meta["subset"] == "cities"}
    items = dataset_items(_config(), "cities", PLAIN, [], rows)

    parts = {"train": items.train, "val": (items.val,), "test": (items.test,)}
    for split, sets in parts.items():
        for stimuli in sets:
            assert stimuli.groups.shape[1] == 2
            for first, second in stimuli.groups:
                uid = stimuli.uids[first]
                assert uid in statements
                assert statements[uid].split == split
                assert stimuli.uids[second] == uid.replace(
                    "cities", "neg_cities"
                )
                assert stimuli.labels[first] + stimuli.labels[second] == 1
            assert all("truthfulness" in text for text in stimuli.texts)
    assert [s.groups.shape[0] for s in items.train] == [4, 4]
    held_out = sorted(items.test.uids[g[0]] for g in items.test.groups)
    assert held_out == sorted(
        uid for uid, e in statements.items() if e.split == "test"
    )
    assert items.val.groups.shape[0] == 8
    assert items.text_of["got-neg_cities-3"] == "town 3 is not in land 3."


@pytest.mark.parametrize("broken", ["shorter", "same_label"])
def test_a_negation_set_must_negate_its_statements_row_by_row(
    broken: str,
) -> None:
    rows = _truth_sets()
    if broken == "shorter":
        rows = [e for e in rows if e.uid != "got-neg_cities-23"]
    else:
        rows = [
            dataclasses.replace(e, label=1 - e.label)
            if e.uid == "got-neg_cities-5"
            else e
            for e in rows
        ]
    with pytest.raises(ValueError, match="neg_cities does not negate cities"):
        dataset_items(_config(), "cities", PLAIN, [], rows)


def test_a_truth_set_without_examples_is_an_error() -> None:
    with pytest.raises(ValueError, match="no examples for larger_than"):
        dataset_items(_config(), "larger_than", PLAIN, [], _truth_sets())


def test_sentiment_trains_on_the_r1_sets_and_halves_the_validation_split() -> (
    None
):
    examples = _sentiment()
    config = _config()
    items = dataset_items(config, "sentiment", PLAIN, examples, [])

    train = r1_sentiment.select_examples(
        examples, split="train", min_words=r1_sentiment.Config().min_words
    )
    assert len(items.train) == config.n_seeds
    for i, stimuli in enumerate(items.train):
        expected = r1_sentiment._stimuli(
            train, PLAIN, config.n_pairs, derive_seed(config.seed, "train", i)
        )
        assert stimuli.texts == expected.texts
        assert stimuli.uids == expected.uids
        np.testing.assert_array_equal(stimuli.labels, expected.labels)
        np.testing.assert_array_equal(stimuli.pairs, expected.pairs)
    halves = task_splits(
        READING_TASKS["sentiment"], {"sst2": examples}, seed=config.seed
    )
    assert set(items.val.uids) <= {e.uid for e in halves.select}
    assert set(items.test.uids) <= {e.uid for e in halves.report}
    assert items.test.groups.shape[0] == min(
        sum(e.label == label for e in halves.report) for label in (0, 1)
    )
    assert items.text_of["sst2-train-1-0"] == "a good and great film 0"


def test_sentiment_contrast_pairs_render_each_text_with_both_completions() -> (
    None
):
    items = dataset_items(_config(), "sentiment", PLAIN, _sentiment(), [])

    texts = completion_texts(items.test, items.text_of, PLAIN)

    assert len(texts) == 2 * len(items.test.uids)
    for j, uid in enumerate(items.test.uids):
        assert texts[2 * j] == (
            "USER: Consider the following example: ''' "
            f"{items.text_of[uid]} '''\nBetween negative and positive, the "
            "sentiment of this example is ASSISTANT: positive"
        )
        assert texts[2 * j + 1] == texts[2 * j].removesuffix("positive") + (
            "negative"
        )
