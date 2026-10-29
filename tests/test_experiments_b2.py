"""Tests for the B2 finder baselines experiment."""

import dataclasses
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.activations import Activations
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import b2_finder_baselines as b2
from tomography.experiments import r1_sentiment
from tomography.experiments.b2_finder_baselines import Config
from tomography.experiments.b2_finder_baselines import completion_texts
from tomography.experiments.b2_finder_baselines import dataset_items
from tomography.experiments.base import derive_seed
from tomography.experiments.sweep import sweep_from_activations
from tomography.experiments.tb1_text_baselines import TextSet
from tomography.experiments.tb1_text_baselines import interval
from tomography.experiments.tb1_text_baselines import seed_accuracy
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import task_splits
from tomography.reading.ccs import CCSProbe
from tomography.reading.metrics import bootstrap_ci
from tomography.reading.metrics import group_accuracy
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


def _part(
    h: NDArray[np.float64],
    labels: NDArray[np.int64],
    texts: tuple[str, ...],
    uids: tuple[str, ...],
) -> b2.Part:
    """A truth part whose groups are the rows (2i, 2i + 1), statement first."""
    n = labels.shape[0]
    groups = np.arange(n, dtype=np.int64).reshape(-1, 2)
    stimuli = StimulusSet(
        texts=texts,
        response_spans=(None,) * n,
        concept_spans=(None,) * n,
        labels=labels,
        pairs=groups,
        groups=groups,
        uids=uids,
    )
    zeros = np.zeros(n, dtype=np.int64)
    act = Activations(
        data=h.astype(np.float32),
        layers=LAYERS,
        position="last",
        row_stimulus=np.arange(n, dtype=np.int64),
        row_rank=zeros,
        row_token=zeros,
        row_token_id=zeros,
        n_stimuli=n,
    )
    return b2.Part(stimuli, act, b2.pair_contrast(stimuli, act))


def _offset_part(
    n_pairs: int, seed: int, *, d: int = 8, offset: float = 3.0
) -> b2.Part:
    """Statement and negation sets offset along the truth direction.

    Row 2i is a statement and row 2i + 1 its negation; the statement is true
    for even i. Truth adds 1 along feature 0, and the statements sit `offset`
    above and the negations `offset` below on that feature, so the raw rows
    of a pair always rank the statement first, while each set centred on its
    own mean gives the truth.
    """
    rng = np.random.default_rng(seed)
    truth = np.where(np.arange(n_pairs) % 2 == 0, 1.0, -1.0)
    base = rng.standard_normal((len(LAYERS), n_pairs, d))
    shift = np.zeros_like(base)
    shift[..., 0] = truth + offset
    h = np.empty((len(LAYERS), 2 * n_pairs, d))
    h[:, 0::2] = base + shift + 0.1 * rng.standard_normal(base.shape)
    h[:, 1::2] = base - shift + 0.1 * rng.standard_normal(base.shape)
    labels = np.empty(2 * n_pairs, dtype=np.int64)
    labels[0::2] = truth > 0
    labels[1::2] = truth < 0
    rows = range(2 * n_pairs)
    uids = tuple(f"got-{'neg_' * (r % 2)}cities-{seed}-{r // 2}" for r in rows)
    texts = tuple(
        f"claim {r // 2} {'does not hold' if r % 2 else 'holds'}" for r in rows
    )
    return _part(h, labels, texts, uids)


def _offset_dataset(n_seeds: int = 1) -> b2.Dataset:
    train = tuple(_offset_part(40, seed=10 + i) for i in range(n_seeds))
    val, test = _offset_part(40, seed=2), _offset_part(40, seed=3)
    text_of = {
        uid: text
        for part in (*train, val, test)
        for uid, text in zip(part.stimuli.uids, part.stimuli.texts, strict=True)
    }
    return b2.Dataset("cities", train, val, test, text_of)


def test_ccs_accuracy_comes_from_pair_probabilities_on_offset_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes: list[CCSProbe] = []
    fit_ccs = b2.fit_ccs

    def spy(*args: Any, **kwargs: Any) -> CCSProbe:
        probes.append(fit_ccs(*args, **kwargs))
        return probes[-1]

    monkeypatch.setattr(b2, "fit_ccs", spy)
    data = _offset_dataset()
    config = _config(
        datasets=("cities",), n_seeds=1, ccs_restarts=2, ccs_epochs=500
    )

    scores = b2.score_dataset(config, data)

    (probe,) = probes
    test = data.test
    probability = probe.pair_probabilities(test.act.data, test.contrast.pairs)
    first_true = test.stimuli.labels[test.contrast.pairs[:, 0]] == 1
    by_pairs = ((probability > 0.5) == first_true).mean(axis=1)
    np.testing.assert_allclose(scores.test["ccs"][0], by_pairs)
    assert by_pairs.min() >= 0.95
    by_reader = group_accuracy(
        probe.reader.score(test.act.data),
        test.stimuli.groups,
        test.stimuli.labels,
    )
    assert by_reader.max() <= 0.6


def test_the_finders_and_two_controls_come_from_the_shared_sweep() -> None:
    data = _offset_dataset(n_seeds=2)
    config = _config(datasets=("cities",), seed=3)

    scores = b2.score_dataset(config, data)

    def prepared(part: b2.Part) -> tuple[StimulusSet, Activations]:
        return part.stimuli, part.act

    evals = {"val": prepared(data.val), "test": prepared(data.test)}
    for finder in b2.FINDERS:
        sweep = sweep_from_activations(
            [prepared(part) for part in data.train],
            evals,
            method=finder,
            seed=config.seed,
        )
        for split, accuracy in (("val", scores.val), ("test", scores.test)):
            series = sweep.per_seed(split)
            np.testing.assert_array_equal(accuracy[finder], series[0])
            if finder == "pca_diff":
                np.testing.assert_array_equal(accuracy["random"], series[1])
                np.testing.assert_array_equal(accuracy["shuffled"], series[2])
    assert scores.layers == LAYERS
    assert set(scores.val) == set(scores.test) == set(b2.LAYER_SERIES)


def test_the_text_controls_are_the_tb1_classifiers_on_the_test_pairs() -> None:
    data = _offset_dataset(n_seeds=2)

    scores = b2.score_dataset(_config(datasets=("cities",)), data)

    test = TextSet(
        texts=tuple(data.text_of[uid] for uid in data.test.stimuli.uids),
        uids=data.test.stimuli.uids,
        labels=data.test.stimuli.labels,
        groups=data.test.stimuli.groups,
    )
    assert b2.TEXT_CONTROLS == ("text_word", "text_char")
    assert set(scores.text) == set(b2.TEXT_CONTROLS)
    for analyzer in ("word", "char"):
        expected = [
            seed_accuracy(
                [data.text_of[uid] for uid in part.stimuli.uids],
                part.stimuli.labels,
                test,
                analyzer=analyzer,
            )
            for part in data.train
        ]
        np.testing.assert_allclose(scores.text[f"text_{analyzer}"], expected)


def _planted_scores() -> b2.Scores:
    """Random accuracies with a validation peak at layer 2 for LAT."""
    rng = np.random.default_rng(0)
    val = {s: 0.5 * rng.random((3, 3)) for s in b2.LAYER_SERIES}
    val["pca_diff"][:, 1] = 0.9
    test = {s: rng.random((3, 3)) for s in b2.LAYER_SERIES}
    text = {c: rng.random(3) for c in b2.TEXT_CONTROLS}
    return b2.Scores((1, 2, 3), val, test, text, rng.random((3, 3)))


def test_rows_give_the_validation_layer_wilson_and_bootstrap() -> None:
    scores = _planted_scores()
    config = _config(n_boot=200, seed=4)

    rows = b2.table_rows("cities", scores, 40, config)

    assert [row[1] for row in rows] == [*b2.LAYER_SERIES, *b2.TEXT_CONTROLS]
    table = {row[1]: dict(zip(b2.COLUMNS, row, strict=True)) for row in rows}
    assert table["pca_diff"]["val_layer"] == 2
    for name, row in table.items():
        if name in b2.TEXT_CONTROLS:
            assert row["val_layer"] is None
            per_seed = scores.text[name]
        else:
            at = int(np.argmax(scores.val[name].mean(axis=0)))
            assert row["val_layer"] == scores.layers[at]
            per_seed = scores.test[name][:, at]
        accuracy = float(per_seed.mean())
        assert row["accuracy"] == pytest.approx(accuracy)
        assert [row["wilson_lo"], row["wilson_hi"]] == pytest.approx(
            interval(accuracy, 40)
        )
        lo, hi = bootstrap_ci(per_seed, n_boot=200, seed=4)
        assert [row["boot_lo"], row["boot_hi"]] == pytest.approx([lo, hi])
        assert (row["dataset"], row["n_test_pairs"], row["n_seeds"]) == (
            "cities",
            40,
            3,
        )


def test_lat_is_the_reader_published_finders_baselines_the_rest_controls() -> (
    None
):
    expected = {
        "pca_diff": "reader",
        "mean_diff": "baseline",
        "whitened_mean_diff": "baseline",
        "logistic": "baseline",
        "ccs": "baseline",
        "random": "control",
        "shuffled": "control",
        "text_word": "control",
        "text_char": "control",
    }
    assert dict(b2.KIND_OF) == expected

    rows = b2.table_rows("cities", _planted_scores(), 40, _config())

    assert {row[1]: row[2] for row in rows} == expected
