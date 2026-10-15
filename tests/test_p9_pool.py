"""Tests for the P9 item pool."""

import pytest

from tomography.datasets.records import QAExample
from tomography.datasets.records import Split
from tomography.knowledge import ItemLabel
from tomography.knowledge import Label
from tomography.p9.pool import GROUPS
from tomography.p9.pool import LabeledItem
from tomography.p9.pool import group_of


def _example(
    uid: str,
    source: str | None,
    kind: str = "real",
    split: Split = "train",
) -> QAExample:
    meta = {} if source is None else {"source": source}
    return QAExample(
        uid=uid,
        question=f"question {uid}",
        answers=("gold",),
        kind=kind,
        split=split,
        meta=meta,
    )


def _label(uid: str, label: Label = "known") -> ItemLabel:
    correct = label == "known"
    return ItemLabel(
        uid=uid,
        label=label,
        basis="behavior",
        greedy="gold" if correct else "other",
        greedy_correct=correct,
        samples=("gold",) * 5 if correct else ("other",) * 5,
        n_correct=5 if correct else 0,
    )


@pytest.mark.parametrize(
    ("source", "kind", "expected"),
    [
        ("popqa", "real", "popqa"),
        ("synthal", "fictional", "synthal"),
        ("generated", "fictional", "fictional"),
        ("kuq", "kuq:future unknown", "kuq"),
        ("falseqa", "false_premise", "falseqa"),
        ("triviaqa", "open", "triviaqa"),
        ("nq_open", "open", "nq_open"),
        ("wikidata", "real", "award"),
        ("wikidata", "post_cutoff", "award"),
        ("wikidata", "unknowable_detail", "detail"),
    ],
    ids=[
        "popqa",
        "synthal",
        "generated_is_fictional",
        "kuq",
        "falseqa",
        "triviaqa",
        "nq_open",
        "wikidata_real_is_award",
        "wikidata_post_cutoff_is_award",
        "wikidata_detail",
    ],
)
def test_group_of_maps_every_source(
    source: str, kind: str, expected: str
) -> None:
    assert group_of(_example("u", source, kind)) == expected
    assert expected in GROUPS


def test_groups_lists_the_nine_groups_in_pool_order() -> None:
    assert GROUPS == (
        "popqa",
        "synthal",
        "fictional",
        "award",
        "detail",
        "falseqa",
        "kuq",
        "triviaqa",
        "nq_open",
    )


@pytest.mark.parametrize(
    "source", ["mystery", None], ids=["unknown_source", "missing_source"]
)
def test_group_of_unknown_source_names_the_uid(source: str | None) -> None:
    with pytest.raises(ValueError, match="item-7"):
        group_of(_example("item-7", source))


def test_labeled_item_group_follows_the_example() -> None:
    item = LabeledItem(_example("a", "popqa"), _label("a"))

    assert item.group == "popqa"
    assert item.guess is None


def test_labeled_item_rejects_a_label_of_another_uid() -> None:
    with pytest.raises(ValueError, match="b"):
        LabeledItem(_example("a", "popqa"), _label("b"))


@pytest.mark.parametrize("label", ["unknown", "uncertain"])
def test_labeled_item_rejects_a_verdict_on_a_label_that_is_not_known(
    label: Label,
) -> None:
    with pytest.raises(ValueError, match="known"):
        LabeledItem(_example("a", "popqa"), _label("a", label), "modal_answer")


@pytest.mark.parametrize("label", ["known", "unknown", "uncertain"])
def test_contrast_label_is_the_raw_label_without_a_verdict(
    label: Label,
) -> None:
    item = LabeledItem(_example("a", "popqa"), _label("a", label))

    assert item.contrast_label == label


@pytest.mark.parametrize(
    "verdict",
    ["excluded_stratum", "guessable_stratum", "modal_answer"],
)
def test_contrast_label_of_a_failed_known_is_uncertain(verdict: str) -> None:
    item = LabeledItem(
        _example("a", "popqa"),
        _label("a", "known"),
        verdict,  # type: ignore[arg-type]
    )

    assert item.contrast_label == "uncertain"
    assert item.label.label == "known"
