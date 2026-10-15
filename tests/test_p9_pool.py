"""Tests for the P9 item pool."""

import pytest

from tomography.datasets.records import QAExample
from tomography.datasets.records import Split
from tomography.knowledge import ItemLabel
from tomography.knowledge import Label
from tomography.p9.pool import GROUPS
from tomography.p9.pool import LabeledItem
from tomography.p9.pool import group_of
from tomography.p9.pool import subsample


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


def _qa(uid: str, relation: str, popularity: int | None = None) -> QAExample:
    return QAExample(
        uid=uid,
        question=f"question {uid}",
        answers=("gold",),
        kind="real",
        split="train",
        meta={
            "source": "popqa",
            "relation": relation,
            "popularity": popularity,
        },
    )


def _relation(example: QAExample) -> str:
    return str(example.meta["relation"])


def _is_popular(example: QAExample) -> bool:
    popularity = example.meta["popularity"]
    return isinstance(popularity, int) and popularity >= 10_000


def _two_relations() -> list[QAExample]:
    """Relations "a" and "b" with 20 popular and 20 obscure items each."""
    return [
        _qa(f"{relation}{i:02d}", relation, 20_000 if i < 20 else 5)
        for relation in ("a", "b")
        for i in range(40)
    ]


def test_subsample_caps_every_stratum() -> None:
    kept = subsample(_two_relations(), 7, seed=0, by=_relation)

    assert sorted(_relation(e) for e in kept) == ["a"] * 7 + ["b"] * 7


def test_subsample_keeps_the_original_order() -> None:
    items = _two_relations()

    kept = subsample(items, 7, seed=0, by=_relation)

    positions = [items.index(e) for e in kept]
    assert positions == sorted(positions)


def test_subsample_is_deterministic() -> None:
    items = _two_relations()

    first = subsample(items, 7, seed=3, by=_relation)

    assert first == subsample(items, 7, seed=3, by=_relation)


def test_subsample_does_not_depend_on_the_input_order() -> None:
    items = _two_relations()

    forward = subsample(items, 7, seed=3, by=_relation)
    backward = subsample(items[::-1], 7, seed=3, by=_relation)

    assert {e.uid for e in forward} == {e.uid for e in backward}


def test_subsample_changes_with_the_seed() -> None:
    items = _two_relations()

    first = {e.uid for e in subsample(items, 10, seed=0, by=_relation)}
    second = {e.uid for e in subsample(items, 10, seed=1, by=_relation)}

    assert first != second


def test_subsample_without_a_key_caps_the_whole_list() -> None:
    assert len(subsample(_two_relations(), 5, seed=0)) == 5


def test_subsample_none_keeps_everything() -> None:
    items = _two_relations()

    assert subsample(items, None, seed=0, by=_relation) == items


def test_subsample_keeps_a_short_stratum_whole() -> None:
    items = [*_two_relations()[:40], _qa("c00", "c"), _qa("c01", "c")]

    kept = subsample(items, 5, seed=0, by=_relation)

    assert sorted(e.uid for e in kept if _relation(e) == "c") == [
        "c00",
        "c01",
    ]


def test_subsample_rejects_a_negative_cap() -> None:
    with pytest.raises(ValueError, match="-1"):
        subsample(_two_relations(), -1, seed=0)


def test_subsample_zero_keeps_nothing() -> None:
    assert subsample(_two_relations(), 0, seed=0) == []


@pytest.mark.parametrize("share", [-0.1, 1.5], ids=["below", "above"])
def test_subsample_rejects_a_share_outside_the_unit_interval(
    share: float,
) -> None:
    with pytest.raises(ValueError, match="share"):
        subsample(_two_relations(), 4, seed=0, prefer=_is_popular, share=share)


def test_subsample_prefer_half_gives_exactly_half_popular() -> None:
    kept = subsample(
        _two_relations(),
        10,
        seed=0,
        by=_relation,
        prefer=_is_popular,
        share=0.5,
    )

    assert len(kept) == 20
    assert sum(_is_popular(e) for e in kept) == 10


def test_subsample_prefer_all_gives_only_popular_items() -> None:
    kept = subsample(
        _two_relations(),
        10,
        seed=0,
        by=_relation,
        prefer=_is_popular,
        share=1.0,
    )

    assert len(kept) == 20
    assert all(_is_popular(e) for e in kept)


def test_subsample_tops_up_from_the_others_when_few_are_preferred() -> None:
    items = [_qa(f"p{i}", "a", 20_000) for i in range(3)] + [
        _qa(f"o{i}", "a", 5) for i in range(30)
    ]

    kept = subsample(items, 10, seed=0, prefer=_is_popular, share=0.5)

    assert len(kept) == 10
    assert sum(_is_popular(e) for e in kept) == 3


def test_subsample_tops_up_from_the_preferred_when_few_are_others() -> None:
    items = [_qa(f"p{i}", "a", 20_000) for i in range(30)] + [
        _qa(f"o{i}", "a", 5) for i in range(2)
    ]

    kept = subsample(items, 10, seed=0, prefer=_is_popular, share=0.5)

    assert len(kept) == 10
    assert sum(_is_popular(e) for e in kept) == 8
