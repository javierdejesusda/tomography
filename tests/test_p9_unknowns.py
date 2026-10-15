"""Tests for the contrasts of the P9 reading study."""

import dataclasses

import pytest

from tomography.datasets.records import MetaValue
from tomography.datasets.records import QAExample
from tomography.datasets.records import Split
from tomography.knowledge import ItemLabel
from tomography.knowledge import Label
from tomography.p9.pool import GuessVerdict
from tomography.p9.pool import LabeledItem
from tomography.p9.unknowns import ENTITY_TYPES
from tomography.p9.unknowns import POP_EDGES
from tomography.p9.unknowns import UNKNOWN_TYPES
from tomography.p9.unknowns import TypePool
from tomography.p9.unknowns import panel_of
from tomography.p9.unknowns import stratum_name
from tomography.p9.unknowns import type_pools

KNOWLEDGE_NAMES = [
    "fictional_synthal",
    "fictional_generated",
    "obscure_real",
    "obscure_real_pop",
    "post_cutoff",
    "unknowable_detail",
    "false_premise",
    "kuq_false_assumption",
    "kuq_future_unknown",
]


def _item(
    uid: str,
    source: str,
    kind: str,
    label: Label,
    *,
    split: Split = "train",
    guess: GuessVerdict | None = None,
    **meta: MetaValue,
) -> LabeledItem:
    example = QAExample(
        uid=uid,
        question=f"question {uid}",
        answers=("gold",),
        kind=kind,
        split=split,
        meta={"source": source, **meta},
    )
    correct = label == "known"
    item_label = ItemLabel(
        uid=uid,
        label=label,
        basis="behavior",
        greedy="gold" if correct else "other",
        greedy_correct=correct,
        samples=("gold",) * 5 if correct else ("other",) * 5,
        n_correct=5 if correct else 0,
    )
    return LabeledItem(example, item_label, guess)


def _popqa(
    uid: str,
    label: Label,
    relation: str,
    popularity: int,
    *,
    split: Split = "train",
    guess: GuessVerdict | None = None,
) -> LabeledItem:
    return _item(
        uid,
        "popqa",
        "real",
        label,
        split=split,
        guess=guess,
        relation=relation,
        popularity=popularity,
    )


def _items() -> list[LabeledItem]:
    return [
        _popqa("pq-f-k0", "known", "father", 50_000),
        _popqa("pq-f-k1", "known", "father", 5_000, split="val"),
        _popqa("pq-f-k2", "known", "father", 50, split="test"),
        _popqa("pq-f-u0", "unknown", "father", 10),
        _popqa("pq-f-u1", "unknown", "father", 500, split="val"),
        _popqa("pq-f-x0", "uncertain", "father", 700),
        _popqa("pq-s-k0", "known", "sport", 90_000, guess="excluded_stratum"),
        _popqa("pq-s-k1", "known", "sport", 80_000, guess="excluded_stratum"),
        _popqa("pq-s-u0", "unknown", "sport", 20),
        _item("sy-0", "synthal", "fictional", "unknown", relation="father"),
        _item(
            "sy-1",
            "synthal",
            "fictional",
            "unknown",
            split="test",
            relation="sport",
        ),
        _item("sy-2", "synthal", "fictional", "uncertain", relation="father"),
        _item("fi-0", "generated", "fictional", "unknown", relation="father"),
        _item(
            "fi-1",
            "generated",
            "fictional",
            "unknown",
            split="val",
            relation="sport",
        ),
        _item(
            "aw-r-k0",
            "wikidata",
            "real",
            "known",
            relation="Award0",
            year=2005,
        ),
        _item(
            "aw-r-u0",
            "wikidata",
            "real",
            "unknown",
            split="val",
            relation="Award0",
            year=2006,
        ),
        _item(
            "aw-r-x0",
            "wikidata",
            "real",
            "uncertain",
            relation="Award0",
            year=2007,
        ),
        _item(
            "aw-p-u0",
            "wikidata",
            "post_cutoff",
            "unknown",
            split="test",
            relation="Award0",
            year=2020,
        ),
        _item(
            "aw-p-k0",
            "wikidata",
            "post_cutoff",
            "known",
            relation="Award0",
            year=2021,
        ),
        _item(
            "de-0",
            "wikidata",
            "unknowable_detail",
            "unknown",
            relation="ordinary_date",
        ),
        _item(
            "de-1",
            "wikidata",
            "unknowable_detail",
            "known",
            relation="ordinary_date",
        ),
        _item("fq-0", "falseqa", "false_premise", "unknown"),
        _item(
            "ku-fa-0",
            "kuq",
            "kuq:false assumption",
            "unknown",
            category="false assumption",
        ),
        _item(
            "ku-fu-0",
            "kuq",
            "kuq:future unknown",
            "unknown",
            category="future unknown",
        ),
        _item(
            "ku-fu-1",
            "kuq",
            "kuq:future unknown",
            "known",
            category="future unknown",
        ),
        _item("tq-0", "triviaqa", "open", "known"),
        _item("tq-1", "triviaqa", "open", "unknown", split="val"),
        _item("nq-0", "nq_open", "open", "known", split="test"),
    ]


def _uids(side: tuple[LabeledItem, ...]) -> set[str]:
    return {item.example.uid for item in side}


def test_constants_name_the_types_in_table_order() -> None:
    assert POP_EDGES == (1_000, 10_000, 100_000)
    assert ENTITY_TYPES == (
        "fictional_synthal",
        "fictional_generated",
        "obscure_real",
        "obscure_real_pop",
        "post_cutoff",
    )
    assert UNKNOWN_TYPES == (
        "real_vs_synthal",
        "real_vs_generated",
        "pre_vs_post_cutoff",
    )


def test_type_pools_builds_nine_knowledge_and_three_unknown_contrasts() -> None:
    pools = type_pools(_items())

    assert list(pools) == [*KNOWLEDGE_NAMES, *UNKNOWN_TYPES]
    assert {name: pools[name].contrast for name in pools} == {
        **dict.fromkeys(KNOWLEDGE_NAMES, "knowledge"),
        **dict.fromkeys(UNKNOWN_TYPES, "unknown_type"),
    }
    assert all(pools[name].name == name for name in pools)


def test_type_pools_selects_the_sides_of_every_contrast() -> None:
    pools = type_pools(_items())
    popqa_known = {"pq-f-k0", "pq-f-k1", "pq-f-k2"}
    open_known = {"tq-0", "nq-0"}
    obscure = {"pq-f-u0", "pq-f-u1", "pq-s-u0"}

    expected = {
        "fictional_synthal": (popqa_known, {"sy-0", "sy-1"}),
        "fictional_generated": (popqa_known, {"fi-0", "fi-1"}),
        "obscure_real": (popqa_known, obscure),
        "obscure_real_pop": (popqa_known, obscure),
        "post_cutoff": ({"aw-r-k0"}, {"aw-p-u0"}),
        "unknowable_detail": (open_known, {"de-0"}),
        "false_premise": (open_known, {"fq-0"}),
        "kuq_false_assumption": (open_known, {"ku-fa-0"}),
        "kuq_future_unknown": (open_known, {"ku-fu-0"}),
        "real_vs_synthal": (obscure, {"sy-0", "sy-1"}),
        "real_vs_generated": (obscure, {"fi-0", "fi-1"}),
        "pre_vs_post_cutoff": ({"aw-r-u0"}, {"aw-p-u0"}),
    }
    for name, (positive, negative) in expected.items():
        assert _uids(pools[name].positive) == positive, name
        assert _uids(pools[name].negative) == negative, name


def test_every_item_of_a_side_carries_the_label_of_the_side() -> None:
    pools = type_pools(_items())

    for name, pool in pools.items():
        negatives = {item.contrast_label for item in pool.negative}
        assert negatives == {"unknown"}, name
        positives = {item.contrast_label for item in pool.positive}
        want = "known" if pool.contrast == "knowledge" else "unknown"
        assert positives == {want}, name


def test_a_known_that_failed_the_guess_control_is_on_no_side() -> None:
    pools = type_pools(_items())

    for pool in pools.values():
        assert not _uids(pool.positive) & {"pq-s-k0", "pq-s-k1"}
        assert not _uids(pool.negative) & {"pq-s-k0", "pq-s-k1"}


def test_uncertain_items_are_in_no_pool() -> None:
    pools = type_pools(_items())
    uncertain = {"pq-f-x0", "sy-2", "aw-r-x0"}

    for pool in pools.values():
        assert not _uids(pool.positive) & uncertain
        assert not _uids(pool.negative) & uncertain


def test_items_outside_every_contrast_stay_out() -> None:
    pools = type_pools(_items())
    used = {
        uid
        for pool in pools.values()
        for uid in _uids(pool.positive) | _uids(pool.negative)
    }

    assert not used & {"aw-p-k0", "de-1", "ku-fu-1", "tq-1"}


def test_strata_follow_the_contrast() -> None:
    pools = type_pools(_items())

    assert {name: pool.strata for name, pool in pools.items()} == {
        "fictional_synthal": "relation",
        "fictional_generated": "relation",
        "obscure_real": "relation",
        "obscure_real_pop": "relation_popularity",
        "post_cutoff": "relation",
        "unknowable_detail": "none",
        "false_premise": "none",
        "kuq_false_assumption": "none",
        "kuq_future_unknown": "none",
        "real_vs_synthal": "relation",
        "real_vs_generated": "relation",
        "pre_vs_post_cutoff": "relation",
    }


def test_knowledge_off_keeps_the_three_unknown_type_contrasts_only() -> None:
    items = _items()
    full = type_pools(items)
    pools = type_pools(items, knowledge=False)

    assert list(pools) == list(UNKNOWN_TYPES)
    assert {pool.contrast for pool in pools.values()} == {"unknown_type"}
    for name in UNKNOWN_TYPES:
        assert pools[name] == full[name]


def test_a_contrast_needs_both_sides() -> None:
    items = [
        item
        for item in _items()
        if not (item.group == "popqa" and item.contrast_label == "known")
    ]

    pools = type_pools(items)

    assert list(pools) == [
        "post_cutoff",
        "unknowable_detail",
        "false_premise",
        "kuq_false_assumption",
        "kuq_future_unknown",
        *UNKNOWN_TYPES,
    ]


def test_kuq_categories_are_sorted() -> None:
    pools = type_pools(list(reversed(_items())))

    kuq = [name for name in pools if name.startswith("kuq_")]
    assert kuq == ["kuq_false_assumption", "kuq_future_unknown"]


def test_an_empty_pool_has_no_contrast() -> None:
    assert type_pools([]) == {}
    assert type_pools([], knowledge=False) == {}


def test_split_returns_the_positive_and_negative_items_of_one_split() -> None:
    pool = type_pools(_items())["obscure_real"]

    positive, negative = pool.split("val")
    assert [item.example.uid for item in positive] == ["pq-f-k1"]
    assert [item.example.uid for item in negative] == ["pq-f-u1"]
    positive, negative = pool.split("train")
    assert [item.example.uid for item in positive] == ["pq-f-k0"]
    assert [item.example.uid for item in negative] == ["pq-f-u0", "pq-s-u0"]
    positive, negative = pool.split("test")
    assert [item.example.uid for item in positive] == ["pq-f-k2"]
    assert negative == []


def test_a_pool_is_immutable() -> None:
    pool = type_pools(_items())["obscure_real"]

    with pytest.raises(dataclasses.FrozenInstanceError):
        pool.name = "other"  # type: ignore[misc]
    assert isinstance(pool, TypePool)
    assert isinstance(pool.positive, tuple)
    assert isinstance(pool.negative, tuple)


def test_stratum_name_of_none_is_empty() -> None:
    item = _popqa("a", "known", "father", 5)

    assert stratum_name(item, "none") == ""
    assert (
        stratum_name(_item("b", "falseqa", "false_premise", "unknown"), "none")
        == ""
    )


def test_stratum_name_by_relation() -> None:
    assert (
        stratum_name(_popqa("a", "known", "father", 5), "relation") == "father"
    )


@pytest.mark.parametrize(
    ("popularity", "bin_index"),
    [
        (0, 0),
        (999, 0),
        (1_000, 1),
        (9_999, 1),
        (10_000, 2),
        (99_999, 2),
        (100_000, 3),
        (5_000_000, 3),
    ],
)
def test_stratum_name_by_relation_and_popularity_bin(
    popularity: int, bin_index: int
) -> None:
    item = _popqa("a", "known", "father", popularity)

    assert stratum_name(item, "relation_popularity") == f"father|{bin_index}"


@pytest.mark.parametrize("popularity", [None, "many", 12.5])
def test_stratum_name_without_an_integer_popularity_raises(
    popularity: MetaValue,
) -> None:
    item = _item(
        "pq-0",
        "popqa",
        "real",
        "known",
        relation="father",
        popularity=popularity,
    )

    with pytest.raises(ValueError, match="pq-0"):
        stratum_name(item, "relation_popularity")


def test_stratum_name_without_a_popularity_field_raises() -> None:
    item = _item("pq-1", "popqa", "real", "known", relation="father")

    with pytest.raises(ValueError, match="pq-1"):
        stratum_name(item, "relation_popularity")


def test_the_relation_modes_ignore_popularity() -> None:
    item = _item("pq-2", "popqa", "real", "known", relation="father")

    assert stratum_name(item, "relation") == "father"
    assert stratum_name(item, "none") == ""


@pytest.mark.parametrize("name", ENTITY_TYPES)
def test_panel_of_entity_types(name: str) -> None:
    assert panel_of(name) == "entity"


@pytest.mark.parametrize("name", UNKNOWN_TYPES)
def test_panel_of_unknown_types(name: str) -> None:
    assert panel_of(name) == "unknown"


@pytest.mark.parametrize(
    "name",
    [
        "unknowable_detail",
        "false_premise",
        "kuq_false_assumption",
        "kuq_future_unknown",
    ],
)
def test_panel_of_question_types(name: str) -> None:
    assert panel_of(name) == "question"
