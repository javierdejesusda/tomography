"""Tests for the P9 item pool."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from tomography.datasets.records import QAExample
from tomography.datasets.records import Split
from tomography.datasets.records import assign_splits
from tomography.knowledge import ItemLabel
from tomography.knowledge import Label
from tomography.model import LoadedModel
import tomography.p9.pool as pool_module
from tomography.p9.pool import GROUPS
from tomography.p9.pool import GuessVerdict
from tomography.p9.pool import LabeledItem
from tomography.p9.pool import PoolLoaders
from tomography.p9.pool import PoolSizes
from tomography.p9.pool import assign_award_kinds
from tomography.p9.pool import build_pool
from tomography.p9.pool import group_of
from tomography.p9.pool import name_token_counter
from tomography.p9.pool import pool_from_records
from tomography.p9.pool import pool_records
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


RELATIONS = ("father", "mother", "occupation", "sport")


def _words(name: str) -> int:
    return len(name.split())


def _make(prefix: str, n: int, source: str, **fields: Any) -> list[QAExample]:
    """Builds `n` synthetic questions; `fields` hold per-index callables."""
    uids = [f"{prefix}:{i}" for i in range(n)]
    splits = assign_splits(uids, dataset=prefix)
    return [
        QAExample(
            uid=uid,
            question=fields["question"](i),
            answers=fields["answers"](i) if "answers" in fields else (),
            kind=fields["kind"](i),
            split=split,
            meta={"source": source, **fields["meta"](i)},
        )
        for i, (uid, split) in enumerate(zip(uids, splits, strict=True))
    ]


def _popqa(n: int = 480) -> list[QAExample]:
    """PopQA-like items; one in four relation cells is popular."""
    return _make(
        "popqa",
        n,
        "popqa",
        question=lambda i: (
            f"Who is the {RELATIONS[i % 4]} of Person Number{i}?"
        ),
        answers=lambda i: (f"Answer {i}",),
        kind=lambda i: "real",
        meta=lambda i: {
            "relation": RELATIONS[i % 4],
            "subject": f"Person Number{i}",
            "popularity": 50_000 if (i // 4) % 4 == 0 else 100,
        },
    )


def _loaders(popqa: list[QAExample] | None = None) -> PoolLoaders:
    synthal = _make(
        "synthal",
        240,
        "synthal",
        question=lambda i: f"Who is the {RELATIONS[i % 4]} of Zed Qux{i}?",
        kind=lambda i: "fictional",
        meta=lambda i: {
            "relation": RELATIONS[i % 4],
            "subject": f"Zed Qux{i}",
            "popularity": None,
        },
    )
    awards = _make(
        "wikidata:award",
        105,
        "wikidata",
        question=lambda i: f"Who won the Award{i // 35} in {1990 + i % 35}?",
        answers=lambda i: (f"Laureate {i}",),
        kind=lambda i: "real",
        meta=lambda i: {
            "relation": f"Award{i // 35}",
            "year": 1990 + i % 35,
            "popularity": None,
        },
    )
    details = _make(
        "wikidata:detail",
        60,
        "wikidata",
        question=lambda i: f"What did Person{i} eat on 3 May 1850?",
        kind=lambda i: "unknowable_detail",
        meta=lambda i: {"relation": "ordinary_date", "popularity": 5_000},
    )
    falseqa = _make(
        "falseqa",
        80,
        "falseqa",
        question=lambda i: f"Why did the moon landing happen in {1900 + i}?",
        kind=lambda i: "false_premise",
        meta=lambda i: {},
    )
    kuq = _make(
        "kuq",
        120,
        "kuq",
        question=lambda i: f"What will happen in {2100 + i}?",
        kind=lambda i: "kuq:future" if i < 60 else "kuq:assumption",
        meta=lambda i: {"category": "future" if i < 60 else "assumption"},
    )
    trivia = _make(
        "triviaqa",
        300,
        "triviaqa",
        question=lambda i: f"Trivia question {i}?",
        answers=lambda i: (f"T{i}",),
        kind=lambda i: "open",
        meta=lambda i: {},
    )
    nq = _make(
        "nq_open",
        200,
        "nq_open",
        question=lambda i: f"NQ question {i}?",
        answers=lambda i: (f"N{i}",),
        kind=lambda i: "open",
        meta=lambda i: {},
    )
    source = _popqa() if popqa is None else popqa
    return PoolLoaders(
        popqa=lambda: source,
        synthal=lambda: synthal,
        kuq=lambda: kuq,
        falseqa=lambda: falseqa,
        triviaqa=lambda: trivia,
        nq_open=lambda: nq,
        awards=lambda: awards,
        details=lambda: details,
    )


SIZES = PoolSizes(
    popqa_per_relation=100,
    kuq_per_category=40,
    falseqa=50,
    triviaqa=200,
    nq_open=100,
)


def _counts(pool: list[QAExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in pool:
        counts[group_of(example)] = counts.get(group_of(example), 0) + 1
    return counts


def test_pool_sizes_defaults() -> None:
    sizes = PoolSizes()

    assert sizes.popqa_per_relation is None
    assert sizes.synthal_per_relation is None
    assert sizes.kuq_per_category == 200
    assert sizes.falseqa == 500
    assert sizes.triviaqa == 1000
    assert sizes.nq_open == 1000
    assert sizes.popular_min == 10_000
    assert sizes.popular_share == 0.5


def test_build_pool_sizes_follow_the_caps() -> None:
    pool = build_pool(_loaders(), SIZES, count_tokens=_words, seed=3)

    assert _counts(pool) == {
        "popqa": 400,
        "fictional": 400,
        "synthal": 240,
        "award": 105,
        "detail": 60,
        "falseqa": 50,
        "kuq": 80,
        "triviaqa": 200,
        "nq_open": 100,
    }


def test_build_pool_orders_the_groups() -> None:
    pool = build_pool(_loaders(), SIZES, count_tokens=_words, seed=3)

    order = list(dict.fromkeys(group_of(e) for e in pool))
    assert order == [
        "popqa",
        "fictional",
        "synthal",
        "award",
        "detail",
        "falseqa",
        "kuq",
        "triviaqa",
        "nq_open",
    ]


def test_build_pool_is_deterministic_and_seeded() -> None:
    first = build_pool(_loaders(), SIZES, count_tokens=_words, seed=3)

    assert first == build_pool(_loaders(), SIZES, count_tokens=_words, seed=3)
    other = build_pool(_loaders(), SIZES, count_tokens=_words, seed=4)
    assert [e.uid for e in first] != [e.uid for e in other]


def test_build_pool_defaults_keep_every_popqa_item() -> None:
    pool = build_pool(_loaders(), PoolSizes(), count_tokens=_words, seed=0)

    assert _counts(pool)["popqa"] == 480
    assert _counts(pool)["synthal"] == 240


def test_build_pool_generated_names_match_their_popqa_item() -> None:
    pool = build_pool(_loaders(), SIZES, count_tokens=_words, seed=3)

    popqa = {e.uid: e for e in pool if group_of(e) == "popqa"}
    generated = [e for e in pool if group_of(e) == "fictional"]
    assert len(generated) == len(popqa)
    for example in generated:
        real = popqa[str(example.meta["matched_uid"])]
        assert example.uid == f"fictional:{real.uid}"
        assert example.answers == ()
        assert example.split == real.split
        assert example.meta["relation"] == real.meta["relation"]
        assert _words(str(example.meta["subject"])) == _words(
            str(real.meta["subject"])
        )
        assert str(real.meta["subject"]) not in example.question
        assert "Answer" not in example.question


def test_build_pool_drops_one_token_subjects_from_the_generator_only() -> None:
    stubs = [
        dataclasses.replace(e, meta={**e.meta, "subject": "Paris"})
        if i % 10 == 0
        else e
        for i, e in enumerate(_popqa())
    ]

    pool = build_pool(_loaders(stubs), SIZES, count_tokens=_words, seed=3)

    one_token = [
        e
        for e in pool
        if group_of(e) == "popqa" and e.meta["subject"] == "Paris"
    ]
    assert one_token
    assert _counts(pool)["popqa"] == 400
    assert _counts(pool)["fictional"] == 400 - len(one_token)


@pytest.mark.parametrize(
    ("share", "expected"), [(0.5, 40), (1.0, 80)], ids=["half", "all"]
)
def test_build_pool_oversamples_popular_popqa_items(
    share: float, expected: int
) -> None:
    sizes = PoolSizes(popqa_per_relation=20, popular_share=share)

    pool = build_pool(_loaders(), sizes, count_tokens=_words, seed=0)

    popqa = [e for e in pool if group_of(e) == "popqa"]
    assert len(popqa) == 80
    assert sum(_is_popular(e) for e in popqa) == expected


def test_build_pool_popular_min_sets_the_preference_threshold() -> None:
    sizes = PoolSizes(
        popqa_per_relation=20, popular_min=60_000, popular_share=1.0
    )

    pool = build_pool(_loaders(), sizes, count_tokens=_words, seed=0)

    popqa = [e for e in pool if group_of(e) == "popqa"]
    assert len(popqa) == 80
    assert sum(_is_popular(e) for e in popqa) < 80


def test_build_pool_rejects_duplicate_uids() -> None:
    popqa = _popqa()
    twin = dataclasses.replace(popqa[1], uid=popqa[0].uid)

    with pytest.raises(ValueError, match="duplicate uid 'popqa:0'"):
        build_pool(
            _loaders([popqa[0], twin, *popqa[2:]]),
            PoolSizes(),
            count_tokens=_words,
        )


def test_build_pool_rejects_an_unknown_source() -> None:
    odd = dataclasses.replace(
        _popqa()[0],
        meta={"source": "mystery", "relation": "father", "subject": "A B"},
    )

    with pytest.raises(ValueError, match="popqa:0"):
        build_pool(_loaders([odd]), PoolSizes(), count_tokens=_words)


def test_name_token_counter_counts_the_name_after_a_space(
    tiny_arch: LoadedModel,
) -> None:
    count = name_token_counter(tiny_arch)

    expected = len(
        tiny_arch.tokenizer(" Zed Qux", add_special_tokens=False)["input_ids"]
    )
    assert count("Zed Qux") == expected
    assert count("Zed") >= 1
    assert count("Zed Qux Zed Qux") > count("Zed")


def _awards() -> list[QAExample]:
    return [*_loaders().awards(), _popqa(1)[0]]


def test_assign_award_kinds_splits_at_the_cutoff_year() -> None:
    out = assign_award_kinds(_awards(), 2000)

    awards = [e for e in out if group_of(e) == "award"]
    for example in awards:
        year = example.meta["year"]
        assert isinstance(year, int)
        assert example.kind == ("post_cutoff" if year > 2000 else "real")
        assert example.meta["cutoff_year"] == 2000
    assert {e.kind for e in awards} == {"real", "post_cutoff"}
    cutoff_rows = [e for e in awards if e.meta["year"] == 2000]
    assert cutoff_rows
    assert {e.kind for e in cutoff_rows} == {"real"}


def test_assign_award_kinds_leaves_other_groups_untouched() -> None:
    items = _awards()

    out = assign_award_kinds(items, 2000)

    assert out[-1] is items[-1]
    assert [e.uid for e in out] == [e.uid for e in items]
    assert "cutoff_year" not in items[0].meta


def test_assign_award_kinds_rejects_a_non_int_year() -> None:
    bad = dataclasses.replace(
        _awards()[0], meta={"source": "wikidata", "year": "1999"}
    )

    with pytest.raises(ValueError, match="wikidata:award:0"):
        assign_award_kinds([bad], 2000)


def _labeled_items() -> list[LabeledItem]:
    pool = build_pool(_loaders(), SIZES, count_tokens=_words, seed=3)
    items = []
    for i, example in enumerate(pool):
        verdict: GuessVerdict | None = None
        if example.answers:
            known = i % 3 == 0
            label = ItemLabel(
                uid=example.uid,
                label="known" if known else "unknown",
                basis="behavior",
                greedy="gold" if known else "",
                greedy_correct=known,
                samples=("gold", "x", "") if known else ("x", "y", "z"),
                n_correct=1 if known else 0,
            )
            if known and i % 2 == 0:
                verdict = "modal_answer"
        else:
            label = ItemLabel(
                example.uid, "unknown", "construction", "", False, (), 0
            )
        items.append(LabeledItem(example, label, verdict))
    return items


def test_pool_records_are_flat_json_dicts() -> None:
    items = _labeled_items()

    records = pool_records(items)

    assert len(records) == len(items)
    assert set(records[0]) == {
        "uid",
        "question",
        "answers",
        "kind",
        "split",
        "meta",
        "label",
        "basis",
        "greedy",
        "greedy_correct",
        "samples",
        "n_correct",
        "guess",
    }
    json.dumps(records)


def test_pool_records_round_trip_through_json() -> None:
    items = _labeled_items()
    assert any(i.guess is not None for i in items)

    back = pool_from_records(json.loads(json.dumps(pool_records(items))))

    assert back == items
    assert all(isinstance(i.example.answers, tuple) for i in back)
    assert all(isinstance(i.label.samples, tuple) for i in back)
    assert all(isinstance(i.example.meta, dict) for i in back)


def test_default_loaders_bind_the_dataset_loaders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, dict[str, Any]] = {}

    def fake(name: str) -> Any:
        def load(**kwargs: Any) -> list[QAExample]:
            calls[name] = kwargs
            return []

        return load

    for name in (
        "load_popqa",
        "load_synthal",
        "load_kuq",
        "load_falseqa",
        "load_triviaqa",
        "load_nq_open",
        "load_post_cutoff",
        "load_unknowable_detail",
    ):
        monkeypatch.setattr(pool_module, name, fake(name))
    loaders = PoolLoaders.default(
        offline=True, seed=7, n_detail=11, cache=tmp_path
    )

    for field in dataclasses.fields(loaders):
        assert getattr(loaders, field.name)() == []

    assert len(calls) == 8
    for name, kwargs in calls.items():
        assert kwargs["offline"] is True, name
        assert kwargs["cache"] == tmp_path, name
    assert calls["load_post_cutoff"]["cutoff_year"] == 9999
    assert calls["load_unknowable_detail"]["seed"] == 7
    assert calls["load_unknowable_detail"]["limit"] == 11
