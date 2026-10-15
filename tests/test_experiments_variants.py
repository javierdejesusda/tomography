"""Tests for the shared template-variant runner."""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
from pathlib import Path
from typing import Any
from typing import get_args

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.activations import Activations
from tomography.datasets import emotions
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import variants as variants_module
from tomography.experiments.base import derive_seed
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import Prepared
from tomography.experiments.sweep import ReadingConfig
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import sweep_from_activations
from tomography.experiments.variants import FAMILIES
from tomography.experiments.variants import FAMILY_OF
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import VARIANT_KINDS
from tomography.experiments.variants import TaskSource
from tomography.experiments.variants import Variant
from tomography.experiments.variants import VariantKind
from tomography.experiments.variants import VariantSweep
from tomography.experiments.variants import eval_pairs
from tomography.experiments.variants import eval_seed
from tomography.experiments.variants import expand_tasks
from tomography.experiments.variants import load_sources
from tomography.experiments.variants import pair_capacity
from tomography.experiments.variants import prepare
from tomography.experiments.variants import target_of
from tomography.experiments.variants import task_examples
from tomography.experiments.variants import task_set
from tomography.experiments.variants import task_sources
from tomography.experiments.variants import task_splits
from tomography.experiments.variants import template_variants
from tomography.experiments.variants import variant_cosines
from tomography.experiments.variants import variant_sweep
from tomography.model import LoadedModel
from tomography.reading.directions import Method
from tomography.templates import EMOTION_PERSONAS
from tomography.templates import EMOTIONS
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat
from tomography.templates import ConceptTemplate
from tomography.templates import FunctionTemplate


def test_variant_kinds_keep_the_documented_order() -> None:
    assert VARIANT_KINDS == (
        "paper",
        "no_template",
        "bare",
        "no_chat_tags",
        "neutral_frame",
        "no_trailing_question",
        "no_trailing_space",
        "paraphrase",
        "scaffold",
        "alternate",
        "concept_swap",
        "framing_swap",
    )
    assert FAMILIES == ("frame", "wording", "concept")


def test_every_kind_belongs_to_one_family() -> None:
    assert set(FAMILY_OF) == set(VARIANT_KINDS)
    assert set(FAMILY_OF.values()) == set(FAMILIES)
    frame = VARIANT_KINDS[: VARIANT_KINDS.index("no_trailing_space") + 1]
    assert [k for k, f in FAMILY_OF.items() if f == "frame"] == list(frame)
    assert [k for k, f in FAMILY_OF.items() if f == "wording"] == [
        "paraphrase",
        "scaffold",
        "alternate",
    ]
    assert [k for k, f in FAMILY_OF.items() if f == "concept"] == [
        "concept_swap",
        "framing_swap",
    ]


def test_the_emotion_order_matches_the_dataset_class_ids() -> None:
    assert EMOTIONS == emotions.EMOTIONS


def test_reading_tasks_hold_the_tasks_of_r1_r7_r2_and_r4() -> None:
    assert list(READING_TASKS) == [
        "sentiment",
        "truth",
        "truth_cities",
        "truth_common",
        "utility",
        *(f"emotion_{e}" for e in EMOTIONS),
    ]
    assert all(t.name == name for name, t in READING_TASKS.items())


def test_sentiment_is_r1s_setup_on_sst2() -> None:
    task = READING_TASKS["sentiment"]
    assert task.source == "sst2"
    assert task.template is PAPER_TEMPLATES["concept"]
    assert task.concept == "sentiment"
    assert task.noun == "positive sentiment"
    assert task.pairing == "label_contrastive"
    assert task.splits == ("train", "val", "val")
    assert task.min_words == 3
    assert task.target is None


def test_truth_is_r7s_setup_on_the_facts() -> None:
    task = READING_TASKS["truth"]
    assert task.source == "facts"
    assert task.template is PAPER_TEMPLATES["concept"]
    assert task.concept == "truthfulness"
    assert task.noun == "truthfulness"
    assert task.pairing == "label_contrastive"
    assert task.splits == ("train", "val", "test")
    assert task.min_words == 1


def test_utility_has_no_concept_slot_and_pairs_by_group() -> None:
    task = READING_TASKS["utility"]
    assert task.source == "ethics_utility"
    assert task.template is PAPER_TEMPLATES["utility"]
    assert "concept" not in task.template.slots
    assert task.concept is None
    assert task.noun == "happiness"
    assert task.word == "happiness"
    assert task.pairing == "by_group"
    assert task.splits == ("train", "val", "test")


@pytest.mark.parametrize("emotion", EMOTIONS)
def test_each_emotion_reads_one_class_against_the_rest(emotion: str) -> None:
    task = READING_TASKS[f"emotion_{emotion}"]
    assert task.source == "emotions"
    assert task.template is PAPER_TEMPLATES["emotion_concept"]
    assert task.concept == emotion
    assert task.noun == emotion
    assert task.pairing == "one_vs_rest"
    assert task.target_class == EMOTIONS.index(emotion)
    assert task.framing == EMOTIONS.index(emotion)
    assert task.alternates == (PAPER_TEMPLATES["emotion_concept_paper"],)


@pytest.mark.parametrize(
    ("name", "subset"),
    [("truth_cities", "cities"), ("truth_common", "common_claim_true_false")],
)
def test_the_extra_truth_sources_are_scored_on_the_facts_items(
    name: str, subset: str
) -> None:
    task = READING_TASKS[name]
    assert task.source == "geometry_of_truth"
    assert task.subset == subset
    assert task.target == "truth"
    assert task.template is READING_TASKS["truth"].template
    assert task.concept == READING_TASKS["truth"].concept
    assert task.pairing == READING_TASKS["truth"].pairing


def test_task_sources_are_known_source_names() -> None:
    assert {t.source for t in READING_TASKS.values()} == set(
        get_args(TaskSource)
    )


def test_expand_tasks_replaces_emotion_by_the_six_in_order() -> None:
    assert expand_tasks(["emotion", "truth"]) == (
        *(f"emotion_{e}" for e in EMOTIONS),
        "truth",
    )


def test_expand_tasks_keeps_order_and_drops_duplicates() -> None:
    got = expand_tasks(["truth", "emotion", "emotion_anger", "truth"])
    assert got == ("truth", *(f"emotion_{e}" for e in EMOTIONS))


def test_expand_tasks_rejects_an_unknown_name() -> None:
    with pytest.raises(
        ValueError,
        match=r"unknown task 'x', expected one of \[.*\] or 'emotion'",
    ):
        expand_tasks(["sentiment", "x"])


PLAIN = ChatFormat("plain")
SENTIMENT_VARIANTS = [
    "paper",
    "no_template",
    "bare",
    "no_chat_tags",
    "neutral_frame",
    "no_trailing_question",
    "no_trailing_space",
    "paraphrase_0",
    "paraphrase_1",
    "paraphrase_2",
    "scaffold_0",
    "scaffold_1",
    "scaffold_2",
    "concept_swap_0",
    "concept_swap_1",
    "concept_swap_2",
    "concept_swap_3",
]
EMOTION_VARIANTS = [
    "paper",
    "no_template",
    "bare",
    "no_chat_tags",
    "neutral_frame",
    "no_trailing_question",
    "no_trailing_space",
    "paraphrase_0",
    "paraphrase_1",
    "scaffold_0",
    "scaffold_1",
    "scaffold_2",
    "alternate_0",
    "concept_swap_0",
    "concept_swap_1",
    "concept_swap_2",
    "concept_swap_3",
    "framing_swap",
]


def _names(task: str, chat: ChatFormat = PLAIN) -> list[str]:
    variants, _ = template_variants(READING_TASKS[task], chat)
    return [v.name for v in variants]


def _template_chat() -> ChatFormat:
    def render(messages: list[dict[str, str]]) -> str:
        return f"<u>{messages[0]['content']}</u><a>"

    return ChatFormat("template", render_fn=render)


@pytest.mark.parametrize("task", ["sentiment", "truth", "truth_cities"])
def test_a_concept_task_has_seventeen_variants(task: str) -> None:
    assert _names(task) == SENTIMENT_VARIANTS


def test_an_emotion_has_eighteen_variants() -> None:
    assert _names("emotion_happiness") == EMOTION_VARIANTS


def test_utility_has_nine_variants() -> None:
    assert _names("utility") == [
        "paper",
        "no_template",
        "bare",
        "no_chat_tags",
        "neutral_frame",
        "no_trailing_question",
        "scaffold_0",
        "scaffold_1",
        "scaffold_2",
    ]


def test_the_paper_variant_comes_first_and_is_the_task_template() -> None:
    for name, task in READING_TASKS.items():
        variants, _ = template_variants(task, PLAIN)
        assert variants[0].name == "paper", name
        assert variants[0].template is task.template
        assert variants[0].chat is PLAIN
        assert variants[0].concept == task.concept


def test_variants_follow_the_kind_order_whatever_the_request_order() -> None:
    kinds: tuple[VariantKind, ...] = ("scaffold", "bare", "paper")
    variants, _ = template_variants(
        READING_TASKS["sentiment"], PLAIN, kinds=kinds
    )
    assert [v.name for v in variants] == [
        "paper",
        "bare",
        "scaffold_0",
        "scaffold_1",
        "scaffold_2",
    ]


def test_each_variant_carries_the_family_of_its_kind() -> None:
    variants, _ = template_variants(READING_TASKS["emotion_anger"], PLAIN)
    assert all(v.family == FAMILY_OF[v.kind] for v in variants)
    assert {v.kind for v in variants} == set(VARIANT_KINDS)


def test_no_family_needs_more_than_eight_series_with_paper() -> None:
    for task in READING_TASKS.values():
        variants, _ = template_variants(task, PLAIN)
        for family in FAMILIES:
            members = {v.name for v in variants if v.family == family}
            assert len(members | {"paper"}) <= 8, (task.name, family)


def test_skipped_kinds_come_with_the_reason() -> None:
    _, skipped = template_variants(READING_TASKS["sentiment"], PLAIN)
    assert skipped == {
        "alternate": "task 'sentiment' has no alternate wording",
        "framing_swap": "task 'sentiment' has no function framing",
    }
    _, skipped = template_variants(READING_TASKS["utility"], PLAIN)
    assert skipped == {
        "no_trailing_space": (
            "template 'utility' has assistant text after the tag"
        ),
        "paraphrase": "no paraphrase for template 'utility'",
        "alternate": "task 'utility' has no alternate wording",
        "concept_swap": "template 'utility' has no concept slot",
        "framing_swap": "task 'utility' has no function framing",
    }
    _, skipped = template_variants(READING_TASKS["emotion_fear"], PLAIN)
    assert skipped == {}


def test_a_requested_kind_that_does_not_apply_is_skipped_not_raised() -> None:
    variants, skipped = template_variants(
        READING_TASKS["utility"], PLAIN, kinds=("paper", "concept_swap")
    )
    assert [v.name for v in variants] == ["paper"]
    assert list(skipped) == ["concept_swap"]


def test_unknown_kinds_raise() -> None:
    kinds: Sequence[Any] = ("paper", "shuffle")
    with pytest.raises(ValueError, match="unknown variant kinds"):
        template_variants(READING_TASKS["sentiment"], PLAIN, kinds=kinds)


def test_kinds_without_paper_raise() -> None:
    with pytest.raises(
        ValueError,
        match="kinds must include 'paper', the reference of every comparison",
    ):
        template_variants(
            READING_TASKS["sentiment"], PLAIN, kinds=("bare", "no_template")
        )


def _render(task: str, name: str, chat: ChatFormat = PLAIN) -> str:
    variants, _ = template_variants(READING_TASKS[task], chat)
    (variant,) = (v for v in variants if v.name == name)
    template = variant.template
    assert isinstance(template, ConceptTemplate)
    return template.render(
        "the stimulus", variant.concept or "", variant.chat
    ).text


def test_bare_is_the_stimulus_with_nothing_around_it() -> None:
    assert _render("sentiment", "bare") == "the stimulus"


def test_no_trailing_space_drops_only_the_lone_space() -> None:
    paper = _render("sentiment", "paper")
    assert paper.endswith("ASSISTANT: ")
    assert _render("sentiment", "no_trailing_space") == paper[:-1]


def test_no_trailing_space_needs_the_plain_chat_format() -> None:
    task = READING_TASKS["sentiment"]
    for chat in (_template_chat(), ChatFormat("none")):
        variants, skipped = template_variants(task, chat)
        assert "no_trailing_space" not in [v.name for v in variants]
        assert "plain chat format" in skipped["no_trailing_space"]


def test_the_other_variants_keep_the_chat_format_they_were_given() -> None:
    chat = _template_chat()
    variants, _ = template_variants(READING_TASKS["sentiment"], chat)
    assert all(v.chat is chat for v in variants)


def test_a_variant_keeps_the_concept_only_while_it_has_the_slot() -> None:
    variants, _ = template_variants(READING_TASKS["emotion_anger"], PLAIN)
    with_slot = {
        "paper",
        "no_chat_tags",
        "no_trailing_question",
        "no_trailing_space",
        "paraphrase_0",
        "paraphrase_1",
        "alternate_0",
    }
    assert {v.name for v in variants if v.concept == "anger"} == with_slot
    assert all(v.concept is None for v in variants if v.name not in with_slot)


def test_framing_swap_is_the_function_template_trained_at_response() -> None:
    variants, _ = template_variants(READING_TASKS["emotion_anger"], PLAIN)
    by_name = {v.name: v for v in variants}
    framing = by_name["framing_swap"]
    assert isinstance(framing.template, FunctionTemplate)
    assert framing.template.pos_personas == EMOTION_PERSONAS["anger"][0]
    assert framing.position == "response"
    assert not framing.evaluable
    assert all(
        v.position is None and v.evaluable
        for name, v in by_name.items()
        if name != "framing_swap"
    )


def test_the_alternate_is_the_second_emotion_wording() -> None:
    variants, _ = template_variants(READING_TASKS["emotion_fear"], PLAIN)
    (alternate,) = (v for v in variants if v.kind == "alternate")
    assert alternate.name == "alternate_0"
    assert alternate.template is PAPER_TEMPLATES["emotion_concept_paper"]


def _sst2(n_train: int = 12, n_val: int = 20) -> list[Example]:
    """Train phrases (two too short, two of exactly three words) and val."""
    rows = []
    for label in (0, 1):
        for i in range(n_train):
            rows.append(
                Example(
                    f"sst2-train-{label}-{i}",
                    f"train phrase {label} number {i}",
                    label,
                    "train",
                )
            )
        rows.append(
            Example(f"sst2-train-short-{label}", "so good", label, "train")
        )
        rows.append(
            Example(
                f"sst2-train-edge-{label}",
                "just fine today" if label else "quite dull today",
                label,
                "train",
            )
        )
        for i in range(n_val):
            rows.append(
                Example(
                    f"sst2-val-{label}-{i}",
                    f"val sentence {label} number {i}",
                    label,
                    "val",
                )
            )
    return rows


def _facts(per_class: int = 10) -> list[Example]:
    return [
        Example(
            f"facts-{split}-{label}-{i}",
            f"fact {split} {label} {i}",
            label,
            split,
        )
        for split in ("train", "val", "test")
        for label in (0, 1)
        for i in range(per_class)
    ]


def _utility(pairs: int = 8) -> list[Example]:
    rows = []
    for split in ("train", "val", "test"):
        for i in range(pairs):
            pair = f"ethics-{split}-{i}"
            for suffix, label in (("a", 1), ("b", 0)):
                rows.append(
                    Example(
                        f"{pair}:{suffix}",
                        f"scenario {split} {i} {suffix}",
                        label,
                        split,
                        {"pair": pair},
                    )
                )
    return rows


def _emotions(per_class: int = 4) -> list[Example]:
    return [
        Example(
            f"emotion-{name}-{split}-{i}",
            f"{name} scenario {split} {i}",
            label,
            split,
            {"emotion": name},
        )
        for split in ("train", "val", "test")
        for label, name in enumerate(EMOTIONS)
        for i in range(per_class)
    ]


def _truth_sets(per_class: int = 6) -> list[Example]:
    return [
        Example(
            f"got-{subset}-{split}-{label}-{i}",
            f"{subset} claim {split} {label} {i}",
            label,
            split,
            {"subset": subset},
        )
        for subset in ("cities", "common_claim_true_false", "neg_cities")
        for split in ("train", "val", "test")
        for label in (0, 1)
        for i in range(per_class)
    ]


def _sources() -> dict[str, list[Example]]:
    return {
        "sst2": _sst2(),
        "facts": _facts(),
        "ethics_utility": _utility(),
        "emotions": _emotions(),
        "geometry_of_truth": _truth_sets(),
    }


def _uids(examples: Sequence[Example]) -> set[str]:
    return {e.uid for e in examples}


def test_task_examples_take_the_source_and_filter_the_subset() -> None:
    sources = _sources()
    assert task_examples(READING_TASKS["sentiment"], sources) == sources["sst2"]
    cities = task_examples(READING_TASKS["truth_cities"], sources)
    assert cities
    assert {e.meta["subset"] for e in cities} == {"cities"}
    assert len(cities) == 12 * 3


def test_task_examples_name_the_missing_source() -> None:
    with pytest.raises(ValueError, match="no examples for source 'sst2'"):
        task_examples(READING_TASKS["sentiment"], {"facts": _facts()})


def test_target_of_a_source_is_its_target_and_of_a_task_itself() -> None:
    assert target_of(READING_TASKS["truth_cities"]) is READING_TASKS["truth"]
    assert target_of(READING_TASKS["truth_common"]) is READING_TASKS["truth"]
    assert target_of(READING_TASKS["truth"]) is READING_TASKS["truth"]
    assert target_of(READING_TASKS["utility"]) is READING_TASKS["utility"]


def test_a_target_must_share_template_concept_and_pairing() -> None:
    task = dataclasses.replace(READING_TASKS["truth_cities"], target="utility")
    with pytest.raises(ValueError, match="differ in template, concept or"):
        target_of(task)


def test_eval_seeds_are_keyed_by_the_target_and_the_split() -> None:
    cities = READING_TASKS["truth_cities"]
    truth = READING_TASKS["truth"]
    assert eval_seed(cities, "report", 3) == derive_seed(
        3, "eval", "truth", "report"
    )
    assert eval_seed(cities, "report", 3) == eval_seed(truth, "report", 3)
    assert eval_seed(truth, "report", 3) != eval_seed(truth, "select", 3)
    assert eval_seed(truth, "report", 3) != eval_seed(truth, "report", 4)
    assert eval_seed(READING_TASKS["utility"], "report", 3) != eval_seed(
        truth, "report", 3
    )


def test_training_examples_drop_the_phrases_below_min_words() -> None:
    splits = task_splits(READING_TASKS["sentiment"], _sources(), seed=0)
    assert len(splits.train) == 26
    assert all(e.split == "train" for e in splits.train)
    assert all(len(e.text.split()) >= 3 for e in splits.train)
    kept = {e.uid for e in splits.train}
    assert {"sst2-train-edge-0", "sst2-train-edge-1"} <= kept
    assert not {"sst2-train-short-0", "sst2-train-short-1"} & kept


def test_sst2_halves_its_validation_split_by_uid() -> None:
    task = READING_TASKS["sentiment"]
    splits = task_splits(task, _sources(), seed=0)
    val = {e.uid for e in _sst2() if e.split == "val"}
    assert _uids(splits.select).isdisjoint(_uids(splits.report))
    assert _uids(splits.select) | _uids(splits.report) == val
    assert len(splits.select) == len(val) // 2
    assert all(e.split == "val" for e in (*splits.select, *splits.report))


def test_the_halves_are_deterministic_under_the_seed_and_vary_with_it() -> None:
    task = READING_TASKS["sentiment"]
    first = task_splits(task, _sources(), seed=0)
    again = task_splits(task, _sources(), seed=0)
    other = task_splits(task, _sources(), seed=1)
    assert first == again
    assert _uids(first.select) != _uids(other.select)
    assert _uids(first.train) == _uids(other.train)


def test_the_halves_do_not_depend_on_the_order_of_the_examples() -> None:
    task = READING_TASKS["sentiment"]
    shuffled = {"sst2": list(reversed(_sst2()))}
    first = task_splits(task, _sources(), seed=2)
    second = task_splits(task, shuffled, seed=2)
    assert _uids(first.select) == _uids(second.select)
    assert _uids(first.report) == _uids(second.report)


def test_a_group_task_halves_whole_groups() -> None:
    task = dataclasses.replace(
        READING_TASKS["utility"], splits=("train", "val", "val")
    )
    splits = task_splits(task, _sources(), seed=0)
    pairs = [
        {str(e.meta["pair"]) for e in part}
        for part in (splits.select, splits.report)
    ]
    assert pairs[0].isdisjoint(pairs[1])
    assert len(pairs[0]) == len(pairs[1]) == 4
    assert len(splits.select) == len(splits.report) == 8


def test_distinct_splits_are_used_as_they_are() -> None:
    sources = _sources()
    splits = task_splits(READING_TASKS["truth"], sources, seed=0)
    assert {e.split for e in splits.train} == {"train"}
    assert {e.split for e in splits.select} == {"val"}
    assert {e.split for e in splits.report} == {"test"}
    emotion = task_splits(READING_TASKS["emotion_fear"], sources, seed=0)
    assert len(emotion.train) == len(emotion.select) == 6 * 4


def test_a_source_is_trained_on_itself_and_scored_on_the_target_items() -> None:
    sources = _sources()
    cities = task_splits(READING_TASKS["truth_cities"], sources, seed=0)
    truth = task_splits(READING_TASKS["truth"], sources, seed=0)
    assert {e.meta["subset"] for e in cities.train} == {"cities"}
    assert {e.split for e in cities.train} == {"train"}
    assert cities.select == truth.select
    assert cities.report == truth.report


def test_a_source_needs_the_examples_of_its_target() -> None:
    only_got = {"geometry_of_truth": _truth_sets()}
    with pytest.raises(ValueError, match="no examples for source 'facts'"):
        task_splits(READING_TASKS["truth_cities"], only_got, seed=0)


@pytest.mark.parametrize("part", ["train", "val", "test"])
def test_a_part_without_a_pair_raises_naming_task_and_split(part: str) -> None:
    facts = [e for e in _facts() if e.split != part or e.label == 1]
    names = {"train": "train", "val": "select", "test": "report"}
    with pytest.raises(
        ValueError, match=f"task 'truth' has no pair in its {names[part]} "
    ):
        task_splits(READING_TASKS["truth"], {"facts": facts}, seed=0)


def test_pair_capacity_of_a_label_contrastive_task_is_the_rarer_label() -> None:
    task = READING_TASKS["truth"]
    rows = [Example(f"u{i}", "t", int(i < 3), "val") for i in range(10)]
    assert pair_capacity(task, rows) == 3
    assert pair_capacity(task, []) == 0


def test_pair_capacity_of_one_vs_rest_counts_the_target_class() -> None:
    task = READING_TASKS["emotion_anger"]
    labels = [2, 2, 2, 0, 1, 3, 4, 5, 5, 5]
    rows = [Example(f"u{i}", "t", c, "val") for i, c in enumerate(labels)]
    assert pair_capacity(task, rows) == 3
    rows = [Example(f"u{i}", "t", 2, "val") for i in range(4)]
    assert pair_capacity(task, rows) == 0


def test_pair_capacity_of_a_group_task_counts_the_valid_groups() -> None:
    task = READING_TASKS["utility"]

    def row(uid: str, label: int, pair: str) -> Example:
        return Example(uid, "t", label, "val", {"pair": pair})

    rows = [
        row("a1", 1, "a"),
        row("a2", 0, "a"),
        row("b1", 1, "b"),
        row("b2", 1, "b"),
        row("c1", 0, "c"),
        row("c2", 0, "c"),
        row("d1", 1, "d"),
        row("e1", 1, "e"),
        row("e2", 0, "e"),
        row("e3", 0, "e"),
    ]
    assert pair_capacity(task, rows) == 2


def test_eval_pairs_are_capped_by_the_capacity() -> None:
    task = READING_TASKS["truth"]
    rows = [Example(f"u{i}", "t", int(i < 5), "val") for i in range(20)]
    assert eval_pairs(task, rows, 256) == 5
    assert eval_pairs(task, rows, 3) == 3
    assert eval_pairs(task, rows, None) is None


CONFIG = ReadingConfig(experiment="t1", model="tiny")


def _evaluable(task: str) -> list[Variant]:
    variants, _ = template_variants(READING_TASKS[task], PLAIN)
    return [v for v in variants if v.evaluable]


def test_evaluable_variants_share_their_items() -> None:
    task = READING_TASKS["sentiment"]
    examples = task_splits(task, _sources(), seed=0).report
    sets = [
        task_set(task, v, examples, n_pairs=6, seed=3)
        for v in _evaluable("sentiment")
    ]
    assert len(sets) == 17
    first = sets[0]
    for other in sets[1:]:
        assert other.uids == first.uids
        np.testing.assert_array_equal(other.labels, first.labels)
        np.testing.assert_array_equal(other.pairs, first.pairs)
        np.testing.assert_array_equal(other.groups, first.groups)
    assert first.groups.shape == (6, 2)
    assert len({s.texts for s in sets}) > 1


def test_task_set_takes_every_pair_when_n_pairs_is_none() -> None:
    task = READING_TASKS["truth"]
    examples = task_splits(task, _sources(), seed=0).select
    paper = _evaluable("truth")[0]
    stimuli = task_set(task, paper, examples, n_pairs=None, seed=0)
    assert stimuli.groups.shape[0] == pair_capacity(task, examples) == 10


def test_task_set_is_a_function_of_the_seed() -> None:
    task = READING_TASKS["truth"]
    examples = task_splits(task, _sources(), seed=0).train
    paper = _evaluable("truth")[0]
    a = task_set(task, paper, examples, n_pairs=5, seed=1)
    b = task_set(task, paper, examples, n_pairs=5, seed=1)
    c = task_set(task, paper, examples, n_pairs=5, seed=2)
    assert a.texts == b.texts
    assert a.texts != c.texts


def test_the_concept_span_marks_the_concept_word() -> None:
    task = READING_TASKS["emotion_fear"]
    examples = task_splits(task, _sources(), seed=0).train
    variants, _ = template_variants(task, PLAIN)
    by_name = {v.name: v for v in variants}
    paper = task_set(task, by_name["paper"], examples, n_pairs=4, seed=0)
    for text, span in zip(paper.texts, paper.concept_spans, strict=True):
        assert span is not None
        assert text[span[0] : span[1]] == "fear"
    swapped = task_set(
        task, by_name["concept_swap_0"], examples, n_pairs=4, seed=0
    )
    assert swapped.concept_spans == (None,) * len(swapped.texts)
    assert "furniture" in swapped.texts[0]


def test_a_group_task_keeps_both_members_of_a_pair() -> None:
    task = READING_TASKS["utility"]
    examples = task_splits(task, _sources(), seed=0).report
    stimuli = task_set(
        task, _evaluable("utility")[0], examples, n_pairs=5, seed=0
    )
    assert stimuli.groups.shape == (5, 2)
    by_uid = {e.uid: e for e in examples}
    for group in stimuli.groups:
        members = [by_uid[stimuli.uids[i]] for i in group]
        assert len({m.meta["pair"] for m in members}) == 1
        assert sorted(m.label for m in members) == [0, 1]


def test_one_vs_rest_labels_the_target_emotion() -> None:
    task = READING_TASKS["emotion_anger"]
    examples = task_splits(task, _sources(), seed=0).train
    stimuli = task_set(
        task, _evaluable("emotion_anger")[0], examples, n_pairs=4, seed=0
    )
    by_uid = {e.uid: e for e in examples}
    assert stimuli.groups.shape == (4, 2)
    for uid, label in zip(stimuli.uids, stimuli.labels, strict=True):
        assert (by_uid[uid].meta["emotion"] == "anger") == bool(label)


def test_the_function_variant_uses_persona_pairs() -> None:
    task = READING_TASKS["emotion_fear"]
    examples = task_splits(task, _sources(), seed=0).train
    variants, _ = template_variants(task, PLAIN)
    (framing,) = (v for v in variants if v.kind == "framing_swap")
    stimuli = task_set(task, framing, examples, n_pairs=5, seed=0)
    assert len(stimuli.texts) == 10
    assert stimuli.labels.tolist() == [1, 0] * 5
    assert [sorted(pair) for pair in stimuli.pairs.tolist()] == [
        [2 * i, 2 * i + 1] for i in range(5)
    ]
    assert all(span is not None for span in stimuli.response_spans)
    assert stimuli.uids[0::2] == stimuli.uids[1::2]
    assert all("extremely" in text for text in stimuli.texts)


def test_prepare_reads_layers_one_to_n_by_default(
    tiny_arch: LoadedModel,
) -> None:
    task = READING_TASKS["sentiment"]
    examples = task_splits(task, _sources(), seed=0).train
    stimuli = task_set(
        task, _evaluable("sentiment")[0], examples, n_pairs=3, seed=0
    )
    got, act = prepare(tiny_arch, stimuli, "last", CONFIG)
    assert got is stimuli
    assert act.layers == tuple(range(1, tiny_arch.n_layers + 1))
    assert act.position == "last"
    assert act.data.shape[:2] == (tiny_arch.n_layers, len(stimuli.texts))


def test_prepare_forwards_the_reading_settings(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    real = variants_module.stimulus_activations

    def spy(
        loaded: LoadedModel, stimuli: StimulusSet, **kwargs: Any
    ) -> Activations:
        calls.append(kwargs)
        return real(loaded, stimuli, **kwargs)

    monkeypatch.setattr(variants_module, "stimulus_activations", spy)
    task = READING_TASKS["sentiment"]
    examples = task_splits(task, _sources(), seed=0).train
    stimuli = task_set(
        task, _evaluable("sentiment")[0], examples, n_pairs=3, seed=0
    )
    config = dataclasses.replace(CONFIG, layers=(2,), bos="none", batch_size=2)
    _, act = prepare(tiny_arch, stimuli, "mean", config)
    assert calls == [
        {"layers": (2,), "position": "mean", "bos": "none", "batch_size": 2}
    ]
    assert act.layers == (2,)


DIM = 64
LAYERS = (1, 2)


def _planted(
    n_pairs: int, *, seed: int, axis: int, offset: float = 3.0
) -> Prepared:
    """Pairs whose label-1 member sits `offset` along the direction `axis`.

    The in-pair order is random, so the mean difference cancels and only the
    direction of the differences carries the signal.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_pairs
    first = rng.integers(0, 2, size=n_pairs).astype(bool)
    labels = np.zeros(n, dtype=np.int64)
    labels[2 * np.arange(n_pairs)] = first
    labels[2 * np.arange(n_pairs) + 1] = ~first
    axes = np.stack(
        [
            np.linalg.qr(np.random.default_rng(1000 + k).normal(size=(DIM, 2)))[
                0
            ]
            for k in range(len(LAYERS))
        ]
    )
    direction = axes[:, :, axis]
    signed = (2 * labels - 1)[None, :, None] * offset
    data = rng.normal(size=(len(LAYERS), n, DIM)) + signed * direction[:, None]
    pairs = np.arange(n, dtype=np.int64).reshape(n_pairs, 2)
    stimuli = StimulusSet(
        texts=tuple(f"text {i}" for i in range(n)),
        response_spans=(None,) * n,
        concept_spans=(None,) * n,
        labels=labels,
        pairs=pairs,
        groups=pairs.copy(),
        uids=tuple(f"u{i // 2}" for i in range(n)),
    )
    act = Activations(
        data=data.astype(np.float32),
        layers=LAYERS,
        position="last",
        row_stimulus=np.arange(n, dtype=np.int64),
        row_rank=np.zeros(n, dtype=np.int64),
        row_token=np.zeros(n, dtype=np.int64),
        row_token_id=np.zeros(n, dtype=np.int64),
        n_stimuli=n,
    )
    return stimuli, act


def _sweep_inputs() -> tuple[
    dict[str, list[Prepared]], dict[str, dict[str, Prepared]]
]:
    """Readers of axes 0 and 1 and a train-only variant on axis 0."""
    train = {
        "paper": [_planted(40, seed=10 + i, axis=0) for i in range(3)],
        "other": [_planted(40, seed=20 + i, axis=1) for i in range(3)],
        "func": [_planted(40, seed=30 + i, axis=0) for i in range(3)],
    }
    evals = {
        split: {
            "paper": _planted(100, seed=seed, axis=0),
            "other": _planted(100, seed=seed + 1, axis=1),
        }
        for split, seed in (("select", 90), ("report", 95))
    }
    return train, evals


def _variant_sweep(
    method: Method = "logistic",
) -> VariantSweep:
    train, evals = _sweep_inputs()
    return variant_sweep(
        train, evals, method=method, preprocess="center", seed=4
    )


def test_the_diagonal_equals_a_plain_sweep_bit_for_bit() -> None:
    train, evals = _sweep_inputs()
    sweep = _variant_sweep()
    for name in ("paper", "other"):
        direct = sweep_from_activations(
            train[name],
            {f"report/{name}": evals["report"][name]},
            method="logistic",
            preprocess="center",
            seed=4,
        )
        np.testing.assert_array_equal(
            sweep.at("report", name, name), direct.per_seed(f"report/{name}")
        )


def test_the_sweep_axes_are_split_trained_evaluated_series_seed_layer() -> None:
    sweep = _variant_sweep()
    assert sweep.splits == ("select", "report")
    assert sweep.train_variants == ("paper", "other", "func")
    assert sweep.eval_variants == ("paper", "other")
    assert sweep.layers == LAYERS
    assert sweep.accuracy.shape == (2, 3, 2, len(SERIES), 3, len(LAYERS))
    assert len(sweep.readers) == 3
    assert all(len(row) == 3 for row in sweep.readers)
    assert sweep.at("report", "other", "paper").shape == (
        len(SERIES),
        3,
        len(LAYERS),
    )


def test_a_reader_transfers_to_its_own_direction_only() -> None:
    sweep = _variant_sweep()
    reader = SERIES.index("reader")
    own = sweep.at("report", "paper", "paper")[reader].mean()
    away = sweep.at("report", "paper", "other")[reader].mean()
    assert own > 0.9
    assert away < 0.65
    shuffled = sweep.at("report", "paper", "paper")[SERIES.index("shuffled")]
    assert abs(shuffled.mean() - 0.5) < 0.1


def test_select_and_report_are_scored_on_their_own_sets() -> None:
    sweep = _variant_sweep()
    select = sweep.at("select", "paper", "paper")
    report = sweep.at("report", "paper", "paper")
    assert not np.array_equal(select, report)


def test_own_scores_an_evaluable_variant_on_its_own_items() -> None:
    sweep = _variant_sweep()
    np.testing.assert_array_equal(
        sweep.own("report", "other"), sweep.at("report", "other", "other")
    )


def test_own_scores_a_train_only_variant_on_the_reference_items() -> None:
    sweep = _variant_sweep()
    np.testing.assert_array_equal(
        sweep.own("report", "func"), sweep.at("report", "func", "paper")
    )
    np.testing.assert_array_equal(
        sweep.own("select", "func", reference="other"),
        sweep.at("select", "func", "other"),
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.at("test", "paper", "paper"),
        lambda s: s.at("report", "nope", "paper"),
        lambda s: s.at("report", "paper", "func"),
        lambda s: s.own("report", "nope"),
    ],
    ids=["split", "trained", "evaluated", "variant"],
)
def test_unknown_names_raise_a_clear_error(
    call: Callable[[VariantSweep], object],
) -> None:
    with pytest.raises(ValueError, match=r"unknown .* expected one of"):
        call(_variant_sweep())


def test_directions_are_the_unit_readers_times_their_sign() -> None:
    sweep = _variant_sweep()
    directions = sweep.directions
    assert directions.shape == (3, 3, len(LAYERS), DIM)
    assert directions.dtype == np.float32
    np.testing.assert_allclose(
        np.linalg.norm(directions, axis=-1), 1.0, rtol=1e-5
    )
    for v, row in enumerate(sweep.readers):
        for n, reader in enumerate(row):
            np.testing.assert_array_equal(
                directions[v, n], reader.directions * reader.signs[:, None]
            )


def test_oriented_directions_point_from_label_0_to_label_1() -> None:
    train, _ = _sweep_inputs()
    sweep = _variant_sweep()
    for v, name in enumerate(sweep.train_variants):
        for n, (stimuli, act) in enumerate(train[name]):
            positive = act.data[:, stimuli.labels == 1].mean(axis=1)
            negative = act.data[:, stimuli.labels == 0].mean(axis=1)
            gap = np.einsum(
                "kd,kd->k", positive - negative, sweep.directions[v, n]
            )
            assert (gap > 0).all()


def test_variant_sweep_rejects_an_empty_training_input() -> None:
    _, evals = _sweep_inputs()
    with pytest.raises(ValueError, match="training variant"):
        variant_sweep({}, evals, method="logistic", preprocess="center", seed=0)


@pytest.mark.parametrize(
    "evals",
    [{}, {"select": {}, "report": {}}],
    ids=["no-splits", "no-variants"],
)
def test_variant_sweep_rejects_an_empty_evaluation_input(
    evals: dict[str, dict[str, Prepared]],
) -> None:
    train, _ = _sweep_inputs()
    with pytest.raises(ValueError, match="evaluation variant"):
        variant_sweep(
            train, evals, method="logistic", preprocess="center", seed=0
        )


def test_variant_sweep_needs_the_same_evaluation_variants_in_every_split() -> (
    None
):
    train, evals = _sweep_inputs()
    del evals["select"]["other"]
    with pytest.raises(ValueError, match="same evaluation variants"):
        variant_sweep(
            train, evals, method="logistic", preprocess="center", seed=0
        )


def _axes(
    *rows: Sequence[int], layers: int = 1, dim: int = 4
) -> NDArray[np.float32]:
    """Builds (V, N, K, d) directions: entry i of a row is the unit axis i."""
    out = np.zeros((len(rows), len(rows[0]), layers, dim), dtype=np.float32)
    for v, row in enumerate(rows):
        for n, axis in enumerate(row):
            sign = -1.0 if axis < 0 else 1.0
            out[v, n, :, abs(axis) - 1] = sign
    return out


def test_identical_readers_have_cosine_one() -> None:
    cos = variant_cosines(_axes([1, 2], [1, 2]))
    assert cos.shape == (1, 2, 2)
    assert cos.dtype == np.float64
    assert cos[0, 0, 1] == pytest.approx(1.0)


def test_a_sign_flip_gives_minus_one() -> None:
    cos = variant_cosines(_axes([1, 2], [-1, -2]))
    assert cos[0, 0, 1] == pytest.approx(-1.0)


def test_off_diagonal_pairs_readers_of_the_same_seed() -> None:
    cos = variant_cosines(_axes([1, 2], [1, 3]))
    assert cos[0, 0, 1] == pytest.approx(0.5)


def test_the_off_diagonal_is_symmetric() -> None:
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(3, 4, 2, 8))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    cos = variant_cosines(directions.astype(np.float32))
    for k in range(2):
        np.testing.assert_allclose(cos[k], cos[k].T, atol=1e-12)


def test_the_diagonal_is_the_mean_over_seed_pairs() -> None:
    cos = variant_cosines(_axes([1, 2, 1], [1, 1, 1]))
    assert cos[0, 0, 0] == pytest.approx(1 / 3)
    assert cos[0, 1, 1] == pytest.approx(1.0)


def test_each_layer_has_its_own_matrix() -> None:
    directions = np.concatenate(
        [_axes([1, 2], [1, 2]), _axes([1, 2], [2, 1])], axis=2
    )
    cos = variant_cosines(directions)
    assert cos.shape == (2, 2, 2)
    assert cos[0, 0, 1] == pytest.approx(1.0)
    assert cos[1, 0, 1] == pytest.approx(0.0)


def test_cosines_need_two_seeds() -> None:
    with pytest.raises(ValueError, match="N >= 2"):
        variant_cosines(_axes([1], [1]))


def test_cosines_need_a_four_dimensional_input() -> None:
    with pytest.raises(ValueError, match=r"\(V, N, K, d\)"):
        variant_cosines(np.zeros((2, 3, 4), dtype=np.float32))


LOADERS = {
    "sst2": "load_sst2",
    "facts": "load_facts",
    "ethics_utility": "load_ethics_utility",
    "emotions": "load_emotions",
    "geometry_of_truth": "load_geometry_of_truth",
}


def _fake_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    for source, loader in LOADERS.items():

        def fake(source: str = source, **kwargs: Any) -> list[Example]:
            calls.append((source, kwargs))
            return [Example(f"{source}-0", "text", 1, "train")]

        monkeypatch.setattr(variants_module, loader, fake)
    return calls


def test_load_sources_calls_each_built_loader_once_per_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _fake_loaders(monkeypatch)
    sources: list[TaskSource] = ["facts", "sst2", "facts", "geometry_of_truth"]
    loaded = load_sources(sources, cache=tmp_path, offline=True)
    assert list(loaded) == ["facts", "sst2", "geometry_of_truth"]
    assert [source for source, _ in calls] == list(loaded)
    assert all(
        kwargs == {"cache": tmp_path, "offline": True} for _, kwargs in calls
    )
    assert loaded["sst2"][0].uid == "sst2-0"


def test_load_sources_defaults_to_the_online_default_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_loaders(monkeypatch)
    load_sources(["emotions", "ethics_utility"])
    assert calls == [
        ("emotions", {"cache": None, "offline": False}),
        ("ethics_utility", {"cache": None, "offline": False}),
    ]


def test_load_sources_rejects_an_unknown_source() -> None:
    sources: Sequence[Any] = ["sst2", "imdb"]
    with pytest.raises(ValueError, match="unknown source 'imdb'"):
        load_sources(sources)


def test_task_sources_lists_each_source_once_targets_included() -> None:
    assert task_sources(["truth_cities", "truth", "utility"]) == [
        "geometry_of_truth",
        "facts",
        "ethics_utility",
    ]
    assert task_sources(["emotion", "sentiment"]) == ["emotions", "sst2"]


def test_task_sources_rejects_an_unknown_task() -> None:
    with pytest.raises(ValueError, match="unknown task 'nope'"):
        task_sources(["sentiment", "nope"])


def _short_emotions(per_class: int = 4) -> list[Example]:
    """Emotion scenarios short enough for the tiny models' 128 positions."""
    words = ("sun", "rain", "dog", "gift")
    return [
        Example(
            f"emotion-{name}-{split}-{i}",
            f"{words[i % 4]} {i}",
            label,
            split,
            {"emotion": name},
        )
        for split in ("train", "val", "test")
        for label, name in enumerate(EMOTIONS)
        for i in range(per_class)
    ]


def test_every_variant_reads_end_to_end_on_the_tiny_models(
    tiny_arch: LoadedModel,
) -> None:
    task = READING_TASKS["emotion_happiness"]
    config = dataclasses.replace(CONFIG, n_seeds=2, n_pairs=4, n_eval_pairs=3)
    variants, skipped = template_variants(task, chat_format_for(tiny_arch))
    assert len(variants) + len(skipped) >= 17
    splits = task_splits(task, {"emotions": _short_emotions()}, seed=0)
    train = {
        v.name: [
            prepare(
                tiny_arch,
                task_set(
                    task,
                    v,
                    splits.train,
                    n_pairs=config.n_pairs,
                    seed=derive_seed(config.seed, "train", i),
                ),
                v.position or "last",
                config,
            )
            for i in range(config.n_seeds)
        ]
        for v in variants
    }
    evals = {
        name: {
            v.name: prepare(
                tiny_arch,
                task_set(
                    task,
                    v,
                    part,
                    n_pairs=config.n_eval_pairs,
                    seed=eval_seed(task, name, config.seed),
                ),
                "last",
                config,
            )
            for v in variants
            if v.evaluable
        }
        for name, part in (("select", splits.select), ("report", splits.report))
    }
    uids = {stimuli.uids for stimuli, _ in evals["report"].values()}
    assert len(uids) == 1
    for name, sets in train.items():
        position = "response" if name == "framing_swap" else "last"
        assert all(act.position == position for _, act in sets)
    sweep = variant_sweep(
        train,
        evals,
        method="logistic",
        preprocess="center",
        seed=config.seed,
    )
    n_layers = tiny_arch.n_layers
    assert sweep.accuracy.shape == (
        2,
        len(variants),
        len(variants) - 1,
        len(SERIES),
        2,
        n_layers,
    )
    assert np.isfinite(sweep.accuracy).all()
    direct = sweep_from_activations(
        train["paper"],
        {"report/paper": evals["report"]["paper"]},
        method="logistic",
        preprocess="center",
        seed=config.seed,
    )
    np.testing.assert_array_equal(
        sweep.at("report", "paper", "paper"), direct.per_seed("report/paper")
    )
    cosines = variant_cosines(sweep.directions)
    assert cosines.shape == (n_layers, len(variants), len(variants))
    assert np.isfinite(cosines).all()
    assert np.abs(cosines).max() <= 1.0 + 1e-6
