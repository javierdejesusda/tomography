import collections
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace

import numpy as np
import pytest

from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.pairing import persona_pairs
from tomography.reading.metrics import group_accuracy
from tomography.reading.reader import RepReader
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat
from tomography.templates import ConceptTemplate
from tomography.templates import FunctionTemplate


@dataclass(frozen=True)
class Ex:
    uid: str
    text: str
    label: int
    meta: Mapping[str, str | int | float | None] = field(default_factory=dict)


def fake_qwen3_render(messages: list[dict[str, str]]) -> str:
    prompt = "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
    )
    return prompt + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


CHATS = {
    "plain": ChatFormat("plain"),
    "none": ChatFormat("none"),
    "template": ChatFormat("template", render_fn=fake_qwen3_render),
}
CONCEPT = PAPER_TEMPLATES["concept"]
HONESTY = PAPER_TEMPLATES["honesty"]
PERSONAS = FunctionTemplate(
    "personas",
    "Pretend you're {persona} person.",
    pos_personas=("an honest", "a truthful", "a candid"),
    neg_personas=("an untruthful", "a deceitful", "a lying"),
)
TINY = ConceptTemplate("tiny", "{stimulus}")
TINY_PERSONAS = FunctionTemplate(
    "tiny_personas",
    "{persona}",
    pos_personas=("p0", "p1", "p2"),
    neg_personas=("n0", "n1", "n2"),
)


def labeled(n: int, positives: int) -> list[Ex]:
    return [
        Ex(f"u{i}", f"text number {i}", int(i < positives)) for i in range(n)
    ]


def span_text(stimuli: StimulusSet, index: int) -> str:
    span = stimuli.response_spans[index]
    assert span is not None
    return stimuli.texts[index][span[0] : span[1]]


def grouped(sizes: list[int]) -> tuple[list[Ex], list[str]]:
    examples: list[Ex] = []
    group_ids: list[str] = []
    for g, size in enumerate(sizes):
        for c in range(size):
            examples.append(Ex(f"g{g}:{c}", f"a{g}{c}", int(c == g % size)))
            group_ids.append(f"g{g}")
    return examples, group_ids


def fingerprint(stimuli: StimulusSet) -> tuple[object, ...]:
    return (
        stimuli.texts,
        stimuli.uids,
        stimuli.labels.tolist(),
        stimuli.pairs.tolist(),
        stimuli.groups.tolist(),
    )


EIGHT = [Ex(f"u{i}", f"t{i}", int(i < 4)) for i in range(8)]
GROUPED = grouped([4, 3, 4])
EXISTING_MODES = {
    "label_contrastive": (
        {"examples": EIGHT, "template": TINY, "n_pairs": 3, "seed": 11},
        (
            ("t0", "t7", "t1", "t5", "t3", "t6"),
            ("u0", "u7", "u1", "u5", "u3", "u6"),
            [1, 0, 1, 0, 1, 0],
            [[0, 1], [2, 3], [4, 5]],
            [[0, 1], [2, 3], [4, 5]],
        ),
    ),
    "random": (
        {"examples": EIGHT, "template": TINY, "n_pairs": 3, "seed": 12},
        (
            ("t2", "t6", "t3", "t5", "t1", "t7"),
            ("u2", "u6", "u3", "u5", "u1", "u7"),
            [1, 0, 1, 0, 1, 0],
            [[0, 1], [2, 3], [4, 5]],
            [[5, 2], [4, 1], [0, 3]],
        ),
    ),
    "one_vs_rest": (
        {
            "examples": [Ex(f"u{i}", f"t{i}", i % 3) for i in range(9)],
            "template": TINY,
            "target_class": 2,
            "n_pairs": 3,
            "seed": 13,
        },
        (
            ("t6", "t8", "t2", "t0", "t5", "t7"),
            ("u6", "u8", "u2", "u0", "u5", "u7"),
            [0, 1, 1, 0, 1, 0],
            [[0, 1], [2, 3], [4, 5]],
            [[0, 1], [2, 3], [4, 5]],
        ),
    ),
    "persona": (
        {
            "examples": EIGHT[:4],
            "template": TINY_PERSONAS,
            "n_pairs": 3,
            "seed": 14,
        },
        (
            ("p1 t0", "n1 t0", "p0 t3", "n2 t3", "p2 t2", "n1 t2"),
            ("u0", "u0", "u3", "u3", "u2", "u2"),
            [1, 0, 1, 0, 1, 0],
            [[0, 1], [3, 2], [4, 5]],
            [[0, 1], [3, 2], [4, 5]],
        ),
    ),
    "by_group": (
        {
            "examples": GROUPED[0],
            "template": TINY,
            "group_ids": GROUPED[1],
            "seed": 15,
        },
        (
            (
                *("a00", "a01", "a02", "a03"),
                *("a20", "a21", "a22", "a23"),
                *("a10", "a11", "a12"),
            ),
            (
                *("g0:0", "g0:1", "g0:2", "g0:3"),
                *("g2:0", "g2:1", "g2:2", "g2:3"),
                *("g1:0", "g1:1", "g1:2"),
            ),
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [[0, 3], [6, 7], [9, 8]],
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, -1]],
        ),
    ),
}


@pytest.mark.parametrize("pairing", list(EXISTING_MODES))
def test_existing_pairing_modes_keep_their_output(pairing: str) -> None:
    arguments, expected = EXISTING_MODES[pairing]

    stimuli = build_stimulus_set(
        chat=ChatFormat("none"),
        pairing=pairing,  # type: ignore[arg-type]
        **arguments,  # type: ignore[arg-type]
    )

    assert fingerprint(stimuli) == expected


def test_label_contrastive_set_has_pair_major_layout() -> None:
    examples = labeled(10, 5)

    stimuli = build_stimulus_set(
        examples,
        CONCEPT,
        ChatFormat("plain"),
        pairing="label_contrastive",
        concept="sentiment",
        n_pairs=3,
        seed=1,
    )

    assert len(stimuli.texts) == 6
    assert stimuli.pairs.dtype == np.int64
    np.testing.assert_array_equal(stimuli.pairs, np.arange(6).reshape(3, 2))
    np.testing.assert_array_equal(stimuli.groups, stimuli.pairs)
    assert stimuli.labels.dtype == np.int64
    assert (stimuli.labels[stimuli.pairs].sum(axis=1) == 1).all()


def test_stimuli_carry_their_source_example() -> None:
    examples = labeled(6, 3)
    by_uid = {e.uid: e for e in examples}

    stimuli = build_stimulus_set(
        examples,
        CONCEPT,
        ChatFormat("plain"),
        pairing="label_contrastive",
        concept="sentiment",
        seed=0,
    )

    assert len(stimuli.uids) == len(stimuli.texts) == 6
    for i, uid in enumerate(stimuli.uids):
        assert span_text(stimuli, i) == by_uid[uid].text
        assert stimuli.labels[i] == by_uid[uid].label
        concept_span = stimuli.concept_spans[i]
        assert concept_span is not None
        assert stimuli.texts[i][concept_span[0] : concept_span[1]] == (
            "sentiment"
        )


def test_label_contrastive_defaults_to_all_available_pairs() -> None:
    stimuli = build_stimulus_set(
        labeled(9, 3),
        CONCEPT,
        ChatFormat("none"),
        pairing="label_contrastive",
        concept="mood",
    )

    assert len(stimuli.pairs) == 3
    assert len(stimuli.texts) == 6


def test_build_is_deterministic_and_seed_dependent() -> None:
    examples = labeled(40, 20)

    def build(seed: int) -> StimulusSet:
        return build_stimulus_set(
            examples,
            CONCEPT,
            ChatFormat("none"),
            pairing="label_contrastive",
            concept="mood",
            n_pairs=8,
            seed=seed,
        )

    assert build(3).texts == build(3).texts
    assert build(3).texts != build(4).texts


def test_build_does_not_mutate_its_inputs() -> None:
    examples = labeled(10, 5)
    snapshot = list(examples)

    build_stimulus_set(
        examples,
        CONCEPT,
        ChatFormat("none"),
        pairing="label_contrastive",
        concept="mood",
        seed=2,
    )

    assert examples == snapshot


def test_strata_keep_pairs_inside_one_stratum() -> None:
    examples = [
        Ex(f"u{i}", f"text {i}", i % 2, {"relation": f"r{(i // 2) % 3}"})
        for i in range(24)
    ]
    relations = [str(e.meta["relation"]) for e in examples]

    stimuli = build_stimulus_set(
        examples,
        CONCEPT,
        ChatFormat("none"),
        pairing="label_contrastive",
        concept="knowledge",
        strata=relations,
        n_pairs=6,
        seed=0,
    )

    by_uid = {e.uid: e for e in examples}
    for first, second in stimuli.pairs:
        meta_first = by_uid[stimuli.uids[first]].meta["relation"]
        meta_second = by_uid[stimuli.uids[second]].meta["relation"]
        assert meta_first == meta_second
    assert len(stimuli.pairs) == 6


@pytest.mark.parametrize("pairing", ["random", "one_vs_rest"])
def test_strata_are_only_valid_for_label_contrastive(pairing: str) -> None:
    examples = labeled(6, 3)

    with pytest.raises(ValueError, match="strata"):
        build_stimulus_set(
            examples,
            CONCEPT,
            ChatFormat("none"),
            pairing=pairing,  # type: ignore[arg-type]
            concept="mood",
            target_class=1,
            strata=["a"] * 6,
        )


def test_concept_is_required_for_a_concept_slot() -> None:
    with pytest.raises(ValueError, match="concept"):
        build_stimulus_set(
            labeled(6, 3),
            CONCEPT,
            ChatFormat("none"),
            pairing="label_contrastive",
        )


def test_concept_is_optional_without_a_concept_slot() -> None:
    examples = [
        Ex(f"u{i}", f"Choice {i}", i % 2, {"question": "Q?"}) for i in range(6)
    ]

    stimuli = build_stimulus_set(
        examples,
        PAPER_TEMPLATES["utility"],
        ChatFormat("plain"),
        pairing="label_contrastive",
    )

    assert all(span is None for span in stimuli.concept_spans)


def test_random_pairing_uses_every_selected_stimulus_once() -> None:
    stimuli = build_stimulus_set(
        labeled(9, 4),
        CONCEPT,
        ChatFormat("none"),
        pairing="random",
        concept="mood",
        seed=0,
    )

    assert len(stimuli.pairs) == 4
    assert len(stimuli.texts) == 8
    assert sorted(stimuli.pairs.ravel().tolist()) == list(range(8))


def test_random_pairing_groups_hold_the_label_matches() -> None:
    stimuli = build_stimulus_set(
        labeled(20, 10),
        CONCEPT,
        ChatFormat("none"),
        pairing="random",
        concept="mood",
        n_pairs=5,
        seed=0,
    )

    assert stimuli.groups.shape[1] == 2
    assert len(stimuli.groups) == min(
        int(stimuli.labels.sum()), int((1 - stimuli.labels).sum())
    )
    assert (stimuli.labels[stimuli.groups].sum(axis=1) == 1).all()


def test_random_pairing_groups_are_empty_when_labels_do_not_match() -> None:
    stimuli = build_stimulus_set(
        [Ex(f"u{i}", f"t{i}", 1) for i in range(6)],
        CONCEPT,
        ChatFormat("none"),
        pairing="random",
        concept="mood",
    )

    assert stimuli.groups.shape == (0, 2)


def test_random_pairing_rejects_too_many_pairs() -> None:
    with pytest.raises(ValueError, match="n_pairs"):
        build_stimulus_set(
            labeled(6, 3),
            CONCEPT,
            ChatFormat("none"),
            pairing="random",
            concept="mood",
            n_pairs=4,
        )


def test_random_pairing_rejects_a_single_example() -> None:
    with pytest.raises(ValueError, match="at least two"):
        build_stimulus_set(
            labeled(1, 1),
            CONCEPT,
            ChatFormat("none"),
            pairing="random",
            concept="mood",
        )


@pytest.mark.parametrize(
    "pairing",
    ["by-group", "onevsrest", "", "Persona"],
    ids=["hyphen", "no-underscore", "empty", "capital"],
)
def test_unknown_pairing_is_rejected(pairing: str) -> None:
    with pytest.raises(ValueError, match="pairing"):
        build_stimulus_set(
            labeled(6, 3),
            CONCEPT,
            ChatFormat("none"),
            pairing=pairing,  # type: ignore[arg-type]
            concept="mood",
        )


def test_one_vs_rest_labels_mark_the_target_class() -> None:
    examples = [Ex(f"u{i}", f"scenario {i}", i % 3) for i in range(30)]

    stimuli = build_stimulus_set(
        examples,
        CONCEPT,
        ChatFormat("none"),
        pairing="one_vs_rest",
        concept="fear",
        target_class=2,
        n_pairs=7,
        seed=0,
    )

    by_uid = {e.uid: e for e in examples}
    assert len(stimuli.pairs) == 7
    assert len(stimuli.texts) == 14
    for i, uid in enumerate(stimuli.uids):
        assert stimuli.labels[i] == int(by_uid[uid].label == 2)
    assert (stimuli.labels[stimuli.pairs].sum(axis=1) == 1).all()
    np.testing.assert_array_equal(stimuli.groups, stimuli.pairs)


def test_one_vs_rest_requires_a_target_class() -> None:
    with pytest.raises(ValueError, match="target_class"):
        build_stimulus_set(
            labeled(6, 3),
            CONCEPT,
            ChatFormat("none"),
            pairing="one_vs_rest",
            concept="mood",
        )


def test_non_binary_labels_are_rejected_outside_one_vs_rest() -> None:
    examples = [Ex(f"u{i}", f"t{i}", i % 3) for i in range(6)]

    with pytest.raises(ValueError, match="binary"):
        build_stimulus_set(
            examples,
            CONCEPT,
            ChatFormat("none"),
            pairing="label_contrastive",
            concept="mood",
        )


@pytest.mark.parametrize("chat", CHATS.values(), ids=CHATS)
def test_persona_pairing_renders_both_poles_of_each_statement(
    chat: ChatFormat,
) -> None:
    examples = [Ex(f"s{i}", f"statement {i}", 1) for i in range(5)]

    stimuli = build_stimulus_set(
        examples, HONESTY, chat, pairing="persona", seed=0
    )

    assert len(stimuli.texts) == 10
    np.testing.assert_array_equal(stimuli.pairs, persona_pairs(10))
    np.testing.assert_array_equal(stimuli.groups, stimuli.pairs)
    np.testing.assert_array_equal(stimuli.labels, [1, 0] * 5)
    for i in range(10):
        assert span_text(stimuli, i) == f"statement {stimuli.uids[i][1:]}"
        assert stimuli.concept_spans[i] is None
    assert "an honest" in stimuli.texts[0]
    assert "an untruthful" in stimuli.texts[1]
    assert stimuli.uids[0] == stimuli.uids[1]
    assert len(set(stimuli.uids)) == 5


def test_persona_pairing_ignores_the_example_labels() -> None:
    examples = [Ex(f"s{i}", f"statement {i}", 0) for i in range(3)]

    stimuli = build_stimulus_set(
        examples, HONESTY, ChatFormat("none"), pairing="persona"
    )

    np.testing.assert_array_equal(stimuli.labels, [1, 0] * 3)


def test_persona_pairing_n_pairs_selects_that_many_statements() -> None:
    examples = [Ex(f"s{i}", f"statement {i}", 1) for i in range(20)]

    stimuli = build_stimulus_set(
        examples,
        HONESTY,
        ChatFormat("none"),
        pairing="persona",
        n_pairs=4,
        seed=7,
    )

    assert len(stimuli.pairs) == 4
    assert len(stimuli.texts) == 8
    assert len(set(stimuli.uids)) == 4


def test_persona_pairing_picks_personas_by_seed() -> None:
    examples = [Ex(f"s{i}", f"statement {i}", 1) for i in range(30)]

    def build(seed: int) -> StimulusSet:
        return build_stimulus_set(
            examples, PERSONAS, ChatFormat("none"), pairing="persona", seed=seed
        )

    first = build(1)
    assert first.texts == build(1).texts
    assert first.texts != build(2).texts
    used = {text.split()[3] for text in first.texts[0::2]}
    assert len(used) > 1


def test_persona_pairing_fills_the_instruction_slot_from_meta() -> None:
    examples = [
        Ex("q0", "Paris.", 1, {"instruction": "Name a capital."}),
        Ex("q1", "Rome.", 1, {"instruction": "Name another capital."}),
    ]

    stimuli = build_stimulus_set(
        examples,
        PAPER_TEMPLATES["honesty_control"],
        ChatFormat("plain"),
        pairing="persona",
    )

    for text, uid in zip(stimuli.texts, stimuli.uids, strict=True):
        assert ("Name a capital." in text) == (uid == "q0")


def test_missing_instruction_raises_naming_the_example() -> None:
    examples = [Ex("q9", "Paris.", 1)]

    with pytest.raises(ValueError, match="q9"):
        build_stimulus_set(
            examples,
            PAPER_TEMPLATES["honesty_control"],
            ChatFormat("plain"),
            pairing="persona",
        )


def test_function_templates_need_persona_pairing() -> None:
    with pytest.raises(ValueError, match="persona"):
        build_stimulus_set(
            labeled(6, 3),
            HONESTY,
            ChatFormat("none"),
            pairing="label_contrastive",
        )


def test_concept_templates_reject_persona_pairing() -> None:
    with pytest.raises(ValueError, match="persona"):
        build_stimulus_set(
            labeled(6, 3),
            CONCEPT,
            ChatFormat("none"),
            pairing="persona",
            concept="mood",
        )


def choice_examples() -> tuple[list[Ex], list[str]]:
    examples: list[Ex] = []
    group_ids: list[str] = []
    for g, size in enumerate([4, 5, 4, 4]):
        for c in range(size):
            examples.append(
                Ex(
                    f"g{g}:{c}",
                    f"answer {g}-{c}",
                    int(c == g % size),
                    {"question": f"question {g}?"},
                )
            )
            group_ids.append(f"g{g}")
    return examples, group_ids


def test_by_group_builds_ragged_groups_and_matching_pairs() -> None:
    examples, group_ids = choice_examples()

    stimuli = build_stimulus_set(
        examples,
        PAPER_TEMPLATES["truthfulqa"],
        ChatFormat("none"),
        pairing="by_group",
        group_ids=group_ids,
        seed=0,
    )

    assert stimuli.groups.shape == (4, 5)
    assert stimuli.pairs.shape == (4, 2)
    assert len(stimuli.texts) == 17
    assert sorted(int(x) for x in stimuli.groups[stimuli.groups >= 0]) == list(
        range(17)
    )
    sizes = sorted(int((row >= 0).sum()) for row in stimuli.groups)
    assert sizes == [4, 4, 4, 5]
    for row, pair in zip(stimuli.groups, stimuli.pairs, strict=True):
        members = [int(i) for i in row if i >= 0]
        assert int(stimuli.labels[members].sum()) == 1
        assert pair[0] in members
        assert pair[1] in members
        assert stimuli.labels[pair].sum() == 1
        assert len({stimuli.uids[i].split(":")[0] for i in members}) == 1


def test_by_group_fills_the_question_slot_from_meta() -> None:
    examples, group_ids = choice_examples()

    stimuli = build_stimulus_set(
        examples,
        PAPER_TEMPLATES["truthfulqa"],
        ChatFormat("none"),
        pairing="by_group",
        group_ids=group_ids,
    )

    for i, text in enumerate(stimuli.texts):
        group = stimuli.uids[i].split(":")[0][1:]
        assert f"Question: question {group}?" in text
        assert span_text(stimuli, i).startswith("answer ")


def test_by_group_n_pairs_selects_that_many_groups() -> None:
    examples, group_ids = choice_examples()

    stimuli = build_stimulus_set(
        examples,
        PAPER_TEMPLATES["arc"],
        ChatFormat("none"),
        pairing="by_group",
        group_ids=group_ids,
        n_pairs=2,
        seed=5,
    )

    assert stimuli.pairs.shape == (2, 2)
    assert stimuli.groups.shape[0] == 2
    assert len(stimuli.texts) == int((stimuli.groups >= 0).sum())


def test_by_group_requires_one_positive_and_a_negative_per_group() -> None:
    examples, group_ids = choice_examples()
    broken = [*examples[:1], Ex("x", "extra", 1, {"question": "q"})]
    broken_ids = [group_ids[0], group_ids[0]]

    with pytest.raises(ValueError, match="g0"):
        build_stimulus_set(
            broken,
            PAPER_TEMPLATES["truthfulqa"],
            ChatFormat("none"),
            pairing="by_group",
            group_ids=broken_ids,
        )


def test_by_group_needs_group_ids_of_the_right_length() -> None:
    examples, group_ids = choice_examples()

    with pytest.raises(ValueError, match="group_ids"):
        build_stimulus_set(
            examples,
            PAPER_TEMPLATES["truthfulqa"],
            ChatFormat("none"),
            pairing="by_group",
        )
    with pytest.raises(ValueError, match="group_ids"):
        build_stimulus_set(
            examples,
            PAPER_TEMPLATES["truthfulqa"],
            ChatFormat("none"),
            pairing="by_group",
            group_ids=group_ids[:-1],
        )


def within_group(
    examples: list[Ex], group_ids: list[str] | None, **kwargs: int
) -> StimulusSet:
    return build_stimulus_set(
        examples,
        TINY,
        ChatFormat("none"),
        pairing="within_group",
        group_ids=group_ids,
        **kwargs,  # type: ignore[arg-type]
    )


def test_within_group_pairs_consecutive_members_of_each_shuffled_group() -> (
    None
):
    examples, group_ids = grouped([4, 4, 4])
    by_uid = {e.uid: e for e in examples}

    stimuli = within_group(examples, group_ids, seed=3)

    group_of = [uid.split(":")[0] for uid in stimuli.uids]
    np.testing.assert_array_equal(stimuli.pairs, np.arange(12).reshape(6, 2))
    assert all(group_of[a] == group_of[b] for a, b in stimuli.pairs)
    assert collections.Counter(group_of[a] for a, _ in stimuli.pairs) == {
        "g0": 2,
        "g1": 2,
        "g2": 2,
    }
    assert sorted(stimuli.uids) == sorted(by_uid)
    assert group_of[::4] == ["g0", "g1", "g2"]
    assert stimuli.texts == tuple(by_uid[uid].text for uid in stimuli.uids)
    assert stimuli.labels.tolist() == [by_uid[u].label for u in stimuli.uids]
    assert stimuli.groups.shape == (3, 4)
    assert sorted(stimuli.groups.ravel().tolist()) == list(range(12))
    for row in stimuli.groups:
        assert len({group_of[i] for i in row}) == 1
        assert stimuli.labels[row].sum() == 1


def test_within_group_is_deterministic_and_seed_dependent() -> None:
    examples, group_ids = grouped([4] * 6)

    first = within_group(examples, group_ids, seed=5)

    assert fingerprint(first) == fingerprint(
        within_group(examples, group_ids, seed=5)
    )
    assert first.uids != within_group(examples, group_ids, seed=6).uids


def test_within_group_pairs_ignore_the_labels() -> None:
    examples, group_ids = grouped([4, 4, 4])
    moved = [replace(e, label=int(e.uid.endswith(":3"))) for e in examples]

    original = within_group(examples, group_ids, seed=8)
    relabeled = within_group(moved, group_ids, seed=8)

    assert relabeled.uids == original.uids
    np.testing.assert_array_equal(relabeled.pairs, original.pairs)


def test_within_group_pads_an_odd_group_with_a_copy_left_out_of_its_group() -> (
    None
):
    examples, group_ids = grouped([5, 4])
    by_uid = {e.uid: e for e in examples}

    stimuli = within_group(examples, group_ids, seed=0)

    assert len(stimuli.texts) == 10
    assert len(stimuli.pairs) == 5
    assert stimuli.groups.shape == (2, 5)
    row = [int(i) for i in stimuli.groups[0]]
    assert sorted(stimuli.uids[i] for i in row) == [f"g0:{c}" for c in range(5)]
    (pad,) = set(range(6)) - set(row)
    assert by_uid[stimuli.uids[pad]].label == 0
    assert stimuli.labels[pad] == 0
    assert stimuli.texts[pad] == by_uid[stimuli.uids[pad]].text
    assert stimuli.groups[1].tolist() == [6, 7, 8, 9, -1]
    assert collections.Counter(
        stimuli.uids[i].split(":")[0] for i in stimuli.pairs.ravel()
    ) == {"g0": 6, "g1": 4}


def test_within_group_pad_never_copies_the_gold_member() -> None:
    examples, group_ids = grouped([5])

    for seed in range(100):
        stimuli = within_group(examples, group_ids, seed=seed)

        (pad,) = set(range(6)) - set(stimuli.groups[0].tolist())
        assert stimuli.uids[pad] != "g0:0"


def test_within_group_rejects_n_pairs() -> None:
    examples, group_ids = grouped([4, 4])

    with pytest.raises(ValueError, match="n_pairs"):
        within_group(examples, group_ids, n_pairs=1)


def test_within_group_needs_group_ids_and_one_positive_per_group() -> None:
    examples, group_ids = grouped([4, 4])
    two_positives = [
        replace(e, label=1) if e.uid == "g1:0" else e for e in examples
    ]

    with pytest.raises(ValueError, match="within_group pairing needs"):
        within_group(examples, None)
    with pytest.raises(ValueError, match="g1"):
        within_group(two_positives, group_ids)


def test_within_group_reading_turns_gold_at_the_minimum_into_sign_minus() -> (
    None
):
    examples, group_ids = grouped([4] * 6)
    stimuli = within_group(examples, group_ids, seed=0)
    h = np.random.default_rng(0).normal(scale=0.01, size=(1, 24, 3))
    h[0, :, 0] -= 5.0 * stimuli.labels

    reader = RepReader.fit(
        "pca_diff",
        h,
        [0],
        labels=stimuli.labels,
        pairs=stimuli.pairs,
        groups=stimuli.groups,
    )

    assert reader.signs.tolist() == [-1.0]
    raw = reader.raw_score(h)[0]
    for row in stimuli.groups:
        assert stimuli.labels[row[np.argmin(raw[row])]] == 1
    accuracy = group_accuracy(reader.score(h), stimuli.groups, stimuli.labels)
    assert accuracy.tolist() == [1.0]


def test_missing_question_raises_naming_the_example() -> None:
    examples = [Ex(f"u{i}", f"a{i}", i % 2) for i in range(6)]

    with pytest.raises(
        ValueError, match=r"example 'u\d' needs meta\['question'\]"
    ):
        build_stimulus_set(
            examples,
            PAPER_TEMPLATES["truthfulqa"],
            ChatFormat("none"),
            pairing="label_contrastive",
        )


def test_question_meta_must_be_text() -> None:
    examples = [Ex(f"u{i}", f"a{i}", i % 2, {"question": 7}) for i in range(6)]

    with pytest.raises(ValueError, match="question"):
        build_stimulus_set(
            examples,
            PAPER_TEMPLATES["truthfulqa"],
            ChatFormat("none"),
            pairing="label_contrastive",
        )


def test_empty_examples_are_rejected() -> None:
    with pytest.raises(ValueError, match="example"):
        build_stimulus_set(
            [],
            CONCEPT,
            ChatFormat("none"),
            pairing="label_contrastive",
            concept="mood",
        )


def test_n_pairs_must_be_positive() -> None:
    with pytest.raises(ValueError, match="n_pairs"):
        build_stimulus_set(
            labeled(6, 3),
            CONCEPT,
            ChatFormat("none"),
            pairing="random",
            concept="mood",
            n_pairs=0,
        )


@pytest.mark.parametrize("chat", CHATS.values(), ids=CHATS)
def test_concept_spans_hold_under_every_chat_format(chat: ChatFormat) -> None:
    template = ConceptTemplate("t", "About {concept}: {stimulus}\nDone")

    stimuli = build_stimulus_set(
        labeled(8, 4),
        template,
        chat,
        pairing="label_contrastive",
        concept="mood",
        seed=3,
    )

    for i in range(len(stimuli.texts)):
        assert span_text(stimuli, i).startswith("text number")
        span = stimuli.concept_spans[i]
        assert span is not None
        assert stimuli.texts[i][span[0] : span[1]] == "mood"
