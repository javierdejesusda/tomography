"""Tests for the P10a quantity behavior experiment."""

import dataclasses
from typing import Any

import pytest

from tomography.datasets.quantities import FAMILIES
from tomography.datasets.quantities import QuantityQuestion
from tomography.experiments import p10a_quantities
from tomography.experiments.p10a_quantities import FORCED_INSTRUCTION
from tomography.experiments.p10a_quantities import FORCED_SHOTS
from tomography.experiments.p10a_quantities import FREE_SHOTS
from tomography.experiments.p10a_quantities import UNIT_WORDS
from tomography.experiments.p10a_quantities import Answer
from tomography.experiments.p10a_quantities import Config
from tomography.experiments.p10a_quantities import build_prompt
from tomography.experiments.p10a_quantities import score_answer
from tomography.experiments.sweep import chat_format_for
from tomography.model import MODELS
from tomography.model import LoadedModel

DEFAULT_HASH = "989620d61461"


def test_config_defaults_and_round_trip() -> None:
    config = Config()

    assert config.experiment == "p10a"
    assert config.model == "gpt2"
    assert config.families is None
    assert config.n_per_side == 120
    assert config.min_sitelinks == 5
    assert config.conditions == ("free", "forced")
    assert config.formats == ("fewshot", "chat")
    assert config.n_samples == 5
    assert config.temperature == 1.0
    assert config.max_new_tokens == 32
    assert config.rel_tol == 0.05
    assert config.n_hand_check == 120
    assert config.batch_size == 8
    assert config.offline is False
    assert config.seed == 0
    assert Config.from_dict(config.to_dict()) == config


def test_a_family_subset_round_trips() -> None:
    config = Config(families=("height", "duration"))

    assert Config.from_dict(config.to_dict()) == config


def test_the_default_config_hash_is_stable() -> None:
    assert Config().config_hash() == DEFAULT_HASH


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("families", ()),
        ("families", ("height", "height")),
        ("families", ("height", "no such family")),
        ("n_per_side", 0),
        ("n_samples", 0),
        ("temperature", 0.0),
        ("max_new_tokens", 0),
        ("rel_tol", -0.1),
        ("n_hand_check", -1),
        ("batch_size", 0),
        ("conditions", ()),
        ("conditions", ("free", "free")),
        ("conditions", ("free", "pressured")),
        ("formats", ()),
        ("formats", ("fewshot", "fewshot")),
        ("formats", ("zeroshot",)),
    ],
)
def test_bad_settings_raise_before_any_work(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        Config(**{field: value})


def test_unit_words_cover_every_unit_the_loader_can_emit() -> None:
    units = {
        side.unit
        for family in FAMILIES
        for side in (family.range_side, family.exact_side)
    }

    assert units <= set(UNIT_WORDS)
    for unit in units:
        assert unit in UNIT_WORDS[unit]


@pytest.mark.parametrize(
    ("unit", "words"),
    [
        ("centimetres", {"cm", "centimeters", "centimeter", "centimetre"}),
        ("metres", {"m", "meters", "meter", "metre"}),
        ("days", {"day"}),
        ("minutes", {"min", "mins", "minute"}),
    ],
)
def test_unit_words_hold_spellings_and_abbreviations(
    unit: str, words: set[str]
) -> None:
    assert words <= set(UNIT_WORDS[unit])


def test_an_unknown_unit_fails_loudly() -> None:
    with pytest.raises(ValueError, match="furlongs"):
        p10a_quantities.unit_words("furlongs")


def test_the_free_and_forced_shots_share_their_questions() -> None:
    assert [q for q, _ in FREE_SHOTS] == [q for q, _ in FORCED_SHOTS]
    assert len(FREE_SHOTS) == 6


def test_free_shots_mix_exact_and_range_answers() -> None:
    answers = [a for _, a in FREE_SHOTS]

    assert sum(" to " in a for a in answers) == 3
    assert sum(" to " not in a for a in answers) == 3


def test_forced_shots_replace_each_range_by_one_number_inside_it() -> None:
    for (_, free), (_, forced) in zip(FREE_SHOTS, FORCED_SHOTS, strict=True):
        assert forced.isdigit()
        if " to " in free:
            low, high = (int(part) for part in free.split(" to "))
            assert low <= int(forced) <= high
        else:
            assert forced == free


def test_the_forced_range_replacements_are_72_8_and_15() -> None:
    replaced = [
        forced
        for (_, free), (_, forced) in zip(FREE_SHOTS, FORCED_SHOTS, strict=True)
        if " to " in free
    ]

    assert replaced == ["72", "8", "15"]


def test_the_free_fewshot_prompt_is_shots_then_the_question() -> None:
    prompt = build_prompt("How tall is X?", "free", "fewshot", None)

    expected = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in FREE_SHOTS)
    assert prompt == f"{expected}Q: How tall is X?\nA:"


def test_the_forced_fewshot_prompt_opens_with_the_instruction() -> None:
    prompt = build_prompt("How tall is X?", "forced", "fewshot", None)

    expected = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in FORCED_SHOTS)
    assert FORCED_INSTRUCTION == "Answer each question with one exact number."
    assert prompt == f"{FORCED_INSTRUCTION}\n\n{expected}Q: How tall is X?\nA:"


@pytest.mark.parametrize(
    ("condition", "instruction"),
    [
        ("free", "Answer the question briefly."),
        ("forced", "Answer with one exact number only."),
    ],
)
def test_the_chat_prompt_renders_the_instruction_and_question(
    tiny_qwen3: LoadedModel, condition: str, instruction: str
) -> None:
    post_trained = dataclasses.replace(
        tiny_qwen3, checkpoint=MODELS["qwen3-1.7b"]
    )
    chat = chat_format_for(post_trained)

    prompt = build_prompt("How tall is X?", condition, "chat", chat)

    assert (
        prompt
        == chat.render_turn(f"{instruction}\n\nQuestion: How tall is X?")[0]
    )
    assert instruction in prompt


def test_an_unknown_condition_or_format_raises() -> None:
    with pytest.raises(ValueError, match="condition"):
        build_prompt("Q?", "pressured", "fewshot", None)
    with pytest.raises(ValueError, match="format"):
        build_prompt("Q?", "free", "zeroshot", None)


def _item(**overrides: Any) -> QuantityQuestion:
    fields: dict[str, Any] = {
        "uid": "height-range-Q1",
        "family": "height",
        "kind": "range",
        "entity": "Beagle",
        "entity_id": "Q1",
        "question": "How tall is a Beagle, in centimetres?",
        "low": 33.0,
        "high": 41.0,
        "unit": "centimetres",
        "sitelinks": 40,
        "split": "train",
        "retrieved_on": "2026-10-17",
        "sha256": "ab" * 32,
    }
    return QuantityQuestion(**{**fields, **overrides})


def _score(text: str, fmt: str = "fewshot", **overrides: Any) -> Answer:
    return score_answer(_item(**overrides), text, fmt, rel_tol=0.05)


def test_a_fewshot_answer_is_cut_at_the_first_newline() -> None:
    text = " 35 cm\nQ: How many keys?\nA: 88"

    answer = _score(text)

    assert (answer.form, answer.low, answer.high) == ("point", 35.0, 35.0)
    assert answer.text == text


def test_a_fewshot_completion_that_starts_a_new_line_is_no_answer() -> None:
    answer = _score("\nQ: What year was 1969?\nA: 1969")

    assert answer.form == "none"
    assert answer.correct is None


def test_a_chat_answer_reads_its_first_non_empty_line() -> None:
    answer = _score("\n\n35 to 40 cm\nBeagles are small.", "chat")

    assert (answer.form, answer.low, answer.high) == ("range", 35.0, 40.0)


def test_the_entity_digits_are_not_the_answer() -> None:
    item = {
        "entity": "Boeing 737-800",
        "question": "What is the wingspan of the Boeing 737-800, in metres?",
        "unit": "metres",
        "low": 35.0,
        "high": 35.0,
        "kind": "exact",
    }

    answer = _score(" The Boeing 737-800 has a wingspan of 35.8.", **item)

    assert (answer.form, answer.low) == ("point", 35.8)
    assert answer.correct is True


def test_the_quantity_followed_by_the_unit_wins() -> None:
    answer = _score(" 3 dogs weigh 38 cm", unit="centimetres")

    assert answer.low == 38.0


@pytest.mark.parametrize(
    ("text", "form", "correct"),
    [
        (" 37", "point", True),
        (" 30", "point", False),
        (" 32", "point", True),
        (" 31", "point", False),
        (" 20 to 34", "range", True),
        (" 50 to 60", "range", False),
        (" I do not know", "abstain", None),
        (" tall", "none", None),
    ],
)
def test_the_answer_is_graded_against_the_sourced_range(
    text: str, form: str, correct: bool | None
) -> None:
    answer = _score(text)

    assert answer.form == form
    assert answer.correct is correct


def test_the_relative_tolerance_widens_the_range() -> None:
    item = _item()

    tight = score_answer(item, " 42", "fewshot", rel_tol=0.0)
    loose = score_answer(item, " 42", "fewshot", rel_tol=0.05)

    assert (tight.correct, loose.correct) == (False, True)


def test_an_approximate_marker_is_kept() -> None:
    assert _score(" about 37 cm").approximate is True
    assert _score(" 37 cm").approximate is False


@pytest.mark.parametrize(
    ("unit", "text", "flag"),
    [
        ("centimetres", " 37 cm", False),
        ("centimetres", " 37", False),
        ("centimetres", " 0.37 m", True),
        ("centimetres", " 14 inches", True),
        ("centimetres", " 37 centimeters", False),
        ("metres", " 35.8 m", False),
        ("metres", " 35.8m", False),
        ("metres", " 3580 cm", True),
        ("metres", " 117 feet", True),
        ("metres", " 0.0358 km", True),
        ("days", " 63 days", False),
        ("days", " 9 months", True),
        ("days", " 2 weeks", True),
        ("days", " 63", False),
        ("days", " 2 months or 63 days", False),
        ("minutes", " 142 min", False),
        ("minutes", " 142 mins", False),
        ("minutes", " 2 hours", True),
        ("minutes", " 2 hours 22 minutes", False),
        ("minutes", " I do not know", False),
        ("minutes", " about a century", False),
    ],
)
def test_a_unit_mismatch_needs_a_number_and_a_foreign_unit(
    unit: str, text: str, flag: bool
) -> None:
    assert _score(text, unit=unit).unit_mismatch is flag


def test_an_entity_name_that_holds_a_unit_word_is_not_a_mismatch() -> None:
    film = {"entity": "Days of Heaven", "unit": "minutes"}

    assert _score(" Days of Heaven runs 94", **film).unit_mismatch is False
    assert _score(" Days of Heaven runs 94 hours", **film).unit_mismatch is True


def test_an_unknown_unit_fails_loudly_when_scoring() -> None:
    with pytest.raises(ValueError, match="furlongs"):
        _score(" 3", unit="furlongs")
