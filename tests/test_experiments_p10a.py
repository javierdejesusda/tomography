"""Tests for the P10a quantity behavior experiment."""

import dataclasses

import pytest

from tomography.datasets.quantities import FAMILIES
from tomography.experiments import p10a_quantities
from tomography.experiments.p10a_quantities import FORCED_INSTRUCTION
from tomography.experiments.p10a_quantities import FORCED_SHOTS
from tomography.experiments.p10a_quantities import FREE_SHOTS
from tomography.experiments.p10a_quantities import UNIT_WORDS
from tomography.experiments.p10a_quantities import Config
from tomography.experiments.p10a_quantities import build_prompt
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
