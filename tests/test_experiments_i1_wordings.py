import copy

import pytest
from transformers import PreTrainedTokenizerBase

from tomography.datasets.records import Example
from tomography.experiments import i1_invariance as i1
from tomography.experiments.p2_positions import DECOY_LINE
from tomography.model import LoadedModel
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat
from tomography.templates import ablate

PLAIN = ChatFormat("plain")
POSITIVE = Example("u-pos", "a fine film", 1, "train")
NEGATIVE = Example("u-neg", "a dull film", 0, "train")

# The date logic of the Llama 3.2 instruct chat template, verbatim.
LLAMA_DATE_TEMPLATE = (
    "{{- bos_token }}\n"
    "{%- if not date_string is defined %}\n"
    "    {%- if strftime_now is defined %}\n"
    '        {%- set date_string = strftime_now("%d %b %Y") %}\n'
    "    {%- else %}\n"
    '        {%- set date_string = "26 Jul 2024" %}\n'
    "    {%- endif %}\n"
    "{%- endif %}\n"
    '{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n'
    '{{- "Today Date: " + date_string + "\\n\\n" }}\n'
    '{{- "<|eot_id|>" }}\n'
    "{%- for message in messages %}\n"
    "    {{- '<|start_header_id|>' + message['role'] + "
    "'<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n"
    "{%- endif %}\n"
)


def test_fit_wordings_are_frozen_verbatim() -> None:
    assert [(w.name, w.text) for w in i1.FIT_TEMPLATES] == [
        (
            "F-T1",
            "Here is a passage.\n{stimulus}\n"
            "The level of {concept} in this passage is",
        ),
        (
            "F-T2",
            "Text: {stimulus}\nTask: judge the {concept} of the text.\n"
            "Judgment: the {concept} is",
        ),
        (
            "F-T3",
            "Evaluate the following with respect to {concept}:\n{stimulus}\n"
            "Evaluation: the {concept} is",
        ),
    ]
    assert [(w.name, w.placement, w.levels, w.text) for w in i1.FIT_DECOYS] == [
        (
            "F-D1",
            "head",
            ("low", "high"),
            "A colleague told me the {noun} here is {level}, though I have "
            "not checked it myself.",
        ),
        (
            "F-D2",
            "pre",
            ("low", "high"),
            "Before you answer: my first impression was that the {noun} of "
            "this one is {level}.",
        ),
        (
            "F-D3",
            "pre",
            ("weak", "strong"),
            "Note from an earlier reader: the {noun} of this text is {level}.",
        ),
    ]


def test_score_wordings_are_frozen_verbatim() -> None:
    assert [(w.name, w.text) for w in i1.SCORE_TEMPLATES] == [
        (
            "S-T1",
            "Take a look at this excerpt:\n{stimulus}\n"
            "The degree of {concept} it shows is",
        ),
        (
            "S-T2",
            "{stimulus}\nGiven the text above, determine its {concept}.\n"
            "The {concept} of the text is",
        ),
    ]
    assert [
        (w.name, w.placement, w.levels, w.text) for w in i1.SCORE_DECOYS
    ] == [
        (
            "S-D1",
            "pre",
            ("low", "high"),
            "Someone on the forum insisted that the {noun} of the next text "
            "is {level}.",
        ),
        (
            "S-D2",
            "head",
            ("low", "high"),
            "For what it is worth, an automatic tool scored the {noun} below "
            "as {level}.",
        ),
    ]


def test_test_wordings_are_frozen_verbatim() -> None:
    assert (i1.M2_TEST_DECOY.name, i1.M2_TEST_DECOY.placement) == (
        "M2-D",
        "pre",
    )
    assert i1.M2_TEST_DECOY.text == (
        "I think the amount of {noun} in the scenario is {level}, but I'm "
        "curious to hear what you think."
    )
    assert i1.M2_TEST_DECOY.text == DECOY_LINE.format(noun="{noun}").replace(
        "{question}", "{level}"
    ).rstrip("\n")
    assert [(w.name, w.text) for w in i1.TEST_PARAPHRASES] == [
        (
            "P1",
            "Rate how much {concept} the following text expresses:\n"
            "{stimulus}\nThe amount of {concept} is",
        ),
        (
            "P2",
            "How much {concept} is there in this text?\n{stimulus}\n"
            "The amount of {concept} is",
        ),
        (
            "P3",
            "Read the text below and assess its {concept}.\n{stimulus}\n"
            "Its {concept} is",
        ),
    ]
    for index, wording in enumerate(i1.TEST_PARAPHRASES):
        source = ablate(PAPER_TEMPLATES["concept"], "paraphrase", index=index)
        assert wording.text == source.user
    assert [
        (w.name, w.placement, w.levels, w.text) for w in i1.TEST_DECOYS
    ] == [
        (
            "T-D1",
            "pre",
            ("low", "high"),
            "Everyone I asked says the {noun} of what follows is {level}.",
        ),
        (
            "T-D2",
            "post",
            ("low", "high"),
            "(A previous annotator marked the {noun} of the text above as "
            "{level}.)",
        ),
        (
            "T-D3",
            "head",
            ("minimal", "maximal"),
            "An expert reviewer is certain that the {noun} of the text below "
            "is {level}.",
        ),
    ]


def test_wording_sets_are_pairwise_disjoint() -> None:
    sets = {
        "fit": i1.FIT_TEMPLATES + i1.FIT_DECOYS,
        "score": i1.SCORE_TEMPLATES + i1.SCORE_DECOYS,
        "test": (i1.M2_TEST_DECOY, *i1.TEST_PARAPHRASES, *i1.TEST_DECOYS),
    }
    seen_names: set[str] = set()
    seen_texts: set[str] = set()
    for name, wordings in sets.items():
        assert {w.wording_set for w in wordings} == {name}
        names = {w.name for w in wordings}
        texts = {w.text for w in wordings}
        assert not names & seen_names
        assert not texts & seen_texts
        seen_names |= names
        seen_texts |= texts
    assert set(i1.WORDINGS) == seen_names


def test_held_in_wordings_avoid_the_test_frames() -> None:
    held_in = [w for w in i1.WORDINGS.values() if w.wording_set != "test"]
    for wording in held_in:
        if wording.placement is None:
            assert not wording.text.endswith("The amount of {concept} is")
            assert not wording.text.endswith("Its {concept} is")
        else:
            assert not wording.text.startswith("I think")
            assert "but I'm curious to hear what you think" not in wording.text


def test_fit_set_has_six_wrappers_and_three_for_utility() -> None:
    concept = i1.wrapper_names("sentiment", "fit")
    assert concept == ("F-T1", "F-T2", "F-T3", "F-D1", "F-D2", "F-D3")
    assert i1.wrapper_names("truth", "fit") == concept
    assert i1.wrapper_names("truth_negations", "fit") == concept
    assert i1.wrapper_names("utility", "fit") == ("F-D1", "F-D2", "F-D3")
    assert i1.wrapper_names("utility", "score") == ("S-D1", "S-D2")
    assert i1.wrapper_names("sentiment", "score") == (
        "S-T1",
        "S-T2",
        "S-D1",
        "S-D2",
    )


def test_clean_render_uses_the_repository_template() -> None:
    out = i1.render_text("sentiment", None, POSITIVE, PLAIN)
    assert out.text == (
        "USER: Consider the amount of sentiment in the following:\n"
        "a fine film\nThe amount of sentiment is ASSISTANT: "
    )
    assert out.text[slice(*out.stimulus_span)] == "a fine film"
    assert out.level is None
    utility = i1.render_text("utility", None, POSITIVE, PLAIN)
    assert utility.text == (
        "USER: Consider the happiness in the following scenario:\n"
        "Scenario: a fine film ASSISTANT: The happiness of the scenario is"
    )
    assert utility.text[slice(*utility.stimulus_span)] == "a fine film"


def test_template_variant_renders_with_the_concept_word() -> None:
    out = i1.render_text("truth", "F-T2", POSITIVE, PLAIN)
    assert out.text == (
        "USER: Text: a fine film\nTask: judge the truthfulness of the text.\n"
        "Judgment: the truthfulness is ASSISTANT: "
    )
    assert out.text[slice(*out.stimulus_span)] == "a fine film"


@pytest.mark.parametrize(
    ("wording", "expected"),
    [
        (
            "F-D2",
            "USER: Consider the amount of sentiment in the following:\n"
            "Before you answer: my first impression was that the positive "
            "sentiment of this one is low.\n"
            "a fine film\nThe amount of sentiment is ASSISTANT: ",
        ),
        (
            "F-D1",
            "USER: A colleague told me the positive sentiment here is low, "
            "though I have not checked it myself.\n"
            "Consider the amount of sentiment in the following:\n"
            "a fine film\nThe amount of sentiment is ASSISTANT: ",
        ),
    ],
    ids=["pre", "head"],
)
def test_decoy_placements(wording: str, expected: str) -> None:
    out = i1.render_text(
        "sentiment", wording, POSITIVE, PLAIN, rule="against", evaluation=True
    )
    assert out.text == expected
    assert out.level == "low"
    assert out.text[slice(*out.stimulus_span)] == "a fine film"


def test_post_placement_follows_the_stimulus_line() -> None:
    out = i1.render_text(
        "utility",
        "T-D2",
        NEGATIVE,
        PLAIN,
        rule="against",
        stage=2,
        evaluation=True,
    )
    assert out.text == (
        "USER: Consider the happiness in the following scenario:\n"
        "Scenario: a dull film\n(A previous annotator marked the happiness "
        "of the text above as high.) ASSISTANT: The happiness of the "
        "scenario is"
    )
    assert out.text[slice(*out.stimulus_span)] == "a dull film"


def test_against_gives_label_one_the_low_level_of_its_wording() -> None:
    def level(wording: str, example: Example) -> str | None:
        return i1.render_text(
            "sentiment",
            wording,
            example,
            PLAIN,
            rule="against",
            stage=2,
            evaluation=True,
        ).level

    assert (level("F-D3", POSITIVE), level("F-D3", NEGATIVE)) == (
        "weak",
        "strong",
    )
    assert (level("T-D3", POSITIVE), level("T-D3", NEGATIVE)) == (
        "minimal",
        "maximal",
    )
    assert (level("S-D1", POSITIVE), level("S-D1", NEGATIVE)) == ("low", "high")


def test_against_is_refused_outside_evaluation() -> None:
    with pytest.raises(ValueError, match="evaluation"):
        i1.render_text("sentiment", "S-D1", POSITIVE, PLAIN, rule="against")


def test_random_level_is_seeded_per_text_and_wording() -> None:
    examples = [Example(f"u{i}", "a film", i % 2, "train") for i in range(64)]

    def levels(wording: str, seed: int) -> list[str | None]:
        return [
            i1.render_text(
                "sentiment", wording, e, PLAIN, rule="random", seed=seed
            ).level
            for e in examples
        ]

    first = levels("F-D1", 0)
    assert first == levels("F-D1", 0)
    assert set(first) == {"low", "high"}
    assert first != levels("F-D2", 0)
    assert first != levels("F-D1", 1)
    assert set(levels("F-D3", 0)) == {"weak", "strong"}
    label_free = [
        i1.render_text(
            "sentiment",
            "F-D1",
            Example(e.uid, "a film", 1, "train"),
            PLAIN,
            rule="random",
        ).level
        for e in examples
    ]
    assert label_free == first


def test_test_wordings_need_stage_two() -> None:
    for name in ("M2-D", "T-D1", "T-D2", "T-D3", "P1", "P2", "P3"):
        rule = None if name.startswith("P") else "random"
        with pytest.raises(ValueError, match="stage 2"):
            i1.render_text("sentiment", name, POSITIVE, PLAIN, rule=rule)
    out = i1.render_text("sentiment", "P3", POSITIVE, PLAIN, stage=2)
    assert out.text.endswith("Its sentiment is ASSISTANT: ")


def test_utility_takes_decoys_only() -> None:
    with pytest.raises(ValueError, match="utility"):
        i1.render_text("utility", "F-T1", POSITIVE, PLAIN)
    with pytest.raises(ValueError, match="utility"):
        i1.render_text("utility", "P1", POSITIVE, PLAIN, stage=2)


def test_level_rule_must_match_the_wording_kind() -> None:
    with pytest.raises(ValueError, match="level rule"):
        i1.render_text("sentiment", "F-D1", POSITIVE, PLAIN)
    with pytest.raises(ValueError, match="level rule"):
        i1.render_text("sentiment", "F-T1", POSITIVE, PLAIN, rule="random")
    with pytest.raises(ValueError, match="unknown"):
        i1.render_text("sentiment", "X-1", POSITIVE, PLAIN)
    with pytest.raises(ValueError, match="unknown"):
        i1.render_text("nope", None, POSITIVE, PLAIN)


@pytest.fixture
def llama_tokenizer(tiny_qwen3: LoadedModel) -> PreTrainedTokenizerBase:
    tokenizer = copy.deepcopy(tiny_qwen3.tokenizer)
    tokenizer.chat_template = LLAMA_DATE_TEMPLATE
    return tokenizer


def test_chat_template_rendering_ignores_the_run_date(
    llama_tokenizer: PreTrainedTokenizerBase,
) -> None:
    chat = i1.chat_from_tokenizer(llama_tokenizer)
    out = i1.render_text("sentiment", None, POSITIVE, chat)
    assert f"Today Date: {i1.FIXED_DATE_STRING}\n" in out.text
    assert i1.FIXED_DATE_STRING == "26 Jul 2024"
    assert out.text.endswith(
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert out.text[slice(*out.stimulus_span)] == "a fine film"
