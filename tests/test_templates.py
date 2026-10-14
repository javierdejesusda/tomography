from typing import Any

import pytest

from tomography.templates import ABLATION_KINDS
from tomography.templates import EMOTIONS
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import UNRELATED_CONCEPTS
from tomography.templates import ChatFormat
from tomography.templates import ConceptTemplate
from tomography.templates import FunctionTemplate
from tomography.templates import ablate
from tomography.templates import emotion_function_template


def fake_qwen3_render(messages: list[dict[str, str]]) -> str:
    """Mimics the Qwen3 chat template with thinking disabled."""
    prompt = "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
    )
    return prompt + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def fake_qwen3_format() -> ChatFormat:
    return ChatFormat("template", render_fn=fake_qwen3_render)


def test_plain_turn_is_the_reference_layout() -> None:
    text, user_start, assistant_start = ChatFormat("plain").render_turn(
        "Pretend you are honest.", "The sky is blue"
    )

    assert text == "USER: Pretend you are honest. ASSISTANT: The sky is blue"
    assert text[user_start:].startswith("Pretend")
    assert text[assistant_start:] == "The sky is blue"


def test_plain_turn_without_assistant_keeps_the_trailing_space() -> None:
    text, _, assistant_start = ChatFormat("plain").render_turn("Question")

    assert text == "USER: Question ASSISTANT: "
    assert assistant_start == len(text)


def test_plain_turn_accepts_other_tags_and_separator() -> None:
    chat = ChatFormat("plain", "Q:", "A:", turn_sep="\n")

    text, user_start, assistant_start = chat.render_turn("Why?", "Because")

    assert text == "Q: Why?\nA: Because"
    assert text[user_start : user_start + 4] == "Why?"
    assert text[assistant_start:] == "Because"


def test_none_turn_is_the_user_text_alone() -> None:
    assert ChatFormat("none").render_turn("Just text") == ("Just text", 0, 9)


def test_none_turn_appends_the_assistant_text_after_the_separator() -> None:
    text, user_start, assistant_start = ChatFormat("none").render_turn(
        "Persona.", "Reply"
    )

    assert text == "Persona. Reply"
    assert user_start == 0
    assert text[assistant_start:] == "Reply"


def test_template_turn_places_the_user_text_exactly() -> None:
    text, user_start, assistant_start = fake_qwen3_format().render_turn(
        "How are you?", "Fine"
    )

    assert text == (
        "<|im_start|>user\nHow are you?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nFine"
    )
    assert text[user_start : user_start + len("How are you?")] == "How are you?"
    assert text[assistant_start:] == "Fine"


@pytest.mark.parametrize(
    "user",
    ["user", "assistant", "<|im_end|>", "think", " padded \n"],
    ids=["role-word", "role-word-2", "special", "template-word", "whitespace"],
)
def test_template_turn_locates_user_text_that_also_occurs_in_the_template(
    user: str,
) -> None:
    text, user_start, _ = fake_qwen3_format().render_turn(user)

    assert text[user_start : user_start + len(user)] == user
    assert user_start == len("<|im_start|>user\n")


def test_template_turn_rejects_a_template_that_alters_the_user_text() -> None:
    def stripping_render(messages: list[dict[str, str]]) -> str:
        return fake_qwen3_render(
            [{**m, "content": m["content"].strip()} for m in messages]
        )

    chat = ChatFormat("template", render_fn=stripping_render)

    with pytest.raises(ValueError, match="user text"):
        chat.render_turn(" padded ")


def test_template_kind_requires_a_render_function() -> None:
    with pytest.raises(ValueError, match="render_fn"):
        ChatFormat("template")


@pytest.mark.parametrize("kind", ["plain", "none"])
def test_other_kinds_reject_a_render_function(kind: Any) -> None:
    with pytest.raises(ValueError, match="render_fn"):
        ChatFormat(kind, render_fn=fake_qwen3_render)


class StubTokenizer:
    """Records the chat template call of ``ChatFormat.from_tokenizer``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> str:
        self.calls.append(kwargs)
        return fake_qwen3_render(messages)


def test_from_tokenizer_disables_thinking_and_adds_the_generation_prompt() -> (
    None
):
    tokenizer = StubTokenizer()

    chat = ChatFormat.from_tokenizer(tokenizer)  # type: ignore[arg-type]
    text, _, assistant_start = chat.render_turn("Hi", "Hello")

    assert chat.kind == "template"
    assert text.endswith("<think>\n\n</think>\n\nHello")
    assert text[assistant_start:] == "Hello"
    assert all(
        call["enable_thinking"] is False
        and call["add_generation_prompt"] is True
        and call["tokenize"] is False
        for call in tokenizer.calls
    )


GENERIC = ConceptTemplate(
    "generic",
    "Consider the amount of {concept} in the following:\n{stimulus}\n"
    "The amount of {concept} is",
)
QA = ConceptTemplate(
    "qa",
    "Question: {question}\nAnswer: {stimulus}\nThe amount of truth is",
)
PERSONA = FunctionTemplate(
    "persona",
    "Pretend you're {persona} person making statements about the world.",
    pos_personas=("an honest", "a truthful"),
    neg_personas=("an untruthful", "a deceitful"),
)

CHAT_FORMATS = {
    "plain": ChatFormat("plain"),
    "plain-qa": ChatFormat("plain", "Q:", "A:", turn_sep="\n"),
    "none": ChatFormat("none"),
    "template": fake_qwen3_format(),
}


@pytest.mark.parametrize("chat", CHAT_FORMATS.values(), ids=CHAT_FORMATS)
def test_concept_spans_equal_the_substituted_slots(chat: ChatFormat) -> None:
    stimulus = " a gripping, beautiful film"

    rendered = GENERIC.render(stimulus, "sentiment", chat)

    assert rendered.response_span is not None
    start, end = rendered.response_span
    assert rendered.text[start:end] == stimulus
    assert rendered.concept_span is not None
    c_start, c_end = rendered.concept_span
    assert rendered.text[c_start:c_end] == "sentiment"


def test_concept_span_is_the_last_occurrence_after_the_stimulus() -> None:
    rendered = GENERIC.render("text", "sentiment", ChatFormat("none"))

    assert rendered.response_span is not None
    assert rendered.concept_span is not None
    assert rendered.concept_span[0] >= rendered.response_span[1]
    assert rendered.text.endswith("The amount of sentiment is")


def test_untagged_concept_text_is_exactly_the_filled_template() -> None:
    rendered = GENERIC.render("Great.", "sentiment", ChatFormat("none"))

    assert rendered.text == (
        "Consider the amount of sentiment in the following:\n"
        "Great.\nThe amount of sentiment is"
    )


def test_plain_concept_text_wraps_the_filled_template_in_tags() -> None:
    rendered = QA.render("Paris", "truth", ChatFormat("plain"), question="Q1?")

    assert rendered.text == (
        "USER: Question: Q1?\nAnswer: Paris\nThe amount of truth is ASSISTANT: "
    )
    assert rendered.concept_span is None


def test_concept_slots_are_filled_verbatim_without_reexpansion() -> None:
    stimulus = "{concept} {stimulus} {question} {0} }{"

    rendered = GENERIC.render(stimulus, "{x}", ChatFormat("none"))

    assert rendered.response_span is not None
    start, end = rendered.response_span
    assert rendered.text[start:end] == stimulus
    assert rendered.text.count("{x}") == 2


def test_concept_stimulus_keeps_whitespace() -> None:
    stimulus = "  padded\n"

    rendered = GENERIC.render(stimulus, "mood", ChatFormat("plain"))

    assert rendered.response_span is not None
    start, end = rendered.response_span
    assert rendered.text[start:end] == stimulus


def test_concept_question_slot_requires_a_question() -> None:
    with pytest.raises(ValueError, match="question"):
        QA.render("Paris", "truth", ChatFormat("none"))


def test_concept_template_without_question_slot_ignores_it() -> None:
    rendered = GENERIC.render("x", "mood", ChatFormat("none"), question="Q?")

    assert "Q?" not in rendered.text


def test_concept_assistant_text_follows_the_assistant_tag() -> None:
    template = ConceptTemplate(
        "utility",
        "Consider the happiness in the following scenario:\n"
        "Scenario: {stimulus}",
        assistant="The happiness of the scenario is",
    )

    rendered = template.render("I won.", "happiness", ChatFormat("plain"))

    assert rendered.text == (
        "USER: Consider the happiness in the following scenario:\n"
        "Scenario: I won. ASSISTANT: The happiness of the scenario is"
    )
    assert rendered.response_span is not None
    start, end = rendered.response_span
    assert rendered.text[start:end] == "I won."


def test_untagged_template_ignores_the_chat_format() -> None:
    template = ConceptTemplate("bare", "{stimulus}", tagged=False)

    rendered = template.render("Hi", "mood", ChatFormat("plain"))

    assert rendered.text == "Hi"


def test_concept_template_needs_exactly_one_stimulus_slot() -> None:
    with pytest.raises(ValueError, match="stimulus"):
        ConceptTemplate("bad", "{stimulus} and {stimulus}")
    with pytest.raises(ValueError, match="stimulus"):
        ConceptTemplate("bad", "no slot")


def test_concept_template_rejects_function_slots() -> None:
    with pytest.raises(ValueError, match="persona"):
        ConceptTemplate("bad", "{persona} {stimulus}")


def test_concept_template_reports_its_slots() -> None:
    assert GENERIC.slots == {"concept", "stimulus"}
    assert QA.slots == {"question", "stimulus"}


@pytest.mark.parametrize("chat", CHAT_FORMATS.values(), ids=CHAT_FORMATS)
def test_function_response_span_equals_the_response(chat: ChatFormat) -> None:
    response = "The sky is blue"

    rendered = PERSONA.render(response, "pos", chat)

    assert rendered.response_span is not None
    start, end = rendered.response_span
    assert rendered.text[start:end] == response
    assert rendered.text.endswith(response)
    assert rendered.concept_span is None


def test_function_text_matches_the_reference_honesty_layout() -> None:
    rendered = PERSONA.render("The sky", "pos", ChatFormat("plain"))

    user_tag, assistant_tag = "USER:", "ASSISTANT:"
    expected = (
        f"{user_tag} Pretend you're an honest person making statements "
        f"about the world. {assistant_tag} " + "The sky"
    )
    assert rendered.text == expected


def test_function_index_picks_the_persona_of_the_pole() -> None:
    chat = ChatFormat("none")

    pos = PERSONA.render("x", "pos", chat, index=1)
    neg = PERSONA.render("x", "neg", chat, index=1)

    assert "a truthful" in pos.text
    assert "a deceitful" in neg.text


@pytest.mark.parametrize("index", [-1, 2], ids=["negative", "too-large"])
def test_function_index_out_of_range_raises(index: int) -> None:
    with pytest.raises(ValueError, match="index"):
        PERSONA.render("x", "pos", ChatFormat("none"), index=index)


def test_function_pole_must_be_pos_or_neg() -> None:
    with pytest.raises(ValueError, match="pole"):
        PERSONA.render("x", "mid", ChatFormat("none"))  # type: ignore[arg-type]


def test_function_instruction_slot_is_filled_and_required() -> None:
    template = FunctionTemplate(
        "control",
        "{instruction} Give {persona} answer.",
        pos_personas=("a truthful",),
        neg_personas=("an untruthful",),
    )

    rendered = template.render(
        "Paris", "neg", ChatFormat("plain"), instruction="Name a capital."
    )

    assert rendered.text == (
        "USER: Name a capital. Give an untruthful answer. ASSISTANT: Paris"
    )
    with pytest.raises(ValueError, match="instruction"):
        template.render("Paris", "neg", ChatFormat("plain"))


def test_function_empty_response_gives_an_empty_span_at_the_end() -> None:
    rendered = PERSONA.render("", "pos", ChatFormat("plain"))

    assert rendered.response_span == (len(rendered.text), len(rendered.text))


def test_function_qwen3_response_follows_the_generation_prompt() -> None:
    rendered = PERSONA.render("Hi", "neg", fake_qwen3_format())

    assert rendered.text.endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\nHi"
    )


def test_function_template_needs_a_persona_slot_and_personas() -> None:
    with pytest.raises(ValueError, match="persona"):
        FunctionTemplate("bad", "no slot", ("a",), ("b",))
    with pytest.raises(ValueError, match="personas"):
        FunctionTemplate("bad", "{persona}", (), ("b",))


def test_function_template_rejects_concept_slots() -> None:
    with pytest.raises(ValueError, match="stimulus"):
        FunctionTemplate("bad", "{persona} {stimulus}", ("a",), ("b",))


def test_function_template_reports_its_slots() -> None:
    assert PERSONA.slots == {"persona"}


def test_registry_has_the_documented_keys() -> None:
    assert set(PAPER_TEMPLATES) == {
        "concept",
        "truthfulqa",
        "honesty",
        "honesty_control",
        "arc",
        "utility",
        "morality",
        "emotion_concept",
        "emotion_concept_paper",
        "emotion_function",
    }


def test_registry_template_names_match_their_keys() -> None:
    for key, template in PAPER_TEMPLATES.items():
        assert template.name == key


def test_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        PAPER_TEMPLATES["new"] = GENERIC


def render_concept(key: str, chat: ChatFormat, **kwargs: Any) -> str:
    template = PAPER_TEMPLATES[key]
    assert isinstance(template, ConceptTemplate)
    return template.render("STIM", "CONCEPT", chat, **kwargs).text


def test_generic_concept_template_is_the_paper_form() -> None:
    text = render_concept("concept", ChatFormat("none"))

    assert text == (
        "Consider the amount of CONCEPT in the following:\n"
        "STIM\nThe amount of CONCEPT is"
    )


def test_truthfulqa_template_is_the_untagged_paper_form() -> None:
    text = render_concept("truthfulqa", ChatFormat("none"), question="QUE")

    assert text == (
        "Consider the amount of truthfulness in the following answer:\n"
        "Question: QUE\nAnswer: STIM\n"
        "The amount of truthfulness in the answer is"
    )


def test_arc_template_is_the_untagged_paper_form() -> None:
    text = render_concept("arc", ChatFormat("none"), question="QUE")

    assert text == (
        "Consider the correctness of the answer to the following question:\n"
        "Question: QUE\nAnswer: STIM\n"
        "The probability of the answer being correct is"
    )


def test_utility_template_is_the_tagged_paper_form() -> None:
    text = render_concept("utility", ChatFormat("plain"))

    assert text == (
        "USER: Consider the happiness in the following scenario:\n"
        "Scenario: STIM ASSISTANT: The happiness of the scenario is"
    )


def test_morality_template_is_the_tagged_paper_form() -> None:
    text = render_concept("morality", ChatFormat("plain"))

    assert text == (
        "USER: Consider the acceptability of the action in the following "
        "scenario:\nScenario: STIM ASSISTANT: "
    )


def test_emotion_concept_template_matches_the_reference_code() -> None:
    u, a, emotion, s = "USER:", "ASSISTANT:", "fear", "STIM"
    reference = (
        f"{u} Consider the {emotion} of the following scenario:\n"
        f"Scenario: {s}\nAnswer: {a} "
    )

    assert (
        render_concept("emotion_concept", ChatFormat("plain"))
        .replace("CONCEPT", emotion)
        .replace("STIM", s)
        == reference
    )


def test_emotion_concept_paper_template_is_the_paper_form() -> None:
    text = render_concept("emotion_concept_paper", ChatFormat("plain"))

    assert text == (
        "USER: Consider the CONCEPT in the following scenario:\n"
        "Scenario: STIM ASSISTANT: "
    )


def test_emotion_function_template_matches_the_reference_code() -> None:
    template = PAPER_TEMPLATES["emotion_function"]
    assert isinstance(template, FunctionTemplate)
    u, a, emo, scenario = "USER:", "ASSISTANT:", "joyful", "Today"

    text = template.render(scenario, "pos", ChatFormat("plain")).text

    assert template.pos_personas[0] == emo
    assert text == f"{u} Act as if you are extremely {emo}. {a} {scenario}"


def test_honesty_template_uses_the_reference_personas() -> None:
    template = PAPER_TEMPLATES["honesty"]
    assert isinstance(template, FunctionTemplate)

    assert template.pos_personas == ("an honest",)
    assert template.neg_personas == ("an untruthful",)
    text = template.render("STIM", "neg", ChatFormat("plain")).text
    assert text == (
        "USER: Pretend you're an untruthful person making statements about "
        "the world. ASSISTANT: STIM"
    )


def test_honesty_control_template_matches_the_lorra_layout() -> None:
    template = PAPER_TEMPLATES["honesty_control"]
    assert isinstance(template, FunctionTemplate)

    text = template.render(
        "Paris.", "pos", ChatFormat("plain"), instruction="Name a capital."
    ).text

    assert text == (
        "USER: Name a capital. Give a truthful answer. ASSISTANT: Paris."
    )


@pytest.mark.parametrize("emotion", EMOTIONS)
def test_emotion_function_template_has_three_personas_per_pole(
    emotion: str,
) -> None:
    template = emotion_function_template(emotion)

    assert len(template.pos_personas) == 3
    assert len(template.neg_personas) == 3
    assert template.name == "emotion_function"


def test_emotion_function_template_uses_the_reference_adjectives() -> None:
    fear = emotion_function_template("fear")

    assert fear.pos_personas == ("fearful", "scared", "frightened")
    assert fear.neg_personas == ("fearless", "bold", "unafraid")


def test_emotion_function_template_rejects_an_unknown_emotion() -> None:
    with pytest.raises(ValueError, match="emotion"):
        emotion_function_template("boredom")


def test_ablation_kinds_are_the_nine_p1_variants() -> None:
    assert len(ABLATION_KINDS) == 9
    assert len(set(ABLATION_KINDS)) == 9


def test_ablate_paper_returns_the_template_itself() -> None:
    assert ablate(GENERIC, "paper") is GENERIC


def test_ablate_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kind"):
        ablate(GENERIC, "shuffle")


def test_ablate_no_template_leaves_the_stimulus_alone() -> None:
    bare = ablate(GENERIC, "no_template")
    plain = bare.render("STIM", "mood", ChatFormat("plain"))

    assert plain.text == "USER: STIM ASSISTANT: "


def test_ablate_no_chat_tags_drops_the_tags_but_keeps_the_wording() -> None:
    untagged = ablate(GENERIC, "no_chat_tags")

    rendered = untagged.render("STIM", "mood", ChatFormat("plain"))

    assert (
        rendered.text == GENERIC.render("STIM", "mood", ChatFormat("none")).text
    )


def test_ablate_no_chat_tags_applies_to_function_templates() -> None:
    untagged = ablate(PERSONA, "no_chat_tags")

    rendered = untagged.render("STIM", "pos", ChatFormat("plain"))

    assert rendered.text.startswith("Pretend you're")


def test_ablate_concept_swap_replaces_the_concept_word() -> None:
    swapped = ablate(GENERIC, "concept_swap", index=1)

    rendered = swapped.render("STIM", "mood", ChatFormat("none"))

    assert "mood" not in rendered.text
    assert UNRELATED_CONCEPTS[1] in rendered.text
    assert rendered.concept_span is None


def test_ablate_concept_swap_needs_a_concept_slot() -> None:
    with pytest.raises(ValueError, match="concept"):
        ablate(QA, "concept_swap")


def test_ablate_no_trailing_question_cuts_the_text_after_the_stimulus() -> None:
    cut = ablate(GENERIC, "no_trailing_question")

    rendered = cut.render("STIM", "mood", ChatFormat("none"))

    assert rendered.text == (
        "Consider the amount of mood in the following:\nSTIM"
    )


def test_ablate_no_trailing_question_drops_the_assistant_text() -> None:
    cut = ablate(PAPER_TEMPLATES["utility"], "no_trailing_question")

    rendered = cut.render("STIM", "happiness", ChatFormat("plain"))

    assert rendered.text.endswith("Scenario: STIM ASSISTANT: ")


def test_ablate_no_trailing_question_needs_a_trailing_question() -> None:
    with pytest.raises(ValueError, match="trailing"):
        ablate(PAPER_TEMPLATES["morality"], "no_trailing_question")


def test_ablate_paraphrase_keeps_the_slots_and_changes_the_wording() -> None:
    template = PAPER_TEMPLATES["concept"]

    for index in range(3):
        paraphrased = ablate(template, "paraphrase", index=index)
        assert isinstance(paraphrased, ConceptTemplate)
        assert paraphrased.user != template.user
        assert paraphrased.slots == {"concept", "stimulus"}


def test_ablate_paraphrase_applies_to_function_templates() -> None:
    template = PAPER_TEMPLATES["honesty"]

    paraphrased = ablate(template, "paraphrase", index=0)

    assert isinstance(paraphrased, FunctionTemplate)
    assert paraphrased.user != template.user
    assert paraphrased.pos_personas == ("an honest",)


def test_ablate_paraphrase_index_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="index"):
        ablate(PAPER_TEMPLATES["concept"], "paraphrase", index=99)


def test_ablate_paraphrase_needs_a_known_template() -> None:
    with pytest.raises(ValueError, match="paraphrase"):
        ablate(GENERIC, "paraphrase")


def test_ablate_neutral_frame_does_not_mention_the_concept() -> None:
    framed = ablate(GENERIC, "neutral_frame")

    rendered = framed.render("STIM", "mood", ChatFormat("none"))

    assert "mood" not in rendered.text
    assert "STIM" in rendered.text


def test_ablate_scaffold_ends_with_a_conclusion_stub() -> None:
    scaffold = ablate(GENERIC, "scaffold")

    rendered = scaffold.render("STIM", "mood", ChatFormat("none"))

    assert rendered.text == "STIM\nConclusion: This movie is"


@pytest.mark.parametrize(
    "kind",
    ["no_template", "concept_swap", "no_trailing_question", "neutral_frame"],
)
def test_concept_only_ablations_reject_function_templates(kind: str) -> None:
    with pytest.raises(ValueError, match="concept template"):
        ablate(PERSONA, kind)


def test_ablate_scaffold_rejects_function_templates() -> None:
    with pytest.raises(ValueError, match="concept template"):
        ablate(PERSONA, "scaffold")


def test_ablate_framing_swap_maps_emotion_concept_to_function() -> None:
    swapped = ablate(
        PAPER_TEMPLATES["emotion_concept"], "framing_swap", index=3
    )

    assert isinstance(swapped, FunctionTemplate)
    assert swapped.pos_personas == ("fearful", "scared", "frightened")


def test_ablate_framing_swap_maps_emotion_function_to_concept() -> None:
    swapped = ablate(PAPER_TEMPLATES["emotion_function"], "framing_swap")

    assert swapped is PAPER_TEMPLATES["emotion_concept"]


def test_ablate_framing_swap_needs_a_counterpart() -> None:
    with pytest.raises(ValueError, match="counterpart"):
        ablate(PAPER_TEMPLATES["truthfulqa"], "framing_swap")


def test_ablations_compose_through_the_base_name() -> None:
    untagged = ablate(PAPER_TEMPLATES["concept"], "no_chat_tags")

    paraphrased = ablate(untagged, "paraphrase")

    assert isinstance(paraphrased, ConceptTemplate)
    assert not paraphrased.tagged
    assert paraphrased.name.startswith("concept:")


@pytest.mark.parametrize(
    "kind", [k for k in ABLATION_KINDS if k != "framing_swap"]
)
def test_ablated_concept_templates_keep_exact_stimulus_spans(
    kind: str,
) -> None:
    template = ablate(PAPER_TEMPLATES["concept"], kind)
    chat = ChatFormat("plain")

    rendered = template.render("a stimulus", "mood", chat)

    assert rendered.response_span is not None
    start, end = rendered.response_span
    assert rendered.text[start:end] == "a stimulus"
