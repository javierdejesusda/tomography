from typing import Any

import pytest

from tomography.templates import ChatFormat
from tomography.templates import ConceptTemplate
from tomography.templates import FunctionTemplate


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
