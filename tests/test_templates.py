from typing import Any

import pytest

from tomography.templates import ChatFormat


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
