"""Chat formats and stimulus templates with exact character spans.

Rendering never strips or normalizes text: the text of every marked span is
the substituted slot itself, so ``text[start:end]`` can be mapped to tokens
through tokenizer offsets.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal
from typing import Self

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

Span = tuple[int, int]
ChatRenderFn = Callable[[list[dict[str, str]]], str]

_USER_SENTINEL = "TOMOGRAPHY_USER_TEXT"


@dataclass(frozen=True)
class ChatFormat:
    """How a user turn and an assistant turn become one prompt string.

    Attributes:
        kind: ``"plain"`` writes ``USER:`` and ``ASSISTANT:`` style tags (base
            models), ``"none"`` writes no tags, and ``"template"`` delegates
            to ``render_fn`` (post-trained models with a chat template).
        user_tag: Tag before the user text for the plain kind.
        assistant_tag: Tag before the assistant text for the plain kind.
        turn_sep: Separator between the user text and the next tag (plain) or
            the assistant text (none).
        render_fn: Maps chat messages to the prompt text including the
            generation prompt. Required for, and only for, the template kind.
    """

    kind: Literal["plain", "none", "template"]
    user_tag: str = "USER:"
    assistant_tag: str = "ASSISTANT:"
    turn_sep: str = " "
    render_fn: ChatRenderFn | None = None

    def __post_init__(self) -> None:
        """Checks that ``render_fn`` is set exactly for the template kind."""
        if (self.kind == "template") != (self.render_fn is not None):
            raise ValueError(
                "render_fn is required for, and only for, the template kind"
            )

    @classmethod
    def from_tokenizer(cls, tokenizer: "PreTrainedTokenizerBase") -> Self:
        """Builds a template format from a tokenizer's chat template.

        The prompt ends with the generation prompt and thinking is disabled
        (``enable_thinking=False``), which for Qwen3 appends an empty think
        block after the assistant tag.

        Args:
            tokenizer: A tokenizer that defines a chat template.

        Returns:
            A format of the template kind.
        """

        def render(messages: list[dict[str, str]]) -> str:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if not isinstance(prompt, str):
                raise TypeError("the chat template did not return text")
            return prompt

        return cls("template", render_fn=render)

    def render_turn(
        self, user: str, assistant: str = ""
    ) -> tuple[str, int, int]:
        """Renders one user turn followed by an assistant turn.

        The plain layout is exactly
        ``f"{user_tag} {user}{turn_sep}{assistant_tag} {assistant}"``. With an
        empty ``assistant`` the trailing space stays.

        Args:
            user: User text, inserted verbatim.
            assistant: Assistant text, inserted verbatim after the prompt.

        Returns:
            The full text, the character offset where ``user`` starts and the
            offset where ``assistant`` starts.

        Raises:
            ValueError: If a chat template alters the user text.
        """
        if self.kind == "none":
            if not assistant:
                return user, 0, len(user)
            return (
                f"{user}{self.turn_sep}{assistant}",
                0,
                len(user) + len(self.turn_sep),
            )
        if self.kind == "plain":
            head = f"{self.user_tag} "
            prompt = f"{head}{user}{self.turn_sep}{self.assistant_tag} "
            return prompt + assistant, len(head), len(prompt)
        prompt, user_start = self._render_template(user)
        return prompt + assistant, user_start, len(prompt)

    def _render_template(self, user: str) -> tuple[str, int]:
        """Renders a user turn through the template and locates its text."""
        assert self.render_fn is not None
        marked = self.render_fn([{"role": "user", "content": _USER_SENTINEL}])
        if marked.count(_USER_SENTINEL) != 1:
            raise ValueError("cannot locate the user text in the chat template")
        user_start = marked.index(_USER_SENTINEL)
        suffix = marked[user_start + len(_USER_SENTINEL) :]
        prompt = self.render_fn([{"role": "user", "content": user}])
        if prompt != marked[:user_start] + user + suffix:
            raise ValueError("the chat template altered the user text")
        return prompt, user_start
