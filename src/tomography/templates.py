"""Chat formats and stimulus templates with exact character spans.

Rendering never strips or normalizes text: the text of every marked span is
the substituted slot itself, so ``text[start:end]`` can be mapped to tokens
through tokenizer offsets.
"""

from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING
from typing import Literal
from typing import Self

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

Span = tuple[int, int]
ChatRenderFn = Callable[[list[dict[str, str]]], str]

_USER_SENTINEL = "TOMOGRAPHY_USER_TEXT"
_SLOT = re.compile(r"\{(concept|stimulus|question|persona|instruction)\}")
_CONCEPT_SLOTS = frozenset({"concept", "stimulus", "question"})
_FUNCTION_SLOTS = frozenset({"persona", "instruction"})


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


@dataclass(frozen=True)
class Rendered:
    """A rendered stimulus and its marked character spans.

    Attributes:
        text: The full prompt text.
        response_span: The marked span: the assistant response for function
            templates and the stimulus text for concept templates.
        concept_span: The concept word in the user turn, if the template has
            a concept slot.
    """

    text: str
    response_span: Span | None
    concept_span: Span | None


@dataclass(frozen=True)
class ConceptTemplate:
    """A template that asks about a concept in a given stimulus text.

    The user text may hold the slots ``{concept}``, ``{question}`` and
    exactly one ``{stimulus}``. Slots are filled in a single pass, so filled
    values are never expanded again.

    Attributes:
        name: Identifier of the template.
        user: The user text with its slots.
        assistant: Literal text that follows the assistant tag.
        tagged: If false, the chat format is ignored and the text is rendered
            without tags (the "no chat tags" ablation).
    """

    name: str
    user: str
    assistant: str = ""
    tagged: bool = True

    def __post_init__(self) -> None:
        """Validates the slots of the user text."""
        _check_slots(self.name, self.user, _CONCEPT_SLOTS)
        if _SLOT.findall(self.user).count("stimulus") != 1:
            raise ValueError(
                f"template {self.name!r} needs exactly one {{stimulus}} slot"
            )

    @property
    def slots(self) -> frozenset[str]:
        """The slot names that occur in the user text."""
        return frozenset(_SLOT.findall(self.user))

    def render(
        self,
        stimulus: str,
        concept: str,
        chat: ChatFormat,
        *,
        question: str | None = None,
    ) -> Rendered:
        """Renders one stimulus.

        Args:
            stimulus: The stimulus text, inserted verbatim.
            concept: The concept word, inserted verbatim. Ignored when the
                template has no ``{concept}`` slot.
            chat: The chat format that wraps the user text.
            question: The question for a ``{question}`` slot.

        Returns:
            The text, the stimulus span as ``response_span`` and the last
            concept occurrence of the user turn as ``concept_span``.

        Raises:
            ValueError: If the template has a ``{question}`` slot and
                ``question`` is None.
        """
        if "question" in self.slots and question is None:
            raise ValueError(f"template {self.name!r} needs a question")
        values = {
            "concept": concept,
            "stimulus": stimulus,
            "question": question or "",
        }
        user, spans = _fill(self.user, values)
        chat = chat if self.tagged else ChatFormat("none")
        text, user_start, _ = chat.render_turn(user, self.assistant)
        concept_span = None
        if "concept" in spans:
            start, end = spans["concept"][-1]
            concept_span = (user_start + start, user_start + end)
        start, end = spans["stimulus"][0]
        response_span = (user_start + start, user_start + end)
        return Rendered(text, response_span, concept_span)


@dataclass(frozen=True)
class FunctionTemplate:
    """A template that makes the model act out a function through a persona.

    The user text holds a ``{persona}`` slot and optionally an
    ``{instruction}`` slot. The response follows the assistant tag and is the
    marked span.

    Attributes:
        name: Identifier of the template.
        user: The user text with its slots.
        pos_personas: Persona wordings of the positive pole (honest, happy).
        neg_personas: Persona wordings of the negative pole.
        tagged: If false, the chat format is ignored and the text is rendered
            without tags (the "no chat tags" ablation).
    """

    name: str
    user: str
    pos_personas: tuple[str, ...]
    neg_personas: tuple[str, ...]
    tagged: bool = True

    def __post_init__(self) -> None:
        """Validates the slots and the personas."""
        _check_slots(self.name, self.user, _FUNCTION_SLOTS)
        if "persona" not in self.slots:
            raise ValueError(f"template {self.name!r} needs a {{persona}} slot")
        if not self.pos_personas or not self.neg_personas:
            raise ValueError(f"template {self.name!r} needs personas per pole")

    @property
    def slots(self) -> frozenset[str]:
        """The slot names that occur in the user text."""
        return frozenset(_SLOT.findall(self.user))

    def render(
        self,
        response: str,
        pole: Literal["pos", "neg"],
        chat: ChatFormat,
        *,
        index: int = 0,
        instruction: str | None = None,
    ) -> Rendered:
        """Renders one response under the persona of a pole.

        Args:
            response: The response text, inserted verbatim.
            pole: Which persona list to use.
            chat: The chat format that wraps the turns.
            index: Position in the persona list of the pole.
            instruction: The text for an ``{instruction}`` slot.

        Returns:
            The text and the response span as ``response_span``.

        Raises:
            ValueError: If ``pole`` is unknown, ``index`` is out of range, or
                an ``{instruction}`` slot has no ``instruction``.
        """
        if pole not in ("pos", "neg"):
            raise ValueError(f"pole must be 'pos' or 'neg', got {pole!r}")
        personas = self.pos_personas if pole == "pos" else self.neg_personas
        if not 0 <= index < len(personas):
            raise ValueError(
                f"index must be in [0, {len(personas)}), got {index}"
            )
        if "instruction" in self.slots and instruction is None:
            raise ValueError(f"template {self.name!r} needs an instruction")
        values = {"persona": personas[index], "instruction": instruction or ""}
        user, _ = _fill(self.user, values)
        chat = chat if self.tagged else ChatFormat("none")
        text, _, start = chat.render_turn(user, response)
        return Rendered(text, (start, start + len(response)), None)


def _check_slots(name: str, user: str, allowed: frozenset[str]) -> None:
    """Raises if the user text has a slot outside ``allowed``."""
    unknown = set(_SLOT.findall(user)) - allowed
    if unknown:
        raise ValueError(
            f"template {name!r} has unsupported slots {sorted(unknown)}"
        )


def _fill(
    text: str, values: Mapping[str, str]
) -> tuple[str, dict[str, list[Span]]]:
    """Fills slots in one pass and records the span of every filled slot."""
    parts: list[str] = []
    spans: dict[str, list[Span]] = {}
    written = 0
    last = 0
    for match in _SLOT.finditer(text):
        literal = text[last : match.start()]
        slot = match.group(1)
        value = values[slot]
        parts.extend((literal, value))
        written += len(literal)
        spans.setdefault(slot, []).append((written, written + len(value)))
        written += len(value)
        last = match.end()
    parts.append(text[last:])
    return "".join(parts), spans
