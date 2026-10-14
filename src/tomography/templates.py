"""Chat formats and stimulus templates with exact character spans.

Rendering never strips or normalizes text: the text of every marked span is
the substituted slot itself, so ``text[start:end]`` can be mapped to tokens
through tokenizer offsets.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
import re
from types import MappingProxyType
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


Template = ConceptTemplate | FunctionTemplate

EMOTION_PERSONAS: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] = (
    MappingProxyType(
        {
            "happiness": (
                ("joyful", "happy", "cheerful"),
                ("dejected", "unhappy", "dispirited"),
            ),
            "sadness": (
                ("sad", "depressed", "miserable"),
                ("cheerful", "optimistic", "happy"),
            ),
            "anger": (
                ("angry", "furious", "irritated"),
                ("pleased", "calm", "peaceful"),
            ),
            "fear": (
                ("fearful", "scared", "frightened"),
                ("fearless", "bold", "unafraid"),
            ),
            "disgust": (
                ("disgusted", "sicken", "revolted"),
                ("approved", "delighted", "satisfied"),
            ),
            "surprise": (
                ("surprised", "shocked", "astonished"),
                ("unimpressed", "indifferent", "bored"),
            ),
        }
    )
)
EMOTIONS = tuple(EMOTION_PERSONAS)

UNRELATED_CONCEPTS = ("furniture", "geometry", "weather", "cooking")
SCAFFOLDS = ("This movie is", "This text is", "Overall, it was")
ABLATION_KINDS = (
    "paper",
    "no_template",
    "no_chat_tags",
    "concept_swap",
    "no_trailing_question",
    "paraphrase",
    "neutral_frame",
    "scaffold",
    "framing_swap",
)

_PARAPHRASES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "concept": (
            "Rate how much {concept} the following text expresses:\n"
            "{stimulus}\nThe amount of {concept} is",
            "How much {concept} is there in this text?\n"
            "{stimulus}\nThe amount of {concept} is",
            "Read the text below and assess its {concept}.\n"
            "{stimulus}\nIts {concept} is",
        ),
        "emotion_concept": (
            "Rate the {concept} in the following scenario:\n"
            "Scenario: {stimulus}\nAnswer:",
            "Think about the {concept} of this scenario:\n"
            "Scenario: {stimulus}\nAnswer:",
        ),
        "honesty": (
            "Imagine you are {persona} person stating facts about the world.",
            "Act as {persona} person who makes claims about the world.",
            "Suppose you are {persona} speaker describing the world.",
        ),
        "emotion_function": (
            "Behave as if you feel extremely {persona}.",
            "Respond as someone who is extremely {persona}.",
        ),
    }
)
_FRAMING_COUNTERPART = MappingProxyType(
    {
        "emotion_concept": "emotion_function",
        "emotion_concept_paper": "emotion_function",
        "emotion_function": "emotion_concept",
    }
)


def emotion_function_template(emotion: str) -> FunctionTemplate:
    """Builds the emotion function template of the reference code.

    Args:
        emotion: One of ``EMOTIONS``.

    Returns:
        The template ``Act as if you are extremely {persona}.`` with three
        adjectives and three antonyms of the emotion as personas.

    Raises:
        ValueError: If the emotion is unknown.
    """
    if emotion not in EMOTION_PERSONAS:
        raise ValueError(f"unknown emotion {emotion!r}, expected {EMOTIONS}")
    pos, neg = EMOTION_PERSONAS[emotion]
    return FunctionTemplate(
        "emotion_function", "Act as if you are extremely {persona}.", pos, neg
    )


PAPER_TEMPLATES: Mapping[str, Template] = MappingProxyType(
    {
        template.name: template
        for template in (
            ConceptTemplate(
                "concept",
                "Consider the amount of {concept} in the following:\n"
                "{stimulus}\nThe amount of {concept} is",
            ),
            ConceptTemplate(
                "truthfulqa",
                "Consider the amount of truthfulness in the following "
                "answer:\nQuestion: {question}\nAnswer: {stimulus}\n"
                "The amount of truthfulness in the answer is",
            ),
            ConceptTemplate(
                "arc",
                "Consider the correctness of the answer to the following "
                "question:\nQuestion: {question}\nAnswer: {stimulus}\n"
                "The probability of the answer being correct is",
            ),
            ConceptTemplate(
                "utility",
                "Consider the happiness in the following scenario:\n"
                "Scenario: {stimulus}",
                assistant="The happiness of the scenario is",
            ),
            ConceptTemplate(
                "morality",
                "Consider the acceptability of the action in the following "
                "scenario:\nScenario: {stimulus}",
            ),
            ConceptTemplate(
                "emotion_concept",
                "Consider the {concept} of the following scenario:\n"
                "Scenario: {stimulus}\nAnswer:",
            ),
            ConceptTemplate(
                "emotion_concept_paper",
                "Consider the {concept} in the following scenario:\n"
                "Scenario: {stimulus}",
            ),
            FunctionTemplate(
                "honesty",
                "Pretend you're {persona} person making statements about the "
                "world.",
                pos_personas=("an honest",),
                neg_personas=("an untruthful",),
            ),
            FunctionTemplate(
                "honesty_control",
                "{instruction} Give {persona} answer.",
                pos_personas=("a truthful",),
                neg_personas=("an untruthful",),
            ),
            emotion_function_template("happiness"),
        )
    }
)


def ablate(template: Template, kind: str, *, index: int = 0) -> Template:
    """Returns a variant of a template for the prompt ablation study.

    Kinds (``ABLATION_KINDS``): ``paper`` is the template itself;
    ``no_template`` keeps only the stimulus; ``no_chat_tags`` keeps the
    wording but ignores the chat format; ``concept_swap`` replaces the concept
    word with ``UNRELATED_CONCEPTS[index]`` (the variant then has no concept
    span); ``no_trailing_question`` cuts the text after the stimulus;
    ``paraphrase`` uses wording number ``index`` of a known template;
    ``neutral_frame`` wraps the stimulus in a frame that names no concept;
    ``scaffold`` is a conclusion stub ending in ``SCAFFOLDS[index]``;
    ``framing_swap`` returns the other framing of the emotion templates (for a
    concept template ``index`` picks the emotion in ``EMOTIONS``).

    Args:
        template: The template to vary. Chained ablations resolve known
            templates through the name before the first colon.
        kind: One of ``ABLATION_KINDS``.
        index: Selects among the options of the kinds that have several.

    Returns:
        A new template, or ``template`` itself for ``paper``.

    Raises:
        ValueError: If the kind is unknown or does not apply to the template,
            or ``index`` is out of range.
    """
    if kind not in ABLATION_KINDS:
        raise ValueError(f"unknown ablation kind {kind!r}: {ABLATION_KINDS}")
    if kind == "paper":
        return template
    if kind == "no_chat_tags":
        return replace(template, name=_name(template, kind), tagged=False)
    if kind == "paraphrase":
        return _paraphrase(template, index)
    if kind == "framing_swap":
        return _swap_framing(template, index)
    if not isinstance(template, ConceptTemplate):
        raise ValueError(
            f"ablation {kind!r} needs a concept template, got {template.name!r}"
        )
    if kind == "concept_swap":
        if "concept" not in template.slots:
            raise ValueError(f"template {template.name!r} has no concept slot")
        word = _pick(UNRELATED_CONCEPTS, index)
        user = template.user.replace("{concept}", word)
        return replace(template, name=_name(template, kind, index), user=user)
    name = _name(template, kind, index if kind == "scaffold" else None)
    if kind == "no_trailing_question":
        stimulus_end = next(
            m.end()
            for m in _SLOT.finditer(template.user)
            if m.group(1) == "stimulus"
        )
        if stimulus_end == len(template.user) and not template.assistant:
            raise ValueError(f"template {template.name!r} has no trailing text")
        user = template.user[:stimulus_end]
    elif kind == "no_template":
        user = "{stimulus}"
    elif kind == "neutral_frame":
        user = "Consider the following text:\n{stimulus}"
    else:
        user = f"{{stimulus}}\nConclusion: {_pick(SCAFFOLDS, index)}"
    return replace(template, name=name, user=user, assistant="")


def _name(template: Template, kind: str, index: int | None = None) -> str:
    """Names an ablated template after its source and the kind."""
    suffix = "" if index is None else f"#{index}"
    return f"{template.name}:{kind}{suffix}"


def _pick(options: Sequence[str], index: int) -> str:
    """Returns ``options[index]`` or raises a ``ValueError``."""
    if not 0 <= index < len(options):
        raise ValueError(f"index must be in [0, {len(options)}), got {index}")
    return options[index]


def _paraphrase(template: Template, index: int) -> Template:
    """Swaps the wording of a known template for paraphrase ``index``."""
    options = _PARAPHRASES.get(template.name.partition(":")[0])
    if options is None:
        raise ValueError(f"no paraphrase for template {template.name!r}")
    name = _name(template, "paraphrase", index)
    return replace(template, name=name, user=_pick(options, index))


def _swap_framing(template: Template, index: int) -> Template:
    """Returns the concept or function counterpart of an emotion template."""
    base = template.name.partition(":")[0]
    if base not in _FRAMING_COUNTERPART:
        raise ValueError(f"template {template.name!r} has no counterpart")
    if isinstance(template, FunctionTemplate):
        return PAPER_TEMPLATES[_FRAMING_COUNTERPART[base]]
    if not 0 <= index < len(EMOTIONS):
        raise ValueError(f"index must be in [0, {len(EMOTIONS)}), got {index}")
    return emotion_function_template(EMOTIONS[index])
