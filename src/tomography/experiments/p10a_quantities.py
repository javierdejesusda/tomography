"""P10a: how models answer quantity questions with a range or one value.

Each question asks for a quantity whose sourced answer is either a range (the
height of a dog breed, the gestation period of a species) or one exact value
(the height of a building, the length of a film). The experiment asks every
model for a free answer and for an answer under the instruction to give one
exact number, in a few-shot format that every model shares and in the chat
template of post-trained models. For each question it keeps one greedy
completion and several sampled ones, reads each completion as a point, a
range, an abstention or nothing, and writes the rates with Wilson intervals,
the gap in hedging between range and exact items, how often a forced point
lands inside the sourced range, and a hand-check file for the answer parser.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from tomography.datasets.quantities import DEFAULT_MIN_SITELINKS
from tomography.datasets.quantities import FAMILIES
from tomography.experiments.base import ExperimentConfig
from tomography.knowledge import few_shot_prompt
from tomography.templates import ChatFormat

CONDITIONS: Final = ("free", "forced")
FORMATS: Final = ("fewshot", "chat")

FORCED_INSTRUCTION: Final = "Answer each question with one exact number."
CHAT_INSTRUCTIONS: Final = {
    "free": "Answer the question briefly.",
    "forced": "Answer with one exact number only.",
}
FREE_SHOTS: Final[tuple[tuple[str, str], ...]] = (
    ("How many keys does a standard piano have?", "88"),
    (
        "What is the resting heart rate of a healthy adult, in beats per "
        "minute?",
        "60 to 100",
    ),
    ("How many days are in a leap year?", "366"),
    ("How many hours of sleep does an adult need per night?", "7 to 9"),
    ("In what year did the first crewed Moon landing happen?", "1969"),
    ("How long does a house cat live, in years?", "12 to 18"),
)
FORCED_SHOTS: Final[tuple[tuple[str, str], ...]] = (
    ("How many keys does a standard piano have?", "88"),
    (
        "What is the resting heart rate of a healthy adult, in beats per "
        "minute?",
        "72",
    ),
    ("How many days are in a leap year?", "366"),
    ("How many hours of sleep does an adult need per night?", "8"),
    ("In what year did the first crewed Moon landing happen?", "1969"),
    ("How long does a house cat live, in years?", "15"),
)

UNIT_WORDS: Final[dict[str, tuple[str, ...]]] = {
    "centimetres": (
        "centimetres",
        "centimetre",
        "centimeters",
        "centimeter",
        "cm",
    ),
    "metres": ("metres", "metre", "meters", "meter", "m"),
    "days": ("days", "day"),
    "minutes": ("minutes", "minute", "mins", "min"),
}
UNIT_DIMENSIONS: Final = {
    "centimetres": "length",
    "metres": "length",
    "days": "time",
    "minutes": "time",
}


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """P10a settings: the questions, the prompts and the sampling.

    Attributes:
        families: Names of the question families to load; None loads all.
        n_per_side: Largest number of items per family and kind.
        min_sitelinks: Smallest sitelink count of an entity.
        conditions: "free" asks for an answer; "forced" tells the model to
            give one exact number.
        formats: "fewshot" is the raw few-shot prompt every model gets;
            "chat" is the chat template, used only for post-trained models
            and skipped for the others.
        n_samples: Sampled completions per question, besides the greedy one.
        temperature: Sampling temperature.
        max_new_tokens: Generation budget per answer.
        rel_tol: Relative tolerance of the grader around the sourced range.
        n_hand_check: Rows of the hand-check file.
        batch_size: Prompts per generation call chunk.
        offline: Whether the questions are read from the cache only.
    """

    experiment: str = "p10a"
    model: str = "gpt2"
    families: tuple[str, ...] | None = None
    n_per_side: int = 120
    min_sitelinks: int = DEFAULT_MIN_SITELINKS
    conditions: tuple[str, ...] = CONDITIONS
    formats: tuple[str, ...] = FORMATS
    n_samples: int = 5
    temperature: float = 1.0
    max_new_tokens: int = 32
    rel_tol: float = 0.05
    n_hand_check: int = 120
    batch_size: int = 8
    offline: bool = False

    def __post_init__(self) -> None:
        """Checks the settings that would act silently or fail late.

        Raises:
            ValueError: If `families` is empty, repeats a name or names a
                family the loader does not have, `conditions` or `formats`
                is empty, repeats a name or holds an unknown one, a count is
                below its minimum (`n_per_side`, `n_samples`,
                `max_new_tokens` and `batch_size` need 1, `n_hand_check` and
                `min_sitelinks` need 0), `temperature` is not positive or
                `rel_tol` is negative.
        """
        if self.families is not None:
            _check_names("families", self.families, [f.name for f in FAMILIES])
        _check_names("conditions", self.conditions, CONDITIONS)
        _check_names("formats", self.formats, FORMATS)
        for name in ("n_per_side", "n_samples", "max_new_tokens", "batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        for name in ("n_hand_check", "min_sitelinks"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if not self.temperature > 0:
            raise ValueError(
                f"temperature must be positive: {self.temperature}"
            )
        if not self.rel_tol >= 0:
            raise ValueError(f"rel_tol must not be negative: {self.rel_tol}")


def _check_names(
    field: str, names: tuple[str, ...], allowed: Sequence[str]
) -> None:
    """Checks that a setting names at least one known option, each once.

    Args:
        field: The setting, for the message.
        names: The chosen options.
        allowed: The known options.

    Raises:
        ValueError: If `names` is empty, repeats an option or holds one that
            is not in `allowed`.
    """
    if not names:
        raise ValueError(f"{field} must not be empty")
    if len(set(names)) != len(names):
        raise ValueError(f"{field} repeats a name: {names}")
    unknown = sorted(set(names) - set(allowed))
    if unknown:
        raise ValueError(f"{field} names unknown {unknown}; known: {allowed}")


def unit_words(unit: str) -> tuple[str, ...]:
    """Returns the words that name a question unit in an answer.

    Args:
        unit: The unit of a `QuantityQuestion`, such as "metres".

    Returns:
        Its spellings and abbreviations.

    Raises:
        ValueError: If the unit is not one the experiment knows.
    """
    try:
        return UNIT_WORDS[unit]
    except KeyError:
        raise ValueError(
            f"unknown unit {unit!r}; known units: {sorted(UNIT_WORDS)}"
        ) from None


def build_prompt(
    question: str, condition: str, fmt: str, chat: ChatFormat | None
) -> str:
    """Builds the prompt of one question.

    Args:
        question: The question text.
        condition: "free" or "forced".
        fmt: "fewshot" or "chat".
        chat: The model's chat format; used for "chat" only.

    Returns:
        The few-shot prompt, whose forced version opens with the
        instruction, or the chat turn with the instruction and the question.

    Raises:
        ValueError: If the condition or the format is unknown, or the format
            is "chat" without a chat format.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    if fmt == "fewshot":
        if condition == "free":
            return few_shot_prompt(question, FREE_SHOTS)
        return (
            f"{FORCED_INSTRUCTION}\n\n{few_shot_prompt(question, FORCED_SHOTS)}"
        )
    if fmt == "chat":
        if chat is None:
            raise ValueError("the chat format needs a chat template")
        instruction = CHAT_INSTRUCTIONS[condition]
        return chat.render_turn(f"{instruction}\n\nQuestion: {question}")[0]
    raise ValueError(f"unknown format {fmt!r}")
