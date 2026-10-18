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
import re
from typing import Any
from typing import Final

import numpy as np
from numpy.typing import NDArray

from tomography.datasets.quantities import DEFAULT_MIN_SITELINKS
from tomography.datasets.quantities import FAMILIES
from tomography.datasets.quantities import QuantityQuestion
from tomography.experiments.base import ExperimentConfig
from tomography.knowledge import few_shot_prompt
from tomography.p10.answers import answer_text
from tomography.p10.answers import grade
from tomography.p10.answers import parse_answer
from tomography.reading.metrics import wilson_ci
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
DIMENSION_WORDS: Final[dict[str, tuple[str, ...]]] = {
    "length": (
        "millimetres",
        "millimetre",
        "millimeters",
        "millimeter",
        "mm",
        "centimetres",
        "centimetre",
        "centimeters",
        "centimeter",
        "cm",
        "metres",
        "metre",
        "meters",
        "meter",
        "m",
        "kilometres",
        "kilometre",
        "kilometers",
        "kilometer",
        "km",
        "inches",
        "inch",
        "feet",
        "foot",
        "ft",
        "yards",
        "yard",
        "miles",
        "mile",
    ),
    "time": (
        "seconds",
        "second",
        "secs",
        "sec",
        "minutes",
        "minute",
        "mins",
        "min",
        "hours",
        "hour",
        "hrs",
        "hr",
        "days",
        "day",
        "weeks",
        "week",
        "months",
        "month",
        "years",
        "year",
    ),
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


def _word_pattern(words: Sequence[str]) -> re.Pattern[str]:
    """Compiles a pattern for whole words, longest first, ignoring case."""
    ordered = sorted(words, key=len, reverse=True)
    alternatives = "|".join(re.escape(word) for word in ordered)
    return re.compile(rf"(?<![A-Za-z])(?:{alternatives})(?![A-Za-z])", re.I)


@dataclass(frozen=True)
class Answer:
    """One completion read against a question.

    Attributes:
        text: The generated text, uncut.
        form: "point", "range", "abstain" or "none".
        low: The value, or the lower end of a range; None without one.
        high: The value, or the upper end of a range; None without one.
        approximate: Whether the answer marks itself as approximate.
        correct: Whether the value or range fits the sourced range within the
            tolerance; None for an abstention or an answer without a value.
        unit_mismatch: Whether the answer line has a number, none of the
            question's unit words and a unit word of the same dimension, such
            as "9 months" for a question in days. Values are not converted.
    """

    text: str
    form: str
    low: float | None
    high: float | None
    approximate: bool
    correct: bool | None
    unit_mismatch: bool


def _unit_mismatch(line: str, item: QuantityQuestion) -> bool:
    """Whether a line with a number names a foreign unit of the dimension."""
    own = unit_words(item.unit)
    foreign = [
        word
        for word in DIMENSION_WORDS[UNIT_DIMENSIONS[item.unit]]
        if word not in own
    ]
    text = re.sub(re.escape(item.entity), " ", line, flags=re.I)
    return bool(
        re.search(r"\d", text)
        and not _word_pattern(own).search(text)
        and _word_pattern(foreign).search(text)
    )


def score_answer(
    item: QuantityQuestion, text: str, fmt: str, *, rel_tol: float
) -> Answer:
    """Reads one completion of a question.

    A few-shot completion is cut at its first newline, because the model
    goes on to write the next question there. A chat completion is read from
    its first non-empty line. The entity label is removed before any number
    is read, and the quantity followed by a unit word wins.

    Args:
        item: The question the completion answers.
        text: The generated text.
        fmt: "fewshot" or "chat".
        rel_tol: Relative tolerance of the grader.

    Returns:
        The reading of the completion.

    Raises:
        ValueError: If the question's unit is not one the experiment knows.
    """
    words = unit_words(item.unit)
    line = text.split("\n", 1)[0] if fmt == "fewshot" else text
    parsed = parse_answer(line, ignore=(item.entity,), unit_words=words)
    has_number = parsed.form in ("point", "range")
    return Answer(
        text=text,
        form=parsed.form,
        low=parsed.low,
        high=parsed.high,
        approximate=parsed.approximate,
        correct=grade(parsed, item.low, item.high, rel_tol=rel_tol),
        unit_mismatch=has_number and _unit_mismatch(answer_text(line), item),
    )


@dataclass(frozen=True)
class Outcome:
    """What a model answered to one question in one condition and format.

    Attributes:
        greedy: The greedy completion.
        samples: The sampled completions.
    """

    greedy: Answer
    samples: tuple[Answer, ...]


Cell = tuple[str, str]
Results = dict[Cell, list[Outcome]]

DECODINGS: Final = ("greedy", "sampled")
POOLED: Final = "all"
N_BOOT: Final = 2000
METRICS: Final = (
    "point",
    "range",
    "approximate",
    "abstain",
    "none",
    "unit_mismatch",
    "correct",
)
_KINDS: Final = ("range", "exact")


def _answers(outcome: Outcome, decoding: str) -> tuple[Answer, ...]:
    """Returns the answers of one decoding of an outcome."""
    return (outcome.greedy,) if decoding == "greedy" else outcome.samples


def _counts(answers: Sequence[Answer]) -> dict[str, int]:
    """Counts the forms, flags and correct answers of a set of answers."""
    return {
        "n": len(answers),
        "n_gradable": sum(a.correct is not None for a in answers),
        "point": sum(a.form == "point" for a in answers),
        "range": sum(a.form == "range" for a in answers),
        "abstain": sum(a.form == "abstain" for a in answers),
        "none": sum(a.form == "none" for a in answers),
        "approximate": sum(a.approximate for a in answers),
        "unit_mismatch": sum(a.unit_mismatch for a in answers),
        "correct": sum(a.correct is True for a in answers),
    }


def _rate(count: int, n: int) -> list[float | None]:
    """Returns a rate and its Wilson 95 percent interval; None when n is 0."""
    if n == 0:
        return [None, None, None]
    low, high = wilson_ci(count, n)
    return [count / n, low, high]


def _rate_columns(name: str) -> list[str]:
    return [name, f"{name}_rate", f"{name}_lo", f"{name}_hi"]


def _families_of(items: Sequence[QuantityQuestion]) -> list[str]:
    """Lists the families of the items, sorted, then the pooled label."""
    return [*sorted({item.family for item in items}), POOLED]


def _members(
    items: Sequence[QuantityQuestion], family: str, kind: str
) -> list[int]:
    """Returns the positions of the items of a family and kind."""
    return [
        i
        for i, item in enumerate(items)
        if item.kind == kind and family in (item.family, POOLED)
    ]


def build_table(
    items: Sequence[QuantityQuestion], results: Results
) -> tuple[list[str], list[list[Any]]]:
    """Builds the form and correctness rates per family, kind and cell.

    Every row covers the answers of one family (or all of them pooled), one
    kind, one condition, one format and one decoding. For the sampled
    decoding a trial is one completion, so the Wilson intervals treat the
    samples of an item as independent and are narrower than the item-level
    uncertainty; the hedge gap uses an item bootstrap instead.

    Args:
        items: The questions.
        results: The outcomes per (condition, format), aligned with items.

    Returns:
        The column names and the rows. Rates are over all answers, except
        `correct`, which is over the gradable ones (`n_gradable`).
    """
    columns = [
        "family",
        "kind",
        "condition",
        "format",
        "decoding",
        "n",
        "n_gradable",
        *(c for m in METRICS for c in _rate_columns(m)),
    ]
    rows: list[list[Any]] = []
    for family in _families_of(items):
        for kind in _KINDS:
            members = _members(items, family, kind)
            if not members:
                continue
            for (condition, fmt), outcomes in results.items():
                for decoding in DECODINGS:
                    answers = [
                        a
                        for i in members
                        for a in _answers(outcomes[i], decoding)
                    ]
                    counts = _counts(answers)
                    rows.append(
                        [
                            family,
                            kind,
                            condition,
                            fmt,
                            decoding,
                            counts["n"],
                            counts["n_gradable"],
                        ]
                    )
                    for metric in METRICS:
                        n = counts["n_gradable" if metric == "correct" else "n"]
                        rows[-1] += [
                            counts[metric],
                            *_rate(counts[metric], n),
                        ]
    return columns, rows


def bootstrap_gap_ci(
    range_rates: NDArray[np.float64],
    exact_rates: NDArray[np.float64],
    *,
    n_boot: int = N_BOOT,
    seed: int,
) -> tuple[float, float]:
    """Bootstraps the 95 percent interval of a difference of mean rates.

    Items are resampled with replacement within each kind.

    Args:
        range_rates: Per-item rate on the range items.
        exact_rates: Per-item rate on the exact items.
        n_boot: Number of resamples.
        seed: Seed of the resampling.

    Returns:
        The 2.5th and 97.5th percentiles of the resampled gaps.
    """
    rng = np.random.default_rng(seed)
    drawn = [
        rates[rng.integers(0, len(rates), size=(n_boot, len(rates)))].mean(
            axis=1
        )
        for rates in (range_rates, exact_rates)
    ]
    low, high = np.percentile(drawn[0] - drawn[1], [2.5, 97.5])
    return float(low), float(high)


def _item_rates(
    outcomes: Sequence[Outcome], members: Sequence[int], decoding: str
) -> NDArray[np.float64]:
    """Returns each member's share of range answers."""
    return np.array(
        [
            np.mean(
                [a.form == "range" for a in _answers(outcomes[i], decoding)]
            )
            for i in members
        ],
        dtype=np.float64,
    )


def hedge_gap_table(
    items: Sequence[QuantityQuestion], results: Results, *, seed: int
) -> tuple[list[str], list[list[Any]]]:
    """Builds the gap in range answers between range and exact items.

    The gap is the mean over range items of the share of range answers minus
    the same mean over exact items, per family and pooled, condition, format
    and decoding. The interval resamples items within each kind.

    Args:
        items: The questions.
        results: The outcomes per (condition, format), aligned with items.
        seed: Seed of the bootstrap.

    Returns:
        The column names and the rows; a family without items of both kinds
        has no row.
    """
    columns = [
        "family",
        "condition",
        "format",
        "decoding",
        "n_range_items",
        "n_exact_items",
        "range_rate_on_range_items",
        "range_rate_on_exact_items",
        "gap",
        "gap_lo",
        "gap_hi",
    ]
    rows: list[list[Any]] = []
    for family in _families_of(items):
        on_range = _members(items, family, "range")
        on_exact = _members(items, family, "exact")
        if not (on_range and on_exact):
            continue
        for (condition, fmt), outcomes in results.items():
            for decoding in DECODINGS:
                high = _item_rates(outcomes, on_range, decoding)
                low = _item_rates(outcomes, on_exact, decoding)
                gap_lo, gap_hi = bootstrap_gap_ci(high, low, seed=seed)
                rows.append(
                    [
                        family,
                        condition,
                        fmt,
                        decoding,
                        len(on_range),
                        len(on_exact),
                        float(high.mean()),
                        float(low.mean()),
                        float(high.mean() - low.mean()),
                        gap_lo,
                        gap_hi,
                    ]
                )
    return columns, rows


def forced_in_range_table(
    items: Sequence[QuantityQuestion], results: Results, *, rel_tol: float
) -> tuple[list[str], list[list[Any]]]:
    """Builds how often a forced greedy point lands in the sourced range.

    Only greedy point answers under the forced condition count. A point is
    inside when it lies in `[low, high]` and inside the band when it lies in
    the range widened by `rel_tol`, as the grader does. Exact items are the
    reference: their range is one value.

    Args:
        items: The questions.
        results: The outcomes per (condition, format), aligned with items.
        rel_tol: Relative tolerance of the band.

    Returns:
        The column names and the rows, one per family (and pooled), kind and
        format; no rows without the forced condition.
    """
    columns = [
        "family",
        "kind",
        "format",
        "n_items",
        "n_point",
        *_rate_columns("inside"),
        *_rate_columns("inside_band"),
    ]
    rows: list[list[Any]] = []
    for family in _families_of(items):
        for kind in _KINDS:
            members = _members(items, family, kind)
            if not members:
                continue
            for (condition, fmt), outcomes in results.items():
                if condition != "forced":
                    continue
                points = [
                    (items[i], outcomes[i].greedy.low)
                    for i in members
                    if outcomes[i].greedy.form == "point"
                ]
                inside = sum(
                    item.low <= value <= item.high
                    for item, value in points
                    if value is not None
                )
                band = sum(
                    value <= item.high * (1 + rel_tol)
                    and value >= item.low * (1 - rel_tol)
                    for item, value in points
                    if value is not None
                )
                rows.append(
                    [
                        family,
                        kind,
                        fmt,
                        len(members),
                        len(points),
                        inside,
                        *_rate(inside, len(points)),
                        band,
                        *_rate(band, len(points)),
                    ]
                )
    return columns, rows


def width_table(
    items: Sequence[QuantityQuestion], results: Results
) -> tuple[list[str], list[list[Any]]]:
    """Builds range and correct rates by the width tertile of range items.

    The width of a range item is `(high - low) / midpoint`. Range items of
    all families are ordered by width (ties by uid) and cut into three equal
    groups; with fewer than three items the empty groups have no row.

    Args:
        items: The questions.
        results: The outcomes per (condition, format), aligned with items.

    Returns:
        The column names and the rows, one per condition, format, decoding
        and non-empty tertile.
    """
    columns = [
        "condition",
        "format",
        "decoding",
        "tertile",
        "width_lo",
        "width_hi",
        "n_items",
        "n",
        "n_gradable",
        *_rate_columns("range"),
        *_rate_columns("correct"),
    ]
    members = _members(items, POOLED, "range")

    def width(i: int) -> float:
        item = items[i]
        return (item.high - item.low) / ((item.high + item.low) / 2)

    ordered = sorted(members, key=lambda i: (width(i), items[i].uid))
    tertiles: list[list[int]] = [[], [], []]
    for rank, i in enumerate(ordered):
        tertiles[rank * 3 // len(ordered)].append(i)
    rows: list[list[Any]] = []
    for (condition, fmt), outcomes in results.items():
        for decoding in DECODINGS:
            for k, group in enumerate(tertiles, start=1):
                if not group:
                    continue
                answers = [
                    a for i in group for a in _answers(outcomes[i], decoding)
                ]
                counts = _counts(answers)
                rows.append(
                    [
                        condition,
                        fmt,
                        decoding,
                        k,
                        width(group[0]),
                        width(group[-1]),
                        len(group),
                        counts["n"],
                        counts["n_gradable"],
                        counts["range"],
                        *_rate(counts["range"], counts["n"]),
                        counts["correct"],
                        *_rate(counts["correct"], counts["n_gradable"]),
                    ]
                )
    return columns, rows
