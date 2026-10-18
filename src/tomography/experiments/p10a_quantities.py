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
import csv
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
from typing import Final

import numpy as np
from numpy.typing import NDArray

from tomography.control.generate import Completion
from tomography.control.generate import generate
from tomography.datasets.quantities import DEFAULT_MIN_SITELINKS
from tomography.datasets.quantities import FAMILIES
from tomography.datasets.quantities import QuantityQuestion
from tomography.datasets.quantities import load_quantities
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.sweep import chat_format_for
from tomography.knowledge import few_shot_prompt
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.p10.answers import answer_text
from tomography.p10.answers import grade
from tomography.p10.answers import parse_answer
from tomography.reading.metrics import wilson_ci
from tomography.templates import ChatFormat
from tomography.viz.style import PX
from tomography.viz.style import categorical_colors
from tomography.viz.style import figure_of
from tomography.viz.style import new_figure
from tomography.viz.style import save_figure
from tomography.viz.style import style_context

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


def _csv_cell(value: object) -> object:
    return "" if value is None else value


def _write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[Sequence[Any]]
) -> None:
    """Writes a CSV with unix line ends; None becomes an empty cell."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_csv_cell(value) for value in row])


def _value_text(answer: Answer) -> str:
    if answer.low is None or answer.high is None:
        return ""
    if answer.form == "range":
        return f"{answer.low:g} to {answer.high:g}"
    return f"{answer.low:g}"


def _grade_text(answer: Answer) -> str:
    if answer.correct is None:
        return ""
    return "correct" if answer.correct else "incorrect"


def hand_check_table(
    items: Sequence[QuantityQuestion],
    results: Results,
    *,
    n: int,
    seed: int,
) -> tuple[list[str], list[list[Any]]]:
    """Builds the rows a person checks the parser and the grader on.

    Items are drawn one at a time from each (family, kind) stratum in turn,
    each stratum in a seeded random order, until `n` rows are chosen or the
    items run out, so the strata have equal quotas as far as their sizes
    allow. The rows keep the order of `items`.

    Args:
        items: The questions.
        results: The outcomes per (condition, format), aligned with items.
        n: Largest number of rows.
        seed: Seed of the draw.

    Returns:
        The column names and the rows: the item with its Wikidata URL and
        gold range, for each cell the greedy completion (newlines shown as
        a backslash and n), its form, value, grade and unit flag, and empty
        columns for the verdict of the person.
    """
    strata: dict[tuple[str, str], list[int]] = {}
    for i, item in enumerate(items):
        strata.setdefault((item.family, item.kind), []).append(i)
    rng = np.random.default_rng(seed)
    queues = [list(rng.permutation(strata[key])) for key in sorted(strata)]
    chosen: list[int] = []
    while len(chosen) < n and any(queues):
        for queue in queues:
            if queue and len(chosen) < n:
                chosen.append(int(queue.pop()))
    columns = [
        "uid",
        "wikidata_url",
        "family",
        "kind",
        "question",
        "gold_low",
        "gold_high",
        "unit",
    ]
    for condition, fmt in results:
        stem = f"{condition}_{fmt}"
        columns += [
            f"{stem}_{field}"
            for field in ("completion", "form", "value", "grade", "unit_flag")
        ]
    columns += ["human_form", "human_correct", "human_unit_mismatch"]
    rows: list[list[Any]] = []
    for i in sorted(chosen):
        item = items[i]
        row: list[Any] = [
            item.uid,
            f"https://www.wikidata.org/wiki/{item.entity_id}",
            item.family,
            item.kind,
            item.question,
            item.low,
            item.high,
            item.unit,
        ]
        for outcomes in results.values():
            greedy = outcomes[i].greedy
            row += [
                greedy.text.replace("\n", "\\n"),
                greedy.form,
                _value_text(greedy),
                _grade_text(greedy),
                "mismatch" if greedy.unit_mismatch else "",
            ]
        rows.append([*row, "", "", ""])
    return columns, rows


def _item_records(
    items: Sequence[QuantityQuestion], results: Results
) -> list[dict[str, Any]]:
    """Lists each item with every parsed answer, for the reading study."""
    return [
        {
            **asdict(item),
            "answers": {
                f"{condition}/{fmt}": {
                    "greedy": asdict(outcomes[i].greedy),
                    "samples": [asdict(a) for a in outcomes[i].samples],
                }
                for (condition, fmt), outcomes in results.items()
            },
        }
        for i, item in enumerate(items)
    ]


def _summary(
    config: Config,
    items: Sequence[QuantityQuestion],
    formats: Sequence[str],
    n_hand_check: int,
) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    snapshots: dict[str, dict[str, dict[str, list[str]]]] = {}
    for item in items:
        counts.setdefault(item.family, {}).setdefault(item.kind, 0)
        counts[item.family][item.kind] += 1
        stamp = snapshots.setdefault(item.family, {}).setdefault(
            item.kind, {"retrieved_on": [], "sha256": []}
        )
        for field, value in (
            ("retrieved_on", item.retrieved_on),
            ("sha256", item.sha256),
        ):
            if value not in stamp[field]:
                stamp[field].append(value)
    for kinds in snapshots.values():
        for stamp in kinds.values():
            stamp["retrieved_on"].sort()
            stamp["sha256"].sort()
    return {
        "model": config.model,
        "n_items": len(items),
        "families": sorted(counts),
        "counts": counts,
        "snapshots": snapshots,
        "conditions": list(config.conditions),
        "formats": list(formats),
        "hand_check_rows": n_hand_check,
        "config": config.to_dict(),
    }


def _formats_for(
    config: Config, loaded: LoadedModel
) -> tuple[list[str], ChatFormat | None]:
    """Picks the formats to run: chat only for a post-trained model."""
    chat = chat_format_for(loaded)
    templated = chat.kind == "template"
    formats = [f for f in config.formats if f != "chat" or templated]
    if not formats:
        raise ValueError(
            f"no format applies to {config.model!r}: the chat format needs a "
            "post-trained model and `formats` has no other"
        )
    return formats, chat if templated else None


def _ask(
    config: Config,
    loaded: LoadedModel,
    items: Sequence[QuantityQuestion],
    condition: str,
    fmt: str,
    chat: ChatFormat | None,
) -> list[Outcome]:
    """Gets a greedy and sampled completions of every item in one cell."""
    prompts = [build_prompt(i.question, condition, fmt, chat) for i in items]
    greedy = generate(
        loaded,
        prompts,
        max_new_tokens=config.max_new_tokens,
        batch_size=config.batch_size,
    )
    sampled = generate(
        loaded,
        prompts,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        num_samples=config.n_samples,
        seed=derive_seed(config.seed, condition, fmt),
        batch_size=config.batch_size,
    )

    def read(item: QuantityQuestion, completion: Completion) -> Answer:
        return score_answer(item, completion.text, fmt, rel_tol=config.rel_tol)

    return [
        Outcome(
            greedy=read(item, one[0]),
            samples=tuple(read(item, c) for c in many),
        )
        for item, one, many in zip(items, greedy, sampled, strict=True)
    ]


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    items: Sequence[QuantityQuestion],
) -> None:
    """Asks every question in every condition and format and writes the files.

    Writes `table.csv`, `hedge_gap.csv`, `forced_in_range.csv`, `width.csv`,
    `hand_check.csv`, `items.json` (every parsed answer; the export step
    leaves it out) and `summary.json`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        items: The questions.

    Raises:
        ValueError: If there are no questions, a question has a unit the
            experiment does not know (checked before any generation), or no
            format applies to the model.
    """
    if not items:
        raise ValueError("no questions to ask")
    for unit in {item.unit for item in items}:
        unit_words(unit)
    formats, chat = _formats_for(config, loaded)
    results: Results = {
        (condition, fmt): _ask(config, loaded, items, condition, fmt, chat)
        for condition in config.conditions
        for fmt in formats
    }
    for name, (columns, rows) in {
        "table": build_table(items, results),
        "hedge_gap": hedge_gap_table(items, results, seed=config.seed),
        "forced_in_range": forced_in_range_table(
            items, results, rel_tol=config.rel_tol
        ),
        "width": width_table(items, results),
    }.items():
        _write_csv(run_dir.path / f"{name}.csv", columns, rows)
    columns, rows = hand_check_table(
        items, results, n=config.n_hand_check, seed=config.seed
    )
    _write_csv(run_dir.path / "hand_check.csv", columns, rows)
    run_dir.write_json("items", _item_records(items, results))
    run_dir.write_json("summary", _summary(config, items, formats, len(rows)))


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the questions and the model, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    families = [
        f
        for f in FAMILIES
        if config.families is None or f.name in config.families
    ]
    items = load_quantities(
        families=families,
        n_per_side=config.n_per_side,
        min_sitelinks=config.min_sitelinks,
        seed=config.seed,
        offline=config.offline,
    )
    execute(config, run_dir, load_model(config.model), items)


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the share of each answer form by kind and condition.

    The bars are the pooled greedy answers of the few-shot format (the chat
    format when the run has no few-shot cells), with Wilson 95 percent
    intervals.

    Args:
        run_dir: A finished P10a run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths.
    """
    with (run_dir.path / "table.csv").open(encoding="utf-8", newline="") as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["family"] == POOLED and r["decoding"] == "greedy"
        ]
    fmt = "fewshot" if any(r["format"] == "fewshot" for r in rows) else "chat"
    groups = [
        r
        for condition in CONDITIONS
        for kind in _KINDS[::-1]
        for r in rows
        if (r["condition"], r["kind"], r["format"]) == (condition, kind, fmt)
    ]
    forms = ("point", "range", "abstain", "none")
    colors = categorical_colors(len(forms))
    with style_context():
        _, axes = new_figure(aspect=0.6)
        ax = axes[0, 0]
        for k, (form, color) in enumerate(zip(forms, colors, strict=True)):
            x = np.arange(len(groups)) + (k - 1.5) * 0.18
            rate = np.array([float(g[f"{form}_rate"]) for g in groups])
            low = np.array([float(g[f"{form}_lo"]) for g in groups])
            high = np.array([float(g[f"{form}_hi"]) for g in groups])
            ax.errorbar(
                x,
                rate,
                yerr=[rate - low, high - rate],
                fmt="o",
                ms=3,
                color=color,
                elinewidth=PX,
                label=form,
            )
        ax.set_xticks(
            np.arange(len(groups)),
            [f"{g['kind']}\n{g['condition']}" for g in groups],
        )
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(f"Kind and condition ({fmt} format)")
        ax.set_ylabel("Share of greedy answers")
        ax.legend(
            ncol=len(forms),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            frameon=False,
        )
    return save_figure(figure_of(ax), out_dir, "form_shares")
