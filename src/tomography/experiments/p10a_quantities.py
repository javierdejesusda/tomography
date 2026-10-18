"""P10a: how models answer quantity questions with a range or one value.

Each question asks for a quantity whose sourced answer is either a range (the
height of a dog breed, the gestation period of a species) or one exact value
(the height of a building, the length of a film). The experiment asks every
model for a free answer and for an answer under the instruction to give one
exact number, in a few-shot format that every model shares and in the chat
template of post-trained models. For each question it keeps one greedy
completion and several sampled ones, reads each completion as a point, a
range, an abstention or nothing, and writes the rates with intervals (Wilson
for the greedy answers, an item bootstrap for the sampled ones),
the gap in hedging between range and exact items, how often a forced point
lands inside the sourced range, and a hand-check file for the answer parser.
A test-only set of people with one exact height is asked too, to tell whether
hedging follows the kind of answer or the animacy of the thing asked about.
"""

from collections.abc import Sequence
import csv
from dataclasses import asdict
from dataclasses import dataclass
import itertools
from pathlib import Path
import re
from typing import Any
from typing import Final
import warnings

import numpy as np
from numpy.typing import NDArray

from tomography.control.generate import Completion
from tomography.control.generate import generate
from tomography.datasets.quantities import DEFAULT_MIN_SITELINKS
from tomography.datasets.quantities import FAMILIES
from tomography.datasets.quantities import PEOPLE_FAMILY
from tomography.datasets.quantities import QuantityQuestion
from tomography.datasets.quantities import load_people_heights
from tomography.datasets.quantities import load_quantities
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.sweep import chat_format_for
from tomography.knowledge import few_shot_prompt
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.p10.answers import Parsed
from tomography.p10.answers import answer_text
from tomography.p10.answers import grade
from tomography.p10.answers import parse_answer
from tomography.reading.metrics import wilson_ci
from tomography.templates import ChatFormat
from tomography.viz.style import COLUMN_WIDTH
from tomography.viz.style import PX
from tomography.viz.style import categorical_colors
from tomography.viz.style import new_figure
from tomography.viz.style import save_figure
from tomography.viz.style import style_context

CONDITIONS: Final = ("free", "forced")
FORMATS: Final = ("fewshot", "chat")
SHOT_ORDERS: Final = ("range_last", "exact_last")
NO_ORDER: Final = "none"
BOTH: Final = "both"

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
    ("What is the normal pH of human blood?", "7.35 to 7.45"),
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
    ("What is the normal pH of human blood?", "7.4"),
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
        shot_orders: The orders of the few-shot examples. "range_last" is
            the alternating list that ends on a range answer; "exact_last"
            is the same list reversed, so it ends on an exact answer and
            only the form of the last shot differs. The chat format has no
            shots.
        n_samples: Sampled completions per question, besides the greedy one.
        temperature: Sampling temperature.
        max_new_tokens: Generation budget per answer.
        rel_tol: Relative tolerance of the grader around the sourced range.
        n_hand_check: Rows of each hand-check file.
        batch_size: Prompts per generation call chunk.
        offline: Whether the questions are read from the cache only.
        people: Whether to also ask the person-height items; they get their
            own rows and the hedge diagnostic, and no pooled statistic
            counts them.
    """

    experiment: str = "p10a"
    model: str = "gpt2"
    families: tuple[str, ...] | None = None
    n_per_side: int = 120
    min_sitelinks: int = DEFAULT_MIN_SITELINKS
    conditions: tuple[str, ...] = CONDITIONS
    formats: tuple[str, ...] = FORMATS
    shot_orders: tuple[str, ...] = SHOT_ORDERS
    n_samples: int = 5
    temperature: float = 1.0
    max_new_tokens: int = 32
    rel_tol: float = 0.05
    n_hand_check: int = 120
    batch_size: int = 8
    offline: bool = False
    people: bool = True

    def __post_init__(self) -> None:
        """Checks the settings that would act silently or fail late.

        Raises:
            ValueError: If `families` is empty, repeats a name or names a
                family the loader does not have, `conditions`, `formats` or
                `shot_orders` is empty, repeats a name or holds an unknown
                one, a count is below its minimum (`n_per_side`, `n_samples`,
                `max_new_tokens` and `batch_size` need 1, `n_hand_check` and
                `min_sitelinks` need 0), `temperature` is not positive or
                `rel_tol` is negative.
        """
        if self.families is not None:
            _check_names("families", self.families, [f.name for f in FAMILIES])
        _check_names("conditions", self.conditions, CONDITIONS)
        _check_names("formats", self.formats, FORMATS)
        _check_names("shot_orders", self.shot_orders, SHOT_ORDERS)
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


def shots_for(condition: str, order: str) -> tuple[tuple[str, str], ...]:
    """Returns the few-shot examples of a condition in a shot order.

    Args:
        condition: "free" or "forced".
        order: "range_last" for the constants' order, "exact_last" for the
            reverse.

    Returns:
        The (question, answer) pairs.

    Raises:
        ValueError: If the condition or the shot order is unknown.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    if order not in SHOT_ORDERS:
        raise ValueError(f"unknown shot order {order!r}")
    shots = FREE_SHOTS if condition == "free" else FORCED_SHOTS
    return shots if order == "range_last" else tuple(reversed(shots))


def build_prompt(
    question: str,
    condition: str,
    fmt: str,
    chat: ChatFormat | None,
    order: str = "range_last",
) -> str:
    """Builds the prompt of one question.

    Args:
        question: The question text.
        condition: "free" or "forced".
        fmt: "fewshot" or "chat".
        chat: The model's chat format; used for "chat" only.
        order: The shot order of the few-shot format; the chat format has no
            shots and ignores it.

    Returns:
        The few-shot prompt, whose forced version opens with the
        instruction, or the chat turn with the instruction and the question.

    Raises:
        ValueError: If the condition, the format or the shot order is
            unknown, or the format is "chat" without a chat format.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    if fmt == "fewshot":
        demos = few_shot_prompt(question, shots_for(condition, order))
        if condition == "free":
            return demos
        return f"{FORCED_INSTRUCTION}\n\n{demos}"
    if fmt == "chat":
        if chat is None:
            raise ValueError("the chat format needs a chat template")
        instruction = CHAT_INSTRUCTIONS[condition]
        return chat.render_turn(f"{instruction}\n\nQuestion: {question}")[0]
    raise ValueError(f"unknown format {fmt!r}")


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
        unit_mismatch: Whether the answer gives a number in a unit of the
            same dimension as the question's, such as "9 months" for a
            question in days, or a compound such as "2 hours 22 minutes".
            Values are not converted.
        finished: Whether the model ended the completion itself, with an
            end-of-sequence token, within the token budget.
        truncated: Whether the token budget cut the answer: not finished,
            and for a few-shot completion no newline was reached either.
    """

    text: str
    form: str
    low: float | None
    high: float | None
    approximate: bool
    correct: bool | None
    unit_mismatch: bool
    finished: bool = True
    truncated: bool = False


_QUANTITY: Final = r"\d[\d,]*(?:\.\d+)?"
_NUMBER_WORDS: Final = {
    "a": 1.0,
    "an": 1.0,
    "half": 0.5,
    "twelve": 12.0,
    "eleven": 11.0,
    "three": 3.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "ten": 10.0,
    "one": 1.0,
    "two": 2.0,
}
_JOINER: Final = re.compile(r"[\s,]*(?:and)?[\s,]*", re.I)


def _unit_mismatch(line: str, item: QuantityQuestion, parsed: Parsed) -> bool:
    """Whether a number is given in a foreign unit of the dimension.

    A unit word counts only when it directly follows a number, so "I'm" and
    "second" in "ranked second" are not units. The answer is flagged when
    none of the question's unit words follows the chosen quantity and a
    foreign unit follows some number ("9 months"), or when a foreign unit
    sits in a compound answer, next to another quantity with a unit
    ("2 hours 22 minutes", "an hour and 22 minutes"), where one of the two
    is the chosen quantity. The first member may be a number word (a, an,
    half, one to twelve). Alternatives joined by "or" and restatements in
    parentheses are not compound.

    Args:
        line: The answer line.
        item: The question.
        parsed: The reading of the line.

    Returns:
        Whether the answer is flagged; values are not converted.
    """
    own = {word.lower() for word in unit_words(item.unit)}
    words = DIMENSION_WORDS[UNIT_DIMENSIONS[item.unit]]
    foreign = {word.lower() for word in words if word.lower() not in own}
    alternatives = "|".join(
        re.escape(word) for word in sorted(words, key=len, reverse=True)
    )
    pattern = re.compile(
        rf"(?:(?P<number>{_QUANTITY})\s*"
        rf"|(?<![A-Za-z'])(?P<word>{'|'.join(_NUMBER_WORDS)})\s+)"
        rf"(?P<unit>{alternatives})(?![A-Za-z])",
        re.I,
    )
    text = re.sub(re.escape(item.entity), " ", line, flags=re.I)
    pairs = [
        (
            m,
            float(m["number"].replace(",", ""))
            if m["number"]
            else _NUMBER_WORDS[m["word"].lower()],
            m["unit"].lower(),
            m["number"] is not None,
        )
        for m in pattern.finditer(text)
    ]
    chosen = {parsed.low, parsed.high}
    if any(
        unit in foreign and digits for _, _, unit, digits in pairs
    ) and not any(
        unit in own and value in chosen for _, value, unit, _ in pairs
    ):
        return True
    return any(
        (first_unit in foreign or second_unit in foreign)
        and (first_value in chosen or second_value in chosen)
        and second_digits
        and _JOINER.fullmatch(text[first.end() : second.start()]) is not None
        for (first, first_value, first_unit, _), (
            second,
            second_value,
            second_unit,
            second_digits,
        ) in itertools.pairwise(pairs)
    )


def score_answer(
    item: QuantityQuestion,
    text: str,
    fmt: str,
    *,
    rel_tol: float,
    finished: bool = True,
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
        finished: Whether the completion ended with an end-of-sequence token.

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
        unit_mismatch=has_number
        and _unit_mismatch(answer_text(line), item, parsed),
        finished=finished,
        truncated=not finished and (fmt == "chat" or "\n" not in text),
    )


@dataclass(frozen=True)
class Outcome:
    """What a model answered to one question in one cell.

    Attributes:
        greedy: The greedy completion.
        samples: The sampled completions.
    """

    greedy: Answer
    samples: tuple[Answer, ...]


Cell = tuple[str, str, str]
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
    "truncated",
    "correct",
)
_KINDS: Final = ("range", "exact")
_HEIGHT: Final = "height"


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
        "truncated": sum(a.truncated for a in answers),
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
    """Returns the positions of the items of a family and kind.

    The pooled label covers the `FAMILIES` items; the people are their own
    family only.
    """
    return [
        i
        for i, item in enumerate(items)
        if item.kind == kind
        and (
            family == item.family
            or (family == POOLED and item.family != PEOPLE_FAMILY)
        )
    ]


def cluster_cis(
    hits: NDArray[np.float64],
    trials: NDArray[np.float64],
    *,
    n_boot: int = N_BOOT,
    seed: int,
) -> NDArray[np.float64]:
    """Bootstraps 95 percent intervals of rates by resampling items.

    Each item contributes a numerator and a denominator per metric (its
    count of a form and its number of answers), and an item is resampled
    with all its answers, so the answers of one item are not treated as
    independent. A resample whose denominator is zero for a metric is left
    out of that metric's interval.

    Args:
        hits: Per-item counts, shape (items, metrics).
        trials: Per-item denominators, same shape.
        n_boot: Number of resamples.
        seed: Seed of the resampling.

    Returns:
        The 2.5th and 97.5th percentiles of the resampled rates, shape
        (metrics, 2); NaN for a metric no resample could rate.
    """
    rng = np.random.default_rng(seed)
    n_items = hits.shape[0]
    drawn = rng.integers(0, n_items, size=(n_boot, n_items))
    top = hits[drawn].sum(axis=1)
    bottom = trials[drawn].sum(axis=1)
    rates = np.full(top.shape, np.nan)
    np.divide(top, bottom, out=rates, where=bottom > 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cis = np.nanpercentile(rates, [2.5, 97.5], axis=0)
    return np.asarray(cis.T, dtype=np.float64)


def _rate_cells(
    per_item: Sequence[dict[str, int]],
    metrics: Sequence[str],
    decoding: str,
    seed: int,
) -> tuple[str, list[str], dict[str, list[float | None]]]:
    """Rates and 95 percent intervals of the metrics of one group of items.

    The greedy decoding has one answer per item and gets Wilson intervals;
    the sampled decoding gets the item bootstrap of `cluster_cis`. A sampled
    rate of exactly 0 or 1 makes every resample agree, so its interval would
    collapse to a point; it gets the Wilson interval with the number of items
    that have a denominator as n instead. The rate of `correct` is over the
    gradable answers and the others are over all.

    Args:
        per_item: The `_counts` of each item's answers.
        metrics: The metrics to rate.
        decoding: "greedy" or "sampled".
        seed: Seed of the bootstrap.

    Returns:
        The interval method ("wilson", "item_bootstrap", "item_wilson" or
        "item_bootstrap+item_wilson" when a sampled group mixes both), the
        metrics whose interval is the item Wilson one, and, per metric, its
        rate, lower and upper bounds (None when it has no denominator).
    """
    totals = {key: sum(c[key] for c in per_item) for key in per_item[0]}

    def denominator(metric: str, item: dict[str, int]) -> int:
        return item["n_gradable" if metric == "correct" else "n"]

    if decoding == "greedy":
        return (
            "wilson",
            [],
            {m: _rate(totals[m], denominator(m, totals)) for m in metrics},
        )
    hits = np.array(
        [[c[m] for m in metrics] for c in per_item], dtype=np.float64
    )
    trials = np.array(
        [[denominator(m, c) for m in metrics] for c in per_item],
        dtype=np.float64,
    )
    cis = cluster_cis(hits, trials, seed=seed)
    cells: dict[str, list[float | None]] = {}
    item_wilson: list[str] = []
    for k, metric in enumerate(metrics):
        n = denominator(metric, totals)
        if n == 0 or np.isnan(cis[k]).any():
            cells[metric] = [None, None, None]
        elif totals[metric] in (0, n):
            n_items = int((trials[:, k] > 0).sum())
            low, high = wilson_ci(n_items * totals[metric] // n, n_items)
            cells[metric] = [totals[metric] / n, low, high]
            item_wilson.append(metric)
        else:
            cells[metric] = [
                totals[metric] / n,
                float(cis[k, 0]),
                float(cis[k, 1]),
            ]
    rated = sum(cells[m][0] is not None for m in metrics)
    if not item_wilson:
        method = "item_bootstrap"
    elif len(item_wilson) == rated:
        method = "item_wilson"
    else:
        method = "item_bootstrap+item_wilson"
    return method, item_wilson, cells


def build_table(
    items: Sequence[QuantityQuestion], results: Results, *, seed: int
) -> tuple[list[str], list[list[Any]]]:
    """Builds the form and correctness rates per family, kind and cell.

    Every row covers the answers of one family (or all of them pooled), one
    kind, one cell and one decoding. The `interval` column says how the
    bounds were made: Wilson for the greedy decoding, and for the sampled
    decoding an item bootstrap (2000 resamples of the items with all their
    samples), because the samples of an item are not independent.

    Args:
        items: The questions.
        results: The outcomes per cell, aligned with items.
        seed: Seed of the bootstrap.

    Returns:
        The column names and the rows. Rates are over all answers, except
        `correct`, which is over the gradable ones (`n_gradable`).
    """
    columns = [
        "family",
        "kind",
        "condition",
        "format",
        "shot_order",
        "decoding",
        "interval",
        "item_wilson",
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
            for (condition, fmt, order), outcomes in results.items():
                for decoding in DECODINGS:
                    per_item = [
                        _counts(_answers(outcomes[i], decoding))
                        for i in members
                    ]
                    method, wilson, cells = _rate_cells(
                        per_item, METRICS, decoding, seed
                    )
                    n = sum(c["n"] for c in per_item)
                    rows.append(
                        [
                            family,
                            kind,
                            condition,
                            fmt,
                            order,
                            decoding,
                            method,
                            ";".join(wilson),
                            n,
                            sum(c["n_gradable"] for c in per_item),
                        ]
                    )
                    for metric in METRICS:
                        count = sum(c[metric] for c in per_item)
                        rows[-1] += [count, *cells[metric]]
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
    outcome_sets: Sequence[Sequence[Outcome]],
    members: Sequence[int],
    decoding: str,
) -> NDArray[np.float64]:
    """Returns each member's share of range answers.

    With several outcome sets (the shot orders) an item's share is the mean
    of its shares in the sets.
    """
    return np.array(
        [
            np.mean(
                [
                    np.mean(
                        [a.form == "range" for a in _answers(set_[i], decoding)]
                    )
                    for set_ in outcome_sets
                ]
            )
            for i in members
        ],
        dtype=np.float64,
    )


def _order_groups(
    results: Results,
) -> list[tuple[str, str, str, Sequence[Sequence[Outcome]]]]:
    """Lists the outcome sets of each cell, then of each cell's orders.

    A condition and format that ran in several shot orders get an extra
    "both" entry holding all its outcome sets.
    """
    groups: list[tuple[str, str, str, Sequence[Sequence[Outcome]]]] = [
        (condition, fmt, order, [outcomes])
        for (condition, fmt, order), outcomes in results.items()
    ]
    for condition, fmt in dict.fromkeys((c, f) for c, f, _ in results):
        sets = [
            o for (c, f, _), o in results.items() if (c, f) == (condition, fmt)
        ]
        if len(sets) > 1:
            groups.append((condition, fmt, BOTH, sets))
    return groups


def hedge_gap_table(
    items: Sequence[QuantityQuestion], results: Results, *, seed: int
) -> tuple[list[str], list[list[Any]]]:
    """Builds the gap in range answers between range and exact items.

    The gap is the mean over range items of the share of range answers minus
    the same mean over exact items, per family and pooled, condition, format,
    shot order and decoding. The interval resamples items within each kind.
    When a condition and format ran in several shot orders, an extra "both"
    row uses for each item the mean of its range share over the orders.

    Args:
        items: The questions.
        results: The outcomes per cell, aligned with items.
        seed: Seed of the bootstrap.

    Returns:
        The column names and the rows; a family without items of both kinds
        has no row.
    """
    columns = [
        "family",
        "condition",
        "format",
        "shot_order",
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
        for condition, fmt, order, outcome_sets in _order_groups(results):
            for decoding in DECODINGS:
                range_rates = _item_rates(outcome_sets, on_range, decoding)
                exact_rates = _item_rates(outcome_sets, on_exact, decoding)
                gap_lo, gap_hi = bootstrap_gap_ci(
                    range_rates, exact_rates, seed=seed
                )
                rows.append(
                    [
                        family,
                        condition,
                        fmt,
                        order,
                        decoding,
                        len(on_range),
                        len(on_exact),
                        float(range_rates.mean()),
                        float(exact_rates.mean()),
                        float(range_rates.mean() - exact_rates.mean()),
                        gap_lo,
                        gap_hi,
                    ]
                )
    return columns, rows


def diagnostic_table(
    items: Sequence[QuantityQuestion], results: Results, *, seed: int
) -> tuple[list[str], list[list[Any]]]:
    """Builds the hedge rate of people against breeds and buildings.

    The hedge rate is the mean over items of the share of range answers, as
    in `hedge_gap_table`. It is taken on three groups: the people
    (`person_height`, all exact), the range items of the height family
    (breeds, living things) and its exact items (buildings, artifacts). The
    rows are per condition, format, shot order (each and "both") and
    decoding, with the rate interval of `build_table` for each group: Wilson
    for one greedy answer per item, an item bootstrap otherwise (also the
    greedy "both" rows, whose two answers per item are not independent),
    and the item Wilson interval at a rate of 0 or 1. The two differences,
    people minus breeds and people minus buildings, get the bootstrap
    interval of `bootstrap_gap_ci`. People heights are also less familiar to
    a model than breed heights, so a difference mixes kind with knowledge:
    the comparison is a diagnostic, not a clean contrast.

    Args:
        items: The questions.
        results: The outcomes per cell, aligned with items.
        seed: Seed of the bootstraps.

    Returns:
        The column names and the rows; no rows without people or without
        range and exact items of the height family.
    """
    names = ("people", "breeds", "buildings")
    differences = ("people_minus_breeds", "people_minus_buildings")
    columns = [
        "condition",
        "format",
        "shot_order",
        "decoding",
        *(f"n_{name}" for name in names),
        *(
            column
            for name in names
            for column in (*_rate_columns(name), f"{name}_interval")
        ),
        *(
            column
            for name in differences
            for column in (name, f"{name}_lo", f"{name}_hi")
        ),
    ]
    members = (
        [i for i, item in enumerate(items) if item.family == PEOPLE_FAMILY],
        _members(items, _HEIGHT, "range"),
        _members(items, _HEIGHT, "exact"),
    )
    rows: list[list[Any]] = []
    if not all(members):
        return columns, rows
    for condition, fmt, order, outcome_sets in _order_groups(results):
        for decoding in DECODINGS:
            single_greedy = decoding == "greedy" and len(outcome_sets) == 1
            rates = [_item_rates(outcome_sets, m, decoding) for m in members]
            row: list[Any] = [condition, fmt, order, decoding]
            row += [len(m) for m in members]
            for m in members:
                per_item = [
                    _counts(
                        [
                            a
                            for outcomes in outcome_sets
                            for a in _answers(outcomes[i], decoding)
                        ]
                    )
                    for i in m
                ]
                method, _, cells = _rate_cells(
                    per_item,
                    ("range",),
                    "greedy" if single_greedy else "sampled",
                    seed,
                )
                row += [sum(c["range"] for c in per_item), *cells["range"]]
                row.append(method)
            for other in rates[1:]:
                low, high = bootstrap_gap_ci(rates[0], other, seed=seed)
                row += [float(rates[0].mean() - other.mean()), low, high]
            rows.append(row)
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
        results: The outcomes per cell (condition, format, order), aligned
            with items.
        rel_tol: Relative tolerance of the band.

    Returns:
        The column names and the rows, one per family (and pooled), kind and
        format; no rows without the forced condition.
    """
    columns = [
        "family",
        "kind",
        "format",
        "shot_order",
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
            for (condition, fmt, order), outcomes in results.items():
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
                        order,
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
    items: Sequence[QuantityQuestion], results: Results, *, seed: int
) -> tuple[list[str], list[list[Any]]]:
    """Builds range and correct rates by the width tertile of range items.

    The width of a range item is `(high - low) / midpoint`. The range items
    of a family are ordered by width (ties by uid) and cut into three equal
    groups by rank; the pooled "all" rows cut all range items together with
    their own tertiles. With fewer than three items the empty groups have no
    row. Intervals are Wilson for the greedy decoding and an item bootstrap
    for the sampled one, as in `build_table`.

    Args:
        items: The questions.
        results: The outcomes per cell, aligned with items.
        seed: Seed of the bootstrap.

    Returns:
        The column names and the rows, one per family (and pooled), cell,
        decoding and non-empty tertile.
    """
    columns = [
        "family",
        "condition",
        "format",
        "shot_order",
        "decoding",
        "interval",
        "item_wilson",
        "tertile",
        "width_lo",
        "width_hi",
        "n_items",
        "n",
        "n_gradable",
        *_rate_columns("range"),
        *_rate_columns("correct"),
    ]

    def width(i: int) -> float:
        item = items[i]
        return (item.high - item.low) / ((item.high + item.low) / 2)

    rows: list[list[Any]] = []
    for family in _families_of(items):
        members = _members(items, family, "range")
        ordered = sorted(members, key=lambda i: (width(i), items[i].uid))
        tertiles: list[list[int]] = [[], [], []]
        for rank, i in enumerate(ordered):
            tertiles[rank * 3 // len(ordered)].append(i)
        for (condition, fmt, order), outcomes in results.items():
            for decoding in DECODINGS:
                for k, group in enumerate(tertiles, start=1):
                    if not group:
                        continue
                    per_item = [
                        _counts(_answers(outcomes[i], decoding)) for i in group
                    ]
                    method, wilson, cells = _rate_cells(
                        per_item, ("range", "correct"), decoding, seed
                    )
                    rows.append(
                        [
                            family,
                            condition,
                            fmt,
                            order,
                            decoding,
                            method,
                            ";".join(wilson),
                            k,
                            width(group[0]),
                            width(group[-1]),
                            len(group),
                            sum(c["n"] for c in per_item),
                            sum(c["n_gradable"] for c in per_item),
                            sum(c["range"] for c in per_item),
                            *cells["range"],
                            sum(c["correct"] for c in per_item),
                            *cells["correct"],
                        ]
                    )
    return columns, rows


def _csv_cell(value: object) -> object:
    return "" if value is None else value


def _write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    encoding: str = "utf-8",
) -> None:
    """Writes a CSV with unix line ends; None becomes an empty cell."""
    with path.open("w", encoding=encoding, newline="") as handle:
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


def _stem(condition: str, fmt: str, order: str) -> str:
    """Names a cell in a column: the shot order is left out for chat."""
    parts = [condition, fmt] if order == NO_ORDER else [condition, fmt, order]
    return "_".join(parts)


def _shown(text: str) -> str:
    """Shows the newlines of a completion as a backslash and n."""
    return text.replace("\n", "\\n")


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
        results: The outcomes per cell (condition, format, order), aligned
            with items.
        n: Largest number of rows.
        seed: Seed of the draw.

    Returns:
        The column names and the rows: the item with its Wikidata URL and
        gold range, for each cell the greedy completion (newlines shown as
        a backslash and n), its form, value, grade and unit flag, and empty
        columns for the verdict of the person. A cell is named by its
        condition, format and shot order (no order for chat).
    """
    strata: dict[tuple[str, str], list[int]] = {}
    for i, item in enumerate(items):
        strata.setdefault((item.family, item.kind), []).append(i)
    chosen = _stratified(strata, n, np.random.default_rng(seed))
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
    for condition, fmt, order in results:
        stem = _stem(condition, fmt, order)
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
                _shown(greedy.text),
                greedy.form,
                _value_text(greedy),
                _grade_text(greedy),
                "mismatch" if greedy.unit_mismatch else "",
            ]
        rows.append([*row, "", "", ""])
    return columns, rows


def _stratified[T](
    strata: dict[tuple[str, str], list[T]], n: int, rng: np.random.Generator
) -> list[T]:
    """Draws up to n members, one stratum at a time, in seeded random order."""
    queues = [
        [strata[key][j] for j in rng.permutation(len(strata[key]))]
        for key in sorted(strata)
    ]
    chosen: list[T] = []
    while len(chosen) < n and any(queues):
        for queue in queues:
            if queue and len(chosen) < n:
                chosen.append(queue.pop())
    return chosen


def hand_check_samples_table(
    items: Sequence[QuantityQuestion],
    results: Results,
    *,
    n: int,
    seed: int,
) -> tuple[list[str], list[list[Any]]]:
    """Builds the sampled completions a person checks, one per row.

    The draw is stratified by family and kind like `hand_check_table`. About
    a quarter of the rows (`n // 4`) go to completions that are most likely
    misread: no number (form "none"), an abstention, or a unit flag. They are
    taken from what exists and the other rows come from the remaining
    completions; when those run out, flagged ones fill the rest. The rows are
    ordered by item, cell and sample index.

    Args:
        items: The questions.
        results: The outcomes per cell, aligned with items.
        n: Largest number of rows.
        seed: Seed of the draw.

    Returns:
        The column names and the rows: the item with its Wikidata URL and
        gold range, the cell, the sample index, the completion (newlines
        shown as a backslash and n), its form, value, grade and unit flag,
        and empty columns for the verdict of the person.
    """
    cells = list(results)
    flagged: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
    plain: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
    for i, item in enumerate(items):
        for c, cell in enumerate(cells):
            for k, answer in enumerate(results[cell][i].samples):
                pool = (
                    flagged
                    if answer.form in ("none", "abstain")
                    or answer.unit_mismatch
                    else plain
                )
                pool.setdefault((item.family, item.kind), []).append((i, c, k))
    rng = np.random.default_rng(seed)
    chosen = _stratified(flagged, n // 4, rng)
    chosen += _stratified(plain, n - len(chosen), rng)
    if len(chosen) < n:
        left = {
            key: [m for m in members if m not in chosen]
            for key, members in flagged.items()
        }
        chosen += _stratified(left, n - len(chosen), rng)
    columns = [
        "uid",
        "wikidata_url",
        "family",
        "kind",
        "question",
        "gold_low",
        "gold_high",
        "unit",
        "condition",
        "format",
        "shot_order",
        "sample",
        "completion",
        "form",
        "value",
        "grade",
        "unit_flag",
        "human_form",
        "human_correct",
        "human_unit_mismatch",
    ]
    rows: list[list[Any]] = []
    for i, c, k in sorted(chosen):
        item = items[i]
        condition, fmt, order = cells[c]
        answer = results[cells[c]][i].samples[k]
        rows.append(
            [
                item.uid,
                f"https://www.wikidata.org/wiki/{item.entity_id}",
                item.family,
                item.kind,
                item.question,
                item.low,
                item.high,
                item.unit,
                condition,
                fmt,
                order,
                k,
                _shown(answer.text),
                answer.form,
                _value_text(answer),
                _grade_text(answer),
                "mismatch" if answer.unit_mismatch else "",
                "",
                "",
                "",
            ]
        )
    return columns, rows


def _item_records(
    items: Sequence[QuantityQuestion], results: Results
) -> list[dict[str, Any]]:
    """Lists each item with every parsed answer, for the reading study."""
    return [
        {
            **asdict(item),
            "answers": {
                f"{condition}/{fmt}/{order}": {
                    "greedy": asdict(outcomes[i].greedy),
                    "samples": [asdict(a) for a in outcomes[i].samples],
                }
                for (condition, fmt, order), outcomes in results.items()
            },
        }
        for i, item in enumerate(items)
    ]


def _summary(
    config: Config,
    items: Sequence[QuantityQuestion],
    formats: Sequence[str],
    n_hand_check: int,
    n_hand_check_samples: int,
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
        "shot_orders": list(config.shot_orders) if "fewshot" in formats else [],
        "hand_check_rows": n_hand_check,
        "hand_check_sample_rows": n_hand_check_samples,
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
    cell: Cell,
    chat: ChatFormat | None,
) -> list[Outcome]:
    """Gets a greedy and sampled completions of every item in one cell."""
    condition, fmt, order = cell
    prompts = [
        build_prompt(i.question, condition, fmt, chat, order) for i in items
    ]
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
        seed=derive_seed(config.seed, *cell),
        batch_size=config.batch_size,
    )

    def read(item: QuantityQuestion, completion: Completion) -> Answer:
        return score_answer(
            item,
            completion.text,
            fmt,
            rel_tol=config.rel_tol,
            finished=completion.finished,
        )

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
    """Asks every question in every cell and writes the files.

    A cell is a condition, a format and, for the few-shot format, a shot
    order; the chat format has no shot order.

    Writes `table.csv`, `hedge_gap.csv`, `forced_in_range.csv`, `width.csv`,
    `diagnostic.csv` (only when `config.people` is set and the items hold
    people and the height family), `hand_check.csv`,
    `hand_check_samples.csv`, `items.json` (every parsed answer; the export
    step leaves it out) and `summary.json`.

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
    cells: list[Cell] = [
        (condition, fmt, order)
        for condition in config.conditions
        for fmt in formats
        for order in (config.shot_orders if fmt == "fewshot" else (NO_ORDER,))
    ]
    results: Results = {
        cell: _ask(config, loaded, items, cell, chat) for cell in cells
    }
    tables = {
        "table": build_table(items, results, seed=config.seed),
        "hedge_gap": hedge_gap_table(items, results, seed=config.seed),
        "forced_in_range": forced_in_range_table(
            items, results, rel_tol=config.rel_tol
        ),
        "width": width_table(items, results, seed=config.seed),
    }
    diagnostic = (
        diagnostic_table(items, results, seed=config.seed)
        if config.people
        else ([], [])
    )
    if diagnostic[1]:
        tables["diagnostic"] = diagnostic
    for name, (columns, rows) in tables.items():
        _write_csv(run_dir.path / f"{name}.csv", columns, rows)
    columns, rows = hand_check_table(
        items, results, n=config.n_hand_check, seed=config.seed
    )
    _write_csv(
        run_dir.path / "hand_check.csv", columns, rows, encoding="utf-8-sig"
    )
    sample_columns, sample_rows = hand_check_samples_table(
        items, results, n=config.n_hand_check, seed=config.seed
    )
    _write_csv(
        run_dir.path / "hand_check_samples.csv",
        sample_columns,
        sample_rows,
        encoding="utf-8-sig",
    )
    run_dir.write_json("items", _item_records(items, results))
    summary = _summary(config, items, formats, len(rows), len(sample_rows))
    if diagnostic[1]:
        summary["diagnostic"] = [
            dict(zip(diagnostic[0], row, strict=True)) for row in diagnostic[1]
        ]
    run_dir.write_json("summary", summary)


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
    if config.people:
        items = [*items, *load_people_heights(offline=config.offline)]
    execute(config, run_dir, load_model(config.model), items)


def panel_orders(rows: Sequence[dict[str, str]]) -> tuple[str, list[str]]:
    """Picks the format and the shot orders the plot shows.

    Args:
        rows: The rows of `table.csv`.

    Returns:
        The few-shot format when the run has it, else chat, and the shot
        orders of that format in the order of the table.
    """
    fmt = "fewshot" if any(r["format"] == "fewshot" for r in rows) else "chat"
    orders: list[str] = []
    for row in rows:
        if row["format"] == fmt and row["shot_order"] not in orders:
            orders.append(row["shot_order"])
    return fmt, orders


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the share of each answer form by kind and condition.

    One panel per shot order shows the pooled greedy answers of the few-shot
    format (one chat panel when the run has no few-shot cells), with Wilson
    95 percent intervals.

    Args:
        run_dir: A finished P10a run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths.
    """
    with (run_dir.path / "table.csv").open(encoding="utf-8", newline="") as f:
        table = list(csv.DictReader(f))
    fmt, orders = panel_orders(table)
    rows = [
        r for r in table if r["family"] == POOLED and r["decoding"] == "greedy"
    ]
    forms = ("point", "range", "abstain", "none")
    colors = categorical_colors(len(forms))
    with style_context():
        fig, axes = new_figure(
            1, len(orders), width=COLUMN_WIDTH * len(orders), sharey=True
        )
        for ax, order in zip(axes[0], orders, strict=True):
            groups = [
                r
                for condition in CONDITIONS
                for kind in _KINDS[::-1]
                for r in rows
                if (r["condition"], r["kind"], r["format"], r["shot_order"])
                == (condition, kind, fmt, order)
            ]
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
            ax.set_title(f"{fmt}, {order}" if order != NO_ORDER else fmt)
        axes[0, 0].set_ylabel("Share of greedy answers")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            ncol=len(forms),
            loc="outside upper center",
            frameon=False,
        )
    return save_figure(fig, out_dir, "form_shares")
