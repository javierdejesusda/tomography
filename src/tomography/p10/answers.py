"""Reading a quantity answer: a point, a range, an abstention or nothing.

A completion is cut to its first non-empty line and that line decides
everything: the first quantity on it is the answer, and an abstention phrase
counts only when the line holds no quantity. So that the digits of a name or a
year are not read as the answer, the caller can remove strings such as the
entity label from the line, and can name unit words so that the first quantity
followed by one wins. Values are read in whatever unit the model used ("9
months" for a days question is 9); the experiment reports unit mismatches from
its hand check. Abstention reuses the P9 phrase patterns, so "does not know"
means the same thing in both studies. Grading compares the parsed value or
range with a gold interval widened by a relative tolerance.

Negative numbers are out of scope: every P10 quantity is positive, and a
leading minus sign reads as a range separator.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Literal

from tomography.p9.behavior import ABSTAIN_REGEXES
from tomography.p9.behavior import is_abstention

UNIT_WORDS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS_WORDS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
SCALE_WORDS: dict[str, int] = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1_000_000,
    "dozen": 12,
}
_WORD_VALUES: dict[str, int] = {"a": 1, **UNIT_WORDS, **TENS_WORDS}


def _alternatives(words: list[str]) -> str:
    """Joins words into a regex alternation, longest first.

    Args:
        words: The words.

    Returns:
        The alternation, without a group.
    """
    return "|".join(sorted(words, key=len, reverse=True))


_DIGIT_WORDS = _alternatives([w for w, v in UNIT_WORDS.items() if 1 <= v <= 9])
_SMALL = (
    rf"(?:(?:{_alternatives(list(TENS_WORDS))})(?:-|\s+)(?:{_DIGIT_WORDS})"
    rf"|{_alternatives(list(TENS_WORDS))}|{_alternatives(list(UNIT_WORDS))})"
    r"(?!\w)"
)
_HUNDRED = (
    rf"(?:(?:a|{_DIGIT_WORDS})\s+)?hundred(?!\w)"
    rf"(?:\s+(?:and\s+)?{_SMALL})?"
)
_THOUSAND = (
    rf"(?:(?:{_HUNDRED}|a|{_SMALL})\s+)?(?:thousand|million)(?!\w)"
    rf"(?:\s+(?:and\s+)?(?:{_HUNDRED}|{_SMALL}))?"
)
_DOZEN = rf"(?:(?:a|{_DIGIT_WORDS})\s+)?dozen(?!\w)"
_DIGITS = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_NUMBER = re.compile(
    rf"(?P<digits>{_DIGITS})(?:(?P<k>k)(?!\w)"
    rf"|\s+(?P<dscale>{_alternatives(list(SCALE_WORDS))})(?!\w))?"
    rf"|(?<!\w)(?P<words>{_THOUSAND}|{_HUNDRED}|{_DOZEN}|{_SMALL})",
    re.IGNORECASE,
)
EXTRA_ABSTAIN_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<!\w)no one (?:knows|can (?:know|say|tell))(?!\w)"),
)
_VAGUE = re.compile(
    r"(?<!\w)(?:(?:several|a\s+few|a\s+couple\s+of)\s+"
    r"(?:hundred|thousand|million)"
    r"|(?:hundreds|dozens|thousands|millions)\s+of)(?!\w)",
    re.IGNORECASE,
)


LEADING_MARKERS: tuple[str, ...] = (
    "about",
    "around",
    "approximately",
    "approx.",
    "roughly",
    "nearly",
    "almost",
    "some",
    "up to",
    "over",
    "more than",
    "less than",
    "at least",
    "under",
    "maybe",
    "perhaps",
    "probably",
    "several",
    "a few",
    "a couple of",
)
SYMBOL_MARKERS: tuple[str, ...] = ("~", "\u2248")
HABIT_MARKERS: tuple[str, ...] = (
    "typically",
    "usually",
    "on average",
    "average",
    "generally",
)
_LEADING = re.compile(
    r"(?<!\w)(?:"
    + "|".join(re.escape(m).replace(r"\ ", r"\s+") for m in LEADING_MARKERS)
    + r")(?!\w)|"
    + "|".join(re.escape(m) for m in SYMBOL_MARKERS),
    re.IGNORECASE,
)
TRAILING_MARKERS: tuple[str, ...] = ("or so", "give or take", "or thereabouts")
_TRAILING = re.compile(
    r"(?<!\w)(?:"
    + "|".join(re.escape(m).replace(r"\ ", r"\s+") for m in TRAILING_MARKERS)
    + r")(?!\w)",
    re.IGNORECASE,
)
_HABIT = re.compile(
    r"(?<!\w)(?:"
    + "|".join(re.escape(m).replace(r"\ ", r"\s+") for m in HABIT_MARKERS)
    + r")(?!\w)",
    re.IGNORECASE,
)

_UNIT = r"(?:[^\W\d_]+\.?(?:\s+[^\W\d_]+\.?)?)?"
_JOINER = re.compile(
    rf"\s*{_UNIT}\s*"
    r"(?:(?P<pm>\+/-|\+-|\u00b1)|(?P<join>-|\u2010|\u2011|\u2013|\u2014|\u2212|~|to|or)"
    r"|(?P<and>and))\s*",
    re.IGNORECASE,
)
_BETWEEN = re.compile(r"between\s*$", re.IGNORECASE)

Form = Literal["point", "range", "abstain", "none"]


@dataclass(frozen=True)
class Parsed:
    """What one completion answered.

    Attributes:
        form: "point" for one value, "range" for two, "abstain" when the
            completion says it does not know, "none" for anything else.
        low: The value, or the lower bound of a range; None without a
            quantity.
        high: The value, or the upper bound of a range; equal to `low` for a
            point; None without a quantity.
        approximate: Whether the answer marks itself as approximate. It is
            False for "abstain" and for "none" except a vague quantity
            ("several hundred", "dozens of"), which is "none" with
            `approximate` True: it has no usable value to grade.
    """

    form: Form
    low: float | None
    high: float | None
    approximate: bool


def answer_text(completion: str) -> str:
    """Returns the first non-empty line of a completion, stripped.

    Args:
        completion: The generated text.

    Returns:
        The line, or an empty string when the completion has no text.
    """
    for line in completion.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _read(match: re.Match[str]) -> tuple[float, int | None]:
    """Converts a number match to its value.

    Args:
        match: A match of `_NUMBER`.

    Returns:
        The value, and the factor of its only scale word ("six hundred",
        "3 thousand", "30k") or None when it has none or several.
    """
    if match["digits"] is not None:
        value = float(match["digits"].replace(",", ""))
        if match["k"]:
            return value * 1000, 1000
        if match["dscale"]:
            scale = SCALE_WORDS[match["dscale"].lower()]
            return value * scale, scale
        return value, None
    words = [
        w for w in re.split(r"[-\s]+", match["words"].lower()) if w != "and"
    ]
    total = current = 0.0
    for word in words:
        if word in ("thousand", "million"):
            total += (current or 1) * SCALE_WORDS[word]
            current = 0.0
        elif word in SCALE_WORDS:
            current = (current or 1) * SCALE_WORDS[word]
        else:
            current += _WORD_VALUES[word]
    scales = [w for w in words if w in SCALE_WORDS]
    ends_in_scale = words[-1] in SCALE_WORDS
    only_scale = SCALE_WORDS[scales[0]] if len(scales) == 1 else None
    return total + current, only_scale if ends_in_scale else None


def _quantity(
    text: str, first: re.Match[str]
) -> tuple[float, float, int] | None:
    """Reads the point or range that starts at a number.

    Args:
        text: The answer line.
        first: The match of `_NUMBER` that opens the quantity.

    Returns:
        The lower and upper bounds, equal for a point, and the index where
        the quantity ends; None when `first` is a bare "one" that does not
        open a range.
    """
    value, factor = _read(first)
    joined = _JOINER.match(text, first.end())
    second = _NUMBER.match(text, joined.end()) if joined else None
    if joined and second:
        between = _BETWEEN.search(text[: first.start()])
        if not joined["and"] or between:
            other, other_factor = _read(second)
            if joined["pm"]:
                return value - other, value + other, second.end()
            if factor is None and other_factor is not None:
                value *= other_factor
            return min(value, other), max(value, other), second.end()
    if first["words"] and first["words"].lower() == "one":
        return None
    return value, value, first.end()


def _without(text: str, ignore: Sequence[str]) -> str:
    """Replaces every ignored string in a text with a space.

    Args:
        text: The answer line.
        ignore: The strings to remove, matched case-insensitively; empty
            strings are skipped.

    Returns:
        The text without them.
    """
    labels = [re.escape(label) for label in ignore if label]
    if not labels:
        return text
    return re.sub(_alternatives(labels), " ", text, flags=re.IGNORECASE)


def _unit_pattern(unit_words: Sequence[str]) -> re.Pattern[str] | None:
    """Compiles a pattern for a unit word right after a quantity.

    Args:
        unit_words: The unit words; empty strings are skipped.

    Returns:
        The pattern, or None when there is no unit word.
    """
    units = [re.escape(unit) for unit in unit_words if unit]
    if not units:
        return None
    return re.compile(rf"\s*(?:{_alternatives(units)})(?!\w)", re.IGNORECASE)


def parse_answer(
    completion: str,
    *,
    ignore: Sequence[str] = (),
    unit_words: Sequence[str] = (),
) -> Parsed:
    """Parses a completion into a point, a range, an abstention or nothing.

    A vague quantity such as "several hundred" has no usable value, so it is
    "none" but still approximate: the one case where `approximate` is True
    without a value.

    Args:
        completion: The generated text.
        ignore: Strings removed from the answer line, case-insensitively,
            before any number is read, such as the entity label "Boeing
            737-800" whose digits are not the answer.
        unit_words: Unit words such as "m" or "minutes". When given, the
            answer is the first quantity followed by one of them ("35.8 m",
            "35.8m", "160 to 188 cm"), and the first quantity when none is.

    Returns:
        The parse of its first non-empty line.
    """
    text = _without(answer_text(completion), ignore)
    unit = _unit_pattern(unit_words)
    vague = _VAGUE.search(text)
    chosen: tuple[re.Match[str], tuple[float, float, int]] | None = None
    for first in _NUMBER.finditer(text):
        if vague and vague.start() <= first.start():
            break
        quantity = _quantity(text, first)
        if quantity is None:
            continue
        chosen = chosen or (first, quantity)
        if unit is None:
            break
        if unit.match(text, quantity[2]) or unit.match(text, first.end()):
            chosen = first, quantity
            break
    if chosen is not None:
        first, (low, high, _) = chosen
        approximate = bool(
            _LEADING.search(text[: first.start()])
            or _HABIT.search(text)
            or _TRAILING.search(text[first.end() :])
        )
        form: Form = "point" if low == high else "range"
        return Parsed(form, low, high, approximate)
    if vague:
        return Parsed("none", None, None, True)
    abstained = is_abstention(text, (*ABSTAIN_REGEXES, *EXTRA_ABSTAIN_REGEXES))
    return Parsed("abstain" if abstained else "none", None, None, False)


def grade(
    parsed: Parsed, low: float, high: float, *, rel_tol: float = 0.05
) -> bool | None:
    """Grades a parsed answer against a gold interval.

    The interval is widened to `[low * (1 - rel_tol), high * (1 + rel_tol)]`.
    A point is correct when it lies inside the widened interval and a range
    is correct when it overlaps it.

    Args:
        parsed: The parsed answer.
        low: The lower bound of the gold answer.
        high: The upper bound of the gold answer; equal to `low` for an
            exact answer.
        rel_tol: The relative tolerance on each side.

    Returns:
        Whether the answer is correct; None for an abstention or an answer
        without a quantity, which are not graded.

    Raises:
        ValueError: If `low` is negative, `low` exceeds `high`, or `rel_tol`
            is negative.
    """
    if low < 0 or low > high:
        raise ValueError(
            f"gold must satisfy 0 <= low <= high, got {low}, {high}"
        )
    if rel_tol < 0:
        raise ValueError(f"rel_tol must be non-negative, got {rel_tol}")
    if parsed.low is None or parsed.high is None:
        return None
    return parsed.low <= high * (1 + rel_tol) and parsed.high >= low * (
        1 - rel_tol
    )
