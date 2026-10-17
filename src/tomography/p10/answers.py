"""Reading a quantity answer: a point, a range, an abstention or nothing.

A completion is cut to its first non-empty line and the first quantity of that
line decides what it is. Abstention reuses the P9 phrase patterns, so "does
not know" means the same thing in both studies. Grading compares the parsed
value or range with a gold interval widened by a relative tolerance.
"""

from dataclasses import dataclass
import re
from typing import Literal

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
    "twenty": 20,
}
TENS_WORDS: dict[str, int] = {
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_TENS_AND_TWENTY = {**TENS_WORDS, "twenty": 20}
_SINGLE_DIGIT_WORDS = {w: v for w, v in UNIT_WORDS.items() if 1 <= v <= 9}

_DIGITS = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_TENS_PATTERN = "|".join(_TENS_AND_TWENTY)
_UNIT_PATTERN = "|".join(_SINGLE_DIGIT_WORDS)
_HUNDRED_PATTERN = "|".join(["a", *_SINGLE_DIGIT_WORDS])
_NUMBER = re.compile(
    rf"(?P<digits>{_DIGITS})"
    rf"|(?<![\w])(?:(?P<mult>{_HUNDRED_PATTERN})\s+)?(?P<hundred>hundred)(?!\w)"
    rf"|(?<![\w])(?P<tens>{_TENS_PATTERN})(?:-(?P<unit>{_UNIT_PATTERN}))?(?!\w)"
    rf"|(?<![\w])(?P<word>{'|'.join(UNIT_WORDS)})(?!\w)",
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
_HABIT = re.compile(
    r"(?<!\w)(?:"
    + "|".join(re.escape(m).replace(r"\ ", r"\s+") for m in HABIT_MARKERS)
    + r")(?!\w)",
    re.IGNORECASE,
)

_UNIT = r"(?:[^\W\d_]+\.?(?:\s+[^\W\d_]+\.?)?)?"
_JOINER = re.compile(
    rf"\s*{_UNIT}\s*"
    r"(?:(?P<pm>\+/-|\+-|\u00b1)|(?P<join>-|\u2013|\u2014|to|or)"
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
        approximate: Whether the answer marks itself as approximate.
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


def _value(match: re.Match[str]) -> float:
    """Converts a number match to its value.

    Args:
        match: A match of `_NUMBER`.

    Returns:
        The value.
    """
    if match["digits"] is not None:
        return float(match["digits"].replace(",", ""))
    if match["hundred"] is not None:
        factor = match["mult"].lower() if match["mult"] else "a"
        return 100.0 * _SINGLE_DIGIT_WORDS.get(factor, 1)
    if match["tens"] is not None:
        tens = _TENS_AND_TWENTY[match["tens"].lower()]
        unit = (
            _SINGLE_DIGIT_WORDS[match["unit"].lower()] if match["unit"] else 0
        )
        return float(tens + unit)
    return float(UNIT_WORDS[match["word"].lower()])


def _joins(before: str, joiner: re.Match[str]) -> bool:
    """Tells whether a joiner may link two numbers into a range.

    Args:
        before: The text before the first number.
        joiner: A match of `_JOINER`.

    Returns:
        False for "and" unless "between" opens the range.
    """
    return not joiner["and"] or bool(_BETWEEN.search(before))


def parse_answer(completion: str) -> Parsed:
    """Parses a completion into a point, a range, an abstention or nothing.

    Args:
        completion: The generated text.

    Returns:
        The parse of its first non-empty line.
    """
    text = answer_text(completion)
    first = _NUMBER.search(text)
    if first is None:
        form: Form = "abstain" if is_abstention(completion) else "none"
        return Parsed(form, None, None, False)
    value = _value(first)
    low, high = value, value
    joined = _JOINER.match(text, first.end())
    second = _NUMBER.match(text, joined.end()) if joined else None
    if joined and second and _joins(text[: first.start()], joined):
        other = _value(second)
        if joined["pm"]:
            low, high = value - other, value + other
        else:
            low, high = min(value, other), max(value, other)
    approximate = bool(
        _LEADING.search(text[: first.start()]) or _HABIT.search(text)
    )
    form: Form = "point" if low == high else "range"
    return Parsed(form, low, high, approximate)


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
