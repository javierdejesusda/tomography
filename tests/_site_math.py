"""Independent formulas and cell formatting for the table tests.

These re-derive the interval formulas and the rounding rules from their
definitions, so a table test never calls the site code it checks.
"""

import math

Z = 1.959963984540054
MINUS = chr(0x2212)


def fmt(value: float, digits: int = 3) -> str:
    """Rounds first, then signs: no negative zero, true minus."""
    text = f"{abs(value):.{digits}f}"
    if value < 0 and float(text) != 0.0:
        return MINUS + text
    return text


def diff(p1: float, p2: float, n1: int, n2: int | None = None) -> tuple:
    """Difference of two accuracies with the conservative interval."""
    n2 = n1 if n2 is None else n2
    half = Z * math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return p1 - p2, p1 - p2 - half, p1 - p2 + half


def wilson(p: float, n: int) -> tuple[float, float]:
    """Wilson score interval."""
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def excludes_zero(lo: float, hi: float) -> bool:
    """Whether an interval lies on one side of zero."""
    return lo > 0 or hi < 0


def groups(models: list[str]) -> tuple[int, ...]:
    """Row index of the first row of GPT-2 and of Qwen3 for these models."""
    first_qwen = next(i for i, m in enumerate(models) if m.startswith("qwen3"))
    return (0, first_qwen)
