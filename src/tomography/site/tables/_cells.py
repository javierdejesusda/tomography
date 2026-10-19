"""Cell text and file readers shared by Tables 4 to 8.

Missing values are written `n/a`, minus signs are plain hyphens and every
number is rounded where it is written, never earlier.
"""

import csv
import json
from pathlib import Path
from typing import Any

MISSING = "n/a"


def num(value: float | None, digits: int = 3) -> str:
    """Writes a number with a fixed count of decimals.

    Args:
        value: The number, or None when it is not stored.
        digits: Decimals.

    Returns:
        The text, or `n/a` for None.
    """
    if value is None:
        return MISSING
    return f"{value:.{digits}f}"


def interval(
    value: float | None,
    lo: float | None,
    hi: float | None,
    digits: int = 3,
) -> str:
    """Writes a value with its interval in brackets.

    Args:
        value: The estimate.
        lo: Lower end, or None when there is no interval.
        hi: Upper end, or None when there is no interval.
        digits: Decimals.

    Returns:
        Text such as `0.914 [0.895, 0.931]`, the bare value without an
        interval, or `n/a` without a value.
    """
    if value is None:
        return MISSING
    if lo is None or hi is None:
        return num(value, digits)
    return f"{num(value, digits)} [{num(lo, digits)}, {num(hi, digits)}]"


def percent(share: float | None) -> str:
    """Writes a share as a whole percent without the sign.

    Args:
        share: A fraction between 0 and 1, or None.

    Returns:
        The rounded percent, or `n/a` for None.
    """
    if share is None:
        return MISSING
    return f"{round(100 * share)}"


def count(part: int, total: int) -> str:
    """Writes `part / total` with thousands separators.

    Args:
        part: The numerator.
        total: The denominator.

    Returns:
        Text such as `1,377 / 21,988`.
    """
    return f"{part:,} / {total:,}"


def _cell(text: str) -> float | str | None:
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def read_csv(path: Path) -> list[dict[str, Any]]:
    """Reads a result CSV; numbers are parsed and blanks become None.

    Args:
        path: The file.

    Returns:
        One dict per row.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {k: _cell(v) for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]


def read_table(path: Path) -> list[dict[str, Any]]:
    """Reads a `table.json`-style result file as records.

    Args:
        path: The file, stored as `{columns, rows}` or as a record list.

    Returns:
        One dict per row.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))["data"]
    if isinstance(payload, dict):
        return [
            dict(zip(payload["columns"], row, strict=True))
            for row in payload["rows"]
        ]
    records: list[dict[str, Any]] = payload
    return records


def check_align(align: str, columns: int) -> str:
    """Validates the alignment string of a table.

    Args:
        align: One `l` or `r` per column.
        columns: Number of columns.

    Returns:
        The alignment string.

    Raises:
        ValueError: If it has a wrong length or a letter other than l or r.
    """
    if len(align) != columns or set(align) - {"l", "r"}:
        raise ValueError(f"bad alignment {align!r} for {columns} columns")
    return align
