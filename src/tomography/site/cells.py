"""Cell text, CSV text and file readers shared by the tables and figures.

Missing values are written `n/a`. Every number is rounded where it is
written, never earlier, then signed: a value that rounds to zero has no sign
and a negative one uses the true minus sign.
"""

from collections.abc import Sequence
import csv
import io
import json
from pathlib import Path
from typing import Any

from tomography.viz import axes

MISSING = "n/a"
MINUS = axes.MINUS
PLUS_MINUS = chr(0x00B1)


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
    text = f"{abs(value):.{digits}f}"
    if value < 0 and float(text) != 0.0:
        return MINUS + text
    return text


def pm(value: float | None, half: float | None, digits: int = 3) -> str:
    """Writes a value with a symmetric half-width.

    Args:
        value: The estimate, or None when it is not stored.
        half: The half-width of the interval.
        digits: Decimals.

    Returns:
        Text such as `0.109 ± 0.020`, or `n/a` without a value.
    """
    if value is None or half is None:
        return MISSING
    return f"{num(value, digits)} {PLUS_MINUS} {num(abs(half), digits)}"


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


def csv_text(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Writes rows as CSV text; None becomes an empty field.

    Args:
        header: Column names.
        rows: One sequence of cells per row.

    Returns:
        The CSV document.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(["" if cell is None else cell for cell in row])
    return buffer.getvalue()
