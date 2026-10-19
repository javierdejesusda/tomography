"""Pure readers for the files of one run directory.

Every JSON file of a run is `{"_meta": ..., "data": ...}`. Tables are stored as
`{columns, rows}`, as a list of records or as CSV. These functions hide those
shapes, hold no state and never write.
"""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

Record = dict[str, Any]


def _number(text: str) -> float | str | None:
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def data(run: Path, name: str) -> Any:
    """Reads the `data` payload of a JSON result file.

    Args:
        run: Run directory.
        name: File name such as `summary.json`.

    Returns:
        The decoded payload.
    """
    return json.loads((run / name).read_text(encoding="utf-8"))["data"]


def summary(run: Path) -> Record:
    """Reads `summary.json` of a run.

    Args:
        run: Run directory.

    Returns:
        The summary payload.
    """
    result: Record = data(run, "summary.json")
    return result


def rows(run: Path, name: str) -> list[Record]:
    """Reads a table file as a list of records.

    Args:
        run: Run directory.
        name: A JSON table or a CSV file name.

    Returns:
        One dict per row; CSV numbers are parsed and blanks become None.
    """
    if name.endswith(".csv"):
        with (run / name).open(newline="", encoding="utf-8") as handle:
            return [
                {k: _number(v) for k, v in row.items()}
                for row in csv.DictReader(handle)
            ]
    payload = data(run, name)
    if isinstance(payload, dict):
        columns = payload["columns"]
        return [dict(zip(columns, row, strict=True)) for row in payload["rows"]]
    result: list[Record] = payload
    return result


def arrays(run: Path, name: str) -> dict[str, np.ndarray]:
    """Reads every array of an `.npz` file.

    Args:
        run: Run directory.
        name: File name such as `curves.npz`.

    Returns:
        Array name to array.
    """
    with np.load(run / name) as stored:
        return {key: stored[key] for key in stored.files}
