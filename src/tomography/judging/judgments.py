"""Reads the judgments an external judge returns for one batch.

A judgments file is JSON lines: one object per item with `item_id` and one
integer score of 0, 1 or 2 per criterion of the batch. Anything but exactly
one valid judgment per exported item is refused as a whole.
"""

from collections.abc import Sequence
import json

import numpy as np
from numpy.typing import NDArray

from tomography.judging.batches import Batch
from tomography.judging.rubric import SCORES

_MAX_LISTED = 20


class JudgmentError(ValueError):
    """A judgments file does not hold one valid judgment per item."""


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _score_problem(name: str, value: object) -> str | None:
    """Describes why a value is not a score, or returns None for a score."""
    # bool is an int in Python, and a JSON true is not a score.
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in SCORES
    ):
        return f"{name} must be 0, 1 or 2, got {value!r}"
    return None


def parse_judgments(
    text: str, batch: Batch, criteria: Sequence[str]
) -> NDArray[np.int64]:
    """Checks a judgments file against its batch and reads the scores.

    Blank lines are skipped. A line is valid when it is a JSON object whose
    fields are exactly `item_id` and the criteria, the item belongs to the
    batch and has no earlier line, and every score is an integer of 0, 1 or
    2 (not a boolean, a float or text).

    Args:
        text: The content of the judgments file.
        batch: The exported batch the judgments answer.
        criteria: The criteria of the export, in order.

    Returns:
        The scores of shape `(items, criteria)`, in the item order of the
        batch whatever the line order of the file.

    Raises:
        JudgmentError: If a line is not valid or an item of the batch has no
            line. The message names the batch and lists the first problems
            with their line numbers, then how many more there are.
    """
    expected = {"item_id", *criteria}
    rows = {item_id: row for row, item_id in enumerate(batch.item_ids)}
    scores = np.zeros((len(rows), len(criteria)), dtype=np.int64)
    seen: set[str] = set()
    problems: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"line {number}: not JSON")
            continue
        if not isinstance(entry, dict):
            problems.append(f"line {number}: not a JSON object")
            continue
        missing = sorted(expected - set(entry))
        unexpected = sorted(set(entry) - expected)
        if missing:
            problems.append(
                f"line {number}: missing fields: {', '.join(missing)}"
            )
        if unexpected:
            problems.append(
                f"line {number}: unexpected fields: {', '.join(unexpected)}"
            )
        if missing or unexpected:
            continue
        item_id = entry["item_id"]
        if not isinstance(item_id, str):
            problems.append(
                f"line {number}: item_id must be text, got {item_id!r}"
            )
            continue
        if item_id not in rows:
            problems.append(f"line {number}: unknown item {item_id!r}")
            continue
        if item_id in seen:
            problems.append(f"line {number}: duplicate item {item_id!r}")
            continue
        seen.add(item_id)
        for column, name in enumerate(criteria):
            problem = _score_problem(name, entry[name])
            if problem is None:
                scores[rows[item_id], column] = entry[name]
            else:
                problems.append(f"line {number}: {problem}")
    absent = [item_id for item_id in batch.item_ids if item_id not in seen]
    if absent:
        problems.append(
            f"missing {_plural(len(absent), 'item')}: {', '.join(absent)}"
        )
    if problems:
        listed = problems[:_MAX_LISTED]
        if len(problems) > len(listed):
            listed.append(f"and {len(problems) - len(listed)} more")
        raise JudgmentError(f"batch {batch.batch_id}: " + "; ".join(listed))
    return scores
