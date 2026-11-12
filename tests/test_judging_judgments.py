"""Tests for reading the judgments an external judge returns."""

import json

import numpy as np
import pytest

from tomography.judging.batches import Batch
from tomography.judging.judgments import JudgmentError
from tomography.judging.judgments import parse_judgments

BATCH = Batch(
    batch_id="0a1b2c3d",
    judge=1,
    item_ids=("aa", "bb", "cc"),
    keys=("k0", "k1", "k2"),
)
BASE = ("concept", "fluency")
POST = ("concept", "instruction", "fluency")


def _lines(*rows: object) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def _row(item_id: str, concept: object = 1, fluency: object = 2) -> dict:
    return {"item_id": item_id, "concept": concept, "fluency": fluency}


GOOD = _lines(_row("aa", 0, 1), _row("bb", 1, 2), _row("cc", 2, 0))


def test_scores_come_back_in_batch_order_and_criteria_order() -> None:
    shuffled = _lines(_row("cc", 2, 0), _row("aa", 0, 1), _row("bb", 1, 2))

    scores = parse_judgments(shuffled, BATCH, BASE)

    assert scores.dtype == np.int64
    assert scores.tolist() == [[0, 1], [1, 2], [2, 0]]
    assert parse_judgments(GOOD, BATCH, BASE).tolist() == scores.tolist()


def test_a_post_trained_batch_reads_the_instruction_score() -> None:
    text = _lines(
        {"item_id": "aa", "fluency": 2, "instruction": 1, "concept": 0},
        {"item_id": "bb", "fluency": 0, "instruction": 2, "concept": 1},
        {"item_id": "cc", "fluency": 1, "instruction": 0, "concept": 2},
    )

    scores = parse_judgments(text, BATCH, POST)

    assert scores.tolist() == [[0, 1, 2], [1, 2, 0], [2, 0, 1]]


def test_blank_lines_and_windows_line_ends_are_ignored() -> None:
    text = "\r\n" + GOOD.replace("\n", "\r\n\r\n")

    assert parse_judgments(text, BATCH, BASE).tolist() == [
        [0, 1],
        [1, 2],
        [2, 0],
    ]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (_lines(_row("aa"), _row("bb")), "missing 1 item: cc"),
        (_lines(_row("aa")), "missing 2 items: bb, cc"),
        (GOOD + _lines(_row("bb")), "line 4: duplicate item 'bb'"),
        (GOOD + _lines(_row("zz")), "line 4: unknown item 'zz'"),
        (
            _lines(_row("aa", 3), _row("bb"), _row("cc")),
            "line 1: concept must be 0, 1 or 2, got 3",
        ),
        (
            _lines(_row("aa", -1), _row("bb"), _row("cc")),
            "line 1: concept must be 0, 1 or 2, got -1",
        ),
        (
            _lines(_row("aa"), _row("bb", 1, 2.0), _row("cc")),
            "line 2: fluency must be 0, 1 or 2, got 2.0",
        ),
        (
            _lines(_row("aa"), _row("bb", "2"), _row("cc")),
            "line 2: concept must be 0, 1 or 2, got '2'",
        ),
        (
            _lines(_row("aa"), _row("bb"), _row("cc", True)),
            "line 3: concept must be 0, 1 or 2, got True",
        ),
        (
            _lines(_row("aa"), _row("bb"), {"item_id": "cc", "concept": 1}),
            "line 3: missing fields: fluency",
        ),
        (
            _lines(
                _row("aa"),
                _row("bb"),
                {**_row("cc"), "instruction": 1},
            ),
            "line 3: unexpected fields: instruction",
        ),
        (
            _lines(_row("aa"), _row("bb"), {**_row("cc"), "note": "fine"}),
            "line 3: unexpected fields: note",
        ),
        (
            _lines(_row("aa"), _row("bb"), {"concept": 1, "fluency": 1}),
            "line 3: missing fields: item_id",
        ),
        (
            _lines(_row("aa"), _row("bb"), _row(7)),  # type: ignore[arg-type]
            "line 3: item_id must be text, got 7",
        ),
        (_lines(_row("aa"), _row("bb")) + "{not json\n", "line 3: not JSON"),
        (
            _lines(_row("aa"), _row("bb"), [1, 2]),
            "line 3: not a JSON object",
        ),
        ("", "missing 3 items: aa, bb, cc"),
    ],
    ids=[
        "one_missing",
        "two_missing",
        "duplicate",
        "unknown_id",
        "above_range",
        "below_range",
        "float_score",
        "text_score",
        "boolean_score",
        "missing_criterion",
        "wrong_criterion",
        "extra_field",
        "no_item_id",
        "numeric_item_id",
        "malformed_line",
        "array_line",
        "empty_file",
    ],
)
def test_anything_but_one_valid_judgment_per_item_is_refused(
    text: str, message: str
) -> None:
    with pytest.raises(JudgmentError) as error:
        parse_judgments(text, BATCH, BASE)

    assert message in str(error.value)
    assert "batch 0a1b2c3d" in str(error.value)


def test_a_base_batch_refuses_the_instruction_criterion_on_every_line() -> None:
    text = _lines(
        *({**_row(item_id), "instruction": 2} for item_id in BATCH.item_ids)
    )

    with pytest.raises(JudgmentError, match="unexpected fields: instruction"):
        parse_judgments(text, BATCH, BASE)


def test_every_problem_is_reported_at_once() -> None:
    text = _lines(_row("aa", 5), _row("zz"), _row("aa"))

    with pytest.raises(JudgmentError) as error:
        parse_judgments(text, BATCH, BASE)

    message = str(error.value)
    assert "line 1: concept must be 0, 1 or 2, got 5" in message
    assert "line 2: unknown item 'zz'" in message
    assert "line 3: duplicate item 'aa'" in message
    assert "missing 2 items: bb, cc" in message


def test_a_long_list_of_problems_is_cut_with_a_count() -> None:
    text = _lines(*(_row(f"x{i}") for i in range(30)))

    with pytest.raises(JudgmentError) as error:
        parse_judgments(text, BATCH, BASE)

    message = str(error.value)
    assert "line 20: unknown item 'x19'" in message
    assert "line 21" not in message
    assert "and 11 more" in message


def test_a_judgment_error_is_a_value_error() -> None:
    assert issubclass(JudgmentError, ValueError)
