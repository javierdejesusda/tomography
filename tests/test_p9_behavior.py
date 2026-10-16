"""Tests for the abstention detector, the behavior rates and the shift CI."""

import csv
import math
from pathlib import Path
import re

import numpy as np
import pytest

from tomography.control.generate import Completion
from tomography.p9.behavior import ABSTAIN_PATTERNS
from tomography.p9.behavior import ABSTAIN_REGEXES
from tomography.p9.behavior import CHECK_COLUMNS
from tomography.p9.behavior import BehaviorRates
from tomography.p9.behavior import CompletionRecord
from tomography.p9.behavior import Outcome
from tomography.p9.behavior import abs_slope_gap_ci
from tomography.p9.behavior import behavior_rates
from tomography.p9.behavior import classify
from tomography.p9.behavior import is_abstention
from tomography.p9.behavior import item_slopes
from tomography.p9.behavior import select_coefficient
from tomography.p9.behavior import shift_ci
from tomography.p9.behavior import slope_ci
from tomography.p9.behavior import write_completion_check


def _outcomes(answered_correct: int, answered_wrong: int, abstained: int):
    return [
        *[Outcome(False, True)] * answered_correct,
        *[Outcome(False, False)] * answered_wrong,
        *[Outcome(True, False)] * abstained,
    ]


def _rates(balanced: float) -> BehaviorRates:
    return BehaviorRates(10, 10, 0.0, balanced, 1.0, 0.0, balanced)


def _record(
    uid: str, abstained: bool, *, coeff: float = 0.0, text: str = "x"
) -> CompletionRecord:
    return CompletionRecord(
        uid, "real", coeff, f"Q {uid}?", text, abstained, not abstained
    )


def test_the_eleven_patterns_are_compiled_in_order() -> None:
    assert len(ABSTAIN_PATTERNS) == 11
    assert len(ABSTAIN_REGEXES) == 11
    assert all(isinstance(r, re.Pattern) for r in ABSTAIN_REGEXES)
    assert ABSTAIN_PATTERNS[0] == "i (?:do not|don't|dont) know"
    assert ABSTAIN_PATTERNS[7] == r"unknown(?! soldiers?\b)"


def test_the_check_columns() -> None:
    assert CHECK_COLUMNS == (
        "uid",
        "kind",
        "coeff",
        "question",
        "completion",
        "abstained",
        "correct",
        "human_label",
    )


@pytest.mark.parametrize(
    "text",
    [
        "I don't know who that is.",
        "I don\N{RIGHT SINGLE QUOTATION MARK}t know who that is.",
        "I'm not sure about this.",
        "I\N{RIGHT SINGLE QUOTATION MARK}m not sure about this.",
        "There is no information about this person.",
        "I do not know.",
        "i dont know",
        "I DON'T KNOW",
        "I am not sure.",
        "I'm unable to help with that.",
        "I am not able to say.",
        "I cannot answer that.",
        "I can't find anything.",
        "I could not verify this.",
        "There is no reliable data.",
        "No such record exists.",
        "I am not aware of such a person.",
        "I'm not familiar with him.",
        "That person doesn't exist.",
        "This does not have any information.",
        "They don't have information.",
        "Unknown.",
        "It cannot be determined.",
        "This cannot be answered.",
        "Do not know.",
        "1. Do not know.",
        "- Don't know",
        "Answer:\nDo not know",
        "The author of Figu is not publicly known.",
        "The producer of North is not publicly disclosed.",
        'The director of the film "Student" is not widely publicized in'
        " mainstream sources.",
        "The answer is not known.",
        "The producer of In Conflict is not specified in the provided"
        " information.",
        "The author is unknown.",
    ],
)
def test_phrases_that_abstain(text: str) -> None:
    assert is_abstention(text)


@pytest.mark.parametrize(
    "text",
    [
        "Paris.",
        "Paris is the capital.",
        "",
        "He knows the answer.",
        "The unknowns remain.",
        "hi do not know",
        "He is not sure.",
        "The author is John Smith.",
        "He was not only a poet but also a painter.",
        "Known as the Iron Lady.",
        "He is known as Smith.",
        "The Tomb of the Unknown Soldier.",
        "He is not known for his films.",
        "He is not known to have written it.",
        "They do not know the answer.",
        "The film is not available on Netflix.",
    ],
)
def test_phrases_that_answer(text: str) -> None:
    assert not is_abstention(text)


def test_the_patterns_are_a_parameter() -> None:
    assert is_abstention("foo bar", [re.compile("foo")])
    assert not is_abstention("I do not know", [re.compile("foo")])


def test_classify_scores_an_alias_match_of_the_extracted_answer() -> None:
    assert classify("Paris.", ["Paris"]) == Outcome(False, True)
    assert classify("Paris is the capital.", ["Paris"]) == Outcome(False, True)
    assert classify("Lyon.", ["Paris"]) == Outcome(False, False)


def test_classify_reads_the_first_line_of_the_answer() -> None:
    assert classify("Lyon\nParis", ["Paris"]) == Outcome(False, False)


def test_classify_an_abstention_is_never_correct() -> None:
    assert classify("I do not know.", ["Paris"]) == Outcome(True, False)
    assert classify("I don't know, maybe Paris.", ["Paris"]) == Outcome(
        True, False
    )


def test_classify_without_gold_answers_is_never_correct() -> None:
    assert classify("Paris.", []) == Outcome(False, False)


def test_classify_takes_a_completion() -> None:
    completion = Completion("Paris.", (1, 2), True)

    assert classify(completion, ["Paris"]) == Outcome(False, True)
    assert classify(Completion("I'm not sure.", (3,), True), ["x"]).abstained


def test_behavior_rates_of_the_prototype_counts() -> None:
    rates = behavior_rates(_outcomes(8, 0, 2), _outcomes(0, 4, 6))

    assert rates.n_known == 10
    assert rates.n_unknown == 10
    assert rates.false_refusal == pytest.approx(0.2)
    assert rates.correct_refusal == pytest.approx(0.6)
    assert rates.coverage == pytest.approx(0.6)
    assert rates.wrong_among_answered == pytest.approx(1 / 3)
    assert rates.balanced == pytest.approx(0.7)


def test_behavior_rates_sides_can_differ_in_size() -> None:
    rates = behavior_rates(_outcomes(3, 1, 0), _outcomes(0, 2, 0))

    assert (rates.n_known, rates.n_unknown) == (4, 2)
    assert rates.coverage == 1.0
    assert rates.wrong_among_answered == pytest.approx(0.5)
    assert rates.balanced == pytest.approx(0.5)


def test_behavior_rates_with_nothing_answered() -> None:
    rates = behavior_rates(_outcomes(0, 0, 3), _outcomes(0, 0, 5))

    assert rates.coverage == 0.0
    assert math.isnan(rates.wrong_among_answered)
    assert rates.false_refusal == 1.0
    assert rates.correct_refusal == 1.0
    assert rates.balanced == 0.5


@pytest.mark.parametrize("side", ["known", "unknown"])
def test_behavior_rates_need_both_sides(side: str) -> None:
    full = _outcomes(1, 0, 0)
    pair = ([], full) if side == "known" else (full, [])

    with pytest.raises(ValueError, match="empty"):
        behavior_rates(*pair)


def test_select_coefficient_takes_the_largest_balanced() -> None:
    coeffs = [-0.2, 0.0, 0.2]

    assert (
        select_coefficient(coeffs, [_rates(0.4), _rates(0.5), _rates(0.9)])
        == 0.2
    )
    assert (
        select_coefficient(coeffs, [_rates(0.9), _rates(0.5), _rates(0.4)])
        == -0.2
    )


def test_select_coefficient_ties_go_to_the_smaller_magnitude_then_value() -> (
    None
):
    same = [_rates(0.6)] * 3

    assert select_coefficient([0.2, -0.1, 0.1], same) == -0.1
    assert select_coefficient([0.1, -0.1], same[:2]) == -0.1
    assert select_coefficient([-0.2, 0.0, 0.2], same) == 0.0


def test_select_coefficient_zero_wins_a_tie_with_a_nonzero_value() -> None:
    assert select_coefficient([0.1, 0.0], [_rates(0.6), _rates(0.6)]) == 0.0


def test_select_coefficient_rounds_to_twelve_digits() -> None:
    nearly = [_rates(0.6), _rates(0.6 + 1e-15)]

    assert select_coefficient([0.0, 0.1], nearly) == 0.0
    assert select_coefficient([0.1, 0.0], nearly[::-1]) == 0.0


def test_select_coefficient_a_real_gain_beats_the_smaller_magnitude() -> None:
    assert (
        select_coefficient([0.0, 0.4], [_rates(0.6), _rates(0.6 + 1e-9)]) == 0.4
    )


@pytest.mark.parametrize(
    ("coeffs", "rates", "message"),
    [
        ([0.0, 0.1], [_rates(0.5)], "2 coefficients and 1 rates"),
        ([], [], "empty"),
    ],
    ids=["mismatch", "empty"],
)
def test_select_coefficient_rejects_bad_input(
    coeffs: list[float], rates: list[BehaviorRates], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        select_coefficient(coeffs, rates)


def test_shift_ci_is_the_mean_of_the_paired_difference() -> None:
    base = np.array([1.0, 2.0, 3.0, 4.0])
    steered = np.array([2.0, 4.0, 4.0, 6.0])

    mean, lo, hi = shift_ci(base, steered, n_boot=200)

    assert mean == pytest.approx(1.5)
    assert 1.0 <= lo <= mean <= hi <= 2.0


def test_shift_ci_of_a_constant_shift_has_no_width() -> None:
    base = np.arange(6.0)

    assert shift_ci(base, base + 0.5, n_boot=50) == pytest.approx(
        (0.5, 0.5, 0.5)
    )


def test_shift_ci_keeps_only_the_finite_pairs() -> None:
    base = np.array([1.0, np.nan, 3.0, 4.0, np.inf])
    steered = np.array([2.0, 5.0, np.nan, 5.0, 7.0])

    mean, _, _ = shift_ci(base, steered, n_boot=50)

    assert mean == pytest.approx(1.0)


def test_shift_ci_is_deterministic_and_seeded() -> None:
    rng = np.random.default_rng(1)
    base, steered = rng.normal(size=30), rng.normal(size=30)

    first = shift_ci(base, steered, n_boot=100, seed=3)

    assert shift_ci(base, steered, n_boot=100, seed=3) == first
    assert shift_ci(base, steered, n_boot=100, seed=4) != first


def test_shift_ci_alpha_sets_the_coverage() -> None:
    rng = np.random.default_rng(2)
    base, steered = rng.normal(size=40), rng.normal(size=40)

    _, lo95, hi95 = shift_ci(base, steered, n_boot=400, alpha=0.05)
    _, lo50, hi50 = shift_ci(base, steered, n_boot=400, alpha=0.5)

    assert hi50 - lo50 < hi95 - lo95


def test_shift_ci_rejects_no_finite_pair_and_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="finite"):
        shift_ci(np.array([np.nan, 1.0]), np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="shape"):
        shift_ci(np.zeros(3), np.zeros(4))


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_completion_check_has_the_columns_and_empty_human_labels(
    tmp_path: Path,
) -> None:
    records = [_record("a", True, text="I don't know"), _record("b", False)]
    path = tmp_path / "check.csv"

    written = write_completion_check(path, records, n=10)

    rows = _read(path)
    assert written == len(rows) == 2
    assert tuple(rows[0]) == CHECK_COLUMNS
    assert {r["human_label"] for r in rows} == {""}
    by_uid = {r["uid"]: r for r in rows}
    assert by_uid["a"]["abstained"] == "1"
    assert by_uid["a"]["correct"] == "0"
    assert by_uid["b"]["abstained"] == "0"
    assert by_uid["b"]["correct"] == "1"
    assert by_uid["a"]["completion"] == "I don't know"
    assert by_uid["a"]["question"] == "Q a?"


def test_the_completion_check_alternates_abstained_and_answered(
    tmp_path: Path,
) -> None:
    records = [_record(f"s{i}", i % 3 == 0) for i in range(30)]
    path = tmp_path / "check.csv"

    written = write_completion_check(path, records, n=8)

    flags = [r["abstained"] for r in _read(path)]
    assert written == 8
    assert flags == ["1", "0"] * 4


def test_the_completion_check_continues_when_one_kind_runs_out(
    tmp_path: Path,
) -> None:
    records = [
        _record("a0", True),
        *[_record(f"k{i}", False) for i in range(6)],
    ]
    path = tmp_path / "check.csv"

    written = write_completion_check(path, records, n=5)

    flags = [r["abstained"] for r in _read(path)]
    assert written == 5
    assert flags == ["1", "0", "0", "0", "0"]


def test_the_completion_check_is_seeded(tmp_path: Path) -> None:
    records = [_record(f"s{i}", i % 2 == 0) for i in range(40)]

    write_completion_check(tmp_path / "a.csv", records, n=10, seed=1)
    write_completion_check(tmp_path / "b.csv", records, n=10, seed=1)
    write_completion_check(tmp_path / "c.csv", records, n=10, seed=2)

    a, b, c = ([r["uid"] for r in _read(tmp_path / f"{n}.csv")] for n in "abc")
    assert a == b
    assert a != c


def test_the_completion_check_keeps_multiline_completions(
    tmp_path: Path,
) -> None:
    records = [_record("a", False, text='Paris\n"France"')]
    path = tmp_path / "check.csv"

    write_completion_check(path, records)

    assert _read(path)[0]["completion"] == 'Paris\n"France"'


def test_the_completion_check_of_no_records_writes_the_header(
    tmp_path: Path,
) -> None:
    path = tmp_path / "check.csv"

    assert write_completion_check(path, []) == 0
    assert _read(path) == []
    assert path.read_text(encoding="utf-8").startswith("uid,kind,coeff")


GRID = (-0.4, -0.2, 0.0, 0.2, 0.4)


def test_item_slopes_of_a_step_down_and_a_step_up() -> None:
    values = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    np.testing.assert_allclose(item_slopes(GRID, values), [-1.5, 1.5])


def test_item_slopes_of_a_constant_and_of_a_line() -> None:
    values = np.array(
        [[0.7, -0.3], [0.7, 0.1], [0.7, 0.5], [0.7, 0.9], [0.7, 1.3]]
    )
    np.testing.assert_allclose(
        item_slopes(GRID, values), [0.0, 2.0], atol=1e-12
    )


def test_item_slopes_on_the_real_grid_with_one_spike() -> None:
    grid = (-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4)
    values = np.array([1.0, 0, 0, 0, 0, 0, 0]).reshape(7, 1)
    assert item_slopes(grid, values)[0] == pytest.approx(-20 / 21)


def test_item_slopes_on_an_asymmetric_grid() -> None:
    values = np.array([0.0, 1.0, 2.0]).reshape(3, 1)
    assert item_slopes((0.0, 1.0, 3.0), values)[0] == pytest.approx(9 / 14)


def test_item_slopes_is_nan_only_for_a_column_with_a_non_finite_value() -> None:
    values = np.array(
        [[1.0, 0.0], [1.0, np.nan], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    slopes = item_slopes(GRID, values)
    assert slopes[0] == pytest.approx(-1.5)
    assert math.isnan(slopes[1])


def test_item_slopes_rejects_a_shape_mismatch_and_one_coefficient() -> None:
    with pytest.raises(ValueError, match="rows"):
        item_slopes(GRID, np.zeros((4, 2)))
    with pytest.raises(ValueError, match="distinct"):
        item_slopes((0.1, 0.1, 0.1), np.zeros((3, 2)))


def test_slope_ci_is_the_mean_slope_with_an_item_interval() -> None:
    mean, lo, hi = slope_ci(np.array([-1.5, -1.5, 0.0]), n_boot=200)
    assert mean == pytest.approx(-1.0)
    assert -1.5 <= lo <= -1.0 <= hi <= 0.0


def test_slope_ci_of_identical_slopes_has_no_width() -> None:
    assert slope_ci(np.array([0.5, 0.5, 0.5]), n_boot=50) == (0.5, 0.5, 0.5)


def test_slope_ci_is_deterministic_and_seeded() -> None:
    slopes = np.array([-1.5, 0.0, 1.5, 0.3])
    assert slope_ci(slopes, n_boot=100, seed=3) == slope_ci(
        slopes, n_boot=100, seed=3
    )


def test_slope_ci_drops_an_item_without_a_finite_slope() -> None:
    mean, _, _ = slope_ci(np.array([-1.5, np.nan, -0.5]), n_boot=50)
    assert mean == pytest.approx(-1.0)


def test_slope_ci_against_a_reference_is_paired() -> None:
    same = np.array([-1.5, 0.0])
    assert slope_ci(same, same, n_boot=100) == (0.0, 0.0, 0.0)
    lat = np.array([-1.5, -1.5])
    assert slope_ci(lat, np.zeros(2), n_boot=100) == (-1.5, -1.5, -1.5)


def test_abs_slope_gap_is_the_difference_of_the_absolute_mean_slopes() -> None:
    lat = np.array([-1.5, -1.5])
    assert abs_slope_gap_ci(lat, np.array([1.0, 1.0]), n_boot=50) == (
        0.5,
        0.5,
        0.5,
    )
    assert abs_slope_gap_ci(lat, np.array([2.0, 2.0]), n_boot=50) == (
        -0.5,
        -0.5,
        -0.5,
    )


def test_abs_slope_gap_resamples_both_arms_on_the_same_items() -> None:
    same = np.array([-3.0, 0.0, -1.0, 2.0])
    assert abs_slope_gap_ci(same, same, n_boot=100) == (0.0, 0.0, 0.0)


def test_abs_slope_gap_interval_covers_the_estimate() -> None:
    lat = np.array([-2.0, -1.0, 0.0, -3.0])
    ref = np.array([0.0, 1.0, 1.0, 0.0])
    gap, lo, hi = abs_slope_gap_ci(lat, ref, n_boot=200, seed=1)
    assert gap == pytest.approx(1.5 - 0.5)
    assert lo <= gap <= hi
    assert abs_slope_gap_ci(lat, ref, n_boot=200, seed=1) == (gap, lo, hi)


def test_abs_slope_gap_drops_an_item_not_finite_in_both_arms() -> None:
    lat = np.array([-2.0, np.nan, -2.0])
    ref = np.array([1.0, 5.0, np.nan])
    assert abs_slope_gap_ci(lat, ref, n_boot=20)[0] == pytest.approx(1.0)


def test_abs_slope_gap_rejects_a_shape_mismatch_and_no_finite_item() -> None:
    with pytest.raises(ValueError, match="shape"):
        abs_slope_gap_ci(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="finite"):
        abs_slope_gap_ci(np.array([np.nan]), np.array([1.0]))
