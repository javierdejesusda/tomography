"""Tests for the quantity answer parser and its grading against a range."""

import pytest

from tomography.p10.answers import Parsed
from tomography.p10.answers import answer_text
from tomography.p10.answers import grade
from tomography.p10.answers import parse_answer

EN_DASH = "\u2013"
EM_DASH = "\u2014"
PLUS_MINUS = "\u00b1"


def _point(value: float, approximate: bool = False) -> Parsed:
    """Builds the expected parse of a single value.

    Args:
        value: The value.
        approximate: Whether an approximation marker applies.

    Returns:
        The parse.
    """
    return Parsed("point", value, value, approximate)


def _range(low: float, high: float, approximate: bool = False) -> Parsed:
    """Builds the expected parse of a range.

    Args:
        low: The lower bound.
        high: The upper bound.
        approximate: Whether an approximation marker applies.

    Returns:
        The parse.
    """
    return Parsed("range", low, high, approximate)


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("", ""),
        ("  35 cm  ", "35 cm"),
        ("\n\n  35 cm\nIt is a dog.", "35 cm"),
        ("   \n \t\n40", "40"),
        ("first\nsecond", "first"),
    ],
)
def test_answer_text_is_the_first_non_empty_line(completion, expected):
    assert answer_text(completion) == expected


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("35", _point(35)),
        ("35 cm", _point(35)),
        ("1.5 metres", _point(1.5)),
        ("1,200 metres", _point(1200)),
        ("1,200.5 m", _point(1200.5)),
        ("10, 12", _point(10)),
        ("1,20 m", _point(1)),
        ("twenty", _point(20)),
        ("Twenty-Five", _point(25)),
        ("thirty", _point(30)),
        ("ninety-nine", _point(99)),
        ("zero", _point(0)),
        ("a hundred metres", _point(100)),
        ("one hundred", _point(100)),
        ("two hundred days", _point(200)),
        ("someone said often", None),
        ("The tower is 330 m tall, 12 m wide", _point(330)),
    ],
)
def test_a_point_is_the_first_number(completion, expected):
    if expected is None:
        assert parse_answer(completion).form == "none"
    else:
        assert parse_answer(completion) == expected


def test_a_number_on_a_later_line_is_ignored():
    parsed = parse_answer("It depends.\n42 metres")
    assert parsed.form == "none"
    assert parsed.low is None
    assert parsed.high is None


def test_a_leading_blank_line_is_skipped():
    assert parse_answer("\n\n42 metres") == _point(42)


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("33-41", _range(33, 41)),
        ("33 - 41 cm", _range(33, 41)),
        (f"33{EN_DASH}41 cm", _range(33, 41)),
        (f"33{EM_DASH}41 cm", _range(33, 41)),
        ("30 to 40 cm", _range(30, 40)),
        ("30 cm to 40 cm", _range(30, 40)),
        ("from 30 to 40 cm", _range(30, 40)),
        ("between 9 and 13 years", _range(9, 13)),
        ("between 9 years and 13 years", _range(9, 13)),
        ("12 or 13", _range(12, 13)),
        ("12 days or 13 days", _range(12, 13)),
        ("41 to 33", _range(33, 41)),
        ("ten-twelve", _range(10, 12)),
        ("ten to twelve days", _range(10, 12)),
        ("1,200 to 1,500 m", _range(1200, 1500)),
        ("1.5-2.5 m", _range(1.5, 2.5)),
        ("330 +/- 5", _range(325, 335)),
        (f"330 {PLUS_MINUS} 5 m", _range(325, 335)),
        (f"330{PLUS_MINUS}5", _range(325, 335)),
        ("twenty-five to thirty", _range(25, 30)),
    ],
)
def test_a_range_has_two_quantities_joined(completion, expected):
    assert parse_answer(completion) == expected


@pytest.mark.parametrize(
    "completion",
    [
        "twenty-five",
        "9 and 13 years",
        "30 cm. 40 cm",
        "10, 12",
        "about 30 or so",
    ],
)
def test_two_numbers_that_are_not_joined_make_a_point(completion):
    assert parse_answer(completion).form == "point"


@pytest.mark.parametrize(
    "marker",
    [
        "about",
        "around",
        "approximately",
        "approx.",
        "Roughly",
        "nearly",
        "almost",
        "some",
        "~",
        "up to",
        "over",
        "more than",
        "less than",
        "at least",
        "under",
        "maybe",
        "perhaps",
        "probably",
    ],
)
def test_a_marker_before_the_first_quantity_is_approximate(marker):
    assert parse_answer(f"{marker} 30 cm") == _point(30, approximate=True)


@pytest.mark.parametrize(
    "completion",
    [
        "Typically 30 cm",
        "30 cm, usually",
        "30 cm on average",
        "The average is 30 cm",
        "Generally 30 cm",
    ],
)
def test_a_habit_word_anywhere_in_the_line_is_approximate(completion):
    parsed = parse_answer(completion)
    assert parsed == _point(30, approximate=True)


@pytest.mark.parametrize(
    "completion",
    [
        "30 cm",
        "The tower is 330 m, about as tall as a building",
        "It is someone's 30 cm",
        "Moreover 30 cm",
        "30 cm\nUsually less",
        "Reunder 30 cm",
    ],
)
def test_no_marker_in_the_right_place_is_exact(completion):
    assert parse_answer(completion).approximate is False


def test_a_marker_before_a_range_flags_it_and_keeps_the_range():
    assert parse_answer("about 30 to 40 cm") == _range(30, 40, True)
    assert parse_answer("from around 30 to 40") == _range(30, 40, True)


def test_a_range_without_a_marker_is_not_approximate():
    assert parse_answer("30-40 cm") == _range(30, 40, False)


def test_a_hedge_before_a_number_beats_the_abstention_words():
    assert parse_answer("I'm not sure, maybe 12") == _point(12, True)
    assert parse_answer("I'm not sure, 12") == _point(12, False)


def test_an_almost_equal_sign_is_a_marker():
    assert parse_answer("\u2248 12") == _point(12, approximate=True)
    assert parse_answer("~12") == _point(12, approximate=True)


@pytest.mark.parametrize(
    "completion",
    [
        "I don't know",
        "I'm not sure.",
        "I don\u2019t know the height",
        "\n\nI do not know",
        "Unknown",
    ],
)
def test_no_number_and_an_abstention_phrase_is_an_abstention(completion):
    assert parse_answer(completion) == Parsed("abstain", None, None, False)


@pytest.mark.parametrize(
    "completion",
    ["", "   \n", "Tall.", "A Beagle is a small dog.", "Hmm\n12 cm"],
)
def test_no_number_and_no_abstention_phrase_is_none(completion):
    assert parse_answer(completion) == Parsed("none", None, None, False)


def test_a_number_beats_an_abstention_phrase():
    assert parse_answer("I don't know, 12").form == "point"


@pytest.mark.parametrize(
    ("parsed", "gold", "expected"),
    [
        (Parsed("abstain", None, None, False), (33, 41), None),
        (Parsed("none", None, None, False), (33, 41), None),
        (_point(33), (33, 41), True),
        (_point(41), (33, 41), True),
        (_point(37), (33, 41), True),
        (_point(20), (33, 41), False),
        (_point(100), (100, 100), True),
        (_point(105), (100, 100), True),
        (_point(105.01), (100, 100), False),
        (_point(95), (100, 100), True),
        (_point(94.99), (100, 100), False),
        (_range(30, 40), (33, 41), True),
        (_range(10, 20), (33, 41), False),
        (_range(50, 60), (33, 41), False),
        (_range(90, 110), (100, 100), True),
        (_range(105, 120), (100, 100), True),
        (_range(105.5, 120), (100, 100), False),
        (_range(80, 95), (100, 100), True),
        (_range(80, 94.5), (100, 100), False),
    ],
)
def test_grade_with_the_default_tolerance(parsed, gold, expected):
    assert grade(parsed, *gold) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(24, True), (23.99, False), (50, True), (50.01, False)],
)
def test_the_tolerance_edges_are_inclusive(value, expected):
    assert grade(_point(value), 32, 40, rel_tol=0.25) is expected


def test_a_range_touching_only_the_widened_edge_is_correct():
    assert grade(_range(50, 70), 32, 40, rel_tol=0.25) is True
    assert grade(_range(50.5, 70), 32, 40, rel_tol=0.25) is False
    assert grade(_range(10, 24), 32, 40, rel_tol=0.25) is True
    assert grade(_range(10, 23.5), 32, 40, rel_tol=0.25) is False


def test_a_zero_tolerance_needs_the_value_inside_the_gold():
    assert grade(_point(41), 33, 41, rel_tol=0.0) is True
    assert grade(_point(41.5), 33, 41, rel_tol=0.0) is False


@pytest.mark.parametrize(
    ("low", "high", "rel_tol", "message"),
    [
        (41, 33, 0.05, "gold"),
        (-1, 5, 0.05, "gold"),
        (33, 41, -0.01, "rel_tol"),
    ],
)
def test_grade_rejects_an_invalid_gold_or_tolerance(
    low, high, rel_tol, message
):
    with pytest.raises(ValueError, match=message):
        grade(_point(35), low, high, rel_tol=rel_tol)


def test_grade_validates_even_when_there_is_nothing_to_grade():
    with pytest.raises(ValueError, match="gold"):
        grade(Parsed("none", None, None, False), 41, 33)


@pytest.mark.parametrize(
    "completion",
    ["No one knows", "No one knows the answer.", "one", "One of a kind"],
)
def test_a_bare_one_is_not_a_number(completion):
    assert parse_answer(completion).form in ("abstain", "none")
    assert parse_answer(completion).low is None


def test_no_one_knows_is_an_abstention():
    assert parse_answer("No one knows").form == "abstain"


def test_a_bare_one_is_skipped_for_the_next_number():
    parsed = parse_answer("One of the smallest hounds, about 38 cm")
    assert parsed == _point(38, True)


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("one to two", _range(1, 2)),
        ("zero to one", _range(0, 1)),
        ("one hundred twenty", _point(120)),
        ("one hundred and twenty", _point(120)),
        ("two hundred fifty", _point(250)),
        ("a hundred and five metres", _point(105)),
        ("twenty five", _point(25)),
        ("Thirty  Five days", _point(35)),
        ("a thousand", _point(1000)),
        ("one thousand two hundred", _point(1200)),
        ("twenty thousand", _point(20000)),
        ("3 thousand", _point(3000)),
        ("1.5 million", _point(1.5e6)),
        ("3 hundred metres", _point(300)),
        ("30k", _point(30000)),
        ("1.5K km", _point(1500)),
        ("30km", _point(30)),
        ("a dozen", _point(12)),
        ("two dozen", _point(24)),
        ("5 to 6 hundred", _range(500, 600)),
        ("five to six hundred days", _range(500, 600)),
        ("2 or 3 thousand", _range(2000, 3000)),
        ("50 to 600", _range(50, 600)),
        ("1 thousand to 2 thousand", _range(1000, 2000)),
    ],
)
def test_compound_and_scale_words(completion, expected):
    assert parse_answer(completion) == expected


@pytest.mark.parametrize(
    "completion",
    [
        "several hundred",
        "a few hundred metres",
        "a couple of hundred",
        "hundreds of",
        "Dozens of days",
        "thousands of years",
        "several thousand",
    ],
)
def test_a_vague_quantity_is_approximate_and_has_no_value(completion):
    assert parse_answer(completion) == Parsed("none", None, None, True)


@pytest.mark.parametrize("marker", ["several", "a few", "a couple of"])
def test_a_count_word_before_a_real_number_is_a_marker(marker):
    assert parse_answer(f"{marker} 30 cm") == _point(30, approximate=True)


@pytest.mark.parametrize(
    "dash",
    ["\u2212", "\u2010", "\u2011", "~", " ~ ", "\u2212 "],
)
def test_more_range_separators(dash):
    assert parse_answer(f"30{dash}40 cm") == _range(30, 40)


def test_a_leading_tilde_is_a_marker_not_a_separator():
    assert parse_answer("~12") == _point(12, approximate=True)
    assert parse_answer("~30~40") == _range(30, 40, approximate=True)


@pytest.mark.parametrize(
    "completion",
    ["30 cm or so", "30 give or take", "30 cm or thereabouts", "30, OR SO"],
)
def test_a_trailing_marker_is_approximate(completion):
    assert parse_answer(completion) == _point(30, approximate=True)


def test_a_trailing_marker_after_a_range_flags_it():
    assert parse_answer("30 to 40 or so") == _range(30, 40, approximate=True)


@pytest.mark.parametrize(
    "completion",
    [
        "Tall.\nQ: How old is X?\nA: I don't know",
        "Let me think.\nI don't know",
        "Hmm\nNo one knows",
    ],
)
def test_only_the_answer_line_can_abstain(completion):
    assert parse_answer(completion) == Parsed("none", None, None, False)


METRES = ("m", "metres", "meters")


@pytest.mark.parametrize(
    ("completion", "label", "expected"),
    [
        (
            "The Boeing 737-800 has a wingspan of 35.8 m.",
            "Boeing 737-800",
            _point(35.8),
        ),
        (
            "The Airbus A320 has a wingspan of 35.8 metres.",
            "Airbus A320",
            _point(35.8),
        ),
        ("The Su-27 has a wingspan of 14.7 m.", "Su-27", _point(14.7)),
        ("The B-52 has a wingspan of 56.4 m", "B-52", _point(56.4)),
        (
            "The Cessna 172 has a wingspan of about 11 m.",
            "Cessna 172",
            _point(11, approximate=True),
        ),
        ("The boeing 737-800 spans 35.8 m.", "BOEING 737-800", _point(35.8)),
    ],
)
def test_an_ignored_label_is_not_read_as_the_answer(
    completion, label, expected
):
    assert parse_answer(completion, ignore=(label,)) == expected


def test_an_ignored_label_does_not_merge_its_neighbours():
    assert parse_answer("12B-523", ignore=("B-52",)) == _point(12)


def test_the_longest_ignored_label_is_removed_first():
    completion = "The Boeing 737-800 has a wingspan of 35.8 m."
    assert parse_answer(
        completion, ignore=("Boeing", "Boeing 737-800")
    ) == _point(35.8)


def test_an_empty_ignored_label_changes_nothing():
    assert parse_answer("30 to 40 cm", ignore=("",)) == _range(30, 40)


@pytest.mark.parametrize(
    ("completion", "units", "expected"),
    [
        ("The Boeing 737-800 has a wingspan of 35.8 m", METRES, _point(35.8)),
        (
            "The Airbus A320 has a wingspan of 35.8 metres.",
            METRES,
            _point(35.8),
        ),
        ("The Su-27 has a wingspan of 14.7m.", METRES, _point(14.7)),
        ("The B-52 has a wingspan of 56.4 M", METRES, _point(56.4)),
        (
            "The Cessna 172 has a wingspan of about 11 m.",
            METRES,
            _point(11, approximate=True),
        ),
        ("Pulp Fiction (1994) runs 154 minutes.", ("minutes",), _point(154)),
        (
            "It was released in 1994 and runs 142 minutes.",
            ("minutes",),
            _point(142),
        ),
        ("Born 1990, it is 160 to 188 cm tall", ("cm",), _range(160, 188)),
        ("Born 1990, it is 160 cm to 188 tall", ("cm",), _range(160, 188)),
        (
            "In 1990 it was about 30-35 m",
            METRES,
            _range(30, 35, approximate=True),
        ),
    ],
)
def test_a_quantity_followed_by_a_unit_word_is_preferred(
    completion, units, expected
):
    assert parse_answer(completion, unit_words=units) == expected


@pytest.mark.parametrize(
    "completion",
    ["In 1994 it ran 154 minutes", "In 1994 it was 5 million"],
)
def test_a_unit_word_must_end_at_a_word_boundary(completion):
    assert parse_answer(completion, unit_words=("m",)) == _point(1994)


def test_without_a_unit_word_the_first_quantity_is_the_answer():
    assert parse_answer("The Cessna 172 spans 11", unit_words=METRES) == _point(
        172
    )


def test_ignore_and_unit_words_combine():
    completion = "The Boeing 737-800 (1998) has a wingspan of 35.8 m."
    assert parse_answer(
        completion, ignore=("Boeing 737-800",), unit_words=METRES
    ) == _point(35.8)


def test_unit_words_leave_an_abstention_alone():
    assert parse_answer("I don't know", unit_words=METRES) == Parsed(
        "abstain", None, None, False
    )


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("The Boeing 737-800 has a wingspan of 35.8 m.", _range(737, 800)),
        ("Pulp Fiction (1994) runs 154 minutes.", _point(1994)),
    ],
)
def test_by_default_the_first_quantity_still_wins(completion, expected):
    assert parse_answer(completion) == expected
