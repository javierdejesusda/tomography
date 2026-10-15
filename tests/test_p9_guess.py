"""Tests for the guess control on known labels."""

import json

import pytest

from tomography.datasets.records import QAExample
from tomography.knowledge import ItemLabel
from tomography.knowledge import Label
from tomography.p9.guess import GUESS_GROUPS
from tomography.p9.guess import StratumGuess
from tomography.p9.guess import apply_guess_control
from tomography.p9.guess import guess_stratum
from tomography.p9.guess import guess_verdict
from tomography.p9.guess import stratum_guesses
from tomography.p9.pool import LabeledItem
from tomography.p9.pool import pool_from_records
from tomography.p9.pool import pool_records


def _item(
    uid: str,
    relation: str,
    greedy: str,
    gold: str,
    label: Label = "known",
    source: str = "popqa",
    kind: str = "real",
) -> LabeledItem:
    """A behavior-labeled item whose greedy answer is `greedy`."""
    example = QAExample(
        uid=uid,
        question=f"question {uid}",
        answers=(gold,),
        kind=kind,
        split="train",
        meta={"relation": relation, "popularity": 5, "source": source},
    )
    correct = label == "known"
    return LabeledItem(
        example,
        ItemLabel(
            uid=uid,
            label=label,
            basis="behavior",
            greedy=greedy,
            greedy_correct=correct,
            samples=(greedy,) * 5,
            n_correct=5 if correct else 0,
        ),
    )


def _construction(uid: str, source: str = "synthal") -> LabeledItem:
    example = QAExample(
        uid=uid,
        question=f"question {uid}",
        answers=(),
        kind="fictional",
        split="train",
        meta={"relation": "father", "popularity": None, "source": source},
    )
    return LabeledItem(
        example, ItemLabel(uid, "unknown", "construction", "", False, (), 0)
    )


def _pilot_items() -> list[LabeledItem]:
    """Five pilot-shaped strata and one item that belongs to none."""
    father = [
        _item(
            f"f{i}",
            "father",
            "John" if i < 6 else f"Dad {i}",
            "John" if i < 2 else f"Dad {i}",
            "known" if i < 2 or i >= 6 else "unknown",
        )
        for i in range(10)
    ]
    mother = [
        _item(
            f"m{i}",
            "mother",
            "Mary" if i < 3 else f"Mom {i}",
            "Mary" if i == 0 else f"Mom {i}",
            "unknown" if i in (1, 2) else "known",
        )
        for i in range(20)
    ]
    sport = [_item(f"s{i}", "sport", "football", "football") for i in range(4)]
    trivia = [
        _item(f"t{i}", "", f"T{i}", f"T{i}", source="triviaqa")
        for i in range(5)
    ]
    occupation = [
        *(
            _item(f"e{i}", "occupation", "", "Poet", "unknown")
            for i in range(5)
        ),
        _item("e5", "occupation", "Poet", "Poet"),
        _item("e6", "occupation", "Singer", "Singer"),
        _item("e7", "occupation", "Poet", "Painter", "unknown"),
        _item("e8", "occupation", "Singer", "Dancer", "unknown"),
    ]
    return [
        *father,
        *mother,
        *sport,
        *trivia,
        *occupation,
        _construction("g0"),
    ]


def test_guess_groups() -> None:
    assert GUESS_GROUPS == ("popqa", "award", "triviaqa", "nq_open")


@pytest.mark.parametrize(
    ("source", "kind", "expected"),
    [
        ("popqa", "real", "popqa:father"),
        ("wikidata", "real", "award:father"),
        ("wikidata", "post_cutoff", "award:father"),
        ("triviaqa", "open", "triviaqa"),
        ("nq_open", "open", "nq_open"),
        ("synthal", "fictional", None),
        ("generated", "fictional", None),
        ("wikidata", "unknowable_detail", None),
        ("falseqa", "false_premise", None),
        ("kuq", "kuq:future", None),
    ],
    ids=[
        "popqa",
        "award_real",
        "award_post_cutoff",
        "triviaqa",
        "nq_open",
        "synthal",
        "generated",
        "detail",
        "falseqa",
        "kuq",
    ],
)
def test_guess_stratum_per_group(
    source: str, kind: str, expected: str | None
) -> None:
    item = _item("a", "father", "x", "x", source=source, kind=kind)

    assert guess_stratum(item) == expected


def test_guess_stratum_ignores_construction_items() -> None:
    assert guess_stratum(_construction("a", source="popqa")) is None


def test_stratum_guesses_cover_the_pilot_strata_sorted() -> None:
    guesses = stratum_guesses(_pilot_items())

    assert list(guesses) == [
        "popqa:father",
        "popqa:mother",
        "popqa:occupation",
        "popqa:sport",
        "triviaqa",
    ]


def test_stratum_guess_of_a_guessable_relation() -> None:
    guesses = stratum_guesses(_pilot_items())

    assert guesses["popqa:father"] == StratumGuess(
        "popqa:father", 10, "john", 0.6, 0.2, False, False
    )


def test_stratum_guess_of_an_eligible_relation() -> None:
    guess = stratum_guesses(_pilot_items())["popqa:mother"]

    assert guess.modal == "mary"
    assert guess.modal_share == pytest.approx(0.15)
    assert guess.guess_accuracy == pytest.approx(0.05)
    assert guess.eligible
    assert not guess.excluded


def test_stratum_guess_of_an_excluded_relation() -> None:
    guess = stratum_guesses(_pilot_items())["popqa:sport"]

    assert guess.excluded
    assert not guess.eligible
    assert guess.guess_accuracy == 1.0


def test_empty_answers_never_form_the_mode_and_ties_go_alphabetically() -> None:
    guess = stratum_guesses(_pilot_items())["popqa:occupation"]

    assert guess.modal == "poet"
    assert guess.modal_share == pytest.approx(2 / 9)
    assert guess.guess_accuracy == pytest.approx(6 / 9)
    assert not guess.eligible


def test_no_answer_seen_twice_gives_no_guess() -> None:
    guess = stratum_guesses(_pilot_items())["triviaqa"]

    assert (guess.modal, guess.modal_share, guess.guess_accuracy) == (
        "",
        0.0,
        0.0,
    )
    assert guess.eligible


def test_min_modal_count_sets_the_floor_of_a_guess_pattern() -> None:
    guess = stratum_guesses(_pilot_items(), min_modal_count=1)["triviaqa"]

    assert guess.modal == "t0"
    assert guess.modal_share == pytest.approx(0.2)
    assert guess.guess_accuracy == pytest.approx(0.2)
    assert not guess.eligible


def test_min_modal_count_below_one_raises() -> None:
    with pytest.raises(ValueError, match="min_modal_count"):
        stratum_guesses(_pilot_items(), min_modal_count=0)


def test_max_guess_accuracy_and_excluded_are_parameters() -> None:
    guesses = stratum_guesses(
        _pilot_items(), max_guess_accuracy=0.04, excluded=()
    )

    assert not guesses["popqa:mother"].eligible
    assert not guesses["popqa:sport"].excluded


def test_the_modal_answer_is_normalized() -> None:
    items = [
        _item("a", "father", "John", "x"),
        _item("b", "father", "john.", "y"),
        _item("c", "father", "The John", "z"),
        _item("d", "father", "Mark", "Mark"),
    ]

    guess = stratum_guesses(items)["popqa:father"]

    assert guess.modal == "john"
    assert guess.modal_share == pytest.approx(0.75)


def test_a_constant_award_winner_is_caught() -> None:
    items = [
        _item(
            f"a{i}",
            "Nobel Prize",
            "Marie Curie",
            "Marie Curie" if i < 3 else f"Laureate {i}",
            "known" if i < 3 else "unknown",
            source="wikidata",
        )
        for i in range(10)
    ]

    guesses = stratum_guesses(items)

    assert guesses["award:Nobel Prize"].modal == "marie curie"
    assert not guesses["award:Nobel Prize"].eligible
    controlled = apply_guess_control(items, guesses)
    assert [i.guess for i in controlled[:3]] == ["guessable_stratum"] * 3


def test_guess_verdict_failures_of_the_pilot_strata() -> None:
    items = _pilot_items()

    controlled = {
        i.example.uid: i
        for i in apply_guess_control(items, stratum_guesses(items))
    }

    assert {controlled[u].guess for u in ("f0", "f1", "f6", "f9")} == {
        "guessable_stratum"
    }
    assert controlled["m0"].guess == "modal_answer"
    assert {controlled[f"m{i}"].guess for i in range(3, 20)} == {None}
    assert {controlled[f"s{i}"].guess for i in range(4)} == {"excluded_stratum"}
    assert {controlled[f"e{i}"].guess for i in (5, 6)} == {"guessable_stratum"}
    assert {controlled[f"t{i}"].guess for i in range(5)} == {None}


def test_guess_verdict_order_is_excluded_then_guessable_then_modal() -> None:
    items = [_item(f"s{i}", "sport", "football", "football") for i in range(4)]
    guesses = stratum_guesses(items)

    assert guess_verdict(items[0], guesses) == "excluded_stratum"
    eligible = StratumGuess("popqa:sport", 4, "football", 1.0, 0.0, False, True)
    assert guess_verdict(items[0], {"popqa:sport": eligible}) == "modal_answer"
    guessable = StratumGuess(
        "popqa:sport", 4, "football", 1.0, 0.5, False, False
    )
    assert (
        guess_verdict(items[0], {"popqa:sport": guessable})
        == "guessable_stratum"
    )


@pytest.mark.parametrize("label", ["unknown", "uncertain"])
def test_guess_verdict_is_none_for_a_label_that_is_not_known(
    label: Label,
) -> None:
    items = [
        _item(f"s{i}", "sport", "football", "football", label) for i in range(4)
    ]

    assert guess_verdict(items[0], stratum_guesses(items)) is None


def test_guess_verdict_is_none_for_construction_items() -> None:
    assert guess_verdict(_construction("g"), {}) is None


def test_guess_verdict_needs_the_stratum_of_a_known_item() -> None:
    with pytest.raises(ValueError, match="popqa:father"):
        guess_verdict(_item("a", "father", "x", "x"), {})


def test_apply_guess_control_keeps_raw_labels_and_order() -> None:
    items = _pilot_items()

    controlled = apply_guess_control(items, stratum_guesses(items))

    assert [i.example for i in controlled] == [i.example for i in items]
    assert [i.label for i in controlled] == [i.label for i in items]
    by_uid = {i.example.uid: i for i in controlled}
    assert by_uid["m0"].label.label == "known"
    assert by_uid["m0"].contrast_label == "uncertain"
    assert by_uid["m3"].contrast_label == "known"
    assert by_uid["f3"].contrast_label == "unknown"
    assert by_uid["g0"].guess is None
    assert all(i.guess is None for i in controlled if i.label.label != "known")


def test_the_verdict_survives_the_record_round_trip() -> None:
    items = _pilot_items()
    controlled = apply_guess_control(items, stratum_guesses(items))
    assert any(i.guess is not None for i in controlled)

    back = pool_from_records(json.loads(json.dumps(pool_records(controlled))))

    assert [i.guess for i in back] == [i.guess for i in controlled]
