"""Tests for the training-cutoff probe and the label tables."""

import math

import numpy as np
import pytest

from tomography.datasets.records import QAExample
from tomography.knowledge import ItemLabel
from tomography.knowledge import Label
from tomography.p9.pool import GuessVerdict
from tomography.p9.pool import LabeledItem
from tomography.p9.probe import CutoffEstimate
from tomography.p9.probe import YearRate
from tomography.p9.probe import estimate_cutoff
from tomography.p9.probe import label_table
from tomography.p9.probe import popularity_table
from tomography.p9.probe import year_rates


def _labeled(
    example: QAExample,
    label: Label = "known",
    guess: GuessVerdict | None = None,
) -> LabeledItem:
    correct = label == "known"
    return LabeledItem(
        example,
        ItemLabel(
            uid=example.uid,
            label=label,
            basis="behavior",
            greedy="gold" if correct else "other",
            greedy_correct=correct,
            samples=("gold",) * 5 if correct else ("other",) * 5,
            n_correct=5 if correct else 0,
        ),
        guess,
    )


def _award(
    uid: str,
    year: object,
    label: Label = "known",
    guess: GuessVerdict | None = None,
    kind: str = "real",
) -> LabeledItem:
    example = QAExample(
        uid=uid,
        question=f"question {uid}",
        answers=("gold",),
        kind=kind,
        split="train",
        meta={
            "source": "wikidata",
            "relation": "Award",
            "year": year,  # type: ignore[dict-item]
        },
    )
    return _labeled(example, label, guess)


def _popqa(
    uid: str,
    relation: str,
    popularity: object,
    label: Label = "known",
    guess: GuessVerdict | None = None,
) -> LabeledItem:
    example = QAExample(
        uid=uid,
        question=f"question {uid}",
        answers=("gold",),
        kind="real",
        split="train",
        meta={
            "source": "popqa",
            "relation": relation,
            "popularity": popularity,  # type: ignore[dict-item]
        },
    )
    return _labeled(example, label, guess)


def _other(uid: str, source: str, kind: str, label: Label) -> LabeledItem:
    example = QAExample(
        uid=uid,
        question=f"question {uid}",
        answers=("gold",),
        kind=kind,
        split="train",
        meta={"source": source, "relation": "r", "popularity": 7},
    )
    return _labeled(example, label)


def _planted(known_per_year: list[int], n: int = 20) -> list[YearRate]:
    return [
        YearRate(2000 + i, n, known) for i, known in enumerate(known_per_year)
    ]


def _loglik(k: int, n: int) -> float:
    p = k / n
    return k * math.log(p) + (n - k) * math.log1p(-p)


def test_year_rate_is_the_known_share() -> None:
    assert YearRate(2001, 8, 2).rate == 0.25


def test_year_rates_count_award_items_by_year_sorted() -> None:
    items = [
        _award("a", 2001, "known"),
        _award("b", 1999, "unknown"),
        _award("c", 2001, "unknown"),
        _award("d", 2001, "known"),
        _award("e", 1999, "uncertain"),
        _popqa("p", "father", 5),
    ]

    rates = year_rates(items)

    assert rates == [YearRate(1999, 2, 0), YearRate(2001, 3, 2)]


def test_year_rates_count_a_failed_known_as_not_known() -> None:
    items = [
        _award("a", 2001, "known"),
        _award("b", 2001, "known", guess="modal_answer"),
    ]

    assert year_rates(items) == [YearRate(2001, 2, 1)]


def test_year_rates_of_a_pool_without_awards_is_empty() -> None:
    assert year_rates([_popqa("p", "father", 5)]) == []


def test_year_rates_reject_a_year_that_is_not_an_int() -> None:
    with pytest.raises(ValueError, match="bad"):
        year_rates([_award("bad", "2001")])


def test_estimate_cutoff_finds_a_clean_step() -> None:
    rates = _planted([10] * 5 + [1] * 5)

    estimate = estimate_cutoff(rates)

    expected_gain = _loglik(50, 100) + _loglik(5, 100) - _loglik(55, 200)
    assert estimate.year == 2004
    assert estimate.rate_before == pytest.approx(0.5)
    assert estimate.rate_after == pytest.approx(0.05)
    assert estimate.gain == pytest.approx(expected_gain)


def test_estimate_cutoff_returns_the_last_year_of_the_high_regime() -> None:
    estimate = estimate_cutoff(_planted([12] * 13 + [1] * 12))

    assert isinstance(estimate, CutoffEstimate)
    assert estimate.year == 2012
    assert estimate.rate_before == pytest.approx(0.6)
    assert estimate.rate_after == pytest.approx(0.05)
    assert estimate.gain > 10


def test_estimate_cutoff_on_noisy_planted_labels() -> None:
    rng = np.random.default_rng(1)
    items = [
        _award(
            f"a{award}-{year}",
            year,
            "known"
            if rng.random() < (0.6 if year <= 2012 else 0.05)
            else "unknown",
        )
        for award in range(3)
        for year in range(1990, 2025)
    ]

    estimate = estimate_cutoff(year_rates(items))

    assert abs(estimate.year - 2012) <= 2
    assert estimate.gain > 10
    assert estimate.rate_before > estimate.rate_after


def test_estimate_cutoff_sorts_the_years() -> None:
    rates = _planted([10] * 5 + [1] * 5)

    assert estimate_cutoff(rates[::-1]).year == 2004


def test_estimate_cutoff_keeps_only_splits_where_the_rate_drops() -> None:
    rates = _planted([2] * 4 + [18] * 4 + [2] * 4)

    estimate = estimate_cutoff(rates)

    assert estimate.year == 2007
    assert estimate.rate_before > estimate.rate_after


def test_estimate_cutoff_respects_min_years() -> None:
    rates = _planted([10] * 2 + [1] * 8)

    assert estimate_cutoff(rates, min_years=1).year == 2001
    assert estimate_cutoff(rates, min_years=3).year >= 2002


def test_estimate_cutoff_tries_the_last_admissible_split() -> None:
    estimate = estimate_cutoff(_planted([10] * 7 + [1] * 3), min_years=3)

    assert estimate.year == 2006


def test_estimate_cutoff_rejects_a_flat_rate() -> None:
    with pytest.raises(ValueError, match="drop"):
        estimate_cutoff(_planted([5] * 10))


def test_estimate_cutoff_rejects_a_rising_rate() -> None:
    with pytest.raises(ValueError, match="drop"):
        estimate_cutoff(_planted([1] * 5 + [10] * 5))


def test_estimate_cutoff_rejects_too_few_years() -> None:
    with pytest.raises(ValueError, match="6"):
        estimate_cutoff(_planted([10, 10, 1, 1, 1]), min_years=3)


def test_estimate_cutoff_rejects_a_min_years_below_one() -> None:
    with pytest.raises(ValueError, match="min_years"):
        estimate_cutoff(_planted([10, 1]), min_years=0)


def test_label_table_has_one_row_per_group_and_kind() -> None:
    items = [
        _popqa("p0", "father", 5, "known"),
        _popqa("p1", "father", 5, "known", guess="modal_answer"),
        _popqa("p2", "father", 5, "unknown"),
        _popqa("p3", "father", 5, "uncertain"),
        _award("a0", 2001, "known", kind="real"),
        _award("a1", 2020, "unknown", kind="post_cutoff"),
        _other("t0", "triviaqa", "open", "known"),
        _other("k0", "kuq", "kuq:future", "unknown"),
    ]

    rows = label_table(items)

    assert rows == [
        ["popqa", "real", 4, 2, 1, 1, 2],
        ["award", "post_cutoff", 1, 0, 0, 1, 0],
        ["award", "real", 1, 1, 1, 0, 0],
        ["kuq", "kuq:future", 1, 0, 0, 1, 0],
        ["triviaqa", "open", 1, 1, 1, 0, 0],
    ]
    assert all(row[2] == sum(row[4:]) for row in rows)


def test_popularity_table_bins_popqa_items_per_relation() -> None:
    items = [
        _popqa("a", "father", 0, "known"),
        _popqa("b", "father", 999, "unknown"),
        _popqa("c", "father", 1_000, "known"),
        _popqa("d", "father", 10_000, "known", guess="modal_answer"),
        _popqa("e", "father", 100_000, "uncertain"),
        _popqa("f", "mother", 50_000, "known"),
        _award("w", 2001),
        _other("t", "triviaqa", "open", "known"),
    ]

    rows = popularity_table(items)

    assert rows == [
        ["father", "[0, 1000)", 2, 1, 1, 1, 0],
        ["father", "[1000, 10000)", 1, 1, 1, 0, 0],
        ["father", "[10000, 100000)", 1, 1, 0, 0, 1],
        ["father", "[100000, inf)", 1, 0, 0, 0, 1],
        ["mother", "[10000, 100000)", 1, 1, 1, 0, 0],
    ]


def test_popularity_table_takes_custom_edges() -> None:
    items = [_popqa("a", "father", 3), _popqa("b", "father", 12)]

    rows = popularity_table(items, edges=(10,))

    assert [row[1] for row in rows] == ["[0, 10)", "[10, inf)"]


def test_popularity_table_rejects_a_popularity_that_is_not_an_int() -> None:
    with pytest.raises(ValueError, match="bad"):
        popularity_table([_popqa("bad", "father", None)])


@pytest.mark.parametrize(
    "edges", [(), (10, 5), (5, 5)], ids=["empty", "descending", "repeated"]
)
def test_popularity_table_rejects_unordered_edges(
    edges: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="edges"):
        popularity_table([_popqa("a", "father", 3)], edges=edges)
