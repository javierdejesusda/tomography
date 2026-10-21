"""Tests for the discrete semantic entropy of sampled answers."""

from collections.abc import Sequence
import math

import pytest

from tomography.p9.semantic_entropy import cluster_entropy
from tomography.p9.semantic_entropy import entailment_pairs
from tomography.p9.semantic_entropy import semantic_clusters
from tomography.p9.semantic_entropy import semantic_entropy

QUESTION = "What is the capital of France?"


def text(answer: str) -> str:
    """The text the entailment model sees for one answer."""
    return f"{QUESTION} {answer}"


def both(first: str, second: str) -> set[tuple[str, str]]:
    """Entailment in both directions between two answers."""
    return {
        (text(first), text(second)),
        (text(second), text(first)),
    }


def verdicts(
    answers: Sequence[str], *relations: set[tuple[str, str]]
) -> dict[tuple[str, str], bool]:
    """Hand-made verdicts for every pair the model would be asked about."""
    entailed: set[tuple[str, str]] = set().union(*relations)
    return {
        pair: pair in entailed for pair in entailment_pairs(QUESTION, answers)
    }


class StubEntailer:
    """Entailment from a hand-made set of directed text pairs.

    Every call is recorded.
    """

    def __init__(self, *relations: set[tuple[str, str]]) -> None:
        self.pairs: set[tuple[str, str]] = set().union(*relations)
        self.calls: list[tuple[list[str], list[str]]] = []

    def __call__(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> list[bool]:
        self.calls.append((list(premises), list(hypotheses)))
        return [
            (premise, hypothesis) in self.pairs
            for premise, hypothesis in zip(premises, hypotheses, strict=True)
        ]


@pytest.mark.parametrize(
    ("cluster_ids", "expected"),
    [
        ([0], 0.0),
        ([0, 0, 0, 0, 0], 0.0),
        ([0, 1], math.log(2)),
        ([0, 1, 2, 3, 4], math.log(5)),
        ([0, 0, 1, 2], 1.5 * math.log(2)),
        ([0, 1, 1, 0, 2], -(0.4 * math.log(0.4) * 2 + 0.2 * math.log(0.2))),
    ],
    ids=["one", "identical", "split", "singletons", "half", "shuffled"],
)
def test_cluster_entropy_is_the_natural_log_entropy_of_the_shares(
    cluster_ids: list[int], expected: float
) -> None:
    assert cluster_entropy(cluster_ids) == pytest.approx(expected, abs=1e-12)


def test_cluster_entropy_does_not_depend_on_the_id_values() -> None:
    assert cluster_entropy([7, 7, 3]) == pytest.approx(
        cluster_entropy([0, 0, 1]), abs=1e-12
    )


def test_cluster_entropy_rejects_no_samples() -> None:
    with pytest.raises(ValueError, match="at least one"):
        cluster_entropy([])


def test_entailment_pairs_cover_both_directions_of_distinct_answer_pairs() -> (
    None
):
    answers = ["Paris", " Paris", "Lyon", "", "  ", "Nice", "Paris\n"]

    pairs = entailment_pairs(QUESTION, answers)

    distinct = ["Paris", "Lyon", "Nice"]
    assert len(pairs) == 6
    assert set(pairs) == {
        (text(first), text(second))
        for first in distinct
        for second in distinct
        if first != second
    }


@pytest.mark.parametrize(
    "answers",
    [[], ["Paris"], ["Paris", " Paris", "Paris\n"], ["", " ", "\n"]],
    ids=["none", "one", "identical", "empty"],
)
def test_entailment_pairs_need_two_distinct_non_empty_answers(
    answers: list[str],
) -> None:
    assert entailment_pairs(QUESTION, answers) == []


def test_answers_with_entailment_both_ways_share_a_cluster() -> None:
    answers = [
        "Paris",
        "The city of Paris",
        "Lyon",
        "Paris, France",
        "Lyon, France",
    ]
    lookup = verdicts(
        answers,
        both("Paris", "The city of Paris"),
        both("Paris", "Paris, France"),
        both("Lyon", "Lyon, France"),
    )

    assert semantic_clusters(QUESTION, answers, lookup) == [0, 0, 1, 0, 1]


def test_one_direction_of_entailment_is_not_enough() -> None:
    answers = ["William Shakespeare", "Shakespeare"]
    lookup = verdicts(
        answers, {(text("William Shakespeare"), text("Shakespeare"))}
    )

    assert semantic_clusters(QUESTION, answers, lookup) == [0, 1]


def test_the_clusters_depend_on_the_order_of_the_samples() -> None:
    relations = (both("A", "B"), both("B", "C"))
    in_order = ["A", "B", "C"]
    reordered = ["B", "A", "C"]

    assert semantic_clusters(
        QUESTION, in_order, verdicts(in_order, *relations)
    ) == [0, 0, 1]
    assert semantic_clusters(
        QUESTION, reordered, verdicts(reordered, *relations)
    ) == [0, 0, 0]


def test_a_later_cluster_head_takes_over_a_sample_an_earlier_one_claimed() -> (
    None
):
    answers = ["A", "D", "C"]
    lookup = verdicts(answers, both("A", "C"), both("D", "C"))

    assert semantic_clusters(QUESTION, answers, lookup) == [0, 1, 1]


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (["Paris"] * 5, [0] * 5),
        ([" Paris", "Paris\n", "Paris"], [0, 0, 0]),
        (["Paris"], [0]),
        (["", " ", "\n"], [0, 0, 0]),
    ],
    ids=["identical", "outer_whitespace", "one_sample", "only_empty"],
)
def test_answers_that_need_no_verdict_are_clustered_without_one(
    answers: list[str], expected: list[int]
) -> None:
    assert semantic_clusters(QUESTION, answers, {}) == expected


def test_empty_answers_cluster_together_and_apart_from_every_answer() -> None:
    answers = ["", "Paris", "   ", "Paris, France"]
    lookup = verdicts(answers, both("Paris", "Paris, France"))

    assert semantic_clusters(QUESTION, answers, lookup) == [0, 1, 0, 1]


def test_no_samples_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one"):
        semantic_clusters(QUESTION, [], {})


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (["Paris"] * 5, 0.0),
        (["Paris", "Lyon", "Nice", "Lille", "Metz"], math.log(5)),
        (
            ["Paris", "The city of Paris", "Lyon"],
            -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3)),
        ),
    ],
    ids=["identical", "all_different", "two_clusters"],
)
def test_semantic_entropy_is_the_entropy_of_the_clusters(
    answers: list[str], expected: float
) -> None:
    entailer = StubEntailer(both("Paris", "The city of Paris"))

    value = semantic_entropy(QUESTION, answers, entailer)

    assert value == pytest.approx(expected, abs=1e-12)


def test_semantic_entropy_asks_the_model_about_every_pair_in_one_call() -> None:
    entailer = StubEntailer()
    answers = ["Paris", "Lyon", "Nice", "Lille", "Metz"]

    semantic_entropy(QUESTION, answers, entailer)

    assert len(entailer.calls) == 1
    premises, hypotheses = entailer.calls[0]
    assert list(zip(premises, hypotheses, strict=True)) == entailment_pairs(
        QUESTION, answers
    )
    assert len(premises) == 2 * (5 * 4 // 2)


def test_semantic_entropy_does_not_call_the_model_without_pairs() -> None:
    entailer = StubEntailer()

    semantic_entropy(QUESTION, ["Paris", " Paris", "", "Paris"], entailer)

    assert entailer.calls == []


def test_an_entailer_that_answers_for_other_pairs_is_rejected() -> None:
    def short(premises: Sequence[str], hypotheses: Sequence[str]) -> list[bool]:
        return [True]

    with pytest.raises(ValueError, match="one verdict per pair"):
        semantic_entropy(QUESTION, ["Paris", "Lyon"], short)
