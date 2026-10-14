import dataclasses
from typing import Protocol
from typing import runtime_checkable

import pytest

from tomography.datasets.records import ChoiceExample
from tomography.datasets.records import Example
from tomography.datasets.records import QAExample
from tomography.datasets.records import assign_splits


@runtime_checkable
class ExampleLike(Protocol):
    uid: str
    text: str
    label: int
    meta: object


def test_example_defaults_to_empty_meta() -> None:
    example = Example(uid="a", text="hello", label=1, split="train")

    assert example.meta == {}


def test_example_meta_is_not_shared_between_instances() -> None:
    first = Example(uid="a", text="x", label=0, split="train")
    second = Example(uid="b", text="y", label=0, split="train")

    assert first.meta is not second.meta


def test_example_is_frozen() -> None:
    example = Example(uid="a", text="hello", label=1, split="train")

    with pytest.raises(dataclasses.FrozenInstanceError):
        example.label = 0  # type: ignore[misc]


def test_example_satisfies_the_example_like_protocol() -> None:
    example = Example(uid="a", text="hello", label=1, split="val")

    assert isinstance(example, ExampleLike)


def test_choice_example_flatten_gives_one_example_per_choice() -> None:
    item = ChoiceExample(
        uid="q7",
        question="Which is hot?",
        choices=("ice", "fire", "snow"),
        correct=1,
        split="test",
    )

    flat = item.flatten()

    assert [e.uid for e in flat] == ["q7:0", "q7:1", "q7:2"]
    assert [e.text for e in flat] == ["ice", "fire", "snow"]
    assert [e.label for e in flat] == [0, 1, 0]
    assert {e.split for e in flat} == {"test"}


def test_choice_example_flatten_sets_group_and_question_meta() -> None:
    item = ChoiceExample(
        uid="q7",
        question="Which is hot?",
        choices=("ice", "fire"),
        correct=1,
        split="train",
    )

    for example in item.flatten():
        assert example.meta["group"] == "q7"
        assert example.meta["question"] == "Which is hot?"


def test_choice_example_flatten_keeps_the_item_meta() -> None:
    item = ChoiceExample(
        uid="q7",
        question="Which is hot?",
        choices=("ice", "fire"),
        correct=0,
        split="train",
        meta={"category": "Science"},
    )

    for example in item.flatten():
        assert example.meta["category"] == "Science"


def test_choice_example_flatten_does_not_mutate_the_item_meta() -> None:
    item = ChoiceExample(
        uid="q7",
        question="Which is hot?",
        choices=("ice", "fire"),
        correct=0,
        split="train",
        meta={"category": "Science"},
    )

    item.flatten()

    assert dict(item.meta) == {"category": "Science"}


@pytest.mark.parametrize("correct", [-1, 2], ids=["negative", "past_the_end"])
def test_choice_example_flatten_rejects_an_out_of_range_index(
    correct: int,
) -> None:
    item = ChoiceExample(
        uid="q7",
        question="Which is hot?",
        choices=("ice", "fire"),
        correct=correct,
        split="train",
    )

    with pytest.raises(ValueError, match="q7"):
        item.flatten()


def test_qa_example_defaults_to_empty_meta() -> None:
    item = QAExample(
        uid="p1",
        question="Who wrote it?",
        answers=("Ada",),
        kind="real",
        split="train",
    )

    assert item.meta == {}
    assert item.answers == ("Ada",)


def test_assign_splits_returns_one_valid_split_per_uid() -> None:
    uids = [f"u{i}" for i in range(50)]

    splits = assign_splits(uids, dataset="demo")

    assert len(splits) == 50
    assert set(splits) <= {"train", "val", "test"}


def test_assign_splits_is_deterministic() -> None:
    uids = [f"u{i}" for i in range(200)]

    first = assign_splits(uids, dataset="demo", seed=3)
    second = assign_splits(uids, dataset="demo", seed=3)

    assert first == second


def test_assign_splits_ignores_row_order() -> None:
    uids = [f"u{i}" for i in range(200)]
    forward = dict(zip(uids, assign_splits(uids, dataset="demo"), strict=True))
    reversed_uids = uids[::-1]
    backward = dict(
        zip(
            reversed_uids,
            assign_splits(reversed_uids, dataset="demo"),
            strict=True,
        )
    )

    assert forward == backward


def test_assign_splits_gives_the_same_split_to_a_repeated_uid() -> None:
    splits = assign_splits(["a", "b", "a"], dataset="demo")

    assert splits[0] == splits[2]


def test_assign_splits_depends_on_the_dataset_name() -> None:
    uids = [f"u{i}" for i in range(200)]

    first = assign_splits(uids, dataset="one")
    second = assign_splits(uids, dataset="two")

    assert first != second


def test_assign_splits_depends_on_the_seed() -> None:
    uids = [f"u{i}" for i in range(200)]

    first = assign_splits(uids, dataset="demo", seed=0)
    second = assign_splits(uids, dataset="demo", seed=1)

    assert first != second


def test_assign_splits_follows_the_fractions() -> None:
    uids = [f"u{i}" for i in range(10_000)]

    splits = assign_splits(uids, dataset="demo")

    for name, share in (("train", 0.6), ("val", 0.2), ("test", 0.2)):
        assert splits.count(name) / len(uids) == pytest.approx(share, abs=0.02)


def test_assign_splits_with_all_mass_on_one_split() -> None:
    uids = [f"u{i}" for i in range(100)]

    assert set(
        assign_splits(uids, dataset="demo", fractions=(1.0, 0.0, 0.0))
    ) == {"train"}
    assert set(
        assign_splits(uids, dataset="demo", fractions=(0.0, 1.0, 0.0))
    ) == {"val"}
    assert set(
        assign_splits(uids, dataset="demo", fractions=(0.0, 0.0, 1.0))
    ) == {"test"}


def test_assign_splits_empty_input_gives_empty_output() -> None:
    assert assign_splits([], dataset="demo") == []


@pytest.mark.parametrize(
    "fractions",
    [(0.5, 0.2, 0.2), (0.7, 0.2, 0.2), (1.2, -0.1, -0.1), (0.0, 0.0, 0.0)],
    ids=["sums_low", "sums_high", "negative_part", "all_zero"],
)
def test_assign_splits_rejects_invalid_fractions(
    fractions: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError, match="fractions"):
        assign_splits(["a"], dataset="demo", fractions=fractions)


def test_assign_splits_recipe_is_frozen() -> None:
    uids = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
    ]

    splits = assign_splits(uids, dataset="golden", seed=7)

    assert splits == [
        "train",
        "test",
        "test",
        "train",
        "val",
        "test",
        "train",
        "train",
    ]
