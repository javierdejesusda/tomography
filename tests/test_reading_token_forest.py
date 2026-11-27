"""Tests for literal token blocks and scores."""

import dataclasses

import numpy as np
import pytest

from tomography.activations import Activations
from tomography.datasets.stimulus import StimulusSet
from tomography.model import LoadedModel
from tomography.reading import token_forest
from tomography.reading.ensembles import GroupedForest


def _items(texts: tuple[str, ...]) -> StimulusSet:
    pairs = np.arange(len(texts)).reshape(-1, 2)
    return StimulusSet(
        texts=texts,
        response_spans=tuple((2, len(text) - 2) for text in texts),
        concept_spans=(None,) * len(texts),
        labels=np.tile((1, 0), len(pairs)),
        pairs=pairs,
        groups=pairs,
        uids=tuple(map(str, range(len(texts)))),
    )


@pytest.mark.parametrize(
    ("alignment", "expected"),
    [
        ("right", [[0, 11, 12], [23, 24, 25], [31, 32, 33]]),
        ("left", [[11, 12, 0], [23, 24, 25], [31, 32, 33]]),
    ],
)
def test_layout_pads_and_truncates(
    alignment: token_forest.Alignment, expected: list[list[int]]
) -> None:
    # The first row of every prompt is outside its stimulus span.
    activations = np.array(
        [
            [
                [10],
                [11],
                [12],
                [20],
                [21],
                [22],
                [23],
                [24],
                [25],
                [30],
                [31],
                [32],
                [33],
            ]
        ],
        dtype=np.float32,
    )
    layout = token_forest.TokenLayout((4,), 3, 1, alignment)
    packed = np.empty((3, 3), dtype=np.float32)
    layout.pack(packed, activations, np.array([3, 6, 4]), np.array([2, 5, 3]))
    np.testing.assert_array_equal(packed, expected)


def test_layout_keeps_only_the_seven_layer_groups() -> None:
    layers = token_forest.literal_layers(tuple(range(1, 29)))
    assert layers == (4, 8, 12, 16, 20, 24, 28)
    layout = token_forest.TokenLayout(layers, 3, 2, "right")
    assert [len(group) for group in layout.groups] == [6] * 7
    assert [column for group in layout.groups for column in group] == list(
        range(42)
    )


def test_token_forest_uses_grouped_node_lookup() -> None:
    layout = token_forest.TokenLayout((4, 8, 12, 16), 3, 2, "right")
    items = _items(("a", "b") * 8)
    rows = np.random.default_rng(8).normal(size=(16, 24)).astype(np.float32)
    forest = token_forest.fit_token_forest(layout, rows, items, 7, 2)
    assert isinstance(forest, GroupedForest)
    assert all(len(columns) == 18 for columns in forest.columns)
    expected = np.mean(
        [
            tree.predict_proba(rows[:, columns])[:, 1]
            for tree, columns in zip(forest.trees, forest.columns, strict=True)
        ],
        axis=0,
    )
    np.testing.assert_array_equal(forest.score(rows), expected)


def test_length_scores_count_the_full_span_with_training_orientation(
    tiny_gpt2: LoadedModel,
) -> None:
    items = _items(("a cat.", "a cat and dog and fox."))
    lengths = token_forest.span_lengths(
        tiny_gpt2, items, bos=None, batch_size=1
    )
    assert 0 < lengths[0] < lengths[1]
    scorer = token_forest.fit_length_control(items, lengths)
    np.testing.assert_array_equal(scorer(np.array([2, 200])), [-2, -200])
    assert token_forest.fit_length_control(items, lengths[::-1])(
        np.array([2, 200])
    ).tolist() == [2, 200]


@pytest.mark.parametrize("bad_span", [None, (1000, 1001)])
def test_missing_start_fails_before_a_forward(
    tiny_gpt2: LoadedModel,
    monkeypatch: pytest.MonkeyPatch,
    bad_span: tuple[int, int] | None,
) -> None:
    items = _items(("a cat.", "a dog."))
    broken = dataclasses.replace(
        items, response_spans=(items.response_spans[0], bad_span)
    )

    def forward(*args: object, **kwargs: object) -> Activations:
        pytest.fail("a forward ran before span validation")

    monkeypatch.setattr(tiny_gpt2.decoder, "forward", forward)
    with pytest.raises(ValueError, match=r"stimulus 1 .*stimulus start"):
        token_forest.score_splits(
            tiny_gpt2,
            (items,),
            items,
            (broken,),
            (1,),
            bos=None,
            batch_size=1,
            alignment="right",
            n_trees=1,
            seeds=(0,),
        )


def test_streamed_scores_use_each_draws_training_width(
    tiny_gpt2: LoadedModel,
) -> None:
    short = _items(("a cat.", "a dog."))
    long = _items(("a cat and fox.", "a dog and a very very long tail."))
    draws = token_forest.score_splits(
        tiny_gpt2,
        (short, long),
        short,
        (long,),
        (1,),
        bos="prepend",
        batch_size=1,
        alignment="right",
        n_trees=3,
        seeds=(2, 3),
    )
    for train, draw in zip((short, long), draws, strict=True):
        lengths = token_forest.span_lengths(
            tiny_gpt2, train, bos="prepend", batch_size=2
        )
        assert draw.width == max(lengths)
        assert draw.forest_report.shape == (1, 2)
        assert draw.forest_select.shape == (2,)
        assert np.all((draw.forest_report >= 0) & (draw.forest_report <= 1))
    assert draws[0].width < draws[1].width
    assert max(abs(draws[0].length_report[0])) > draws[0].width
