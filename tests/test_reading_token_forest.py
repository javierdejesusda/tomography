"""Tests for literal token blocks and scores."""

import dataclasses

import numpy as np
import pytest
import torch

from tomography import model
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
        ("left", [[11, 12, 0], [21, 22, 23], [31, 32, 33]]),
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


@pytest.mark.parametrize("alignment", ["right", "left"])
def test_layout_places_each_layer_block_in_its_column_group(
    alignment: token_forest.Alignment,
) -> None:
    n_layers, depth, width = 3, 2, 3
    counts, lengths = np.array([4, 6]), np.array([2, 5])
    values = np.arange(n_layers * counts.sum() * depth, dtype=np.float32)
    cells = values.reshape(n_layers, counts.sum(), depth)
    layout = token_forest.TokenLayout((4, 8, 12), width, depth, alignment)
    packed = np.empty((2, layout.n_columns), dtype=np.float32)
    layout.pack(packed, cells, counts, lengths)
    # Prompt 0 is short and padded; prompt 1 keeps three of its five tokens.
    tokens = {
        "right": ([None, 2, 3], [7, 8, 9]),
        "left": ([2, 3, None], [5, 6, 7]),
    }[alignment]
    for row, picks in enumerate(tokens):
        for layer, group in enumerate(layout.groups):
            expected = np.concatenate(
                [
                    np.zeros(depth) if pick is None else cells[layer, pick]
                    for pick in picks
                ]
            )
            np.testing.assert_array_equal(packed[row, list(group)], expected)


def test_layout_rejects_an_unknown_alignment() -> None:
    with pytest.raises(ValueError, match="'middle'"):
        token_forest.TokenLayout((4,), 3, 1, "middle")  # type: ignore[arg-type]


def test_layout_rejects_a_span_longer_than_its_prompt() -> None:
    layout = token_forest.TokenLayout((4,), 3, 1, "right")
    cells = np.zeros((1, 5, 1), dtype=np.float32)
    packed = np.empty((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match=r"prompt 1 .* 4 .* 3"):
        layout.pack(packed, cells, np.array([2, 3]), np.array([2, 4]))


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


def _marked(texts: tuple[str, ...], stimulus: str) -> StimulusSet:
    """Items whose span is the last occurrence of `stimulus` in each text."""
    starts = [text.rindex(stimulus) for text in texts]
    return dataclasses.replace(
        _items(texts),
        response_spans=tuple(
            (start, start + len(stimulus)) for start in starts
        ),
    )


def test_span_lengths_skip_bos_when_the_stimulus_starts_the_text(
    tiny_gpt2: LoadedModel,
) -> None:
    texts = ("a cat sat.", "a dog ran far.")
    items = dataclasses.replace(
        _items(texts),
        response_spans=tuple((0, len(text)) for text in texts),
    )
    lengths = token_forest.span_lengths(
        tiny_gpt2, items, bos="prepend", batch_size=2
    )
    expected = tiny_gpt2.encode(texts, bos="none").lengths
    np.testing.assert_array_equal(lengths, expected)


def test_span_lengths_start_after_a_prefix_line_with_the_same_words(
    tiny_gpt2: LoadedModel,
) -> None:
    prefixes = ("Note: a cat sat.\nReview:", "Review:")
    texts = tuple(prefix + " a cat sat.\nAnswer:" for prefix in prefixes)
    items = _marked(texts, "a cat sat.")
    lengths = token_forest.span_lengths(
        tiny_gpt2, items, bos="none", batch_size=2
    )
    # The leading space belongs to the first stimulus token.
    whole = tiny_gpt2.encode(texts, bos="none").lengths
    before = tiny_gpt2.encode(prefixes, bos="none").lengths
    np.testing.assert_array_equal(lengths, np.subtract(whole, before))
    assert lengths[0] == lengths[1]


def test_length_control_keeps_a_positive_sign_on_a_tie() -> None:
    items = _items(("a cat.", "a dog.", "a fox.", "a hen."))
    scorer = token_forest.fit_length_control(items, np.array([3, 3, 5, 5]))
    np.testing.assert_array_equal(scorer(np.array([2, 7])), [2, 7])


def test_score_splits_rejects_an_unknown_alignment_before_tokenizing(
    tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = _items(("a cat.", "a dog."))

    def tokenize(*args: object, **kwargs: object) -> None:
        pytest.fail("tokenized before checking the alignment")

    monkeypatch.setattr(token_forest, "span_lengths", tokenize)
    with pytest.raises(ValueError, match="'middle'"):
        token_forest.score_splits(
            tiny_gpt2,
            (items,),
            items,
            (items,),
            (1,),
            bos=None,
            batch_size=1,
            alignment="middle",  # type: ignore[arg-type]
            n_trees=1,
            seeds=(0,),
        )


@pytest.mark.parametrize("bos", ["prepend", "none"])
def test_packed_rows_equal_hidden_states_from_the_stimulus_start(
    tiny_gpt2: LoadedModel, bos: model.BosMode
) -> None:
    # Mid-text stimuli after a decoy-like line, unequal lengths in one batch.
    texts = (
        "Note: a cat sat.\nReview: a cat sat on the mat.\nAnswer:",
        "Review: a dog ran.\nAnswer:",
    )
    stimuli = ("a cat sat on the mat.", "a dog ran.")
    items = dataclasses.replace(
        _items(texts),
        response_spans=tuple(
            (text.index(s), text.index(s) + len(s))
            for text, s in zip(texts, stimuli, strict=True)
        ),
    )
    # Below the top layer, whose hidden state is after the final norm.
    layers = (1, 2)
    lengths = token_forest.span_lengths(tiny_gpt2, items, bos=bos, batch_size=2)
    assert lengths[0] != lengths[1]
    layout = token_forest.TokenLayout(
        layers, int(lengths.max()), tiny_gpt2.d_model, "right"
    )
    packed = np.empty((2, layout.n_columns), dtype=np.float32)
    for span, cells, totals in token_forest._cell_batches(
        tiny_gpt2, items, layers, bos, 2
    ):
        layout.pack(packed[span], cells, totals, lengths[span])

    blocks = packed.reshape(2, len(layers), layout.width, tiny_gpt2.d_model)
    for row, (text, (span_start, _)) in enumerate(
        zip(texts, items.response_spans, strict=True)
    ):
        encoded = tiny_gpt2.encode([text], bos=bos)
        with torch.no_grad():
            hidden = tiny_gpt2.model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                output_hidden_states=True,
            ).hidden_states
        offsets = encoded.offsets[0]
        start = next(
            token
            for token, (lo, hi) in enumerate(offsets)
            if hi > span_start and hi > lo
        )
        size = encoded.lengths[0] - start
        assert size == lengths[row]
        for at, layer in enumerate(layers):
            expected = hidden[layer][0, start : encoded.lengths[0]].numpy()
            np.testing.assert_allclose(
                blocks[row, at, layout.width - size :], expected, atol=1e-5
            )
            assert not blocks[row, at, : layout.width - size].any()


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
