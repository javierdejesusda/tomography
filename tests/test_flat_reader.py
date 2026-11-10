"""Tests for the flat reader: a linear reader over the aligned cell matrix."""

from collections.abc import Sequence
import warnings

import numpy as np
from numpy.typing import NDArray
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from tomography.activations import Activations
from tomography.reading.flat import FlatReader
from tomography.reading.flat import fit_flat
from tomography.reading.probes import StalledFitWarning

N_SEGMENTS = 3
LAYERS = (2, 5)
WIDTH = 3
TRAIN_COUNTS = (
    (2, 3, 1),
    (1, 5, 1),
    (0, 2, 2),
    (2, 1, 1),
    (1, 4, 1),
    (2, 2, 1),
    (1, 1, 2),
    (0, 3, 1),
    (2, 2, 2),
    (1, 5, 1),
    (2, 1, 1),
    (1, 2, 2),
)
TRAIN_LABELS = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
# The widths the training counts give: each segment's longest.
TRAIN_WIDTHS = (2, 5, 2)


def _acts(
    counts: Sequence[Sequence[int]],
    *,
    seed: int = 0,
    layers: tuple[int, ...] = LAYERS,
) -> tuple[Activations, NDArray[np.int8]]:
    """Random cells of stimuli with counts[i][s] tokens in segment s.

    Each stimulus's segments follow in code order, as in a prompt, and the
    cells sit off zero with unequal spreads per layer, as real ones do.
    """
    codes = [
        np.repeat(np.arange(len(row)), row).astype(np.int8) for row in counts
    ]
    sizes = np.array([c.size for c in codes])
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(len(layers), int(sizes.sum()), WIDTH))
    data = data * np.arange(1, len(layers) + 1)[:, None, None] + 3.0
    acts = Activations(
        data=data.astype(np.float32),
        layers=layers,
        position="response",
        row_stimulus=np.repeat(np.arange(len(counts)), sizes),
        row_rank=np.concatenate([np.arange(n) for n in sizes]),
        row_token=np.concatenate([np.arange(1, n + 1) for n in sizes]),
        row_token_id=np.zeros(int(sizes.sum()), dtype=np.int64),
        n_stimuli=len(counts),
    )
    return acts, np.concatenate(codes)


def _scaling(
    acts: Activations,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Each layer's mean row and root mean square norm of centered rows."""
    rows = acts.data.astype(np.float64)
    centers = rows.mean(axis=1)
    centered = rows - centers[:, None]
    scales = np.sqrt(np.square(centered).sum(axis=(1, 2)) / rows.shape[1])
    return centers, scales


def _dense(
    acts: Activations,
    segments: NDArray[np.int8],
    widths: Sequence[int],
    scaling: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """The flat matrix by the documented rule, one text at a time.

    Each segment gets `widths[s]` slots, in code order; its last token takes
    its last slot, the tokens before it the slots before; a token with no
    slot left is dropped, and an empty slot holds 0, the training mean once
    standardized.
    """
    centers, scales = scaling
    n_layers, _, width = acts.data.shape
    offsets = np.cumsum(widths) - np.array(widths)
    flat = np.zeros((acts.n_stimuli, n_layers, int(sum(widths)), width))
    for i in range(acts.n_stimuli):
        for s, slots in enumerate(widths):
            rows = np.flatnonzero((acts.row_stimulus == i) & (segments == s))
            kept = rows[max(rows.size - slots, 0) :]
            for j, row in enumerate(kept):
                slot = offsets[s] + slots - kept.size + j
                cells = acts.data[:, row].astype(np.float64)
                flat[i, :, slot] = (cells - centers) / scales[:, None]
    return flat.reshape(acts.n_stimuli, -1)


def _fit(
    acts: Activations, segments: NDArray[np.int8], **kwargs: float
) -> FlatReader:
    return fit_flat(
        acts,
        segments=segments,
        labels=TRAIN_LABELS,
        n_segments=N_SEGMENTS,
        **kwargs,
    )


def test_the_widths_are_each_segments_longest_training_run() -> None:
    acts, segments = _acts(TRAIN_COUNTS)

    reader = fit_flat(
        acts, segments=segments, labels=TRAIN_LABELS, n_segments=4
    )

    assert reader.widths.tolist() == [*TRAIN_WIDTHS, 0]
    assert reader.layers == LAYERS
    assert reader.values.shape == (len(LAYERS), sum(TRAIN_WIDTHS), WIDTH)
    assert reader.shifts.shape == (len(LAYERS), sum(TRAIN_WIDTHS))
    assert reader.method == "flat"


def _by_hand(
    reader: FlatReader, acts: Activations, segments: NDArray[np.int8]
) -> NDArray[np.float64]:
    """Scores each text cell by cell at its right-aligned slot."""
    widths = reader.widths.tolist()
    offsets = np.cumsum(widths) - np.array(widths)
    scores = np.full(acts.n_stimuli, reader.bias)
    for i in range(acts.n_stimuli):
        for s, slots in enumerate(widths):
            rows = np.flatnonzero((acts.row_stimulus == i) & (segments == s))
            for j, row in enumerate(rows[::-1][:slots]):
                slot = offsets[s] + slots - 1 - j
                for k in range(len(reader.layers)):
                    cell = acts.data[k, row].astype(np.float64)
                    scores[i] += (
                        reader.values[k, slot] @ cell - reader.shifts[k, slot]
                    )
    return scores


def test_each_cell_is_read_at_its_slot_counted_from_its_segments_end() -> None:
    rng = np.random.default_rng(1)
    widths = np.array([2, 4, 0, 1])
    size = int(widths.sum())
    reader = FlatReader(
        layers=LAYERS,
        widths=widths,
        values=rng.normal(size=(len(LAYERS), size, WIDTH)),
        shifts=rng.normal(size=(len(LAYERS), size)),
        bias=0.25,
    )
    # Shorter, equal and longer runs than the widths, and a segment with
    # no slots.
    acts, segments = _acts([(1, 4, 0, 1), (3, 2, 2, 0), (0, 6, 1, 2)])

    scores = reader.score(acts, segments)

    assert scores.shape == (1, 3)
    assert scores.dtype == np.float32
    np.testing.assert_allclose(
        scores[0], _by_hand(reader, acts, segments), rtol=1e-6
    )


def test_an_empty_slot_reads_the_training_mean() -> None:
    train, segments = _acts(TRAIN_COUNTS)
    reader = _fit(train, segments)
    centers = _scaling(train)[0]
    # The second text has one token of segment 0, so slot 0 is empty; a
    # first token equal to each layer's training mean fills it.
    one = np.flatnonzero(train.row_stimulus == 1)
    filled = np.concatenate([centers[:, None], train.data[:, one]], axis=1)
    acts = Activations(
        data=filled.astype(np.float32),
        layers=LAYERS,
        position="response",
        row_stimulus=np.zeros(one.size + 1, dtype=np.int64),
        row_rank=np.arange(one.size + 1),
        row_token=np.arange(1, one.size + 2),
        row_token_id=np.zeros(one.size + 1, dtype=np.int64),
        n_stimuli=1,
    )
    alone = Activations(
        data=train.data[:, one],
        layers=LAYERS,
        position="response",
        row_stimulus=np.zeros(one.size, dtype=np.int64),
        row_rank=np.arange(one.size),
        row_token=np.arange(1, one.size + 1),
        row_token_id=np.zeros(one.size, dtype=np.int64),
        n_stimuli=1,
    )

    with_mean = reader.score(acts, np.concatenate([[0], segments[one]]))
    without = reader.score(alone, segments[one])

    assert reader.dropped(acts, np.concatenate([[0], segments[one]])).sum() == 0
    # Only the float32 rounding of the mean separates them.
    np.testing.assert_allclose(with_mean, without, rtol=1e-5, atol=1e-5)


def _first_cut(
    acts: Activations, segments: NDArray[np.int8], cut: dict[int, int]
) -> tuple[Activations, NDArray[np.int8]]:
    """Removes the first `cut[s]` tokens of segment s from every text."""
    keep = np.ones(segments.size, dtype=bool)
    for i in range(acts.n_stimuli):
        for s, n in cut.items():
            rows = np.flatnonzero((acts.row_stimulus == i) & (segments == s))
            keep[rows[:n]] = False
    stimulus = acts.row_stimulus[keep]
    counts = np.bincount(stimulus, minlength=acts.n_stimuli)
    return (
        Activations(
            data=acts.data[:, keep],
            layers=acts.layers,
            position=acts.position,
            row_stimulus=stimulus,
            row_rank=np.concatenate([np.arange(n) for n in counts]),
            row_token=acts.row_token[keep],
            row_token_id=acts.row_token_id[keep],
            n_stimuli=acts.n_stimuli,
        ),
        segments[keep],
    )


def test_a_longer_segment_loses_its_first_cells() -> None:
    train, segments = _acts(TRAIN_COUNTS)
    reader = _fit(train, segments)
    # Every text runs 3 tokens past the width of segment 1.
    longer = [(1, TRAIN_WIDTHS[1] + 3, 1), (2, TRAIN_WIDTHS[1] + 3, 2)]
    acts, codes = _acts(longer, seed=7)

    dropped = reader.dropped(acts, codes)
    cut, cut_codes = _first_cut(acts, codes, {1: 3})

    np.testing.assert_array_equal(
        dropped, [[0, 3 * len(LAYERS), 0], [0, 3 * len(LAYERS), 0]]
    )
    np.testing.assert_array_equal(
        reader.score(acts, codes), reader.score(cut, cut_codes)
    )
    assert reader.dropped(cut, cut_codes).sum() == 0


def test_a_segment_without_training_cells_has_no_weights() -> None:
    # No training text has a token of segment 1, the decoy line of m3.
    counts = [(a, 0, c) for a, _, c in TRAIN_COUNTS]
    train, segments = _acts(counts)
    reader = _fit(train, segments)
    acts, codes = _acts([(1, 2, 1), (2, 4, 2), (0, 1, 1)], seed=3)

    without, without_codes = _first_cut(acts, codes, {1: 4})

    assert reader.widths.tolist() == [2, 0, 2]
    assert reader.values.shape[1] == 4
    np.testing.assert_array_equal(
        reader.score(acts, codes), reader.score(without, without_codes)
    )
    np.testing.assert_array_equal(
        reader.dropped(acts, codes)[:, 1], np.array([2, 4, 1]) * len(LAYERS)
    )


@pytest.mark.parametrize("c", [0.01, 1.0, 100.0])
def test_the_dual_fit_equals_scikit_learns_primal_fit(c: float) -> None:
    train, segments = _acts(TRAIN_COUNTS)
    scaling = _scaling(train)
    flat = _dense(train, segments, TRAIN_WIDTHS, scaling)
    # More inputs than texts, as in the real runs.
    assert flat.shape[1] > flat.shape[0]
    primal = LogisticRegression(C=c, tol=1e-12, max_iter=100_000).fit(
        flat, TRAIN_LABELS
    )
    report, codes = _acts([(2, 6, 1), (1, 2, 3), (0, 5, 2)], seed=5)
    report_flat = _dense(report, codes, TRAIN_WIDTHS, scaling)

    reader = _fit(train, segments, c=c, tol=1e-10)

    np.testing.assert_allclose(
        reader.score(train, segments)[0],
        primal.decision_function(flat),
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        reader.score(report, codes)[0],
        primal.decision_function(report_flat),
        rtol=1e-4,
        atol=1e-5,
    )
    # The weights, in standardized units, are scikit-learn's.
    standardized = reader.values * scaling[1][:, None, None]
    np.testing.assert_allclose(
        standardized.reshape(-1),
        primal.coef_[0],
        rtol=1e-4,
        atol=1e-5 * np.abs(primal.coef_).max(),
    )


def test_a_refit_is_bitwise_identical() -> None:
    train, segments = _acts(TRAIN_COUNTS)

    first = _fit(train, segments, c=10.0, tol=1e-10)
    second = _fit(train, segments, c=10.0, tol=1e-10)

    assert first.bias == second.bias
    np.testing.assert_array_equal(first.widths, second.widths)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.shifts, second.shifts)


def test_a_fit_cut_short_warns_that_it_did_not_converge() -> None:
    train, segments = _acts(TRAIN_COUNTS)

    with pytest.warns(ConvergenceWarning, match=r"^the flat reader did not"):
        _fit(train, segments, c=100.0, tol=1e-10, max_iter=1)


def test_a_fit_that_cannot_leave_its_zero_start_warns_that_it_stalled() -> None:
    # Two pairs of texts with equal cells and opposite labels: at zero the
    # gradient of every weight and of the bias is exactly zero.
    acts, segments = _acts([(1, 2, 1), (1, 2, 1)])
    # Text 0 twice, then text 1 twice.
    order = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7]
    twice = Activations(
        data=acts.data[:, order],
        layers=LAYERS,
        position="response",
        row_stimulus=np.repeat(np.arange(4), 4),
        row_rank=np.tile(np.arange(4), 4),
        row_token=np.tile(np.arange(1, 5), 4),
        row_token_id=np.zeros(16, dtype=np.int64),
        n_stimuli=4,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        with pytest.warns(StalledFitWarning, match=r"^the flat reader stalled"):
            fit_flat(
                twice,
                segments=np.tile(segments[:4], 4),
                labels=np.array([1, 0, 1, 0]),
                n_segments=N_SEGMENTS,
            )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("short", r"^segments must have one code per row \(21\), got 20$"),
        ("code", r"^segment codes must be in \[0, 3\), got 3$"),
        ("negative", r"^segment codes must be in \[0, 3\), got -1$"),
    ],
)
def test_a_fit_refuses_malformed_segments(change: str, message: str) -> None:
    acts, segments = _acts(TRAIN_COUNTS[:4])
    codes = segments.copy()
    if change == "short":
        codes = codes[:-1]
    else:
        codes[0] = 3 if change == "code" else -1

    with pytest.raises(ValueError, match=message):
        fit_flat(
            acts,
            segments=codes,
            labels=TRAIN_LABELS[:4],
            n_segments=N_SEGMENTS,
        )


def test_a_fit_refuses_a_nonpositive_c_and_one_class() -> None:
    acts, segments = _acts(TRAIN_COUNTS[:4])

    with pytest.raises(ValueError, match=r"^c must be positive, got 0.0$"):
        _fit_four(acts, segments, TRAIN_LABELS[:4], c=0.0)
    with pytest.raises(ValueError, match=r"^labels must contain both classes"):
        _fit_four(acts, segments, np.ones(4, dtype=np.int64))


def _fit_four(
    acts: Activations,
    segments: NDArray[np.int8],
    labels: NDArray[np.int64],
    c: float = 1.0,
) -> FlatReader:
    return fit_flat(
        acts, segments=segments, labels=labels, n_segments=N_SEGMENTS, c=c
    )


def test_a_score_refuses_other_layers_and_malformed_segments() -> None:
    train, segments = _acts(TRAIN_COUNTS)
    reader = _fit(train, segments)
    other, codes = _acts([(1, 1, 1)], layers=(2, 6))
    acts, codes = _acts([(1, 1, 1)])

    with pytest.raises(ValueError, match=r"^acts must hold the layers"):
        reader.score(other, codes)
    with pytest.raises(ValueError, match=r"one code per row \(3\), got 2$"):
        reader.score(acts, codes[:2])
    with pytest.raises(ValueError, match=r"must be in \[0, 3\), got 5$"):
        reader.dropped(acts, np.array([0, 1, 5], dtype=np.int8))
