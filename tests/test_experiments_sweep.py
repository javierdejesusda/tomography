"""Tests for the shared reading sweep."""

from collections.abc import Mapping
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.activations import Activations
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import Prepared
from tomography.experiments.sweep import shuffle_within_groups
from tomography.experiments.sweep import sweep_from_activations

DIM = 64
LAYERS = (1, 2)


def _planted(
    n_pairs: int,
    *,
    seed: int,
    layers: Sequence[int] = LAYERS,
    offset: float = 3.0,
) -> Prepared:
    """Pairs whose label-1 member sits `offset` along a fixed direction.

    The in-pair order is random, so the mean difference cancels and only the
    direction of the differences carries the signal.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_pairs
    labels = np.zeros(n, dtype=np.int64)
    first_is_positive = rng.integers(0, 2, size=n_pairs).astype(bool)
    labels[2 * np.arange(n_pairs)] = first_is_positive
    labels[2 * np.arange(n_pairs) + 1] = ~first_is_positive
    direction = np.random.default_rng(1234).normal(size=(len(layers), DIM))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    signed = (2 * labels - 1)[None, :, None] * offset
    data = rng.normal(size=(len(layers), n, DIM)) + signed * direction[:, None]
    pairs = _pair_groups(n_pairs)
    stimuli = StimulusSet(
        texts=tuple(f"text {i}" for i in range(n)),
        response_spans=(None,) * n,
        concept_spans=(None,) * n,
        labels=labels,
        pairs=pairs,
        groups=pairs.copy(),
        uids=tuple(f"u{i // 2}" for i in range(n)),
    )
    act = Activations(
        data=data.astype(np.float32),
        layers=tuple(layers),
        position="last",
        row_stimulus=np.arange(n, dtype=np.int64),
        row_rank=np.zeros(n, dtype=np.int64),
        row_token=np.zeros(n, dtype=np.int64),
        row_token_id=np.zeros(n, dtype=np.int64),
        n_stimuli=n,
    )
    return stimuli, act


def _inputs(
    n_train: int = 3,
) -> tuple[list[Prepared], Mapping[str, Prepared]]:
    train = [_planted(40, seed=10 + i) for i in range(n_train)]
    evals = {
        "val": _planted(200, seed=95),
        "other": _planted(50, seed=98),
    }
    return train, evals


def _pair_groups(n_pairs: int) -> NDArray[np.int64]:
    return np.arange(2 * n_pairs, dtype=np.int64).reshape(n_pairs, 2)


def _pair_labels(n_pairs: int, seed: int) -> NDArray[np.int64]:
    first = np.random.default_rng(seed).integers(0, 2, size=n_pairs)
    labels = np.zeros(2 * n_pairs, dtype=np.int64)
    labels[2 * np.arange(n_pairs) + (1 - first)] = 1
    return labels


def test_shuffle_keeps_one_positive_per_group() -> None:
    groups = _pair_groups(30)
    labels = _pair_labels(30, seed=1)
    shuffled = shuffle_within_groups(labels, groups, seed=3)
    assert shuffled.dtype == np.int64
    assert shuffled.shape == labels.shape
    assert (shuffled[groups].sum(axis=1) == 1).all()


def test_shuffle_handles_ragged_groups_and_keeps_other_rows() -> None:
    groups = np.array([[0, 1, 2], [4, 5, -1]], dtype=np.int64)
    labels = np.array([1, 0, 0, 1, 0, 1, 1], dtype=np.int64)
    shuffled = shuffle_within_groups(labels, groups, seed=0)
    assert shuffled[[0, 1, 2]].sum() == 1
    assert shuffled[[4, 5]].sum() == 1
    assert shuffled[3] == 1
    assert shuffled[6] == 1


def test_shuffle_ignores_padding_anywhere_in_a_group() -> None:
    groups = np.array([[-1, 0, 1], [3, -1, 2]], dtype=np.int64)
    labels = np.array([1, 0, 0, 1], dtype=np.int64)
    for seed in range(20):
        shuffled = shuffle_within_groups(labels, groups, seed=seed)
        assert shuffled[[0, 1]].sum() == 1
        assert shuffled[[2, 3]].sum() == 1


def test_shuffle_is_deterministic_per_seed_and_varies_across_seeds() -> None:
    groups = _pair_groups(40)
    labels = _pair_labels(40, seed=2)
    first = shuffle_within_groups(labels, groups, seed=5)
    again = shuffle_within_groups(labels, groups, seed=5)
    other = shuffle_within_groups(labels, groups, seed=6)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_shuffle_does_not_modify_its_input() -> None:
    groups = _pair_groups(10)
    labels = _pair_labels(10, seed=0)
    before = labels.copy()
    shuffle_within_groups(labels, groups, seed=1)
    np.testing.assert_array_equal(labels, before)


def test_shuffle_positive_position_is_uniform() -> None:
    groups = np.tile(np.arange(4, dtype=np.int64), (400, 1)) + 4 * np.arange(
        400, dtype=np.int64
    ).reshape(-1, 1)
    labels = np.zeros(1600, dtype=np.int64)
    labels[groups[:, 0]] = 1
    shuffled = shuffle_within_groups(labels, groups, seed=0)
    counts = np.bincount(shuffled[groups].argmax(axis=1), minlength=4)
    assert (np.abs(counts / 400 - 0.25) < 0.08).all()


def test_shuffle_rejects_groups_sharing_a_row() -> None:
    groups = np.array([[0, 1], [1, 2]], dtype=np.int64)
    labels = np.array([1, 0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="share"):
        shuffle_within_groups(labels, groups, seed=0)


def test_shuffle_rejects_non_binary_labels() -> None:
    groups = np.array([[0, 1]], dtype=np.int64)
    labels = np.array([1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="binary"):
        shuffle_within_groups(labels, groups, seed=0)


def test_shuffle_rejects_group_without_one_positive() -> None:
    groups = np.array([[0, 1]], dtype=np.int64)
    labels = np.array([1, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="exactly one"):
        shuffle_within_groups(labels, groups, seed=0)


def test_sweep_shapes_and_eval_order() -> None:
    train, evals = _inputs()
    result = sweep_from_activations(train, evals, seed=0)
    assert result.layers == LAYERS
    assert result.eval_names == ("val", "other")
    assert result.accuracy.shape == (2, len(SERIES), 3, len(LAYERS))
    assert len(result.readers) == 3
    assert all(reader.layers == LAYERS for reader in result.readers)
    per_seed = result.per_seed("other")
    assert per_seed.shape == (len(SERIES), 3, len(LAYERS))
    np.testing.assert_array_equal(per_seed, result.accuracy[1])


def test_sweep_per_seed_rejects_an_unknown_name() -> None:
    train, evals = _inputs(1)
    result = sweep_from_activations(train, evals, seed=0)
    with pytest.raises(ValueError, match="nope"):
        result.per_seed("nope")


def test_sweep_reader_recovers_a_planted_direction() -> None:
    train, evals = _inputs()
    result = sweep_from_activations(train, evals, seed=0)
    reader = result.per_seed("val")[SERIES.index("reader")]
    np.testing.assert_array_equal(reader, np.ones_like(reader))


def test_sweep_shuffled_series_sits_at_chance() -> None:
    train, evals = _inputs()
    result = sweep_from_activations(train, evals, seed=0)
    shuffled = result.per_seed("val")[SERIES.index("shuffled")]
    assert np.abs(shuffled - 0.5).max() < 0.08


def test_sweep_random_series_is_below_the_reader() -> None:
    train, evals = _inputs()
    result = sweep_from_activations(train, evals, seed=0)
    per_seed = result.per_seed("val")
    assert (per_seed[SERIES.index("random")] < per_seed[0]).all()


def test_sweep_is_deterministic_per_seed() -> None:
    train, evals = _inputs()
    first = sweep_from_activations(train, evals, seed=4)
    again = sweep_from_activations(train, evals, seed=4)
    other = sweep_from_activations(train, evals, seed=5)
    np.testing.assert_array_equal(first.accuracy, again.accuracy)
    assert not np.array_equal(first.accuracy, other.accuracy)


def test_sweep_rejects_empty_train_and_evals() -> None:
    train, evals = _inputs(1)
    with pytest.raises(ValueError, match="train"):
        sweep_from_activations([], evals, seed=0)
    with pytest.raises(ValueError, match="evals"):
        sweep_from_activations(train, {}, seed=0)


def test_sweep_rejects_mismatching_layers() -> None:
    train, evals = _inputs(1)
    odd = {"val": _planted(20, seed=3, layers=(1, 3))}
    with pytest.raises(ValueError, match="layers"):
        sweep_from_activations(train, odd, seed=0)
    with pytest.raises(ValueError, match="layers"):
        sweep_from_activations(
            [train[0], _planted(20, seed=4, layers=(1, 3))], evals, seed=0
        )
