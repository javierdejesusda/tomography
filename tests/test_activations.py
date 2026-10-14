import numpy as np
import pytest

from tomography.activations import Activations


def make_activations(counts: list[int]) -> Activations:
    row_stimulus = np.repeat(np.arange(len(counts)), counts).astype(np.int64)
    row_rank = np.concatenate([np.arange(c) for c in counts]).astype(np.int64)
    n_rows = int(sum(counts))
    return Activations(
        data=np.zeros((1, n_rows, 2), dtype=np.float32),
        layers=(1,),
        position="response",
        row_stimulus=row_stimulus,
        row_rank=row_rank,
        row_token=np.arange(n_rows, dtype=np.int64),
        row_token_id=np.arange(n_rows, dtype=np.int64),
        n_stimuli=len(counts),
    )


def test_row_values_repeats_each_stimulus_value_for_its_rows() -> None:
    act = make_activations([2, 1, 3])
    labels = np.array([1, 0, 1])

    assert act.row_values(labels).tolist() == [1, 1, 0, 1, 1, 1]
    matrix = np.arange(6).reshape(3, 2)
    assert act.row_values(matrix).shape == (6, 2)


def test_row_values_rejects_a_wrong_length() -> None:
    with pytest.raises(ValueError, match="n_stimuli"):
        make_activations([1, 1]).row_values(np.array([1, 0, 1]))


def test_row_groups_is_the_identity_for_one_row_positions() -> None:
    act = make_activations([1, 1, 1, 1, 1])
    groups = np.array([[0, 1, 2], [3, 4, -1]], dtype=np.int64)

    assert act.row_groups(groups).tolist() == groups.tolist()


def test_row_groups_aligns_members_by_rank_and_keeps_padding() -> None:
    act = make_activations([2, 2, 2, 2])
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)
    padded = np.array([[0, 1, -1]], dtype=np.int64)

    assert act.row_groups(pairs).tolist() == [[0, 2], [1, 3], [4, 6], [5, 7]]
    assert act.row_groups(padded).tolist() == [[0, 2, -1], [1, 3, -1]]


def test_row_groups_rejects_or_truncates_unequal_row_counts() -> None:
    act = make_activations([3, 2, 1])
    groups = np.array([[0, 1]], dtype=np.int64)

    with pytest.raises(ValueError, match="group 0"):
        act.row_groups(groups)
    assert act.row_groups(groups, ragged="truncate").tolist() == [
        [0, 3],
        [1, 4],
    ]
    assert act.row_groups(
        np.array([[0, 2]], dtype=np.int64), ragged="truncate"
    ).tolist() == [[0, 5]]


def test_row_groups_rejects_malformed_groups() -> None:
    act = make_activations([1, 1])

    with pytest.raises(ValueError, match="2D"):
        act.row_groups(np.array([0, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="no members"):
        act.row_groups(np.array([[-1, -1]], dtype=np.int64))
    with pytest.raises(ValueError, match="n_stimuli"):
        act.row_groups(np.array([[0, 2]], dtype=np.int64))
    with pytest.raises(ValueError, match="ragged"):
        act.row_groups(np.array([[0, 1]], dtype=np.int64), ragged="pad")  # type: ignore[arg-type]
