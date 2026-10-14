import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.reading.reader import RepReader


def small_reader() -> RepReader:
    return RepReader.from_arrays(
        "hand",
        [4, 9],
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 1.0], [0.0, 0.0]]),
        np.array([1.0, -1.0]),
    )


def small_h() -> NDArray[np.float32]:
    return np.array(
        [[[2, 3], [0, 0], [1, 5]], [[1, 2], [3, 4], [0, 1]]],
        dtype=np.float32,
    )


def test_score_applies_the_mean_the_direction_and_the_sign() -> None:
    scores = small_reader().score(small_h())

    assert scores.dtype == np.float32
    np.testing.assert_allclose(scores, [[1, -1, 0], [-2, -4, -1]])


def test_raw_score_leaves_out_the_sign() -> None:
    raw = small_reader().raw_score(small_h())

    assert raw.dtype == np.float32
    np.testing.assert_allclose(raw, [[1, -1, 0], [2, 4, 1]])


def test_from_arrays_stores_float32_unit_directions() -> None:
    reader = RepReader.from_arrays(
        "prompt_diff",
        (7,),
        np.array([[3.0, 4.0]]),
        np.array([[0.5, 0.5]]),
    )

    assert reader.method == "prompt_diff"
    assert reader.layers == (7,)
    assert reader.directions.dtype == np.float32
    np.testing.assert_allclose(reader.directions, [[0.6, 0.8]], atol=1e-6)
    assert reader.means.dtype == np.float32
    assert reader.signs.dtype == np.float32
    np.testing.assert_array_equal(reader.signs, [1.0])


def test_from_arrays_copies_its_inputs() -> None:
    directions = np.array([[1.0, 0.0]], dtype=np.float32)

    reader = RepReader.from_arrays(
        "copy", [0], directions, np.zeros((1, 2), dtype=np.float32)
    )
    directions[0, 0] = 5.0

    assert reader.directions[0, 0] == 1.0


@pytest.mark.parametrize(
    ("layers", "directions", "means", "signs", "message"),
    [
        ([0], np.ones(2), np.ones((1, 2)), None, "directions"),
        ([0], np.ones((1, 2)), np.ones((1, 3)), None, "means"),
        ([0, 1], np.ones((1, 2)), np.ones((1, 2)), None, "layers"),
        ([0], np.ones((1, 2)), np.ones((1, 2)), np.array([0.5]), "signs"),
        ([0], np.ones((1, 2)), np.ones((1, 2)), np.ones(2), "signs"),
        ([0], np.zeros((1, 2)), np.ones((1, 2)), None, "zero"),
    ],
    ids=[
        "directions-rank",
        "means-shape",
        "layers-length",
        "sign-value",
        "signs-shape",
        "zero-direction",
    ],
)
def test_from_arrays_rejects_inconsistent_arrays(
    layers: list[int],
    directions: NDArray[np.float64],
    means: NDArray[np.float64],
    signs: NDArray[np.float64] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RepReader.from_arrays("bad", layers, directions, means, signs)


@pytest.mark.parametrize(
    "shape",
    [(3, 2), (1, 3, 2), (2, 3, 5)],
    ids=["two-dimensional", "wrong-layer-count", "wrong-width"],
)
def test_score_rejects_activations_of_the_wrong_shape(
    shape: tuple[int, ...],
) -> None:
    reader = small_reader()

    with pytest.raises(ValueError, match="shape"):
        reader.score(np.zeros(shape, dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        reader.raw_score(np.zeros(shape, dtype=np.float32))
