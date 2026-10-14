from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.reading.directions import Method
from tomography.reading.directions import fit_direction


class Planted(NamedTuple):
    h: NDArray[np.float32]
    pairs: NDArray[np.int64]
    labels: NDArray[np.int64]
    direction: NDArray[np.float64]


def make_planted(
    n_pairs: int,
    d: int = 32,
    *,
    seed: int = 0,
    signal: float = 3.0,
    noise: float = 1.0,
    random_order: bool = True,
) -> Planted:
    """Builds pairs whose members differ along one planted unit direction.

    Rows 2i and 2i + 1 form pair i with a random in-pair order, so the mean
    difference is near zero as in the reference datasets.
    """
    rng = np.random.default_rng(seed)
    planted = rng.standard_normal(d)
    planted /= np.linalg.norm(planted)
    base = 2.0 * rng.standard_normal((n_pairs, d))
    pos = base + signal * planted + noise * rng.standard_normal((n_pairs, d))
    neg = base - signal * planted + noise * rng.standard_normal((n_pairs, d))
    pos_first = (
        rng.random(n_pairs) < 0.5 if random_order else np.ones(n_pairs, bool)
    )
    h = np.empty((2 * n_pairs, d))
    h[0::2] = np.where(pos_first[:, None], pos, neg)
    h[1::2] = np.where(pos_first[:, None], neg, pos)
    h += 10.0 * rng.standard_normal(d)
    labels = np.empty(2 * n_pairs, dtype=np.int64)
    labels[0::2] = pos_first
    labels[1::2] = ~pos_first
    pairs = np.stack(
        [np.arange(0, 2 * n_pairs, 2), np.arange(1, 2 * n_pairs, 2)], axis=1
    ).astype(np.int64)
    return Planted(h.astype(np.float32), pairs, labels, planted)


def abs_cosine(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    return float(abs(a64 @ b64) / (np.linalg.norm(a64) * np.linalg.norm(b64)))


def test_pca_diff_recovers_planted_direction() -> None:
    data = make_planted(128, seed=1)

    fit = fit_direction("pca_diff", data.h, pairs=data.pairs)

    assert abs_cosine(fit.direction, data.direction) > 0.95


def test_pca_diff_is_the_top_singular_vector_of_centered_differences() -> None:
    data = make_planted(40, seed=2)
    h64 = data.h.astype(np.float64)
    diffs = h64[data.pairs[:, 0]] - h64[data.pairs[:, 1]]
    centered = diffs - diffs.mean(axis=0)
    top = np.linalg.svd(centered, full_matrices=False)[2][0]

    fit = fit_direction("pca_diff", data.h, pairs=data.pairs)

    assert abs_cosine(fit.direction, top) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(fit.mean, diffs.mean(axis=0), atol=1e-5)
    assert fit.raw_norm == pytest.approx(1.0)


def test_direction_is_unit_float32_with_largest_coordinate_positive() -> None:
    data = make_planted(40, seed=3)

    fit = fit_direction("pca_diff", data.h, pairs=data.pairs)

    assert fit.direction.dtype == np.float32
    assert fit.direction.shape == (32,)
    assert fit.mean.dtype == np.float32
    assert fit.mean.shape == (32,)
    assert np.linalg.norm(fit.direction) == pytest.approx(1.0, abs=1e-6)
    assert fit.direction[np.argmax(np.abs(fit.direction))] > 0


def test_pca_diff_ignores_seed_and_in_pair_order() -> None:
    data = make_planted(40, seed=4)
    swapped = data.pairs[:, ::-1].copy()

    first = fit_direction("pca_diff", data.h, pairs=data.pairs, seed=0)
    second = fit_direction("pca_diff", data.h, pairs=data.pairs, seed=99)
    flipped = fit_direction("pca_diff", data.h, pairs=swapped)

    np.testing.assert_array_equal(first.direction, second.direction)
    np.testing.assert_allclose(first.direction, flipped.direction, atol=1e-6)


def test_pca_diff_requires_pairs() -> None:
    data = make_planted(10, seed=5)

    with pytest.raises(ValueError, match="pairs"):
        fit_direction("pca_diff", data.h)


def pair_differences(data: Planted) -> NDArray[np.float64]:
    h64 = data.h.astype(np.float64)
    return h64[data.pairs[:, 0]] - h64[data.pairs[:, 1]]


def test_preprocess_none_keeps_the_mean_difference_in_the_fit() -> None:
    data = make_planted(60, seed=6, random_order=False)
    diffs = pair_differences(data)
    top = np.linalg.svd(diffs, full_matrices=False)[2][0]

    centered = fit_direction("pca_diff", data.h, pairs=data.pairs)
    raw = fit_direction("pca_diff", data.h, pairs=data.pairs, preprocess="none")

    assert abs_cosine(raw.direction, top) == pytest.approx(1.0, abs=1e-6)
    assert abs_cosine(raw.direction, data.direction) > 0.95
    assert abs_cosine(centered.direction, data.direction) < 0.6
    np.testing.assert_allclose(raw.mean, diffs.mean(axis=0), atol=1e-5)


def test_preprocess_unit_scales_each_difference_before_centering() -> None:
    data = make_planted(40, seed=7)
    diffs = pair_differences(data)
    unit_diffs = diffs / np.linalg.norm(diffs, axis=1, keepdims=True)
    centered = unit_diffs - unit_diffs.mean(axis=0)
    top = np.linalg.svd(centered, full_matrices=False)[2][0]

    fit = fit_direction("pca_diff", data.h, pairs=data.pairs, preprocess="unit")

    assert abs_cosine(fit.direction, top) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(fit.mean, unit_diffs.mean(axis=0), atol=1e-6)


def test_preprocess_unit_resists_one_outlier_difference() -> None:
    data = make_planted(60, seed=8)
    outlier = np.random.default_rng(80).standard_normal(32)
    h = data.h.copy()
    h[0] += (1000.0 * outlier / np.linalg.norm(outlier)).astype(np.float32)

    plain = fit_direction("pca_diff", h, pairs=data.pairs)
    unit = fit_direction("pca_diff", h, pairs=data.pairs, preprocess="unit")

    assert abs_cosine(plain.direction, data.direction) < 0.6
    assert abs_cosine(unit.direction, data.direction) > 0.95


def test_preprocess_unit_keeps_a_zero_difference_finite() -> None:
    data = make_planted(20, seed=9)
    h = data.h.copy()
    h[1] = h[0]

    fit = fit_direction("pca_diff", h, pairs=data.pairs, preprocess="unit")

    assert np.isfinite(fit.direction).all()
    assert np.isfinite(fit.mean).all()


def test_unknown_preprocess_is_rejected() -> None:
    data = make_planted(10, seed=10)

    with pytest.raises(ValueError, match="preprocess"):
        fit_direction(
            "pca_diff",
            data.h,
            pairs=data.pairs,
            preprocess="whiten",  # type: ignore[arg-type]
        )


def make_blobs(
    n: int, d: int = 32, *, seed: int = 0, signal: float = 3.0
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.float64]]:
    """Builds two unpaired classes separated along one planted direction."""
    rng = np.random.default_rng(seed)
    planted = rng.standard_normal(d)
    planted /= np.linalg.norm(planted)
    labels = (np.arange(n) % 2).astype(np.int64)
    sign = np.where(labels == 1, 1.0, -1.0)[:, None]
    h = signal * sign * planted + rng.standard_normal((n, d))
    h += 10.0 * rng.standard_normal(d)
    return h.astype(np.float32), labels, planted


def test_pca_raw_is_the_top_singular_vector_of_centered_rows() -> None:
    h, _, planted = make_blobs(200, seed=11)
    h64 = h.astype(np.float64)
    top = np.linalg.svd(h64 - h64.mean(axis=0), full_matrices=False)[2][0]

    fit = fit_direction("pca_raw", h)

    assert abs_cosine(fit.direction, top) == pytest.approx(1.0, abs=1e-6)
    assert abs_cosine(fit.direction, planted) > 0.95
    np.testing.assert_allclose(fit.mean, h64.mean(axis=0), atol=1e-4)


def test_mean_diff_matches_a_hand_computed_class_difference() -> None:
    h = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    labels = np.array([1, 1, 0, 0], dtype=np.int64)

    fit = fit_direction("mean_diff", h, labels=labels)

    expected = np.array([-2.0, 3.0]) / np.sqrt(13.0)
    np.testing.assert_allclose(fit.direction, expected, atol=1e-6)
    assert fit.raw_norm == pytest.approx(np.sqrt(13.0))
    np.testing.assert_allclose(fit.mean, [1.0, 1.5])


def test_mean_diff_recovers_planted_direction() -> None:
    h, labels, planted = make_blobs(200, seed=12)

    fit = fit_direction("mean_diff", h, labels=labels)

    assert abs_cosine(fit.direction, planted) > 0.95


def test_mean_diff_needs_labels_with_both_classes() -> None:
    h, labels, _ = make_blobs(20, seed=13)

    with pytest.raises(ValueError, match="labels"):
        fit_direction("mean_diff", h)
    with pytest.raises(ValueError, match="both classes"):
        fit_direction("mean_diff", h, labels=np.ones_like(labels))


def test_kmeans_is_the_centroid_difference_of_two_clusters() -> None:
    h = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0]])

    fit = fit_direction("kmeans", h)

    np.testing.assert_allclose(fit.direction, [1.0, 0.0], atol=1e-6)
    assert fit.raw_norm == pytest.approx(10.0)


def test_kmeans_recovers_planted_direction_without_labels() -> None:
    h, _, planted = make_blobs(200, seed=14, signal=4.0)

    first = fit_direction("kmeans", h, seed=3)
    second = fit_direction("kmeans", h, seed=3)

    assert abs_cosine(first.direction, planted) > 0.95
    np.testing.assert_array_equal(first.direction, second.direction)


def test_random_direction_is_a_seeded_gaussian_draw() -> None:
    h, _, _ = make_blobs(20, seed=15)
    draw = np.random.default_rng(5).standard_normal(32)

    fit = fit_direction("random", h, seed=5)
    other = fit_direction("random", h, seed=6)

    assert abs_cosine(fit.direction, draw) == pytest.approx(1.0, abs=1e-6)
    assert fit.raw_norm == pytest.approx(np.linalg.norm(draw))
    assert fit.direction[np.argmax(np.abs(fit.direction))] > 0
    assert abs_cosine(fit.direction, other.direction) < 0.9
    np.testing.assert_allclose(
        fit.mean, h.astype(np.float64).mean(axis=0), atol=1e-4
    )


def test_logistic_follows_the_axis_of_collinear_classes() -> None:
    h = np.array([[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]])
    labels = np.array([1, 1, 0, 0], dtype=np.int64)

    fit = fit_direction("logistic", h, labels=labels)
    flipped = fit_direction("logistic", h, labels=1 - labels)

    np.testing.assert_allclose(fit.direction, [0.70710678] * 2, atol=1e-5)
    np.testing.assert_allclose(flipped.direction, fit.direction, atol=1e-5)


def test_logistic_recovers_planted_direction_on_separable_data() -> None:
    h, labels, planted = make_blobs(60, seed=16, signal=12.0)

    fit = fit_direction("logistic", h, labels=labels)

    assert abs_cosine(fit.direction, planted) > 0.95


def test_logistic_direction_does_not_depend_on_activation_scale() -> None:
    h, labels, _ = make_blobs(120, seed=17)

    base = fit_direction("logistic", h, labels=labels)
    scaled = fit_direction("logistic", 1000.0 * h, labels=labels)

    np.testing.assert_allclose(scaled.direction, base.direction, atol=1e-4)
    assert scaled.raw_norm == pytest.approx(base.raw_norm / 1000.0, rel=1e-3)


def test_logistic_needs_labels_with_both_classes() -> None:
    h, labels, _ = make_blobs(20, seed=18)

    with pytest.raises(ValueError, match="labels"):
        fit_direction("logistic", h)
    with pytest.raises(ValueError, match="both classes"):
        fit_direction("logistic", h, labels=np.zeros_like(labels))


def test_unknown_method_is_rejected() -> None:
    h, _, _ = make_blobs(10, seed=19)

    with pytest.raises(ValueError, match="method"):
        fit_direction("lda", h)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("h_shape", "message"),
    [((8,), "shape"), ((1, 4), "at least 2"), ((8, 4, 1), "shape")],
    ids=["one-dimensional", "single-row", "three-dimensional"],
)
def test_activations_must_be_a_matrix_with_two_rows(
    h_shape: tuple[int, ...], message: str
) -> None:
    h = np.zeros(h_shape)

    with pytest.raises(ValueError, match=message):
        fit_direction("random", h)


def test_activations_must_be_finite() -> None:
    h, _, _ = make_blobs(10, seed=20)
    h[3, 4] = np.nan

    with pytest.raises(ValueError, match="finite"):
        fit_direction("pca_raw", h)


@pytest.mark.parametrize(
    "pairs",
    [
        np.array([0, 1], dtype=np.int64),
        np.array([[0, 1, 2]], dtype=np.int64),
        np.array([[0, 20]], dtype=np.int64),
        np.array([[-1, 2], [3, 4]], dtype=np.int64),
    ],
    ids=["one-dimensional", "three-columns", "out-of-range", "negative"],
)
def test_pairs_must_be_valid_row_indices(pairs: NDArray[np.int64]) -> None:
    h, _, _ = make_blobs(10, seed=21)

    with pytest.raises(ValueError, match="pairs"):
        fit_direction("pca_diff", h, pairs=pairs)


def test_pca_diff_needs_two_pairs() -> None:
    h, _, _ = make_blobs(10, seed=22)

    with pytest.raises(ValueError, match="at least 2"):
        fit_direction("pca_diff", h, pairs=np.array([[0, 1]], dtype=np.int64))


@pytest.mark.parametrize(
    "labels",
    [
        np.array([0, 1, 0], dtype=np.int64),
        np.array([0, 1, 2, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64),
    ],
    ids=["wrong-length", "not-binary"],
)
def test_labels_must_be_binary_and_match_the_rows(
    labels: NDArray[np.int64],
) -> None:
    h, _, _ = make_blobs(10, seed=23)

    with pytest.raises(ValueError, match="labels"):
        fit_direction("mean_diff", h, labels=labels)


@pytest.mark.parametrize("method", ["mean_diff", "logistic"])
def test_a_zero_length_direction_is_an_error(method: Method) -> None:
    h = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    labels = np.array([1, 0, 1, 0], dtype=np.int64)

    with pytest.raises(ValueError, match="zero norm"):
        fit_direction(method, h, labels=labels)
