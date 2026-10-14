from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
import pytest

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
