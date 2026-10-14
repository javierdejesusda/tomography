"""Tests for the layer analyses: PCA variance and t-SNE."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.analysis import pca_explained_variance


def _planted(
    spectrum: NDArray[np.float64], n: int, d: int, seed: int = 0
) -> NDArray[np.float64]:
    """Returns (2, n, d) data whose centered rows have these singular values."""
    rng = np.random.default_rng(seed)
    layers = []
    for _ in range(2):
        raw = rng.standard_normal((n, d))
        raw -= raw.mean(axis=0)
        u, _ = np.linalg.qr(raw)
        v, _ = np.linalg.qr(rng.standard_normal((d, d)))
        offset = rng.standard_normal(d) * 5.0
        layers.append(u @ np.diag(spectrum) @ v.T + offset)
    return np.stack(layers)


def test_planted_spectrum_gives_the_planted_ratios() -> None:
    s = np.array([5.0, 3.0, 2.0, 1.0, 0.5])
    h = _planted(s, n=12, d=5)
    got = pca_explained_variance(h, n_components=5)
    assert got.shape == (2, 5)
    for layer in got:
        np.testing.assert_allclose(layer, s**2 / np.sum(s**2), atol=1e-9)


def test_random_ratios_are_ordered_and_bounded() -> None:
    h = np.random.default_rng(1).standard_normal((3, 10, 6))
    part = pca_explained_variance(h, n_components=3)
    assert (part >= 0).all()
    assert (np.diff(part, axis=1) <= 1e-12).all()
    assert (part.sum(axis=1) <= 1 + 1e-12).all()
    full = pca_explained_variance(h, n_components=6)
    np.testing.assert_allclose(full.sum(axis=1), 1.0, atol=1e-12)


def test_pairs_use_the_differences() -> None:
    h = np.random.default_rng(2).standard_normal((2, 12, 5))
    pairs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=np.int64)
    got = pca_explained_variance(h, pairs=pairs, n_components=4)
    diffs = h[:, pairs[:, 0]] - h[:, pairs[:, 1]]
    np.testing.assert_allclose(
        got, pca_explained_variance(diffs, n_components=4), atol=1e-12
    )
    assert not np.allclose(got, pca_explained_variance(h, n_components=4))


def test_unit_scales_rows_before_centering() -> None:
    rng = np.random.default_rng(3)
    h = rng.standard_normal((2, 9, 4)) * rng.uniform(0.5, 4.0, size=(2, 9, 1))
    unit = h / np.linalg.norm(h, axis=2, keepdims=True)
    np.testing.assert_allclose(
        pca_explained_variance(h, n_components=4, preprocess="unit"),
        pca_explained_variance(unit, n_components=4),
        atol=1e-12,
    )


def test_none_skips_the_centering() -> None:
    rng = np.random.default_rng(5)
    h = rng.standard_normal((2, 9, 4)) + 20.0
    got = pca_explained_variance(h, n_components=4, preprocess="none")
    for layer, row in zip(got, h, strict=True):
        s = np.linalg.svd(row, compute_uv=False)
        np.testing.assert_allclose(layer, s**2 / np.sum(s**2), atol=1e-12)
    assert not np.allclose(
        got, pca_explained_variance(h, n_components=4, preprocess="center")
    )


def _bad(name: str) -> dict[str, Any]:
    h = np.random.default_rng(6).standard_normal((2, 8, 4))
    good_pairs = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int64)
    cases: dict[str, dict[str, Any]] = {
        "too-many": {"h": h, "n_components": 5},
        "zero": {"h": h, "n_components": 0},
        "two-d": {"h": h[0], "n_components": 2},
        "nan": {"h": np.where(h > 1, np.nan, h), "n_components": 2},
        "pairs-shape": {
            "h": h,
            "pairs": np.zeros((3, 3), dtype=np.int64),
            "n_components": 2,
        },
        "pairs-range": {
            "h": h,
            "pairs": good_pairs + 9,
            "n_components": 2,
        },
        "preprocess": {"h": h, "preprocess": "bogus", "n_components": 2},
        "constant": {"h": np.ones((2, 8, 4)), "n_components": 2},
    }
    return cases[name]


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("too-many", "n_components"),
        ("zero", "n_components"),
        ("two-d", "shape"),
        ("nan", "finite"),
        ("pairs-shape", "pairs"),
        ("pairs-range", "pairs"),
        ("preprocess", "preprocess"),
        ("constant", "variance"),
    ],
    ids=[
        "too-many",
        "zero",
        "two-d",
        "nan",
        "pairs-shape",
        "pairs-range",
        "preprocess",
        "constant",
    ],
)
def test_invalid_inputs_raise(name: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        pca_explained_variance(**_bad(name))
