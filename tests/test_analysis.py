"""Tests for the layer analyses: PCA variance and t-SNE."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.analysis import pca_explained_variance
from tomography.analysis import tsne_embed


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


def _clusters(
    n_per: int = 20, d: int = 16, seed: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    centers = np.zeros((2, d))
    centers[1, 0] = 8.0
    points = np.concatenate(
        [rng.standard_normal((n_per, d)) + c for c in centers]
    )
    return points, np.repeat([0, 1], n_per).astype(np.int64)


def test_tsne_shape_dtype_and_determinism() -> None:
    points, _ = _clusters()
    h = np.stack([points, points[::-1]])
    first = tsne_embed(h, perplexity=5.0, seed=0)
    assert first.shape == (2, 40, 2)
    assert first.dtype == np.float32
    again = tsne_embed(h, perplexity=5.0, seed=0)
    np.testing.assert_array_equal(first, again)


def test_tsne_keeps_separated_clusters_apart() -> None:
    points, cluster = _clusters()
    embedded = tsne_embed(points[None], perplexity=10.0)[0]
    gaps = np.linalg.norm(embedded[:, None] - embedded[None], axis=2)
    np.fill_diagonal(gaps, np.inf)
    nearest = gaps.argmin(axis=1)
    assert (cluster[nearest] == cluster).all()


def test_tsne_embeds_layers_independently() -> None:
    points, _ = _clusters()
    other = np.random.default_rng(9).standard_normal(points.shape)
    alone = tsne_embed(points[None], perplexity=5.0)
    stacked = tsne_embed(np.stack([points, other]), perplexity=5.0)
    np.testing.assert_array_equal(alone[0], stacked[0])


@pytest.mark.parametrize("perplexity", [20.0, 25.0], ids=["equal", "above"])
def test_tsne_rejects_perplexity_not_below_n(perplexity: float) -> None:
    h = np.random.default_rng(0).standard_normal((1, 20, 4))
    with pytest.raises(ValueError, match="below the 20 rows"):
        tsne_embed(h, perplexity=perplexity)


def test_tsne_rejects_bad_h() -> None:
    flat = np.random.default_rng(0).standard_normal((20, 4))
    with pytest.raises(ValueError, match="shape"):
        tsne_embed(flat, perplexity=5.0)
    bad = np.random.default_rng(0).standard_normal((1, 20, 4))
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        tsne_embed(bad, perplexity=5.0)
