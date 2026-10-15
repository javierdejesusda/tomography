"""Tests for the layer analyses: PCA variance and t-SNE."""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest
from sklearn.manifold import TSNE

from tomography.analysis import agreement_band
from tomography.analysis import agreement_pseudo_values
from tomography.analysis import layer_cosines
from tomography.analysis import lens_logits
from tomography.analysis import mean_direction
from tomography.analysis import pca_explained_variance
from tomography.analysis import pole_separation
from tomography.analysis import seed_agreement
from tomography.analysis import standardize
from tomography.analysis import top_tokens
from tomography.analysis import tsne_embed
from tomography.reading.metrics import bootstrap_ci


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


def test_a_truncated_spectrum_keeps_the_ratios_over_all_components() -> None:
    s = np.array([5.0, 3.0, 2.0, 1.0, 0.5])
    h = _planted(s, n=12, d=5)
    got = pca_explained_variance(h, n_components=2)
    assert got.shape == (2, 2)
    for layer in got:
        np.testing.assert_allclose(layer, s[:2] ** 2 / np.sum(s**2), atol=1e-9)
        assert layer.sum() < 1


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


def test_tsne_follows_the_scikit_learn_settings() -> None:
    points, _ = _clusters()
    other = np.random.default_rng(9).standard_normal(points.shape)
    h = np.stack([points, other])
    got = tsne_embed(h, perplexity=6.0, seed=3)
    for k, rows in enumerate(h):
        expected = TSNE(
            n_components=2, init="pca", perplexity=6.0, random_state=3
        ).fit_transform(rows)
        np.testing.assert_array_equal(got[k], expected.astype(np.float32))


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


def _runs(n: int = 4, k: int = 3, d: int = 6, seed: int = 0) -> Any:
    return np.random.default_rng(seed).normal(size=(n, k, d)) * 3.0


def _unit(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def test_layer_cosines_average_the_runs_and_normalize_the_inputs() -> None:
    root = np.sqrt(0.5)
    vectors = np.array(
        [
            [[5.0, 0.0], [0.0, 2.0]],
            [[3.0, 0.0], [1.0, 1.0]],
        ]
    )
    got = layer_cosines(vectors)
    np.testing.assert_allclose(
        got, [[1.0, root / 2], [root / 2, 1.0]], atol=1e-12
    )


def test_layer_cosines_has_a_unit_diagonal_and_is_symmetric() -> None:
    got = layer_cosines(_runs())
    assert got.shape == (3, 3)
    np.testing.assert_allclose(np.diag(got), 1.0, atol=1e-12)
    np.testing.assert_allclose(got, got.T, atol=1e-12)


def test_seed_agreement_is_the_mean_cosine_to_the_other_runs() -> None:
    vectors = _runs()
    got = seed_agreement(vectors)
    unit = _unit(vectors)
    assert got.shape == (4, 3)
    for n in range(4):
        others = [m for m in range(4) if m != n]
        expected = np.mean(
            [np.sum(unit[n] * unit[m], axis=-1) for m in others], axis=0
        )
        np.testing.assert_allclose(got[n], expected, atol=1e-12)


def test_the_mean_agreement_is_the_mean_pairwise_cosine() -> None:
    vectors = _runs(n=5)
    unit = _unit(vectors)
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    pairwise = np.mean(
        [np.sum(unit[i] * unit[j], axis=-1) for i, j in pairs], axis=0
    )
    np.testing.assert_allclose(
        seed_agreement(vectors).mean(axis=0), pairwise, atol=1e-12
    )


def test_seed_agreement_of_identical_and_opposite_runs() -> None:
    one = _runs(n=1)
    np.testing.assert_allclose(
        seed_agreement(np.concatenate([one, 2.0 * one, 0.5 * one])), 1.0
    )
    np.testing.assert_allclose(
        seed_agreement(np.concatenate([one, -one])), -1.0
    )


def test_seed_agreement_needs_two_runs() -> None:
    with pytest.raises(ValueError, match="two runs"):
        seed_agreement(_runs(n=1))


def _mean_pairwise(unit: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mean cosine over all pairs of runs, per layer, by explicit pairs."""
    n = unit.shape[0]
    return np.mean(
        [
            np.sum(unit[i] * unit[j], axis=-1)
            for i in range(n)
            for j in range(i + 1, n)
        ],
        axis=0,
    )


def test_pseudo_values_are_the_leave_one_out_jackknife() -> None:
    unit = _unit(_runs(n=5))
    whole = _mean_pairwise(unit)
    got = agreement_pseudo_values(unit)
    assert got.shape == (5, 3)
    for i in range(5):
        left_out = _mean_pairwise(np.delete(unit, i, axis=0))
        np.testing.assert_allclose(got[i], 5 * whole - 4 * left_out, atol=1e-12)


def test_the_pseudo_values_average_to_the_mean_pairwise_cosine() -> None:
    vectors = _runs(n=6)
    np.testing.assert_allclose(
        agreement_pseudo_values(vectors).mean(axis=0),
        _mean_pairwise(_unit(vectors)),
        atol=1e-12,
    )


def test_pseudo_values_need_three_runs() -> None:
    with pytest.raises(ValueError, match="three runs"):
        agreement_pseudo_values(_runs(n=2))


def test_the_agreement_band_is_the_bootstrap_of_the_pseudo_values() -> None:
    vectors = _runs(n=5)
    lo, hi = agreement_band(vectors, n_boot=50, seed=4)
    expected = bootstrap_ci(agreement_pseudo_values(vectors), n_boot=50, seed=4)
    np.testing.assert_array_equal(lo, expected[0])
    np.testing.assert_array_equal(hi, expected[1])
    assert lo.shape == hi.shape == (3,)


def test_the_band_covers_the_pairwise_mean_in_most_replications() -> None:
    width, sigma = 8, 0.8
    shared = np.zeros(width)
    shared[0] = 1.0

    def seeds(rng: np.random.Generator, n: int) -> NDArray[np.float64]:
        return _unit(shared + sigma * rng.normal(size=(n, 1, width)))

    rng = np.random.default_rng(12345)
    first, second = seeds(rng, 200000)[:, 0], seeds(rng, 200000)[:, 0]
    truth = float(np.sum(first * second, axis=-1).mean())
    hits = 0
    replications = 300
    for i in range(replications):
        lo, hi = agreement_band(
            seeds(np.random.default_rng(1000 + i), 6), n_boot=300, seed=i
        )
        hits += int(lo[0] <= truth <= hi[0])
    assert hits / replications > 0.78


def test_mean_direction_averages_the_unit_directions() -> None:
    vectors = np.array([[[10.0, 0.0]], [[0.0, 0.1]]])
    got = mean_direction(vectors)
    np.testing.assert_allclose(got, [[np.sqrt(0.5), np.sqrt(0.5)]])


def test_mean_direction_is_a_unit_vector_per_layer() -> None:
    got = mean_direction(_runs())
    assert got.shape == (3, 6)
    np.testing.assert_allclose(np.linalg.norm(got, axis=-1), 1.0)


def test_mean_direction_refuses_runs_that_cancel() -> None:
    one = _runs(n=1)
    with pytest.raises(ValueError, match="cancel"):
        mean_direction(np.concatenate([one, -one]))


@pytest.mark.parametrize(
    "function",
    [layer_cosines, seed_agreement, mean_direction, agreement_pseudo_values],
    ids=[
        "layer_cosines",
        "seed_agreement",
        "mean_direction",
        "agreement_pseudo_values",
    ],
)
def test_the_geometry_functions_check_their_input(
    function: Callable[[NDArray[np.float64]], object],
) -> None:
    zero = _runs()
    zero[1, 2] = 0.0
    with pytest.raises(ValueError, match="zero"):
        function(zero)
    with pytest.raises(ValueError, match="shape"):
        function(_runs()[0])
    nan = _runs()
    nan[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        function(nan)


def test_lens_logits_reads_a_direction_through_the_unembedding() -> None:
    rng = np.random.default_rng(0)
    unembedding = rng.normal(size=(7, 4)).astype(np.float32)
    vectors = rng.normal(size=(3, 4))
    got = lens_logits(vectors, unembedding)
    assert got.dtype == np.float32
    assert got.shape == (3, 7)
    np.testing.assert_allclose(
        got, vectors @ unembedding.T.astype(np.float64), rtol=1e-5, atol=1e-6
    )


def test_lens_logits_centers_then_scales_by_the_norm_weight() -> None:
    rng = np.random.default_rng(1)
    unembedding = rng.normal(size=(7, 4)).astype(np.float32)
    scale = rng.uniform(0.5, 2.0, size=4).astype(np.float32)
    vector = rng.normal(size=4)
    centered = vector - vector.mean()
    got = lens_logits(vector, unembedding, scale=scale, center=True)
    assert got.shape == (7,)
    np.testing.assert_allclose(
        got, (centered * scale) @ unembedding.T, rtol=1e-5, atol=1e-6
    )
    scaled_only = lens_logits(vector, unembedding, scale=scale)
    np.testing.assert_allclose(
        scaled_only, (vector * scale) @ unembedding.T, rtol=1e-5, atol=1e-6
    )
    centered_only = lens_logits(vector, unembedding, center=True)
    np.testing.assert_allclose(
        centered_only, centered @ unembedding.T, rtol=1e-5, atol=1e-6
    )


def test_lens_logits_centers_every_direction_of_a_batch_over_its_width() -> (
    None
):
    rng = np.random.default_rng(3)
    unembedding = rng.normal(size=(7, 4)).astype(np.float32)
    scale = rng.uniform(0.5, 2.0, size=4).astype(np.float32)
    batch = rng.normal(size=(3, 4)) + np.array([[5.0], [-3.0], [1.0]])
    got = lens_logits(batch, unembedding, scale=scale, center=True)
    oracle = ((batch - batch.mean(-1, keepdims=True)) * scale) @ unembedding.T
    assert got.shape == (3, 7)
    np.testing.assert_allclose(got, oracle, rtol=1e-4, atol=1e-5)


def test_lens_logits_checks_the_widths() -> None:
    unembedding = np.ones((7, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="width"):
        lens_logits(np.ones(5), unembedding)
    with pytest.raises(ValueError, match="scale"):
        lens_logits(np.ones(4), unembedding, scale=np.ones(3))


def test_standardize_is_the_z_score_over_the_last_axis() -> None:
    logits = np.random.default_rng(2).normal(size=(3, 50)).astype(np.float32)
    got = standardize(logits)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got.mean(axis=-1), 0.0, atol=1e-9)
    np.testing.assert_allclose(got.std(axis=-1), 1.0, atol=1e-9)


def test_standardize_refuses_a_constant_row() -> None:
    logits = np.ones((2, 5), dtype=np.float32)
    logits[0] = [1.0, 2.0, 3.0, 4.0, 5.0]
    with pytest.raises(ValueError, match="constant"):
        standardize(logits)


def test_pole_separation_is_the_gap_between_the_pole_means_in_z() -> None:
    logits = np.array([3.0, 1.0, -1.0, -3.0, 0.0])
    sd = np.sqrt(np.mean(logits**2))
    got = pole_separation(logits, [0, 1], [3])
    np.testing.assert_allclose(got, (2.0 - (-3.0)) / sd)
    np.testing.assert_allclose(pole_separation(logits, [3], [0, 1]), -got)
    np.testing.assert_allclose(pole_separation(logits, [4], [4]), 0.0)


def test_pole_separation_keeps_the_leading_axes() -> None:
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(2, 3, 20))
    got = pole_separation(logits, [0, 1], [2])
    assert got.shape == (2, 3)
    np.testing.assert_allclose(
        got[1, 2], pole_separation(logits[1, 2], [0, 1], [2])
    )


def test_pole_separation_checks_the_poles() -> None:
    logits = np.arange(5.0)
    with pytest.raises(ValueError, match="pole"):
        pole_separation(logits, [], [1])
    with pytest.raises(ValueError, match="pole"):
        pole_separation(logits, [1], [])
    with pytest.raises(ValueError, match="ids"):
        pole_separation(logits, [5], [1])
    with pytest.raises(ValueError, match="ids"):
        pole_separation(logits, [-1], [1])


def test_top_tokens_returns_the_extremes_in_stable_order() -> None:
    logits = np.array([1.0, 5.0, 5.0, 0.0, 3.0, 0.0], dtype=np.float32)
    top, bottom = top_tokens(logits, 3)
    assert top.dtype == bottom.dtype == np.int64
    assert top.tolist() == [1, 2, 4]
    assert bottom.tolist() == [3, 5, 0]


@pytest.mark.parametrize("k", [0, 7], ids=["zero", "above"])
def test_top_tokens_checks_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must"):
        top_tokens(np.arange(6.0), k)


def test_top_tokens_needs_one_row() -> None:
    with pytest.raises(ValueError, match="shape"):
        top_tokens(np.ones((2, 6)), 1)
