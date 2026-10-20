"""Tests for the layer analyses: PCA variance, PCA coordinates and t-SNE."""

from collections.abc import Callable
import itertools
import tracemalloc
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from tomography import analysis
from tomography.analysis import agreement_band
from tomography.analysis import agreement_pseudo_values
from tomography.analysis import column_norms
from tomography.analysis import decoder_cosines
from tomography.analysis import dictionary_directions
from tomography.analysis import layer_cosines
from tomography.analysis import lens_logits
from tomography.analysis import mean_direction
from tomography.analysis import pca_coordinates
from tomography.analysis import pca_explained_variance
from tomography.analysis import pole_separation
from tomography.analysis import seed_agreement
from tomography.analysis import sparse_fit
from tomography.analysis import standardize
from tomography.analysis import top_features
from tomography.analysis import top_overlap
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


def test_pca_coordinates_shape_dtype_and_determinism() -> None:
    h = np.random.default_rng(10).standard_normal((3, 12, 6))
    coordinates, ratios = pca_coordinates(h)
    assert coordinates.shape == (3, 12, 2)
    assert coordinates.dtype == np.float32
    assert ratios.shape == (3, 2)
    again_coordinates, again_ratios = pca_coordinates(h)
    np.testing.assert_array_equal(coordinates, again_coordinates)
    np.testing.assert_array_equal(ratios, again_ratios)


def test_pca_coordinates_of_a_hand_computed_case() -> None:
    # Centered rows (+-1, 0) and (0, +-2) have X^T X = diag(2, 8): the first
    # component is the y axis (8 of 10) and the second the x axis (2 of 10).
    h = np.array([[[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0]]]) + 7.0
    coordinates, ratios = pca_coordinates(h)
    np.testing.assert_allclose(
        coordinates[0],
        [[0.0, 1.0], [0.0, -1.0], [2.0, 0.0], [-2.0, 0.0]],
        atol=1e-6,
    )
    np.testing.assert_allclose(ratios[0], [0.8, 0.2], atol=1e-12)


def test_pca_coordinates_agree_with_the_variance_helper_and_scikit_learn() -> (
    None
):
    h = np.random.default_rng(11).standard_normal((2, 15, 6)) * [
        4,
        2,
        1,
        1,
        1,
        1,
    ]
    coordinates, ratios = pca_coordinates(h)
    np.testing.assert_allclose(
        ratios, pca_explained_variance(h, n_components=2), atol=1e-12
    )
    for k, rows in enumerate(h):
        expected = PCA(n_components=2, svd_solver="full").fit_transform(rows)
        np.testing.assert_allclose(
            np.abs(coordinates[k]), np.abs(expected), atol=1e-5
        )


def test_pca_coordinates_make_the_largest_loading_positive() -> None:
    h = np.random.default_rng(12).standard_normal((4, 30, 5)) + 3.0
    coordinates, _ = pca_coordinates(h)
    for k, rows in enumerate(h):
        centered = rows - rows.mean(axis=0)
        for component in range(2):
            loading = np.linalg.lstsq(
                centered, coordinates[k, :, component].astype(np.float64)
            )[0]
            assert loading[np.argmax(np.abs(loading))] > 0


def test_pca_coordinates_treat_layers_independently() -> None:
    first = np.random.default_rng(13).standard_normal((12, 6))
    second = np.random.default_rng(14).standard_normal((12, 6))
    alone_coordinates, alone_ratios = pca_coordinates(first[None])
    stacked_coordinates, stacked_ratios = pca_coordinates(
        np.stack([first, second])
    )
    np.testing.assert_array_equal(alone_coordinates[0], stacked_coordinates[0])
    np.testing.assert_array_equal(alone_ratios[0], stacked_ratios[0])


def test_pca_coordinates_leave_the_input_untouched() -> None:
    h = np.random.default_rng(15).standard_normal((2, 12, 6)) + 5.0
    before = h.copy()
    pca_coordinates(h)
    np.testing.assert_array_equal(h, before)


def test_pca_coordinates_never_cast_the_whole_stack() -> None:
    h = np.random.default_rng(16).standard_normal((40, 100, 50))
    h32 = h.astype(np.float32)
    del h
    whole = h32.size * 8
    tracemalloc.start()
    try:
        pca_coordinates(h32)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < whole / 4


@pytest.mark.parametrize(
    ("h", "match"),
    [
        (np.ones((6, 3)), "shape"),
        (np.full((1, 6, 3), np.nan), "finite"),
        (np.ones((2, 6, 3)), "variance"),
        (np.random.default_rng(0).standard_normal((1, 6, 1)), "components"),
        (np.random.default_rng(0).standard_normal((1, 1, 6)), "components"),
    ],
    ids=["two-d", "nan", "constant", "one-feature", "one-row"],
)
def test_pca_coordinates_reject_bad_input(h: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        pca_coordinates(h)


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


def test_the_agreement_band_agrees_with_the_bootstrap_at_many_runs() -> None:
    """Approximate agreement check against a bootstrap of the pseudo-values.

    With 40 runs each bound lies within a tenth of the band's width of the
    bootstrap bound.
    """
    vectors = _runs(n=40)
    lo, hi = agreement_band(vectors)
    boot_lo, boot_hi = bootstrap_ci(
        agreement_pseudo_values(vectors), n_boot=20000, seed=0
    )
    assert lo.shape == hi.shape == (3,)
    assert (np.abs(lo - boot_lo) < 0.1 * (hi - lo)).all()
    assert (np.abs(hi - boot_hi) < 0.1 * (hi - lo)).all()


T_4_DF = 2.7764451051977934
T_2_DF = 4.302652729749462


def test_the_agreement_band_is_a_t_interval_of_the_pseudo_values() -> None:
    vectors = _runs(n=5)
    pseudo = agreement_pseudo_values(vectors)
    mean = _mean_pairwise(_unit(vectors))
    half = T_4_DF * pseudo.std(axis=0, ddof=1) / np.sqrt(5)
    assert (np.abs(mean) + half < 1.0).all()
    lo, hi = agreement_band(vectors)
    np.testing.assert_allclose(lo, mean - half, atol=1e-12)
    np.testing.assert_allclose(hi, mean + half, atol=1e-12)


def test_a_wider_alpha_gives_a_narrower_band() -> None:
    vectors = _runs(n=5)
    lo, hi = agreement_band(vectors)
    lo_90, hi_90 = agreement_band(vectors, alpha=0.1)
    assert (lo < lo_90).all()
    assert (hi_90 < hi).all()


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_the_agreement_band_refuses_an_alpha_outside_the_unit_interval(
    alpha: float,
) -> None:
    with pytest.raises(ValueError, match="alpha"):
        agreement_band(_runs(n=5), alpha=alpha)


def test_the_agreement_band_is_clipped_to_the_cosine_range() -> None:
    near = np.array([[[1.0, 0.0]], [[1.0, 0.0]], [[0.8, 0.6]]])
    pseudo = agreement_pseudo_values(near)[:, 0]
    mean = pseudo.mean()
    half = T_2_DF * pseudo.std(ddof=1) / np.sqrt(3)
    assert mean + half > 1.0
    assert mean - half > -1.0
    lo, hi = agreement_band(near)
    assert hi[0] == 1.0
    assert lo[0] == pytest.approx(mean - half)
    split = np.array([[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]])
    lo, hi = agreement_band(split)
    assert (lo[0], hi[0]) == (-1.0, 1.0)


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
        lo, hi = agreement_band(seeds(np.random.default_rng(1000 + i), 6))
        hits += int(lo[0] <= truth <= hi[0])
    assert hits / replications > 0.9


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


def _columns() -> NDArray[np.float32]:
    """Returns a (32, 20) float32 decoder with orthonormal columns."""
    raw = np.random.default_rng(0).standard_normal((32, 20))
    q, _ = np.linalg.qr(raw)
    return q.astype(np.float32)


def _planted_vector(decoder: NDArray[np.float32]) -> NDArray[np.float64]:
    columns = decoder.astype(np.float64)
    return 0.8 * columns[:, 3] - 0.6 * columns[:, 7]


def test_column_norms_are_the_float64_norms_of_the_columns() -> None:
    decoder = _columns()
    scales = np.random.default_rng(2).uniform(0.5, 3.0, size=20)
    scaled = (decoder * scales).astype(np.float32)
    norms = column_norms(scaled)
    assert norms.dtype == np.float64
    assert norms.shape == (20,)
    np.testing.assert_allclose(
        norms, np.linalg.norm(scaled.astype(np.float64), axis=0)
    )
    np.testing.assert_allclose(norms, scales, rtol=1e-5)


def test_column_norms_refuse_a_bad_decoder() -> None:
    decoder = _columns()
    with pytest.raises(ValueError, match="shape"):
        column_norms(decoder[0])
    zero_column = decoder.copy()
    zero_column[:, 4] = 0.0
    with pytest.raises(ValueError, match="zero column"):
        column_norms(zero_column)
    bad_column = decoder.copy()
    bad_column[2, 4] = np.inf
    with pytest.raises(ValueError, match="finite"):
        column_norms(bad_column)


def test_decoder_cosines_reads_the_planted_vector_off_the_columns() -> None:
    decoder = _columns()
    cosines = decoder_cosines(_planted_vector(decoder)[None], decoder)
    assert cosines.dtype == np.float32
    assert cosines.shape == (1, 20)
    assert cosines[0, 3] == pytest.approx(0.8, abs=1e-6)
    assert cosines[0, 7] == pytest.approx(-0.6, abs=1e-6)
    others = np.delete(cosines[0], [3, 7])
    assert np.abs(others).max() < 1e-6


def test_decoder_cosines_ignores_the_scale_of_vectors_and_columns() -> None:
    decoder = _columns()
    vectors = np.random.default_rng(1).standard_normal((4, 32))
    scales = np.random.default_rng(2).uniform(0.5, 3.0, size=20)
    expected = decoder_cosines(vectors, decoder)
    scaled = decoder_cosines(
        vectors * 7.0, (decoder * scales).astype(np.float32)
    )
    np.testing.assert_allclose(scaled, expected, atol=1e-6)
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    np.testing.assert_allclose(
        expected, unit @ decoder.astype(np.float64), atol=1e-6
    )


def test_decoder_cosines_leaves_its_inputs_alone() -> None:
    decoder = _columns()
    decoder *= np.float32(2.0)
    decoder.setflags(write=False)
    vectors = np.random.default_rng(1).standard_normal((2, 32))
    before = vectors.copy()
    decoder_cosines(vectors, decoder)
    np.testing.assert_array_equal(vectors, before)
    assert float(np.linalg.norm(decoder[:, 0])) == pytest.approx(2.0, abs=1e-6)


def test_top_features_orders_the_columns_by_absolute_cosine() -> None:
    decoder = _columns()
    cosines = decoder_cosines(_planted_vector(decoder)[None], decoder)
    top = top_features(cosines, 4)
    assert top.dtype == np.int64
    assert top.shape == (1, 4)
    assert top[0, :2].tolist() == [3, 7]


def test_top_features_gives_ties_to_the_lower_id() -> None:
    cosines = np.array(
        [[0.1, -0.5, 0.5, 0.0, 0.5, -0.1], [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]],
        dtype=np.float32,
    )
    assert top_features(cosines, 3).tolist() == [[1, 2, 4], [0, 1, 2]]


def test_sparse_fit_recovers_the_planted_atoms() -> None:
    decoder = _columns()
    explained, atoms = sparse_fit(_planted_vector(decoder)[None], decoder, 5)
    assert explained.shape == (1, 5)
    assert atoms.shape == (1, 5)
    assert atoms.dtype == np.int64
    assert atoms[0, :2].tolist() == [3, 7]
    np.testing.assert_allclose(
        explained[0], [0.64, 1.0, 1.0, 1.0, 1.0], atol=1e-6
    )


def test_sparse_fit_ignores_scale_and_sign() -> None:
    decoder = _columns()
    vector = _planted_vector(decoder)[None]
    explained, atoms = sparse_fit(vector, decoder, 4)
    scales = np.random.default_rng(2).uniform(0.5, 3.0, size=20)
    for changed, columns in (
        (vector * 9.0, decoder),
        (vector, (decoder * scales).astype(np.float32)),
        (vector, (decoder * 1e-12).astype(np.float32)),
        (-vector, decoder),
    ):
        got_explained, got_atoms = sparse_fit(changed, columns, 4)
        np.testing.assert_allclose(got_explained, explained, atol=1e-6)
        np.testing.assert_array_equal(got_atoms, atoms)


def test_sparse_fit_gives_ties_to_the_lower_id() -> None:
    decoder = _columns()
    decoder[:, 12] = decoder[:, 5]
    decoder[:, 15] = decoder[:, 5]
    vector = decoder[:, 5].astype(np.float64)
    explained, atoms = sparse_fit(vector[None], decoder, 2)
    assert atoms[0, 0] == 5
    assert explained[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_sparse_fit_ties_are_within_1e_6_of_the_best_score() -> None:
    def first_atom(gap: float) -> int:
        angles = np.arccos(np.array([0.1, 0.5, 0.5 + gap, 0.2]))
        decoder = np.stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
        return int(sparse_fit(np.array([[1.0, 0.0]]), decoder, 1)[1][0, 0])

    assert first_atom(5e-7) == 1
    assert first_atom(1e-5) == 2


def test_sparse_fit_survives_a_column_that_is_already_in_the_span() -> None:
    decoder = _columns()
    decoder[:, 0] = 0.0
    decoder[0, 0] = 1.0
    decoder[:, 1] = decoder[:, 0]
    vector = np.zeros((1, 32))
    vector[0, 0] = 1.0
    explained, atoms = sparse_fit(vector, decoder, 3)
    assert atoms[0, :2].tolist() == [0, 1]
    assert np.isfinite(explained).all()
    np.testing.assert_allclose(explained[0], 1.0)


def test_sparse_fit_picks_by_the_residual_and_not_by_the_cosine() -> None:
    decoder = np.array(
        [[1.0, 0.9, 0.0], [0.0, np.sqrt(0.19), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    vector = np.array([[0.8, 0.0, 0.6]])
    cosines = decoder_cosines(vector, decoder)
    assert top_features(cosines, 3).tolist() == [[0, 1, 2]]
    explained, atoms = sparse_fit(vector, decoder, 2)
    assert atoms.tolist() == [[0, 2]]
    np.testing.assert_allclose(explained[0], [0.64, 1.0], atol=1e-6)


def test_sparse_fit_takes_no_more_atoms_than_the_dimension() -> None:
    rng = np.random.default_rng(0)
    decoder = rng.standard_normal((16, 40)).astype(np.float32)
    vectors = rng.standard_normal((2, 16))
    with pytest.raises(ValueError, match="n_atoms"):
        sparse_fit(vectors, decoder, 17)


def test_sparse_fit_is_monotone_bounded_and_picks_distinct_atoms() -> None:
    rng = np.random.default_rng(4)
    decoder = rng.standard_normal((16, 40)).astype(np.float32)
    vectors = rng.standard_normal((6, 16))
    explained, atoms = sparse_fit(vectors, decoder, 16)
    assert (np.diff(explained, axis=1) >= 0.0).all()
    assert explained.min() >= 0.0
    assert explained.max() <= 1.0
    assert all(len(set(row)) == 16 for row in atoms.tolist())
    assert explained[:, -1] == pytest.approx(1.0, abs=1e-5)


def _best_subset_fit(
    vector: NDArray[np.float64], decoder: NDArray[np.float32], size: int
) -> float:
    unit = vector / np.linalg.norm(vector)
    best = 0.0
    for subset in itertools.combinations(range(decoder.shape[1]), size):
        columns = decoder[:, subset].astype(np.float64)
        coefficients, *_ = np.linalg.lstsq(columns, unit, rcond=None)
        fit = 1.0 - float(np.sum((unit - columns @ coefficients) ** 2))
        best = max(best, fit)
    return best


def test_sparse_fit_never_beats_the_best_subset_of_the_same_size() -> None:
    rng = np.random.default_rng(5)
    decoder = rng.standard_normal((12, 14)).astype(np.float32)
    vectors = rng.standard_normal((5, 12))
    explained, _ = sparse_fit(vectors, decoder, 3)
    for i, vector in enumerate(vectors):
        bests = [_best_subset_fit(vector, decoder, s) for s in (1, 2, 3)]
        assert (explained[i] <= np.array(bests) + 1e-6).all()
        assert explained[i, 0] == pytest.approx(bests[0], abs=1e-6)


def test_sparse_fit_reorthogonalizes_nearly_collinear_atoms() -> None:
    rng = np.random.default_rng(7)
    d, f = 40, 30
    q, _ = np.linalg.qr(rng.standard_normal((d, f)))
    columns = np.empty((d, f))
    columns[:, 0] = q[:, 0]
    for j in range(1, f):
        columns[:, j] = columns[:, j - 1] + 1e-6 * q[:, j]
    columns /= np.linalg.norm(columns, axis=0)
    decoder = columns.astype(np.float32)
    vectors = rng.standard_normal((4, d))
    explained, atoms = sparse_fit(vectors, decoder, 20)
    matrix = decoder.astype(np.float64)
    for i, vector in enumerate(vectors):
        unit = vector / np.linalg.norm(vector)
        for t in range(20):
            chosen = matrix[:, atoms[i, : t + 1]]
            coef = np.linalg.lstsq(chosen, unit, rcond=None)[0]
            oracle = 1.0 - np.sum((unit - chosen @ coef) ** 2)
            assert explained[i, t] == pytest.approx(oracle, abs=1e-9)


def test_sparse_fit_matches_the_best_pair_on_a_planted_pair() -> None:
    rng = np.random.default_rng(6)
    decoder = rng.standard_normal((64, 8)).astype(np.float32)
    vector = decoder[:, 2].astype(np.float64) + 0.5 * decoder[:, 5]
    explained, atoms = sparse_fit(vector[None], decoder, 2)
    assert sorted(atoms[0].tolist()) == [2, 5]
    assert _best_subset_fit(vector, decoder, 2) == pytest.approx(1.0, abs=1e-6)
    assert explained[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_sparse_fit_rows_do_not_depend_on_the_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(7)
    decoder = rng.standard_normal((16, 30)).astype(np.float32)
    vectors = rng.standard_normal((5, 16))
    monkeypatch.setattr(analysis, "_FIT_BLOCK", 2)
    explained, atoms = sparse_fit(vectors, decoder, 6)
    for i, vector in enumerate(vectors):
        one_explained, one_atoms = sparse_fit(vector[None], decoder, 6)
        np.testing.assert_allclose(explained[i], one_explained[0], atol=1e-9)
        np.testing.assert_array_equal(atoms[i], one_atoms[0])


def test_dictionary_directions_are_unit_and_stay_in_the_decoder_span() -> None:
    decoder = np.zeros((8, 5), dtype=np.float32)
    decoder[:3] = np.random.default_rng(0).standard_normal((3, 5))
    directions = dictionary_directions(decoder, 50, np.random.default_rng(1))
    assert directions.shape == (50, 8)
    assert directions.dtype == np.float64
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert (directions[:, 3:] == 0.0).all()


def test_dictionary_directions_replay_the_draw_of_the_contract() -> None:
    decoder = _columns()
    got = dictionary_directions(decoder, 7, np.random.default_rng(3))
    g = np.random.default_rng(3).standard_normal((7, 20)).astype(np.float32)
    expected = (g @ decoder.T).astype(np.float64)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(got, expected, atol=1e-7)


def test_dictionary_directions_follow_the_covariance_of_the_decoder() -> None:
    decoder = np.zeros((8, 11), dtype=np.float32)
    decoder[0, :10] = 1.0
    decoder[1, 10] = 1.0
    directions = dictionary_directions(decoder, 4000, np.random.default_rng(0))
    share = float((directions[:, 0] ** 2).mean())
    expected = (10 / 9) * (1 - 1 / np.sqrt(10))
    assert share == pytest.approx(expected, abs=0.02)
    assert abs(share - 0.5) > 0.2


def test_top_overlap_is_the_mean_pairwise_jaccard_of_the_id_sets() -> None:
    ids = np.array([[1, 2, 3], [1, 2, 4], [3, 2, 1]], dtype=np.int64)
    assert top_overlap(ids) == pytest.approx(2 / 3)
    assert top_overlap(np.array([[1, 2], [1, 2]])) == 1.0
    assert top_overlap(np.array([[1, 2], [3, 4], [5, 6]])) == 0.0


def test_the_sae_functions_check_their_input() -> None:
    decoder = _columns()
    vectors = np.ones((2, 32))
    with pytest.raises(ValueError, match="shape"):
        decoder_cosines(np.ones(32), decoder)
    with pytest.raises(ValueError, match="shape"):
        decoder_cosines(vectors, decoder[0])
    with pytest.raises(ValueError, match="width"):
        decoder_cosines(np.ones((2, 31)), decoder)
    with pytest.raises(ValueError, match="zero vector"):
        decoder_cosines(np.zeros((2, 32)), decoder)
    with pytest.raises(ValueError, match="finite"):
        decoder_cosines(np.full((2, 32), np.nan), decoder)
    zero_column = decoder.copy()
    zero_column[:, 4] = 0.0
    with pytest.raises(ValueError, match="zero column"):
        decoder_cosines(vectors, zero_column)
    bad_column = decoder.copy()
    bad_column[2, 4] = np.inf
    with pytest.raises(ValueError, match="finite"):
        sparse_fit(vectors, bad_column, 2)
    with pytest.raises(ValueError, match="zero vector"):
        sparse_fit(np.zeros((2, 32)), decoder, 2)
    for n_atoms in (0, 21):
        with pytest.raises(ValueError, match="n_atoms"):
            sparse_fit(vectors, decoder, n_atoms)
    with pytest.raises(ValueError, match="shape"):
        top_features(np.ones(5), 1)
    for k in (0, 6):
        with pytest.raises(ValueError, match="k must"):
            top_features(np.ones((2, 5)), k)
    with pytest.raises(ValueError, match="finite"):
        top_features(np.full((2, 5), np.nan), 1)
    with pytest.raises(ValueError, match="at least 2"):
        top_overlap(np.array([[1, 2]]))
    with pytest.raises(ValueError, match="shape"):
        top_overlap(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="shape"):
        dictionary_directions(decoder[0], 3, np.random.default_rng(0))
    with pytest.raises(ValueError, match="n must"):
        dictionary_directions(decoder, 0, np.random.default_rng(0))
    with pytest.raises(ValueError, match="finite"):
        dictionary_directions(bad_column, 3, np.random.default_rng(0))
    with pytest.raises(ValueError, match="zero vector"):
        dictionary_directions(np.zeros((8, 5)), 3, np.random.default_rng(0))
