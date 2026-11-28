"""Tests of the invariance readers of the preregistration, sections 4 and 5."""

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.activations import Activations
from tomography.reading import invariance as inv
from tomography.reading.probes import fit_attention


def _renderings(k: int) -> inv.Renderings:
    names = ("clean", *(f"F-{i}" for i in range(1, k)))
    levels: tuple[inv.Level, ...] = ("none", *("random",) * (k - 1))
    return inv.Renderings(names, levels)


def _labelled(
    *,
    n_pairs: int = 8,
    n_layers: int = 3,
    k: int = 4,
    width: int = 6,
    seed: int = 0,
    signal: float = 1.0,
) -> inv.LabelledSet:
    rng = np.random.default_rng(seed)
    n = 2 * n_pairs
    labels = np.tile(np.array([0, 1]), n_pairs)
    pairs = np.repeat(np.arange(n_pairs), 2)
    y = 2.0 * labels - 1.0
    direction = np.zeros(width)
    direction[0] = 1.0
    text = rng.normal(size=(n_layers, n, 1, width))
    wrap = 0.3 * rng.normal(size=(n_layers, n, k, width))
    stim = text + wrap + signal * y[None, :, None, None] * direction
    last = rng.normal(size=(n_layers, n, k, width)) + stim
    return inv.LabelledSet(
        stim=stim.astype(np.float32),
        last=last.astype(np.float32),
        labels=labels.astype(np.int64),
        pairs=pairs.astype(np.int64),
        renderings=_renderings(k),
        layers=tuple(range(10, 10 + n_layers)),
    )


def _pool(
    *, m: int = 20, n_layers: int = 3, k: int = 4, width: int = 6, seed: int = 1
) -> inv.Pool:
    rng = np.random.default_rng(seed)
    stim = rng.normal(size=(n_layers, m, k, width))
    last = rng.normal(size=(n_layers, m, k, width))
    return inv.Pool(
        stim=stim.astype(np.float32),
        last=last.astype(np.float32),
        renderings=_renderings(k),
        layers=tuple(range(10, 10 + n_layers)),
    )


def _literal_sigma_e(phi: NDArray[np.float64]) -> NDArray[np.float64]:
    m, k, d = phi.shape
    total = np.zeros((d, d))
    mu = [phi[:, a].mean(axis=0) for a in range(k)]
    for j in range(m):
        c = [phi[j, a] - mu[a] for a in range(k)]
        mean_c = sum(c) / k
        for a in range(k):
            e = c[a] - mean_c
            total += np.outer(e, e)
    sigma = total / (m * (k - 1))
    return sigma * d / np.trace(sigma)


def test_renderings_must_start_with_the_clean_template() -> None:
    with pytest.raises(ValueError, match="clean"):
        inv.Renderings(("F-1", "clean"), ("random", "none"))


def test_labelled_set_refuses_against_copies() -> None:
    data = _labelled()
    tags = inv.Renderings(
        ("clean", "F-1", "F-2", "S-D1"), ("none", "random", "random", "against")
    )
    with pytest.raises(inv.AgainstCopyError, match="S-D1"):
        inv.LabelledSet(
            stim=data.stim,
            last=data.last,
            labels=data.labels,
            pairs=data.pairs,
            renderings=tags,
            layers=data.layers,
        )


def test_pool_refuses_against_copies() -> None:
    pool = _pool()
    tags = inv.Renderings(
        ("clean", "F-1", "F-2", "S-D2"), ("none", "random", "random", "against")
    )
    with pytest.raises(inv.AgainstCopyError):
        inv.Pool(
            stim=pool.stim, last=pool.last, renderings=tags, layers=pool.layers
        )


def test_covariance_refuses_against_copies() -> None:
    phi = np.random.default_rng(0).normal(size=(5, 3, 4))
    tags = inv.Renderings(
        ("clean", "F-1", "S-D1"), ("none", "random", "against")
    )
    with pytest.raises(inv.AgainstCopyError):
        inv.within_item_covariance(phi, tags)


def test_labelled_set_needs_one_text_of_each_label_per_pair() -> None:
    data = _labelled()
    labels = data.labels.copy()
    labels[1] = 0
    with pytest.raises(ValueError, match="pair"):
        inv.LabelledSet(
            stim=data.stim,
            last=data.last,
            labels=labels,
            pairs=data.pairs,
            renderings=data.renderings,
            layers=data.layers,
        )


def test_sigma_e_matches_the_formula_of_section_4() -> None:
    phi = np.random.default_rng(3).normal(size=(7, 4, 5))
    sigma = inv.within_item_covariance(phi, _renderings(4))
    np.testing.assert_allclose(sigma, _literal_sigma_e(phi), atol=1e-12)


def test_sigma_e_has_trace_equal_to_the_feature_dimension() -> None:
    phi = np.random.default_rng(4).normal(size=(9, 3, 6))
    sigma = inv.within_item_covariance(phi, _renderings(3))
    assert np.trace(sigma) == pytest.approx(6.0)


def test_sigma_e_is_class_free() -> None:
    rng = np.random.default_rng(5)
    phi = rng.normal(size=(10, 4, 5))
    per_text = 50.0 * rng.normal(size=(10, 1, 5))
    per_wrapper = 30.0 * rng.normal(size=(1, 4, 5))
    sigma = inv.within_item_covariance(phi, _renderings(4))
    shifted = inv.within_item_covariance(
        phi + per_text + per_wrapper, _renderings(4)
    )
    np.testing.assert_allclose(shifted, sigma, atol=1e-9)


def test_standardization_divides_a_constant_dimension_by_one() -> None:
    x = np.array([[1.0, 5.0, 2.0], [3.0, 5.0, 6.0]])
    mean, scale = inv.standardization(x)
    np.testing.assert_allclose(mean, [2.0, 5.0, 4.0])
    np.testing.assert_allclose(scale, [1.0, 1.0, 2.0])


def test_folds_split_64_pairs_into_4_folds_of_16_pairs() -> None:
    pairs = np.repeat(np.arange(64), 2)
    folds = inv.fold_ids(pairs, seed=7)
    for fold in range(4):
        assert np.unique(pairs[folds == fold]).size == 16
    for pair in range(64):
        assert np.unique(folds[pairs == pair]).size == 1


def test_folds_follow_a_seeded_permutation() -> None:
    pairs = np.repeat(np.arange(64), 2)
    np.testing.assert_array_equal(
        inv.fold_ids(pairs, seed=7), inv.fold_ids(pairs, seed=7)
    )
    assert not np.array_equal(
        inv.fold_ids(pairs, seed=7), inv.fold_ids(pairs, seed=8)
    )


def test_auroc_counts_ties_as_half() -> None:
    labels = np.array([0, 1, 0, 1])
    assert inv.auroc(np.array([0.0, 1.0, 1.0, 1.0]), labels) == 0.75
    assert inv.auroc(np.zeros(4), labels) == 0.5


def _custom(
    stim: NDArray[np.float64], last: NDArray[np.float64]
) -> inv.LabelledSet:
    n = stim.shape[1]
    return inv.LabelledSet(
        stim=stim,
        last=last,
        labels=np.tile(np.array([0, 1]), n // 2).astype(np.int64),
        pairs=np.repeat(np.arange(n // 2), 2).astype(np.int64),
        renderings=_renderings(stim.shape[2]),
        layers=tuple(range(stim.shape[0])),
    )


def _standardized(
    x: NDArray[np.float64], clean: NDArray[np.float64]
) -> NDArray[np.float64]:
    mean, scale = inv.standardization(clean)
    return (np.asarray(x, dtype=np.float64) - mean) / scale


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(1.0 / (1.0 + np.exp(-x)))


def test_grids_follow_section_5_2() -> None:
    assert inv.LAMBDAS == (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    assert inv.GAMMAS == (1e-2, 1e-1, 1.0, 10.0, 100.0)
    assert inv.ALPHAS == (0.01, 0.1, 0.3, 0.6, 0.9)
    assert 0.0 not in inv.GAMMAS


def test_a1_minimizes_its_exact_objective() -> None:
    data = _labelled(width=5)
    reader = inv.fit_fixed(
        inv.ARMS["A1"], data, None, inv.Candidate((1,), lam=1e-2)
    )
    (block,) = reader.blocks
    clean = data.last[1, :, 0]
    mean, scale = inv.standardization(clean)
    np.testing.assert_allclose(block.mean, mean)
    np.testing.assert_allclose(block.scale, scale)
    assert (block.position, block.layer, block.index) == ("last", 11, 1)
    z = _standardized(clean, clean)
    y = 2.0 * data.labels - 1.0
    r = -y * _sigmoid(-y * (z @ block.weight + reader.bias)) / y.size
    assert np.abs(z.T @ r + 1e-2 * block.weight).max() < 1e-6
    assert abs(r.sum()) < 1e-6
    assert reader.params == {"lambda": 1e-2}


def test_a6_minimizes_its_exact_objective() -> None:
    data = _labelled(width=4, k=3)
    pool = _pool(width=4, k=3)
    reader = inv.fit_fixed(
        inv.ARMS["A6"], data, pool, inv.Candidate((0, 2), lam=0.1, gamma=1.0)
    )
    stim_clean, last_clean = data.stim[0, :, 0], data.last[2, :, 0]
    phi = np.concatenate(
        [
            _standardized(data.stim[0], stim_clean),
            _standardized(data.last[2], last_clean),
        ],
        axis=-1,
    )
    pool_phi = np.concatenate(
        [
            _standardized(pool.stim[0], stim_clean),
            _standardized(pool.last[2], last_clean),
        ],
        axis=-1,
    )
    sigma = inv.within_item_covariance(pool_phi, pool.renderings)
    w = np.concatenate([b.weight for b in reader.blocks])
    n, k, _ = phi.shape
    rows = phi.reshape(n * k, -1)
    y = np.repeat(2.0 * data.labels - 1.0, k)
    r = -y * _sigmoid(-y * (rows @ w + reader.bias)) / (n * k)
    gradient = rows.T @ r + 0.1 * w + 1.0 * sigma @ w
    assert np.abs(gradient).max() < 1e-6
    assert abs(r.sum()) < 1e-6
    assert [b.position for b in reader.blocks] == ["stim", "last"]
    assert reader.params == {"lambda": 0.1, "gamma": 1.0}


def test_a_text_and_its_copies_share_a_total_weight_of_one() -> None:
    data = _labelled(k=4)
    stim = np.repeat(data.stim[:, :, :1], 4, axis=2)
    last = np.repeat(data.last[:, :, :1], 4, axis=2)
    copies = _custom(stim, last)
    candidate = inv.Candidate((1,), lam=1e-3)
    plain = inv.fit_fixed(inv.ARMS["A1"], copies, None, candidate)
    augmented = inv.fit_fixed(inv.ARMS["A3"], copies, None, candidate)
    np.testing.assert_allclose(
        augmented.blocks[0].weight, plain.blocks[0].weight, atol=1e-6
    )


def _nuisance_data(seed: int, n: int) -> tuple[inv.LabelledSet, inv.Pool]:
    rng = np.random.default_rng(seed)
    k, width = 4, 6
    y = 2.0 * np.tile(np.array([0, 1]), n // 2) - 1.0

    def features(signs: NDArray[np.float64]) -> NDArray[np.float64]:
        x = rng.normal(size=(1, signs.size, 1, width)) + 0.1 * rng.normal(
            size=(1, signs.size, k, width)
        )
        x[..., 0] += 1.0 * signs[None, :, None]
        decoy = rng.choice([-1.5, 1.5], size=(signs.size, k))
        decoy[:, 0] = 0.0
        x[..., 1] += 0.8 * signs[None, :, None] + decoy[None]
        return x

    x = features(y)
    data = _custom(x, x.copy())
    pool_x = features(rng.choice([-1.0, 1.0], size=4 * n))
    pool = inv.Pool(
        stim=pool_x, last=pool_x.copy(), renderings=_renderings(k), layers=(0,)
    )
    return data, pool


def test_a6_shrinks_the_weight_on_a_planted_wrapper_nuisance() -> None:
    data, pool = _nuisance_data(seed=2, n=64)
    a5 = inv.fit_fixed(
        inv.ARMS["A5"], data, pool, inv.Candidate((0, 0), lam=1e-2)
    )
    a6 = inv.fit_fixed(
        inv.ARMS["A6"], data, pool, inv.Candidate((0, 0), lam=1e-2, gamma=10.0)
    )

    def nuisance_share(reader: inv.LinearReader) -> float:
        w = np.concatenate([b.weight for b in reader.blocks])
        return float(np.hypot(w[1], w[7]) / np.linalg.norm(w))

    assert nuisance_share(a6) < 0.5 * nuisance_share(a5)


def _separable(informative: tuple[int, ...] = (0, 1, 2)) -> inv.LabelledSet:
    rng = np.random.default_rng(9)
    n, k, width = 16, 3, 4
    y = 2.0 * np.tile(np.array([0, 1]), n // 2) - 1.0
    x = np.zeros((3, n, k, width))
    x[:, :, :, 0] = rng.normal(size=(3, n, k))
    for layer in informative:
        x[layer, :, :, 0] = 5.0 * y[:, None] + 0.01 * rng.normal(size=(n, k))
    return _custom(x, x.copy())


def test_ties_go_to_the_larger_lambda_then_the_shallower_layer() -> None:
    tuning = inv.tune(inv.ARMS["A1"], _separable(), None, seed=0)
    assert np.all(tuning.criteria == 1.0)
    assert tuning.reader.params == {"lambda": 1.0}
    assert tuning.reader.blocks[0].index == 0


def test_ties_go_to_the_larger_gamma_after_lambda() -> None:
    data = _separable()
    pool = inv.Pool(
        stim=data.stim,
        last=data.last,
        renderings=data.renderings,
        layers=data.layers,
    )
    tuning = inv.tune(inv.ARMS["A6"], data, pool, seed=0, layers=(0, 0))
    assert np.all(tuning.criteria == 1.0)
    assert tuning.reader.params == {"lambda": 1.0, "gamma": 100.0}


def test_a_single_position_reader_takes_the_best_layer() -> None:
    data = _separable(informative=(2,))
    tuning = inv.tune(inv.ARMS["A1"], data, None, seed=0)
    assert tuning.reader.blocks[0].index == 2


def test_plain_readers_tune_on_clean_auroc_and_augmented_on_the_worst() -> None:
    data = _labelled(signal=0.6)
    plain = inv.tune(inv.ARMS["A2"], data, None, seed=1)
    augmented = inv.tune(inv.ARMS["A4"], data, None, seed=1)
    for c in range(len(plain.candidates)):
        assert plain.criteria[c] == inv.auroc(plain.oof[c, 0], data.labels)
        worst = min(
            inv.auroc(augmented.oof[c, k], data.labels) for k in range(4)
        )
        assert augmented.criteria[c] == worst
    assert len(plain.candidates) == 3 * len(inv.LAMBDAS)


def test_out_of_fold_scores_come_from_the_training_folds_only() -> None:
    data = _labelled(signal=0.6)
    tuning = inv.tune(inv.ARMS["A3"], data, None, seed=3)
    np.testing.assert_array_equal(
        tuning.folds, inv.fold_ids(data.pairs, seed=3)
    )
    test = np.flatnonzero(tuning.folds == 0)
    train = data.subset(np.flatnonzero(tuning.folds != 0))
    for c in (0, 7, 14):
        candidate = tuning.candidates[c]
        reader = inv.fit_fixed(inv.ARMS["A3"], train, None, candidate)
        for k in range(4):
            expected = reader.score(
                data.stim[:, test, k], data.last[:, test, k]
            )
            np.testing.assert_allclose(
                tuning.oof[c, k, test], expected, atol=1e-9
            )


def test_held_out_labels_do_not_change_out_of_fold_scores() -> None:
    data = _labelled(signal=0.6)
    pool = _pool()
    folds = inv.fold_ids(data.pairs, seed=3)
    swapped = data.labels.copy()
    swapped[folds == 0] = 1 - swapped[folds == 0]
    flipped = inv.LabelledSet(
        stim=data.stim,
        last=data.last,
        labels=swapped,
        pairs=data.pairs,
        renderings=data.renderings,
        layers=data.layers,
    )
    for name, layers in (("A2", None), ("A6", (1, 2))):
        one = inv.tune(inv.ARMS[name], data, pool, seed=3, layers=layers)
        two = inv.tune(inv.ARMS[name], flipped, pool, seed=3, layers=layers)
        np.testing.assert_array_equal(
            one.oof[:, :, folds == 0], two.oof[:, :, folds == 0]
        )


def test_the_final_reader_is_refitted_on_all_texts() -> None:
    data = _labelled(signal=0.6)
    tuning = inv.tune(inv.ARMS["A3"], data, None, seed=3)
    best = tuning.candidates.index(tuning.choice)
    refit = inv.fit_fixed(inv.ARMS["A3"], data, None, tuning.choice)
    x = (data.stim[:, :, 2], data.last[:, :, 2])
    np.testing.assert_allclose(tuning.reader.score(*x), refit.score(*x))
    assert tuning.reader.criterion == tuning.criteria[best]
    assert tuning.criteria[best] == tuning.criteria.max()


def test_two_position_readers_take_the_layers_of_a4_and_a3() -> None:
    data = _labelled(signal=0.6)
    pool = _pool()
    a3 = inv.tune(inv.ARMS["A3"], data, None, seed=5)
    a4 = inv.tune(inv.ARMS["A4"], data, None, seed=5)
    for name in ("A5", "A6", "random_subspace"):
        stim, last = inv.fit_arm(name, data, pool, seed=5).reader.blocks
        assert stim.index == a4.reader.blocks[0].index
        assert last.index == a3.reader.blocks[0].index


def test_random_subspace_keeps_the_spectrum_of_sigma_e() -> None:
    phi = np.random.default_rng(6).normal(size=(30, 4, 8))
    sigma = inv.within_item_covariance(phi, _renderings(4))
    q = inv.haar_orthogonal(8, seed=2)
    np.testing.assert_allclose(q @ q.T, np.eye(8), atol=1e-12)
    rotated = inv.rotate_covariance(sigma, seed=2)
    np.testing.assert_allclose(rotated, q @ sigma @ q.T)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(rotated), np.linalg.eigvalsh(sigma), atol=1e-10
    )
    assert not np.allclose(rotated, sigma)


def _class_scatter(
    z: NDArray[np.float64], groups: NDArray[np.int64]
) -> NDArray[np.float64]:
    total = np.zeros((z.shape[1], z.shape[1]))
    for group in np.unique(groups):
        centred = z[groups == group] - z[groups == group].mean(axis=0)
        total += centred.T @ centred
    return total


def _ulda_weight(
    scatter: NDArray[np.float64], d_hat: NDArray[np.float64], alpha: float
) -> NDArray[np.float64]:
    d = d_hat.size
    shrunk = (1 - alpha) * scatter + alpha * np.trace(scatter) / d * np.eye(d)
    return np.asarray(np.linalg.solve(shrunk, d_hat))


def _far_pool(data: inv.LabelledSet) -> inv.Pool:
    pool = _pool()
    last = pool.last.astype(np.float64)
    last[..., 0] += 50.0 * np.sign(data.last[..., 0].mean())
    return inv.Pool(
        stim=pool.stim,
        last=last,
        renderings=pool.renderings,
        layers=pool.layers,
    )


def test_a7_follows_the_formula_of_section_5_2() -> None:
    data, pool = _labelled(signal=1.5), _pool()
    reader = inv.fit_fixed(
        inv.ARMS["A7"], data, pool, inv.Candidate((1,), alpha=0.3)
    )
    clean = data.last[1, :, 0]
    z = _standardized(clean, clean)
    d_hat = z[data.labels == 1].mean(0) - z[data.labels == 0].mean(0)
    g = inv.fit_fixed(
        inv.ARMS["A1"], data, None, inv.Candidate((1,), lam=inv.PSEUDO_LAMBDA)
    )
    pseudo = (g.score(pool.stim[:, :, 0], pool.last[:, :, 0]) > 0).astype(int)
    assert 2 <= pseudo.sum() <= pseudo.size - 2
    pool_z = _standardized(pool.last[1, :, 0], clean)
    scatter = _class_scatter(pool_z, pseudo) / (pseudo.size - 2)
    np.testing.assert_allclose(
        reader.blocks[0].weight, _ulda_weight(scatter, d_hat, 0.3), rtol=1e-8
    )
    assert reader.fallbacks == 0
    assert reader.params == {"alpha": 0.3}


def test_a8_follows_the_formula_of_section_5_2() -> None:
    data, pool = _labelled(signal=1.5), _pool()
    reader = inv.fit_fixed(
        inv.ARMS["A8"], data, pool, inv.Candidate((2,), alpha=0.1)
    )
    clean = data.last[2, :, 0]
    z = _standardized(data.last[2], clean).mean(axis=1)
    d_hat = z[data.labels == 1].mean(0) - z[data.labels == 0].mean(0)
    g = inv.fit_fixed(
        inv.ARMS["A3"], data, None, inv.Candidate((2,), lam=inv.PSEUDO_LAMBDA)
    )
    pseudo = (g.score(pool.stim[:, :, 0], pool.last[:, :, 0]) > 0).astype(int)
    assert 2 <= pseudo.sum() <= pseudo.size - 2
    pool_z = _standardized(pool.last[2], clean)
    m, k = pool_z.shape[:2]
    scatter = sum(_class_scatter(pool_z[:, a], pseudo) for a in range(k))
    weight = _ulda_weight(scatter / (m * k - 2 * k), d_hat, 0.1)
    np.testing.assert_allclose(reader.blocks[0].weight, weight, rtol=1e-8)


def test_a7_and_a8_fall_back_to_the_pool_mean_and_count_it() -> None:
    data = _labelled(signal=1.5)
    pool = _far_pool(data)
    clean = data.last[1, :, 0]
    pool_z = _standardized(pool.last[1], clean)
    m, k = pool_z.shape[:2]
    one_group = np.zeros(m, dtype=np.int64)
    z = _standardized(clean, clean)
    d_hat = z[data.labels == 1].mean(0) - z[data.labels == 0].mean(0)
    a7 = inv.fit_fixed(
        inv.ARMS["A7"], data, pool, inv.Candidate((1,), alpha=0.3)
    )
    scatter = _class_scatter(pool_z[:, 0], one_group) / (m - 1)
    np.testing.assert_allclose(
        a7.blocks[0].weight, _ulda_weight(scatter, d_hat, 0.3), rtol=1e-8
    )
    assert a7.fallbacks == 1
    aug_z = _standardized(data.last[1], clean).mean(axis=1)
    aug_d = aug_z[data.labels == 1].mean(0) - aug_z[data.labels == 0].mean(0)
    a8 = inv.fit_fixed(
        inv.ARMS["A8"], data, pool, inv.Candidate((1,), alpha=0.3)
    )
    scatter = sum(_class_scatter(pool_z[:, a], one_group) for a in range(k))
    np.testing.assert_allclose(
        a8.blocks[0].weight,
        _ulda_weight(scatter / (m * k - k), aug_d, 0.3),
        rtol=1e-8,
    )
    for name in ("A7", "A8"):
        tuning = inv.tune(inv.ARMS[name], data, pool, seed=0)
        assert tuning.fits == 4 * 3 + 1
        assert tuning.fallbacks == tuning.fits


def test_a7_is_plain_and_a8_augmented() -> None:
    data, pool = _labelled(signal=0.6), _pool()
    a7 = inv.tune(inv.ARMS["A7"], data, pool, seed=2)
    a8 = inv.tune(inv.ARMS["A8"], data, pool, seed=2)
    assert len(a7.candidates) == 3 * len(inv.ALPHAS)
    for c in range(len(a7.candidates)):
        assert a7.criteria[c] == inv.auroc(a7.oof[c, 0], data.labels)
        assert a8.criteria[c] == min(
            inv.auroc(a8.oof[c, k], data.labels) for k in range(4)
        )
    assert a7.fits == a8.fits == 4 * 3 + 1


def test_baselines_reuse_the_arms() -> None:
    for baseline, arm in (
        ("B3", "A1"),
        ("B4", "A2"),
        ("B8", "A3"),
        ("B9", "A4"),
    ):
        assert inv.ARMS[baseline].positions == inv.ARMS[arm].positions
        assert inv.ARMS[baseline].augmented == inv.ARMS[arm].augmented
        assert inv.ARMS[baseline].family == inv.ARMS[arm].family


def test_b1_is_the_difference_of_class_means() -> None:
    data = _labelled(signal=0.6)
    tuning = inv.tune(inv.ARMS["B1"], data, None, seed=0)
    (block,) = tuning.reader.blocks
    clean = data.last[block.index, :, 0]
    z = _standardized(clean, clean)
    d_hat = z[data.labels == 1].mean(0) - z[data.labels == 0].mean(0)
    np.testing.assert_allclose(block.weight, d_hat)
    assert tuning.reader.bias == 0.0
    assert inv.ARMS["B2"].positions == ("stim",)
    assert len(tuning.candidates) == 3


def test_b6_averages_standardized_layer_scores() -> None:
    data = _labelled(signal=0.6)
    tuning = inv.fit_depth_mean(data, seed=4)
    lam = tuning.choice.lam
    assert lam in inv.LAMBDAS
    x = (data.stim[:, :, 1], data.last[:, :, 1])
    expected = np.zeros(data.labels.size)
    for layer in range(3):
        reader = inv.fit_fixed(
            inv.ARMS["A1"], data, None, inv.Candidate((layer,), lam=lam)
        )
        fitted = reader.score(data.stim[:, :, 0], data.last[:, :, 0])
        expected += (reader.score(*x) - fitted.mean()) / fitted.std() / 3
    np.testing.assert_allclose(tuning.reader.score(*x), expected, atol=1e-9)
    for c in range(len(tuning.candidates)):
        assert tuning.criteria[c] == inv.auroc(tuning.oof[c, 0], data.labels)


def _hand_scores(
    labels: NDArray[np.int64], wrong: list[set[int]]
) -> NDArray[np.float64]:
    scores = np.where(labels == 1, 1.0, -1.0) * np.ones((len(wrong), 1))
    for layer, texts in enumerate(wrong):
        for text in texts:
            scores[layer, text] *= -1.0
    return scores


def test_b7_adds_the_layer_with_the_smallest_largest_double_fault() -> None:
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = _hand_scores(labels, [{0, 1}, {0, 1}, {2, 3}, {0, 4}])
    aurocs = np.array([0.9, 0.8, 0.8, 0.7])
    assert inv.greedy_layers(scores, labels, aurocs, size=4) == (0, 2, 3, 1)


def test_b7_breaks_double_fault_ties_by_auroc_then_depth() -> None:
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = _hand_scores(labels, [{0, 1}, {2, 3}, {4, 5}])
    higher = np.array([0.9, 0.7, 0.8])
    assert inv.greedy_layers(scores, labels, higher, size=2) == (0, 2)
    equal = np.array([0.9, 0.8, 0.8])
    assert inv.greedy_layers(scores, labels, equal, size=2) == (0, 1)
    best_is_deep = np.array([0.8, 0.7, 0.8])
    assert inv.greedy_layers(scores, labels, best_is_deep, size=1) == (0,)


def test_b7_stacks_five_layers_on_standardized_out_of_fold_scores() -> None:
    data = _labelled(signal=0.6, n_layers=6)
    stack = inv.fit_stack(data, seed=1)
    assert len(stack.layers) == 5
    assert len(set(stack.layers)) == 5
    a1 = inv.tune(inv.ARMS["A1"], data, None, seed=1)
    assert stack.layers[0] == a1.choice.layers[0]
    y = 2.0 * data.labels - 1.0
    z = (stack.oof - stack.oof_mean[:, None]) / stack.oof_scale[:, None]
    r = -y * _sigmoid(-y * (stack.weight @ z + stack.bias)) / y.size
    assert np.abs(z @ r + inv.STACK_LAMBDA * stack.weight).max() < 1e-6
    x = (data.stim[:, :, 2], data.last[:, :, 2])
    expected = np.full(y.size, stack.bias)
    for k, (layer, lam) in enumerate(
        zip(stack.layers, stack.lambdas, strict=True)
    ):
        reader = inv.fit_fixed(
            inv.ARMS["A1"], data, None, inv.Candidate((layer,), lam=lam)
        )
        standard = (reader.score(*x) - stack.oof_mean[k]) / stack.oof_scale[k]
        expected += stack.weight[k] * standard
    np.testing.assert_allclose(stack.reader.score(*x), expected, atol=1e-9)


def test_random_direction_is_the_m2_random_reader() -> None:
    from tomography.reading.reader import RepReader

    data = _labelled(signal=0.6)
    reader = inv.fit_random_direction(data, index=2, seed=11)
    clean = data.last[2, :, 0]
    z = _standardized(clean, clean)
    reference = RepReader.fit(
        "random",
        z[None],
        [12],
        labels=data.labels,
        pairs=inv.pair_matrix(data.pairs),
        seed=11,
    )
    expected = reference.score(z[None])[0]
    np.testing.assert_allclose(
        reader.score(data.stim[:, :, 0], data.last[:, :, 0]),
        expected,
        atol=1e-5,
    )
    assert np.linalg.norm(reader.blocks[0].weight) == pytest.approx(1.0)


def test_shuffled_labels_stay_within_pairs() -> None:
    data = _labelled(n_pairs=32)
    shuffled = inv.shuffle_within_pairs(data, seed=3)
    assert not np.array_equal(shuffled.labels, data.labels)
    for pair in range(32):
        assert sorted(shuffled.labels[data.pairs == pair]) == [0, 1]
    np.testing.assert_array_equal(
        shuffled.labels, inv.shuffle_within_pairs(data, seed=3).labels
    )
    assert shuffled.stim is data.stim


def test_text_only_is_the_word_tfidf_regression() -> None:
    from tomography.p9.risk import text_scores

    train = ["good film", "bad film", "good plot", "bad plot", "good", "bad"]
    labels = np.array([1, 0, 1, 0, 1, 0])
    test = ["good film plot", "bad"]
    np.testing.assert_array_equal(
        inv.text_only_scores(train, labels, test),
        text_scores(train, labels, test, analyzer="word"),
    )


def _token_acts(seed: int, n: int, layers: tuple[int, ...]) -> Activations:
    rng = np.random.default_rng(seed)
    tokens = 3
    y = np.tile(np.array([-1.0, 1.0]), n // 2)
    data = rng.normal(size=(len(layers), n * tokens, 4))
    data[0, :, 0] += np.repeat(y, tokens)
    return Activations(
        data=data.astype(np.float32),
        layers=layers,
        position="response",
        row_stimulus=np.repeat(np.arange(n), tokens),
        row_rank=np.tile(np.arange(tokens), n),
        row_token=np.tile(np.arange(tokens), n),
        row_token_id=np.zeros(n * tokens, dtype=np.int64),
        n_stimuli=n,
    )


def test_b5_tunes_the_m2_attention_reader_with_c_from_lambda() -> None:
    acts = _token_acts(0, 16, (4, 6))
    labels = np.tile(np.array([0, 1]), 8)
    pairs = np.repeat(np.arange(8), 2)
    tuning = inv.tune_attention(acts, labels, pairs, seed=2)
    assert tuning.layer in (4, 6)
    for c, (index, lam) in enumerate(tuning.candidates):
        assert lam in inv.LAMBDAS
        assert index in (0, 1)
        assert tuning.criteria[c] == inv.auroc(tuning.oof[c], labels)
    index = acts.layers.index(tuning.layer)
    single = replace(acts, data=acts.data[[index]], layers=(tuning.layer,))
    reference = fit_attention(
        single,
        labels=labels,
        pairs=inv.pair_matrix(pairs),
        c=1.0 / (16 * tuning.lam),
    )
    np.testing.assert_allclose(
        tuning.score(acts), reference.score(single)[0], atol=1e-6
    )
