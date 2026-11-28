"""Tests of the invariance readers of the preregistration, sections 4 and 5."""

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.reading import invariance as inv


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
