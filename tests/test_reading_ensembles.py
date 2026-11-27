"""Tests for pair-grouped ensembles."""

import numpy as np
from numpy.typing import NDArray


def test_pair_folds_hold_out_each_pair_exactly_once() -> None:
    from tomography.reading import ensembles

    pairs = np.column_stack([np.arange(64), np.arange(127, 63, -1)])
    seen = []
    for fit, held in ensembles.pair_folds(pairs, n_rows=128, n_folds=8, seed=4):
        assert len(held) == 16
        assert len(fit) == 112
        assert not np.intersect1d(fit, held).size
        held_mask = np.isin(pairs, held)
        assert (held_mask[:, 0] == held_mask[:, 1]).all()
        seen.extend(held)
    assert sorted(seen) == list(range(128))


def test_cross_fit_scales_other_splits_using_only_held_out_scores() -> None:
    from tomography.reading import ensembles

    pairs = np.array([[0, 3], [1, 2]])

    def score(
        fit: NDArray[np.int64], held: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        assert not np.intersect1d(fit, held).size
        out = np.array([1.0, 3.0, 5.0, 7.0])
        return np.column_stack((out[held], np.ones(held.size)))

    train, select, report = ensembles.cross_fit_scores(
        pairs,
        4,
        np.array([[4.0, 2.0], [9.0, 1.0]]),
        np.array([[14.0, 3.0]]),
        score,
        n_folds=2,
        seed=2,
    )
    np.testing.assert_allclose(
        train[:, 0], [-3 / 5**0.5, -1 / 5**0.5, 1 / 5**0.5, 3 / 5**0.5]
    )
    np.testing.assert_allclose(select, [[0, 1], [5 / 5**0.5, 0]])
    np.testing.assert_allclose(report, [[10 / 5**0.5, 2]])


def test_weighted_mean_discounts_chance_and_below() -> None:
    from tomography.reading import ensembles

    scores = np.array([[1.0, 5.0, 100.0, 100.0]])
    result = ensembles.weighted_mean(scores, np.array([1.0, 0.75, 0.5, 0.0]))
    np.testing.assert_allclose(result, [7 / 3])


def test_forest_global_node_lookup_matches_tree_probabilities() -> None:
    from tomography.reading import ensembles

    rng = np.random.default_rng(23)
    train = rng.normal(size=(16, 7))
    labels = np.tile(np.array([0, 1]), 8)
    columns = ((0, 1), (2, 3, 4), (5, 6))
    forest = ensembles.fit_grouped_forest(
        train,
        labels,
        np.arange(16).reshape(-1, 2),
        columns,
        m=2,
        n_trees=12,
        seed=1,
    )
    report = rng.normal(size=(17, 7))
    expected = []
    for tree, used in zip(forest.trees, forest.columns, strict=True):
        assert sum(set(group) <= set(used) for group in columns) == 2
        assert tuple(tree.classes_) == (0, 1)
        expected.append(tree.predict_proba(report[:, used])[:, 1])
    np.testing.assert_allclose(forest.score(report), np.mean(expected, axis=0))


def test_forest_tuning_keeps_smallest_subspace_on_a_tie() -> None:
    from tomography.reading import ensembles

    rows = np.zeros((8, 3))
    labels = np.tile(np.array([0, 1]), 4)
    pairs = np.arange(8).reshape(-1, 2)
    chosen, fits = ensembles.tune_grouped_forest(
        rows,
        labels,
        pairs,
        rows,
        labels,
        pairs,
        ((0,), (1,), (2,)),
        grid=(3, 1, 2),
        n_trees=2,
        seed=5,
    )
    assert chosen == 1
    np.testing.assert_array_equal(fits[chosen].score(rows), np.full(8, 0.5))
