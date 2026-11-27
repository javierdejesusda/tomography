"""Tests for pair-grouped ensembles."""

import numpy as np
from numpy.typing import NDArray
import pytest


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


def test_forest_scoring_does_not_need_retained_local_columns() -> None:
    from tomography.reading import ensembles

    rows = np.random.default_rng(6).normal(size=(12, 8)).astype(np.float32)
    labels = np.tile((0, 1), 6)
    pairs = np.arange(12).reshape(-1, 2)
    groups = (range(4), range(4, 8))
    full = ensembles.fit_grouped_forest(
        rows, labels, pairs, groups, m=1, n_trees=7, seed=1
    )
    compact = ensembles.fit_grouped_forest(
        rows,
        labels,
        pairs,
        groups,
        m=1,
        n_trees=7,
        seed=1,
        retain_columns=False,
    )
    assert len(full.columns) == 7
    assert compact.columns == ()
    np.testing.assert_array_equal(compact.score(rows), full.score(rows))


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


def test_stack_grids_keep_strongest_penalty_at_same_pair_accuracy() -> None:
    from tomography.reading import ensembles

    train = np.zeros((8, 2))
    labels = np.tile(np.array([0, 1]), 4)
    pairs = np.arange(8).reshape(-1, 2)
    for method, grid in (("lr", (0.01, 1.0)), ("mlp", (0.1, 0.01))):
        score, selected = ensembles.tune_stack(
            method, train, labels, train, labels, pairs, grid=grid, seed=42
        )
        assert selected == grid[0]
        assert np.ptp(score(train)) == 0


def test_forest_pair_counts_match_duplicate_rows_on_separated_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sklearn.tree import DecisionTreeClassifier

    from tomography.reading import ensembles

    rows = np.column_stack(([-4, 4, -3, 3, -2, 2, -1, 1], np.zeros(8)))
    pairs = np.arange(8).reshape(-1, 2)
    labels = np.tile([0, 1], 4)
    fit = DecisionTreeClassifier.fit
    seen = []

    def capture_fit(
        tree: DecisionTreeClassifier,
        values: np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray | None = None,
        check_input: bool = True,
    ) -> DecisionTreeClassifier:
        assert values.shape == rows.shape
        assert values.dtype == np.float32
        assert values.flags.f_contiguous
        assert not check_input
        assert sample_weight is not None
        np.testing.assert_array_equal(sample_weight[::2], sample_weight[1::2])
        repeated = np.repeat(np.arange(len(labels)), sample_weight.astype(int))
        duplicate = DecisionTreeClassifier(
            max_features="sqrt", random_state=tree.random_state
        )
        fit(duplicate, values[repeated], target[repeated])
        seen.append(duplicate.predict_proba(rows)[:, 1])
        return fit(tree, values, target, sample_weight, check_input)

    monkeypatch.setattr(DecisionTreeClassifier, "fit", capture_fit)
    settings = {"m": 1, "n_trees": 5, "seed": 11}
    first = ensembles.fit_grouped_forest(
        rows, labels, pairs, ((0, 1),), **settings
    )
    np.testing.assert_array_equal(first.score(rows), np.mean(seen, axis=0))
    second = ensembles.fit_grouped_forest(
        rows, labels, pairs, ((0, 1),), **settings
    )
    np.testing.assert_array_equal(first.score(rows), second.score(rows))


def test_forest_rejects_nonfinite_features_before_training() -> None:
    from tomography.reading import ensembles

    rows = np.array([[0.0], [1.0], [np.nan], [2.0]])
    with pytest.raises(ValueError, match="finite"):
        ensembles.fit_grouped_forest(
            rows,
            np.array([0, 1, 0, 1]),
            np.arange(4).reshape(-1, 2),
            ((0,),),
            m=1,
            n_trees=2,
            seed=3,
        )
