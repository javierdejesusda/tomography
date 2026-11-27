"""Pair-grouped cross-fitting and random-subspace ensembles."""

from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from tomography.reading.metrics import FloatArray
from tomography.reading.metrics import group_accuracy
from tomography.reading.metrics import parse_groups
from tomography.reading.probes import fit_mlp

Indices = NDArray[np.int64]
Scores = NDArray[np.float64]


def pair_folds(
    pairs: Indices, *, n_rows: int, n_folds: int, seed: int
) -> Iterator[tuple[Indices, Indices]]:
    """Yields complementary fit and held-out rows for shuffled whole pairs.

    Args:
        pairs: Two indices per pair, covering each row exactly once.
        n_rows: Number of rows in the training matrix.
        n_folds: Number of nonempty folds; sizes differ by at most one pair.
        seed: Seed used to shuffle the pairs.

    Yields:
        The fit and held-out row indices of each fold, in input row order.

    Raises:
        ValueError: If the pairs do not partition the rows or the fold
            count cannot produce at least two nonempty folds.
    """
    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or not np.issubdtype(pairs.dtype, np.integer)
        or not np.array_equal(np.sort(pairs, axis=None), np.arange(n_rows))
    ):
        raise ValueError("pairs must cover every row exactly once")
    if not 2 <= n_folds <= len(pairs):
        raise ValueError("n_folds must be between 2 and the number of pairs")
    order = np.random.default_rng(seed).permutation(len(pairs))
    for fold in np.array_split(order, n_folds):
        held = np.zeros(n_rows, dtype=bool)
        held[pairs[fold].ravel()] = True
        yield np.flatnonzero(~held), np.flatnonzero(held)


def cross_fit_scores(
    pairs: Indices,
    n_rows: int,
    select: Scores,
    report: Scores,
    score_fold: Callable[[Indices, Indices], Scores],
    *,
    n_folds: int,
    seed: int,
) -> tuple[Scores, Scores, Scores]:
    """Standardizes train, select and report using out-of-fold scores.

    Args:
        pairs: Training row indices of each pair.
        n_rows: Number of training rows.
        select: Full-fit select scores shaped (n_select, n_layers).
        report: Full-fit report scores with layers on the last axis.
        score_fold: Refits on its first argument and scores its second,
            returning a matrix with one column per layer.
        n_folds: Number of pair-grouped folds.
        seed: Random seed for the fold assignment.

    Returns:
        Standardized out-of-fold, select and report scores. Constant
        out-of-fold columns use a scale of one.
    """
    out = np.empty((n_rows, select.shape[1]), dtype=np.float64)
    for fit, held in pair_folds(
        pairs, n_rows=n_rows, n_folds=n_folds, seed=seed
    ):
        out[held] = score_fold(fit, held)
    center = out.mean(axis=0)
    deviation = out.std(axis=0)
    scale = np.where(deviation > 0, deviation, 1.0)
    return (
        (out - center) / scale,
        (select - center) / scale,
        (report - center) / scale,
    )


def weighted_mean(scores: Scores, accuracy: Scores) -> Scores:
    """Averages standardized scores weighted by accuracy above chance.

    Args:
        scores: Standardized scores with layers on the last axis.
        accuracy: Select pair accuracy for each layer.

    Returns:
        Weighted scores; an equal mean if no layer beats chance.
    """
    weights = np.maximum(accuracy - 0.5, 0)
    if not weights.any():
        return scores.mean(axis=-1)
    return np.asarray(np.average(scores, axis=-1, weights=weights))


@dataclass(frozen=True)
class _Nodes:
    """A tree with its split columns mapped back to the input matrix."""

    feature: Indices
    threshold: Scores
    left: Indices
    right: Indices
    probability: Scores

    def score(self, rows: NDArray[np.float32]) -> Scores:
        """Walks one node per live row until every row reaches a leaf."""
        indices = np.arange(rows.shape[0])
        nodes = np.zeros(len(indices), dtype=np.int64)
        while indices.size:
            current = nodes[indices]
            splits = self.feature[current]
            live = splits >= 0
            indices, current, splits = (
                indices[live],
                current[live],
                splits[live],
            )
            go_left = rows[indices, splits] <= self.threshold[current]
            nodes[indices] = np.where(
                go_left, self.left[current], self.right[current]
            )
        return self.probability[nodes]


@dataclass(frozen=True)
class GroupedForest:
    """Pair-bootstrap trees over shared groups of input columns.

    Attributes:
        trees: The fitted sklearn trees, in their local column order.
        columns: Global columns for replaying the fitted trees, or empty
            when only global-node scoring is needed.
    """

    trees: tuple[DecisionTreeClassifier, ...]
    columns: tuple[Indices, ...]
    _nodes: tuple[_Nodes, ...]

    def score(self, rows: FloatArray) -> Scores:
        """Returns mean positive probabilities without per-tree slicing.

        Args:
            rows: A 2D feature matrix in the original global column order.

        Returns:
            The mean positive probability per row.
        """
        features = np.asarray(rows, dtype=np.float32)
        result = np.zeros(features.shape[0], dtype=np.float64)
        for nodes in self._nodes:
            result += nodes.score(features)
        return result / len(self._nodes)


def fit_grouped_forest(
    rows: FloatArray,
    labels: Indices,
    pairs: Indices,
    groups: Sequence[Sequence[int]],
    *,
    m: int,
    n_trees: int,
    seed: int,
    retain_columns: bool = True,
) -> GroupedForest:
    """Fits trees to bootstrapped pairs and random whole feature groups.

    Args:
        rows: Training feature rows.
        labels: Binary labels, with one positive per pair.
        pairs: Indices of both rows in every training pair.
        groups: Global feature column indices of each group (e.g., layer).
        m: Number of groups drawn without replacement for each tree.
        n_trees: Number of trees in the forest.
        seed: Controls the bootstrap, subspace and split draws.
        retain_columns: Keep each tree's full local-to-global mapping for
            sklearn replay. Global-node scoring does not require it.

    Returns:
        A forest with precomputed global split columns.

    Raises:
        ValueError: If there are no trees, m is out of range, rows are not
            finite, labels or pairs are invalid, or a column is out of range.
    """
    if not 1 <= m <= len(groups) or n_trees < 1:
        raise ValueError("forest needs positive n_trees and m within groups")
    features = np.asarray(rows, dtype=np.float32, order="F")
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("forest features must be a finite 2D matrix")
    if (
        labels.shape != (features.shape[0],)
        or not np.isin(labels, (0, 1)).all()
    ):
        raise ValueError("forest needs one binary label per row")
    index = parse_groups(pairs, labels)
    if pairs.shape[1] != 2 or not index.valid.all():
        raise ValueError("forest needs two rows per pair")
    rng = np.random.default_rng(seed)
    layer_columns = tuple(np.asarray(group) for group in groups)
    for group in layer_columns:
        if (
            group.ndim != 1
            or group.size == 0
            or not np.issubdtype(group.dtype, np.integer)
            or group.min() < 0
            or group.max() >= features.shape[1]
        ):
            raise ValueError("forest groups must contain valid column indices")
    scratch = None
    trees = []
    columns = []
    nodes = []
    for _ in range(n_trees):
        bootstrap = pairs[rng.integers(len(pairs), size=len(pairs))].ravel()
        subset = np.sort(rng.choice(len(groups), m, replace=False))
        used = np.concatenate([layer_columns[i] for i in subset])
        if np.all(np.diff(used) == 1):
            values = features[:, used[0] : used[-1] + 1]
        else:
            shape = (len(used), len(labels))
            if scratch is None or scratch.shape != shape:
                scratch = np.empty(shape, dtype=np.float32)
            # Gather columns into a reusable buffer; validation proved
            # every index is in range, so take need not buffer for errors.
            np.take(features.T, used, axis=0, out=scratch, mode="clip")
            values = scratch.T
        weights = np.bincount(bootstrap, minlength=len(labels)).astype(
            np.float64
        )
        tree = DecisionTreeClassifier(
            max_features="sqrt", random_state=int(rng.integers(2**31))
        ).fit(values, labels, sample_weight=weights, check_input=False)
        structure = tree.tree_
        feature = np.full(structure.node_count, -1, dtype=np.int64)
        split = structure.feature >= 0
        feature[split] = used[structure.feature[split]]
        counts = structure.value[:, 0, :]
        probability = counts[:, 1] / counts.sum(axis=1)
        trees.append(tree)
        if retain_columns:
            columns.append(used)
        nodes.append(
            _Nodes(
                feature,
                structure.threshold,
                structure.children_left,
                structure.children_right,
                probability,
            )
        )
    return GroupedForest(tuple(trees), tuple(columns), tuple(nodes))


def tune_grouped_forest(
    train: FloatArray,
    labels: Indices,
    pairs: Indices,
    select: FloatArray,
    select_labels: Indices,
    select_pairs: Indices,
    groups: Sequence[Sequence[int]],
    *,
    grid: Sequence[int],
    n_trees: int,
    seed: int,
) -> tuple[int, dict[int, GroupedForest]]:
    """Fits a grid once and chooses m by select pair accuracy.

    Args:
        train: Training feature matrix.
        labels: Binary training labels.
        pairs: Training row indices grouped by pair.
        select: Validation feature matrix.
        select_labels: Binary validation labels.
        select_pairs: Validation row indices grouped by pair.
        groups: Column indices of the feature groups.
        grid: Candidate group counts.
        n_trees: Trees per candidate.
        seed: Shared random seed for the fits.

    Returns:
        The winning m (smallest on ties) and the fits by m.
    """
    fits = {
        m: fit_grouped_forest(
            train, labels, pairs, groups, m=m, n_trees=n_trees, seed=seed
        )
        for m in sorted(set(grid))
    }
    best = max(
        fits,
        key=lambda m: float(
            group_accuracy(fits[m].score(select), select_pairs, select_labels)
        ),
    )
    return best, fits


def fit_mlp_stack(
    train: Scores, labels: Indices, alpha: float, seed: int
) -> Callable[[Scores], Scores]:
    """Fits the single-hidden-layer M2 MLP to standardized layer scores.

    Args:
        train: Out-of-fold scores, one column per layer.
        labels: Training labels.
        alpha: L2 penalty.
        seed: Seed of the network's initialization.

    Returns:
        The fitted score function.
    """
    fit = fit_mlp(
        train[None],
        (1,),
        labels=labels,
        hidden_sizes=(16,),
        alpha=alpha,
        seed=seed,
    )

    def score(matrix: Scores) -> Scores:
        return np.asarray(fit.score(matrix[None])[0], dtype=np.float64)

    return score


def tune_stack(
    method: Literal["lr", "mlp"],
    train: Scores,
    labels: Indices,
    select: Scores,
    select_labels: Indices,
    select_pairs: Indices,
    *,
    grid: Sequence[float],
    seed: int,
) -> tuple[Callable[[Scores], Scores], float]:
    """Tunes a fitted stack using select pair accuracy.

    Args:
        method: The logistic or single-hidden-layer aggregator.
        train: Standardized out-of-fold training scores.
        labels: Binary training labels.
        select: Standardized select scores from full-data row fits.
        select_labels: Binary select labels.
        select_pairs: Index pairs into the select scores.
        grid: Penalties listed from strongest to weakest.
        seed: Network initialization for each MLP candidate.

    Returns:
        The selected score function and penalty, first on ties.
    """
    if method == "lr":
        fits = [
            LogisticRegression(C=value, max_iter=1000).fit(train, labels)
            for value in grid
        ]
        scorers: list[Callable[[Scores], Scores]] = [
            fit.decision_function for fit in fits
        ]
    else:
        scorers = [fit_mlp_stack(train, labels, alpha, seed) for alpha in grid]
    scores = np.stack([score(select) for score in scorers])
    best = int(np.argmax(group_accuracy(scores, select_pairs, select_labels)))
    return scorers[best], grid[best]
