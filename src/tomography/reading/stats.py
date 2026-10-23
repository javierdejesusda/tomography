"""Paired comparisons, threshold metrics and layer choices for readers.

Two readers evaluated on the same items are compared through the difference
of their metrics: a bootstrap that resamples whole clusters of rows, so rows
of one item are never treated as independent; DeLong's test for two
correlated AUROCs; McNemar's exact test for two classifiers' outcomes; and
Holm's correction over a declared family of such tests. The thresholded
metrics (accuracy, precision, recall and F1, and the true positive rate at a
fixed false positive rate) take their threshold from validation scores, and
the two layer choices, the best validation layer and an ensemble of the top
k, rank the layers by a validation metric.
"""

from collections.abc import Iterator
from collections.abc import Sequence
import math
from typing import Any
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest
from scipy.stats import norm
from scipy.stats import rankdata

from tomography.reading.metrics import FloatArray
from tomography.reading.metrics import auroc
from tomography.reading.metrics import best_layer

NO_RESAMPLE = "no resample holds both classes"


class PairedDiff(NamedTuple):
    """A metric difference between two scorers with a bootstrap interval.

    Attributes:
        difference: The metric of the first scorer minus the second's.
        lower: Lower bound of the percentile interval.
        upper: Upper bound of the percentile interval.
        n_resamples: Resamples behind the interval.
    """

    difference: float
    lower: float
    upper: float
    n_resamples: int


class DelongResult(NamedTuple):
    """DeLong's comparison of two correlated AUROCs.

    Attributes:
        difference: The AUROC of the first scorer minus the second's.
        standard_error: Standard error of the difference.
        z: The difference over its standard error.
        p_value: Two-sided p value of the null of equal AUROCs.
    """

    difference: float
    standard_error: float
    z: float
    p_value: float


class McNemarResult(NamedTuple):
    """McNemar's comparison of two classifiers on the same items.

    Attributes:
        a_only: Items only the first classifier gets right.
        b_only: Items only the second classifier gets right.
        p_value: Two-sided p value of the null of equal accuracy.
    """

    a_only: int
    b_only: int
    p_value: float


class HolmResult(NamedTuple):
    """Holm-adjusted p values of a family, in input order.

    Attributes:
        adjusted: Adjusted p values, shape (m,).
        reject: Whether each null hypothesis is rejected, shape (m,).
    """

    adjusted: NDArray[np.float64]
    reject: NDArray[np.bool_]


class ThresholdMetrics(NamedTuple):
    """Test metrics of a scorer at a threshold chosen on validation.

    Attributes:
        threshold: Scores above it are called positive.
        accuracy: Share of test rows called correctly.
        precision: Share of positive calls that are positives.
        recall: Share of positives called positive.
        f1: Harmonic mean of precision and recall.
    """

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float


class TprAtFpr(NamedTuple):
    """Test rates at a threshold set on the validation negatives.

    Attributes:
        true_positive_rate: Share of test positives scoring above the
            threshold.
        false_positive_rate: Share of test negatives scoring above the
            threshold.
        threshold: Scores above it are called positive.
        negatives_above: Validation negatives scoring above the threshold,
            the tail the threshold rests on.
    """

    true_positive_rate: float
    false_positive_rate: float
    threshold: float
    negatives_above: int


class LayerChoice(NamedTuple):
    """Per-stimulus scores read at the chosen layers.

    Attributes:
        layers: The chosen layer indices, best validation metric first.
        val_scores: Validation score of each stimulus, shape (n_val,).
        test_scores: Test score of each stimulus, shape (n_test,).
    """

    layers: tuple[int, ...]
    val_scores: NDArray[np.float64]
    test_scores: NDArray[np.float64]


def paired_auroc_diff(
    a: FloatArray,
    b: FloatArray,
    labels: NDArray[np.int64],
    clusters: NDArray[np.int64],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedDiff:
    """AUROC of one scorer minus another's, with a clustered bootstrap.

    Both scorers are evaluated on the same resampled rows, so the interval
    keeps their correlation. A resample draws as many clusters as there are,
    with replacement, and keeps every row of a drawn cluster: rows that share
    an item (the members of a pair, the answers of a question) are never
    treated as independent. Clusters are drawn by position in increasing id
    order, so with one row per cluster and ids increasing along the rows,
    the resamples are those of the item bootstrap
    `tomography.p9.risk.auroc_diff_ci` with the same seed and `n_boot`. A
    resample that holds a single class has no AUROC and is skipped. A single
    cluster gives a zero-width interval.

    Args:
        a: Scores of the first scorer, shape (n,); higher should mean
            label 1.
        b: Scores of the second scorer on the same rows, shape (n,).
        labels: Binary labels of shape (n,), both classes present.
        clusters: Integer cluster id of each row, shape (n,): the item the
            row belongs to, such as its pair or question.
        n_boot: Number of resamples drawn.
        alpha: One minus the coverage; 0.05 gives a 95% interval.
        seed: Seed of the resampling.

    Returns:
        The difference on all rows, the percentile interval of the resampled
        differences and the number of resamples kept.

    Raises:
        ValueError: If the shapes disagree, a label is not 0 or 1, a class
            is missing, a score is not finite, the clusters are not
            integers, `n_boot` or `alpha` is out of range, or no resample
            holds both classes.
    """
    _check_resampling(n_boot, alpha)
    pair, ids = _paired_rows(a, b, clusters)
    labels = np.asarray(labels)
    point = auroc(pair, labels)
    diffs: list[float] = []
    for rows in _resampled_rows(ids, n_boot, seed):
        drawn = labels[rows]
        if drawn.min() == drawn.max():
            continue
        values = auroc(pair[:, rows], drawn)
        diffs.append(float(values[0] - values[1]))
    if not diffs:
        raise ValueError(NO_RESAMPLE)
    return _summary(point, diffs, alpha)


def paired_accuracy_diff(
    correct_a: NDArray[np.bool_] | FloatArray,
    correct_b: NDArray[np.bool_] | FloatArray,
    clusters: NDArray[np.int64],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedDiff:
    """Accuracy of one scorer minus another's, with a clustered bootstrap.

    Accuracy is the mean credit over rows: 1 for a correct row, 0 for a
    wrong one, and a fraction where tied scores share the credit. Give one
    row per decision: per stimulus for accuracy at a threshold, or per group
    with the group's credit for pair and multiple-choice accuracy.
    Resampling follows `paired_auroc_diff`, and the accuracy of a resample
    pools the rows of its clusters, so a larger cluster weighs more. A
    single cluster gives a zero-width interval.

    Args:
        correct_a: Credit of the first scorer on each row, in [0, 1], shape
            (n,).
        correct_b: Credit of the second scorer on the same rows, shape (n,).
        clusters: Integer cluster id of each row, shape (n,).
        n_boot: Number of resamples drawn.
        alpha: One minus the coverage; 0.05 gives a 95% interval.
        seed: Seed of the resampling.

    Returns:
        The difference on all rows, the percentile interval of the resampled
        differences and the number of resamples, all of them kept.

    Raises:
        ValueError: If the shapes disagree, there is no row, a credit is
            outside [0, 1], the clusters are not integers, or `n_boot` or
            `alpha` is out of range.
    """
    _check_resampling(n_boot, alpha)
    pair, ids = _paired_rows(correct_a, correct_b, clusters)
    if not ((pair >= 0.0) & (pair <= 1.0)).all():
        raise ValueError("credit must lie in [0, 1]")
    diffs: list[float] = []
    for rows in _resampled_rows(ids, n_boot, seed):
        values = pair[:, rows].mean(axis=1)
        diffs.append(float(values[0] - values[1]))
    return _summary(pair.mean(axis=1), diffs, alpha)


def delong_test(
    a: FloatArray, b: FloatArray, labels: NDArray[np.int64]
) -> DelongResult:
    """DeLong's test for two correlated AUROCs measured on the same rows.

    A scorer's AUROC is the mean over positives of its structural component
    V10, the share of negatives a positive outscores with ties counting
    half, and the mean over negatives of V01, the share of positives that
    outscore a negative. Following DeLong, DeLong and Clarke-Pearson (1988),
    the variance of the difference is var(V10_a - V10_b) / n_pos +
    var(V01_a - V01_b) / n_neg with sample variances; the components come
    from midranks (Sun and Xu, 2014). The test treats rows as independent:
    for rows that share an item, use `paired_auroc_diff`. A zero standard
    error gives z = 0 and p = 1 when the AUROCs are equal, and an infinite z
    with p = 0 otherwise, as R's pROC does.

    Args:
        a: Scores of the first scorer, shape (n,); higher should mean
            label 1.
        b: Scores of the second scorer on the same rows, shape (n,).
        labels: Binary labels of shape (n,), with at least two rows of each
            class.

    Returns:
        The AUROC difference, its standard error, the z statistic and the
        two-sided normal p value.

    Raises:
        ValueError: If the shapes disagree, a label is not 0 or 1, a score
            is not finite, or a class has fewer than two rows.
    """
    pair = _pair(a, b)
    labels = np.asarray(labels)
    aucs = auroc(pair, labels)
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = labels.shape[0] - n_positive
    if min(n_positive, n_negative) < 2:
        raise ValueError(
            "DeLong's test needs at least two positives and two negatives"
        )
    over_positives = []
    over_negatives = []
    for scores in pair:
        overall = rankdata(scores)
        over_positives.append(
            (overall[positive] - rankdata(scores[positive])) / n_negative
        )
        over_negatives.append(
            1.0
            - (overall[~positive] - rankdata(scores[~positive])) / n_positive
        )
    variance = (
        np.var(over_positives[0] - over_positives[1], ddof=1) / n_positive
        + np.var(over_negatives[0] - over_negatives[1], ddof=1) / n_negative
    )
    difference = float(aucs[0] - aucs[1])
    standard_error = float(np.sqrt(variance))
    if standard_error > 0.0:
        z = difference / standard_error
    else:
        z = 0.0 if difference == 0.0 else math.copysign(math.inf, difference)
    return DelongResult(
        difference, standard_error, z, float(2 * norm.sf(abs(z)))
    )


def mcnemar_test(
    correct_a: NDArray[np.bool_] | NDArray[np.integer[Any]],
    correct_b: NDArray[np.bool_] | NDArray[np.integer[Any]],
) -> McNemarResult:
    """McNemar's test for two classifiers judged on the same items.

    Only the discordant items, right for one classifier and wrong for the
    other, carry information. Under the null of equal accuracy each of them
    favors either classifier with probability 1/2, so this is the exact
    two-sided binomial test of `a_only` successes in `a_only + b_only`
    trials (the conditional exact McNemar test). It is used instead of the
    chi-square approximation, with or without continuity correction,
    because discordant counts can be a few dozen or fewer, where that
    approximation is unreliable; the exact test is valid at any count and
    slightly conservative. Give one outcome per independent item (a pair, a
    question), not one per row of a clustered item. Without discordant
    items the p value is 1.

    Args:
        correct_a: Whether the first classifier is right on each item,
            shape (n,), as booleans or 0 and 1.
        correct_b: Whether the second classifier is right on each item.

    Returns:
        The discordant counts and the two-sided exact p value.

    Raises:
        ValueError: If the shapes disagree, there is no item, or an outcome
            is not 0 or 1.
    """
    pair = _pair(correct_a, correct_b)
    if not np.isin(pair, (0.0, 1.0)).all():
        raise ValueError("outcomes must be 0 or 1 (or booleans)")
    first, second = pair.astype(bool)
    a_only = int((first & ~second).sum())
    b_only = int((~first & second).sum())
    if a_only + b_only == 0:
        return McNemarResult(0, 0, 1.0)
    p_value = binomtest(a_only, a_only + b_only, 0.5).pvalue
    return McNemarResult(a_only, b_only, float(p_value))


def holm_correction(p_values: FloatArray, *, alpha: float = 0.05) -> HolmResult:
    """Holm's step-down correction over a declared family of p values.

    Sorted in increasing order, the i-th of m p values (from 1) is
    multiplied by m - i + 1 and capped at 1, and a running maximum keeps the
    adjusted values in the order of the raw ones. A hypothesis is rejected
    when its adjusted p value is at most `alpha`, which holds the
    family-wise error rate at `alpha` under any dependence between the
    tests. Declare the family before looking at the results: the
    correction loses power as the family grows.

    Args:
        p_values: The p values of the family, shape (m,) with m >= 1.
        alpha: The family-wise error rate.

    Returns:
        The adjusted p values and the rejections, both in input order.

    Raises:
        ValueError: If `p_values` is empty or not a vector, a p value is
            outside [0, 1], or `alpha` is not in (0, 1).
    """
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] == 0:
        raise ValueError(
            "p_values must be a non-empty vector, the declared family; got "
            f"shape {values.shape}"
        )
    if not ((values >= 0.0) & (values <= 1.0)).all():
        raise ValueError("p values must lie in [0, 1]")
    _check_alpha(alpha)
    n_tests = values.shape[0]
    order = np.argsort(values, kind="stable")
    scaled = np.minimum(1.0, (n_tests - np.arange(n_tests)) * values[order])
    adjusted = np.empty(n_tests)
    adjusted[order] = np.maximum.accumulate(scaled)
    return HolmResult(adjusted, adjusted <= alpha)


def threshold_metrics(
    val_scores: FloatArray,
    val_labels: NDArray[np.int64],
    test_scores: FloatArray,
    test_labels: NDArray[np.int64],
) -> ThresholdMetrics:
    """Classifies test rows at a threshold chosen on validation scores.

    A score above the threshold is called positive. The threshold maximizes
    the validation balanced accuracy, the mean of the true positive and true
    negative rates, which weighs both classes equally whatever their
    balance. It equals (1 + J) / 2 for Youden's J, the true positive rate
    minus the false positive rate, so the rule also maximizes J. The
    candidates split the sorted distinct validation scores, and the
    threshold is the midpoint of the two scores either side of the best
    split, or minus infinity when calling every validation row positive is
    best. Ties go to the lowest threshold. Precision is 0 when no test row
    is called positive, as with scikit-learn's `zero_division=0`.

    Args:
        val_scores: Validation scores, shape (n_val,).
        val_labels: Validation labels of 0 and 1, shape (n_val,), both
            classes present.
        test_scores: Test scores, shape (n_test,).
        test_labels: Test labels of 0 and 1, shape (n_test,), both classes
            present.

    Returns:
        The threshold and the test accuracy, precision, recall and F1.

    Raises:
        ValueError: If a split's scores and labels differ in length, a label
            is not 0 or 1, a score is not finite, or a split misses a class.
    """
    val, val_positive = _labelled(val_scores, val_labels, "validation")
    _check_both_classes(val_positive, "validation")
    test, test_positive = _labelled(test_scores, test_labels, "test")
    _check_both_classes(test_positive, "test")
    threshold = _balanced_threshold(val, val_positive)
    called = test > threshold
    tp = int((called & test_positive).sum())
    fp = int((called & ~test_positive).sum())
    fn = int((~called & test_positive).sum())
    return ThresholdMetrics(
        threshold,
        float((called == test_positive).mean()),
        tp / (tp + fp) if tp + fp else 0.0,
        tp / (tp + fn),
        2 * tp / (2 * tp + fp + fn),
    )


def tpr_at_fpr(
    val_scores: FloatArray,
    val_labels: NDArray[np.int64],
    test_scores: FloatArray,
    test_labels: NDArray[np.int64],
    *,
    max_fpr: float = 0.01,
) -> TprAtFpr:
    """True positive rate on test at a false positive rate set on validation.

    Only the validation negatives set the threshold. With n of them, k is
    the largest count with k / n <= `max_fpr`, and the threshold is the
    (k + 1)-th highest negative score. A score above the threshold is called
    positive, so at most k validation negatives are (fewer when scores tie
    at the threshold) and the validation false positive rate never exceeds
    `max_fpr`. The threshold rests on few items: at 1%, 256 negatives allow
    2 and fewer than 100 allow none, making the threshold the highest
    negative; `negatives_above` reports that tail.

    Args:
        val_scores: Validation scores, shape (n_val,).
        val_labels: Validation labels of 0 and 1, shape (n_val,), with at
            least one negative.
        test_scores: Test scores, shape (n_test,).
        test_labels: Test labels of 0 and 1, shape (n_test,), both classes
            present.
        max_fpr: The false positive rate allowed on validation, in (0, 1).

    Returns:
        The test true and false positive rates, the threshold and the number
        of validation negatives above it.

    Raises:
        ValueError: If a split's scores and labels differ in length, a label
            is not 0 or 1, a score is not finite, validation has no
            negative, test misses a class, or `max_fpr` is not in (0, 1).
    """
    if not 0.0 < max_fpr < 1.0:
        raise ValueError(f"max_fpr must be in (0, 1), got {max_fpr}")
    val, val_positive = _labelled(val_scores, val_labels, "validation")
    if val_positive.all():
        raise ValueError("validation labels must contain a negative")
    test, test_positive = _labelled(test_scores, test_labels, "test")
    _check_both_classes(test_positive, "test")
    negatives = np.sort(val[~val_positive])[::-1]
    n_negative = negatives.shape[0]
    rates = np.arange(n_negative + 1) / n_negative
    allowed = int(np.searchsorted(rates, max_fpr, side="right")) - 1
    threshold = float(negatives[allowed])
    called = test > threshold
    return TprAtFpr(
        float(called[test_positive].mean()),
        float(called[~test_positive].mean()),
        threshold,
        int((negatives > threshold).sum()),
    )


def best_validation_layer(
    val_metric: FloatArray,
    layers: Sequence[int],
    val_scores: FloatArray,
    test_scores: FloatArray,
) -> LayerChoice:
    """Reads the scores of the layer with the best validation metric.

    The layer is the first maximum of `val_metric`, as chosen by
    `tomography.reading.metrics.best_layer`. Its raw scores are returned: a
    single layer needs no standardization.

    Args:
        val_metric: Validation metric of each layer, shape (K,); higher is
            better, for example pair accuracy or AUROC.
        layers: The distinct layer index of each entry of `val_metric`.
        val_scores: Validation score of each stimulus per layer, shape
            (K, n_val).
        test_scores: Test score of each stimulus per layer, shape
            (K, n_test).

    Returns:
        The chosen layer with its validation and test scores.

    Raises:
        ValueError: If the shapes disagree, a split has no stimulus,
            `val_metric` holds NaN, a layer repeats, or a score is not
            finite.
    """
    metric = _layer_metric(val_metric, layers)
    val = _per_layer(val_scores, "val_scores", len(layers))
    test = _per_layer(test_scores, "test_scores", len(layers))
    layer = best_layer(metric, layers)
    position = list(layers).index(layer)
    return LayerChoice((layer,), val[position], test[position])


def top_k_layers(
    val_metric: FloatArray,
    layers: Sequence[int],
    val_scores: FloatArray,
    test_scores: FloatArray,
    *,
    k: int = 5,
) -> LayerChoice:
    """Averages the standardized scores of the k best validation layers.

    The layers are the k highest entries of `val_metric`, ties going to the
    earlier entry, so the first is the one `best_validation_layer` reads.
    The scores of each chosen layer, validation and test alike, are
    standardized with that layer's validation mean and population standard
    deviation, so no test information enters, and the ensemble score of a
    stimulus is
    their mean over the chosen layers. Standardizing keeps a layer with
    larger scores from dominating the mean. The ensemble averages scores,
    not per-layer metrics.

    Args:
        val_metric: Validation metric of each layer, shape (K,); higher is
            better, for example pair accuracy or AUROC.
        layers: The distinct layer index of each entry of `val_metric`.
        val_scores: Validation score of each stimulus per layer, shape
            (K, n_val).
        test_scores: Test score of each stimulus per layer, shape
            (K, n_test).
        k: Number of layers averaged, between 1 and K.

    Returns:
        The chosen layers, best first, with the ensemble validation and test
        scores.

    Raises:
        ValueError: If the shapes disagree, a split has no stimulus,
            `val_metric` holds NaN, a layer repeats, a score is not finite,
            `k` is out of range, or a chosen layer has constant validation
            scores.
    """
    metric = _layer_metric(val_metric, layers)
    val = _per_layer(val_scores, "val_scores", len(layers))
    test = _per_layer(test_scores, "test_scores", len(layers))
    if not 1 <= k <= len(layers):
        raise ValueError(
            f"k must be between 1 and the number of layers ({len(layers)}), "
            f"got {k}"
        )
    chosen = np.argsort(-metric, kind="stable")[:k]
    picked = val[chosen]
    # A constant row can have a computed deviation of about 1e-17 instead of
    # 0, so constancy is tested on the range.
    if (np.ptp(picked, axis=1) == 0.0).any():
        raise ValueError("a chosen layer has constant validation scores")
    mean = picked.mean(axis=1, keepdims=True)
    std = picked.std(axis=1, keepdims=True)
    return LayerChoice(
        tuple(int(layers[i]) for i in chosen),
        ((picked - mean) / std).mean(axis=0),
        ((test[chosen] - mean) / std).mean(axis=0),
    )


def _balanced_threshold(
    scores: NDArray[np.float64], positive: NDArray[np.bool_]
) -> float:
    values, inverse = np.unique(scores, return_inverse=True)
    n_positive = int(positive.sum())
    n_negative = positive.shape[0] - n_positive
    positive_counts = np.bincount(inverse[positive], minlength=values.shape[0])
    negative_counts = np.bincount(inverse[~positive], minlength=values.shape[0])
    # Split k calls the k lowest distinct scores negative. Calling every score
    # negative is left out: it ties with calling every score positive (k = 0).
    # The balanced accuracy times 2 * n_positive * n_negative is an integer,
    # so ties are exact and go to the lowest threshold.
    positives_above = n_positive - (
        np.cumsum(positive_counts) - positive_counts
    )
    negatives_below = np.cumsum(negative_counts) - negative_counts
    split = int(
        np.argmax(positives_above * n_negative + negatives_below * n_positive)
    )
    if split == 0:
        return -math.inf
    return float((values[split - 1] + values[split]) / 2)


def _layer_metric(
    val_metric: FloatArray, layers: Sequence[int]
) -> NDArray[np.float64]:
    metric = np.asarray(val_metric, dtype=np.float64)
    if (
        metric.ndim != 1
        or metric.shape[0] == 0
        or metric.shape[0] != len(layers)
    ):
        raise ValueError(
            "val_metric must be a non-empty vector with one entry per layer, "
            f"got shape {metric.shape} for {len(layers)} layers"
        )
    if np.isnan(metric).any():
        raise ValueError("val_metric must not contain NaN")
    if len(set(layers)) != len(layers):
        raise ValueError("layers must be distinct")
    return metric


def _per_layer(
    scores: FloatArray, name: str, n_layers: int
) -> NDArray[np.float64]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != n_layers:
        raise ValueError(
            f"{name} must have shape ({n_layers}, n), got {values.shape}"
        )
    if values.shape[1] == 0:
        raise ValueError(f"{name} needs a stimulus")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def _labelled(
    scores: FloatArray, labels: NDArray[np.int64], split: str
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    values = np.asarray(scores, dtype=np.float64)
    marks = np.asarray(labels)
    if values.ndim != 1 or values.shape != marks.shape:
        raise ValueError(
            f"{split} scores and labels must be vectors of the same length, "
            f"got {values.shape} and {marks.shape}"
        )
    if not np.isin(marks, (0, 1)).all():
        raise ValueError(f"{split} labels must be 0 or 1")
    if not np.isfinite(values).all():
        raise ValueError(f"{split} scores must be finite")
    return values, marks == 1


def _check_both_classes(positive: NDArray[np.bool_], split: str) -> None:
    if positive.all() or not positive.any():
        raise ValueError(f"{split} labels must contain both classes")


def _check_resampling(n_boot: int, alpha: float) -> None:
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    _check_alpha(alpha)


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _pair(a: NDArray[Any], b: NDArray[Any]) -> NDArray[np.float64]:
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if first.ndim != 1 or first.shape != second.shape:
        raise ValueError(
            "the two scorers need vectors of the same length, got "
            f"{first.shape} and {second.shape}"
        )
    if first.shape[0] == 0:
        raise ValueError("the scorers need at least one row")
    return np.stack([first, second])


def _paired_rows(
    a: NDArray[Any], b: NDArray[Any], clusters: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    pair = _pair(a, b)
    ids = np.asarray(clusters)
    if ids.shape != pair.shape[1:]:
        raise ValueError(
            f"clusters must have shape {pair.shape[1:]}, got {ids.shape}"
        )
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"clusters must be integers, got {ids.dtype}")
    return pair, ids


def _summary(
    point: NDArray[np.float64], diffs: list[float], alpha: float
) -> PairedDiff:
    lo, hi = np.percentile(diffs, [50 * alpha, 100 - 50 * alpha])
    return PairedDiff(
        float(point[0] - point[1]), float(lo), float(hi), len(diffs)
    )


def _resampled_rows(
    clusters: NDArray[np.int64], n_boot: int, seed: int
) -> Iterator[NDArray[np.int64]]:
    _, inverse = np.unique(clusters, return_inverse=True)
    sizes = np.bincount(inverse)
    starts = np.cumsum(sizes) - sizes
    order = np.argsort(inverse, kind="stable")
    n_clusters = sizes.shape[0]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        drawn = rng.integers(n_clusters, size=n_clusters)
        lengths = sizes[drawn]
        shift = starts[drawn] - (np.cumsum(lengths) - lengths)
        yield order[np.arange(lengths.sum()) + np.repeat(shift, lengths)]
