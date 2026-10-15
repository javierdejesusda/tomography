"""Interval and selective-prediction tools of the P9 study.

The reliability experiment asks how well a score predicts the model's own
errors. An AUROC answers it with one number, and a risk-coverage curve shows
what the score buys when the model may abstain. Both come with bootstrap
intervals over items, and the AUROC difference between two scores is resampled
on the same items so that the pairing is kept.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tomography.reading.metrics import auroc

NO_RESAMPLE = "no bootstrap resample holds both classes"


def auroc_replicates(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    n_boot: int,
    seed: int,
) -> NDArray[np.float64]:
    """AUROC of item bootstrap resamples.

    A resample that holds a single class has no AUROC and is skipped.

    Args:
        scores: Scores, shape (n,).
        labels: 0/1 labels, shape (n,).
        n_boot: Number of resamples drawn.
        seed: Seed of the resampling.

    Returns:
        The AUROC of every usable resample, shape (m,) with m <= n_boot.

    Raises:
        ValueError: If no resample holds both classes.
    """
    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    out = []
    for _ in range(n_boot):
        idx = rng.integers(n, size=n)
        if labels[idx].min() == labels[idx].max():
            continue
        out.append(float(auroc(scores[idx], labels[idx])))
    if not out:
        raise ValueError(NO_RESAMPLE)
    return np.array(out)


def auroc_ci(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """AUROC with a percentile bootstrap interval over items.

    Args:
        scores: Scores, shape (n,).
        labels: 0/1 labels, shape (n,).
        n_boot: Number of resamples.
        alpha: Total tail mass of the interval.
        seed: Seed of the resampling.

    Returns:
        The AUROC, the lower bound and the upper bound.

    Raises:
        ValueError: If no resample holds both classes.
    """
    reps = auroc_replicates(scores, labels, n_boot=n_boot, seed=seed)
    lo, hi = np.percentile(reps, [50 * alpha, 100 - 50 * alpha])
    return float(auroc(scores, labels)), float(lo), float(hi)


def auroc_diff_ci(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """AUROC of one score minus another, with a paired bootstrap interval.

    Both scores are evaluated on the same resampled items, so the interval
    holds the correlation between the two scores.

    Args:
        a: First scores, shape (n,).
        b: Second scores, shape (n,).
        labels: 0/1 labels, shape (n,).
        n_boot: Number of resamples.
        alpha: Total tail mass of the interval.
        seed: Seed of the resampling.

    Returns:
        The difference, the lower bound and the upper bound.

    Raises:
        ValueError: If no resample holds both classes.
    """
    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(n, size=n)
        if labels[idx].min() == labels[idx].max():
            continue
        diffs.append(
            float(auroc(a[idx], labels[idx]) - auroc(b[idx], labels[idx]))
        )
    if not diffs:
        raise ValueError(NO_RESAMPLE)
    lo, hi = np.percentile(diffs, [50 * alpha, 100 - 50 * alpha])
    return float(auroc(a, labels) - auroc(b, labels)), float(lo), float(hi)


def risk_coverage(
    confidence: NDArray[np.float64],
    correct: NDArray[np.int64],
    *,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The risk of answering the k most confident items, for every k.

    Items are shuffled before the stable sort, so ties are broken at random
    and never in the order the data happens to come in.

    Args:
        confidence: Confidence, higher meaning more likely correct, (n,).
        correct: 1 where the answer is correct, shape (n,).
        seed: Seed of the tie-breaking shuffle.

    Returns:
        The coverage k / n and the error rate among the k answered items,
        both of shape (n,).
    """
    n = confidence.shape[0]
    shuffle = np.random.default_rng(seed).permutation(n)
    order = shuffle[np.argsort(-confidence[shuffle], kind="stable")]
    kept = np.cumsum(correct[order])
    k = np.arange(1, n + 1)
    return k / n, 1.0 - kept / k


def aurc(
    confidence: NDArray[np.float64],
    correct: NDArray[np.int64],
    *,
    seed: int = 0,
) -> float:
    """Area under the risk-coverage curve, lower being better.

    Args:
        confidence: Confidence, higher meaning more likely correct, (n,).
        correct: 1 where the answer is correct, shape (n,).
        seed: Seed of the tie-breaking shuffle.

    Returns:
        The mean risk over all coverages.
    """
    return float(risk_coverage(confidence, correct, seed=seed)[1].mean())


def risk_at(
    coverage: NDArray[np.float64],
    risk: NDArray[np.float64],
    grid: Sequence[int],
) -> NDArray[np.float64]:
    """The risk at coverages given in percent.

    Args:
        coverage: Coverage from `risk_coverage`, shape (n,).
        risk: Risk from `risk_coverage`, shape (n,).
        grid: Coverages in percent.

    Returns:
        The risk once the first ceil(g / 100 * n) items are answered, one
        value per grid point.
    """
    n = coverage.shape[0]
    return np.array([risk[max(1, -(-g * n // 100)) - 1] for g in grid])


def risk_band(
    confidence: NDArray[np.float64],
    correct: NDArray[np.int64],
    grid: Sequence[int],
    *,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The risk-coverage curve on a grid with a bootstrap band over items.

    Args:
        confidence: Confidence, higher meaning more likely correct, (n,).
        correct: 1 where the answer is correct, shape (n,).
        grid: Coverages in percent.
        n_boot: Number of resamples.
        alpha: Total tail mass of the band.
        seed: Seed of the resampling and of the tie-breaking shuffles.

    Returns:
        The risk, the lower band and the upper band, each of shape
        (len(grid),).
    """
    point = risk_at(*risk_coverage(confidence, correct, seed=seed), grid)
    rng = np.random.default_rng(seed)
    n = confidence.shape[0]
    reps = []
    for b in range(n_boot):
        idx = rng.integers(n, size=n)
        coverage, risk = risk_coverage(
            confidence[idx], correct[idx], seed=seed + b + 1
        )
        reps.append(risk_at(coverage, risk, grid))
    lo, hi = np.percentile(
        np.stack(reps), [50 * alpha, 100 - 50 * alpha], axis=0
    )
    return point, lo, hi


def stack_scores(
    train_x: NDArray[np.float64],
    train_y: NDArray[np.int64],
    test_x: NDArray[np.float64],
    *,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Combine several scores into one with a logistic regression.

    The scaler and the regression are fitted on the train items only.

    Args:
        train_x: Train features, shape (n_train, f).
        train_y: Train labels, shape (n_train,).
        test_x: Test features, shape (n_test, f).
        seed: Seed of the regression.

    Returns:
        The decision function on the test items, shape (n_test,).
    """
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(max_iter=1000, random_state=seed).fit(
        scaler.transform(train_x), train_y
    )
    return np.asarray(
        model.decision_function(scaler.transform(test_x)), dtype=np.float64
    )


def group_rate(
    train_keys: Sequence[str],
    train_correct: NDArray[np.int64],
    test_keys: Sequence[str],
) -> NDArray[np.float64]:
    """The smoothed accuracy of each key, as a score.

    The rate of a key is (correct + overall) / (n + 1), which shrinks a key
    seen a few times towards the overall train accuracy.

    Args:
        train_keys: Key of every train item.
        train_correct: 1 where the train answer is correct.
        test_keys: Key of every item to score.

    Returns:
        The rate of each test key; a key unseen in train gets the overall
        train accuracy.
    """
    overall = float(train_correct.mean())
    totals: dict[str, list[float]] = {}
    for key, value in zip(train_keys, train_correct, strict=True):
        cell = totals.setdefault(key, [0.0, 0.0])
        cell[0] += float(value)
        cell[1] += 1.0
    return np.array(
        [
            (totals[k][0] + overall) / (totals[k][1] + 1.0)
            if k in totals
            else overall
            for k in test_keys
        ]
    )


def orient_on_train(
    train_scores: NDArray[np.float64],
    train_labels: NDArray[np.int64],
    test_scores: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Give a score the sign that ranks correct answers higher on train.

    Args:
        train_scores: Train scores, shape (n_train,).
        train_labels: 1 where the train answer is correct.
        test_scores: Scores to orient, shape (n_test,).

    Returns:
        The test scores, negated iff the train AUROC is below 0.5.
    """
    if float(auroc(train_scores, train_labels)) >= 0.5:
        return test_scores
    return -test_scores
