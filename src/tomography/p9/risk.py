"""Interval and selective-prediction tools of the P9 study.

The reliability experiment asks how well a score predicts the model's own
errors. An AUROC answers it with one number, and a risk-coverage curve shows
what the score buys when the model may abstain. Both come with bootstrap
intervals over items, and the AUROC difference between two scores is resampled
on the same items so that the pairing is kept.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tomography.reading.metrics import auroc

NO_RESAMPLE = "no bootstrap resample holds both classes"
NO_PAIRED_GROUP = "no group holds both classes"


@dataclass(frozen=True)
class GroupAuroc:
    """The AUROC of one group of items.

    Attributes:
        group: The group name.
        n_pos: Items of the group with label 1.
        n_neg: Items of the group with label 0.
        auroc: The AUROC of the group, or None when either class has fewer
            items than the minimum asked for.
    """

    group: str
    n_pos: int
    n_neg: int
    auroc: float | None


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


def grouped_auroc(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    groups: NDArray[np.str_] | Sequence[str],
) -> float:
    """AUROC within groups, the groups weighted by their pairs.

    The value is the probability that a positive item scores above a negative
    item of the SAME group: the sum over groups of n_pos * n_neg * AUROC_g,
    divided by the sum of n_pos * n_neg. A group is scored with `auroc`, so
    ties share the credit. A group that holds a single class has no pair and
    contributes nothing.

    Args:
        scores: Scores, shape (n,).
        labels: 0/1 labels, shape (n,).
        groups: The group of every item, shape (n,).

    Returns:
        The pair-weighted AUROC.

    Raises:
        ValueError: If the lengths differ, a score is not finite, a label is
            neither 0 nor 1, or no group holds both classes.
    """
    return _weighted_auroc(_paired_cells(scores, labels, groups))


def grouped_auroc_ci(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    groups: NDArray[np.str_] | Sequence[str],
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Grouped AUROC with a percentile bootstrap interval over items.

    Groups that hold a single class are dropped first, as they never
    contribute. A resample draws items with replacement inside each group at
    the size of the group, so a draw never mixes groups, group sizes stay
    fixed and the class counts of a group, hence its pairs and its weight in
    `grouped_auroc`, vary as they would under a new test split. A group that
    holds a single class in a draw contributes nothing to it, and a draw in
    which no group holds both classes is skipped. With one group the interval
    equals `auroc_ci` for the same seed, `n_boot` and `alpha`.

    Args:
        scores: Scores, shape (n,).
        labels: 0/1 labels, shape (n,).
        groups: The group of every item, shape (n,).
        n_boot: Number of resamples, at least 1.
        seed: Seed of the resampling.
        alpha: Total tail mass of the interval, inside (0, 1).

    Returns:
        The grouped AUROC, the lower bound and the upper bound.

    Raises:
        ValueError: If `n_boot` or `alpha` is out of range, the lengths
            differ, a score is not finite, a label is neither 0 nor 1, no
            group holds both classes, or no resample holds both classes in
            any group.
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    items = _paired_items(scores, labels, groups)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        draws = [
            _split(values[idx], marks[idx])
            for values, marks in items
            for idx in (rng.integers(values.size, size=values.size),)
        ]
        usable = [cell for cell in draws if cell[0].size and cell[1].size]
        if usable:
            reps.append(_weighted_auroc(usable))
    if not reps:
        raise ValueError(NO_RESAMPLE)
    lo, hi = np.percentile(reps, [50 * alpha, 100 - 50 * alpha])
    return (
        _weighted_auroc([_split(v, m) for v, m in items]),
        float(lo),
        float(hi),
    )


def per_group_auroc(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    groups: NDArray[np.str_] | Sequence[str],
    *,
    min_class: int,
) -> list[GroupAuroc]:
    """The AUROC of every group, null where a class is too thin.

    Args:
        scores: Scores, shape (n,).
        labels: 0/1 labels, shape (n,).
        groups: The group of every item, shape (n,).
        min_class: The fewest items either class may hold for the AUROC to be
            reported, at least 1.

    Returns:
        One row per group in sorted group order. A group with a class below
        `min_class`, a single-class group among them, has an AUROC of None.

    Raises:
        ValueError: If `min_class` is below 1, the lengths differ, a score is
            not finite, or a label is neither 0 nor 1.
    """
    if min_class < 1:
        raise ValueError(f"min_class must be at least 1, got {min_class}")
    return [
        GroupAuroc(
            group=name,
            n_pos=positive.size,
            n_neg=negative.size,
            auroc=_cell_auroc(positive, negative)
            if min(positive.size, negative.size) >= min_class
            else None,
        )
        for name, positive, negative in _cells(scores, labels, groups)
    ]


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


def text_scores(
    train_texts: Sequence[str],
    train_labels: NDArray[np.int64],
    test_texts: Sequence[str],
    *,
    analyzer: Literal["word", "char"],
) -> NDArray[np.float64]:
    """Scores of a text-only classifier, the surface-form baseline of a reader.

    The features are TF-IDF weights fitted on the train texts only: word
    1-grams and 2-grams seen in at least 2 texts for "word", character 2-grams
    to 5-grams inside words seen in at least 3 texts for "char". A logistic
    regression (C=10, class-balanced) is fitted on the train labels and
    scores the test texts. The fit is deterministic.

    Args:
        train_texts: The texts the classifier is fitted on.
        train_labels: 0/1 labels of the train texts.
        test_texts: The texts that are scored.
        analyzer: "word" or "char" n-grams.

    Returns:
        The decision score of every test text, shape (n_test,), higher
        meaning more likely label 1.

    Raises:
        ValueError: If `analyzer` is neither "word" nor "char", or the train
            texts leave no feature.
    """
    if analyzer == "word":
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, lowercase=True
        )
    elif analyzer == "char":
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=3, lowercase=True
        )
    else:
        raise ValueError(f"analyzer must be 'word' or 'char', got {analyzer!r}")
    train_features = vectorizer.fit_transform(train_texts)
    classifier = LogisticRegression(
        C=10, max_iter=2000, class_weight="balanced"
    )
    classifier.fit(train_features, train_labels)
    scores = classifier.decision_function(vectorizer.transform(test_texts))
    return np.asarray(scores, dtype=np.float64)


def text_auroc(
    train_texts: Sequence[str],
    train_labels: NDArray[np.int64],
    test_texts: Sequence[str],
    test_labels: NDArray[np.int64],
    *,
    analyzer: Literal["word", "char"],
) -> float:
    """AUROC of a text-only classifier, the surface-form baseline of a reader.

    The classifier is the one of `text_scores`.

    Args:
        train_texts: The texts the classifier is fitted on.
        train_labels: 0/1 labels of the train texts.
        test_texts: The texts that are scored.
        test_labels: 0/1 labels of the test texts.
        analyzer: "word" or "char" n-grams.

    Returns:
        The AUROC of the decision scores on the test texts, label 1 positive.

    Raises:
        ValueError: If `analyzer` is neither "word" nor "char", the test
            labels hold one class, or the train texts leave no feature.
    """
    scores = text_scores(
        train_texts, train_labels, test_texts, analyzer=analyzer
    )
    return float(auroc(scores, test_labels))


def popularity_auroc(
    popularity: NDArray[np.float64], labels: NDArray[np.int64]
) -> float:
    """AUROC of the popularity of an item, the baseline of a known contrast.

    The score is `log10(popularity + 1)` and is not oriented: a value below
    0.5 means the positive side is the less popular one.

    Args:
        popularity: Non-negative popularity of each item, shape (n,).
        labels: 0/1 labels, shape (n,).

    Returns:
        The AUROC of the score, label 1 positive.

    Raises:
        ValueError: If the labels hold one class.
    """
    return float(auroc(np.log10(popularity + 1.0), labels))


def _checked_items(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    groups: NDArray[np.str_] | Sequence[str],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.str_]]:
    values = np.asarray(scores, dtype=np.float64)
    marks = np.asarray(labels)
    names = np.asarray(groups)
    if values.ndim != 1 or {marks.shape, names.shape} != {values.shape}:
        raise ValueError(
            f"scores {values.shape}, labels {marks.shape} and groups "
            f"{names.shape} must be vectors of the same length"
        )
    infinite = ~np.isfinite(values)
    if infinite.any():
        raise ValueError(f"scores must be finite, got {values[infinite][0]}")
    other = ~np.isin(marks, (0, 1))
    if other.any():
        raise ValueError(f"labels must be 0 or 1, got {marks[other][0]}")
    return values, marks.astype(np.int64), names


def _group_items(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    groups: NDArray[np.str_] | Sequence[str],
) -> list[tuple[str, NDArray[np.float64], NDArray[np.int64]]]:
    values, marks, names = _checked_items(scores, labels, groups)
    return [
        (str(name), values[names == name], marks[names == name])
        for name in np.unique(names)
    ]


def _split(
    values: NDArray[np.float64], marks: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return values[marks == 1], values[marks == 0]


def _cells(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    groups: NDArray[np.str_] | Sequence[str],
) -> list[tuple[str, NDArray[np.float64], NDArray[np.float64]]]:
    return [
        (name, *_split(values, marks))
        for name, values, marks in _group_items(scores, labels, groups)
    ]


def _paired_items(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    groups: NDArray[np.str_] | Sequence[str],
) -> list[tuple[NDArray[np.float64], NDArray[np.int64]]]:
    paired = [
        (values, marks)
        for _, values, marks in _group_items(scores, labels, groups)
        if 0 < marks.sum() < marks.size
    ]
    if not paired:
        raise ValueError(NO_PAIRED_GROUP)
    return paired


def _paired_cells(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    groups: NDArray[np.str_] | Sequence[str],
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    return [
        _split(values, marks)
        for values, marks in _paired_items(scores, labels, groups)
    ]


def _cell_auroc(
    positive: NDArray[np.float64], negative: NDArray[np.float64]
) -> float:
    labels = np.r_[np.ones(positive.size), np.zeros(negative.size)]
    return float(auroc(np.r_[positive, negative], labels.astype(np.int64)))


def _weighted_auroc(
    cells: Sequence[tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> float:
    weights = [positive.size * negative.size for positive, negative in cells]
    values = [_cell_auroc(positive, negative) for positive, negative in cells]
    total = sum(w * v for w, v in zip(weights, values, strict=True))
    return total / sum(weights)
