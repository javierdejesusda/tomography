"""Decision statistics of the two-position invariance reader study.

Implements, on reader scores, the stage 1 selection rule and gate (sections
6.2 to 6.4) and the stage 2 tests (sections 7.2 to 7.6) of
`docs/preregistration-invariance-reader.md`.

Scores of one reader under one condition form an array of shape
`(n_draws, n_pairs, 2)`: axis 2 holds the label-1 text and the label-0 text
of each pair. A bootstrap resample is a vector of pair multiplicities, so
the AUROC of a resample is the weighted Mann-Whitney statistic
`m^T C m / n_pairs^2`, where `C[i, j]` compares the label-1 text of pair `i`
with the label-0 text of pair `j` (a tie counts 0.5). The statistic is
linear in `C`, so averaging over draws is done once on `C`.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tomography.reading.stats import holm_correction

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

CLEAN = "clean"
ARM_ORDER = ("A1", "A2", "A3", "A4", "A5", "A7", "A8", "A6")
BASELINE_ARMS = ("A1", "A2")
COMPARATOR_ARMS = ("A3", "A4")
N_RESAMPLES = 10_000
GATE_MARGIN = 0.02
CLEAN_MARGIN = 0.01
TIE_MARGIN = 0.005
READER_ARMS = ("A5", "A6", "A7", "A8")
LEAD = "A6"
LEAD_ABLATION = "A5"
STAGE2_BASELINES = tuple(f"B{i}" for i in range(1, 10))
LAST_TOKEN_LR = "B3"
RANDOM_DIRECTION = "random_direction"
ALPHA = 0.05
NON_INFERIORITY_MARGIN = 0.02
FAMILY_SIZES = {"qwen": 16, "llama": 8}
REQUIRED_WINS = {"qwen": 9, "llama": 5}

# Preregistered thresholds are compared with this slack, so that float
# rounding of an exact rational AUROC never decides a boundary case.
_TOL = 1e-12
_CHUNK_ELEMENTS = 4_000_000


def _percentile_interval(values: FloatArray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


@dataclass(frozen=True)
class TaskScores:
    """Scores of several readers on the validation or report pairs of a task.

    Attributes:
        task: Task name; unique within one analysis.
        scores: Reader name to condition name to an array of shape
            `(n_draws, n_pairs, 2)`, axis 2 being (label-1 text, label-0
            text) of the pair. Every reader has the same conditions, one of
            them `clean`, and every array has the same shape.
    """

    task: str
    scores: Mapping[str, Mapping[str, FloatArray]]
    readers: tuple[str, ...] = field(init=False)
    conditions: tuple[str, ...] = field(init=False)
    shape: tuple[int, int, int] = field(init=False)

    def __post_init__(self) -> None:
        """Validates the scores.

        Raises:
            ValueError: If there is no reader, the readers do not share
                their conditions, `clean` is missing, or an array has the
                wrong shape or a value that is not finite.
        """
        if not self.scores:
            raise ValueError(f"task {self.task} has no reader")
        readers = tuple(self.scores)
        conditions = tuple(self.scores[readers[0]])
        if any(set(self.scores[r]) != set(conditions) for r in readers):
            raise ValueError(f"readers of {self.task} differ in conditions")
        if CLEAN not in conditions:
            raise ValueError(f"task {self.task} lacks the clean condition")
        shapes = set()
        for reader in readers:
            for condition in conditions:
                values = np.asarray(self.scores[reader][condition])
                if values.ndim != 3 or values.shape[2] != 2:
                    raise ValueError(
                        f"scores must have shape (n_draws, n_pairs, 2), got"
                        f" {values.shape} for {reader} {condition}"
                    )
                if not np.isfinite(values).all():
                    raise ValueError(f"scores of {reader} are not finite")
                shapes.add(values.shape)
        if len(shapes) != 1:
            raise ValueError(f"scores of {self.task} differ in shape")
        object.__setattr__(self, "readers", readers)
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "shape", shapes.pop())

    @property
    def n_pairs(self) -> int:
        """Number of pairs."""
        return self.shape[1]


def resample_pairs(
    n_pairs: int, n_resamples: int, rng: np.random.Generator
) -> IntArray:
    """Draws bootstrap resamples of pairs as multiplicities.

    Args:
        n_pairs: Number of pairs; each resample draws this many with
            replacement.
        n_resamples: Number of resamples.
        rng: Source of randomness.

    Returns:
        Counts of shape `(n_resamples, n_pairs)`, each row summing to
        `n_pairs`.
    """
    probabilities = np.full(n_pairs, 1.0 / n_pairs)
    return rng.multinomial(n_pairs, probabilities, size=n_resamples)


def _comparisons(scores: FloatArray) -> FloatArray:
    """Returns the draw-averaged comparison matrix of shape (P, P)."""
    positive = scores[:, :, 0][:, :, None]
    negative = scores[:, :, 1][:, None, :]
    wins = (positive > negative) + 0.5 * (positive == negative)
    return np.asarray(wins.mean(axis=0), dtype=np.float64)


class _Kernel:
    """Resampled AUROCs and clean pair accuracies of chosen readers."""

    def __init__(self, task: TaskScores, readers: Sequence[str]) -> None:
        self.readers = tuple(readers)
        self.conditions = task.conditions
        n_pairs = task.n_pairs
        matrices = np.stack(
            [
                _comparisons(np.asarray(task.scores[r][c], dtype=np.float64))
                for r in self.readers
                for c in self.conditions
            ]
        )
        self._n_pairs = n_pairs
        self._cells = matrices.shape[0]
        self._flat = np.ascontiguousarray(
            matrices.transpose(1, 0, 2).reshape(n_pairs, -1)
        )
        clean = self.conditions.index(CLEAN)
        self._diagonals = np.stack(
            [
                np.diagonal(matrices[i * len(self.conditions) + clean])
                for i in range(len(self.readers))
            ],
            axis=1,
        )

    def aurocs(self, counts: IntArray) -> FloatArray:
        """Returns AUROCs of shape (n_resamples, readers, conditions)."""
        weights = np.asarray(counts, dtype=np.float64)
        batch = max(1, _CHUNK_ELEMENTS // (self._cells * self._n_pairs))
        out = np.empty((weights.shape[0], self._cells))
        for start in range(0, weights.shape[0], batch):
            chunk = weights[start : start + batch]
            left = (chunk @ self._flat).reshape(chunk.shape[0], self._cells, -1)
            out[start : start + batch] = np.einsum("bkp,bp->bk", left, chunk)
        out /= self._n_pairs**2
        return out.reshape(weights.shape[0], len(self.readers), -1)

    def clean_accuracy(self, counts: IntArray) -> FloatArray:
        """Returns clean pair accuracies of shape (n_resamples, readers)."""
        weights = np.asarray(counts, dtype=np.float64)
        return np.asarray(weights @ self._diagonals / self._n_pairs)


def _single(scores: FloatArray) -> _Kernel:
    return _Kernel(TaskScores("", {"r": {CLEAN: scores}}), ["r"])


def resampled_auroc(scores: FloatArray, counts: IntArray) -> FloatArray:
    """Computes the draw-averaged AUROC of each resample of pairs.

    Args:
        scores: Array of shape `(n_draws, n_pairs, 2)`.
        counts: Pair multiplicities of shape `(n_resamples, n_pairs)`.

    Returns:
        For each resample, the AUROC over the `2 n_pairs` resampled texts
        (ties count 0.5) of each draw, averaged over draws.
    """
    return _single(scores).aurocs(counts)[:, 0, 0]


def resampled_pair_accuracy(scores: FloatArray, counts: IntArray) -> FloatArray:
    """Computes the draw-averaged pair accuracy of each resample of pairs.

    Args:
        scores: Array of shape `(n_draws, n_pairs, 2)`.
        counts: Pair multiplicities of shape `(n_resamples, n_pairs)`.

    Returns:
        For each resample, the share of resampled pairs whose label-1 text
        scores higher (a tie counts 0.5), averaged over draws.
    """
    return _single(scores).clean_accuracy(counts)[:, 0]


def _point(task: TaskScores, readers: Sequence[str]) -> FloatArray:
    ones = np.ones((1, task.n_pairs), dtype=np.int64)
    return np.asarray(_Kernel(task, readers).aurocs(ones)[0])


@dataclass(frozen=True)
class Stage1Selection:
    """Endpoints and the selection rule of sections 6.2 and 6.3.

    Attributes:
        auroc: Task to arm to condition to draw-averaged AUROC.
        endpoint: Task to arm to worst-condition AUROC.
        best_baseline: Task to the better of A1 and A2 (a tie goes to A1).
        gain: Task to arm to its endpoint minus that of the best baseline.
        mean_gain: Arm to its mean gain over tasks.
        eligible: Arms whose clean AUROC is at least that of A1 minus 0.01
            in every task, in the order of section 6.3.
        winner: The selected arm.
    """

    auroc: Mapping[str, Mapping[str, Mapping[str, float]]]
    endpoint: Mapping[str, Mapping[str, float]]
    best_baseline: Mapping[str, str]
    gain: Mapping[str, Mapping[str, float]]
    mean_gain: Mapping[str, float]
    eligible: tuple[str, ...]
    winner: str


def _stage1_arms(tasks: Sequence[TaskScores]) -> tuple[str, ...]:
    if not tasks:
        raise ValueError("stage 1 needs at least one task")
    if len({task.task for task in tasks}) != len(tasks):
        raise ValueError("task names must be unique")
    arms = set(tasks[0].readers)
    unknown = arms - set(ARM_ORDER)
    if unknown:
        raise ValueError(f"unknown stage 1 arm(s): {sorted(unknown)}")
    missing = set(BASELINE_ARMS + COMPARATOR_ARMS) - arms
    if missing:
        raise ValueError(f"stage 1 needs the arms {sorted(missing)}")
    if any(set(task.readers) != arms for task in tasks):
        raise ValueError("every task needs the same arms")
    return tuple(arm for arm in ARM_ORDER if arm in arms)


def stage1_selection(tasks: Sequence[TaskScores]) -> Stage1Selection:
    """Applies the stage 1 endpoint and selection rule (sections 6.2, 6.3).

    Args:
        tasks: Validation scores of the stage 1 arms, one entry per task.

    Returns:
        The endpoints, gains, eligible arms and the winner.

    Raises:
        ValueError: If a task name repeats, an arm is unknown, A1 to A4 are
            missing, or the tasks do not share their arms.
    """
    arms = _stage1_arms(tasks)
    auroc: dict[str, dict[str, dict[str, float]]] = {}
    endpoint: dict[str, dict[str, float]] = {}
    best: dict[str, str] = {}
    gain: dict[str, dict[str, float]] = {}
    for task in tasks:
        values = _point(task, arms)
        auroc[task.task] = {
            arm: dict(zip(task.conditions, map(float, values[i]), strict=True))
            for i, arm in enumerate(arms)
        }
        worst = {arm: float(values[i].min()) for i, arm in enumerate(arms)}
        endpoint[task.task] = worst
        best[task.task] = max(BASELINE_ARMS, key=lambda arm: worst[arm])
        reference = worst[best[task.task]]
        gain[task.task] = {arm: worst[arm] - reference for arm in arms}
    mean_gain = {
        arm: float(np.mean([gain[t.task][arm] for t in tasks])) for arm in arms
    }
    eligible = tuple(
        arm
        for arm in arms
        if all(
            auroc[t.task][arm][CLEAN]
            >= auroc[t.task]["A1"][CLEAN] - CLEAN_MARGIN - _TOL
            for t in tasks
        )
    )
    top = max(mean_gain[arm] for arm in eligible)
    winner = next(
        arm for arm in eligible if mean_gain[arm] >= top - TIE_MARGIN - _TOL
    )
    return Stage1Selection(
        auroc, endpoint, best, gain, mean_gain, eligible, winner
    )


@dataclass(frozen=True)
class Stage1Gate:
    """The stage 1 gate of section 6.4.

    Attributes:
        selection: The selection of section 6.3.
        comparator: The better of A3 and A4 by mean endpoint over tasks on
            the full validation set (a tie goes to A3).
        statistic: Mean over tasks of the winner's endpoint minus the
            comparator's.
        interval: Two-sided 95% percentile interval of the statistic.
        bootstrap: The statistic in every resample, with the minima and the
            comparator recomputed inside each one.
        passed: Whether the gate is passed.
        reasons: Why the gate failed, empty when it passed.
    """

    selection: Stage1Selection
    comparator: str
    statistic: float
    interval: tuple[float, float]
    bootstrap: FloatArray
    passed: bool
    reasons: tuple[str, ...]


def _comparator(worst: Mapping[str, Any]) -> Any:
    """Returns True where A3 is the comparator, given mean endpoints."""
    return worst["A3"] >= worst["A4"] - _TOL


def stage1_gate(
    tasks: Sequence[TaskScores],
    *,
    rng: np.random.Generator | int,
    n_resamples: int = N_RESAMPLES,
) -> Stage1Gate:
    """Runs the stage 1 selection and its paired bootstrap gate.

    Pairs are resampled within each task, one call of `resample_pairs` per
    task in sorted task-name order; the same resampled pairs serve every
    arm, condition and draw of that task.

    Args:
        tasks: Validation scores of the stage 1 arms, one entry per task.
        rng: Seed or generator of the bootstrap.
        n_resamples: Number of bootstrap resamples.

    Returns:
        The selection, the statistic, its interval and the decision.
    """
    selection = stage1_selection(tasks)
    winner = selection.winner
    readers = (winner, *COMPARATOR_ARMS)
    generator = np.random.default_rng(rng)
    worst = np.zeros((n_resamples, len(readers)))
    for task in sorted(tasks, key=lambda t: t.task):
        counts = resample_pairs(task.n_pairs, n_resamples, generator)
        worst += _Kernel(task, readers).aurocs(counts).min(axis=2)
    worst /= len(tasks)
    use_a3 = _comparator({"A3": worst[:, 1], "A4": worst[:, 2]})
    bootstrap = worst[:, 0] - np.where(use_a3, worst[:, 1], worst[:, 2])
    mean_endpoint = {
        arm: float(np.mean([selection.endpoint[t.task][arm] for t in tasks]))
        for arm in readers
    }
    comparator = "A3" if _comparator(mean_endpoint) else "A4"
    statistic = mean_endpoint[winner] - mean_endpoint[comparator]
    interval = _percentile_interval(bootstrap)
    reasons = []
    if winner in BASELINE_ARMS + COMPARATOR_ARMS:
        reasons.append(
            f"the winner {winner} is a single-position or plain logistic"
            " regression"
        )
    if statistic < GATE_MARGIN - _TOL:
        reasons.append("the statistic is below 0.02")
    if interval[0] <= _TOL:
        reasons.append("the interval is not above 0")
    return Stage1Gate(
        selection,
        comparator,
        statistic,
        interval,
        bootstrap,
        not reasons,
        tuple(reasons),
    )


@dataclass(frozen=True)
class Combination:
    """Report scores of one checkpoint on one task (section 7.1).

    Attributes:
        family: Model family, a key of `FAMILY_SIZES`.
        checkpoint: Checkpoint name.
        reader: Name of the frozen stage 1 winner within `scores`.
        scores: Scores of the reader, the baselines B1 to B9 and any
            controls. Every other name is a control: it never enters the
            best baseline (section 5.5). `random_direction` is the random
            direction control and `A5` the two-position augmented control.
    """

    family: str
    checkpoint: str
    reader: str
    scores: TaskScores

    def __post_init__(self) -> None:
        """Validates the readers of the combination.

        Raises:
            ValueError: If a baseline is missing, the reader is not a
                stage 2 candidate or is missing, or the reader is A6
                without the A5 control.
        """
        names = self.scores.readers
        missing = [b for b in STAGE2_BASELINES if b not in names]
        if missing:
            raise ValueError(f"combination lacks the baselines {missing}")
        if self.reader not in READER_ARMS or self.reader not in names:
            raise ValueError(
                f"the reader must be one of {READER_ARMS} and have scores,"
                f" got {self.reader}"
            )
        if self.reader == LEAD and LEAD_ABLATION not in names:
            raise ValueError("an A6 reader needs the A5 control (section 7.6)")

    @property
    def task(self) -> str:
        """Task name."""
        return self.scores.task


@dataclass(frozen=True)
class Estimate:
    """A point estimate with its two-sided 95% percentile interval.

    Attributes:
        estimate: Value on the full set of pairs.
        interval: 2.5th and 97.5th percentiles over resamples.
    """

    estimate: float
    interval: tuple[float, float]


@dataclass(frozen=True)
class CombinationTest:
    """Endpoint and tests of one combination (sections 7.2 to 7.4, 7.6).

    Attributes:
        family: Model family.
        checkpoint: Checkpoint name.
        task: Task name.
        reader: The reader.
        auroc: Reader to condition to draw-averaged AUROC, for every reader,
            baseline and control.
        worst: Reader to worst-condition AUROC `W`.
        clean_accuracy: Reader to draw-averaged clean pair accuracy.
        best_baseline: The baseline among B1 to B9 with the highest `W`
            (a tie goes to the lower number).
        delta: `W` of the reader minus the highest `W` of B1 to B9.
        delta_interval: Two-sided 95% percentile interval of `delta`.
        delta_bootstrap: `delta` in every resample, with the minima and the
            maximum recomputed inside each one.
        p_value: One-sided bootstrap p value for `delta > 0`.
        clean_difference: Clean pair accuracy of the reader minus B3.
        clean_lower: 2.5th percentile of `clean_difference` over resamples.
        non_inferior: Whether `clean_lower` is above -0.02.
        random_contrast: Reader to condition to its AUROC minus that of the
            random direction control, for every reader but that control;
            None without the control.
        secondary: `W(A6) - W(A5)`; None unless the reader is A6.
        secondary_bootstrap: The secondary contrast in every resample.
    """

    family: str
    checkpoint: str
    task: str
    reader: str
    auroc: Mapping[str, Mapping[str, float]]
    worst: Mapping[str, float]
    clean_accuracy: Mapping[str, float]
    best_baseline: str
    delta: float
    delta_interval: tuple[float, float]
    delta_bootstrap: FloatArray
    p_value: float
    clean_difference: float
    clean_lower: float
    non_inferior: bool
    random_contrast: Mapping[str, Mapping[str, Estimate]] | None
    secondary: Estimate | None
    secondary_bootstrap: FloatArray | None


def _combination_test(
    combination: Combination, counts: IntArray
) -> CombinationTest:
    task = combination.scores
    names = task.readers
    index = {name: i for i, name in enumerate(names)}
    kernel = _Kernel(task, names)
    ones = np.ones((1, task.n_pairs), dtype=np.int64)
    auroc = kernel.aurocs(ones)[0]
    accuracy = kernel.clean_accuracy(ones)[0]
    boot_auroc = kernel.aurocs(counts)
    boot_accuracy = kernel.clean_accuracy(counts)
    worst, boot_worst = auroc.min(axis=-1), boot_auroc.min(axis=-1)
    reader = index[combination.reader]
    baselines = [index[b] for b in STAGE2_BASELINES]
    last = index[LAST_TOKEN_LR]
    delta = float(worst[reader] - worst[baselines].max())
    boot_delta = boot_worst[:, reader] - boot_worst[:, baselines].max(axis=1)
    p_value = (1 + np.count_nonzero(boot_delta <= _TOL)) / (1 + len(counts))
    boot_clean = boot_accuracy[:, reader] - boot_accuracy[:, last]
    clean_lower = float(np.percentile(boot_clean, 2.5))
    random_contrast = None
    if RANDOM_DIRECTION in index:
        control = index[RANDOM_DIRECTION]
        bounds = np.percentile(
            boot_auroc - boot_auroc[:, control : control + 1],
            [2.5, 97.5],
            axis=0,
        )
        random_contrast = {
            name: {
                condition: Estimate(
                    float(auroc[i, c] - auroc[control, c]),
                    (float(bounds[0, i, c]), float(bounds[1, i, c])),
                )
                for c, condition in enumerate(task.conditions)
            }
            for i, name in enumerate(names)
            if i != control
        }
    secondary = secondary_bootstrap = None
    if combination.reader == LEAD:
        ablation = index[LEAD_ABLATION]
        secondary_bootstrap = boot_worst[:, reader] - boot_worst[:, ablation]
        secondary = Estimate(
            float(worst[reader] - worst[ablation]),
            _percentile_interval(secondary_bootstrap),
        )
    return CombinationTest(
        family=combination.family,
        checkpoint=combination.checkpoint,
        task=task.task,
        reader=combination.reader,
        auroc={
            name: dict(zip(task.conditions, map(float, auroc[i]), strict=True))
            for i, name in enumerate(names)
        },
        worst=dict(zip(names, map(float, worst), strict=True)),
        clean_accuracy=dict(zip(names, map(float, accuracy), strict=True)),
        best_baseline=STAGE2_BASELINES[int(np.argmax(worst[baselines]))],
        delta=delta,
        delta_interval=_percentile_interval(boot_delta),
        delta_bootstrap=boot_delta,
        p_value=float(p_value),
        clean_difference=float(accuracy[reader] - accuracy[last]),
        clean_lower=clean_lower,
        non_inferior=clean_lower > -NON_INFERIORITY_MARGIN + _TOL,
        random_contrast=random_contrast,
        secondary=secondary,
        secondary_bootstrap=secondary_bootstrap,
    )


def bootstrap_combination(
    combination: Combination,
    *,
    rng: np.random.Generator | int,
    n_resamples: int = N_RESAMPLES,
) -> CombinationTest:
    """Runs the paired bootstrap of one combination (sections 7.2 to 7.4).

    Both texts of a pair, all conditions, readers and draws follow the
    pair; the resample is one call of `resample_pairs`.

    Args:
        combination: Report scores of the combination.
        rng: Seed or generator of the bootstrap.
        n_resamples: Number of bootstrap resamples.

    Returns:
        The endpoint, its p value and the clean non-inferiority check.
    """
    counts = resample_pairs(
        combination.scores.n_pairs, n_resamples, np.random.default_rng(rng)
    )
    return _combination_test(combination, counts)


def holm_adjust(p_values: FloatArray) -> FloatArray:
    """Adjusts p values of one family with the Holm step-down procedure.

    Args:
        p_values: Unadjusted p values of the family.

    Returns:
        Adjusted p values in the input order; a hypothesis is rejected at
        level alpha when its adjusted p value is below alpha.
    """
    return holm_correction(p_values).adjusted


@dataclass(frozen=True)
class PooledEffect:
    """A mean over combinations with its bootstrap (section 7.5).

    Attributes:
        estimate: Mean over the combinations on the full set of pairs.
        interval: Two-sided 95% percentile interval.
        bootstrap: The mean in every resample.
        above_zero: Whether the lower bound of the interval is above 0.
    """

    estimate: float
    interval: tuple[float, float]
    bootstrap: FloatArray
    above_zero: bool


def _pooled(
    estimates: Sequence[float], boots: Sequence[FloatArray]
) -> PooledEffect:
    bootstrap = np.mean(np.stack(boots), axis=0)
    interval = _percentile_interval(bootstrap)
    return PooledEffect(
        float(np.mean(estimates)), interval, bootstrap, interval[0] > _TOL
    )


@dataclass(frozen=True)
class Stage2Result:
    """The stage 2 decision (sections 7.3 to 7.6).

    Attributes:
        combinations: The test of every combination, in input order.
        holm_p_value: Holm-adjusted p value of each combination, within
            its family.
        win: Whether each combination is a win: adjusted p value below 0.05
            and clean non-inferiority.
        wins: Family to its number of wins.
        pooled: Mean `delta` over all combinations.
        pooled_by_family: Family to its mean `delta`.
        pooled_secondary: Mean A6 minus A5 contrast; None unless the
            reader is A6.
        h2: Whether H2 holds: at least 9 of 16 Qwen wins, at least 5 of 8
            Llama wins and a pooled effect above 0.
    """

    combinations: tuple[CombinationTest, ...]
    holm_p_value: tuple[float, ...]
    win: tuple[bool, ...]
    wins: Mapping[str, int]
    pooled: PooledEffect
    pooled_by_family: Mapping[str, PooledEffect]
    pooled_secondary: PooledEffect | None
    h2: bool


def _check_stage2(combinations: Sequence[Combination]) -> dict[str, int]:
    keys = [(c.family, c.checkpoint, c.task) for c in combinations]
    if len(set(keys)) != len(keys):
        raise ValueError("each checkpoint and task must appear once")
    if len({c.reader for c in combinations}) > 1:
        raise ValueError("every combination needs the same frozen reader")
    pairs: dict[str, int] = {}
    for c in combinations:
        if pairs.setdefault(c.task, c.scores.n_pairs) != c.scores.n_pairs:
            raise ValueError(f"task {c.task} has different numbers of pairs")
    sizes = {f: sum(c.family == f for c in combinations) for f in FAMILY_SIZES}
    if sizes != FAMILY_SIZES or len(keys) != sum(FAMILY_SIZES.values()):
        raise ValueError(f"each family needs exactly {FAMILY_SIZES}")
    return pairs


def stage2(
    combinations: Sequence[Combination],
    *,
    rng: np.random.Generator | int,
    n_resamples: int = N_RESAMPLES,
) -> Stage2Result:
    """Applies the stage 2 tests and success rule (sections 7.2 to 7.6).

    Report pairs are resampled within each task, one call of
    `resample_pairs` per task in sorted task-name order, and the same
    resampled pairs serve every checkpoint of that task, in both families.
    These resamples give the per-combination tests and the pooled effects.

    Args:
        combinations: The 16 Qwen and 8 Llama combinations.
        rng: Seed or generator of the bootstrap.
        n_resamples: Number of bootstrap resamples.

    Returns:
        Every combination's test, the Holm-corrected wins, the pooled
        effects and whether H2 holds.

    Raises:
        ValueError: If a combination repeats, the readers differ, a task
            has different numbers of pairs, or the families do not have
            16 and 8 combinations.
    """
    pairs = _check_stage2(combinations)
    generator = np.random.default_rng(rng)
    counts = {
        task: resample_pairs(pairs[task], n_resamples, generator)
        for task in sorted(pairs)
    }
    tests = tuple(_combination_test(c, counts[c.task]) for c in combinations)
    holm = np.empty(len(tests))
    pooled_by_family = {}
    for family in FAMILY_SIZES:
        members = [i for i, t in enumerate(tests) if t.family == family]
        holm[members] = holm_adjust(
            np.array([tests[i].p_value for i in members])
        )
        pooled_by_family[family] = _pooled(
            [tests[i].delta for i in members],
            [tests[i].delta_bootstrap for i in members],
        )
    win = tuple(
        bool(p < ALPHA - _TOL and t.non_inferior)
        for p, t in zip(holm, tests, strict=True)
    )
    wins = {
        family: sum(
            w for w, t in zip(win, tests, strict=True) if t.family == family
        )
        for family in FAMILY_SIZES
    }
    pooled = _pooled(
        [t.delta for t in tests], [t.delta_bootstrap for t in tests]
    )
    pooled_secondary = None
    if tests[0].secondary is not None:
        pooled_secondary = _pooled(
            [t.secondary.estimate for t in tests if t.secondary is not None],
            [
                t.secondary_bootstrap
                for t in tests
                if t.secondary_bootstrap is not None
            ],
        )
    h2 = pooled.above_zero and all(
        wins[family] >= REQUIRED_WINS[family] for family in FAMILY_SIZES
    )
    return Stage2Result(
        tests,
        tuple(map(float, holm)),
        win,
        wins,
        pooled,
        pooled_by_family,
        pooled_secondary,
        h2,
    )
