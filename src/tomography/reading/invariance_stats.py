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

# Preregistered thresholds are compared with this slack, so that float
# rounding of an exact rational AUROC never decides a boundary case.
_TOL = 1e-12
_CHUNK_ELEMENTS = 4_000_000


def _as_rng(rng: np.random.Generator | int) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


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
    generator = _as_rng(rng)
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
