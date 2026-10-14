"""The shared reading sweep: per-layer, per-seed accuracy curves.

Every reading experiment turns stimulus sets into accuracy curves for the
fitted reader and two baselines through this module, and writes them in the
`curves` results format.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tomography.activations import Activations
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments.base import derive_seed
from tomography.reading.directions import Method
from tomography.reading.directions import Preprocess
from tomography.reading.metrics import group_accuracy
from tomography.reading.metrics import parse_groups
from tomography.reading.reader import RepReader

SERIES: tuple[str, str, str] = ("reader", "random", "shuffled")

Prepared = tuple[StimulusSet, Activations]


@dataclass(frozen=True, eq=False)
class SweepResult:
    """Accuracy of the three series for every evaluation set and layer.

    Attributes:
        layers: Layer index of each entry of the last axis.
        eval_names: Name of each evaluation set, in the order of axis 0.
        accuracy: Array of shape (E, 3, N, K): evaluation set, `SERIES`,
            training set (seed) and layer.
        readers: The N fitted "reader" series, one per training set.
    """

    layers: tuple[int, ...]
    eval_names: tuple[str, ...]
    accuracy: NDArray[np.float64]
    readers: tuple[RepReader, ...]

    def per_seed(self, eval_name: str) -> NDArray[np.float64]:
        """Returns the accuracy of one evaluation set.

        Args:
            eval_name: A name from `eval_names`.

        Returns:
            Array of shape (3, N, K).

        Raises:
            ValueError: If the name is unknown.
        """
        if eval_name not in self.eval_names:
            raise ValueError(
                f"unknown evaluation set {eval_name!r}, expected one of "
                f"{self.eval_names}"
            )
        return np.asarray(self.accuracy[self.eval_names.index(eval_name)])


def sweep_from_activations(
    train: Sequence[Prepared],
    evals: Mapping[str, Prepared],
    *,
    method: Method = "pca_diff",
    preprocess: Preprocess = "center",
    seed: int = 0,
) -> SweepResult:
    """Fits the three series on each training set and scores every eval set.

    For training set i, `reader` is `RepReader.fit(method, ...)` on the real
    labels, `random` a random unit direction oriented by the same sign rule,
    and `shuffled` the `reader` scored against evaluation labels shuffled
    within each group (a permutation null).

    Args:
        train: One `(stimuli, activations)` pair per seed.
        evals: Named evaluation `(stimuli, activations)` pairs.
        method: Direction finder of the `reader` series.
        preprocess: Vector preparation, see `fit_direction`.
        seed: Base seed; each fit and shuffle derives its own.

    Returns:
        The accuracy of every evaluation set, series, seed and layer.

    Raises:
        ValueError: If `train` or `evals` is empty or the activations do not
            share the same layers.
    """
    if not train:
        raise ValueError("train needs at least one training set")
    if not evals:
        raise ValueError("evals needs at least one evaluation set")
    layers = train[0][1].layers
    prepared = [*train, *evals.values()]
    if any(act.layers != layers for _, act in prepared):
        raise ValueError(f"all activations must share the layers {layers}")
    eval_names = tuple(evals)
    accuracy = np.zeros((len(eval_names), len(SERIES), len(train), len(layers)))
    readers = []
    for i, (stimuli, act) in enumerate(train):
        reader = _fit(
            method, stimuli, act, preprocess, derive_seed(seed, "reader", i)
        )
        baseline = _fit(
            "random", stimuli, act, preprocess, derive_seed(seed, "random", i)
        )
        readers.append(reader)
        for e, name in enumerate(eval_names):
            e_stimuli, e_act = evals[name]
            e_groups = e_act.row_groups(e_stimuli.groups)
            e_labels = e_act.row_values(e_stimuli.labels)
            shuffled = shuffle_within_groups(
                e_labels,
                e_groups,
                seed=derive_seed(seed, "shuffled", i, name),
            )
            scores = reader.score(e_act.data)
            accuracy[e, 0, i] = group_accuracy(scores, e_groups, e_labels)
            accuracy[e, 1, i] = group_accuracy(
                baseline.score(e_act.data), e_groups, e_labels
            )
            accuracy[e, 2, i] = group_accuracy(scores, e_groups, shuffled)
    return SweepResult(layers, eval_names, accuracy, tuple(readers))


def shuffle_within_groups(
    labels: NDArray[np.int64], groups: NDArray[np.int64], *, seed: int
) -> NDArray[np.int64]:
    """Moves the label-1 member of every group to a random member.

    This is the permutation null of group accuracy: any scorer sits at
    chance against the shuffled labels.

    Args:
        labels: Binary labels of shape (n,).
        groups: Row groups of shape (g, k) with exactly one label-1 member
            each, padded with -1.
        seed: Seed of the random choice.

    Returns:
        New labels of shape (n,): in every group exactly one uniformly
        chosen member is 1 and the others are 0; rows outside the groups
        keep their label.

    Raises:
        ValueError: If `labels` is not binary, a group does not have exactly
            one label-1 member, or two groups share a row.
    """
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary (0 or 1)")
    index = parse_groups(groups, labels)
    rows = index.members[index.valid]
    if np.unique(rows).size != rows.size:
        raise ValueError("groups must not share rows")
    rng = np.random.default_rng(seed)
    sizes = index.valid.sum(axis=1)
    chosen = np.minimum(
        (rng.random(sizes.shape[0]) * sizes).astype(np.int64), sizes - 1
    )
    shuffled = labels.astype(np.int64, copy=True)
    shuffled[rows] = 0
    valid_first = np.argsort(~index.valid, axis=1, kind="stable")
    columns = valid_first[np.arange(sizes.shape[0]), chosen]
    shuffled[index.members[np.arange(sizes.shape[0]), columns]] = 1
    return shuffled


def _fit(
    method: Method,
    stimuli: StimulusSet,
    act: Activations,
    preprocess: Preprocess,
    seed: int,
) -> RepReader:
    return RepReader.fit(
        method,
        act.data,
        act.layers,
        labels=act.row_values(stimuli.labels),
        pairs=act.row_groups(stimuli.pairs),
        groups=act.row_groups(stimuli.groups),
        preprocess=preprocess,
        seed=seed,
    )
