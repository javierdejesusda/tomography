"""The shared reading sweep: per-layer, per-seed accuracy curves.

Every reading experiment turns stimulus sets into accuracy curves for the
fitted reader and two baselines through this module, and writes them in the
`curves` results format.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
from numpy.typing import NDArray

from tomography.activations import Activations
from tomography.activations import PositionStrategy
from tomography.activations import collect_activations
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.reading.directions import Method
from tomography.reading.directions import Preprocess
from tomography.reading.metrics import bootstrap_ci
from tomography.reading.metrics import group_accuracy
from tomography.reading.metrics import parse_groups
from tomography.reading.reader import RepReader
from tomography.templates import ChatFormat
from tomography.viz.reading import curves_from_arrays
from tomography.viz.reading import plot_curves
from tomography.viz.style import save_figure
from tomography.viz.style import style_context

SERIES: tuple[str, str, str] = ("reader", "random", "shuffled")

Prepared = tuple[StimulusSet, Activations]


@dataclass(frozen=True, kw_only=True)
class ReadingConfig(ExperimentConfig):
    """Fields shared by the reading experiments R1 to R7.

    Attributes:
        n_seeds: Number of training sets (seeds).
        n_pairs: Training pairs (groups) per seed.
        n_eval_pairs: Evaluation groups; None means every group the
            evaluation split allows.
        method: Direction finder.
        preprocess: Vector preparation.
        position: Token positions read on the training sets.
        eval_position: Token positions read on the evaluation sets; None
            means `position`.
        layers: Layer indices; None means 1..L.
        bos: BOS policy; None uses the model's default.
        batch_size: Texts per forward pass.
    """

    n_seeds: int = 10
    n_pairs: int = 64
    n_eval_pairs: int | None = None
    method: Method = "pca_diff"
    preprocess: Preprocess = "center"
    position: PositionStrategy = "last"
    eval_position: PositionStrategy | None = None
    layers: tuple[int, ...] | None = None
    bos: BosMode | None = None
    batch_size: int = 8


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
    within each group (a permutation null). The shuffle moves the positive
    stimulus of a group, so all token rows of a stimulus move together.

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
            shuffled = e_act.row_values(
                shuffle_within_groups(
                    e_stimuli.labels,
                    e_stimuli.groups,
                    seed=derive_seed(seed, "shuffled", i, name),
                )
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


def chat_format_for(loaded: LoadedModel) -> ChatFormat:
    """Returns the chat format a model's prompts are rendered with.

    Args:
        loaded: The wrapped model.

    Returns:
        The tokenizer's chat template iff the checkpoint is post-trained,
        else the plain `USER:` and `ASSISTANT:` format.
    """
    if loaded.checkpoint is not None and loaded.checkpoint.chat_format == (
        "template"
    ):
        return ChatFormat.from_tokenizer(loaded.tokenizer)
    return ChatFormat("plain")


def stimulus_activations(
    loaded: LoadedModel,
    stimuli: StimulusSet,
    *,
    layers: Sequence[int] | None,
    position: PositionStrategy,
    bos: BosMode | None = None,
    batch_size: int = 8,
) -> Activations:
    """Collects the activations of a stimulus set.

    Args:
        loaded: The wrapped model.
        stimuli: The stimuli, with the spans the position needs.
        layers: Layer indices in 0..L; None means all of them.
        position: Which token positions to read.
        bos: BOS policy; None uses the model's default.
        batch_size: Texts per forward pass.

    Returns:
        The activations, one row per read token position.
    """
    return collect_activations(
        loaded,
        stimuli.texts,
        layers=layers,
        position=position,
        response_spans=stimuli.response_spans,
        concept_spans=stimuli.concept_spans,
        bos=bos,
        batch_size=batch_size,
    )


def reading_sweep(
    loaded: LoadedModel,
    train_sets: Sequence[StimulusSet],
    eval_sets: Mapping[str, StimulusSet],
    *,
    method: Method = "pca_diff",
    preprocess: Preprocess = "center",
    position: PositionStrategy = "last",
    eval_position: PositionStrategy | None = None,
    layers: Sequence[int] | None = None,
    seed: int = 0,
    bos: BosMode | None = None,
    batch_size: int = 8,
) -> SweepResult:
    """Collects activations once per set and runs `sweep_from_activations`.

    Args:
        loaded: The wrapped model.
        train_sets: One training stimulus set per seed.
        eval_sets: Named evaluation stimulus sets.
        method: Direction finder of the `reader` series.
        preprocess: Vector preparation, see `fit_direction`.
        position: Token positions read on the training sets.
        eval_position: Token positions read on the evaluation sets; None
            means `position`.
        layers: Layer indices; None means 1..L, because layer 0 is the
            embedding output, which can carry no signal under `last`.
        seed: Base seed of the fits and the shuffles.
        bos: BOS policy; None uses the model's default.
        batch_size: Texts per forward pass.

    Returns:
        The accuracy of every evaluation set, series, seed and layer.

    Raises:
        ValueError: If `train_sets` or `eval_sets` is empty (checked before
            any forward pass) or an input is invalid.
    """
    if not train_sets:
        raise ValueError("train_sets is empty")
    if not eval_sets:
        raise ValueError("eval_sets is empty")
    selected = (
        tuple(range(1, loaded.n_layers + 1))
        if layers is None
        else tuple(layers)
    )

    def collect(stimuli: StimulusSet, where: PositionStrategy) -> Prepared:
        act = stimulus_activations(
            loaded,
            stimuli,
            layers=selected,
            position=where,
            bos=bos,
            batch_size=batch_size,
        )
        return stimuli, act

    where_eval = position if eval_position is None else eval_position
    return sweep_from_activations(
        [collect(stimuli, position) for stimuli in train_sets],
        {name: collect(s, where_eval) for name, s in eval_sets.items()},
        method=method,
        preprocess=preprocess,
        seed=seed,
    )


def write_curves(
    run_dir: RunDir,
    per_seed: NDArray[np.float64],
    layers: Sequence[int],
    *,
    series: Sequence[str],
    stem: str = "curves",
    xlabel: str = "Layer",
    ylabel: str = "Accuracy",
    chance: float | None = None,
    n_layers: int | None = None,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
) -> None:
    """Writes the `curves` arrays and sidecar of one model.

    The arrays hold `x (K,)`, `mean (S, K)`, `lo (S, K)`, `hi (S, K)` and
    `per_seed (S, N, K)`, with a percentile bootstrap band over the seeds.

    Args:
        run_dir: The run to write into.
        per_seed: Values of shape (S, N, K): series, seed, layer.
        layers: The x value of each of the K layers.
        series: The name of each series.
        stem: File stem of the pair (`<stem>.npz` and `<stem>.json`).
        xlabel: Label of the x axis.
        ylabel: Label of the y axis.
        chance: Level of the chance line, or None for no line.
        n_layers: Depth of the model, for a relative-depth axis.
        alpha: One minus the coverage of the band.
        n_boot: Number of bootstrap resamples.
        seed: Seed of the resampling.

    Raises:
        ValueError: If `per_seed` is not 3D or `series` or `layers` do not
            match its shape.
    """
    if per_seed.ndim != 3:
        raise ValueError(
            f"per_seed must have shape (S, N, K), got {per_seed.shape}"
        )
    if len(series) != per_seed.shape[0]:
        raise ValueError(
            f"got {len(series)} series names for {per_seed.shape[0]} series"
        )
    if len(layers) != per_seed.shape[2]:
        raise ValueError(
            f"got {len(layers)} layers for {per_seed.shape[2]} columns"
        )
    lo, hi = bootstrap_ci(
        np.moveaxis(per_seed, 1, 0), alpha=alpha, n_boot=n_boot, seed=seed
    )
    run_dir.write_arrays(
        stem,
        x=np.asarray(layers, dtype=np.float64),
        mean=per_seed.mean(axis=1),
        lo=lo,
        hi=hi,
        per_seed=per_seed,
    )
    run_dir.write_json(
        stem,
        {
            "series": list(series),
            "xlabel": xlabel,
            "ylabel": ylabel,
            "chance": chance,
            "n_layers": n_layers,
        },
    )


def plot_curves_file(
    run_dir: RunDir,
    out_dir: Path,
    *,
    stem: str = "curves",
    name: str | None = None,
) -> list[Path]:
    """Draws a `curves` file written by `write_curves` and saves it.

    Args:
        run_dir: The run holding the pair.
        out_dir: Folder of the figures.
        stem: File stem of the pair.
        name: File name of the figures; None means `stem`.

    Returns:
        The PDF, PNG and SVG paths.
    """
    arrays = run_dir.read_arrays(stem)
    sidecar = cast(dict[str, Any], run_dir.read_json(stem))
    curves = curves_from_arrays(
        arrays["x"], arrays["mean"], arrays["lo"], arrays["hi"]
    )
    with style_context():
        fig = plot_curves(
            curves,
            series=sidecar["series"],
            per_seed=list(arrays["per_seed"]),
            xlabel=sidecar["xlabel"],
            ylabel=sidecar["ylabel"],
            chance=sidecar["chance"],
            n_layers=sidecar["n_layers"],
        )
    return save_figure(fig, out_dir, name or stem)
