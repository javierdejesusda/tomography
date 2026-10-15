"""P4: how much the pairing and the preprocessing matter for a reading.

The pairs a direction is fitted on, the way their differences are prepared,
and the direction finder itself are varied one at a time around a reference
cell, and the first principal components of the pair differences are read
beyond the first, with the component chosen on the training pairs alone.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any
from typing import Literal
from typing import get_args

import numpy as np
from numpy.typing import NDArray

from tomography.datasets.records import Example
from tomography.datasets.stimulus import PairingMode
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.matrices import plot_matrix_file
from tomography.experiments.matrices import write_matrix
from tomography.experiments.sweep import Prepared
from tomography.experiments.sweep import ReadingConfig
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import sweep_from_activations
from tomography.experiments.sweep import write_curves
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import ReadingTask
from tomography.experiments.variants import Variant
from tomography.experiments.variants import eval_pairs
from tomography.experiments.variants import eval_seed
from tomography.experiments.variants import expand_tasks
from tomography.experiments.variants import load_sources
from tomography.experiments.variants import pair_capacity
from tomography.experiments.variants import prepare
from tomography.experiments.variants import task_set
from tomography.experiments.variants import task_splits
from tomography.experiments.variants import template_variants
from tomography.experiments.variants import variant_cosines
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.directions import FloatArray
from tomography.reading.directions import Method
from tomography.reading.directions import Preprocess
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import bootstrap_ci
from tomography.reading.metrics import group_accuracy
from tomography.templates import ChatFormat

Pairing = Literal["matched", "label_contrastive", "random"]
PAIRINGS: tuple[Pairing, ...] = get_args(Pairing)
COLUMNS: tuple[str, ...] = (
    "task",
    "method",
    "pairing",
    "preprocess",
    "layer",
    "select",
    "report",
    "lo",
    "hi",
    "shuffled",
)
COMPONENT_COLUMNS: tuple[str, ...] = (
    "task",
    "pairing",
    "layer",
    "pc1",
    "pc1_lo",
    "pc1_hi",
    "cv",
    "cv_lo",
    "cv_hi",
    "cv_component",
    "explained_pc1",
)
SPLITS: tuple[str, str] = ("select", "report")


def pairing_mode(task: ReadingTask, pairing: Pairing) -> PairingMode:
    """Names the stimulus pairing that builds a pairing of a task.

    Args:
        task: The reading task.
        pairing: `matched` pairs the label-1 member of a group with another
            member of the same group, `label_contrastive` a label-1 with a
            label-0 stimulus anywhere in the examples, and `random` any two
            stimuli, whatever their labels.

    Returns:
        The mode of `build_stimulus_set`.

    Raises:
        ValueError: If the pairing is unknown, or `matched` is asked of a
            task without groups.
    """
    if pairing not in PAIRINGS:
        raise ValueError(
            f"unknown pairing {pairing!r}, expected one of {PAIRINGS}"
        )
    if pairing == "matched":
        if task.pairing != "by_group":
            raise ValueError(
                "matched pairing needs a task with groups, "
                f"{task.name!r} has none"
            )
        return "by_group"
    return pairing


def pairing_set(
    task: ReadingTask,
    variant: Variant,
    examples: Sequence[Example],
    pairing: Pairing,
    *,
    n_pairs: int,
    seed: int,
) -> StimulusSet:
    """Renders training examples and pairs them in one of the three ways.

    Args:
        task: The reading task.
        variant: The variant to render, normally the paper one.
        examples: The training examples.
        pairing: How to pair them, see `pairing_mode`.
        n_pairs: Number of pairs.
        seed: Seed of the draw.

    Returns:
        The stimulus set.

    Raises:
        ValueError: If the pairing does not fit the task or `n_pairs` is
            more than the examples allow.
    """
    return build_stimulus_set(
        examples,
        variant.template,
        variant.chat,
        pairing=pairing_mode(task, pairing),
        concept=variant.concept,
        group_ids=(
            [str(e.meta["pair"]) for e in examples]
            if pairing == "matched"
            else None
        ),
        n_pairs=n_pairs,
        seed=seed,
    )


def diff_components(
    h: FloatArray,
    pairs: NDArray[np.int64],
    *,
    preprocess: Preprocess,
    n_components: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Takes the leading principal components of the pair differences.

    The differences are `h[pairs[:, 0]] - h[pairs[:, 1]]` in the stored
    order, prepared as `fit_direction` prepares them for `pca_diff`, so the
    first component is that direction.

    Args:
        h: Activations of shape (K, n, d).
        pairs: Row indices of shape (m, 2).
        preprocess: `center` subtracts the mean difference, `unit` first
            scales each difference to unit length and then centers, `none`
            does neither.
        n_components: Number of components, from 1 to min(m, d).

    Returns:
        The unit components of shape (K, C, d), each with its largest
        coordinate positive, and their explained-variance ratios of shape
        (K, C): the share of the squared norm of the prepared differences.

    Raises:
        ValueError: If the shapes do not fit, `preprocess` is unknown,
            `n_components` is out of range, or the differences have no
            variance.
    """
    rows = np.asarray(h, dtype=np.float64)
    pair_index = np.asarray(pairs)
    if rows.ndim != 3:
        raise ValueError(
            f"diff_components needs h of shape (K, n, d), got {rows.shape}"
        )
    if pair_index.ndim != 2 or pair_index.shape[1] != 2:
        raise ValueError(
            "diff_components needs pairs of shape (m, 2), got "
            f"{pair_index.shape}"
        )
    if pair_index.size and (
        pair_index.min() < 0 or pair_index.max() >= rows.shape[1]
    ):
        raise ValueError("diff_components needs pairs that index the rows of h")
    if preprocess not in get_args(Preprocess):
        raise ValueError(f"unsupported preprocess {preprocess!r}")
    limit = min(pair_index.shape[0], rows.shape[2])
    if not 1 <= n_components <= limit:
        raise ValueError(
            f"n_components must be between 1 and min(m, d) = {limit}, got "
            f"{n_components}"
        )
    components = np.empty((rows.shape[0], n_components, rows.shape[2]))
    explained = np.empty((rows.shape[0], n_components))
    for k, layer in enumerate(rows):
        vectors = layer[pair_index[:, 0]] - layer[pair_index[:, 1]]
        if preprocess == "unit":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms > 0, norms, 1.0)
        bound = np.finfo(np.float64).eps * max(vectors.shape)
        bound *= float(np.abs(vectors).max())
        if preprocess != "none":
            vectors = vectors - vectors.mean(axis=0)
        _, singular_values, axes = np.linalg.svd(vectors, full_matrices=False)
        if singular_values[0] <= bound:
            raise ValueError("the pair differences have no variance")
        energy = singular_values**2
        explained[k] = energy[:n_components] / energy.sum()
        leading = axes[:n_components]
        top = np.abs(leading).argmax(axis=1)
        leading[leading[np.arange(n_components), top] < 0] *= -1.0
        components[k] = leading
    return components, explained


def component_accuracy(
    train: Prepared, evaluated: Prepared, components: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Scores every component as a reader, oriented on the training groups.

    The sign of each component at each layer is the one for which the
    label-1 member wins at least half of the training groups, ties sharing
    the credit; the evaluated labels play no part in it.

    Args:
        train: The training stimuli and activations that orient the signs.
        evaluated: The stimuli and activations that are scored.
        components: Directions of shape (K, C, d), as `diff_components`
            returns them.

    Returns:
        The group accuracy on `evaluated`, shape (K, C).

    Raises:
        ValueError: If `components` does not fit the activations.
    """
    basis = np.asarray(components, dtype=np.float64)
    accuracy = _group_accuracy(train, basis)
    sign = np.where(accuracy >= 0.5, 1.0, -1.0)
    return _group_accuracy(evaluated, basis, sign)


def cv_component(
    prepared: Prepared,
    *,
    layer_index: int,
    preprocess: Preprocess,
    n_components: int,
    n_folds: int,
    seed: int,
) -> int:
    """Picks the principal component that reads best on held-out pairs.

    The pairs are split into folds at random. For each fold the components
    are fitted on the other folds, each is oriented there so that the
    label-1 member wins at least half of the pairs with distinct labels, and
    it is scored on the held-out pairs with distinct labels, a win counting
    one and a tie one half. The component with the most credit over all
    folds is chosen, the first on a tie. Only the training pairs are used.

    Args:
        prepared: The training stimuli and activations.
        layer_index: Position of the layer within the activations.
        preprocess: How the pair differences are prepared.
        n_components: Number of components to choose among.
        n_folds: Number of folds, from 2 to the number of pairs.
        seed: Seed of the fold assignment.

    Returns:
        The 0-based index of the chosen component.

    Raises:
        ValueError: If the layer is out of range, the folds do not fit the
            pairs and components, or the pairs have no distinct labels to
            orient and score the components.
    """
    stimuli, act = prepared
    n_layers = act.data.shape[0]
    if not 0 <= layer_index < n_layers:
        raise ValueError(
            f"cv_component needs layer_index in [0, {n_layers}), got "
            f"{layer_index}"
        )
    pairs = act.row_groups(stimuli.pairs)
    n_pairs = pairs.shape[0]
    if not 2 <= n_folds <= n_pairs:
        raise ValueError(
            f"n_folds must be between 2 and the {n_pairs} pairs, got {n_folds}"
        )
    if n_pairs - math.ceil(n_pairs / n_folds) < n_components:
        raise ValueError(
            f"folds too small: {n_pairs} pairs in {n_folds} folds leave "
            f"fewer than {n_components} to fit the components"
        )
    labels = act.row_values(stimuli.labels)
    distinct = labels[pairs[:, 0]] != labels[pairs[:, 1]]
    rows = act.data[layer_index : layer_index + 1].astype(np.float64)
    order = np.random.default_rng(seed).permutation(n_pairs)
    credit = np.zeros(n_components)
    for held_out in np.array_split(order, n_folds):
        fit = np.setdiff1d(order, held_out)
        components, _ = diff_components(
            rows, pairs[fit], preprocess=preprocess, n_components=n_components
        )
        fit_pairs = pairs[fit[distinct[fit]]]
        if not len(fit_pairs):
            raise ValueError(
                "cv_component needs pairs with distinct labels outside "
                "every fold to orient the components"
            )
        test_pairs = pairs[held_out[distinct[held_out]]]
        fit_credit = _pair_credit(rows, components, fit_pairs, labels)
        share = fit_credit / len(fit_pairs)
        won = _pair_credit(rows, components, test_pairs, labels)
        credit += np.where(share >= 0.5, won, len(test_pairs) - won)
    return int(credit.argmax())


def _pair_credit(
    rows: NDArray[np.float64],
    components: NDArray[np.float64],
    pairs: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> NDArray[np.float64]:
    scores = components[0] @ rows[0].T
    first, second = pairs.T
    margin = scores[:, first] - scores[:, second]
    margin = np.where(labels[first] > labels[second], margin, -margin)
    credit: NDArray[np.float64] = np.sum(margin > 0, axis=1, dtype=np.float64)
    credit += 0.5 * np.sum(margin == 0, axis=1)
    return credit


def _group_accuracy(
    prepared: Prepared,
    basis: NDArray[np.float64],
    sign: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    stimuli, act = prepared
    if (
        basis.ndim != 3
        or basis.shape[0] != act.data.shape[0]
        or basis.shape[2] != act.data.shape[2]
    ):
        raise ValueError(
            "component_accuracy needs components of shape "
            f"({act.data.shape[0]}, C, {act.data.shape[2]}), got {basis.shape}"
        )
    scores = np.einsum("kcd,knd->kcn", basis, act.data.astype(np.float64))
    if sign is not None:
        scores = scores * sign[..., None]
    return group_accuracy(
        scores,
        act.row_groups(stimuli.groups),
        act.row_values(stimuli.labels),
    )


@dataclass(frozen=True, kw_only=True)
class Config(ReadingConfig):
    """P4 settings: pairings, preprocessings and direction finders compared.

    The shared defaults apply except for the experiment id, the model and the
    number of evaluation pairs. `method`, `preprocess` and the task's own
    pairing (`matched` for a task with groups, else `label_contrastive`)
    name the reference cell: every comparison varies one of the three around
    it, and the components are taken under `preprocess`.

    Attributes:
        tasks: Keys of `READING_TASKS` to read; binary tasks only.
        methods: Direction finders compared, at least two, with `method`
            among them.
        pairings: Ways of pairing the training examples. `matched` is
            skipped for a task without groups, and the task's own pairing
            must remain.
        preprocesses: Vector preparations compared, with `preprocess` among
            them.
        n_components: Principal components of the pair differences scored.
        n_folds: Folds of the cross-validated choice among the components.
        n_boot: Number of bootstrap resamples of the bands and intervals.
    """

    experiment: str = "p4"
    model: str = "gpt2"
    n_eval_pairs: int | None = 256
    tasks: tuple[str, ...] = ("sentiment", "truth", "utility")
    methods: tuple[Method, ...] = (
        "pca_diff",
        "pca_raw",
        "mean_diff",
        "logistic",
        "kmeans",
        "random",
    )
    pairings: tuple[Pairing, ...] = ("matched", "label_contrastive", "random")
    preprocesses: tuple[Preprocess, ...] = ("center", "unit", "none")
    n_components: int = 8
    n_folds: int = 5
    n_boot: int = 2000


@dataclass(frozen=True, eq=False)
class _Plan:
    """What one task reads: its pairings and the sets it will prepare."""

    task: ReadingTask
    own: Pairing
    pairings: tuple[Pairing, ...]
    skipped: dict[str, str]
    train: dict[Pairing, list[StimulusSet]]
    evaluation: dict[str, StimulusSet]


@dataclass(frozen=True, eq=False)
class _Result:
    """The measurements of one task.

    Attributes:
        plan: What was read.
        layers: Layer index of each of the K layers.
        layer: The reading layer, the best select layer of the reference
            cell.
        accuracy: Array of shape (M, P, R, 2, 3, N, K): method, pairing,
            preprocessing, select or report, `SERIES`, seed and layer.
        cosine: Cosines between the oriented directions of the methods under
            the reference pairing and preprocessing, shape (K, M, M).
        directions: Those oriented unit directions, shape (M, N, K, d).
        components: Report accuracy of every component at the reading
            layer, shape (P, N, C).
        explained: Explained-variance share of every component at the
            reading layer, shape (P, N, C).
        cv_component: The 0-based component the cross-validation picked,
            shape (P, N).
        cv_accuracy: The report accuracy of that component, shape (P, N).
    """

    plan: _Plan
    layers: tuple[int, ...]
    layer: int
    accuracy: NDArray[np.float64]
    cosine: NDArray[np.float64]
    directions: NDArray[np.float32]
    components: NDArray[np.float64]
    explained: NDArray[np.float64]
    cv_component: NDArray[np.int64]
    cv_accuracy: NDArray[np.float64]

    @property
    def at(self) -> int:
        """The position of the reading layer among the layers."""
        return self.layers.index(self.layer)


def _check_axis(
    label: str, values: Sequence[str], allowed: Sequence[str]
) -> None:
    if not values:
        raise ValueError(f"{label}s must not be empty")
    for value in values:
        if value not in allowed:
            raise ValueError(
                f"unknown {label} {value!r}, expected one of {tuple(allowed)}"
            )
    if len(set(values)) != len(values):
        raise ValueError(f"{label}s must not repeat, got {tuple(values)}")


def _checked_tasks(config: Config) -> tuple[ReadingTask, ...]:
    if config.n_seeds < 2:
        raise ValueError(f"n_seeds must be at least 2, got {config.n_seeds}")
    if "response" in (config.position, config.eval_position):
        raise ValueError(
            "p4 reads one row per stimulus, not the 'response' position"
        )
    _check_axis("method", config.methods, get_args(Method))
    _check_axis("pairing", config.pairings, PAIRINGS)
    _check_axis("preprocess", config.preprocesses, get_args(Preprocess))
    if len(config.methods) < 2:
        raise ValueError(
            "methods needs at least two direction finders to compare, got "
            f"{config.methods}"
        )
    if config.method not in config.methods:
        raise ValueError(
            f"method {config.method!r} is not in methods {config.methods}"
        )
    if config.preprocess not in config.preprocesses:
        raise ValueError(
            f"preprocess {config.preprocess!r} is not in preprocesses "
            f"{config.preprocesses}"
        )
    if config.n_components < 1:
        raise ValueError(
            f"n_components must be at least 1, got {config.n_components}"
        )
    if not 2 <= config.n_folds <= config.n_pairs:
        raise ValueError(
            f"n_folds must be between 2 and the {config.n_pairs} pairs, got "
            f"{config.n_folds}"
        )
    fit = config.n_pairs - math.ceil(config.n_pairs / config.n_folds)
    if fit < config.n_components:
        raise ValueError(
            f"folds too small: {config.n_pairs} pairs in {config.n_folds} "
            f"folds leave {fit} to fit {config.n_components} components"
        )
    tasks = tuple(READING_TASKS[name] for name in expand_tasks(config.tasks))
    for task in tasks:
        if task.pairing == "one_vs_rest":
            raise ValueError(f"p4 reads binary tasks only, not {task.name!r}")
    return tasks


def _plan(
    config: Config,
    task: ReadingTask,
    chat: ChatFormat,
    examples: Mapping[str, Sequence[Example]],
) -> _Plan:
    """Builds every stimulus set of a task; the model has not run yet."""
    name = task.name
    (variant,), _ = template_variants(task, chat, kinds=("paper",))
    splits = task_splits(task, examples, seed=config.seed)
    own: Pairing = (
        "matched" if task.pairing == "by_group" else "label_contrastive"
    )
    skipped: dict[str, str] = {
        pairing: f"matched pairing needs a task with groups, {name!r} has none"
        for pairing in config.pairings
        if pairing == "matched" and task.pairing != "by_group"
    }
    pairings = tuple(p for p in config.pairings if p not in skipped)
    if own not in pairings:
        raise ValueError(
            f"task {name!r} is read on its own pairing {own!r}, which is not "
            f"in pairings {pairings}"
        )
    capacity = pair_capacity(task, splits.train)
    if config.n_pairs > capacity:
        raise ValueError(
            f"task {name!r} has {capacity} training pairs, fewer than "
            f"n_pairs = {config.n_pairs}"
        )
    train: dict[Pairing, list[StimulusSet]] = {}
    for pairing in pairings:
        try:
            train[pairing] = [
                pairing_set(
                    task,
                    variant,
                    splits.train,
                    pairing,
                    n_pairs=config.n_pairs,
                    seed=derive_seed(config.seed, "train", i),
                )
                for i in range(config.n_seeds)
            ]
        except ValueError as error:
            raise ValueError(
                f"cannot pair the training examples of {name!r} as "
                f"{pairing!r}: {error}"
            ) from error
    evaluation = {
        split: task_set(
            task,
            variant,
            part,
            n_pairs=eval_pairs(task, part, config.n_eval_pairs),
            seed=eval_seed(task, split, config.seed),
        )
        for split, part in zip(
            SPLITS, (splits.select, splits.report), strict=True
        )
    }
    return _Plan(task, own, pairings, skipped, train, evaluation)


def _oriented(
    directions: NDArray[np.float32], signs: NDArray[np.float32]
) -> NDArray[np.float32]:
    return np.asarray(directions * signs[:, None], dtype=np.float32)


def _read_task(loaded: LoadedModel, config: Config, plan: _Plan) -> _Result:
    """Reads every cell of a task and scores its components."""
    position = (
        config.position
        if config.eval_position is None
        else config.eval_position
    )
    evals = {
        split: prepare(loaded, stimuli, position, config)
        for split, stimuli in plan.evaluation.items()
    }
    train = {
        pairing: [prepare(loaded, s, config.position, config) for s in sets]
        for pairing, sets in plan.train.items()
    }
    cells: dict[tuple[int, int, int], NDArray[np.float64]] = {}
    directions: dict[int, NDArray[np.float32]] = {}
    layers: tuple[int, ...] = ()
    reference: tuple[int, int, int] | None = None
    for m, method in enumerate(config.methods):
        for p, pairing in enumerate(plan.pairings):
            for r, preprocess in enumerate(config.preprocesses):
                sweep = sweep_from_activations(
                    train[pairing],
                    evals,
                    method=method,
                    preprocess=preprocess,
                    seed=config.seed,
                )
                layers = sweep.layers
                cells[m, p, r] = np.stack(
                    [sweep.per_seed(split) for split in SPLITS]
                )
                if pairing == plan.own and preprocess == config.preprocess:
                    directions[m] = np.stack(
                        [
                            _oriented(x.directions, x.signs)
                            for x in sweep.readers
                        ]
                    )
                    if method == config.method:
                        reference = (m, p, r)
    assert reference is not None
    accuracy = np.array(
        [
            [
                [cells[m, p, r] for r in range(len(config.preprocesses))]
                for p in range(len(plan.pairings))
            ]
            for m in range(len(config.methods))
        ]
    )
    layer = best_layer(accuracy[reference][0, 0].mean(axis=0), layers)
    at = layers.index(layer)
    shape = (len(plan.pairings), config.n_seeds, config.n_components)
    components = np.zeros(shape)
    explained = np.zeros(shape)
    chosen = np.zeros(shape[:2], dtype=np.int64)
    report = evals["report"]
    for p, pairing in enumerate(plan.pairings):
        for i, prepared in enumerate(train[pairing]):
            stimuli, act = prepared
            basis, share = diff_components(
                act.data,
                act.row_groups(stimuli.pairs),
                preprocess=config.preprocess,
                n_components=config.n_components,
            )
            components[p, i] = component_accuracy(prepared, report, basis)[at]
            explained[p, i] = share[at]
            chosen[p, i] = cv_component(
                prepared,
                layer_index=at,
                preprocess=config.preprocess,
                n_components=config.n_components,
                n_folds=config.n_folds,
                seed=derive_seed(config.seed, "cv", pairing, i),
            )
    stacked = np.stack([directions[m] for m in range(len(config.methods))])
    return _Result(
        plan,
        layers,
        layer,
        accuracy,
        variant_cosines(stacked),
        stacked,
        components,
        explained,
        chosen,
        np.take_along_axis(components, chosen[..., None], axis=2)[..., 0],
    )


def _write_task(
    run_dir: RunDir, config: Config, result: _Result, n_layers: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Writes the files of a task; returns its grid and component rows."""
    plan = result.plan
    name = plan.task.name
    methods, preprocesses = config.methods, config.preprocesses
    run_dir.write_arrays(
        f"grid_{name}",
        layers=np.asarray(result.layers, dtype=np.int64),
        accuracy=result.accuracy,
        cosine=result.cosine,
        components=result.components,
        explained=result.explained,
        cv_accuracy=result.cv_accuracy,
        cv_component=result.cv_component,
    )
    write_matrix(
        run_dir,
        f"methods_{name}",
        result.cosine[result.at],
        rows=methods,
        cols=methods,
        title=(
            f"Cosine between the directions of the methods on {name} at "
            f"layer {result.layer}"
        ),
    )
    ref_m = methods.index(config.method)
    ref_p = plan.pairings.index(plan.own)
    ref_r = preprocesses.index(config.preprocess)
    others = [m for m in range(len(methods)) if m != ref_m]
    unit = result.directions.astype(np.float64)
    stem = f"convergence_{name}"
    write_curves(
        run_dir,
        np.einsum("mnkd,nkd->mnk", unit[others], unit[ref_m]),
        result.layers,
        series=[methods[m] for m in others],
        stem=stem,
        ylabel=f"Cosine to {config.method}",
        n_layers=n_layers,
        n_boot=config.n_boot,
        seed=derive_seed(config.seed, "curves", stem),
    )
    for cut, report, series in (
        ("methods", result.accuracy[:, ref_p, ref_r], methods),
        ("pairing", result.accuracy[ref_m, :, ref_r], plan.pairings),
        ("preprocess", result.accuracy[ref_m, ref_p], preprocesses),
    ):
        stem = f"curves_{name}_{cut}"
        write_curves(
            run_dir,
            report[:, 1, 0],
            result.layers,
            series=series,
            stem=stem,
            chance=0.5,
            n_layers=n_layers,
            n_boot=config.n_boot,
            seed=derive_seed(config.seed, "curves", stem),
        )
    stem = f"components_{name}"
    write_curves(
        run_dir,
        result.components,
        range(1, config.n_components + 1),
        series=plan.pairings,
        stem=stem,
        xlabel="Component",
        chance=0.5,
        n_boot=config.n_boot,
        seed=derive_seed(config.seed, "curves", stem),
    )
    rows = []
    for m, method in enumerate(methods):
        for p, pairing in enumerate(plan.pairings):
            for r, preprocess in enumerate(preprocesses):
                cell = result.accuracy[m, p, r]
                layer = best_layer(cell[0, 0].mean(axis=0), result.layers)
                at = result.layers.index(layer)
                low, high = bootstrap_ci(
                    cell[1, 0, :, at],
                    n_boot=config.n_boot,
                    seed=derive_seed(
                        config.seed, "ci", name, method, pairing, preprocess
                    ),
                )
                values = {
                    "task": name,
                    "method": method,
                    "pairing": pairing,
                    "preprocess": preprocess,
                    "layer": layer,
                    "select": cell[0, 0, :, at].mean(),
                    "report": cell[1, 0, :, at].mean(),
                    "lo": low,
                    "hi": high,
                    "shuffled": cell[1, 2, :, at].mean(),
                }
                rows.append({column: values[column] for column in COLUMNS})
    component_rows = []
    for p, pairing in enumerate(plan.pairings):
        pc1 = result.components[p, :, 0]
        cv = result.cv_accuracy[p]
        pc1_lo, pc1_hi = bootstrap_ci(
            pc1,
            n_boot=config.n_boot,
            seed=derive_seed(config.seed, "ci", name, pairing, "pc1"),
        )
        cv_lo, cv_hi = bootstrap_ci(
            cv,
            n_boot=config.n_boot,
            seed=derive_seed(config.seed, "ci", name, pairing, "cv"),
        )
        values = {
            "task": name,
            "pairing": pairing,
            "layer": result.layer,
            "pc1": pc1.mean(),
            "pc1_lo": pc1_lo,
            "pc1_hi": pc1_hi,
            "cv": cv.mean(),
            "cv_lo": cv_lo,
            "cv_hi": cv_hi,
            "cv_component": int(np.bincount(result.cv_component[p]).argmax())
            + 1,
            "explained_pc1": result.explained[p, :, 0].mean(),
        }
        component_rows.append({c: values[c] for c in COMPONENT_COLUMNS})
    return rows, component_rows


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    examples: Mapping[str, Sequence[Example]],
) -> None:
    """Reads every cell of the grid and writes the study.

    The settings, the training capacity and every stimulus set are checked
    or built before the model runs. Per task it writes `grid_<task>` (the
    accuracy of every cell and seed, the cosines between the methods and the
    scores of the components), `methods_<task>`, `convergence_<task>`,
    `curves_<task>_methods`, `curves_<task>_pairing`,
    `curves_<task>_preprocess` and `components_<task>`; and `table.json`,
    `table_components.json` and `summary.json`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        examples: The examples of each source that the tasks read.

    Raises:
        ValueError: If a setting is invalid, a task is not binary, a source
            is missing, the task's own pairing is not among the pairings, or
            the training examples cannot give `n_pairs` pairs.
    """
    tasks = _checked_tasks(config)
    if config.n_components > loaded.d_model:
        raise ValueError(
            f"n_components must be at most the model width {loaded.d_model}, "
            f"got {config.n_components}"
        )
    chat = chat_format_for(loaded)
    plans = [_plan(config, task, chat, examples) for task in tasks]
    results: dict[str, _Result] = {}
    rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for plan in plans:
        result = _read_task(loaded, config, plan)
        results[plan.task.name] = result
        task_rows, task_component_rows = _write_task(
            run_dir, config, result, loaded.n_layers
        )
        rows.extend(task_rows)
        component_rows.extend(task_component_rows)
    run_dir.write_json("table", rows)
    run_dir.write_json("table_components", component_rows)
    run_dir.write_json(
        "summary",
        {
            "tasks": list(results),
            "layers": {n: r.layer for n, r in results.items()},
            "reference": {
                n: {
                    "method": config.method,
                    "pairing": r.plan.own,
                    "preprocess": config.preprocess,
                }
                for n, r in results.items()
            },
            "skipped": {n: r.plan.skipped for n, r in results.items()},
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the model and the datasets, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    tasks = _checked_tasks(config)
    sources = list(dict.fromkeys(task.source for task in tasks))
    execute(config, run_dir, load_model(config.model), load_sources(sources))


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws every curve and every methods matrix of a P4 run.

    Args:
        run_dir: A finished P4 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths.
    """
    paths: list[Path] = []
    for prefix in ("curves", "convergence", "components"):
        for file in sorted(run_dir.path.glob(f"{prefix}_*.npz")):
            paths += plot_curves_file(run_dir, out_dir, stem=file.stem)
    for file in sorted(run_dir.path.glob("methods_*.npz")):
        paths += plot_matrix_file(
            run_dir,
            out_dir,
            stem=file.stem,
            kind="diverging",
            vmin=-1.0,
            vmax=1.0,
        )
    return paths
