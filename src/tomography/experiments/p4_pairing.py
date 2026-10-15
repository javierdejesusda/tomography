"""P4: how much the pairing and the preprocessing matter for a reading.

The pairs a direction is fitted on, the way their differences are prepared,
and the direction finder itself are varied one at a time around a reference
cell, and the first principal components of the pair differences are read
beyond the first, with the component chosen on the training pairs alone.
"""

from collections.abc import Sequence
import math
from typing import Literal
from typing import get_args

import numpy as np
from numpy.typing import NDArray

from tomography.datasets.records import Example
from tomography.datasets.stimulus import PairingMode
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.experiments.sweep import Prepared
from tomography.experiments.variants import ReadingTask
from tomography.experiments.variants import Variant
from tomography.reading.directions import FloatArray
from tomography.reading.directions import Preprocess
from tomography.reading.metrics import group_accuracy

Pairing = Literal["matched", "label_contrastive", "random"]
PAIRINGS: tuple[Pairing, ...] = get_args(Pairing)


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
