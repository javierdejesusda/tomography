"""Readers beyond one oriented direction per layer.

`RepReader` holds one direction per layer. The readers here keep a
direction only where the scorer is linear: LDA is a `RepReader`, and the
MLP reader has none. All of them satisfy `Reader`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from typing import runtime_checkable

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.neural_network import MLPClassifier

from tomography.reading.directions import FloatArray
from tomography.reading.reader import RepReader
from tomography.reading.reader import _layer_seed
from tomography.reading.reader import orient


@runtime_checkable
class Reader[InputT](Protocol):
    """A fitted reader that scores stimuli from the activations it reads.

    `RepReader` takes activations of shape (K, n, d), one row per stimulus.
    A reader that pools tokens takes the per-token `Activations` of the
    `response` position instead.
    """

    @property
    def method(self) -> str:
        """Name of the method that produced the reader."""
        ...

    @property
    def layers(self) -> tuple[int, ...]:
        """Indices of the layers the reader reads, in input order."""
        ...

    def score(self, data: InputT, /) -> NDArray[np.float32]:
        """Scores every stimulus; higher means more of the positive pole.

        Args:
            data: The activations of the layers in `layers`.

        Returns:
            float32 scores of shape (K, n_stimuli): one row per layer, or a
            single row for a reader that combines its layers.
        """
        ...


def fit_lda(
    h: FloatArray,
    layers: Sequence[int],
    *,
    labels: NDArray[np.int64],
    pairs: NDArray[np.int64] | None = None,
    groups: NDArray[np.int64] | None = None,
    shrinkage: float | None = None,
) -> RepReader:
    """Fits LDA per layer: the mass-mean direction under a shrunk covariance.

    The direction of a layer is `S^-1 (m1 - m0)`, with m1 and m0 the class
    means and S the pooled within-class covariance C (rows centered on
    their class mean, divided by n) shrunk toward a scaled identity:
    `S = (1 - a) C + a tr(C) / d I`. A weight a of 1 gives the mass-mean
    direction `m1 - m0`; smaller weights move toward Fisher's LDA. The
    solve goes through the thin SVD of the centered rows, so it stays cheap
    when d exceeds n. The scorer is linear, so the reader is a `RepReader`
    that can steer: its means are the row means and its signs follow the
    rule of `RepReader.fit` (see `orient`).

    Args:
        h: Activations of shape (K, n, d), K = `len(layers)`.
        layers: Layer index of each slice of `h`.
        labels: Binary labels of shape (n,), 1 for the positive pole.
        pairs: Row index pairs of shape (m, 2) that orient the layers.
        groups: Row groups that orient the layers, see `orient`.
        shrinkage: The weight a, in (0, 1]. None estimates it per layer
            with Ledoit-Wolf on the class-centered rows.

    Returns:
        The reader, named `lda`.

    Raises:
        ValueError: If the shapes disagree, the labels are not binary with
            both classes, `h` is not finite, the weight is outside (0, 1],
            a layer has no within-class variance, or the orientation inputs
            are malformed.
    """
    stack = _checked_stack(h, layers, labels)
    if shrinkage is not None:
        _check_shrinkage(shrinkage)
    positive = labels == 1
    vectors = np.stack(
        [_shrunk_mass_mean(rows, positive, shrinkage) for rows in stack]
    )
    reader = RepReader.from_arrays("lda", layers, vectors, stack.mean(axis=1))
    return orient(reader, h, labels=labels, pairs=pairs, groups=groups)


@dataclass(frozen=True, eq=False)
class MLPReader:
    """Multilayer perceptrons that score standardized activations.

    Each feature is centered and divided by its training standard deviation,
    as scikit-learn recommends for its MLP, so the penalty does not depend on
    the activation scale. A score is the logit of the positive class. The
    reader has no direction, so it cannot steer.

    Attributes:
        method: `mlp` for one MLP per layer, `mlp_joint` for one MLP on the
            layers concatenated feature-wise.
        layers: Layer indices, in the order of the first axis of the input.
        models: The fitted classifiers: one per layer, or one in total.
        centers: Training means of shape (K, d).
        scales: Training standard deviations of shape (K, d), 1 for a
            constant feature.
    """

    method: str
    layers: tuple[int, ...]
    models: tuple[MLPClassifier, ...]
    centers: NDArray[np.float64]
    scales: NDArray[np.float64]

    def score(self, h: FloatArray) -> NDArray[np.float32]:
        """Scores activations with the fitted perceptrons.

        Args:
            h: Activations of shape (K, n, d), K = `len(self.layers)`.

        Returns:
            Logits of shape (K, n) for `mlp` and (1, n) for `mlp_joint`.

        Raises:
            ValueError: If `h` does not have shape (K, n, d).
        """
        n_layers, width = self.centers.shape
        stack = np.asarray(h, dtype=np.float64)
        expected = (n_layers, width)
        if stack.ndim != 3 or (stack.shape[0], stack.shape[2]) != expected:
            raise ValueError(
                f"h must have shape ({n_layers}, n, {width}), got {stack.shape}"
            )
        inputs = (stack - self.centers[:, None]) / self.scales[:, None]
        if len(self.models) == n_layers:
            logits = [
                _logits(model, rows)
                for model, rows in zip(self.models, inputs, strict=True)
            ]
        else:
            logits = [_logits(self.models[0], np.concatenate(inputs, axis=1))]
        return np.stack(logits).astype(np.float32)


def fit_mlp(
    h: FloatArray,
    layers: Sequence[int],
    *,
    labels: NDArray[np.int64],
    hidden_sizes: Sequence[int],
    alpha: float,
    seed: int = 0,
    joint: bool = False,
    max_iter: int = 1000,
) -> MLPReader:
    """Fits one MLP per layer, or one MLP on the concatenated layers.

    The classifiers are scikit-learn's `MLPClassifier` with ReLU units and
    the L-BFGS solver, fitted to convergence of the penalized loss, so the
    hidden sizes and `alpha` are the only knobs to tune, as `c` is for the
    logistic finder. The model of layer k draws its initial weights from a
    seed derived from `seed` and k, as `RepReader.fit` does; the joint model
    draws them from `seed`.

    Args:
        h: Activations of shape (K, n, d), K = `len(layers)`.
        layers: Layer index of each slice of `h`.
        labels: Binary labels of shape (n,), 1 for the positive pole.
        hidden_sizes: Units of each hidden layer.
        alpha: L2 strength, scikit-learn's `alpha`: the penalty is
            alpha / (2 n) times the squared weights.
        seed: Base seed of the initial weights.
        joint: Fit one MLP on the K layers side by side, which scores one
            row, instead of one MLP per layer.
        max_iter: Iteration cap of the solver; scikit-learn warns when a
            fit reaches it.

    Returns:
        The reader, named `mlp`, or `mlp_joint` when `joint` is set.

    Raises:
        ValueError: If the shapes disagree, the labels are not binary with
            both classes, `h` is not finite, or scikit-learn rejects
            `hidden_sizes` or `alpha`.
    """
    stack = _checked_stack(h, layers, labels)
    centers = stack.mean(axis=1)
    deviations = stack.std(axis=1)
    scales = np.where(deviations > 0.0, deviations, 1.0)
    inputs = (stack - centers[:, None]) / scales[:, None]
    if joint:
        blocks = [np.concatenate(inputs, axis=1)]
        seeds = [seed]
    else:
        blocks = list(inputs)
        seeds = [_layer_seed(seed, k) for k in range(len(blocks))]
    models = tuple(
        MLPClassifier(
            hidden_layer_sizes=tuple(hidden_sizes),
            activation="relu",
            solver="lbfgs",
            alpha=alpha,
            max_iter=max_iter,
            random_state=block_seed,
        ).fit(block, labels)
        for block, block_seed in zip(blocks, seeds, strict=True)
    )
    return MLPReader(
        method="mlp_joint" if joint else "mlp",
        layers=tuple(int(layer) for layer in layers),
        models=models,
        centers=centers,
        scales=scales,
    )


def _checked_stack(
    h: FloatArray, layers: Sequence[int], labels: NDArray[np.int64]
) -> NDArray[np.float64]:
    stack = np.asarray(h, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"h must have shape (K, n, d), got {stack.shape}")
    if stack.shape[0] != len(layers):
        raise ValueError(
            f"layers has {len(layers)} entries for {stack.shape[0]} layers"
        )
    if labels.shape != (stack.shape[1],):
        raise ValueError(
            f"labels must have shape ({stack.shape[1]},), got {labels.shape}"
        )
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    if (labels == 1).all() or not (labels == 1).any():
        raise ValueError("labels must contain both classes")
    if not np.isfinite(stack).all():
        raise ValueError("h must be finite")
    return stack


def _check_shrinkage(shrinkage: float) -> None:
    if not 0.0 < shrinkage <= 1.0:
        raise ValueError(f"shrinkage must be in (0, 1], got {shrinkage}")


def _shrunk_mass_mean(
    rows: NDArray[np.float64],
    positive: NDArray[np.bool_],
    shrinkage: float | None,
) -> NDArray[np.float64]:
    positive_mean = rows[positive].mean(axis=0)
    negative_mean = rows[~positive].mean(axis=0)
    centered = rows - np.where(positive[:, None], positive_mean, negative_mean)
    if shrinkage is None:
        shrinkage = float(ledoit_wolf_shrinkage(centered, assume_centered=True))
        _check_shrinkage(shrinkage)
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    variances = singular**2 / rows.shape[0]
    ridge = shrinkage * variances.sum() / rows.shape[1]
    if ridge == 0.0:
        raise ValueError("the rows have no within-class variance")
    gap = positive_mean - negative_mean
    inside = basis @ gap
    shrunk = (1.0 - shrinkage) * variances + ridge
    # Off the row space of the centered rows, S is the ridge times identity.
    direction = basis.T @ (inside / shrunk) + (gap - basis.T @ inside) / ridge
    return np.asarray(direction, dtype=np.float64)


def _logits(
    model: MLPClassifier, rows: NDArray[np.float64]
) -> NDArray[np.float64]:
    # The forward pass stops before the sigmoid: probabilities saturate at
    # 0 and 1, and the resulting ties would blur a ranking.
    hidden = rows
    for weights, bias in zip(
        model.coefs_[:-1], model.intercepts_[:-1], strict=True
    ):
        hidden = np.maximum(hidden @ weights + bias, 0.0)
    output = hidden @ model.coefs_[-1] + model.intercepts_[-1]
    return np.asarray(output[:, 0], dtype=np.float64)
