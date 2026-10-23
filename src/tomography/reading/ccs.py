"""Contrast-consistent search (CCS), an unsupervised probe (Burns et al., 2023).

CCS reads contrast pairs: one item with two opposite answers, such as a
statement and its negation. Each of the two sets is centred on its own
mean, and a logistic probe is trained so that the two members get
complementary probabilities (consistency) away from one half (confidence).
The probe has a bias and two set means, which a `RepReader` cannot hold, so
`fit_ccs` fits it outside `fit_direction` and wraps its direction with
`RepReader.from_arrays`.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import torch

from tomography.reading.directions import FloatArray
from tomography.reading.directions import _check_labels
from tomography.reading.reader import RepReader
from tomography.reading.reader import _layer_seed
from tomography.reading.reader import orient


@dataclass(frozen=True, eq=False)
class CCSProbe:
    """A fitted CCS probe per layer, oriented on the training labels.

    The probability that row x of a set is true is
    `sigmoid(weights[k] . (x - set_mean[k]) + biases[k])`, with the set
    mean of the first or second pair members.

    Score CCS accuracy with `pair_probabilities`, not with `reader`: the
    reader drops the per-set centring, so when the two sets have different
    means (a statement set and a negation set) its pair comparisons are
    shifted by that offset and do not match the probe's predictions.

    Attributes:
        reader: The direction of `weights` per layer, centred on the mean
            of both set means, named `ccs`. Its scores compare rows on one
            linear scale without the bias and the per-set centring; use it
            to steer or to score rows of sets that share a mean.
        weights: Probe weights in activation units, shape (K, d).
        biases: Probe biases, shape (K,).
        first_means: Training mean of the first pair members, (K, d).
        second_means: Training mean of the second pair members, (K, d).
        losses: The CCS loss of the restart kept per layer, shape (K,).
    """

    reader: RepReader
    weights: NDArray[np.float64]
    biases: NDArray[np.float64]
    first_means: NDArray[np.float64]
    second_means: NDArray[np.float64]
    losses: NDArray[np.float64]

    def pair_probabilities(
        self, h: FloatArray, pairs: NDArray[np.int64]
    ) -> NDArray[np.float32]:
        """The CCS estimate that the first member of each pair is true.

        It is `0.5 * (p(x+) + 1 - p(x-))`, each member centred on the
        training mean of its set, so one pair is scored on its own.

        Args:
            h: Activations of shape (K, n, d), K = `len(self.reader.layers)`.
            pairs: Row index pairs of shape (m, 2), first member first.

        Returns:
            Probabilities of shape (K, m).

        Raises:
            ValueError: If `h` or `pairs` is malformed.
        """
        stack = _checked_activations(h, len(self.reader.layers))
        _check_pairs(pairs, stack.shape[1], minimum=1)
        if stack.shape[2] != self.weights.shape[1]:
            raise ValueError(
                f"h must have width {self.weights.shape[1]}, "
                f"got {stack.shape[2]}"
            )
        first = _probabilities(
            stack[:, pairs[:, 0]], self.first_means, self.weights, self.biases
        )
        second = _probabilities(
            stack[:, pairs[:, 1]], self.second_means, self.weights, self.biases
        )
        return (0.5 * (first + 1.0 - second)).astype(np.float32)


def fit_ccs(
    h: FloatArray,
    layers: Sequence[int],
    *,
    pairs: NDArray[np.int64],
    labels: NDArray[np.int64],
    restarts: int = 10,
    epochs: int = 1000,
    lr: float = 0.01,
    seed: int = 0,
) -> CCSProbe:
    """Fits a CCS probe per layer and orients it on the training labels.

    Each set of pair members is centred on its own training mean, and both
    are divided by one scalar, the mean norm of the centred rows (the paper
    divides each set by its own; a shared scalar keeps one linear map for
    both sets). The probe starts from a random unit weight vector and a
    zero bias and is trained full batch with AdamW (the paper's learning
    rate 0.01 by default, weight decay 0.01 as in its code) on the loss
    `mean((p(x+) - (1 - p(x-)))^2 + min(p(x+), p(x-))^2)`; of `restarts`
    runs the one with the lowest final loss is kept. The loss does not fix
    the sign, so the labels orient the reader with the rule of
    `RepReader.fit` on the pairs, as for every finder, scoring each set
    centred on its own mean, so the sign makes the probe's own prediction
    agree with the labels on most training pairs; the probe flips with the
    reader. The labels never enter the fit.

    Args:
        h: Activations of shape (K, n, d), K = `len(layers)`.
        layers: Layer index of each slice of `h`.
        pairs: Row index pairs of shape (m, 2): the first member, such as
            a statement, then its contrast, such as the negation.
        labels: Binary labels of shape (n,), 1 for a true row.
        restarts: Number of random initialisations per layer.
        epochs: Full-batch optimiser steps per initialisation.
        lr: AdamW learning rate.
        seed: Base seed; layer k and restart r draw from (seed, k, r).

    Returns:
        The oriented probe. Score its accuracy with
        `CCSProbe.pair_probabilities`, not with its reader.

    Raises:
        ValueError: If the inputs are malformed, a hyperparameter is not
            positive, a row is both a first and a second member, or a
            layer has no variance within the sets.
    """
    if restarts < 1:
        raise ValueError(f"restarts must be at least 1, got {restarts}")
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    if not lr > 0.0:
        raise ValueError(f"lr must be positive, got {lr}")
    stack = _checked_activations(h, len(layers))
    _check_pairs(pairs, stack.shape[1], minimum=2)
    if np.intersect1d(pairs[:, 0], pairs[:, 1]).size:
        raise ValueError("a row cannot be both a first and a second member")
    _check_labels(labels, stack.shape[1])
    first_rows = stack[:, pairs[:, 0]]
    second_rows = stack[:, pairs[:, 1]]
    first_means = first_rows.mean(axis=1)
    second_means = second_rows.mean(axis=1)
    fits = [
        _fit_layer(
            first_rows[k] - first_means[k],
            second_rows[k] - second_means[k],
            restarts=restarts,
            epochs=epochs,
            lr=lr,
            seed=_layer_seed(seed, k),
        )
        for k in range(stack.shape[0])
    ]
    weights = np.stack([fit[0] for fit in fits])
    biases = np.array([fit[1] for fit in fits])
    centers = 0.5 * (first_means + second_means)
    unsigned = RepReader.from_arrays("ccs", layers, weights, centers)
    # Shifting each set onto the common centre makes the reader's pair
    # comparison the probe's own, so the orientation reads CCS predictions.
    shifted = stack.copy()
    shifted[:, pairs[:, 0]] -= (first_means - centers)[:, None]
    shifted[:, pairs[:, 1]] -= (second_means - centers)[:, None]
    reader = orient(unsigned, shifted, labels=labels, pairs=pairs)
    signs = reader.signs.astype(np.float64)
    return CCSProbe(
        reader=reader,
        weights=signs[:, None] * weights,
        biases=signs * biases,
        first_means=first_means,
        second_means=second_means,
        losses=np.array([fit[2] for fit in fits]),
    )


def _fit_layer(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    *,
    restarts: int,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[NDArray[np.float64], float, float]:
    scale = float(
        np.linalg.norm(np.concatenate([first, second]), axis=1).mean()
    )
    if scale == 0.0:
        raise ValueError("the pair members have no variance within the sets")
    first_t = torch.from_numpy(first / scale)
    second_t = torch.from_numpy(second / scale)
    best: tuple[NDArray[np.float64], float, float] | None = None
    for restart in range(restarts):
        generator = torch.Generator().manual_seed(
            int(np.random.SeedSequence([seed, restart]).generate_state(1)[0])
        )
        weight = torch.randn(
            first.shape[1], dtype=torch.float64, generator=generator
        )
        weight = (weight / weight.norm()).requires_grad_()
        bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.AdamW([weight, bias], lr=lr, weight_decay=0.01)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = _ccs_loss(first_t, second_t, weight, bias)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        with torch.no_grad():
            final = float(_ccs_loss(first_t, second_t, weight, bias))
        if best is None or final < best[2]:
            vector = weight.detach().numpy() / scale
            best = (vector, float(bias.detach()[0]), final)
    assert best is not None
    return best


def _ccs_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    p_first = torch.sigmoid(first @ weight + bias)
    p_second = torch.sigmoid(second @ weight + bias)
    consistency = (p_first - (1.0 - p_second)) ** 2
    confidence = torch.minimum(p_first, p_second) ** 2
    return (consistency + confidence).mean()


def _probabilities(
    rows: NDArray[np.float64],
    means: NDArray[np.float64],
    weights: NDArray[np.float64],
    biases: NDArray[np.float64],
) -> NDArray[np.float64]:
    logits = np.einsum("knd,kd->kn", rows - means[:, None], weights)
    return np.asarray(1.0 / (1.0 + np.exp(-(logits + biases[:, None]))))


def _checked_activations(h: FloatArray, n_layers: int) -> NDArray[np.float64]:
    stack = np.asarray(h, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"h must have shape (K, n, d), got {stack.shape}")
    if stack.shape[0] != n_layers:
        raise ValueError(
            f"layers has {n_layers} entries for {stack.shape[0]} layers"
        )
    if not np.isfinite(stack).all():
        raise ValueError("h must be finite")
    return stack


def _check_pairs(
    pairs: NDArray[np.int64], n_rows: int, *, minimum: int
) -> None:
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must have shape (m, 2), got {pairs.shape}")
    if pairs.shape[0] < minimum:
        raise ValueError(f"CCS needs at least {minimum} pairs")
    if pairs.min() < 0 or pairs.max() >= n_rows:
        raise ValueError("pairs must index the rows of h")
