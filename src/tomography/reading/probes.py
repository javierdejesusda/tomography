"""Readers beyond one oriented direction per layer.

`RepReader` holds one direction per layer. The readers here keep a
direction only where the scorer is linear: LDA and TTPD are `RepReader`s,
the attention reader exports its value heads as one, and the MLP reader
has none. All of them satisfy `Reader`.
"""

from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from typing import Protocol
from typing import runtime_checkable
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from tomography.activations import Activations
from tomography.reading.directions import FloatArray
from tomography.reading.directions import _check_labels
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
    positive = labels == 1
    vectors = np.stack(
        [_shrunk_mass_mean(rows, positive, shrinkage) for rows in stack]
    )
    reader = RepReader.from_arrays("lda", layers, vectors, stack.mean(axis=1))
    return orient(reader, h, labels=labels, pairs=pairs, groups=groups)


def fit_ttpd(
    h: FloatArray,
    layers: Sequence[int],
    *,
    labels: NDArray[np.int64],
    polarities: NDArray[np.int64] | None = None,
    pairs: NDArray[np.int64] | None = None,
    groups: NDArray[np.int64] | None = None,
) -> RepReader:
    """Fits TTPD per layer: a general truth direction plus polarity.

    The reference method (Training of Truth and Polarity Direction) models a
    centered activation as `t g + t p q`, with t = +1 for label 1 and -1
    for label 0, p = +1 for an affirmative and -1 for a negated statement.
    The general truth direction g is the least-squares solution of that
    model; a logistic regression without penalty on the rows finds the
    polarity direction, and a second one, on the projections onto both,
    gives the score. The rows are centered on their mean, the case of one
    topic. That score is linear in the activation, so the reader is the
    `RepReader` whose direction is the combination of g and the polarity
    direction, with the mean placed where the regression's decision is
    zero; a score is the decision divided by the norm of that direction.
    Without polarities the model keeps only `t g`, and a regression on one
    projection is monotone in it, so the reader is g on the centered rows.
    Either way the signs follow the rule of `RepReader.fit` (see `orient`).
    As in the reference code, a regression on separable rows stops at its
    iteration cap without a warning, at the direction reached there.

    Args:
        h: Activations of shape (K, n, d), K = `len(layers)`.
        layers: Layer index of each slice of `h`.
        labels: Binary labels of shape (n,), 1 for true statements.
        polarities: +1 (affirmative) or -1 (negated) per row, both present;
            None when every statement has the same polarity.
        pairs: Row index pairs of shape (m, 2) that orient the layers.
        groups: Row groups that orient the layers, see `orient`.

    Returns:
        The reader, named `ttpd`.

    Raises:
        ValueError: If the shapes disagree, the labels are not binary with
            both classes, `h` is not finite, the polarities are not one
            -1 or +1 per row with both present, or the orientation inputs
            are malformed.
    """
    stack = _checked_stack(h, layers, labels)
    truth = 2.0 * labels - 1.0
    if polarities is None:
        centers = stack.mean(axis=1)
        vectors = [
            np.linalg.lstsq(truth[:, None], rows - center, rcond=None)[0][0]
            for rows, center in zip(stack, centers, strict=True)
        ]
        reader = RepReader.from_arrays(
            "ttpd", layers, np.stack(vectors), centers
        )
        return orient(reader, h, labels=labels, pairs=pairs, groups=groups)
    _check_polarities(polarities, stack.shape[1])
    design = np.stack([truth, truth * polarities], axis=1)
    vectors = []
    means = []
    for rows in stack:
        center = rows.mean(axis=0)
        general = np.linalg.lstsq(design, rows - center, rcond=None)[0][0]
        polar = _unpenalized_logistic(rows, polarities == 1).coef_[0]
        projected = np.stack([rows @ general, rows @ polar], axis=1)
        final = _unpenalized_logistic(projected, labels == 1)
        direction = final.coef_[0] @ np.stack([general, polar])
        vectors.append(direction)
        means.append(
            -final.intercept_[0] * direction / np.square(direction).sum()
        )
    reader = RepReader.from_arrays(
        "ttpd", layers, np.stack(vectors), np.stack(means)
    )
    return orient(reader, h, labels=labels, pairs=pairs, groups=groups)


@dataclass(frozen=True, eq=False)
class MLPReader:
    """Multilayer perceptrons that score standardized activations.

    Each feature is centered and divided by its training standard deviation,
    as scikit-learn recommends for its MLP, so the penalty does not depend on
    the activation scale. A score is the logit of the positive class. The
    reader has no direction, so it cannot steer.

    Attributes:
        layers: Layer indices, in the order of the first axis of the input.
        joint: One MLP on the layers concatenated feature-wise instead of
            one MLP per layer.
        models: The fitted classifiers: one per layer, or one in total.
        centers: Training means of shape (K, d).
        scales: Training standard deviations of shape (K, d), 1 for a
            constant feature.
    """

    layers: tuple[int, ...]
    joint: bool
    models: tuple[MLPClassifier, ...]
    centers: NDArray[np.float64]
    scales: NDArray[np.float64]

    @property
    def method(self) -> str:
        """`mlp`, or `mlp_joint` for a joint reader."""
        return "mlp_joint" if self.joint else "mlp"

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
        stack = np.asarray(h)
        expected = (n_layers, width)
        if stack.ndim != 3 or (stack.shape[0], stack.shape[2]) != expected:
            raise ValueError(
                f"h must have shape ({n_layers}, n, {width}), got {stack.shape}"
            )
        blocks = _mlp_inputs(stack, self.centers, self.scales, self.joint)
        logits = [
            _logits(model, rows)
            for model, rows in zip(self.models, blocks, strict=True)
        ]
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
    blocks = _mlp_inputs(stack, centers, scales, joint)
    seeds = (
        [seed]
        if joint
        else [_layer_seed(seed, k) for k in range(stack.shape[0])]
    )
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
        layers=tuple(int(layer) for layer in layers),
        joint=joint,
        models=models,
        centers=centers,
        scales=scales,
    )


@dataclass(frozen=True, eq=False)
class AttentionReader:
    """Linear readers that pool the tokens of a stimulus with attention.

    A cell is one token row of a stimulus at one layer. A cell x of layer k
    has the attention logit `queries[k] @ x + offsets[k]` and the value
    `values[k] @ x + biases[k]`, and a stimulus scores the mean of its
    cells' values weighted by the softmax of their logits. The softmax runs
    over the tokens of each layer, which gives one score row per layer, or
    over the cells of all layers at once in a joint reader, which gives one
    row. With zero queries the weights are uniform over the tokens of each
    layer, so a per-layer score is the value head applied to the mean of
    the stimulus's tokens, and a joint reader with equal offsets averages
    those scores over its layers. The parameters are in activation units.

    Attributes:
        layers: Layer indices, in the order of the first axis of the input.
        joint: One softmax over the cells of all layers instead of one per
            layer.
        queries: Query vectors of shape (K, d).
        offsets: Logit offsets of shape (K,); they move weight between
            layers, so only a joint reader depends on them.
        values: Value heads of shape (K, d).
        biases: Value biases of shape (K,).
        signs: Orientation of each value head as a direction, shape (K,),
            each -1 or +1, see `rep_reader`.
    """

    layers: tuple[int, ...]
    joint: bool
    queries: NDArray[np.float64]
    offsets: NDArray[np.float64]
    values: NDArray[np.float64]
    biases: NDArray[np.float64]
    signs: NDArray[np.float32]

    @property
    def method(self) -> str:
        """`attention`, or `attention_joint` for a joint reader."""
        return "attention_joint" if self.joint else "attention"

    def score(self, acts: Activations) -> NDArray[np.float32]:
        """Scores every stimulus with its attention-weighted values.

        Args:
            acts: Token rows of the layers in `layers`, such as those of
                the `response` position.

        Returns:
            Scores of shape (K, n_stimuli), or (1, n_stimuli) for a joint
            reader.

        Raises:
            ValueError: If `acts` holds other layers or another width, or a
                stimulus has no rows.
        """
        if acts.layers != self.layers:
            raise ValueError(
                f"acts must hold the layers {self.layers}, got {acts.layers}"
            )
        width = self.queries.shape[1]
        if acts.data.shape[2] != width:
            raise ValueError(
                f"acts must have width {width}, got {acts.data.shape[2]}"
            )
        stimulus = torch.tensor(_row_stimulus(acts))
        logits = np.empty((len(self.layers), stimulus.shape[0]))
        heads = np.empty_like(logits)
        for k, layer_rows in enumerate(acts.data):
            rows = layer_rows.astype(np.float64)
            logits[k] = rows @ self.queries[k] + self.offsets[k]
            heads[k] = rows @ self.values[k] + self.biases[k]
        pooled = torch.stack(
            [
                _pool(
                    torch.from_numpy(logits[block]),
                    torch.from_numpy(heads[block]),
                    stimulus,
                    acts.n_stimuli,
                )
                for block in _blocks(len(self.layers), self.joint)
            ]
        )
        return pooled.numpy().astype(np.float32)

    def rep_reader(self) -> RepReader:
        """Exports the value heads as a reader that can steer.

        The direction of layer k is `values[k]`, and its mean is the point
        on that direction where the value is zero, so the raw score of a
        token is its value divided by the norm of the head. The signs are
        `signs`, which `fit_attention` sets with the rule of
        `RepReader.fit`.

        Returns:
            The reader, named like this one.

        Raises:
            ValueError: If a value head is zero.
        """
        squares = np.square(self.values).sum(axis=1, keepdims=True)
        if not squares.all():
            raise ValueError("a value head is zero, so it has no direction")
        means = -self.biases[:, None] * self.values / squares
        return RepReader.from_arrays(
            self.method, self.layers, self.values, means, self.signs
        )


def fit_attention(
    acts: Activations,
    *,
    labels: NDArray[np.int64],
    pairs: NDArray[np.int64] | None = None,
    groups: NDArray[np.int64] | None = None,
    c: float = 1.0,
    joint: bool = False,
    tol: float = 1e-4,
    max_iter: int = 1000,
) -> AttentionReader:
    """Fits attention-pooled linear readers on the token rows of stimuli.

    The loss is the logistic loss of the stimulus scores plus an L2 penalty
    on the queries and the value heads, weighted as in scikit-learn's
    `LogisticRegression`: `c` times the summed loss plus half the squared
    weights. As in the logistic finder, each layer's rows are first centered
    on their mean and divided by their root mean square norm, so `c` means
    the same in both, and with one token per stimulus the value head is
    that finder's regression. The queries and value heads start at zero, so
    the fit starts from mean pooling, and full-batch L-BFGS minimizes the
    loss in float64 on the CPU with scikit-learn's settings. The fit draws
    no random numbers, so it needs no seed. A joint reader shares one value
    bias across its layers and holds the rows of all of them in float64. The
    value heads are oriented over the token rows the way the sweep orients a
    `RepReader.fit` reader of `response` rows: the labels repeat on every
    row of a stimulus, and `Activations.row_groups` maps the pairs and the
    groups to rows.

    Args:
        acts: Token rows of the layers to read, such as those of the
            `response` position.
        labels: Binary labels of shape (n_stimuli,), 1 for the positive
            pole.
        pairs: Stimulus pairs of shape (m, 2) that orient the value heads.
        groups: Stimulus groups that orient the value heads, see `orient`.
        c: The inverse L2 strength, scikit-learn's `C`: smaller values
            regularize more.
        joint: Fit one reader over the cells of all layers, which scores
            one row, instead of one reader per layer.
        tol: Gradient tolerance of L-BFGS, scikit-learn's `tol`.
        max_iter: Iteration cap of L-BFGS; a fit that stops before
            converging warns with `ConvergenceWarning`.

    Returns:
        The reader.

    Raises:
        ValueError: If the labels do not have one binary entry per stimulus
            with both classes, `c` is not positive, a stimulus has no rows,
            `acts` is not finite, a layer's rows are all equal, a value head
            is zero, or the orientation inputs are malformed (members of a
            group need equal row counts).
    """
    _check_classes(labels, acts.n_stimuli)
    if not c > 0.0:
        raise ValueError(f"c must be positive, got {c}")
    stimulus = torch.tensor(_row_stimulus(acts))
    targets = torch.tensor(labels, dtype=torch.float64)
    n_layers, _, width = acts.data.shape
    queries = np.empty((n_layers, width))
    values = np.empty((n_layers, width))
    biases = np.empty(n_layers)
    offsets = np.empty(n_layers)
    for block in _blocks(n_layers, joint):
        queries[block], offsets[block], values[block], biases[block] = (
            _fit_block(
                acts.data[block],
                acts.layers[block],
                stimulus,
                targets,
                c,
                tol,
                max_iter,
            )
        )
    reader = AttentionReader(
        layers=tuple(int(layer) for layer in acts.layers),
        joint=joint,
        queries=queries,
        offsets=offsets,
        values=values,
        biases=biases,
        signs=np.ones(n_layers, dtype=np.float32),
    )
    oriented = orient(
        reader.rep_reader(),
        acts.data,
        labels=acts.row_values(labels),
        pairs=None if pairs is None else acts.row_groups(pairs),
        groups=None if groups is None else acts.row_groups(groups),
    )
    return replace(reader, signs=oriented.signs)


def _check_classes(labels: NDArray[np.int64], n: int) -> None:
    _check_labels(labels, n)
    if (labels == 1).all() or not (labels == 1).any():
        raise ValueError("labels must contain both classes")


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
    _check_classes(labels, stack.shape[1])
    if not np.isfinite(stack).all():
        raise ValueError("h must be finite")
    return stack


def _check_polarities(polarities: NDArray[np.int64], n: int) -> None:
    values = np.asarray(polarities)
    if values.shape != (n,):
        raise ValueError(
            f"expected one polarity per row ({n}), got shape {values.shape}"
        )
    if not np.isin(values, (-1, 1)).all():
        raise ValueError("polarities must be -1 or +1")
    if (values == 1).all() or (values == -1).all():
        raise ValueError("polarities must contain both polarities")


def _unpenalized_logistic(
    rows: NDArray[np.float64], positive: NDArray[np.bool_]
) -> LogisticRegression:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return LogisticRegression(C=np.inf).fit(rows, positive)


def _shrunk_mass_mean(
    rows: NDArray[np.float64],
    positive: NDArray[np.bool_],
    shrinkage: float | None,
) -> NDArray[np.float64]:
    positive_mean = rows[positive].mean(axis=0)
    negative_mean = rows[~positive].mean(axis=0)
    centered = rows - np.where(positive[:, None], positive_mean, negative_mean)
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    variances = singular**2 / rows.shape[0]
    if not variances.any():
        raise ValueError("the rows have no within-class variance")
    if shrinkage is None:
        shrinkage = float(ledoit_wolf_shrinkage(centered, assume_centered=True))
    if not 0.0 < shrinkage <= 1.0:
        raise ValueError(f"shrinkage must be in (0, 1], got {shrinkage}")
    ridge = shrinkage * variances.sum() / rows.shape[1]
    gap = positive_mean - negative_mean
    inside = basis @ gap
    shrunk = (1.0 - shrinkage) * variances + ridge
    # Off the row space of the centered rows, S is the ridge times identity.
    direction = basis.T @ (inside / shrunk) + (gap - basis.T @ inside) / ridge
    return np.asarray(direction, dtype=np.float64)


def _mlp_inputs(
    stack: FloatArray,
    centers: NDArray[np.float64],
    scales: NDArray[np.float64],
    joint: bool,
) -> Iterable[NDArray[np.float64]]:
    """Standardizes each layer in float64, one layer at a time.

    Returns one block per layer, made when it is read, so a single layer is
    copied at a time; or one block of all layers side by side when joint.
    """
    blocks = (
        _standardized(layer, center, scale)
        for layer, center, scale in zip(stack, centers, scales, strict=True)
    )
    return [np.concatenate(list(blocks), axis=1)] if joint else blocks


def _standardized(
    layer: FloatArray,
    center: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    rows = np.array(layer, dtype=np.float64)
    rows -= center
    rows /= scale
    return rows


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


def _row_stimulus(acts: Activations) -> NDArray[np.int64]:
    counts = np.bincount(acts.row_stimulus, minlength=acts.n_stimuli)
    if not counts.all():
        raise ValueError(f"stimulus {int(np.argmin(counts))} has no rows")
    return np.asarray(acts.row_stimulus, dtype=np.int64)


def _blocks(n_layers: int, joint: bool) -> list[slice]:
    # The layers that share one softmax: all of them, or each one alone.
    if joint:
        return [slice(0, n_layers)]
    return [slice(k, k + 1) for k in range(n_layers)]


def _pool(
    logits: torch.Tensor,
    values: torch.Tensor,
    stimulus: torch.Tensor,
    n_stimuli: int,
) -> torch.Tensor:
    # Pools the cells of shape (layers, rows) into one score per stimulus.
    cells = stimulus.repeat(logits.shape[0])
    logits = logits.reshape(-1)
    values = values.reshape(-1)
    # Subtracting each stimulus's largest logit leaves its softmax unchanged
    # and keeps the exponentials from overflowing.
    peaks = logits.new_full((n_stimuli,), -torch.inf).scatter_reduce(
        0, cells, logits.detach(), "amax"
    )
    weights = torch.exp(logits - peaks[cells])
    total = logits.new_zeros(n_stimuli).scatter_add(0, cells, weights)
    pooled = logits.new_zeros(n_stimuli).scatter_add(0, cells, weights * values)
    return pooled / total


def _fit_block(
    data: NDArray[np.float32],
    layers: tuple[int, ...],
    stimulus: torch.Tensor,
    targets: torch.Tensor,
    c: float,
    tol: float,
    max_iter: int,
) -> tuple[NDArray[np.float64], ...]:
    # Fits the layers of one softmax on scaled rows and returns their
    # queries, offsets, value heads and biases in activation units.
    rows = data.astype(np.float64)
    if not np.isfinite(rows).all():
        raise ValueError("acts must be finite")
    centers = rows.mean(axis=1)
    rows -= centers[:, None]
    squares = np.array([np.square(layer).sum() for layer in rows])
    scales = np.sqrt(squares / rows.shape[1])
    if not scales.all():
        layer = layers[int(np.argmin(scales))]
        raise ValueError(f"the rows of layer {layer} have no variance")
    rows /= scales[:, None, None]
    scaled = torch.from_numpy(rows)
    n_layers, _, width = rows.shape
    size = n_layers * width
    strength = 1.0 / (c * targets.shape[0])

    def split(
        params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        queries = params[:size].view(n_layers, width)
        values = params[size:-1].view(n_layers, width)
        return queries, values, params[-1]

    def objective(
        theta: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        params = torch.tensor(theta, requires_grad=True)
        queries, values, bias = split(params)
        logits = torch.einsum("krd,kd->kr", scaled, queries)
        heads = torch.einsum("krd,kd->kr", scaled, values) + bias
        scores = _pool(logits, heads, stimulus, targets.shape[0])
        penalty = queries.square().sum() + values.square().sum()
        loss = binary_cross_entropy_with_logits(scores, targets)
        loss = loss + 0.5 * strength * penalty
        (gradient,) = torch.autograd.grad(loss, params)
        return float(loss.detach()), gradient.numpy()

    result = minimize(
        objective,
        np.zeros(2 * size + 1),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": max_iter,
            "maxls": 50,
            "gtol": tol,
            "ftol": 64 * np.finfo(np.float64).eps,
        },
    )
    if result.status != 0:
        warnings.warn(
            f"the attention reader did not converge: {result.message}",
            ConvergenceWarning,
            stacklevel=3,
        )
    fitted = split(torch.tensor(result.x, dtype=torch.float64))
    queries = fitted[0].numpy() / scales[:, None]
    values = fitted[1].numpy() / scales[:, None]
    offsets = -(queries * centers).sum(axis=1)
    biases = float(fitted[2]) - (values * centers).sum(axis=1)
    return queries, offsets, values, biases
