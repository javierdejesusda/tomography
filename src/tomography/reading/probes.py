"""Readers beyond one oriented direction per layer.

`RepReader` holds one direction per layer. The readers here keep a
direction only where the scorer is linear: LDA and TTPD are `RepReader`s,
the attention and cell max readers export their value heads as one, and
the MLP reader and the cell readers with an MLP head per layer have none.
All of them satisfy `Reader`.
"""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from functools import partial
from typing import Protocol
from typing import runtime_checkable
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult
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

# L-BFGS-B's `ftol`: it stops when a step lowers the loss by no more than
# this times the loss's magnitude (at least 1).
_FTOL = 64 * np.finfo(np.float64).eps
_NO_DIRECTION = (
    "an MLP head has no single direction per layer, so it cannot be "
    "exported as a RepReader; cell_values gives the value of every cell"
)


class StalledFitWarning(UserWarning):
    """A cell fit whose loss never dropped below its loss at the zero start.

    The fit may still have converged by L-BFGS's own tests, so this is not
    a `ConvergenceWarning`. Its heads stay near zero, so its reader scores
    on a tiny scale: the scores seldom tie, but they carry no fitted signal.
    """


class RestartedFitWarning(UserWarning):
    """A cell max fit that stalled at its zero start and was fit again.

    The second fit starts from the mean pooling solution, split into one
    term per layer, and left the zero start's loss behind, so the reader is
    that fit.
    """


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
        stimulus = _scored_stimulus(acts, self.layers, self.queries.shape[1])
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
        return _value_heads(
            self.method, self.layers, self.values, self.biases, self.signs
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
    groups to rows. An L-BFGS run (one per layer, or one for a joint
    reader) whose loss never drops below its value at the zero start, by
    more than L-BFGS's relative `ftol` of 64 machine epsilons, warns with
    `StalledFitWarning` and keeps the parameters it reached.

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
    stimulus, targets = _fit_inputs(acts, labels, c)
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
    signs = _head_signs(reader.rep_reader(), acts, labels, pairs, groups)
    return replace(reader, signs=signs)


@dataclass(frozen=True, eq=False)
class CellMaxReader:
    """A linear reader that scores a stimulus by its highest cell.

    A cell is one token row of a stimulus at one layer. A cell x of layer k
    has the value `values[k] @ x + biases[k]`, and a stimulus scores the
    largest value among its cells of all layers, which gives one score row.
    The maximum ignores the order of the cells, so stimuli of any length
    need no padding or mask, but it rises with their number: stimuli with
    more cells tend to score higher. The parameters are in activation
    units.

    Attributes:
        layers: Layer indices, in the order of the first axis of the input.
        values: Value heads of shape (K, d).
        biases: Value biases of shape (K,), one per layer, which move the
            layers' cells against each other.
        signs: Orientation of each value head as a direction, shape (K,),
            each -1 or +1, see `rep_reader`.
    """

    layers: tuple[int, ...]
    values: NDArray[np.float64]
    biases: NDArray[np.float64]
    signs: NDArray[np.float32]

    @property
    def method(self) -> str:
        """`cellmax`."""
        return "cellmax"

    def score(self, acts: Activations) -> NDArray[np.float32]:
        """Scores every stimulus with the highest value among its cells.

        Args:
            acts: Token rows of the layers in `layers`, such as those of
                the `response` position.

        Returns:
            Scores of shape (1, n_stimuli).

        Raises:
            ValueError: If `acts` holds other layers or another width, or a
                stimulus has no rows.
        """
        stimulus = _scored_stimulus(acts, self.layers, self.values.shape[1])
        heads = np.empty((len(self.layers), stimulus.shape[0]))
        for k, layer_rows in enumerate(acts.data):
            rows = layer_rows.astype(np.float64)
            heads[k] = rows @ self.values[k] + self.biases[k]
        pooled = _max_pool(torch.from_numpy(heads), stimulus, acts.n_stimuli)
        return pooled[None].numpy().astype(np.float32)

    def rep_reader(self) -> RepReader:
        """Exports the value heads as a reader that can steer.

        As in `AttentionReader.rep_reader`, the direction of layer k is
        `values[k]` and its mean is the point on that direction where the
        value is zero; the signs are `signs`, which `fit_cell_max` sets
        with the rule of `RepReader.fit`.

        Returns:
            The reader, named `cellmax`.

        Raises:
            ValueError: If a value head is zero.
        """
        return _value_heads(
            self.method, self.layers, self.values, self.biases, self.signs
        )


def fit_cell_max(
    acts: Activations,
    *,
    labels: NDArray[np.int64],
    pairs: NDArray[np.int64] | None = None,
    groups: NDArray[np.int64] | None = None,
    c: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000,
) -> CellMaxReader:
    """Fits a linear reader through the maximum over the cells of stimuli.

    The loss is the logistic loss of the stimulus scores, each the highest
    value among the stimulus's cells of all layers, plus an L2 penalty on
    the value heads weighted as in `fit_attention`, so `c` means
    scikit-learn's `C`; the biases are not penalized. The rows are scaled as
    in `fit_attention`, on the training rows only, so with one cell per
    stimulus the reader is the logistic finder's regression. The value
    heads and biases start at zero, where every cell of a stimulus ties and
    the gradient splits evenly among them, so the first step is the one of
    mean pooling; L-BFGS then runs with the settings of `fit_attention` and
    draws no random numbers. The maximum makes the loss piecewise smooth,
    so a fit can stop on a line search: it keeps the parameters reached and
    warns. Along the mean pooling step the maximum can rise in the wrong
    class, for instance when one template cell tops every stimulus, and
    then the loss rises too: the line search shrinks the step to rounding
    size and the fit stalls at its zero start, which need not be a
    minimum. Such a fit runs again, once, from the mean pooling
    solution: the penalized logistic fit of the scores at the zero start,
    each stimulus's mean value over its cells of all layers, split into one
    term per layer (its heads and biases over the number of layers), so a
    cell takes the value its layer adds to the mean. When the
    second run leaves the zero start's loss behind, it is kept and the fit
    warns with `RestartedFitWarning`; a fit that does not stall runs once,
    from zero. When the second run stalls too, the fit keeps the zero
    start's heads, which score on a tiny scale with no fitted signal, and
    warns as in `fit_attention`. The value heads are oriented over the
    token rows as in `fit_attention`.

    Args:
        acts: Token rows of the layers to read, such as those of the
            `response` position.
        labels: Binary labels of shape (n_stimuli,), 1 for the positive
            pole.
        pairs: Stimulus pairs of shape (m, 2) that orient the value heads.
        groups: Stimulus groups that orient the value heads, see `orient`.
        c: The inverse L2 strength, scikit-learn's `C`: smaller values
            regularize more.
        tol: Gradient tolerance of L-BFGS, scikit-learn's `tol`.
        max_iter: Iteration cap of each L-BFGS run; a fit whose kept run
            stops before converging warns with `ConvergenceWarning`.

    Returns:
        The reader.

    Raises:
        ValueError: If the labels do not have one binary entry per stimulus
            with both classes, `c` is not positive, a stimulus has no rows,
            `acts` is not finite, a layer's rows are all equal, a value head
            is zero, or the orientation inputs are malformed (members of a
            group need equal row counts).
    """
    stimulus, targets = _fit_inputs(acts, labels, c)
    values, biases = _fit_max(
        acts.data, acts.layers, stimulus, targets, c, tol, max_iter
    )
    reader = CellMaxReader(
        layers=tuple(int(layer) for layer in acts.layers),
        values=values,
        biases=biases,
        signs=np.ones(len(acts.layers), dtype=np.float32),
    )
    signs = _head_signs(reader.rep_reader(), acts, labels, pairs, groups)
    return replace(reader, signs=signs)


@dataclass(frozen=True, eq=False)
class AttentionMLPReader:
    """A joint attention reader whose cell values come from an MLP per layer.

    A cell x of layer k has the attention logit `queries[k] @ x +
    offsets[k]`, as in a joint `AttentionReader`, and the value
    `outputs[k] @ relu(hidden[k] @ x + hidden_biases[k]) + biases[k]`: one
    hidden layer of ReLU units per layer. A stimulus scores the values of
    its cells of all layers weighted by one softmax of their logits, which
    gives one score row. The parameters are in activation units.

    Attributes:
        layers: Layer indices, in the order of the first axis of the input.
        queries: Query vectors of shape (K, d).
        offsets: Logit offsets of shape (K,).
        hidden: Hidden weights of shape (K, H, d), H units per layer.
        hidden_biases: Hidden biases of shape (K, H).
        outputs: Output weights of shape (K, H).
        biases: Output biases of shape (K,); `fit_attention_mlp` shares one
            across the layers.
    """

    layers: tuple[int, ...]
    queries: NDArray[np.float64]
    offsets: NDArray[np.float64]
    hidden: NDArray[np.float64]
    hidden_biases: NDArray[np.float64]
    outputs: NDArray[np.float64]
    biases: NDArray[np.float64]

    @property
    def method(self) -> str:
        """`attention_joint_mlp`."""
        return "attention_joint_mlp"

    def cell_values(self, acts: Activations) -> NDArray[np.float64]:
        """The value of every cell, in float64.

        Args:
            acts: Token rows of the layers in `layers`.

        Returns:
            The values, shape (K, n_rows), in the layout of `acts.data`.

        Raises:
            ValueError: If `acts` holds other layers or another width, or a
                stimulus has no rows.
        """
        _scored_stimulus(acts, self.layers, self.hidden.shape[2])
        return _mlp_values(acts, self)

    def score(self, acts: Activations) -> NDArray[np.float32]:
        """Scores every stimulus with its attention-weighted cell values.

        Args:
            acts: Token rows of the layers in `layers`, such as those of
                the `response` position.

        Returns:
            Scores of shape (1, n_stimuli).

        Raises:
            ValueError: If `acts` holds other layers or another width, or a
                stimulus has no rows.
        """
        stimulus = _scored_stimulus(acts, self.layers, self.queries.shape[1])
        logits = np.empty((len(self.layers), stimulus.shape[0]))
        for k, layer_rows in enumerate(acts.data):
            rows = layer_rows.astype(np.float64)
            logits[k] = rows @ self.queries[k] + self.offsets[k]
        pooled = _pool(
            torch.from_numpy(logits),
            torch.from_numpy(_mlp_values(acts, self)),
            stimulus,
            acts.n_stimuli,
        )
        return pooled[None].numpy().astype(np.float32)

    def rep_reader(self) -> RepReader:
        """Refuses the export of `AttentionReader.rep_reader`.

        Raises:
            ValueError: Always: an MLP head has no single direction per
                layer; `cell_values` gives the value of every cell.
        """
        raise ValueError(_NO_DIRECTION)


@dataclass(frozen=True, eq=False)
class CellMaxMLPReader:
    """A reader that scores a stimulus by its highest cell under an MLP.

    A cell x of layer k has the value `outputs[k] @ relu(hidden[k] @ x +
    hidden_biases[k]) + biases[k]`, one hidden layer of ReLU units per
    layer, and a stimulus scores the largest value among its cells of all
    layers, as in `CellMaxReader`. The parameters are in activation units.

    Attributes:
        layers: Layer indices, in the order of the first axis of the input.
        hidden: Hidden weights of shape (K, H, d), H units per layer.
        hidden_biases: Hidden biases of shape (K, H).
        outputs: Output weights of shape (K, H).
        biases: Output biases of shape (K,), one per layer.
    """

    layers: tuple[int, ...]
    hidden: NDArray[np.float64]
    hidden_biases: NDArray[np.float64]
    outputs: NDArray[np.float64]
    biases: NDArray[np.float64]

    @property
    def method(self) -> str:
        """`cellmax_mlp`."""
        return "cellmax_mlp"

    def cell_values(self, acts: Activations) -> NDArray[np.float64]:
        """The value of every cell, in float64; `score` takes their maximum.

        Args:
            acts: Token rows of the layers in `layers`.

        Returns:
            The values, shape (K, n_rows), in the layout of `acts.data`.

        Raises:
            ValueError: If `acts` holds other layers or another width, or a
                stimulus has no rows.
        """
        _scored_stimulus(acts, self.layers, self.hidden.shape[2])
        return _mlp_values(acts, self)

    def score(self, acts: Activations) -> NDArray[np.float32]:
        """Scores every stimulus with the highest value among its cells.

        Args:
            acts: Token rows of the layers in `layers`, such as those of
                the `response` position.

        Returns:
            Scores of shape (1, n_stimuli).

        Raises:
            ValueError: If `acts` holds other layers or another width, or a
                stimulus has no rows.
        """
        stimulus = _scored_stimulus(acts, self.layers, self.hidden.shape[2])
        values = torch.from_numpy(_mlp_values(acts, self))
        pooled = _max_pool(values, stimulus, acts.n_stimuli)
        return pooled[None].numpy().astype(np.float32)

    def rep_reader(self) -> RepReader:
        """Refuses the export of `CellMaxReader.rep_reader`.

        Raises:
            ValueError: Always: an MLP head has no single direction per
                layer; `cell_values` gives the value of every cell.
        """
        raise ValueError(_NO_DIRECTION)


def fit_attention_mlp(
    acts: Activations,
    *,
    labels: NDArray[np.int64],
    hidden: int,
    seed: int,
    c: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000,
) -> AttentionMLPReader:
    """Fits a joint attention reader with an MLP value head per layer.

    The reader of `fit_attention(joint=True)` with each layer's value head
    replaced by `hidden` ReLU units and an output weight vector; the
    attention logits stay linear. The loss is the logistic loss of the
    stimulus scores plus an L2 penalty on the queries, the hidden weights
    and the output weights, weighted as in `fit_attention`, so `c` means
    scikit-learn's `C`; the biases are not penalized. The rows are scaled
    as in `fit_attention`, on the training rows only, and L-BFGS runs with
    its settings.

    The loss is not convex, and its zero point is a saddle where every unit
    is the same, so the fit starts from seeded hidden weights: a standard
    normal draw of `np.random.default_rng(seed)` divided by the square root
    of the width, so each unit's weights have a norm near 1 against the
    scaled rows, whose root mean square norm is 1. Every other parameter
    starts at zero, so the start scores every stimulus 0 and the attention
    starts from mean pooling. The draw touches no global random state, and
    two fits with the same seed are identical on one machine. A random start
    was chosen over a warm start from the linear head, which would cost a
    linear fit first and would still need random units beside it, since
    units with zero output weights get no gradient on their hidden weights.

    The stall test compares the final loss with the loss of the all-zero
    reader, log 2, which the start does not reach because of its penalty:
    a fit that never beats it warns with `StalledFitWarning`. Under strong
    regularization the head shrinks to zero this way, since the penalty on
    two layers of weights makes zero a local minimum for small `c`.

    Args:
        acts: Token rows of the layers to read, such as those of the
            `response` position.
        labels: Binary labels of shape (n_stimuli,), 1 for the positive
            pole.
        hidden: Hidden units per layer, at least 1.
        seed: Seed of the start.
        c: The inverse L2 strength, scikit-learn's `C`: smaller values
            regularize more.
        tol: Gradient tolerance of L-BFGS, scikit-learn's `tol`.
        max_iter: Iteration cap of L-BFGS; a fit that stops before
            converging warns with `ConvergenceWarning`.

    Returns:
        The reader. It has no direction to export, see
        `AttentionMLPReader.rep_reader`.

    Raises:
        ValueError: If the labels do not have one binary entry per stimulus
            with both classes, `c` is not positive, `hidden` is below 1, a
            stimulus has no rows, `acts` is not finite, or a layer's rows
            are all equal.
    """
    stimulus, targets = _fit_inputs(acts, labels, c)
    _check_hidden(hidden, 1)
    queries, offsets, weights, unit_biases, outputs, biases = (
        _fit_attention_mlp(
            acts.data,
            acts.layers,
            stimulus,
            targets,
            c,
            tol,
            max_iter,
            _hidden_start(acts.data.shape, hidden, seed),
        )
    )
    return AttentionMLPReader(
        layers=tuple(int(layer) for layer in acts.layers),
        queries=queries,
        offsets=offsets,
        hidden=weights,
        hidden_biases=unit_biases,
        outputs=outputs,
        biases=biases,
    )


def fit_cell_max_mlp(
    acts: Activations,
    *,
    labels: NDArray[np.int64],
    hidden: int,
    seed: int,
    c: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 1000,
) -> CellMaxMLPReader:
    """Fits a cell max reader with an MLP value head per layer.

    The reader of `fit_cell_max` with each layer's value head replaced by
    `hidden` ReLU units and an output weight vector, each layer keeping its
    own output bias. The loss, its penalty on the hidden and output weights,
    the scaling, the settings of L-BFGS, the seeded start and the stall
    test are those of `fit_attention_mlp`. At the start every cell scores 0,
    so the cells of a stimulus tie and the first step is the mean pooling
    one, as in `fit_cell_max`, with the same trap. A fit that stalls runs
    again, once, from the mean pooling solution of `fit_cell_max` put on
    two units of each layer, `relu(v @ z) - relu(-v @ z) = v @ z` for its
    head v, with the other units at their seeded start and zero output
    weights, so every cell starts at the value the linear restart gives it;
    the second run is kept unless it stalls too, and warns with
    `RestartedFitWarning`.

    Args:
        acts: Token rows of the layers to read, such as those of the
            `response` position.
        labels: Binary labels of shape (n_stimuli,), 1 for the positive
            pole.
        hidden: Hidden units per layer, at least 2, so the restart can hold
            a linear head.
        seed: Seed of the start.
        c: The inverse L2 strength, scikit-learn's `C`: smaller values
            regularize more.
        tol: Gradient tolerance of L-BFGS, scikit-learn's `tol`.
        max_iter: Iteration cap of each L-BFGS run; a fit whose kept run
            stops before converging warns with `ConvergenceWarning`.

    Returns:
        The reader. It has no direction to export, see
        `CellMaxMLPReader.rep_reader`.

    Raises:
        ValueError: If the labels do not have one binary entry per stimulus
            with both classes, `c` is not positive, `hidden` is below 2, a
            stimulus has no rows, `acts` is not finite, or a layer's rows
            are all equal.
    """
    stimulus, targets = _fit_inputs(acts, labels, c)
    _check_hidden(hidden, 2)
    weights, unit_biases, outputs, biases = _fit_max_mlp(
        acts.data,
        acts.layers,
        stimulus,
        targets,
        c,
        tol,
        max_iter,
        _hidden_start(acts.data.shape, hidden, seed),
    )
    return CellMaxMLPReader(
        layers=tuple(int(layer) for layer in acts.layers),
        hidden=weights,
        hidden_biases=unit_biases,
        outputs=outputs,
        biases=biases,
    )


def _scored_stimulus(
    acts: Activations, layers: tuple[int, ...], width: int
) -> torch.Tensor:
    # Checks that a reader of these layers and width can score `acts` and
    # returns the stimulus of each row.
    if acts.layers != layers:
        raise ValueError(
            f"acts must hold the layers {layers}, got {acts.layers}"
        )
    if acts.data.shape[2] != width:
        raise ValueError(
            f"acts must have width {width}, got {acts.data.shape[2]}"
        )
    return torch.tensor(_row_stimulus(acts))


def _fit_inputs(
    acts: Activations, labels: NDArray[np.int64], c: float
) -> tuple[torch.Tensor, torch.Tensor]:
    # Checks the labels, `c` and the rows of a fit on cells; returns the
    # stimulus of each row and the labels as float64.
    _check_classes(labels, acts.n_stimuli)
    if not c > 0.0:
        raise ValueError(f"c must be positive, got {c}")
    stimulus = torch.tensor(_row_stimulus(acts))
    return stimulus, torch.tensor(labels, dtype=torch.float64)


def _value_heads(
    method: str,
    layers: tuple[int, ...],
    values: NDArray[np.float64],
    biases: NDArray[np.float64],
    signs: NDArray[np.float32],
) -> RepReader:
    squares = np.square(values).sum(axis=1, keepdims=True)
    if not squares.all():
        raise ValueError("a value head is zero, so it has no direction")
    means = -biases[:, None] * values / squares
    return RepReader.from_arrays(method, layers, values, means, signs)


def _head_signs(
    heads: RepReader,
    acts: Activations,
    labels: NDArray[np.int64],
    pairs: NDArray[np.int64] | None,
    groups: NDArray[np.int64] | None,
) -> NDArray[np.float32]:
    # Orients value heads over the token rows the way the sweep orients a
    # `RepReader.fit` reader of `response` rows.
    oriented = orient(
        heads,
        acts.data,
        labels=acts.row_values(labels),
        pairs=None if pairs is None else acts.row_groups(pairs),
        groups=None if groups is None else acts.row_groups(groups),
    )
    return oriented.signs


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
    # Subtracting each stimulus's largest logit leaves its softmax unchanged
    # and keeps the exponentials from overflowing.
    peaks = _max_pool(logits.detach(), stimulus, n_stimuli)
    cells = stimulus.repeat(logits.shape[0])
    logits = logits.reshape(-1)
    values = values.reshape(-1)
    weights = torch.exp(logits - peaks[cells])
    total = logits.new_zeros(n_stimuli).scatter_add(0, cells, weights)
    pooled = logits.new_zeros(n_stimuli).scatter_add(0, cells, weights * values)
    return pooled / total


def _max_pool(
    values: torch.Tensor, stimulus: torch.Tensor, n_stimuli: int
) -> torch.Tensor:
    # The highest of the cells of shape (layers, rows) of each stimulus.
    # Autograd splits the gradient of a maximum evenly among tied cells.
    cells = stimulus.repeat(values.shape[0])
    return values.new_full((n_stimuli,), -torch.inf).scatter_reduce(
        0, cells, values.reshape(-1), "amax"
    )


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
    scaled, centers, scales = _scaled_cells(data, layers)
    n_layers, _, width = scaled.shape
    size = n_layers * width

    def split(
        params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        queries = params[:size].view(n_layers, width)
        values = params[size:-1].view(n_layers, width)
        return queries, values, params[-1]

    def loss(params: torch.Tensor) -> torch.Tensor:
        queries, values, bias = split(params)
        logits = torch.einsum("krd,kd->kr", scaled, queries)
        heads = torch.einsum("krd,kd->kr", scaled, values) + bias
        scores = _pool(logits, heads, stimulus, targets.shape[0])
        penalty = queries.square().sum() + values.square().sum()
        return _penalized_loss(scores, targets, penalty, c)

    solution = _minimize(loss, 2 * size + 1, tol, max_iter, "attention")
    fitted = split(torch.tensor(solution, dtype=torch.float64))
    queries, query_shifts = _unscaled(fitted[0].numpy(), centers, scales)
    values, value_shifts = _unscaled(fitted[1].numpy(), centers, scales)
    return queries, -query_shifts, values, float(fitted[2]) - value_shifts


def _fit_max(
    data: NDArray[np.float32],
    layers: tuple[int, ...],
    stimulus: torch.Tensor,
    targets: torch.Tensor,
    c: float,
    tol: float,
    max_iter: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Fits the value heads and biases of a cell max reader on scaled rows
    # and returns them in activation units. Kept apart from `fit_cell_max`
    # so the scaled float64 rows are freed before the heads are oriented,
    # and so the warning of `_minimize` reaches the caller of either fit.
    scaled, centers, scales = _scaled_cells(data, layers)
    n_layers, _, width = scaled.shape
    size = n_layers * width

    def loss(params: torch.Tensor) -> torch.Tensor:
        values = params[:size].view(n_layers, width)
        heads = torch.einsum("krd,kd->kr", scaled, values)
        heads = heads + params[size:, None]
        scores = _max_pool(heads, stimulus, targets.shape[0])
        return _penalized_loss(scores, targets, values.square().sum(), c)

    mean_pooled = partial(
        _mean_pooled, scaled, stimulus, targets, c, tol, max_iter
    )
    solution = _minimize(
        loss, size + n_layers, tol, max_iter, "cell max", restart=mean_pooled
    )
    heads = solution[:size].reshape(n_layers, width)
    values, shifts = _unscaled(heads, centers, scales)
    return values, solution[size:] - shifts


def _mean_pooled(
    scaled: torch.Tensor,
    stimulus: torch.Tensor,
    targets: torch.Tensor,
    c: float,
    tol: float,
    max_iter: int,
) -> NDArray[np.float64]:
    # The fit of the scores the zero start of a cell max fit sees: each
    # stimulus scores the mean value of its cells of all layers. The loss is
    # convex, so its minimum is unique but for the biases, which stay equal.
    # It is returned split into one term per layer, its heads and biases
    # over the number of layers, which sum to the mean pooled score: a cell
    # then takes the value its layer adds to that score, so the max starts
    # on the scale of the mean.
    n_layers, _, width = scaled.shape
    size = n_layers * width
    n_stimuli = targets.shape[0]
    counts = torch.bincount(stimulus, minlength=n_stimuli)
    means = scaled.new_zeros((n_layers, n_stimuli, width))
    means.index_add_(1, stimulus, scaled)
    means /= counts[:, None]

    def pooled_loss(params: torch.Tensor) -> torch.Tensor:
        values = params[:size].view(n_layers, width)
        cells = torch.einsum("knd,kd->kn", means, values)
        scores = (cells + params[size:, None]).mean(dim=0)
        return _penalized_loss(scores, targets, values.square().sum(), c)

    start = np.zeros(size + n_layers)
    fitted = _lbfgs(pooled_loss, start, tol, max_iter).x
    return np.asarray(fitted / n_layers, dtype=np.float64)


def _check_hidden(hidden: int, minimum: int) -> None:
    if hidden < minimum:
        raise ValueError(f"hidden must be at least {minimum}, got {hidden}")


def _hidden_start(
    shape: tuple[int, ...], hidden: int, seed: int
) -> NDArray[np.float64]:
    # The seeded hidden weights an MLP fit on cells of shape (K, rows, d)
    # starts from: each unit's weights have a norm near 1.
    n_layers, _, width = shape
    draw = np.random.default_rng(seed).standard_normal(
        (n_layers, hidden, width)
    )
    return np.asarray(draw / np.sqrt(width), dtype=np.float64)


def _mlp_split(
    params: torch.Tensor, n_layers: int, hidden: int, width: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # The hidden weights, hidden biases and output weights of MLP heads,
    # laid out in that order.
    units = n_layers * hidden
    weights = params[: units * width].view(n_layers, hidden, width)
    biases = params[units * width : units * (width + 1)].view(n_layers, hidden)
    outputs = params[units * (width + 1) :].view(n_layers, hidden)
    return weights, biases, outputs


def _mlp_cells(
    scaled: torch.Tensor,
    weights: torch.Tensor,
    biases: torch.Tensor,
    outputs: torch.Tensor,
) -> torch.Tensor:
    # The MLP value of every cell of shape (K, rows, d), without the output
    # bias: one hidden layer of ReLU units per layer.
    units = torch.einsum("krd,khd->krh", scaled, weights) + biases[:, None, :]
    return torch.einsum("krh,kh->kr", torch.relu(units), outputs)


def _mlp_values(
    acts: Activations, reader: AttentionMLPReader | CellMaxMLPReader
) -> NDArray[np.float64]:
    # The value of every cell under a reader's MLP heads, in float64.
    values = np.empty(acts.data.shape[:2])
    for k, layer_rows in enumerate(acts.data):
        rows = layer_rows.astype(np.float64)
        units = rows @ reader.hidden[k].T + reader.hidden_biases[k]
        values[k] = np.maximum(units, 0.0) @ reader.outputs[k]
        values[k] += reader.biases[k]
    return values


def _unscaled_mlp(
    params: NDArray[np.float64],
    hidden: int,
    centers: NDArray[np.float64],
    scales: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    # Moves MLP heads that read scaled rows to activation units: the hidden
    # weights are divided by the scale of their layer and the center is
    # folded into the hidden biases; the output weights are unchanged.
    n_layers, width = centers.shape
    weights, biases, outputs = (
        part.numpy()
        for part in _mlp_split(
            torch.from_numpy(params), n_layers, hidden, width
        )
    )
    raw, shifts = _unscaled(
        weights.reshape(-1, width),
        np.repeat(centers, hidden, axis=0),
        np.repeat(scales, hidden),
    )
    return (
        raw.reshape(n_layers, hidden, width),
        biases - shifts.reshape(n_layers, hidden),
        outputs.copy(),
    )


def _fit_attention_mlp(
    data: NDArray[np.float32],
    layers: tuple[int, ...],
    stimulus: torch.Tensor,
    targets: torch.Tensor,
    c: float,
    tol: float,
    max_iter: int,
    hidden_start: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    # Fits the queries and MLP heads of one softmax over every layer on
    # scaled rows and returns the queries, offsets, hidden weights, hidden
    # biases, output weights and output biases in activation units.
    scaled, centers, scales = _scaled_cells(data, layers)
    n_layers, hidden, width = hidden_start.shape
    size = n_layers * width

    def loss(params: torch.Tensor) -> torch.Tensor:
        queries = params[:size].view(n_layers, width)
        weights, biases, outputs = _mlp_split(
            params[size:-1], n_layers, hidden, width
        )
        logits = torch.einsum("krd,kd->kr", scaled, queries)
        heads = _mlp_cells(scaled, weights, biases, outputs) + params[-1]
        scores = _pool(logits, heads, stimulus, targets.shape[0])
        penalty = (
            queries.square().sum()
            + weights.square().sum()
            + outputs.square().sum()
        )
        return _penalized_loss(scores, targets, penalty, c)

    start = np.zeros(size + n_layers * hidden * (width + 2) + 1)
    start[size : size + hidden_start.size] = hidden_start.ravel()
    solution = _minimize(
        loss, start.size, tol, max_iter, "attention MLP", start=start
    )
    queries, shifts = _unscaled(
        solution[:size].reshape(n_layers, width), centers, scales
    )
    heads = _unscaled_mlp(solution[size:-1], hidden, centers, scales)
    return queries, -shifts, *heads, np.full(n_layers, solution[-1])


def _fit_max_mlp(
    data: NDArray[np.float32],
    layers: tuple[int, ...],
    stimulus: torch.Tensor,
    targets: torch.Tensor,
    c: float,
    tol: float,
    max_iter: int,
    hidden_start: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    # Fits the MLP heads and output biases of a cell max reader on scaled
    # rows and returns the hidden weights, hidden biases, output weights
    # and output biases in activation units.
    scaled, centers, scales = _scaled_cells(data, layers)
    n_layers, hidden, width = hidden_start.shape
    size = n_layers * hidden * (width + 2)

    def loss(params: torch.Tensor) -> torch.Tensor:
        weights, biases, outputs = _mlp_split(
            params[:size], n_layers, hidden, width
        )
        heads = _mlp_cells(scaled, weights, biases, outputs)
        heads = heads + params[size:, None]
        scores = _max_pool(heads, stimulus, targets.shape[0])
        penalty = weights.square().sum() + outputs.square().sum()
        return _penalized_loss(scores, targets, penalty, c)

    def restart() -> NDArray[np.float64]:
        # The mean pooling solution of the linear cell max fit, its head v
        # of each layer on two units: relu(v @ z) - relu(-v @ z) = v @ z.
        linear = _mean_pooled(scaled, stimulus, targets, c, tol, max_iter)
        heads = linear[: n_layers * width].reshape(n_layers, width)
        weights = hidden_start.copy()
        weights[:, 0], weights[:, 1] = heads, -heads
        outputs = np.zeros((n_layers, hidden))
        outputs[:, 0], outputs[:, 1] = 1.0, -1.0
        return np.concatenate(
            [
                weights.ravel(),
                np.zeros(n_layers * hidden),
                outputs.ravel(),
                linear[n_layers * width :],
            ]
        )

    start = np.zeros(size + n_layers)
    start[: hidden_start.size] = hidden_start.ravel()
    solution = _minimize(
        loss,
        start.size,
        tol,
        max_iter,
        "cell max MLP",
        restart=restart,
        start=start,
    )
    heads = _unscaled_mlp(solution[:size], hidden, centers, scales)
    return *heads, solution[size:].copy()


def _scaled_cells(
    data: NDArray[np.float32], layers: tuple[int, ...]
) -> tuple[torch.Tensor, NDArray[np.float64], NDArray[np.float64]]:
    # Centers each layer's rows on their mean and divides them by their root
    # mean square norm, as the logistic finder does, in float64. Returns the
    # scaled rows with the centers and scales of the layers.
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
    return torch.from_numpy(rows), centers, scales


def _unscaled(
    heads: NDArray[np.float64],
    centers: NDArray[np.float64],
    scales: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Moves heads of shape (K, d) that read scaled rows to activation units:
    # a head h reads a row x as `raw @ x - shift`.
    raw = heads / scales[:, None]
    return raw, (raw * centers).sum(axis=1)


def _penalized_loss(
    scores: torch.Tensor, targets: torch.Tensor, penalty: torch.Tensor, c: float
) -> torch.Tensor:
    # scikit-learn's logistic loss divided by c times the number of targets:
    # the mean loss plus half the penalty over c n.
    strength = 1.0 / (c * targets.shape[0])
    loss = binary_cross_entropy_with_logits(scores, targets)
    return loss + 0.5 * strength * penalty


def _minimize(
    loss: Callable[[torch.Tensor], torch.Tensor],
    size: int,
    tol: float,
    max_iter: int,
    name: str,
    restart: Callable[[], NDArray[np.float64]] | None = None,
    start: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    # Minimizes a loss of `size` parameters from zeros, or from `start`,
    # with full-batch L-BFGS in float64 and scikit-learn's settings. A fit
    # whose loss never dropped below its loss at zero stalls: the loss of
    # its zero start, or of the all-zero reader for a fit that starts
    # elsewhere. Given `restart`, which returns a second start, a stalled
    # fit runs again from that start, and the second run is kept unless it
    # stalls too; any other fit runs once. The restart, and a stop before
    # convergence or a stall of the run that is kept, warn from the caller
    # of the public fit, two frames above this one's caller.
    with torch.no_grad():
        initial = float(loss(torch.zeros(size, dtype=torch.float64)))
    first = np.zeros(size) if start is None else start
    result = _lbfgs(loss, first, tol, max_iter)
    stalled = _stalled(initial, result.fun)
    where, at = (
        ("its zero start", "the zero start") if start is None else ("zero",) * 2
    )
    if restart is not None and stalled:
        refit = _lbfgs(loss, restart(), tol, max_iter)
        if not _stalled(initial, refit.fun):
            warnings.warn(
                f"the {name} reader stalled at {where}, so it was "
                f"fit again from the mean pooling solution: its loss "
                f"{float(refit.fun)!r} is below {initial!r}, its loss at "
                f"{at}",
                RestartedFitWarning,
                stacklevel=4,
            )
            result, stalled = refit, False
    if result.status != 0:
        warnings.warn(
            f"the {name} reader did not converge: {result.message}",
            ConvergenceWarning,
            stacklevel=4,
        )
    if stalled:
        warnings.warn(
            f"the {name} reader stalled: its loss {float(result.fun)!r} never "
            f"dropped below {initial!r}, its loss at {at}",
            StalledFitWarning,
            stacklevel=4,
        )
    return np.asarray(result.x, dtype=np.float64)


def _lbfgs(
    loss: Callable[[torch.Tensor], torch.Tensor],
    start: NDArray[np.float64],
    tol: float,
    max_iter: int,
) -> OptimizeResult:
    # Runs full-batch L-BFGS-B on a float64 loss from `start`, with the
    # gradient from autograd and scikit-learn's settings.
    def objective(
        theta: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        params = torch.tensor(theta, requires_grad=True)
        value = loss(params)
        (gradient,) = torch.autograd.grad(value, params)
        return float(value.detach()), gradient.numpy()

    return minimize(
        objective,
        start,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": max_iter,
            "maxls": 50,
            "gtol": tol,
            "ftol": _FTOL,
        },
    )


def _stalled(initial: float, final: float) -> bool:
    # The test `ftol` applies to each step, applied to a whole fit: a drop
    # no larger than rounding of the loss (log 2 at the zero start of both
    # cell fits) is no drop.
    return bool(initial - final <= _FTOL * max(abs(initial), abs(final), 1.0))
