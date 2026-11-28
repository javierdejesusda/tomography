"""Readers of the two-position invariance study, preregistration 4 and 5.

The preregistration is `docs/preregistration-invariance-reader.md`. Labels
are 0 and 1 everywhere in this module, 1 for the positive pole; the
objectives of section 5.2 use `y = 2 label - 1` in {-1, +1}.

A text is rendered under `K` renderings: the clean template first, then each
fit-set wrapper. A rendering carries a decoy level tag, and the containers
that feed fitting, tuning or a covariance (`LabelledSet`, `Pool`,
`within_item_covariance`) raise `AgainstCopyError` on an `against` tag
(section 8.2). Scoring takes plain arrays, so `against` copies can still be
scored in evaluation.

The canonical re-prompt oracle of section 5.5 needs no reader code: it is
the clean rendering of a report text scored by a fitted reader.
"""

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace
from itertools import pairwise
from typing import Literal
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from tomography.activations import Activations
from tomography.experiments.sweep import shuffle_within_groups
from tomography.p9.risk import text_scores
from tomography.reading.probes import AttentionReader
from tomography.reading.probes import fit_attention
from tomography.reading.reader import RepReader

Level = Literal["none", "random", "against"]
FloatArray = NDArray[np.floating]


class AgainstCopyError(ValueError):
    """An `against` copy reached a fit, a tuning step or a covariance."""


@dataclass(frozen=True)
class Renderings:
    """Names and decoy level tags of the renderings of every text.

    Attributes:
        names: Rendering names; the first is "clean", the clean template.
        levels: Decoy level of each rendering: "none" for the clean
            template and template variants, "random" or "against" for a
            decoy wrapper.
    """

    names: tuple[str, ...]
    levels: tuple[Level, ...]

    def __post_init__(self) -> None:
        """Validates the names and the levels.

        Raises:
            ValueError: If the lengths differ, the first rendering is not
                the clean template or a name repeats.
        """
        if len(self.names) != len(self.levels) or not self.names:
            raise ValueError("names and levels need equal, nonzero lengths")
        if self.names[0] != "clean" or self.levels[0] != "none":
            raise ValueError("the first rendering must be the clean template")
        if len(set(self.names)) != len(self.names):
            raise ValueError("rendering names must be unique")

    def require_fittable(self) -> None:
        """Refuses renderings with an `against` level (section 8.2).

        Raises:
            AgainstCopyError: If any rendering has the `against` level.
        """
        against = [
            name
            for name, level in zip(self.names, self.levels, strict=True)
            if level == "against"
        ]
        if against:
            raise AgainstCopyError(
                f"against copies are never fitted, tuned on or pooled: "
                f"{against}"
            )


def _check_features(
    stim: FloatArray,
    last: FloatArray,
    renderings: Renderings,
    layers: tuple[int, ...],
) -> None:
    if stim.ndim != 4 or stim.shape[:3] != last.shape[:3]:
        raise ValueError(
            "stim and last need shape (L, n, K, d) with equal L, n, K"
        )
    if stim.shape[0] != len(layers) or stim.shape[2] != len(renderings.names):
        raise ValueError("axis 0 must match layers and axis 2 the renderings")
    if any(b <= a for a, b in pairwise(layers)):
        raise ValueError("layers must be strictly increasing")


@dataclass(frozen=True, eq=False)
class LabelledSet:
    """Labelled texts of one draw with their wrapped copies.

    Attributes:
        stim: Features at the last stimulus token, shape (L, n, K, d): every
            candidate layer, text and rendering.
        last: Features at the last token, shape (L, n, K, d') with the same
            L, n and K.
        labels: 0/1 labels of shape (n,), 1 for the positive pole.
        pairs: Pair index of each text, shape (n,). Every pair holds two
            texts, one of each label.
        renderings: The renderings of axis 2; none may be `against`.
        layers: Model layer of each index of axis 0, strictly increasing.
    """

    stim: FloatArray
    last: FloatArray
    labels: NDArray[np.int64]
    pairs: NDArray[np.int64]
    renderings: Renderings
    layers: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validates the fields and refuses `against` copies.

        Raises:
            AgainstCopyError: If a rendering has the `against` level.
            ValueError: If the shapes disagree, a label is not 0 or 1, or a
                pair does not hold one text of each label.
        """
        self.renderings.require_fittable()
        _check_features(self.stim, self.last, self.renderings, self.layers)
        n = self.stim.shape[1]
        if self.labels.shape != (n,) or self.pairs.shape != (n,):
            raise ValueError("labels and pairs need shape (n,)")
        if not np.isin(self.labels, (0, 1)).all():
            raise ValueError("labels must be 0 or 1")
        for pair in np.unique(self.pairs):
            members = self.labels[self.pairs == pair]
            if sorted(members.tolist()) != [0, 1]:
                raise ValueError(f"pair {pair} needs one text of each label")

    def subset(self, index: NDArray[np.int64]) -> "LabelledSet":
        """The texts at `index`, with all their copies.

        Args:
            index: Text indices into axis 1.

        Returns:
            A new labelled set.
        """
        return LabelledSet(
            stim=self.stim[:, index],
            last=self.last[:, index],
            labels=self.labels[index],
            pairs=self.pairs[index],
            renderings=self.renderings,
            layers=self.layers,
        )


@dataclass(frozen=True, eq=False)
class Pool:
    """Unlabelled pool texts rendered like the labelled texts (section 4).

    Attributes:
        stim: Features at the last stimulus token, shape (L, M, K, d).
        last: Features at the last token, shape (L, M, K, d').
        renderings: The renderings of axis 2; none may be `against`.
        layers: Model layer of each index of axis 0, strictly increasing.
    """

    stim: FloatArray
    last: FloatArray
    renderings: Renderings
    layers: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validates the fields and refuses `against` copies.

        Raises:
            AgainstCopyError: If a rendering has the `against` level.
            ValueError: If the shapes disagree.
        """
        self.renderings.require_fittable()
        _check_features(self.stim, self.last, self.renderings, self.layers)


def standardization(
    x: FloatArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-dimension statistics of section 5.1.

    Args:
        x: Clean features of the labelled training texts, shape (n, d).

    Returns:
        The mean and the scale of shape (d,): the population standard
        deviation, with 1 where it is zero.
    """
    rows = np.asarray(x, dtype=np.float64)
    mean = rows.mean(axis=0)
    scale = rows.std(axis=0)
    scale[scale == 0.0] = 1.0
    return mean, scale


def within_item_covariance(
    phi: FloatArray, renderings: Renderings
) -> NDArray[np.float64]:
    """The class-free within-item covariance `Sigma_e` of section 4.

    Args:
        phi: Standardized pool features of shape (M, K, d), K renderings
            per pool text.
        renderings: The renderings of axis 1; none may be `against`.

    Returns:
        `Sigma_e` of shape (d, d), rescaled to a trace equal to d.

    Raises:
        AgainstCopyError: If a rendering has the `against` level.
        ValueError: If the shapes disagree, K < 2 or the trace is zero.
    """
    renderings.require_fittable()
    rows = np.asarray(phi, dtype=np.float64)
    m, k, d = rows.shape
    if k != len(renderings.names) or k < 2:
        raise ValueError("axis 1 must hold the K >= 2 renderings")
    centred = rows - rows.mean(axis=0, keepdims=True)
    residual = (centred - centred.mean(axis=1, keepdims=True)).reshape(m * k, d)
    sigma = residual.T @ residual / (m * (k - 1))
    trace = float(np.trace(sigma))
    if trace <= 0.0:
        raise ValueError("Sigma_e has zero trace")
    return np.asarray(sigma * (d / trace), dtype=np.float64)


def fold_ids(
    pairs: NDArray[np.int64], *, seed: int, n_folds: int = 4
) -> NDArray[np.int64]:
    """Cross-validation folds by pair (section 5.3).

    The distinct pairs are permuted with a seeded generator and split into
    `n_folds` folds of equal size (16 pairs each at 64 pairs); every text
    takes the fold of its pair, so a text and its copies stay together.

    Args:
        pairs: Pair index of each text, shape (n,).
        seed: Seed of the permutation.
        n_folds: Number of folds.

    Returns:
        The fold of each text, shape (n,).
    """
    unique = np.unique(pairs)
    order = np.random.default_rng(seed).permutation(unique.size)
    fold_of_pair = np.empty(unique.size, dtype=np.int64)
    for fold, chunk in enumerate(np.array_split(order, n_folds)):
        fold_of_pair[chunk] = fold
    return fold_of_pair[np.searchsorted(unique, pairs)]


def auroc(scores: FloatArray, labels: NDArray[np.int64]) -> float:
    """Area under the ROC curve, ties counted as one half.

    Args:
        scores: Scores of shape (n,), higher for label 1.
        labels: 0/1 labels of shape (n,) with both classes.

    Returns:
        The AUROC.
    """
    ranks = rankdata(scores)
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = labels.size - n_pos
    return float(
        (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


Position = Literal["stim", "last"]
Family = Literal["lr", "mean_diff", "ulda"]
Penalty = Literal["none", "sigma_e", "random_subspace"]
Stats = list[tuple[NDArray[np.float64], NDArray[np.float64]]]
Fit = tuple[NDArray[np.float64], float, bool]

LAMBDAS: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
GAMMAS: tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0)
ALPHAS: tuple[float, ...] = (0.01, 0.1, 0.3, 0.6, 0.9)
PSEUDO_LAMBDA = 1e-2
STACK_LAMBDA = 1e-2


@dataclass(frozen=True)
class Spec:
    """One reader of sections 5.2 to 5.5 that `tune` can tune.

    Attributes:
        name: Arm or baseline id, such as "A6".
        family: "lr" for the logistic objectives, "mean_diff" for the
            difference of means, "ulda" for U-LDA.
        positions: Read positions in the order of the feature vector:
            ("stim", "last") for `phi = [z_stim ; z_last]`.
        augmented: Fit on the wrapped copies and tune on the
            worst-condition AUROC; otherwise clean texts and clean AUROC.
        penalty: "sigma_e" adds `(gamma/2) w^T Sigma_e w`;
            "random_subspace" uses `Q Sigma_e Q^T` instead.
        subspace_seed: Seed of `Q` for the random-subspace control.
    """

    name: str
    family: Family
    positions: tuple[Position, ...]
    augmented: bool
    penalty: Penalty = "none"
    subspace_seed: int = 0


ARMS: dict[str, Spec] = {
    spec.name: spec
    for spec in (
        Spec("A1", "lr", ("last",), False),
        Spec("A2", "lr", ("stim",), False),
        Spec("A3", "lr", ("last",), True),
        Spec("A4", "lr", ("stim",), True),
        Spec("A5", "lr", ("stim", "last"), True),
        Spec("A6", "lr", ("stim", "last"), True, "sigma_e"),
        Spec(
            "random_subspace", "lr", ("stim", "last"), True, "random_subspace"
        ),
        Spec("A7", "ulda", ("last",), False),
        Spec("A8", "ulda", ("last",), True),
        Spec("B1", "mean_diff", ("last",), False),
        Spec("B2", "mean_diff", ("stim",), False),
        Spec("B3", "lr", ("last",), False),
        Spec("B4", "lr", ("stim",), False),
        Spec("B8", "lr", ("last",), True),
        Spec("B9", "lr", ("stim",), True),
    )
}


@dataclass(frozen=True)
class Candidate:
    """One point of the tuning grid.

    Attributes:
        layers: Candidate-layer index (axis 0) per position of the spec.
        lam: `lambda`, 0 when the spec has none.
        gamma: `gamma`, 0 when the spec has none.
        alpha: `alpha`, 0 when the spec has none.
    """

    layers: tuple[int, ...]
    lam: float = 0.0
    gamma: float = 0.0
    alpha: float = 0.0


@dataclass(frozen=True, eq=False)
class Block:
    """The part of a linear reader that reads one position at one layer.

    Attributes:
        position: "stim" or "last".
        layer: Model layer.
        index: Index of that layer on axis 0 of the features.
        mean: Standardization mean, shape (d,).
        scale: Standardization scale, shape (d,).
        weight: Weight on the standardized features, shape (d,).
    """

    position: Position
    layer: int
    index: int
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    weight: NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class LinearReader:
    """A fitted reader whose score is linear in standardized features.

    Attributes:
        name: Arm or baseline id.
        blocks: The positions and layers it reads.
        bias: The intercept.
        params: The chosen hyperparameters by name ("lambda", "gamma",
            "alpha").
        criterion: The cross-validated criterion of the choice, NaN when
            the reader was not tuned.
        fallbacks: 1 if this fit used the U-LDA fallback of section 5.2.
    """

    name: str
    blocks: tuple[Block, ...]
    bias: float
    params: dict[str, float]
    criterion: float = float("nan")
    fallbacks: int = 0

    def score(self, stim: FloatArray, last: FloatArray) -> NDArray[np.float64]:
        """Scores texts: `w^T phi(x) + b`, higher for label 1.

        Args:
            stim: Last stimulus token features, shape (L, n, d), on the
                candidate layers of the fit.
            last: Last token features, shape (L, n, d').

        Returns:
            Scores of shape (n,).
        """
        total = np.full(last.shape[1], self.bias)
        for block in self.blocks:
            x = (stim if block.position == "stim" else last)[block.index]
            z = (np.asarray(x, dtype=np.float64) - block.mean) / block.scale
            total = total + z @ block.weight
        return total


@dataclass(frozen=True, eq=False)
class Tuning:
    """A reader tuned by cross-validation and refitted on all texts.

    Attributes:
        reader: The refitted reader.
        choice: The chosen grid point.
        candidates: Every grid point, in the order of `criteria`.
        criteria: Cross-validated criterion of each grid point.
        oof: Out-of-fold scores, shape (n_candidates, K, n): every text
            under every rendering, scored by the fit without its fold.
        folds: Fold of each text.
        fits: Number of statistic fits: folds times layer choices, plus
            the refit.
        fallbacks: How many of those fits used the U-LDA fallback.
    """

    reader: LinearReader
    choice: Candidate
    candidates: tuple[Candidate, ...]
    criteria: NDArray[np.float64]
    oof: NDArray[np.float64]
    folds: NDArray[np.int64]
    fits: int
    fallbacks: int


def haar_orthogonal(d: int, *, seed: int) -> NDArray[np.float64]:
    """A seeded Haar-random orthogonal matrix.

    Args:
        d: Dimension.
        seed: Seed of the Gaussian draw.

    Returns:
        `Q` of shape (d, d): the QR factor of a Gaussian matrix with the
        signs of the diagonal of R folded in, which makes it Haar.
    """
    gaussian = np.random.default_rng(seed).standard_normal((d, d))
    q, r = np.linalg.qr(gaussian)
    return np.asarray(q * np.sign(np.diag(r)))


def rotate_covariance(
    sigma: NDArray[np.float64], *, seed: int
) -> NDArray[np.float64]:
    """The random-subspace covariance `Q Sigma_e Q^T` of section 5.5.

    Args:
        sigma: `Sigma_e` of shape (d, d).
        seed: Seed of `Q`.

    Returns:
        The rotated covariance: same spectrum, random directions.
    """
    q = haar_orthogonal(sigma.shape[0], seed=seed)
    return np.asarray(q @ sigma @ q.T)


def _features(source: LabelledSet | Pool, position: Position) -> FloatArray:
    return source.stim if position == "stim" else source.last


def _stats(
    data: LabelledSet, positions: tuple[Position, ...], layers: tuple[int, ...]
) -> Stats:
    return [
        standardization(_features(data, position)[index, :, 0])
        for position, index in zip(positions, layers, strict=True)
    ]


def _phi(
    source: LabelledSet | Pool,
    positions: tuple[Position, ...],
    layers: tuple[int, ...],
    stats: Stats,
) -> NDArray[np.float64]:
    return np.concatenate(
        [
            (np.asarray(_features(source, position)[index], np.float64) - mean)
            / scale
            for position, index, (mean, scale) in zip(
                positions, layers, stats, strict=True
            )
        ],
        axis=-1,
    )


def _rows(
    phi: NDArray[np.float64], labels: NDArray[np.int64], augmented: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    n, k, width = phi.shape
    y = 2.0 * labels.astype(np.float64) - 1.0
    if not augmented:
        return phi[:, 0], y, np.full(n, 1.0 / n)
    return (
        phi.reshape(n * k, width),
        np.repeat(y, k),
        np.full(n * k, 1.0 / (n * k)),
    )


def _logistic(
    z: NDArray[np.float64], y: NDArray[np.float64], v: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """Minimizes `sum v log(1 + exp(-y (z theta + b))) + |theta|^2 / 2`."""
    k = z.shape[1]

    def objective(p: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        theta = p[:k]
        margin = y * (z @ theta + p[k])
        loss = float(v @ np.logaddexp(0.0, -margin))
        r = -v * y * expit(-margin)
        gradient = np.empty(k + 1)
        gradient[:k] = z.T @ r + theta
        gradient[k] = r.sum()
        return loss + 0.5 * float(theta @ theta), gradient

    result = minimize(
        objective,
        np.zeros(k + 1),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 20000, "gtol": 1e-10, "ftol": 1e-15},
    )
    if result.status == 1:
        warnings.warn(
            "L-BFGS reached its iteration cap", RuntimeWarning, stacklevel=2
        )
    return np.asarray(result.x[:k]), float(result.x[k])


def _span_fit(
    eigenvalues: NDArray[np.float64],
    vectors: NDArray[np.float64],
    y: NDArray[np.float64],
    v: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Solves the penalized logistic problem in the span of the rows.

    With `P` the penalty matrix and `G = X P^-1 X^T = V E V^T`, the
    objective in `w` equals a unit ridge logistic regression on
    `Z = V E^(1/2)`, with `w = P^-1 X^T V E^(-1/2) theta`. This is an exact
    change of variables, not an approximation.

    Returns:
        The coefficient `c = V E^(-1/2) theta`, so that `w = P^-1 X^T c`,
        and the intercept.
    """
    keep = eigenvalues > max(float(eigenvalues.max()), 0.0) * 1e-12
    basis = vectors[:, keep]
    root = np.sqrt(eigenvalues[keep])
    theta, bias = _logistic(basis * root, y, v)
    return basis @ (theta / root), bias


def _check_pool(pool: Pool | None, data: LabelledSet) -> Pool:
    if pool is None:
        raise ValueError("this reader needs the unlabelled pool")
    if pool.renderings != data.renderings or pool.layers != data.layers:
        raise ValueError("pool and texts need equal renderings and layers")
    return pool


def _lr_group(
    spec: Spec,
    data: LabelledSet,
    pool: Pool | None,
    layers: tuple[int, ...],
    stats: Stats,
    candidates: list[Candidate],
) -> list[Fit]:
    phi = _phi(data, spec.positions, layers, stats)
    x, y, v = _rows(phi, data.labels, spec.augmented)
    fits: list[Fit] = []
    if spec.penalty == "none":
        eigenvalues, vectors = np.linalg.eigh(x @ x.T)
        for candidate in candidates:
            c, bias = _span_fit(eigenvalues / candidate.lam, vectors, y, v)
            fits.append((x.T @ c / candidate.lam, bias, False))
        return fits
    pool_phi = _phi(_check_pool(pool, data), spec.positions, layers, stats)
    spectrum, u = np.linalg.eigh(
        within_item_covariance(pool_phi, data.renderings)
    )
    spectrum = np.clip(spectrum, 0.0, None)
    if spec.penalty == "random_subspace":
        u = haar_orthogonal(u.shape[0], seed=spec.subspace_seed) @ u
    xu = x @ u
    for candidate in candidates:
        inverse = 1.0 / (candidate.lam + candidate.gamma * spectrum)
        eigenvalues, vectors = np.linalg.eigh((xu * inverse) @ xu.T)
        c, bias = _span_fit(eigenvalues, vectors, y, v)
        fits.append((u @ (inverse * (xu.T @ c)), bias, False))
    return fits


def _params(spec: Spec, candidate: Candidate) -> dict[str, float]:
    if spec.family == "ulda":
        return {"alpha": candidate.alpha}
    if spec.family == "mean_diff":
        return {}
    if spec.penalty == "none":
        return {"lambda": candidate.lam}
    return {"lambda": candidate.lam, "gamma": candidate.gamma}


def _reader(
    spec: Spec,
    data: LabelledSet,
    candidate: Candidate,
    stats: Stats,
    fit: Fit,
) -> LinearReader:
    weight, bias, fallback = fit
    blocks = []
    start = 0
    for position, index, (mean, scale) in zip(
        spec.positions, candidate.layers, stats, strict=True
    ):
        stop = start + mean.size
        blocks.append(
            Block(
                position,
                data.layers[index],
                index,
                mean,
                scale,
                weight[start:stop],
            )
        )
        start = stop
    return LinearReader(
        spec.name,
        tuple(blocks),
        bias,
        _params(spec, candidate),
        fallbacks=int(fallback),
    )


def _group_fits(
    spec: Spec,
    data: LabelledSet,
    pool: Pool | None,
    layers: tuple[int, ...],
    stats: Stats,
    candidates: list[Candidate],
) -> list[Fit]:
    if spec.family == "mean_diff":
        phi = _phi(data, spec.positions, layers, stats)
        return [(_class_difference(phi[:, 0], data.labels), 0.0, False)]
    if spec.family == "ulda":
        return _ulda_group(spec, data, pool, layers, stats, candidates)
    return _lr_group(spec, data, pool, layers, stats, candidates)


def _class_difference(
    x: NDArray[np.float64], labels: NDArray[np.int64]
) -> NDArray[np.float64]:
    return np.asarray(x[labels == 1].mean(axis=0) - x[labels == 0].mean(axis=0))


def _ulda_group(
    spec: Spec,
    data: LabelledSet,
    pool: Pool | None,
    layers: tuple[int, ...],
    stats: Stats,
    candidates: list[Candidate],
) -> list[Fit]:
    """U-LDA (A7, A8) at one layer for every `alpha`.

    `d_hat` comes from the labelled texts (and, for A8, their copies, each
    text weighted 1). The pseudo-labeller is the A1 (A8: A3) objective at
    `PSEUDO_LAMBDA` on the same texts and layer; it labels each pool text
    from its clean rendering. `S` is centred per pseudo-class (and, for
    A8, per wrapper) with denominator `K' (M - 2)`, `K'` = 1 for A7 and K
    for A8; if a pseudo-class holds fewer than 2 pool texts, `S` is centred
    on the pool mean instead with denominator `K' (M - 1)`.
    """
    source = _check_pool(pool, data)
    phi = _phi(data, spec.positions, layers, stats)
    texts = phi.mean(axis=1) if spec.augmented else phi[:, 0]
    d_hat = _class_difference(texts, data.labels)
    labeller = Spec("g", "lr", spec.positions, spec.augmented)
    weight, bias, _ = _lr_group(
        labeller,
        data,
        None,
        layers,
        stats,
        [Candidate(layers, lam=PSEUDO_LAMBDA)],
    )[0]
    pool_phi = _phi(source, spec.positions, layers, stats)
    positive = pool_phi[:, 0] @ weight + bias > 0
    m = positive.size
    fallback = min(int(positive.sum()), m - int(positive.sum())) < 2
    renderings = range(pool_phi.shape[1]) if spec.augmented else range(1)
    scatter = np.zeros((d_hat.size, d_hat.size))
    for a in renderings:
        x = pool_phi[:, a]
        if fallback:
            centred = x - x.mean(axis=0)
        else:
            centred = x.copy()
            centred[positive] -= x[positive].mean(axis=0)
            centred[~positive] -= x[~positive].mean(axis=0)
        scatter += centred.T @ centred
    groups = 1 if fallback else 2
    spectrum, u = np.linalg.eigh(scatter / (len(renderings) * (m - groups)))
    spectrum = np.clip(spectrum, 0.0, None)
    target = spectrum.sum() / d_hat.size
    projected = u.T @ d_hat
    return [
        (
            u @ (projected / ((1 - c.alpha) * spectrum + c.alpha * target)),
            0.0,
            fallback,
        )
        for c in candidates
    ]


def _fit_candidates(
    spec: Spec,
    data: LabelledSet,
    pool: Pool | None,
    candidates: list[Candidate],
) -> tuple[list[LinearReader], int]:
    """Fits every candidate, sharing the work of candidates on one layer.

    Returns:
        The readers in the order of `candidates` and the number of layer
        groups whose fit used the U-LDA fallback.
    """
    readers: dict[int, LinearReader] = {}
    fallbacks = 0
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, candidate in enumerate(candidates):
        groups.setdefault(candidate.layers, []).append(i)
    for layers, members in groups.items():
        stats = _stats(data, spec.positions, layers)
        chosen = [candidates[i] for i in members]
        fits = _group_fits(spec, data, pool, layers, stats, chosen)
        fallbacks += int(fits[0][2])
        for i, fit in zip(members, fits, strict=True):
            readers[i] = _reader(spec, data, candidates[i], stats, fit)
    return [readers[i] for i in range(len(candidates))], fallbacks


def fit_fixed(
    spec: Spec, data: LabelledSet, pool: Pool | None, candidate: Candidate
) -> LinearReader:
    """Fits a reader at fixed hyperparameters and layers.

    Every statistic (standardization, `Sigma_e`, `d_hat`, the
    pseudo-labeller, `S`) comes from `data` and `pool` only. Tuning uses
    this same fit inside every fold and for the final refit.

    Args:
        spec: The reader.
        data: The labelled texts of the fit.
        pool: The unlabelled pool; needed by A6, A7, A8 and the
            random-subspace control, ignored by the others.
        candidate: Layers and hyperparameters.

    Returns:
        The fitted reader.

    Raises:
        ValueError: If a needed pool is missing or does not match `data`.
    """
    readers, _ = _fit_candidates(spec, data, pool, [candidate])
    return readers[0]


def _grid(spec: Spec, layer_choices: list[tuple[int, ...]]) -> list[Candidate]:
    if spec.family == "mean_diff":
        return [Candidate(layers) for layers in layer_choices]
    if spec.family == "ulda":
        return [
            Candidate(layers, alpha=alpha)
            for layers in layer_choices
            for alpha in ALPHAS
        ]
    gammas = (0.0,) if spec.penalty == "none" else GAMMAS
    return [
        Candidate(layers, lam=lam, gamma=gamma)
        for layers in layer_choices
        for lam in LAMBDAS
        for gamma in gammas
    ]


def _criterion(
    oof: NDArray[np.float64], labels: NDArray[np.int64], augmented: bool
) -> float:
    if not augmented:
        return auroc(oof[0], labels)
    return min(auroc(scores, labels) for scores in oof)


def _best(candidates: list[Candidate], criteria: NDArray[np.float64]) -> int:
    """Best criterion; ties: larger lambda, gamma, alpha, then shallower."""
    return max(
        range(len(candidates)),
        key=lambda c: (
            criteria[c],
            candidates[c].lam,
            candidates[c].gamma,
            candidates[c].alpha,
            -candidates[c].layers[0],
        ),
    )


def _out_of_fold(
    data: LabelledSet,
    seed: int,
    fit: Callable[[LabelledSet], tuple[list[LinearReader], int]],
) -> tuple[NDArray[np.float64], NDArray[np.int64], int]:
    """Out-of-fold scores of every text under every rendering.

    Returns:
        The scores, shape (n_readers, K, n), the fold of each text and the
        fallbacks the fold fits reported.
    """
    folds = fold_ids(data.pairs, seed=seed)
    oof: NDArray[np.float64] | None = None
    fallbacks = 0
    for fold in np.unique(folds):
        test = np.flatnonzero(folds == fold)
        readers, used = fit(data.subset(np.flatnonzero(folds != fold)))
        fallbacks += used
        if oof is None:
            shape = (len(readers), len(data.renderings.names), folds.size)
            oof = np.empty(shape)
        stim, last = data.stim[:, test], data.last[:, test]
        for c, reader in enumerate(readers):
            for k in range(len(data.renderings.names)):
                oof[c, k, test] = reader.score(stim[:, :, k], last[:, :, k])
    assert oof is not None
    return oof, folds, fallbacks


def tune(
    spec: Spec,
    data: LabelledSet,
    pool: Pool | None,
    *,
    seed: int,
    layers: tuple[int, ...] | None = None,
) -> Tuning:
    """Tunes a reader by 4-fold cross-validation by pair (section 5.3).

    Out-of-fold scores of all texts are pooled and scored once: the clean
    AUROC for plain readers, the minimum AUROC over the clean template and
    each fit-set wrapper for augmented ones. Every statistic is recomputed
    inside each fold from its training folds and the pool. A
    single-position reader searches every candidate layer; a two-position
    reader takes `layers`. Ties go to the larger `lambda`, then `gamma`,
    then `alpha`, then the shallower layer. The choice is refitted on all
    texts.

    Args:
        spec: The reader.
        data: The labelled texts of the draw.
        pool: The unlabelled pool, or None for readers that do not use it.
        seed: Seed of the fold permutation.
        layers: Candidate-layer indices of a two-position reader, in the
            order of `spec.positions`.

    Returns:
        The tuning, with the refitted reader.

    Raises:
        ValueError: If a two-position reader gets no `layers`.
    """
    choices: list[tuple[int, ...]]
    if len(spec.positions) == 1:
        choices = [(index,) for index in range(len(data.layers))]
    elif layers is None:
        raise ValueError(f"{spec.name} needs the layers of its positions")
    else:
        choices = [layers]
    candidates = _grid(spec, choices)
    oof, folds, fallbacks = _out_of_fold(
        data, seed, lambda train: _fit_candidates(spec, train, pool, candidates)
    )
    criteria = np.array(
        [_criterion(scores, data.labels, spec.augmented) for scores in oof]
    )
    best = _best(candidates, criteria)
    readers, used = _fit_candidates(spec, data, pool, [candidates[best]])
    return Tuning(
        reader=replace(readers[0], criterion=float(criteria[best])),
        choice=candidates[best],
        candidates=tuple(candidates),
        criteria=criteria,
        oof=oof,
        folds=folds,
        fits=np.unique(folds).size * len(choices) + 1,
        fallbacks=fallbacks + used,
    )


def two_position_layers(data: LabelledSet, *, seed: int) -> tuple[int, int]:
    """Layers of a two-position reader (section 5.3).

    Args:
        data: The labelled texts of the draw.
        seed: Seed of the fold permutation.

    Returns:
        Candidate-layer indices (stim, last): the layer A4 chooses at the
        last stimulus token and the layer A3 chooses at the last token.
    """
    stim = tune(ARMS["A4"], data, None, seed=seed).choice.layers[0]
    last = tune(ARMS["A3"], data, None, seed=seed).choice.layers[0]
    return stim, last


def fit_arm(
    name: str,
    data: LabelledSet,
    pool: Pool | None,
    *,
    seed: int,
    layers: tuple[int, ...] | None = None,
) -> Tuning:
    """Tunes and refits one arm, baseline or control by name.

    Args:
        name: A key of `ARMS`.
        data: The labelled texts of the draw.
        pool: The unlabelled pool.
        seed: Seed of the fold permutation.
        layers: Layers of a two-position reader; by default those of
            `two_position_layers`.

    Returns:
        The tuning.
    """
    spec = ARMS[name]
    if len(spec.positions) == 2 and layers is None:
        layers = two_position_layers(data, seed=seed)
    return tune(spec, data, pool, seed=seed, layers=layers)


def pair_matrix(pairs: NDArray[np.int64]) -> NDArray[np.int64]:
    """The texts of every pair as rows of indices.

    Args:
        pairs: Pair index of each text, shape (n,), two texts per pair.

    Returns:
        Text indices of shape (m, 2), one row per pair in pair order.
    """
    return np.stack([np.flatnonzero(pairs == p) for p in np.unique(pairs)])


def _depth_mean(data: LabelledSet, lam: float) -> LinearReader:
    """B6 at a fixed `lambda`: the mean of standardized layer scores."""
    n_layers = len(data.layers)
    clean = (data.stim[:, :, 0], data.last[:, :, 0])
    blocks = []
    bias = 0.0
    for index in range(n_layers):
        layer = fit_fixed(ARMS["A1"], data, None, Candidate((index,), lam=lam))
        fitted = layer.score(*clean)
        spread = float(fitted.std()) or 1.0
        scale = spread * n_layers
        (block,) = layer.blocks
        blocks.append(replace(block, weight=block.weight / scale))
        bias += (layer.bias - float(fitted.mean())) / scale
    return LinearReader("B6", tuple(blocks), bias, {"lambda": lam})


def fit_depth_mean(data: LabelledSet, *, seed: int) -> Tuning:
    """B6, the depth mean adapted from HalluTracer (section 5.4).

    One A1 objective per candidate layer at the last token, with one shared
    `lambda`; each layer's score is standardized with the mean and standard
    deviation of its in-sample scores on the clean texts of the fit (zero
    standard deviation: divided by 1), and the reader scores the uniform
    mean over layers. `lambda` is chosen on the cross-validated clean AUROC
    of that mean, ties to the larger `lambda`.

    Args:
        data: The labelled texts of the draw.
        seed: Seed of the fold permutation.

    Returns:
        The tuning; every candidate spans all layers.
    """
    every = tuple(range(len(data.layers)))
    candidates = [Candidate(every, lam=lam) for lam in LAMBDAS]
    oof, folds, _ = _out_of_fold(
        data,
        seed,
        lambda train: ([_depth_mean(train, lam) for lam in LAMBDAS], 0),
    )
    criteria = np.array([auroc(scores[0], data.labels) for scores in oof])
    best = _best(candidates, criteria)
    reader = _depth_mean(data, candidates[best].lam)
    return Tuning(
        reader=replace(reader, criterion=float(criteria[best])),
        choice=candidates[best],
        candidates=tuple(candidates),
        criteria=criteria,
        oof=oof,
        folds=folds,
        fits=np.unique(folds).size + 1,
        fallbacks=0,
    )


def greedy_layers(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    aurocs: NDArray[np.float64],
    *,
    size: int,
    start: int | None = None,
) -> tuple[int, ...]:
    """The greedy layer choice of B7 (section 5.4).

    A text is misclassified by a layer when its prediction (label 1 if and
    only if the score is above 0) differs from its label; the double-fault
    rate of two layers is the share of texts both misclassify. Starting
    from `start`, each step adds the layer that minimizes the largest
    pairwise double-fault rate of the chosen set; ties go to the higher
    AUROC, then the shallower layer.

    Args:
        scores: Out-of-fold scores of shape (L, n).
        labels: 0/1 labels of shape (n,).
        aurocs: Cross-validated AUROC of each layer, shape (L,).
        size: Number of layers to choose.
        start: First layer; by default the highest AUROC, ties to the
            shallower layer.

    Returns:
        Layer indices in the order they were chosen.
    """
    wrong = (scores > 0) != (labels == 1)
    double = (wrong[:, None, :] & wrong[None, :, :]).mean(axis=2)
    n_layers = scores.shape[0]
    if start is None:
        start = max(range(n_layers), key=lambda layer: (aurocs[layer], -layer))
    chosen = [start]

    def largest(layer: int) -> float:
        members = [*chosen, layer]
        return max(
            float(double[a, b]) for a in members for b in members if a < b
        )

    while len(chosen) < min(size, n_layers):
        rest = [layer for layer in range(n_layers) if layer not in chosen]
        chosen.append(
            min(rest, key=lambda layer: (largest(layer), -aurocs[layer], layer))
        )
    return tuple(chosen)


@dataclass(frozen=True, eq=False)
class Stack:
    """B7, the 5-layer stack adapted from Nordby, Pais and Parrack.

    Attributes:
        reader: The stack as one linear reader: refitted layer readers,
            standardized with the out-of-fold statistics, then the stacker.
        layers: Candidate-layer indices in the order they were chosen.
        lambdas: The A1 `lambda` of each chosen layer.
        oof: Out-of-fold clean scores of the chosen layers, shape (5, n).
        oof_mean: Mean of each row of `oof`.
        oof_scale: Standard deviation of each row (zero: 1).
        weight: Stacker weights, shape (5,).
        bias: Stacker intercept.
    """

    reader: LinearReader
    layers: tuple[int, ...]
    lambdas: tuple[float, ...]
    oof: NDArray[np.float64]
    oof_mean: NDArray[np.float64]
    oof_scale: NDArray[np.float64]
    weight: NDArray[np.float64]
    bias: float


def fit_stack(
    data: LabelledSet,
    *,
    seed: int,
    a1: Tuning | None = None,
    size: int = 5,
) -> Stack:
    """B7 (section 5.4): five A1 layers chosen greedily, then stacked.

    Each layer keeps its own A1 tuning (`lambda` with the best
    cross-validated clean AUROC at that layer, ties to the larger). The
    greedy choice runs on the out-of-fold scores of the n texts and starts
    from the layer A1 selects. The stacker is the A1 objective at
    `STACK_LAMBDA` on the chosen out-of-fold scores, each standardized with
    its mean and standard deviation over the n texts.

    Args:
        data: The labelled texts of the draw.
        seed: Seed of the fold permutation.
        a1: The A1 tuning of the draw, recomputed if None.
        size: Number of stacked layers.

    Returns:
        The stack.
    """
    if a1 is None:
        a1 = tune(ARMS["A1"], data, None, seed=seed)
    per_layer = []
    for index in range(len(data.layers)):
        members = [
            c for c, cand in enumerate(a1.candidates) if cand.layers == (index,)
        ]
        per_layer.append(
            max(members, key=lambda c: (a1.criteria[c], a1.candidates[c].lam))
        )
    scores = a1.oof[per_layer, 0]
    order = greedy_layers(
        scores,
        data.labels,
        a1.criteria[per_layer],
        size=size,
        start=a1.choice.layers[0],
    )
    oof = scores[list(order)]
    mean = oof.mean(axis=1)
    spread = oof.std(axis=1)
    spread[spread == 0.0] = 1.0
    z = ((oof - mean[:, None]) / spread[:, None]).T
    y = 2.0 * data.labels.astype(np.float64) - 1.0
    eigenvalues, vectors = np.linalg.eigh(z @ z.T)
    c, bias = _span_fit(
        eigenvalues / STACK_LAMBDA, vectors, y, np.full(y.size, 1.0 / y.size)
    )
    weight = z.T @ c / STACK_LAMBDA
    lambdas = tuple(a1.candidates[per_layer[i]].lam for i in order)
    blocks = []
    total = bias
    for k, (index, lam) in enumerate(zip(order, lambdas, strict=True)):
        layer = fit_fixed(ARMS["A1"], data, None, Candidate((index,), lam=lam))
        (block,) = layer.blocks
        factor = weight[k] / spread[k]
        blocks.append(replace(block, weight=block.weight * factor))
        total += factor * (layer.bias - mean[k])
    reader = LinearReader(
        "B7", tuple(blocks), total, {"stack_lambda": STACK_LAMBDA}
    )
    return Stack(reader, order, lambdas, oof, mean, spread, weight, bias)


def fit_random_direction(
    data: LabelledSet, *, index: int, seed: int
) -> LinearReader:
    """The random-direction control of section 5.5.

    The M2 `random` reader (`RepReader.fit` with method "random") on the
    standardized clean last-token features at candidate layer `index` (the
    layer of B3): a seeded random unit direction, centred on the mean of
    the texts and oriented on the training labels by pair.

    Args:
        data: The labelled texts of the fit.
        index: Candidate-layer index, the layer B3 selects.
        seed: Seed of the direction.

    Returns:
        The reader.
    """
    mean, scale = standardization(data.last[index, :, 0])
    z = (np.asarray(data.last[index, :, 0], np.float64) - mean) / scale
    rep = RepReader.fit(
        "random",
        z[None],
        [data.layers[index]],
        labels=data.labels,
        pairs=pair_matrix(data.pairs),
        seed=seed,
    )
    weight = float(rep.signs[0]) * rep.directions[0].astype(np.float64)
    block = Block("last", data.layers[index], index, mean, scale, weight)
    bias = -float(weight @ rep.means[0].astype(np.float64))
    return LinearReader("random_direction", (block,), bias, {})


def shuffle_within_pairs(data: LabelledSet, *, seed: int) -> LabelledSet:
    """Labels shuffled within pairs, for the shuffled-labels control.

    Uses `shuffle_within_groups` of the sweep, as M2 does; refit the frozen
    winner on the result with `fit_arm`.

    Args:
        data: The labelled texts of the draw.
        seed: Seed of the shuffle.

    Returns:
        The same texts with shuffled labels.
    """
    labels = shuffle_within_groups(
        data.labels, pair_matrix(data.pairs), seed=seed
    )
    return replace(data, labels=labels)


def text_only_scores(
    train_texts: list[str],
    train_labels: NDArray[np.int64],
    test_texts: list[str],
) -> NDArray[np.float64]:
    """The text-only control: word TF-IDF logistic regression (section 5.5).

    Args:
        train_texts: The labelled stimulus texts of the fit.
        train_labels: Their 0/1 labels.
        test_texts: The stimulus texts to score.

    Returns:
        Scores of the test texts.
    """
    return text_scores(train_texts, train_labels, test_texts, analyzer="word")


def _stimuli(acts: Activations, index: NDArray[np.int64]) -> Activations:
    rows = np.isin(acts.row_stimulus, index)
    return replace(
        acts,
        data=acts.data[:, rows],
        row_stimulus=np.searchsorted(index, acts.row_stimulus[rows]),
        row_rank=acts.row_rank[rows],
        row_token=acts.row_token[rows],
        row_token_id=acts.row_token_id[rows],
        n_stimuli=index.size,
    )


def _one_layer(acts: Activations, index: int) -> Activations:
    return replace(acts, data=acts.data[[index]], layers=(acts.layers[index],))


def _attention(
    acts: Activations,
    labels: NDArray[np.int64],
    pairs: NDArray[np.int64],
    lam: float,
) -> AttentionReader:
    return fit_attention(
        acts,
        labels=labels,
        pairs=pair_matrix(pairs),
        c=1.0 / (labels.size * lam),
    )


@dataclass(frozen=True, eq=False)
class AttentionTuning:
    """B5, the M2 attention reader, tuned as in section 5.3.

    Attributes:
        reader: The reader refitted on all texts at the chosen layer.
        layer: The chosen model layer.
        lam: The chosen `lambda`, fitted with `c = 1 / (n lambda)`.
        criterion: Its cross-validated clean AUROC.
        candidates: (layer index in `acts.layers`, `lambda`) per candidate.
        criteria: Cross-validated clean AUROC of each candidate.
        oof: Out-of-fold scores, shape (n_candidates, n).
        folds: Fold of each text.
    """

    reader: AttentionReader
    layer: int
    lam: float
    criterion: float
    candidates: tuple[tuple[int, float], ...]
    criteria: NDArray[np.float64]
    oof: NDArray[np.float64]
    folds: NDArray[np.int64]

    def score(self, acts: Activations) -> NDArray[np.float64]:
        """Scores stimuli from their token rows.

        Args:
            acts: Token rows of the stimuli at layers including `layer`.

        Returns:
            Scores of shape (n_stimuli,).
        """
        single = _one_layer(acts, acts.layers.index(self.layer))
        return np.asarray(self.reader.score(single)[0], dtype=np.float64)


def tune_attention(
    acts: Activations,
    labels: NDArray[np.int64],
    pairs: NDArray[np.int64],
    *,
    seed: int,
) -> AttentionTuning:
    """B5 (section 5.4): `fit_attention` of M2, one reader per layer.

    The reader keeps its own M2 objective and row scaling, not the
    standardization of section 5.1. The grid is `LAMBDAS` mapped to
    `c = 1 / (n lambda)` with n the texts of the fit; the layers are those
    of `acts`, the relative depths 0.4, 0.55, 0.7, 0.85 and 1.0 rounded up,
    which the caller selects. Folds and the clean-AUROC criterion follow
    `tune`; ties go to the larger `lambda`, then the shallower layer.

    Args:
        acts: Token rows of the stimulus tokens of the n labelled texts
            (clean rendering) at the candidate layers.
        labels: 0/1 labels of shape (n,).
        pairs: Pair index of each text, shape (n,).
        seed: Seed of the fold permutation.

    Returns:
        The tuning, with the refitted reader.
    """
    candidates = [
        (index, lam) for index in range(len(acts.layers)) for lam in LAMBDAS
    ]
    folds = fold_ids(pairs, seed=seed)
    oof = np.empty((len(candidates), labels.size))
    for fold in np.unique(folds):
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        held_out = _stimuli(acts, test)
        for j, lam in enumerate(LAMBDAS):
            reader = _attention(
                _stimuli(acts, train), labels[train], pairs[train], lam
            )
            scores = reader.score(held_out)
            for index in range(len(acts.layers)):
                oof[index * len(LAMBDAS) + j, test] = scores[index]
    criteria = np.array([auroc(scores, labels) for scores in oof])
    best = max(
        range(len(candidates)),
        key=lambda c: (criteria[c], candidates[c][1], -candidates[c][0]),
    )
    index, lam = candidates[best]
    reader = _attention(_one_layer(acts, index), labels, pairs, lam)
    return AttentionTuning(
        reader=reader,
        layer=acts.layers[index],
        lam=lam,
        criterion=float(criteria[best]),
        candidates=tuple(candidates),
        criteria=criteria,
        oof=oof,
        folds=folds,
    )
