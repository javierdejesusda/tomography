"""A per-layer linear reader that scores activations along fitted directions."""

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

from tomography.reading.directions import FloatArray
from tomography.reading.directions import Method
from tomography.reading.directions import Preprocess
from tomography.reading.directions import fit_direction
from tomography.reading.metrics import GroupIndex
from tomography.reading.metrics import parse_groups

_FIELDS = ("method", "layers", "directions", "means", "signs")


@dataclass(frozen=True, eq=False)
class RepReader:
    """One unit direction, mean and orientation sign per layer.

    The score of an activation `h` at layer k is
    `signs[k] * (h - means[k]) . directions[k]`, so a higher score means
    more of the positive pole once the signs have been set.

    Attributes:
        method: Name of the method that produced the directions.
        layers: Layer indices, in the order of the first axis of every array.
        directions: Unit vectors of shape (K, d).
        means: Vectors of shape (K, d) subtracted before projecting.
        signs: Orientation of each layer, shape (K,), -1 or +1.
    """

    method: str
    layers: tuple[int, ...]
    directions: NDArray[np.float32]
    means: NDArray[np.float32]
    signs: NDArray[np.float32]

    @classmethod
    def fit(
        cls,
        method: Method,
        h: FloatArray,
        layers: Sequence[int],
        *,
        labels: NDArray[np.int64],
        pairs: NDArray[np.int64] | None = None,
        groups: NDArray[np.int64] | None = None,
        preprocess: Preprocess = "center",
        seed: int = 0,
        refit_at_zero: bool = False,
    ) -> Self:
        """Fits and orients one direction per layer.

        The unpaired `mean_diff` and `logistic` finders fit on the labels;
        for every method the labels also orient each layer with the
        reference rule. Within each group, the sign is positive when the
        label-1 member holds the highest score more often than the lowest,
        negative in the opposite case, and positive on a tie. The groups are
        `groups`, else the `pairs` whose members differ in label. Without
        either, the sign makes the mean score of the label-1 rows exceed
        that of the label-0 rows.

        Layer k draws its randomness from a seed derived from `seed` and k,
        so the layers of a random baseline are independent.

        Args:
            method: Direction finder, see `fit_direction`.
            h: Activations of shape (K, n, d), K = `len(layers)`.
            layers: Layer index of each slice of `h`.
            labels: Binary labels of shape (n,), 1 for the positive pole.
            pairs: Row index pairs of shape (m, 2); needed by `pca_diff`.
            groups: Row groups of shape (g, k) with exactly one label-1
                member each, padded with -1.
            preprocess: Vector preparation, see `fit_direction`.
            seed: Base seed of the stochastic finders.
            refit_at_zero: See `fit_direction`.

        Returns:
            The fitted reader.

        Raises:
            ValueError: If the shapes disagree, a group does not have
                exactly one label-1 member, or the sign cannot be set
                because a class is missing.
        """
        if h.ndim != 3:
            raise ValueError(f"h must have shape (K, n, d), got {h.shape}")
        if h.shape[0] != len(layers):
            raise ValueError(
                f"layers has {len(layers)} entries for {h.shape[0]} layers"
            )
        index = _orientation_groups(pairs, groups, labels)
        fits = [
            fit_direction(
                method,
                h[k],
                labels=labels,
                pairs=pairs,
                preprocess=preprocess,
                seed=_layer_seed(seed, k),
                refit_at_zero=refit_at_zero,
            )
            for k in range(h.shape[0])
        ]
        unsigned = cls(
            method=method,
            layers=tuple(int(layer) for layer in layers),
            directions=np.stack([fit.direction for fit in fits]),
            means=np.stack([fit.mean for fit in fits]),
            signs=np.ones(len(fits), dtype=np.float32),
        )
        signs = _orientation(unsigned._project(h), labels, index)
        return replace(unsigned, signs=signs)

    @classmethod
    def from_arrays(
        cls,
        method: str,
        layers: Sequence[int],
        directions: FloatArray,
        means: FloatArray,
        signs: FloatArray | None = None,
    ) -> Self:
        """Builds a reader from given vectors, such as a prompt difference.

        Directions are rescaled to unit norm, so a score is a projection in
        activation units whatever the scale of the input.

        Args:
            method: Name stored on the reader.
            layers: Layer indices, one per row of `directions`.
            directions: Vectors of shape (K, d), none of them zero.
            means: Vectors of shape (K, d) subtracted before projecting.
            signs: Orientation per layer, shape (K,), each -1 or +1.
                Defaults to +1 everywhere.

        Returns:
            The reader, holding float32 copies of the arrays.

        Raises:
            ValueError: If the shapes disagree, a direction is zero or a
                sign is not -1 or +1.
        """
        vectors, centers, orientation = _checked_arrays(
            layers, directions, means, signs
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms == 0.0).any():
            raise ValueError("directions must not contain a zero vector")
        return cls(
            method=method,
            layers=tuple(int(layer) for layer in layers),
            directions=(vectors / norms).astype(np.float32),
            means=centers.astype(np.float32),
            signs=orientation.astype(np.float32),
        )

    def save(self, path: str | Path) -> None:
        """Writes the reader to one `.npz` file at exactly `path`.

        Args:
            path: Destination file; no suffix is added.
        """
        with Path(path).open("wb") as file:
            np.savez(
                file,
                method=np.array(self.method),
                layers=np.array(self.layers, dtype=np.int64),
                directions=self.directions,
                means=self.means,
                signs=self.signs,
            )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Reads a reader written by `save`, with every array unchanged.

        Args:
            path: File written by `save`.

        Returns:
            The reader, equal to the one that was saved.

        Raises:
            ValueError: If the file lacks an array or the arrays disagree.
        """
        with np.load(Path(path), allow_pickle=False) as data:
            missing = [name for name in _FIELDS if name not in data.files]
            if missing:
                raise ValueError(f"{path} has no array named {missing[0]!r}")
            method = str(data["method"])
            layers = [int(layer) for layer in data["layers"]]
            arrays = [data[name] for name in _FIELDS[2:]]
        vectors, centers, orientation = _checked_arrays(layers, *arrays)
        return cls(
            method=method,
            layers=tuple(layers),
            directions=vectors.astype(np.float32),
            means=centers.astype(np.float32),
            signs=orientation.astype(np.float32),
        )

    def score(self, h: FloatArray) -> NDArray[np.float32]:
        """Scores activations along each layer's oriented direction.

        Args:
            h: Activations of shape (K, n, d), K = `len(self.layers)`.

        Returns:
            Scores of shape (K, n).

        Raises:
            ValueError: If `h` does not have shape (K, n, d).
        """
        return self.signs[:, None] * self.raw_score(h)

    def raw_score(self, h: FloatArray) -> NDArray[np.float32]:
        """Scores activations along each direction, ignoring the signs.

        Args:
            h: Activations of shape (K, n, d), K = `len(self.layers)`.

        Returns:
            Scores of shape (K, n).

        Raises:
            ValueError: If `h` does not have shape (K, n, d).
        """
        return self._project(h).astype(np.float32)

    def _project(self, h: FloatArray) -> NDArray[np.float64]:
        n_layers, width = self.directions.shape
        if h.ndim != 3 or h.shape[0] != n_layers or h.shape[2] != width:
            raise ValueError(
                f"h must have shape ({n_layers}, n, {width}), got {h.shape}"
            )
        scores = np.empty((n_layers, h.shape[1]), dtype=np.float64)
        for k in range(n_layers):
            centered = h[k].astype(np.float64) - self.means[k]
            scores[k] = centered @ self.directions[k].astype(np.float64)
        return scores


def orient(
    reader: RepReader,
    h: FloatArray,
    *,
    labels: NDArray[np.int64],
    pairs: NDArray[np.int64] | None = None,
    groups: NDArray[np.int64] | None = None,
) -> RepReader:
    """Sets each layer's sign with the rule of `RepReader.fit`.

    For readers built from directions that `RepReader.fit` does not find,
    such as LDA or a learned value head.

    Args:
        reader: The reader to orient; its signs are ignored.
        h: Activations of shape (K, n, d) that orient the reader.
        labels: Binary labels of shape (n,), 1 for the positive pole.
        pairs: Row index pairs of shape (m, 2).
        groups: Row groups of shape (g, k) with exactly one label-1 member
            each, padded with -1.

    Returns:
        A copy of the reader with the new signs.

    Raises:
        ValueError: If the shapes disagree, a group does not have exactly
            one label-1 member, the pairs are malformed, or the sign cannot
            be set because a class is missing.
    """
    projections = reader._project(h)
    if labels.shape != (projections.shape[1],):
        raise ValueError(
            f"labels must have shape ({projections.shape[1]},), "
            f"got {labels.shape}"
        )
    index = _orientation_groups(pairs, groups, labels)
    return replace(reader, signs=_orientation(projections, labels, index))


def _checked_arrays(
    layers: Sequence[int],
    directions: FloatArray,
    means: FloatArray,
    signs: FloatArray | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    vectors = np.array(directions, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError(
            f"directions must have shape (K, d), got {vectors.shape}"
        )
    n_layers = vectors.shape[0]
    centers = np.array(means, dtype=np.float64)
    if centers.shape != vectors.shape:
        raise ValueError(
            f"means must have shape {vectors.shape}, got {centers.shape}"
        )
    if len(layers) != n_layers:
        raise ValueError(
            f"layers has {len(layers)} entries for {n_layers} directions"
        )
    orientation = (
        np.ones(n_layers) if signs is None else np.array(signs, np.float64)
    )
    if (
        orientation.shape != (n_layers,)
        or not np.isin(orientation, (-1.0, 1.0)).all()
    ):
        raise ValueError("signs must have shape (K,) with values -1 or +1")
    return vectors, centers, orientation


def _layer_seed(seed: int, layer_position: int) -> int:
    sequence = np.random.SeedSequence([seed, layer_position])
    return int(sequence.generate_state(1)[0])


def _orientation_groups(
    pairs: NDArray[np.int64] | None,
    groups: NDArray[np.int64] | None,
    labels: NDArray[np.int64],
) -> GroupIndex | None:
    if groups is None and pairs is not None:
        groups = _pairs_with_distinct_labels(pairs, labels)
    if groups is None:
        positive = labels == 1
        if positive.all() or not positive.any():
            raise ValueError("labels must contain both classes")
        return None
    return parse_groups(groups, labels)


def _orientation(
    projections: NDArray[np.float64],
    labels: NDArray[np.int64],
    index: GroupIndex | None,
) -> NDArray[np.float32]:
    if index is None:
        positive = labels == 1
        return _sign_of(
            projections[:, positive].mean(axis=1)
            - projections[:, ~positive].mean(axis=1)
        )
    members = projections[:, index.members]
    highest = np.where(index.valid, members, -np.inf).max(axis=-1)
    lowest = np.where(index.valid, members, np.inf).min(axis=-1)
    own = members[:, np.arange(index.positive.shape[0]), index.positive]
    return _sign_of(
        (own == highest).mean(axis=1) - (own == lowest).mean(axis=1)
    )


def _pairs_with_distinct_labels(
    pairs: NDArray[np.int64], labels: NDArray[np.int64]
) -> NDArray[np.int64] | None:
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must have shape (m, 2), got {pairs.shape}")
    if pairs.size and (pairs.min() < 0 or pairs.max() >= labels.shape[0]):
        raise ValueError("pairs must index the rows of h")
    distinct = labels[pairs[:, 0]] != labels[pairs[:, 1]]
    return pairs[distinct] if distinct.any() else None


def _sign_of(gap: NDArray[np.float64]) -> NDArray[np.float32]:
    return np.where(gap < 0, -1.0, 1.0).astype(np.float32)
