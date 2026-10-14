"""A per-layer linear reader that scores activations along fitted directions."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from typing import Self

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


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
            np.ones(n_layers)
            if signs is None
            else np.array(signs, dtype=np.float64)
        )
        if (
            orientation.shape != (n_layers,)
            or not np.isin(orientation, (-1.0, 1.0)).all()
        ):
            raise ValueError("signs must have shape (K,) with values -1 or +1")
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
        n_layers, width = self.directions.shape
        if h.ndim != 3 or h.shape[0] != n_layers or h.shape[2] != width:
            raise ValueError(
                f"h must have shape ({n_layers}, n, {width}), got {h.shape}"
            )
        scores = np.empty((n_layers, h.shape[1]), dtype=np.float32)
        for k in range(n_layers):
            centered = h[k].astype(np.float64) - self.means[k]
            scores[k] = centered @ self.directions[k].astype(np.float64)
        return scores
