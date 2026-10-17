"""Helpers shared by the steering experiments and their random controls."""

from collections.abc import Callable
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from tomography.reading.reader import RepReader


def random_readers(
    layers: Sequence[int], dim: int, n: int, *, seed: int
) -> list[RepReader]:
    """Draws the random readers that control the vector arms.

    Args:
        layers: Layers of every reader; one direction row each.
        dim: Width of the residual stream.
        n: Number of readers.
        seed: Seed of the one generator that draws them in order.

    Returns:
        `n` readers of unit directions with zero means.
    """
    rng = np.random.default_rng(seed)
    zeros = np.zeros((len(layers), dim))
    return [
        RepReader.from_arrays(
            "random", layers, rng.standard_normal((len(layers), dim)), zeros
        )
        for _ in range(n)
    ]


def max_abs(values: NDArray[np.float64]) -> float:
    """Gives the largest absolute finite value; NaN when none is finite.

    Args:
        values: An array of any shape.

    Returns:
        The largest finite magnitude, or NaN.
    """
    finite = np.abs(values[np.isfinite(values)])
    return float(finite.max()) if finite.size else float("nan")


def recording[T](
    make: Callable[[float], T], built: dict[float, T]
) -> Callable[[float], T]:
    """Wraps an arm factory so that the arm of each coefficient is kept.

    Args:
        make: Builds the arm of a coefficient.
        built: Filled in place, coefficient to the arm built for it.

    Returns:
        A factory that calls `make`, stores the result in `built` and
        returns it.
    """

    def record(coeff: float) -> T:
        built[coeff] = make(coeff)
        return built[coeff]

    return record
