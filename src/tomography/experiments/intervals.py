"""Curves with given interval bands over a float axis.

`sweep.write_curves` fixes its band to a bootstrap over seeds and types the
x axis as integer layers. The P9 experiments plot rates by year, risk against
coverage and effects against steering coefficients, whose bands are Wilson
intervals, item bootstraps or min/max ranges computed by the caller. This
module writes such curves in the same `curves` results format, so
`sweep.plot_curves_file` draws them unchanged.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from tomography.experiments.base import RunDir


def write_interval_curves(
    run_dir: RunDir,
    x: Sequence[float],
    mean: NDArray[np.float64],
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    *,
    series: Sequence[str],
    stem: str = "curves",
    xlabel: str,
    ylabel: str,
    chance: float | None = None,
    n_layers: int | None = None,
) -> None:
    """Writes the `curves` arrays and sidecar with given interval bands.

    The arrays hold `x (K,)`, `mean (S, K)`, `lo (S, K)`, `hi (S, K)` and
    `per_seed (S, 1, K)`, a one-row copy of the mean that the plot reader
    expects and that draws as a single hairline under the mean line.

    Args:
        run_dir: The run to write into.
        x: The x value of each of the K points.
        mean: Series values of shape (S, K).
        lo: Lower band bounds of shape (S, K).
        hi: Upper band bounds of shape (S, K).
        series: The name of each series.
        stem: File stem of the pair (`<stem>.npz` and `<stem>.json`).
        xlabel: Label of the x axis.
        ylabel: Label of the y axis.
        chance: Level of the chance line, or None for no line.
        n_layers: Depth of the model, which fixes the x range to 0..n_layers.

    Raises:
        ValueError: If `mean`, `lo` and `hi` do not share one shape (S, K),
            or `series` does not have S names, or `x` does not have K values.
    """
    values = np.asarray(mean, dtype=np.float64)
    lower = np.asarray(lo, dtype=np.float64)
    upper = np.asarray(hi, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"mean must have shape (S, K), got {values.shape}")
    for name, band in (("lo", lower), ("hi", upper)):
        if band.shape != values.shape:
            raise ValueError(
                f"{name} must have the shape of mean {values.shape}, got "
                f"{band.shape}"
            )
    if len(series) != values.shape[0]:
        raise ValueError(
            f"got {len(series)} series names for {values.shape[0]} series"
        )
    if len(x) != values.shape[1]:
        raise ValueError(f"got {len(x)} x values for {values.shape[1]} columns")
    run_dir.write_arrays(
        stem,
        x=np.asarray(x, dtype=np.float64),
        mean=values,
        lo=lower,
        hi=upper,
        per_seed=values[:, None, :],
    )
    run_dir.write_json(
        stem,
        {
            "series": list(series),
            "xlabel": xlabel,
            "ylabel": ylabel,
            "chance": chance,
            "n_layers": n_layers,
        },
    )
