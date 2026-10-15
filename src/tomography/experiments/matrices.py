"""Writer of labeled matrices in the `matrix` results format.

A matrix is stored as a `<stem>.npz` holding `values` of shape (R, C) and a
`<stem>.json` sidecar with the row labels, the column labels and the title.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Literal
from typing import cast

import numpy as np
from numpy.typing import NDArray

from tomography.experiments.base import RunDir
from tomography.viz.geometry import plot_matrix
from tomography.viz.style import save_figure
from tomography.viz.style import style_context


def write_matrix(
    run_dir: RunDir,
    stem: str,
    values: NDArray[np.float64],
    *,
    rows: Sequence[str],
    cols: Sequence[str],
    title: str,
) -> None:
    """Writes the arrays and sidecar of one labeled matrix.

    Args:
        run_dir: The run to write into.
        stem: File stem of the pair (`<stem>.npz` and `<stem>.json`).
        values: Cell values of shape (R, C); NaN marks a blank cell.
        rows: The R row labels, top to bottom.
        cols: The C column labels, left to right.
        title: Title of the figure.

    Raises:
        ValueError: If `values` is not 2-D or the labels do not match its
            shape.
    """
    if values.ndim != 2:
        raise ValueError(
            f"values must be 2-D (rows, cols), got shape {values.shape}"
        )
    n_rows, n_cols = values.shape
    if len(rows) != n_rows:
        raise ValueError(f"got {len(rows)} rows for {n_rows} value rows")
    if len(cols) != n_cols:
        raise ValueError(f"got {len(cols)} cols for {n_cols} value columns")
    run_dir.write_arrays(stem, values=values)
    run_dir.write_json(
        stem, {"rows": list(rows), "cols": list(cols), "title": title}
    )


def plot_matrix_file(
    run_dir: RunDir,
    out_dir: Path,
    *,
    stem: str,
    kind: Literal["sequential", "diverging"] = "sequential",
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[Path]:
    """Draws a matrix written by `write_matrix` and saves it.

    Args:
        run_dir: The run holding the pair.
        out_dir: Folder of the figures.
        stem: File stem of the pair, also the name of the figures.
        kind: "sequential" for magnitudes or "diverging" for signed values.
        vmin: Lower end of the color range; None takes the data minimum.
        vmax: Upper end of the color range; None takes the data maximum.

    Returns:
        The PDF, PNG and SVG paths.
    """
    values = run_dir.read_arrays(stem)["values"]
    sidecar = cast(dict[str, Any], run_dir.read_json(stem))
    with style_context():
        fig = plot_matrix(
            values,
            rows=sidecar["rows"],
            cols=sidecar["cols"],
            title=sidecar["title"],
            kind=kind,
            vmin=vmin,
            vmax=vmax,
        )
    return save_figure(fig, out_dir, stem)
