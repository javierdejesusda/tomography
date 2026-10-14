"""Token scans: reading scores per token and layer.

A scan colors each token of one text by the reading score at each layer, on
a diverging scale where gray is zero and the positive pole (red) means the
concept is present.
"""

from collections.abc import Sequence
from typing import Any

from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from tomography.viz import style

FloatArray = NDArray[np.floating[Any]]

MAX_LAYER_TICKS = 8
LONG_TEXT = 16


def _display(token: str) -> str:
    return token.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def _tick_rows(n_layers: int) -> list[int]:
    step = -(-n_layers // MAX_LAYER_TICKS)
    rows = list(range(0, n_layers, step))
    if rows[-1] != n_layers - 1:
        if 2 * (n_layers - 1 - rows[-1]) >= step:
            rows.append(n_layers - 1)
        else:
            rows[-1] = n_layers - 1
    return rows


def plot_scan(
    scores: FloatArray,
    layers: Sequence[int] | NDArray[np.int64],
    *,
    tokens: Sequence[str],
    mask_first: bool = False,
    vmax: float | None = None,
    cbar_label: str = "Reading score",
    colorbar: bool = True,
    ax: Axes | None = None,
) -> Figure:
    """Plots a token by layer heatmap of reading scores.

    Layer rows run upward from the lowest layer. The color range is
    symmetric around zero so that the gray midpoint always means no signal.

    Args:
        scores: Reading scores of shape (K, T), signed so that positive means
            the concept is present.
        layers: The K layer indices, in row order.
        tokens: The T token strings, in column order.
        mask_first: Whether to blank the first token column (the BOS or the
            first-token artifact) and leave it out of the color range.
        vmax: Largest absolute score on the color scale; None takes the
            largest absolute visible score.
        cbar_label: Label of the color bar.
        colorbar: Whether to add a color bar beside the heatmap.
        ax: Axes to draw on. A new styled figure is created when None.

    Returns:
        The figure holding the axes.

    Raises:
        ValueError: If the shapes of `scores`, `layers` and `tokens` disagree,
            `vmax` is not positive, or masking leaves no token.
    """
    if scores.ndim != 2:
        raise ValueError(
            f"scores must be 2-D (layers, tokens), got shape {scores.shape}"
        )
    n_layers, n_tokens = scores.shape
    if len(layers) != n_layers:
        raise ValueError(f"got {len(layers)} layers for {n_layers} score rows")
    if len(tokens) != n_tokens:
        raise ValueError(
            f"got {len(tokens)} tokens for {n_tokens} score columns"
        )
    if vmax is not None and vmax <= 0:
        raise ValueError(f"vmax must be positive, got {vmax}")
    if mask_first and n_tokens < 2:
        raise ValueError("mask_first would leave no token to show")

    mask = np.zeros(scores.shape, dtype=bool)
    mask[:, :1] = mask_first
    if vmax is None:
        vmax = float(np.abs(scores[~mask]).max()) or 1.0
    cmap = style.diverging_cmap().with_extremes(bad=style.SURFACE)

    if ax is None:
        width = style.TEXT_WIDTH if n_tokens > LONG_TEXT else style.COLUMN_WIDTH
        _, axes = style.new_figure(width=width, aspect=0.6)
        ax = axes[0, 0]
    with style.style_context():
        image = ax.imshow(
            np.ma.masked_array(scores, mask),
            cmap=cmap,
            norm=Normalize(-vmax, vmax),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        rows = _tick_rows(n_layers)
        ax.set_yticks(rows, [str(int(layers[i])) for i in rows])
        ax.set_xticks(
            range(n_tokens),
            [_display(token) for token in tokens],
            rotation=90,
            fontsize=6,
        )
        ax.set_ylabel("Layer")
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig = style.figure_of(ax)
        if colorbar:
            bar = fig.colorbar(image, ax=ax, label=cbar_label, pad=0.02)
            bar.outline.set_visible(False)
            bar.ax.tick_params(length=0)
    return fig
