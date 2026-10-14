"""Token scans: reading scores per token and layer.

A scan colors each token of one text by the reading score at each layer, on
a diverging scale where gray is zero and the positive pole (red) means the
concept is present.
"""

from collections.abc import Sequence
import math
from typing import Any

from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.text import Text
import numpy as np
from numpy.typing import NDArray

from tomography.viz import style

FloatArray = NDArray[np.floating[Any]]

MAX_LAYER_TICKS = 8
LONG_TEXT = 16
STRIP_FONT_SIZE = 8
STRIP_PAD = 0.15
STRIP_LINE_HEIGHT = 0.22


def _display(token: str) -> str:
    return token.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def _color_limit(visible: FloatArray, vmax: float | None) -> float:
    if vmax is None:
        return float(np.abs(visible).max(initial=0.0)) or 1.0
    if vmax <= 0:
        raise ValueError(f"vmax must be positive, got {vmax}")
    return vmax


def _measure(ax: Axes, text: str) -> float:
    probe = ax.text(0.0, 0.0, text, fontsize=STRIP_FONT_SIZE)
    width = float(probe.get_window_extent().width)
    probe.remove()
    return width


def _tick_rows(n_layers: int) -> list[int]:
    step = math.ceil(n_layers / MAX_LAYER_TICKS)
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
    if mask_first and n_tokens < 2:
        raise ValueError("mask_first would leave no token to show")

    mask = np.zeros(scores.shape, dtype=bool)
    mask[:, :1] = mask_first
    vmax = _color_limit(scores[~mask], vmax)
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


def plot_token_strip(
    scores: FloatArray,
    tokens: Sequence[str],
    *,
    mask_first: bool = False,
    vmax: float | None = None,
    width: float = style.TEXT_WIDTH,
    ax: Axes | None = None,
) -> Figure:
    """Highlights the tokens of one text by their reading score.

    Tokens run as wrapped text, each on a background from the same diverging
    scale as `plot_scan`; the text takes white or ink by its fill.

    Args:
        scores: Reading scores of shape (T,), positive meaning the concept
            is present.
        tokens: The T token strings; a leading space becomes a gap.
        mask_first: Whether to leave the first token unfilled and out of the
            color range.
        vmax: Largest absolute score on the color scale; None takes the
            largest absolute visible score.
        width: Figure width in inches when `ax` is None; the figure height
            follows the number of wrapped lines.
        ax: Axes to fill as given. A new styled figure is created when None.

    Returns:
        The figure holding the axes.

    Raises:
        ValueError: If `scores` is not 1-D, the token count differs, `vmax`
            is not positive, or masking leaves no token.
    """
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D (tokens,), got {scores.shape}")
    if len(tokens) != len(scores):
        raise ValueError(f"got {len(tokens)} tokens for {len(scores)} scores")
    if mask_first and len(tokens) < 2:
        raise ValueError("mask_first would leave no token to show")
    limit = _color_limit(scores[1:] if mask_first else scores, vmax)
    cmap = style.diverging_cmap()
    norm = Normalize(-limit, limit)

    owned = ax is None
    if ax is None:
        _, axes = style.new_figure(width=width)
        ax = axes[0, 0]
    fig = style.figure_of(ax)
    with style.style_context():
        ax.set_axis_off()
        fig.draw_without_rendering()
        ax_width = ax.get_window_extent().width
        pad = STRIP_PAD * STRIP_FONT_SIZE * fig.dpi / 72
        space = _measure(ax, " ")
        texts: list[Text] = []
        lead_gaps: list[bool] = []
        for i, token in enumerate(tokens):
            shown = _display(token)
            lead_gaps.append(shown.startswith(" "))
            shown = shown.lstrip(" ") or " "
            masked = mask_first and i == 0
            fill = "none" if masked else cmap(norm(scores[i]))
            texts.append(
                ax.text(
                    0.0,
                    0.5,
                    shown,
                    transform=ax.transAxes,
                    ha="left",
                    va="center",
                    fontsize=STRIP_FONT_SIZE,
                    color=(
                        style.INK_SECONDARY if masked else style.ink_on(fill)
                    ),
                    bbox={
                        "boxstyle": f"square,pad={STRIP_PAD}",
                        "facecolor": fill,
                        "edgecolor": "none",
                    },
                )
            )
        slots: list[tuple[int, float]] = []
        row, cursor = 0, 0.0
        for text, lead_gap in zip(texts, lead_gaps, strict=True):
            gap = space if lead_gap and cursor > 0 else 0.0
            extent = text.get_window_extent().width + 2 * pad
            if cursor > 0 and cursor + gap + extent > ax_width:
                row, cursor, gap = row + 1, 0.0, 0.0
            slots.append((row, cursor + gap + pad))
            cursor += gap + extent
        n_lines = row + 1
        for text, (line, x) in zip(texts, slots, strict=True):
            text.set_position((x / ax_width, 1 - (line + 0.5) / n_lines))
        if owned:
            fig.set_size_inches(width, n_lines * STRIP_LINE_HEIGHT + 0.1)
    return fig
