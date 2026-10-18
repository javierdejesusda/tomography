"""Shared axes, styles and captions for the paper figures.

Layer axes use relative depth (layer over layer count) and size axes use the
log of the parameter count, so every figure reads the same way. Base and
post-trained checkpoints are drawn apart: filled markers and solid lines for
GPT-2 and the base models, open markers and dashed lines for post-trained.
"""

from collections.abc import Sequence
import dataclasses

from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
import numpy as np

from tomography.viz import paper

GPT2_SIZES = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
QWEN3_SIZES = ("0.6b", "1.7b", "4b")
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "paper_blue", ["#f7fbff", paper.SERIES[0]]
)


@dataclasses.dataclass(frozen=True)
class ModelStyle:
    """How one checkpoint is drawn.

    Attributes:
        family: `GPT-2` or `Qwen3`.
        color: Hex color, one hue per family and darker for larger sizes.
        linestyle: Solid for base and GPT-2, dashed for post-trained.
        filled: Filled markers for base and GPT-2, open for post-trained.
        marker: Matplotlib marker; circle for GPT-2, square for Qwen3.
    """

    family: str
    color: str
    linestyle: str
    filled: bool
    marker: str


def relative_depth(layers: Sequence[float], count: int) -> list[float]:
    """Converts layer indices to relative depth.

    Args:
        layers: Layer indices, 1 to `count`.
        count: Number of layers of the model.

    Returns:
        Layer divided by the layer count.

    Raises:
        ValueError: If the layer count is not positive.
    """
    if count <= 0:
        raise ValueError("layer count must be positive")
    return [layer / count for layer in layers]


def model_style(key: str) -> ModelStyle:
    """Gives the fixed look of a checkpoint.

    Args:
        key: Checkpoint key such as `gpt2-xl` or `qwen3-4b-base`.

    Returns:
        The style.

    Raises:
        KeyError: If the family or size is unknown.
    """
    if key.startswith("gpt2"):
        ramp = paper.size_ramp("gpt2", len(GPT2_SIZES))
        return ModelStyle("GPT-2", ramp[GPT2_SIZES.index(key)], "-", True, "o")
    size = key.removeprefix("qwen3-").removesuffix("-base")
    ramp = paper.size_ramp("qwen3", len(QWEN3_SIZES))
    post = not key.endswith("-base")
    return ModelStyle(
        "Qwen3",
        ramp[QWEN3_SIZES.index(size)],
        "--" if post else "-",
        not post,
        "s",
    )


def depth_axis(ax: Axes) -> None:
    """Sets an x axis for relative depth from 0 to 1.

    Args:
        ax: The axes.
    """
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("relative depth")


def _size_text(value: float, _pos: int) -> str:
    if value >= 1e9:
        return f"{value / 1e9:g}B"
    return f"{value / 1e6:g}M"


def size_axis(
    ax: Axes,
    counts: Sequence[float],
    ticks: Sequence[float] = (1e8, 3e8, 1e9, 3e9),
) -> None:
    """Sets a logarithmic x axis for parameter counts.

    Args:
        ax: The axes.
        counts: The counts that must fit on the axis.
        ticks: Where to put labelled ticks.
    """
    ax.set_xscale("log")
    ax.set_xlim(min(counts) / 1.6, max(counts) * 1.6)
    ax.xaxis.set_major_locator(ticker.FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_size_text))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("parameters")


def layer_curve(
    ax: Axes,
    depth: Sequence[float],
    mean: Sequence[float],
    lo: Sequence[float],
    hi: Sequence[float],
    model: str,
) -> None:
    """Draws one model's curve over relative depth with its interval band.

    Args:
        ax: The axes.
        depth: Relative depth of each point.
        mean: Value at each depth.
        lo: Lower end of the interval.
        hi: Upper end of the interval.
        model: Checkpoint key, which fixes color and line style.
    """
    style = model_style(model)
    ax.plot(
        depth,
        mean,
        color=style.color,
        ls=style.linestyle,
        lw=1.0,
        zorder=3,
    )
    ax.fill_between(
        depth, lo, hi, color=style.color, alpha=0.18, lw=0, zorder=2
    )


def heatmap(
    ax: Axes, matrix: np.ndarray, vmin: float, vmax: float
) -> AxesImage:
    """Draws a matrix on a sequential scale with the diagonal outlined.

    No numbers are written in the cells.

    Args:
        ax: The axes.
        matrix: Square matrix of values.
        vmin: Value of the lightest color.
        vmax: Value of the darkest color.

    Returns:
        The image, for a color bar.
    """
    image = ax.imshow(
        matrix, cmap=SEQUENTIAL, vmin=vmin, vmax=vmax, interpolation="nearest"
    )
    for i in range(min(matrix.shape)):
        ax.add_patch(
            Rectangle(
                (i - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                ec=paper.INK,
                lw=0.6,
                zorder=3,
            )
        )
    return image


def caption(claim: str, detail: str) -> str:
    """Builds a caption: the claim sentence, then what is plotted.

    Args:
        claim: One sentence with the finding; it must end with a period.
        detail: What is plotted, baselines and what each interval covers.

    Returns:
        The caption text.

    Raises:
        ValueError: If the claim does not end with a period.
    """
    if not claim.endswith("."):
        raise ValueError("a claim is one sentence ending with a period")
    return f"{claim} {detail}".strip()
