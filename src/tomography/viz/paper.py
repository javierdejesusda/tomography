"""Print-ready figure style for the results site and the paper figures.

One style covers every figure: the serif face of the NeurIPS and ICLR body
text, 7 to 8.5 pt type, 0.5 pt axes, Okabe-Ito hues in a fixed order, grey
for random baselines, and a text width of 5.5 in. Rendering is deterministic:
text is converted to paths and the SVG and PDF headers carry no dates.
"""

from collections.abc import Iterator
import colorsys
import contextlib
import io
from typing import Any
from typing import cast

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Annotation

FULL_WIDTH = 5.5
HALF_WIDTH = 2.7
ROW_HEIGHT = 0.16

SERIES: tuple[str, ...] = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
NEUTRALS: tuple[str, ...] = ("#4d4d4d", "#888888")
RANDOM_BAND = "#d9d9d9"
CHANCE = "#7f7f7f"
INK = "#1a1a1a"

TICK = 7
LABEL = 8
LETTER = 8.5

RAMP_LIGHTNESS = (0.46, 0.2)

RC: dict[str, Any] = {
    "font.family": ["serif"],
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": TICK,
    "xtick.labelsize": TICK,
    "ytick.labelsize": TICK,
    "axes.labelsize": LABEL,
    "axes.titlesize": LABEL,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "legend.fontsize": TICK,
    "legend.frameon": False,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "grid.linewidth": 0.3,
    "grid.color": "#e6e6e6",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "path",
    "svg.hashsalt": "tomography",
    "pdf.fonttype": 42,
}

FAMILY_BASE = {"qwen3": SERIES[0], "gpt2": SERIES[1]}

NAMES = {
    "gpt2": "GPT-2 small",
    "gpt2-medium": "GPT-2 medium",
    "gpt2-large": "GPT-2 large",
    "gpt2-xl": "GPT-2 XL",
}


@contextlib.contextmanager
def paper_style() -> Iterator[None]:
    """Applies the paper rc parameters inside a `with` block.

    Yields:
        Nothing; the previous rc parameters return on exit.
    """
    with mpl.rc_context(cast("dict[Any, Any]", RC)):
        yield


def family(model: str) -> str:
    """Names the model family of a checkpoint key.

    Args:
        model: Checkpoint key such as `qwen3-4b-base`.

    Returns:
        `GPT-2` or `Qwen3`.
    """
    return "GPT-2" if model.startswith("gpt2") else "Qwen3"


def display_name(key: str) -> str:
    """Writes a checkpoint key, or a key plus a variant, in paper form.

    Args:
        key: A checkpoint key, optionally followed by a space and a variant.

    Returns:
        The paper name, or the input when it is not a checkpoint key.
    """
    model, _, variant = key.partition(" ")
    if model in NAMES:
        name = NAMES[model]
    elif model.startswith("qwen3-"):
        size, _, suffix = model.removeprefix("qwen3-").partition("-")
        name = f"Qwen3-{size.upper()}" + (
            f"-{suffix.title()}" if suffix else ""
        )
    else:
        return key
    if variant:
        variant = "few-shot" if variant == "fewshot" else variant
        return f"{name}, {variant}"
    return name


def size_ramp(family_key: str, count: int) -> list[str]:
    """Builds one hue per family with lightness ordered by size.

    Args:
        family_key: `qwen3` or `gpt2`.
        count: Number of sizes in the family.

    Returns:
        Hex colors from the smallest (lightest) to the largest (darkest).

    Raises:
        KeyError: If the family is unknown.
    """
    base = FAMILY_BASE[family_key]
    if count == 1:
        return [base]
    red, green, blue = (int(base[i : i + 2], 16) / 255 for i in (1, 3, 5))
    hue, _, sat = colorsys.rgb_to_hls(red, green, blue)
    hi, lo = RAMP_LIGHTNESS
    colors = []
    for i in range(count):
        light = hi + (lo - hi) * i / (count - 1)
        r, g, b = colorsys.hls_to_rgb(hue, light, sat)
        colors.append(
            f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"
        )
    return colors


def panel_letter(ax: Axes, letter: str) -> Annotation:
    """Writes the bold panel letter above the top left corner of an axes.

    Args:
        ax: The panel.
        letter: The letter without parentheses.

    Returns:
        The text artist.
    """
    return ax.annotate(
        f"({letter})",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=LETTER,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def svg_bytes(fig: Figure) -> bytes:
    """Renders a figure as deterministic SVG.

    Args:
        fig: The figure.

    Returns:
        The SVG document with text as paths and no date.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="svg", metadata={"Date": None})
    return buffer.getvalue()


def pdf_bytes(fig: Figure) -> bytes:
    """Renders a figure as deterministic vector PDF.

    Args:
        fig: The figure.

    Returns:
        The PDF document with embedded TrueType text and no creation date.
    """
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="pdf",
        metadata={"CreationDate": None, "Creator": "tomography"},
    )
    return buffer.getvalue()
