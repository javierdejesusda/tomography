"""Print-ready figure style for the results site and the paper figures.

`ENC` is the encoding dictionary of the page: every hue, fill and line style
has one meaning, and no figure picks a colour of its own.

One style covers every figure: the Computer Modern face of the NeurIPS and
ICLR body text, 8 to 9 pt type, 0.5 pt axes, Okabe-Ito hues in a fixed order,
grey for random baselines, and a text width of 5.5 in. Rendering is
deterministic: text is converted to paths and the SVG and PDF headers carry
no dates.
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
from matplotlib.text import Text
from matplotlib.transforms import ScaledTranslation

FULL_WIDTH = 5.5
HALF_WIDTH = 2.7
ROW_HEIGHT = 0.16

SERIES: tuple[str, ...] = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
NEUTRALS: tuple[str, ...] = ("#4d4d4d", "#888888")
RANDOM_BAND = "#d9d9d9"
CHANCE = "#7f7f7f"
INK = "#1a1a1a"
WHITE = "#ffffff"
TILE_LABEL = "tile_label"

TICK = 8
LABEL = 8
TITLE = 9
LETTER = 9

RAMP_LIGHTNESS = {"gpt2": (0.46, 0.2), "qwen3": (0.40, 0.2)}
REFERENCE_WIDTH = 0.5
TREND_WIDTH = 0.6
DATA_WIDTH = 1.0
WHISKER_WIDTH = 0.6

RC: dict[str, Any] = {
    "font.family": ["serif"],
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": TICK,
    "xtick.labelsize": TICK,
    "ytick.labelsize": TICK,
    "axes.labelsize": LABEL,
    "axes.titlesize": TITLE,
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

LINESTYLES: dict[str, Any] = {
    "solid": "-",
    "post_trained": (0, (3, 2)),
    "reference": (0, (1, 1.5)),
    "trend": (0, (4, 1.5, 1, 1.5)),
}

MEASURE_MARKERS: dict[str, Any] = {
    "direction": "o",
    "random": "x",
    "text": ("^", "v"),
    "paper_finder": "D",
    "popularity": "s",
    "answer_probability": "p",
    "probe": "P",
}
ROLE_KEY_MARKER = "h"
CATEGORY_MARKERS: tuple[str, ...] = ("*", "<", ">")
VARIANT_MARKERS: dict[str, str] = {
    "frame": CATEGORY_MARKERS[0],
    "wording": CATEGORY_MARKERS[1],
    "concept": CATEGORY_MARKERS[2],
}
SET_MARKERS: dict[str, str] = {
    "cities": CATEGORY_MARKERS[0],
    "translations": CATEGORY_MARKERS[1],
}
PROMPT_MARKERS: dict[str, str] = {
    "positive": CATEGORY_MARKERS[2],
    "negative": CATEGORY_MARKERS[1],
}
ITEM_SET_MARKERS: dict[str, str] = {
    "range": CATEGORY_MARKERS[0],
    "exact": CATEGORY_MARKERS[1],
    "people": CATEGORY_MARKERS[2],
}
ENCODING_RULES = """Encoding rules of every figure.
1. The checkpoint colour ramps (vermilion GPT-2, blue Qwen3) mark checkpoint
   identity only, never a measure.
2. One marker per measure family, site-wide (`MEASURE_MARKERS`): the direction
   under test is a circle, a random direction a grey x, text-only classifiers
   triangles, the paper's finder a diamond, and each other baseline
   (popularity, answer probability, supervised probe) its own shape in a
   non-ramp ink.
3. Filled markers mean GPT-2 and Qwen3 base, open ones post-trained; every
   legend that uses fill carries the hexagon role key (`ROLE_KEY_MARKER`).
4. Post-trained lines use one style (`LINESTYLES["post_trained"]`) everywhere.
5. A grey band has one meaning per figure and is keyed in its legend.
6. A categorical set that is not a measure (prompt variants, statement sets,
   item sets) uses `CATEGORY_MARKERS`, keyed in its figure, never a measure
   shape; shapes may repeat across figures because each figure keys them."""


def _build_enc() -> dict[str, Any]:
    """Builds the encoding dictionary after the ramp function exists."""
    return {
        "gpt2": tuple(size_ramp("gpt2", 4)),
        "qwen3": tuple(size_ramp("qwen3", 3)),
        "direction": INK,
        "random_band": RANDOM_BAND,
        "random_marker": "#888888",
        "reference": CHANCE,
        "text": "#009E73",
        "paper": "#CC79A7",
        "popularity": "#E69F00",
        "own_probability": "#56B4E9",
        "unused": "#F0E442",
        "secondary_text": "#4d4d4d",
    }


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
    hi, lo = RAMP_LIGHTNESS[family_key]
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
        rf"$\mathbf{{({letter})}}$",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=LETTER,
        ha="right",
        va="bottom",
    )


TITLE_GAP = 4.0


def panel_title(
    ax: Axes, letter: str, title: str, size: float | None = None
) -> tuple[Annotation, Text]:
    """Writes a bold panel letter and the left title of a panel on one baseline.

    The title is the axes title, shifted right so that it starts `TITLE_GAP`
    points after the letter; layout still reserves its height.

    Args:
        ax: The panel.
        letter: The letter without parentheses.
        title: The title; may be empty.
        size: Font size of both; the axes title size when None.

    Returns:
        The letter and the title.
    """
    mark = ax.annotate(
        rf"$\mathbf{{({letter})}}$",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, mpl.rcParams["axes.titlepad"]),
        textcoords="offset points",
        fontsize=size or ax.title.get_fontsize(),
        ha="left",
        va="baseline",
    )
    root = ax.get_figure(root=True)
    assert root is not None
    shift = mark.get_window_extent().width * 72.0 / root.dpi + TITLE_GAP
    name = ax.set_title(title, loc="left")
    if size is not None:
        name.set_fontsize(size)
    name.set_transform(
        ax.transAxes
        + cast(Any, ax).titleOffsetTrans
        + ScaledTranslation(shift / 72.0, 0.0, root.dpi_scale_trans)
    )
    return mark, name


def svg_bytes(fig: Figure) -> bytes:
    """Renders a figure as deterministic SVG.

    Args:
        fig: The figure.

    Returns:
        The SVG document with text as paths and no date.
    """
    buffer = io.BytesIO()
    with paper_style():
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
    with paper_style():
        fig.savefig(
            buffer,
            format="pdf",
            metadata={"CreationDate": None, "Creator": "tomography"},
        )
    return buffer.getvalue()


def _luminance(color: Any) -> float:
    linear = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        for c in mpl.colors.to_rgb(color)
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def tile_ink(rgba: Any) -> str:
    """Picks the ink for a number written on a heat-map tile.

    Args:
        rgba: Any matplotlib colour of the tile, such as a colormap value.

    Returns:
        `INK` or `WHITE`, whichever has the higher WCAG contrast with the
        tile. Mark the text with `TILE_LABEL` as its gid so that the palette
        guard accepts white.
    """
    lum = _luminance(rgba)
    against_white = (1.0 + 0.05) / (lum + 0.05)
    against_ink = (lum + 0.05) / (_luminance(INK) + 0.05)
    return WHITE if against_white >= against_ink else INK


ENC: dict[str, Any] = _build_enc()
