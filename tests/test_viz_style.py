"""Tests for the shared figure style, color tokens and save helper."""

import itertools
from pathlib import Path
import struct

import matplotlib as mpl
from matplotlib.colors import Colormap
from matplotlib.colors import to_hex
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
import numpy as np
import pytest

from tomography.viz import style

SRGB_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
LMS_TO_LAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)


def oklab(color: str) -> np.ndarray:
    rgb = np.array(to_rgb(color))
    linear = np.where(
        rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4
    )
    return LMS_TO_LAB @ np.cbrt(SRGB_TO_LMS @ linear)


def test_categorical_slots_keep_the_documented_order() -> None:
    assert style.CATEGORICAL == (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    )


def test_adjacent_categorical_slots_differ_for_normal_vision() -> None:
    labs = [oklab(color) for color in style.CATEGORICAL]

    distances = [
        100 * np.linalg.norm(a - b) for a, b in itertools.pairwise(labs)
    ]

    assert min(distances) >= 15


@pytest.mark.parametrize("n", [1, 3, 8], ids=["one", "three", "eight"])
def test_categorical_colors_are_the_first_slots_in_order(n: int) -> None:
    assert style.categorical_colors(n) == list(style.CATEGORICAL[:n])


def test_categorical_colors_refuse_a_ninth_series() -> None:
    with pytest.raises(ValueError, match="eight"):
        style.categorical_colors(9)


def sample_cmap(cmap: Colormap, n: int = 257) -> list[str]:
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def test_sequential_cmap_runs_one_hue_from_light_to_dark() -> None:
    colors = sample_cmap(style.sequential_cmap())

    lightness = [oklab(color)[0] for color in colors]

    assert colors[0] == "#cde2fb"
    assert colors[-1] == "#0d366b"
    assert all(a >= b for a, b in itertools.pairwise(lightness))


def test_diverging_cmap_has_a_gray_midpoint() -> None:
    cmap = style.diverging_cmap()

    middle = oklab(to_hex(cmap(0.5)))

    assert np.hypot(middle[1], middle[2]) < 0.01


def test_diverging_cmap_puts_blue_below_and_red_above() -> None:
    cmap = style.diverging_cmap()

    low, high = to_rgb(cmap(0.0)), to_rgb(cmap(1.0))

    assert low[2] > low[0]
    assert high[0] > high[2]


def test_diverging_cmap_arms_are_equally_light() -> None:
    colors = sample_cmap(style.diverging_cmap())
    lightness = np.array([oklab(color)[0] for color in colors])

    assert np.abs(lightness - lightness[::-1]).max() < 0.03


def test_style_context_restores_the_previous_rc_params() -> None:
    before = mpl.rcParams["font.size"]

    with style.style_context():
        inside = mpl.rcParams["font.size"]

    assert inside != before
    assert mpl.rcParams["font.size"] == before


def test_new_figure_returns_a_grid_of_axes() -> None:
    fig, axes = style.new_figure(2, 3)

    assert isinstance(fig, Figure)
    assert axes.shape == (2, 3)
    assert all(ax.figure is fig for ax in axes.flat)


def test_new_figure_sizes_the_figure_from_the_panel_aspect() -> None:
    fig, _ = style.new_figure(2, 4, width=7.0, aspect=0.5)

    assert fig.get_size_inches() == pytest.approx((7.0, 7.0 * 0.5 * 2 / 4))


def test_new_figure_draws_recessive_axes() -> None:
    _, axes = style.new_figure()

    ax = axes[0, 0]

    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()


def test_lines_keep_the_style_whatever_the_ambient_rc() -> None:
    ambient = {"lines.linewidth": 4.0, "lines.markersize": 12.0}
    with mpl.rc_context(ambient), style.style_context():
        _, axes = style.new_figure()
        (line,) = axes[0, 0].plot([0, 1], [0, 1], marker="o")

    assert to_hex(line.get_color()) == style.CATEGORICAL[0]
    assert line.get_linewidth() == pytest.approx(2 * style.PX)
    assert line.get_markersize() == pytest.approx(8 * style.PX)
    assert line.get_solid_capstyle() == "round"


def sample_figure() -> Figure:
    fig, axes = style.new_figure()
    axes[0, 0].plot([0, 1, 2], [0, 1, 0])
    axes[0, 0].set_xlabel("Layer")
    return fig


def test_save_figure_writes_pdf_png_and_svg(tmp_path: Path) -> None:
    out_dir = tmp_path / "figures"

    paths = style.save_figure(sample_figure(), out_dir, "curve")

    assert [path.name for path in paths] == [
        "curve.pdf",
        "curve.png",
        "curve.svg",
    ]
    assert all(path.parent == out_dir for path in paths)
    assert all(path.stat().st_size > 0 for path in paths)


@pytest.mark.parametrize(
    ("extension", "signature"),
    [("pdf", b"%PDF"), ("png", b"\x89PNG"), ("svg", b"<?xml")],
    ids=["pdf", "png", "svg"],
)
def test_saved_files_have_the_format_signature(
    tmp_path: Path, extension: str, signature: bytes
) -> None:
    style.save_figure(sample_figure(), tmp_path, "curve", formats=(extension,))

    data = (tmp_path / f"curve.{extension}").read_bytes()

    assert data.startswith(signature)


def test_png_is_saved_at_the_requested_resolution(tmp_path: Path) -> None:
    fig = sample_figure()

    style.save_figure(fig, tmp_path, "curve", dpi=200, formats=("png",))

    header = (tmp_path / "curve.png").read_bytes()[16:24]
    width, height = struct.unpack(">II", header)
    fig_width, fig_height = fig.get_size_inches()
    assert (width, height) == (round(fig_width * 200), round(fig_height * 200))


def test_save_figure_honors_a_format_subset(tmp_path: Path) -> None:
    paths = style.save_figure(
        sample_figure(), tmp_path, "curve", formats=("png",)
    )

    assert [path.name for path in paths] == ["curve.png"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["curve.png"]


def test_save_figure_rejects_an_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tiff"):
        style.save_figure(sample_figure(), tmp_path, "curve", formats=("tiff",))


@pytest.mark.parametrize("extension", ["pdf", "png", "svg"])
def test_saving_twice_gives_identical_bytes(
    tmp_path: Path, extension: str
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    formats = (extension,)
    style.save_figure(sample_figure(), first, "curve", formats=formats)
    style.save_figure(sample_figure(), second, "curve", formats=formats)

    assert (first / f"curve.{extension}").read_bytes() == (
        second / f"curve.{extension}"
    ).read_bytes()


@pytest.mark.parametrize(
    ("extension", "marker"),
    [
        ("pdf", b"/CreationDate"),
        ("png", b"Software"),
        ("svg", b"<dc:date>"),
    ],
    ids=["pdf", "png", "svg"],
)
def test_saved_files_carry_no_timestamp_or_tool_stamp(
    tmp_path: Path, extension: str, marker: bytes
) -> None:
    style.save_figure(sample_figure(), tmp_path, "curve", formats=(extension,))

    data = (tmp_path / f"curve.{extension}").read_bytes()

    assert marker not in data


def test_svg_uses_unix_line_endings(tmp_path: Path) -> None:
    style.save_figure(sample_figure(), tmp_path, "curve", formats=("svg",))

    assert b"\r\n" not in (tmp_path / "curve.svg").read_bytes()


def test_figure_of_returns_the_figure_that_holds_the_axes() -> None:
    fig, axes = style.new_figure(1, 2)

    assert style.figure_of(axes[0, 1]) is fig


@pytest.mark.parametrize(
    ("fill", "expected"),
    [
        ("#0d366b", style.SURFACE),
        ("#762221", style.SURFACE),
        ("#cde2fb", style.INK),
        ("#f0efec", style.INK),
    ],
    ids=["dark-blue", "dark-red", "light-blue", "neutral"],
)
def test_ink_on_picks_the_text_color_with_more_contrast(
    fill: str, expected: str
) -> None:
    assert style.ink_on(fill) == expected
