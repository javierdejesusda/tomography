import io

import matplotlib as mpl
from matplotlib.figure import Figure
import pytest

from tomography.viz import paper


def _luminance(color: str) -> float:
    rgb = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb
    ]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(color: str) -> float:
    return 1.05 / (_luminance(color) + 0.05)


def test_rc_matches_the_neurips_figure_standard() -> None:
    with paper.paper_style():
        rc = mpl.rcParams
        assert rc["font.family"] == ["serif"]
        assert rc["font.serif"][0] == "cmr10"
        assert rc["mathtext.fontset"] == "cm"
        assert rc["axes.formatter.use_mathtext"] is True
        assert rc["xtick.labelsize"] == 8
        assert rc["ytick.labelsize"] == 8
        assert rc["axes.labelsize"] == 8
        assert rc["axes.titlesize"] == 9
        assert rc["legend.fontsize"] == 8
        assert rc["axes.linewidth"] == 0.5
        assert rc["axes.spines.top"] is False
        assert rc["axes.spines.right"] is False
        assert rc["xtick.direction"] == "out"
        assert rc["xtick.major.size"] == 2.5
        assert rc["xtick.major.width"] == 0.5
        assert rc["lines.linewidth"] == 1.0
        assert rc["lines.markersize"] == 3.5
        assert rc["axes.grid"] is False
        assert rc["svg.fonttype"] == "path"
        assert rc["pdf.fonttype"] == 42
        assert rc["svg.hashsalt"]


def test_style_does_not_leak_outside_the_context() -> None:
    before = mpl.rcParams["axes.linewidth"]
    with paper.paper_style():
        pass
    assert mpl.rcParams["axes.linewidth"] == before


def test_series_colors_are_okabe_ito_in_fixed_order() -> None:
    assert paper.SERIES[:4] == ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for color in paper.SERIES[:3]:
        assert _contrast(color) >= 3.0


def test_neutral_series_are_distinguishable_greys_with_contrast() -> None:
    assert len(set(paper.NEUTRALS)) == len(paper.NEUTRALS)
    for color in paper.NEUTRALS:
        assert _contrast(color) >= 3.0


def test_size_ramps_get_darker_with_size_and_keep_contrast() -> None:
    for family in ("qwen3", "gpt2"):
        ramp = paper.size_ramp(family, 4)
        lum = [_luminance(c) for c in ramp]
        assert lum == sorted(lum, reverse=True)
        assert len(set(ramp)) == 4
        assert all(_contrast(c) >= 3.0 for c in ramp)
    assert paper.size_ramp("qwen3", 1) == [paper.SERIES[0]]


def test_model_names_use_the_paper_form() -> None:
    names = {
        "gpt2": "GPT-2 small",
        "gpt2-xl": "GPT-2 XL",
        "qwen3-0.6b-base": "Qwen3-0.6B-Base",
        "qwen3-4b": "Qwen3-4B",
        "qwen3-1.7b fewshot": "Qwen3-1.7B, few-shot",
        "qwen3-1.7b concept": "Qwen3-1.7B, concept",
    }
    for key, name in names.items():
        assert paper.display_name(key) == name
    assert paper.display_name("something else") == "something else"
    assert paper.family("gpt2-large") == "GPT-2"
    assert paper.family("qwen3-4b-base") == "Qwen3"


def test_panel_letter_sits_at_the_top_left_outside_the_axes() -> None:
    with paper.paper_style():
        fig = Figure(figsize=(2.7, 1.5))
        ax = fig.subplots()
        text = paper.panel_letter(ax, "b")
    assert text.get_text() == r"$\mathbf{(b)}$"
    assert text.get_fontsize() == 9
    assert text.get_position()[1] >= 1.0


def test_svg_and_pdf_are_byte_deterministic() -> None:
    def build() -> Figure:
        with paper.paper_style():
            fig = Figure(figsize=(2.7, 1.5))
            ax = fig.subplots()
            ax.plot([0, 1, 2], [0.4, 0.8, 0.6])
            ax.set_xlabel("coefficient")
            return fig

    first, second = build(), build()
    with paper.paper_style():
        assert paper.svg_bytes(first) == paper.svg_bytes(second)
        assert paper.pdf_bytes(first) == paper.pdf_bytes(second)
        svg = paper.svg_bytes(first).decode("utf-8")
    assert "<dc:date>" not in svg
    assert "<text" not in svg


def test_saved_figure_has_its_natural_size_in_inches() -> None:
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, 2.0))
        fig.subplots()
        svg = paper.svg_bytes(fig).decode("utf-8")
    assert 'width="396pt"' in svg
    assert io.BytesIO(paper.pdf_bytes(fig)).read(4) == b"%PDF"


def test_unknown_family_ramp_is_rejected() -> None:
    with pytest.raises(KeyError):
        paper.size_ramp("llama", 2)


def test_every_text_artist_resolves_to_a_computer_modern_font() -> None:
    from matplotlib import font_manager
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.text import Text

    with paper.paper_style():
        fig = Figure(figsize=(2.7, 1.5))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.plot([-1, 0, 1], [0.4, 0.6, 0.5], label="series")
        ax.set_title("GPT-2 small, Qwen3-0.6B-Base")
        ax.set_xlabel("relative depth")
        ax.legend()
        paper.panel_letter(ax, "a")
        fig.canvas.draw()
        texts = [t for t in fig.findobj(Text) if t.get_text()]
        assert texts
        for text in texts:
            path = font_manager.findfont(text.get_fontproperties())
            name = path.replace("\\", "/").rsplit("/", 1)[-1]
            assert name.startswith("cm"), (text.get_text(), name)
