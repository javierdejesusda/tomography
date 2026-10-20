import io
import itertools

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


def test_tile_ink_is_white_on_dark_cividis_and_ink_on_light() -> None:
    cmap = mpl.colormaps["cividis"]
    assert paper.tile_ink(cmap(0.0)) == "#ffffff"
    assert paper.tile_ink(cmap(1.0)) == paper.INK


def test_tile_ink_has_at_least_the_contrast_of_the_other_choice() -> None:
    cmap = mpl.colormaps["cividis"]
    for value in [i / 20 for i in range(21)]:
        rgba = cmap(value)
        fill = "#" + "".join(f"{round(c * 255):02x}" for c in rgba[:3])
        chosen = paper.tile_ink(rgba)
        other = "#ffffff" if chosen == paper.INK else paper.INK

        def ratio(ink: str, fill: str = fill) -> float:
            hi, lo = sorted((_luminance(ink), _luminance(fill)), reverse=True)
            return (hi + 0.05) / (lo + 0.05)

        assert ratio(chosen) >= ratio(other) - 1e-6


def test_panel_title_puts_the_title_after_the_letter_on_one_baseline() -> None:
    with paper.paper_style():
        fig = Figure(figsize=(2.7, 1.5))
        ax = fig.subplots()
        letter, title = paper.panel_title(ax, "a", "GPT-2")
        gap = title.get_window_extent().x0 - letter.get_window_extent().x1
    assert letter.get_text() == r"$\mathbf{(a)}$"
    assert title.get_text() == "GPT-2"
    assert gap >= 2 * fig.dpi / 72
    assert letter.get_va() == title.get_va() == "baseline"
    assert ax.get_title(loc="left") == "GPT-2"
    top = ax.transAxes.transform((0.0, 1.0))[1]
    assert title.get_window_extent().y0 >= top
    assert title.get_transform().transform(title.get_position())[
        1
    ] == pytest.approx(top + 6.0 * fig.dpi / 72)
    assert title.get_fontsize() == ax.title.get_fontsize()


def test_panel_title_takes_a_smaller_size_for_dense_grids() -> None:
    with paper.paper_style():
        fig = Figure(figsize=(2.7, 1.5))
        ax = fig.subplots()
        letter, title = paper.panel_title(ax, "a", "dense", size=6.5)
    assert letter.get_fontsize() == title.get_fontsize() == 6.5


def test_one_marker_per_measure_family_and_the_role_key_is_its_own() -> None:
    markers = list(paper.MEASURE_MARKERS.values())
    flat = [
        m
        for item in markers
        for m in (item if isinstance(item, tuple) else (item,))
    ]
    assert len(flat) == len(set(flat))
    assert "h" not in flat
    assert paper.MEASURE_MARKERS["direction"] == "o"
    assert paper.MEASURE_MARKERS["random"] == "x"
    assert paper.MEASURE_MARKERS["paper_finder"] == "D"
    assert set(paper.MEASURE_MARKERS["text"]) == {"^", "v"}


def test_the_encoding_rules_are_stated_in_the_module() -> None:
    assert paper.ENCODING_RULES.count("\n") >= 4
    for word in ("identity", "Filled", "open", "circle", "band"):
        assert word in paper.ENCODING_RULES


def test_rule_three_keys_role_with_circles_not_the_hexagon() -> None:
    parts = paper.ENCODING_RULES.split("\n3. ")[1].split("\n4. ")
    rule = " ".join(parts[0].split())
    assert "CIRCLE" in rule
    assert "hexagon" not in paper.ENCODING_RULES


def test_category_markers_stay_clear_of_measures_and_the_role_key() -> None:
    measures = {
        m
        for item in paper.MEASURE_MARKERS.values()
        for m in (item if isinstance(item, tuple) else (item,))
    }
    assert len(paper.CATEGORY_MARKERS) == len(set(paper.CATEGORY_MARKERS))
    assert not measures & set(paper.CATEGORY_MARKERS)
    assert "h" not in paper.CATEGORY_MARKERS


def test_each_named_category_map_has_distinct_shapes_from_the_tuple() -> None:
    maps = {
        "VARIANT_MARKERS": ("frame", "wording", "concept"),
        "SET_MARKERS": ("cities", "translations"),
        "ITEM_SET_MARKERS": ("range", "exact", "people"),
    }
    for name, keys in maps.items():
        found = getattr(paper, name)
        assert tuple(found) == keys
        assert len(set(found.values())) == len(keys)
        assert set(found.values()) <= set(paper.CATEGORY_MARKERS)
    assert "6." in paper.ENCODING_RULES
    assert "CATEGORY_MARKERS" in paper.ENCODING_RULES


def test_prompting_arms_have_their_own_category_shapes() -> None:
    assert tuple(paper.PROMPT_MARKERS) == ("positive", "negative")
    values = set(paper.PROMPT_MARKERS.values())
    assert len(values) == 2
    assert values <= set(paper.CATEGORY_MARKERS)


def _lightness(color: str) -> float:
    import colorsys

    rgb = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    return colorsys.rgb_to_hls(*rgb)[1]


def test_adjacent_sizes_in_a_ramp_are_clearly_separable() -> None:
    for family, count in (("gpt2", 4), ("qwen3", 3)):
        ramp = paper.size_ramp(family, count)
        lights = [_lightness(c) for c in ramp]
        steps = [a - b for a, b in itertools.pairwise(lights)]
        assert min(steps) >= 0.08, (family, steps)
        assert all(_contrast(c) >= 3.0 for c in ramp)


def test_the_darkest_step_keeps_its_hue_against_black() -> None:
    for family, count in (("gpt2", 4), ("qwen3", 3)):
        darkest = paper.size_ramp(family, count)[-1]
        assert (_luminance(darkest) + 0.05) / 0.05 >= 1.8, family
        assert _lightness(darkest) >= 0.18, family


def test_markers_are_large_enough_and_edges_survive_print() -> None:
    assert 4.5 <= paper.MARKER_SIZE <= 5.0
    assert 0.8 <= paper.MARKER_EDGE <= 1.0


def test_random_is_one_grey_and_text_green_is_reserved() -> None:
    assert paper.ENC["random_marker"] == paper.RANDOM_GREY
    assert 0.0 < paper.RANDOM_BAND_ALPHA < 1.0
    assert paper.TEXT_GREEN == paper.ENC["text"] == "#009E73"
    others = [
        v for k, v in paper.ENC.items() if k != "text" and isinstance(v, str)
    ]
    assert paper.TEXT_GREEN not in others
    assert paper.TEXT_GREEN not in paper.ENC["gpt2"] + paper.ENC["qwen3"]
    for key in ("popularity", "own_probability"):
        assert _luminance(paper.ENC[key]) < 0.2
        assert paper.ENC[key] != paper.ENC["random_marker"]


def test_at_most_six_shapes_with_one_meaning_each() -> None:
    assert paper.SHAPES == {
        "direction": "o",
        "random": "x",
        "text": "^",
        "baseline": "s",
        "paper_finder": "D",
        "probe": "P",
    }
    assert len(set(paper.SHAPES.values())) == len(paper.SHAPES) <= 6


def test_one_sequential_and_one_diverging_map() -> None:
    import matplotlib as mpl

    assert paper.SEQUENTIAL_CMAP == "cividis"
    assert mpl.colormaps[paper.SEQUENTIAL_CMAP] is not None
    diverging = paper.diverging_cmap()
    assert tuple(diverging(0.5)[:3]) == pytest.approx(
        (0.9686, 0.9686, 0.9686), abs=0.01
    )
    ends = [diverging(0.0), diverging(1.0)]
    assert abs(_lum_rgba(ends[0]) - _lum_rgba(ends[1])) < 0.35
    assert _lum_rgba(diverging(0.5)) > _lum_rgba(ends[0])


def _lum_rgba(rgba: tuple[float, ...]) -> float:
    lin = [
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        for c in rgba[:3]
    ]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def test_type_and_line_rules_are_constants() -> None:
    assert paper.MIN_TEXT == 7.0
    assert paper.TEXT_GREY_DARKEST == "#4d4d4d"
    assert paper.DATA_WIDTH_RANGE == (1.0, 1.5)
    assert paper.DENSE_MARKER_RANGE == (3.0, 3.5)
    assert paper.REFERENCE_WIDTH == 0.5
    assert paper.WHISKER_WIDTH == 0.8


def test_the_glossary_rules_are_stated_and_no_map_is_called_deprecated() -> (
    None
):
    for word in ("chance", "grey", "plus", "cividis", "7 pt"):
        assert word in paper.ENCODING_RULES
    assert "deprecated" not in paper.ENCODING_RULES


def test_rule_four_applies_only_where_role_is_not_given_by_the_layout() -> None:
    rule = " ".join(
        paper.ENCODING_RULES.split("\n4. ")[1].split("\n5. ")[0].split()
    )
    assert "where role is not given by the panel or row block" in rule
    assert "dashed segment may carry a state" in rule
    assert "rule 8" in rule


def test_the_hexagon_role_key_is_gone() -> None:
    assert not hasattr(paper, "ROLE_KEY_MARKER")


def test_the_rules_state_the_constants_the_figures_use() -> None:
    rules = " ".join(paper.ENCODING_RULES.split())
    low, high = paper.DENSE_MARKER_RANGE
    assert f"{paper.MARKER_SIZE:g} pt markers" in rules
    assert f"{low:.1f} to {high:.1f} pt in dense panels" in rules
    assert f"{paper.MARKER_EDGE:g} pt edge" in rules
    low, high = paper.DATA_WIDTH_RANGE
    assert f"data lines {low:.1f} to {high:.1f} pt" in rules
    assert f"whiskers {paper.WHISKER_WIDTH:g} pt" in rules
    assert paper.MEASURE_MARKERS["answer_probability"] == "p"
    assert "pentagon" in rules
    assert paper.QUESTION_SET_MARKERS["known"] == "<"
    assert "left triangle" in rules


def test_the_dotted_line_rule_covers_zero_as_well_as_chance() -> None:
    rule = " ".join(
        paper.ENCODING_RULES.split("\n5. ")[1].split("\n6. ")[0].split()
    )
    assert "chance" in rule
    assert "zero" in rule
    assert "always means chance" not in rule
