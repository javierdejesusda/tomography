"""Checks that the figures read the encoding table of `viz/paper.py`."""

from pathlib import Path
import runpy
from typing import Any

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import pytest

from tomography.site import registry
from tomography.site.figs import _rows as rowkit
from tomography.site.manifest import Manifest
from tomography.viz import paper

_L = runpy.run_path(str(Path(__file__).with_name("_legends.py")))
RESULTS = Path(__file__).resolve().parents[1] / "results"
KEYS = (
    "teaser",
    "finders",
    "reading",
    "wording",
    "truth",
    "a_layers",
    "a_finders_gpt2",
    "a_finders_qwen3",
    "a_wording",
)
FILL_KEYS = ("teaser", "finders", "reading", "truth")


def _pattern(style: Any) -> Any:
    return Line2D([], [], ls=style)._unscaled_dash_pattern


POST = _pattern(paper.LINESTYLES["post_trained"])
KNOWN = {
    _pattern(paper.LINESTYLES[k])
    for k in ("solid", "post_trained", "reference", "trend")
}


def _flatten(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    return {m for v in value for m in _flatten(v)}


MEASURES = _flatten(list(paper.MEASURE_MARKERS.values()))
ALLOWED = MEASURES | set(paper.CATEGORY_MARKERS) | {paper.ROLE_KEY_MARKER}


@pytest.fixture(scope="module")
def built() -> dict[str, registry.FigureOutput]:
    manifest = Manifest(RESULTS)
    return {k: registry.build_figure(k, manifest) for k in KEYS}


def _lines(fig: Figure) -> list[Line2D]:
    found = [ln for ax in fig.axes for ln in ax.get_lines()]
    for leg in [*fig.legends, *(a.get_legend() for a in fig.axes)]:
        if leg is not None:
            found.extend(h for h in leg.legend_handles if isinstance(h, Line2D))
    return found


SEGMENT_KEYS = {"|", "_"}


def _markers(fig: Figure) -> set[str]:
    return {
        str(ln.get_marker())
        for ln in _lines(fig)
        if str(ln.get_marker()) not in ("None", "", " ", *SEGMENT_KEYS)
    }


@pytest.mark.parametrize("key", KEYS)
def test_every_marker_comes_from_the_encoding_table(
    built: dict[str, registry.FigureOutput], key: str
) -> None:
    extra = _markers(built[key].figure) - ALLOWED
    assert extra == set()


@pytest.mark.parametrize("key", KEYS)
def test_no_legend_symbol_means_two_things(
    built: dict[str, registry.FigureOutput], key: str
) -> None:
    assert _L["marker_clashes"](built[key].figure) == []


@pytest.mark.parametrize("key", FILL_KEYS)
def test_a_legend_that_uses_fill_carries_the_hexagon_role_key(
    built: dict[str, registry.FigureOutput], key: str
) -> None:
    fig = built[key].figure
    legends = [*fig.legends, *(a.get_legend() for a in fig.axes)]
    handles = [
        h
        for leg in legends
        if leg is not None
        for h in leg.legend_handles
        if isinstance(h, Line2D)
    ]
    hexes = [h for h in handles if h.get_marker() == paper.ROLE_KEY_MARKER]
    fills = {h.get_markerfacecolor() == "white" for h in hexes}
    assert fills == {True, False}


def test_the_paper_finder_is_a_diamond_and_the_logistic_a_circle(
    built: dict[str, registry.FigureOutput],
) -> None:
    for key in ("finders", "a_finders_gpt2", "a_finders_qwen3"):
        marks = _markers(built[key].figure)
        assert paper.MEASURE_MARKERS["paper_finder"] in marks
        assert paper.MEASURE_MARKERS["direction"] in marks


def test_the_probability_reader_of_figure_6_uses_the_answer_marker(
    built: dict[str, registry.FigureOutput],
) -> None:
    marks = _markers(built["truth"].figure)
    assert paper.MEASURE_MARKERS["answer_probability"] in marks
    assert paper.MEASURE_MARKERS["probe"] not in marks
    assert paper.MEASURE_MARKERS["paper_finder"] not in marks


@pytest.mark.parametrize("key", ["a_layers", "a_finders_qwen3"])
def test_post_trained_lines_use_the_one_style(
    built: dict[str, registry.FigureOutput], key: str
) -> None:
    styles = {ln._unscaled_dash_pattern for ln in _lines(built[key].figure)}
    assert POST in styles
    assert styles <= KNOWN | {(0, None)}


def test_the_gpt2_curves_have_no_post_trained_style(
    built: dict[str, registry.FigureOutput],
) -> None:
    fig = built["a_finders_gpt2"].figure
    styles = {ln._unscaled_dash_pattern for ln in _lines(fig)}
    assert POST not in styles


def test_figure_6_marks_the_statement_sets_with_the_set_markers(
    built: dict[str, registry.FigureOutput],
) -> None:
    marks = _markers(built["truth"].figure)
    assert set(paper.SET_MARKERS.values()) <= marks
    for measure in ("text", "popularity"):
        assert not _flatten(paper.MEASURE_MARKERS[measure]) & marks


@pytest.mark.parametrize("key", ["truth"])
def test_the_star_is_drawn_at_the_shared_size(
    built: dict[str, registry.FigureOutput], key: str
) -> None:
    star = paper.CATEGORY_MARKERS[0]
    sizes = {
        ln.get_markersize()
        for ln in _lines(built[key].figure)
        if ln.get_marker() == star
    }
    assert sizes
    assert min(sizes) >= rowkit.STAR_SIZE


@pytest.mark.parametrize("key", ["reading", "wording"])
def test_a_redesigned_figure_uses_at_most_six_shapes(
    built: dict[str, registry.FigureOutput], key: str
) -> None:
    assert len(_markers(built[key].figure)) <= 6


@pytest.mark.parametrize("key", ["reading", "wording"])
def test_figures_4_and_5_use_no_category_shapes(
    built: dict[str, registry.FigureOutput], key: str
) -> None:
    marks = _markers(built[key].figure)
    assert not marks & set(paper.CATEGORY_MARKERS)
    assert not marks & (MEASURES - {"o", "D", "x"})


def test_figure_4_draws_only_circles_and_the_role_key(
    built: dict[str, registry.FigureOutput],
) -> None:
    marks = _markers(built["reading"].figure)
    assert marks == {paper.MEASURE_MARKERS["direction"], paper.ROLE_KEY_MARKER}


def test_figure_5_draws_circles_a_diamond_and_the_role_free_rows(
    built: dict[str, registry.FigureOutput],
) -> None:
    marks = _markers(built["wording"].figure)
    assert marks == {
        paper.MEASURE_MARKERS["direction"],
        paper.MEASURE_MARKERS["paper_finder"],
    }
