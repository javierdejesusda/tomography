from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.markers import MarkerStyle
import numpy as np
import pytest

from tomography.site import stats
from tomography.site.figs import _rows
from tomography.viz import axes
from tomography.viz import paper


def test_join_and_writes_a_list_phrase() -> None:
    assert _rows.join_and([]) == ""
    assert _rows.join_and(["a"]) == "a"
    assert _rows.join_and(["a", "b"]) == "a and b"
    assert _rows.join_and(["a", "b", "c"]) == "a, b and c"


def test_number_uses_the_true_minus_and_never_a_negative_zero() -> None:
    assert _rows.number(-0.5) == f"{axes.MINUS}0.500"
    assert _rows.number(-0.0004) == "0.000"
    assert _rows.number(0.1234, 2) == "0.12"


def test_marker_fill_follows_the_role() -> None:
    for model in ("gpt2-xl", "qwen3-4b-base"):
        style = _rows.fill(model, "#123456")
        assert style["markerfacecolor"] == "#123456"
        assert style["markeredgecolor"] == "#123456"
    post = _rows.fill("qwen3-4b", "#123456")
    assert post["markerfacecolor"] == "white"
    assert post["markeredgecolor"] == "#123456"


def test_new_figure_is_page_wide_with_constrained_layout() -> None:
    fig = _rows.new_figure(3.0)
    assert tuple(fig.get_size_inches()) == (paper.FULL_WIDTH, 3.0)
    assert isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine)


def test_series_handle_is_filled_or_open() -> None:
    filled = _rows.series_handle("a", "o", "#123456")
    opened = _rows.series_handle("b", "o", "#123456", filled=False)
    assert filled.get_markerfacecolor() == "#123456"
    assert opened.get_markerfacecolor() == "white"
    assert opened.get_markeredgecolor() == "#123456"


def test_row_axis_puts_the_first_checkpoint_at_the_top() -> None:
    fig = _rows.new_figure(2.0)
    ax = fig.add_subplot()
    rows = axes.model_rows(["gpt2", "gpt2-medium"])
    _rows.row_axis(ax, rows, labels=True)
    low, high = ax.get_ylim()
    assert low > high
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        "GPT-2 small",
        "GPT-2 medium",
    ]


def test_run_counts_counts_each_run_under_the_role_of_its_checkpoint() -> None:
    counts = _rows.run_counts(
        {
            ("gpt2-xl", "fewshot"): True,
            ("qwen3-4b-base", "fewshot"): True,
            ("qwen3-4b", "fewshot"): True,
            ("qwen3-4b", "concept"): False,
        }
    )
    assert counts["gpt2"] == stats.RoleCount(1, 1)
    assert counts["base"] == stats.RoleCount(1, 1)
    assert counts["post"] == stats.RoleCount(1, 2)


def test_runs_text_names_the_roles_and_skips_empty_ones() -> None:
    counts = {
        "gpt2": stats.RoleCount(3, 3),
        "base": stats.RoleCount(0, 0),
        "post": stats.RoleCount(3, 6),
    }
    assert _rows.runs_text(counts) == (
        "3 of 3 GPT-2 runs and 3 of 6 post-trained runs"
    )


def test_freeze_layout_pins_the_axes_to_six_decimals_and_drops_the_solver() -> (
    None
):
    fig = _rows.new_figure(2.0)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.37, 2.11])
    for cell in grid:
        fig.add_subplot(cell).set_title("title")
    _rows.freeze_layout(fig)
    assert not isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine)
    for ax in fig.axes:
        bounds = ax.get_position().bounds
        assert all(abs(v - round(v, 6)) < 1e-12 for v in bounds)
        assert min(bounds[2:]) > 0.1


def test_the_hexagon_role_key_is_drawn_larger_than_a_data_marker() -> None:
    key = _rows.series_handle("a", paper.ROLE_KEY_MARKER, "#123456")
    data = _rows.series_handle("b", "o", "#123456")
    assert key.get_markersize() >= 6.0
    assert key.get_markersize() > data.get_markersize()


def test_stars_are_drawn_larger_and_the_category_shapes_are_all_filled() -> (
    None
):
    assert _rows.marker_size("*") > _rows.marker_size("o")
    assert _rows.marker_size(paper.ROLE_KEY_MARKER) >= 6.0
    fig = _rows.new_figure(2.0)
    ax = fig.add_subplot()
    _rows.dot(ax, "gpt2", 0.0, 0.5, "*", "#123456")
    (line,) = ax.lines
    assert line.get_markersize() == _rows.marker_size("*")
    for marker in paper.CATEGORY_MARKERS:
        assert marker not in ("1", "2", "3", "4")


def _area(marker: str, size: float) -> float:
    style = MarkerStyle(marker)
    path = style.get_path().transformed(style.get_transform())
    total = 0.0
    for poly in path.to_polygons(closed_only=True):
        x, y = poly[:, 0], poly[:, 1]
        total += 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return float(total * size * size)


def test_category_shapes_and_the_circle_have_about_the_same_area() -> None:
    circle = _area("o", _rows.marker_size("o"))
    for marker in paper.CATEGORY_MARKERS:
        ratio = _area(marker, _rows.marker_size(marker)) / circle
        assert 0.65 <= ratio <= 1.0, (marker, ratio)


def test_filled_category_markers_get_a_thin_white_edge() -> None:
    fig = _rows.new_figure(2.0)
    ax = fig.add_subplot()
    _rows.dot(ax, "gpt2", 0.0, 0.5, "<", "#123456")
    _rows.dot(ax, "qwen3-4b", 1.0, 0.5, "<", "#123456")
    _rows.dot(ax, "gpt2", 2.0, 0.5, "o", "#123456")
    filled, opened, circle = ax.lines
    assert filled.get_markeredgecolor() == "white"
    assert filled.get_markeredgewidth() == pytest.approx(0.3)
    assert opened.get_markeredgecolor() == "#123456"
    assert circle.get_markeredgecolor() == "#123456"
    key = _rows.series_handle("k", "<", "#123456")
    assert key.get_markeredgecolor() == "white"
