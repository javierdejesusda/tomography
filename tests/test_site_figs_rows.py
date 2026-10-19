from matplotlib.layout_engine import ConstrainedLayoutEngine

from tomography.site.figs import _rows
from tomography.viz import axes
from tomography.viz import paper


def test_negative_ticks_are_mathtext_so_the_minus_can_be_drawn() -> None:
    assert _rows.tick_text(-0.3) == r"$\mathdefault{-0.3}$"
    assert _rows.tick_text(-1.0) == r"$\mathdefault{-1}$"


def test_zero_and_positive_ticks_are_plain_trimmed_numbers() -> None:
    assert _rows.tick_text(0.0) == "0"
    assert _rows.tick_text(-0.0) == "0"
    assert _rows.tick_text(0.25) == "0.25"
    assert _rows.tick_text(1.0) == "1"
    assert _rows.tick_formatter()(0.3, 0) == "0.3"


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
