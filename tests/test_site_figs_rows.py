from matplotlib.layout_engine import ConstrainedLayoutEngine

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


def test_all_or_per_role_says_all_only_when_every_checkpoint_holds() -> None:
    every = {
        "gpt2": stats.RoleCount(3, 3),
        "base": stats.RoleCount(3, 3),
        "post": stats.RoleCount(3, 3),
    }
    assert _rows.all_or_per_role(every) == "all 9 checkpoints"
    some = {**every, "post": stats.RoleCount(2, 3)}
    assert _rows.all_or_per_role(some) == (
        "3 of 3 GPT-2, 3 of 3 Qwen3 base and 2 of 3 post-trained checkpoints"
    )


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
