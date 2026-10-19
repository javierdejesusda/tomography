from matplotlib.figure import Figure
import numpy as np
import pytest

from tomography.site import store
from tomography.viz import axes as shared
from tomography.viz import paper


def test_relative_depth_divides_the_layer_index_by_the_layer_count() -> None:
    depth = shared.relative_depth([1, 6, 12], 12)
    assert depth == pytest.approx([1 / 12, 0.5, 1.0])
    with pytest.raises(ValueError, match="positive"):
        shared.relative_depth([1], 0)


def test_model_style_keeps_families_and_roles_apart() -> None:
    gpt2 = shared.model_style("gpt2-xl")
    base = shared.model_style("qwen3-4b-base")
    post = shared.model_style("qwen3-4b")
    assert gpt2.family == "GPT-2"
    assert base.family == post.family == "Qwen3"
    assert base.color == post.color
    assert base.linestyle == "-"
    assert post.linestyle == "--"
    assert base.filled
    assert not post.filled
    assert gpt2.color != base.color


def test_darker_means_larger_inside_a_family() -> None:
    def lightness(color: str) -> float:
        r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
        return 0.299 * r + 0.587 * g + 0.114 * b

    sizes = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
    lums = [lightness(shared.model_style(k).color) for k in sizes]
    assert lums == sorted(lums, reverse=True)
    qwen = ["qwen3-0.6b-base", "qwen3-1.7b-base", "qwen3-4b-base"]
    lums = [lightness(shared.model_style(k).color) for k in qwen]
    assert lums == sorted(lums, reverse=True)


def test_depth_axis_runs_from_zero_to_one() -> None:
    with paper.paper_style():
        ax = Figure().subplots()
        shared.depth_axis(ax)
    assert ax.get_xlim() == pytest.approx((0.0, 1.0), abs=0.05)
    assert ax.get_xlabel() == "relative depth"


def test_size_axis_is_logarithmic_in_parameters() -> None:
    with paper.paper_style():
        ax = Figure().subplots()
        shared.size_axis(ax, [1.2e8, 1.5e9, 4e9])
    assert ax.get_xscale() == "log"
    low, high = ax.get_xlim()
    assert low < 1.2e8
    assert high > 4e9
    assert ax.get_xlabel() == "parameters"


def test_layer_curve_draws_a_line_and_an_interval() -> None:
    with paper.paper_style():
        ax = Figure().subplots()
        depth = np.linspace(0.1, 1.0, 10)
        mean = np.linspace(0.5, 0.9, 10)
        shared.layer_curve(
            ax, depth, mean, mean - 0.02, mean + 0.02, "qwen3-4b"
        )
    line = ax.get_lines()[0]
    assert line.get_linestyle() == "--"
    assert len(ax.collections) == 1


def test_heatmap_uses_a_sequential_scale_and_outlines_the_diagonal() -> None:
    with paper.paper_style():
        ax = Figure().subplots()
        image = shared.heatmap(ax, np.full((4, 4), 0.8), vmin=0.5, vmax=1.0)
    assert image.get_clim() == (0.5, 1.0)
    assert len(ax.patches) == 4
    assert not ax.texts


def test_caption_joins_claim_and_detail_with_single_spaces() -> None:
    text = shared.caption("Reading rises with depth.", "Lines are means.")
    assert text == "Reading rises with depth. Lines are means."
    with pytest.raises(ValueError, match="period"):
        shared.caption("No period", "x")


MINUS = chr(0x2212)


def test_unit_axis_sets_the_same_ticks_and_labels_on_any_zero_to_one_axis() -> (
    None
):
    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])
    shared.unit_axis(ax, "x")
    shared.unit_axis(ax, "y")
    fig.canvas.draw()  # type: ignore[no-untyped-call]
    for axis in (ax.xaxis, ax.yaxis):
        assert [float(t) for t in axis.get_ticklocs()] == [
            0,
            0.25,
            0.5,
            0.75,
            1,
        ]
        assert [t.get_text() for t in axis.get_ticklabels()] == [
            "0",
            "0.25",
            "0.5",
            "0.75",
            "1",
        ]


def test_minus_formatter_writes_the_true_minus_and_trims_zeros() -> None:
    fmt = shared.minus_formatter()
    assert fmt(-0.5, 0) == f"{MINUS}0.5"
    assert fmt(0.25, 0) == "0.25"
    assert fmt(1.0, 0) == "1"
    assert fmt(100.0, 0) == "100"
    assert fmt(-1e-16, 0) == "0"
    assert fmt(0.0, 0) == "0"


def test_role_dodge_moves_base_and_post_trained_apart_in_decades() -> None:
    assert shared.role_dodge("gpt2-large") == 0.0
    assert shared.role_dodge("qwen3-1.7b-base") == -0.035
    assert shared.role_dodge("qwen3-1.7b") == 0.035


def test_ref_line_is_a_thin_dotted_grey_line_behind_the_data() -> None:
    ax = Figure().subplots()
    line = shared.ref_line(ax, 0.5, "y")
    assert line.get_color() == "#7f7f7f"
    assert line.get_linewidth() == 0.5
    assert line.get_linestyle() == "--"
    assert line.get_zorder() < 2
    assert list(line.get_ydata()) == [0.5, 0.5]
    vertical = shared.ref_line(ax, 3.0, "x")
    assert list(vertical.get_xdata()) == [3.0, 3.0]


def test_model_rows_leave_a_gap_after_gpt2_and_between_qwen3_pairs() -> None:
    rows = shared.model_rows(list(store.MODEL_ORDER))
    assert [rows[m] for m in store.MODEL_ORDER] == pytest.approx(
        [0, 1, 2, 3, 4.6, 5.6, 6.85, 7.85, 9.1, 10.1]
    )


def test_model_rows_only_gap_between_neighbours_that_are_present() -> None:
    rows = shared.model_rows(["gpt2", "qwen3-4b", "gpt2-xl"])
    assert rows["gpt2"] == 0.0
    assert rows["gpt2-xl"] == 1.0
    assert rows["qwen3-4b"] == pytest.approx(2.6)
