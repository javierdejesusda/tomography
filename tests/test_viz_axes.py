from matplotlib.figure import Figure
import numpy as np
import pytest

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
