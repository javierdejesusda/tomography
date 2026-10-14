"""Tests for the layer-curve primitive."""

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_hex
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np
import pytest

from tomography.viz import reading
from tomography.viz import style


def make_arrays(
    n_series: int = 2, n_points: int = 6
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(n_points, dtype=np.float64)
    mean = np.linspace(0.5, 0.9, n_points) + 0.02 * np.arange(n_series)[:, None]
    return x, mean, mean - 0.05, mean + 0.05


def make_curves(
    n_series: int = 2, n_points: int = 6
) -> list[reading.CurveArrays]:
    x, mean, lo, hi = make_arrays(n_series, n_points)
    return reading.curves_from_arrays(x, mean, lo, hi)


def curve_lines(ax: Axes) -> dict[str, Line2D]:
    return {
        line.get_label(): line
        for line in ax.get_lines()
        if not line.get_label().startswith("_")
    }


def bands(ax: Axes) -> list[PolyCollection]:
    return [c for c in ax.collections if isinstance(c, PolyCollection)]


def test_curves_from_arrays_splits_series_that_share_x() -> None:
    x, mean, lo, hi = make_arrays(n_series=3)

    curves = reading.curves_from_arrays(x, mean, lo, hi)

    assert len(curves) == 3
    for i, (cx, cmean, clo, chi) in enumerate(curves):
        np.testing.assert_array_equal(cx, x)
        np.testing.assert_array_equal(cmean, mean[i])
        np.testing.assert_array_equal(clo, lo[i])
        np.testing.assert_array_equal(chi, hi[i])


def test_curves_from_arrays_rejects_mismatched_shapes() -> None:
    x, mean, lo, hi = make_arrays()

    with pytest.raises(ValueError, match="shape"):
        reading.curves_from_arrays(x, mean, lo[:, :-1], hi)


def test_plot_curves_draws_one_line_and_band_per_series() -> None:
    x, mean, lo, hi = make_arrays()

    fig = reading.plot_curves(
        reading.curves_from_arrays(x, mean, lo, hi), series=["a", "b"]
    )

    ax = fig.axes[0]
    lines = curve_lines(ax)
    assert list(lines) == ["a", "b"]
    for i, label in enumerate(["a", "b"]):
        np.testing.assert_array_equal(lines[label].get_xdata(), x)
        np.testing.assert_array_equal(lines[label].get_ydata(), mean[i])
    assert len(bands(ax)) == 2
    for i, band in enumerate(bands(ax)):
        ys = band.get_paths()[0].vertices[:, 1]
        assert np.isin(lo[i], ys).all()
        assert np.isin(hi[i], ys).all()


def test_series_keep_the_categorical_order_and_a_faint_band() -> None:
    fig = reading.plot_curves(make_curves(n_series=3), series=["a", "b", "c"])

    ax = fig.axes[0]
    for i, line in enumerate(curve_lines(ax).values()):
        assert to_hex(line.get_color()) == style.CATEGORICAL[i]
    for i, band in enumerate(bands(ax)):
        assert to_rgb(band.get_facecolor()[0]) == pytest.approx(
            to_rgb(style.CATEGORICAL[i])
        )
        assert band.get_alpha() == pytest.approx(0.1)


def test_plot_curves_labels_the_axes() -> None:
    fig = reading.plot_curves(
        make_curves(n_series=1),
        series=["a"],
        xlabel="Relative depth",
        ylabel="Paired accuracy",
    )

    ax = fig.axes[0]
    assert ax.get_xlabel() == "Relative depth"
    assert ax.get_ylabel() == "Paired accuracy"


def test_a_chance_level_is_drawn_and_named() -> None:
    fig = reading.plot_curves(make_curves(n_series=1), series=["a"], chance=0.5)

    ax = fig.axes[0]
    (level,) = [
        line
        for line in ax.get_lines()
        if np.all(np.asarray(line.get_ydata()) == 0.5)
    ]
    assert level.get_label() not in curve_lines(ax)
    assert [text.get_text() for text in ax.texts] == ["chance"]


def test_the_chance_label_sits_in_the_right_margin() -> None:
    fig = reading.plot_curves(make_curves(n_series=1), series=["a"], chance=0.5)

    (label,) = fig.axes[0].texts
    assert label.get_position()[0] > 1.0


def test_no_chance_line_is_drawn_by_default() -> None:
    fig = reading.plot_curves(make_curves(n_series=1), series=["a"])

    ax = fig.axes[0]
    assert len(ax.get_lines()) == 1
    assert not ax.texts


def test_a_legend_names_two_or_more_series() -> None:
    fig = reading.plot_curves(make_curves(), series=["a", "b"])

    legend = fig.axes[0].get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["a", "b"]


def test_the_legend_sits_above_the_plot_area() -> None:
    fig = reading.plot_curves(make_curves(), series=["a", "b"])

    fig.canvas.draw()
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_window_extent().y0 >= ax.get_window_extent().y1 - 1


@pytest.mark.parametrize(
    ("labels", "columns"),
    [
        (
            ["GPT-2 small", "GPT-2 medium", "GPT-2 large", "GPT-2 XL"],
            3,
        ),
        (
            [
                "Qwen3 0.6B base",
                "Qwen3 0.6B post",
                "Qwen3 1.7B base",
                "Qwen3 1.7B post",
            ],
            2,
        ),
        (
            [
                "Honesty on TruthfulQA, validation",
                "Honesty on TruthfulQA, test split",
            ],
            1,
        ),
        ([f"model-{i:04d}" for i in range(8)], 3),
        (list("abcdefgh"), 4),
    ],
    ids=[
        "gpt2-sizes",
        "qwen3-pairs",
        "two-long",
        "eight-medium",
        "eight-short",
    ],
)
def test_the_legend_wraps_to_fit_the_default_figure(
    labels: list[str], columns: int
) -> None:
    x, mean, lo, hi = make_arrays(n_series=len(labels))

    fig = reading.plot_curves(
        reading.curves_from_arrays(x, mean, lo, hi), series=labels
    )

    fig.canvas.draw()
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    box = legend.get_window_extent()
    assert box.x0 >= 0
    assert box.x1 <= fig.bbox.width
    assert ax.get_window_extent().width >= 0.6 * fig.bbox.width
    starts = {round(t.get_window_extent().x0) for t in legend.get_texts()}
    assert len(starts) == columns


def test_a_single_series_gets_no_legend() -> None:
    fig = reading.plot_curves(make_curves(n_series=1), series=["a"])

    assert fig.axes[0].get_legend() is None


def test_the_legend_can_be_switched_off() -> None:
    fig = reading.plot_curves(
        make_curves(),
        series=["a", "b"],
        legend=False,
    )

    assert fig.axes[0].get_legend() is None


def test_seed_curves_are_faint_hairlines_outside_the_legend() -> None:
    x, mean, lo, hi = make_arrays()
    per_seed = np.random.default_rng(0).uniform(0.4, 1.0, (3, len(x)))

    fig = reading.plot_curves(
        reading.curves_from_arrays(x, mean, lo, hi),
        series=["a", "b"],
        per_seed=[per_seed, None],
    )

    ax = fig.axes[0]
    faint = [line for line in ax.get_lines() if line.get_label() == "_seed"]
    assert len(faint) == 3
    for line, values in zip(faint, per_seed, strict=True):
        np.testing.assert_array_equal(line.get_ydata(), values)
        assert line.get_linewidth() < style.PX * 2
        assert line.get_alpha() is not None
        assert line.get_alpha() < 0.5
    legend = ax.get_legend()
    assert legend is not None
    assert len(legend.get_texts()) == 2


def test_series_of_different_lengths_share_one_figure() -> None:
    def one_series(k: int) -> reading.CurveArrays:
        mean = np.linspace(0.3, 0.9, k)
        return np.linspace(0, 1, k), mean, mean - 0.05, mean + 0.05

    fig = reading.plot_curves(
        [one_series(4), one_series(12)], series=["small", "large"]
    )

    lines = curve_lines(fig.axes[0])
    assert len(lines["small"].get_xdata()) == 4
    assert len(lines["large"].get_xdata()) == 12


def test_layer_ticks_are_whole_numbers() -> None:
    x, mean, lo, hi = make_arrays(n_series=1, n_points=4)

    fig = reading.plot_curves(
        reading.curves_from_arrays(x, mean, lo, hi), series=["a"]
    )

    fig.canvas.draw()
    ticks = fig.axes[0].get_xticks()
    inside = ticks[(ticks >= x[0]) & (ticks <= x[-1])]
    assert np.all(inside == np.round(inside))


def test_the_full_depth_can_set_the_x_range() -> None:
    fig = reading.plot_curves(
        make_curves(n_series=1, n_points=4),
        series=["a"],
        n_layers=12,
    )

    assert fig.axes[0].get_xlim() == (0, 12)


def test_the_y_range_can_be_fixed() -> None:
    fig = reading.plot_curves(
        make_curves(n_series=1),
        series=["a"],
        ylim=(0.0, 1.0),
    )

    assert fig.axes[0].get_ylim() == (0.0, 1.0)


def test_plot_curves_draws_on_a_given_axes() -> None:
    fig, axes = style.new_figure(1, 2)

    returned = reading.plot_curves(
        make_curves(n_series=1),
        series=["a"],
        ax=axes[0, 1],
    )

    assert returned is fig
    assert curve_lines(axes[0, 1])
    assert not axes[0, 0].get_lines()


def test_plot_curves_rejects_a_label_count_mismatch() -> None:
    with pytest.raises(ValueError, match="series"):
        reading.plot_curves(make_curves(), series=["a"])


def test_plot_curves_rejects_more_than_eight_series() -> None:
    with pytest.raises(ValueError, match="eight"):
        reading.plot_curves(
            make_curves(n_series=9),
            series=[str(i) for i in range(9)],
        )


def test_plot_curves_rejects_no_series() -> None:
    with pytest.raises(ValueError, match="at least one"):
        reading.plot_curves([], series=[])


def test_plot_curves_rejects_swapped_bounds() -> None:
    x, mean, lo, hi = make_arrays(n_series=1)

    with pytest.raises(ValueError, match="lo"):
        reading.plot_curves(
            reading.curves_from_arrays(x, mean, hi, lo), series=["a"]
        )


def test_plot_curves_rejects_seed_curves_of_the_wrong_length() -> None:
    x, mean, lo, hi = make_arrays(n_series=1)

    with pytest.raises(ValueError, match="per_seed"):
        reading.plot_curves(
            reading.curves_from_arrays(x, mean, lo, hi),
            series=["a"],
            per_seed=[np.zeros((2, len(x) + 1))],
        )


def test_curve_figure_saves_in_all_formats(tmp_path: Path) -> None:
    fig = reading.plot_curves(
        make_curves(),
        series=["a", "b"],
        chance=0.5,
    )

    paths = style.save_figure(fig, tmp_path, "curves")

    assert [path.suffix for path in paths] == [".pdf", ".png", ".svg"]
    assert all(path.stat().st_size > 0 for path in paths)
