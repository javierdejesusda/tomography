"""Tests for the matrix and embedding primitives."""

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import to_hex
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import numpy as np
import pytest

from tomography.viz import geometry
from tomography.viz import style

ROWS = ["happy", "sad", "angry"]
COLS = ["a", "b", "c", "d"]


def make_values(n_rows: int = 3, n_cols: int = 4) -> np.ndarray:
    return np.random.default_rng(0).uniform(0.1, 0.9, (n_rows, n_cols))


def matrix_image(ax: Axes) -> AxesImage:
    (image,) = ax.images
    return image


def test_plot_matrix_draws_the_values_with_row_zero_on_top() -> None:
    values = make_values()

    fig = geometry.plot_matrix(values, rows=ROWS, cols=COLS)

    image = matrix_image(fig.axes[0])
    assert image.origin == "upper"
    np.testing.assert_array_equal(np.ma.getdata(image.get_array()), values)


def test_sequential_matrix_runs_light_to_dark_over_the_data_range() -> None:
    values = make_values()

    fig = geometry.plot_matrix(values, rows=ROWS, cols=COLS)

    image = matrix_image(fig.axes[0])
    assert (image.norm.vmin, image.norm.vmax) == (values.min(), values.max())
    assert to_hex(image.cmap(0.0)) == style.SEQUENTIAL_STEPS[0]
    assert to_hex(image.cmap(1.0)) == style.SEQUENTIAL_STEPS[-1]


def test_sequential_range_can_be_fixed() -> None:
    fig = geometry.plot_matrix(
        make_values(), rows=ROWS, cols=COLS, vmin=0.0, vmax=1.0
    )

    norm = matrix_image(fig.axes[0]).norm
    assert (norm.vmin, norm.vmax) == (0.0, 1.0)


def test_diverging_matrix_is_symmetric_with_a_gray_zero() -> None:
    values = make_values() - 0.4

    fig = geometry.plot_matrix(values, rows=ROWS, cols=COLS, kind="diverging")

    image = matrix_image(fig.axes[0])
    limit = np.abs(values).max()
    assert (image.norm.vmin, image.norm.vmax) == (-limit, limit)
    assert to_rgb(image.cmap(image.norm(0.0))) == pytest.approx(
        to_rgb(style.NEUTRAL), abs=2 / 255
    )


def test_diverging_matrix_range_uses_the_larger_bound() -> None:
    fig = geometry.plot_matrix(
        make_values() - 0.4,
        rows=ROWS,
        cols=COLS,
        kind="diverging",
        vmin=-1.0,
        vmax=0.5,
    )

    norm = matrix_image(fig.axes[0]).norm
    assert (norm.vmin, norm.vmax) == (-1.0, 1.0)


def test_ticks_and_title_come_from_the_sidecar_fields() -> None:
    fig = geometry.plot_matrix(
        make_values(), rows=ROWS, cols=COLS, title="Cosine"
    )

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ROWS
    assert [t.get_text() for t in ax.get_xticklabels()] == COLS
    assert ax.get_title() == "Cosine"


def test_small_matrices_write_each_value_in_its_cell() -> None:
    values = make_values()

    fig = geometry.plot_matrix(values, rows=ROWS, cols=COLS)

    ax = fig.axes[0]
    assert sorted(t.get_text() for t in ax.texts) == sorted(
        f"{v:.2f}" for v in values.flat
    )


def test_large_matrices_are_not_annotated_unless_asked() -> None:
    values = make_values(13, 13)
    names = [str(i) for i in range(13)]

    plain = geometry.plot_matrix(values, rows=names, cols=names)
    forced = geometry.plot_matrix(values, rows=names, cols=names, annotate=True)

    assert not plain.axes[0].texts
    assert len(forced.axes[0].texts) == 169


def test_cell_text_takes_white_or_ink_by_its_fill() -> None:
    values = np.array([[0.0, 1.0]])

    fig = geometry.plot_matrix(values, rows=["r"], cols=["lo", "hi"])

    by_text = {t.get_text(): t.get_color() for t in fig.axes[0].texts}
    assert by_text == {"0.00": style.INK, "1.00": style.SURFACE}


def test_annotation_format_is_configurable() -> None:
    fig = geometry.plot_matrix(
        np.array([[0.5]]), rows=["r"], cols=["c"], fmt="{:.1f}"
    )

    assert [t.get_text() for t in fig.axes[0].texts] == ["0.5"]


def test_cells_are_separated_by_surface_colored_gaps() -> None:
    fig = geometry.plot_matrix(make_values(), rows=ROWS, cols=COLS)

    ax = fig.axes[0]
    np.testing.assert_allclose(
        ax.xaxis.get_minorticklocs(), np.arange(-0.5, len(COLS))
    )
    gridline = ax.xaxis.get_minor_ticks()[0].gridline
    assert to_hex(gridline.get_color()) == style.SURFACE
    assert gridline.get_linewidth() >= 2 * style.PX
    assert not ax.get_axisbelow()


def test_missing_cells_stay_blank_and_unannotated() -> None:
    values = make_values()
    values[1, 2] = np.nan

    fig = geometry.plot_matrix(values, rows=ROWS, cols=COLS)

    ax = fig.axes[0]
    assert np.ma.getmaskarray(matrix_image(ax).get_array())[1, 2]
    assert len(ax.texts) == values.size - 1


def test_matrix_colorbar_can_be_labeled_or_dropped() -> None:
    labeled = geometry.plot_matrix(
        make_values(), rows=ROWS, cols=COLS, cbar_label="Accuracy"
    )
    dropped = geometry.plot_matrix(
        make_values(), rows=ROWS, cols=COLS, colorbar=False
    )

    assert len(labeled.axes) == 2
    assert labeled.axes[1].get_ylabel() == "Accuracy"
    assert len(dropped.axes) == 1


def test_plot_matrix_draws_on_a_given_axes() -> None:
    fig, axes = style.new_figure(1, 2)

    returned = geometry.plot_matrix(
        make_values(), rows=ROWS, cols=COLS, ax=axes[0, 1]
    )

    assert returned is fig
    assert axes[0, 1].images
    assert not axes[0, 0].images


@pytest.mark.parametrize(
    ("values", "rows", "cols", "kwargs", "message"),
    [
        (np.zeros(4), ["a"], ["b"], {}, "2-D"),
        (np.zeros((3, 4)), ROWS[:2], COLS, {}, "rows"),
        (np.zeros((3, 4)), ROWS, COLS[:3], {}, "cols"),
        (np.zeros((3, 4)), ROWS, COLS, {"kind": "rainbow"}, "kind"),
        (np.zeros((3, 4)), ROWS, COLS, {"vmin": 1.0, "vmax": 0.0}, "vmin"),
    ],
    ids=["not-2d", "rows", "cols", "kind", "range"],
)
def test_plot_matrix_rejects_unusable_inputs(
    values: np.ndarray,
    rows: list[str],
    cols: list[str],
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        geometry.plot_matrix(values, rows=rows, cols=cols, **kwargs)


def test_matrix_figure_saves_in_all_formats(tmp_path: Path) -> None:
    fig = geometry.plot_matrix(make_values(), rows=ROWS, cols=COLS)

    paths = style.save_figure(fig, tmp_path, "matrix")

    assert [path.suffix for path in paths] == [".pdf", ".png", ".svg"]
    assert all(path.stat().st_size > 0 for path in paths)


def make_embedding(
    n_layers: int = 2, n_points: int = 30, n_classes: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    xy = rng.normal(size=(n_layers, n_points, 2))
    labels = np.arange(n_points, dtype=np.int64) % n_classes
    return xy, labels, np.arange(n_layers) * 4


def panels(fig: Figure) -> list[Axes]:
    return [ax for ax in fig.axes if ax.get_visible()]


def clouds(ax: Axes) -> list[PathCollection]:
    return [c for c in ax.collections if isinstance(c, PathCollection)]


def test_embedding_draws_one_panel_per_layer_titled_by_layer() -> None:
    xy, labels, layers = make_embedding(n_layers=3)

    fig = geometry.plot_embedding(xy, labels, layers, classes=["a", "b", "c"])

    assert [ax.get_title() for ax in panels(fig)] == [
        "Layer 0",
        "Layer 4",
        "Layer 8",
    ]


def test_embedding_draws_each_class_with_its_hue_and_marker() -> None:
    xy, labels, layers = make_embedding()

    fig = geometry.plot_embedding(xy, labels, layers, classes=["a", "b", "c"])

    ax = panels(fig)[1]
    drawn = clouds(ax)
    assert len(drawn) == 3
    for c, cloud in enumerate(drawn):
        np.testing.assert_allclose(cloud.get_offsets(), xy[1][labels == c])
        assert to_hex(cloud.get_facecolor()[0]) == style.CATEGORICAL[c]
    shapes = [tuple(c.get_paths()[0].vertices.ravel()) for c in drawn]
    assert len(set(shapes)) == 3


def test_embedding_names_the_classes_in_one_legend() -> None:
    xy, labels, layers = make_embedding()

    fig = geometry.plot_embedding(xy, labels, layers, classes=["a", "b", "c"])

    (legend,) = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == ["a", "b", "c"]


def test_embedding_has_no_axis_scale_and_rasterizes_its_points() -> None:
    xy, labels, layers = make_embedding()

    fig = geometry.plot_embedding(xy, labels, layers, classes=["a", "b", "c"])

    for ax in panels(fig):
        assert len(ax.get_xticks()) == 0
        assert len(ax.get_yticks()) == 0
        assert all(c.get_rasterized() for c in clouds(ax))


def test_embedding_wraps_panels_into_the_requested_columns() -> None:
    xy, labels, layers = make_embedding(n_layers=5)

    fig = geometry.plot_embedding(
        xy, labels, layers, classes=["a", "b", "c"], ncols=2
    )

    rows, cols = fig.axes[0].get_subplotspec().get_gridspec().get_geometry()
    assert (rows, cols) == (3, 2)
    assert len(panels(fig)) == 5


def test_many_classes_become_one_vs_rest_rows() -> None:
    names = ["c0", "c1", "c2", "c3", "c4", "c5"]
    xy, labels, layers = make_embedding(n_points=60, n_classes=6)

    fig = geometry.plot_embedding(xy, labels, layers, classes=names)

    assert len(fig.axes) == 12
    spec = fig.axes[0].get_subplotspec().get_gridspec().get_geometry()
    assert spec == (6, 2)
    first_column = fig.axes[0::2]
    assert [ax.get_ylabel() for ax in first_column] == names
    assert [ax.get_title() for ax in fig.axes[:2]] == ["Layer 0", "Layer 4"]
    assert all(ax.get_title() == "" for ax in fig.axes[2:])


def test_one_vs_rest_panels_highlight_only_the_row_class() -> None:
    xy, labels, layers = make_embedding(n_points=60, n_classes=6)
    names = ["c0", "c1", "c2", "c3", "c4", "c5"]

    fig = geometry.plot_embedding(xy, labels, layers, classes=names)

    ax = fig.axes[2 * 3 + 1]
    rest, focus = sorted(clouds(ax), key=lambda c: len(c.get_offsets()))[::-1]
    np.testing.assert_allclose(focus.get_offsets(), xy[1][labels == 3])
    np.testing.assert_allclose(rest.get_offsets(), xy[1][labels != 3])
    assert to_hex(focus.get_facecolor()[0]) == style.CATEGORICAL[0]
    assert to_hex(rest.get_facecolor()[0]) != style.CATEGORICAL[0]


def test_one_vs_rest_legend_explains_the_two_encodings() -> None:
    xy, labels, layers = make_embedding(n_points=60, n_classes=6)

    fig = geometry.plot_embedding(
        xy, labels, layers, classes=[f"c{i}" for i in range(6)]
    )

    (legend,) = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == [
        "class of the row",
        "other classes",
    ]


def test_one_vs_rest_rows_cannot_be_wrapped() -> None:
    xy, labels, layers = make_embedding(n_points=60, n_classes=6)

    with pytest.raises(ValueError, match="ncols"):
        geometry.plot_embedding(
            xy, labels, layers, classes=[f"c{i}" for i in range(6)], ncols=2
        )


@pytest.mark.parametrize(
    ("xy_shape", "n_labels", "n_layers", "classes", "message"),
    [
        ((2, 30), 30, 2, ["a", "b", "c"], "shape"),
        ((2, 30, 3), 30, 2, ["a", "b", "c"], "shape"),
        ((2, 30, 2), 29, 2, ["a", "b", "c"], "labels"),
        ((2, 30, 2), 30, 3, ["a", "b", "c"], "layers"),
        ((2, 30, 2), 30, 2, ["a", "b"], "classes"),
        ((2, 30, 2), 30, 2, [], "classes"),
    ],
    ids=[
        "xy-2d",
        "xy-3-coords",
        "labels",
        "layers",
        "few-classes",
        "no-classes",
    ],
)
def test_plot_embedding_rejects_unusable_inputs(
    xy_shape: tuple[int, ...],
    n_labels: int,
    n_layers: int,
    classes: list[str],
    message: str,
) -> None:
    labels = np.arange(n_labels, dtype=np.int64) % 3

    with pytest.raises(ValueError, match=message):
        geometry.plot_embedding(
            np.zeros(xy_shape),
            labels,
            np.arange(n_layers),
            classes=classes,
        )


def test_embedding_figure_saves_in_all_formats(tmp_path: Path) -> None:
    xy, labels, layers = make_embedding()
    fig = geometry.plot_embedding(xy, labels, layers, classes=["a", "b", "c"])

    paths = style.save_figure(fig, tmp_path, "embedding")

    assert [path.suffix for path in paths] == [".pdf", ".png", ".svg"]
    assert all(path.stat().st_size > 0 for path in paths)
