import csv
import io
from pathlib import Path
import runpy
from typing import Any

import matplotlib as mpl
from matplotlib.patches import Rectangle
import numpy as np
import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
SPEC = "a_knowledge_matrix"
MODEL = "qwen3-4b-base"
HEIGHT = 3.6
SLACK = 0.02


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(output.csv)))


def _matrix() -> tuple[list[str], np.ndarray]:
    path = _R["run"]("p9b", MODEL)
    names = _R["data"](path, "matrix.json")["rows"]
    values = np.asarray(_R["npz"](path, "matrix.npz")["values"], dtype=float)
    return list(names), values


def _table() -> dict[str, dict[str, Any]]:
    payload = _R["data"](_R["run"]("p9b", MODEL), "table.json")
    return {
        r[0]: dict(zip(payload["columns"], r, strict=True))
        for r in payload["rows"]
    }


def _close_types() -> list[str]:
    names, values = _matrix()
    table = _table()
    return [
        t
        for i, t in enumerate(names)
        if max(table[t]["text_word"] or 0.0, table[t]["text_char"] or 0.0)
        >= values[i, i] - SLACK
    ]


def test_the_close_type_count_is_computed_from_the_files(
    output: registry.FigureOutput,
) -> None:
    close = _close_types()
    names, _ = _matrix()
    assert len(close) == 9
    assert output.numbers["close"] == len(close)
    assert output.numbers["types"] == len(names)
    assert output.claim == (
        "A classifier on the words or characters alone comes within "
        f"{SLACK:.3f} of the reader or beats it on {len(close)} of "
        f"{len(names)} question types."
    )
    assert "Qwen3-4B-Base" in output.detail


def test_size_guards_and_layout(output: registry.FigureOutput) -> None:
    fig = output.figure
    assert tuple(fig.get_size_inches()) == (paper.FULL_WIDTH, HEIGHT)
    assert checks.guard_problems(fig, appendix=True) == []
    assert checks.problems(fig) == []


def test_svg_and_pdf_bytes_are_the_same_for_the_same_input() -> None:
    first = registry.build_figure(SPEC, Manifest(RESULTS))
    second = registry.build_figure(SPEC, Manifest(RESULTS))
    with paper.paper_style():
        assert paper.svg_bytes(first.figure) == paper.svg_bytes(second.figure)
        assert paper.pdf_bytes(first.figure) == paper.pdf_bytes(second.figure)


def _best(t: str) -> float:
    row = _table()[t]
    return max(row["text_word"] or 0.0, row["text_char"] or 0.0)


def _diff(i: int, t: str, values: np.ndarray) -> float:
    return float(values[i, i]) - _best(t)


def test_matrix_is_cividis_on_half_to_one_with_a_white_diagonal_outline(
    output: registry.FigureOutput,
) -> None:
    names, values = _matrix()
    matrix = output.figure.axes[0]
    image = matrix.images[0]
    assert np.asarray(image.get_array()) == pytest.approx(values)
    assert image.get_clim() == (0.5, 1.0)
    assert image.get_cmap().name == mpl.colormaps[paper.SEQUENTIAL_CMAP].name
    frames = [p for p in matrix.patches if isinstance(p, Rectangle)]
    assert len(frames) == len(names)
    assert {p.get_edgecolor()[:3] for p in frames} == {(1.0, 1.0, 1.0)}
    assert not any(p.get_fill() for p in frames)
    cells = [r for r in _rows(output) if r["panel"] == "a"]
    assert len(cells) == len(names) ** 2
    for r in cells:
        i, j = names.index(r["fitted"]), names.index(r["tested"])
        assert float(r["value"]) == pytest.approx(values[i, j])


def test_numbers_are_written_only_on_the_diagonal_and_below_point_eight(
    output: registry.FigureOutput,
) -> None:
    names, values = _matrix()
    matrix = output.figure.axes[0]
    shown = {
        (round(t.get_position()[1]), round(t.get_position()[0])): t.get_text()
        for t in matrix.texts
        if t.get_gid() == paper.TILE_LABEL
    }
    want = {
        (i, j)
        for i in range(len(names))
        for j in range(len(names))
        if i == j or values[i, j] < 0.8
    }
    assert set(shown) == want
    for (i, j), text in shown.items():
        assert text == f"{values[i, j]:.2f}"


def test_the_title_names_the_checkpoint(output: registry.FigureOutput) -> None:
    matrix = output.figure.axes[0]
    assert "Qwen3-4B-Base" in matrix.get_title(loc="left")


def test_panel_b_is_the_reader_minus_the_best_text_classifier_with_a_band(
    output: registry.FigureOutput,
) -> None:
    names, values = _matrix()
    diff_ax = output.figure.axes[1]
    dots = [ln for ln in diff_ax.lines if ln.get_gid() == "difference"]
    assert len(dots) == len(names)
    for ln in dots:
        i = round(float(ln.get_ydata()[0]))
        assert float(ln.get_xdata()[0]) == pytest.approx(
            _diff(i, names[i], values)
        )
    ahead = [ln for ln in dots if ln.get_markerfacecolor() != "white"]
    assert len(ahead) == sum(
        _diff(i, t, values) > SLACK for i, t in enumerate(names)
    )
    bands = [p for p in diff_ax.patches if p.get_gid() == "band"]
    assert len(bands) == 1
    box = bands[0].get_bbox()
    assert box.x0 == pytest.approx(-SLACK)
    assert box.x1 == pytest.approx(SLACK)
    csv_rows = [
        r
        for r in _rows(output)
        if r["panel"] == "b" and r["series"] == "difference"
    ]
    assert len(csv_rows) == len(names)
    assert output.numbers["ahead"] == sum(
        _diff(i, t, values) > SLACK for i, t in enumerate(names)
    )


def test_random_is_a_grey_tick_in_its_own_narrow_column(
    output: registry.FigureOutput,
) -> None:
    names, _ = _matrix()
    table = _table()
    random_ax = output.figure.axes[2]
    ticks = [ln for ln in random_ax.lines if ln.get_gid() == "random"]
    assert len(ticks) == len(names)
    assert {ln.get_color() for ln in ticks} == {paper.RANDOM_GREY}
    got = sorted(float(np.mean(ln.get_xdata())) for ln in ticks)
    assert got == pytest.approx(sorted(table[t]["random"] for t in names))
    assert (
        random_ax.get_position().width
        < output.figure.axes[1].get_position().width
    )


def test_panels_share_rows_and_carry_letters_and_no_dashes(
    output: registry.FigureOutput,
) -> None:
    matrix, diff_ax, random_ax = output.figure.axes[:3]
    assert matrix.get_shared_y_axes().joined(matrix, diff_ax)
    assert matrix.get_shared_y_axes().joined(matrix, random_ax)
    checks.draw(output.figure)
    letters = [
        t.get_text()
        for ax in (matrix, diff_ax)
        for t in ax.texts
        if t.get_text().startswith(r"$\mathbf")
    ]
    assert sorted(letters) == [r"$\mathbf{(a)}$", r"$\mathbf{(b)}$"]
    text = f"{output.claim} {output.detail} {output.csv}"
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text
    assert "_" not in f"{output.claim} {output.detail}"
    labels = [t.get_text() for t in matrix.get_yticklabels()]
    assert all("_" not in label for label in labels)


def test_the_caption_says_what_the_band_and_the_marks_mean(
    output: registry.FigureOutput,
) -> None:
    detail = output.detail
    assert "0.020" in detail
    assert "best text classifier" in detail
    assert "filled" in detail
    assert "grey tick" in detail
    assert "(c)" in detail


def test_the_x_labels_of_b_and_c_are_separate_two_line_labels(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    checks.draw(fig)
    labels = [
        ax.xaxis.get_label()
        for ax in fig.axes
        if ax.xaxis.get_label()
        .get_text()
        .startswith(("reader minus", "random"))
    ]
    assert len(labels) == 2
    assert all(chr(10) in t.get_text() for t in labels)
    boxes = [t.get_window_extent() for t in labels]
    assert boxes[0].x1 + 6 < boxes[1].x0


def test_the_random_ticks_are_thick_and_wide_enough_to_find(
    output: registry.FigureOutput,
) -> None:
    random_ax = output.figure.axes[2]
    ticks = [ln for ln in random_ax.lines if ln.get_gid() == "random"]
    for ln in ticks:
        assert ln.get_linewidth() >= 2.0
        x = ln.get_xdata()
        assert max(x) - min(x) >= 0.08
    checks.draw(output.figure)
    sizes = {
        round(t.get_fontsize(), 1)
        for ax in output.figure.axes
        for t in [*ax.get_xticklabels(), *ax.get_yticklabels()]
        if t.get_text()
    }
    assert min(sizes) >= paper.MIN_TEXT


def test_the_colour_bar_ticks_are_plain_text_at_the_tick_size(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    fig.canvas.draw()
    (bar,) = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    labels = [t for t in bar.get_xticklabels() if t.get_text()]
    assert labels
    for label in labels:
        assert "$" not in label.get_text()
        assert label.get_fontsize() >= paper.TICK
