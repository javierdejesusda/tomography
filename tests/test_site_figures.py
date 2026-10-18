import itertools
from pathlib import Path

from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
import pytest

from tomography.site import figures
from tomography.site import findings
from tomography.site import panels
from tomography.site import store as store_module
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _all_figures() -> list[findings.Figure]:
    store = store_module.Store(RESULTS)
    return [f for item in findings.build_findings(store) for f in item.figures]


FIGURES = _all_figures()


def _drawn(fig: Figure) -> RendererBase:
    canvas = FigureCanvasAgg(fig)
    with paper.paper_style():
        canvas.draw()
    return canvas.get_renderer()


def _texts(fig: Figure) -> list[Text]:
    unique = {id(t): t for t in fig.findobj(Text)}
    return [t for t in unique.values() if t.get_text() and t.get_visible()]


def _boxes(fig: Figure) -> list[tuple[str, tuple[float, float, float, float]]]:
    renderer = _drawn(fig)
    boxes = []
    for text in _texts(fig):
        box = text.get_window_extent(renderer)
        boxes.append((text.get_text(), (box.x0, box.y0, box.x1, box.y1)))
    return boxes


def _overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return (
        a[0] < b[2] - 0.5
        and b[0] < a[2] - 0.5
        and (a[1] < b[3] - 0.5 and b[1] < a[3] - 0.5)
    )


def test_there_are_figures_from_the_committed_results() -> None:
    assert len(FIGURES) >= 8


@pytest.mark.parametrize("index", range(len(FIGURES)))
def test_figure_is_print_width_and_has_one_letter_per_panel(
    index: int,
) -> None:
    source = FIGURES[index]
    fig = figures.render(source)
    assert fig.get_size_inches()[0] == paper.FULL_WIDTH
    letters = [
        t.get_text() for t in _texts(fig) if t.get_fontweight() == "bold"
    ]
    expected = [f"({chr(97 + i)})" for i in range(len(source.panels))]
    assert letters == expected


@pytest.mark.parametrize("index", range(len(FIGURES)))
def test_no_text_overlaps_or_leaves_the_figure(index: int) -> None:
    fig = figures.render(FIGURES[index])
    boxes = _boxes(fig)
    width, height = (fig.get_size_inches() * fig.dpi).tolist()
    for label, (x0, y0, x1, y1) in boxes:
        assert min(x0, y0) >= -0.5, label
        assert x1 <= width + 0.5, label
        assert y1 <= height + 0.5, label
    for (la, a), (lb, b) in itertools.combinations(boxes, 2):
        assert not _overlap(a, b), (la, lb)


@pytest.mark.parametrize("index", range(len(FIGURES)))
def test_no_marker_covers_a_tick_label(index: int) -> None:
    fig = figures.render(FIGURES[index])
    renderer = _drawn(fig)
    ticks = []
    for ax in fig.axes:
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            if label.get_text():
                box = label.get_window_extent(renderer)
                ticks.append((label.get_text(), box))
    for ax in fig.axes:
        for line in ax.get_lines():
            if line.get_marker() in ("None", None, ""):
                continue
            for x, y in line.get_xydata():
                px, py = ax.transData.transform((x, y))
                for text, box in ticks:
                    assert not (
                        box.x0 - 2 <= px <= box.x1 + 2
                        and box.y0 - 2 <= py <= box.y1 + 2
                    ), text


@pytest.mark.parametrize("index", range(len(FIGURES)))
def test_rows_are_about_a_sixth_of_an_inch(index: int) -> None:
    fig = figures.render(FIGURES[index])
    for ax in fig.axes:
        low, high = ax.get_ylim()
        if low < high:
            continue
        inches = ax.get_position().height * fig.get_size_inches()[1]
        assert inches / (low - high) == pytest.approx(
            paper.ROW_HEIGHT, rel=0.02
        )


def test_figures_are_byte_identical_when_built_twice() -> None:
    for source in FIGURES[:3]:
        assert figures.svg(source) == figures.svg(source)
        assert figures.pdf(source) == figures.pdf(source)


def test_svg_is_vector_without_text_elements_or_dates() -> None:
    text = figures.svg(FIGURES[0]).decode("utf-8")
    assert text.startswith("<?xml")
    assert "<text" not in text
    assert "<dc:date>" not in text
    assert 'width="396pt"' in text


def test_csv_lists_every_plotted_number() -> None:
    dots = next(f for f in FIGURES if isinstance(f.panels[0], panels.DotPanel))
    panel = dots.panels[0]
    assert isinstance(panel, panels.DotPanel)
    lines = figures.csv_text(dots).splitlines()
    assert lines[0].startswith("panel,row,series")
    marks = sum(len(r.marks) for p in dots.panels for r in p.rows)
    assert len(lines) - 1 == marks
    first = panel.rows[0]
    assert repr(first.marks[0].value) in lines[1]


def test_csv_of_line_panels_lists_each_point() -> None:
    lines_fig = next(
        f for f in FIGURES if isinstance(f.panels[0], panels.LinePanel)
    )
    count = sum(
        len(s.points)
        for p in lines_fig.panels
        if isinstance(p, panels.LinePanel)
        for s in p.series
    )
    assert len(figures.csv_text(lines_fig).splitlines()) - 1 == count


def test_model_rows_use_paper_names() -> None:
    fig = figures.render(FIGURES[0])
    labels = " ".join(
        t.get_text() for ax in fig.axes for t in ax.get_yticklabels()
    )
    assert "gpt2" not in labels
    assert "qwen3" not in labels


@pytest.mark.parametrize("index", range(len(FIGURES)))
def test_every_mark_lies_inside_its_axis_range(index: int) -> None:
    for panel in FIGURES[index].panels:
        if not isinstance(panel, panels.DotPanel):
            continue
        low, high = panel.domain
        for row in panel.rows:
            for mark in row.marks:
                assert low <= mark.value <= high, (row.label, mark.value)
