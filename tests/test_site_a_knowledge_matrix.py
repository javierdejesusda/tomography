import csv
import io
from pathlib import Path
import runpy
from typing import Any

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


def test_matrix_cells_are_the_stored_values_with_no_numbers(
    output: registry.FigureOutput,
) -> None:
    names, values = _matrix()
    matrix = output.figure.axes[0]
    assert np.asarray(matrix.images[0].get_array()) == pytest.approx(values)
    assert matrix.images[0].get_clim() == (0.5, 1.0)
    assert len(matrix.texts) == 1
    assert matrix.texts[0].get_text() == r"$\mathbf{(a)}$"
    cells = [r for r in _rows(output) if r["panel"] == "a"]
    assert len(cells) == len(names) ** 2
    for r in cells:
        i, j = names.index(r["fitted"]), names.index(r["tested"])
        assert float(r["value"]) == pytest.approx(values[i, j])


def test_strip_has_the_reader_both_classifiers_and_random_per_type(
    output: registry.FigureOutput,
) -> None:
    names, values = _matrix()
    table = _table()
    rows = [r for r in _rows(output) if r["panel"] == "b"]
    assert len(rows) == 4 * len(names)
    want = {
        "reader": lambda i, t: values[i, i],
        "words": lambda i, t: table[t]["text_word"],
        "characters": lambda i, t: table[t]["text_char"],
        "random": lambda i, t: table[t]["random"],
    }
    for i, t in enumerate(names):
        for series, get in want.items():
            (got,) = [
                r for r in rows if r["fitted"] == t and r["series"] == series
            ]
            assert float(got["value"]) == pytest.approx(get(i, t))


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
        f"{SLACK:.3f} of the reader on {len(close)} of {len(names)} "
        "question types."
    )
    assert "Qwen3-4B-Base" in output.detail


def test_size_guards_and_layout(output: registry.FigureOutput) -> None:
    fig = output.figure
    assert tuple(fig.get_size_inches()) == (paper.FULL_WIDTH, HEIGHT)
    assert checks.guard_problems(fig, appendix=True) == []
    assert checks.problems(fig) == []


def test_panels_share_rows_and_carry_letters_and_no_dashes(
    output: registry.FigureOutput,
) -> None:
    matrix, strip = output.figure.axes[:2]
    assert matrix.get_shared_y_axes().joined(matrix, strip)
    letters = [
        t.get_text()
        for ax in (matrix, strip)
        for t in ax.texts
        if t.get_text().startswith(r"$\mathbf")
    ]
    assert sorted(letters) == [r"$\mathbf{(a)}$", r"$\mathbf{(b)}$"]
    text = f"{output.claim} {output.detail} {output.csv}"
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text
    assert "_" not in f"{output.claim} {output.detail}"
    checks.draw(output.figure)
    labels = [t.get_text() for t in matrix.get_yticklabels()]
    assert all("_" not in label for label in labels)


def test_svg_and_pdf_bytes_are_the_same_for_the_same_input() -> None:
    first = registry.build_figure(SPEC, Manifest(RESULTS))
    second = registry.build_figure(SPEC, Manifest(RESULTS))
    with paper.paper_style():
        assert paper.svg_bytes(first.figure) == paper.svg_bytes(second.figure)
        assert paper.pdf_bytes(first.figure) == paper.pdf_bytes(second.figure)


def test_the_reader_is_drawn_above_the_classifier_and_random_markers(
    output: registry.FigureOutput,
) -> None:
    ax_b = output.figure.axes[1]
    marked = {
        ln.get_marker(): ln.get_zorder()
        for ln in ax_b.lines
        if ln.get_marker() not in ("None", "", None)
    }
    reader = marked[paper.MEASURE_MARKERS["direction"]]
    others = [
        marked[paper.MEASURE_MARKERS["random"]],
        *(marked[m] for m in paper.MEASURE_MARKERS["text"]),
    ]
    assert all(reader > z for z in others)
