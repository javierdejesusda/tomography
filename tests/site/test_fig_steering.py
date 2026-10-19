import csv
import io
import itertools
import json
from pathlib import Path
import re
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
import numpy as np
import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[2] / "results"
SPEC = "steering"
MAX_RATIO = 1.5
MIN_DISTINCT = 0.15


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(output.csv)))


def _c1() -> dict[str, list[dict[str, Any]]]:
    found = {}
    for model in Manifest(RESULTS).models("c1"):
        (run,) = [p for p in (RESULTS / "c1" / model).iterdir() if p.is_dir()]
        payload = json.loads((run / "table.json").read_text(encoding="utf-8"))
        data = payload["data"]
        found[model] = [
            dict(zip(data["columns"], row, strict=True)) for row in data["rows"]
        ]
    return found


def _linear(rows: list[dict[str, Any]], coeff: float, key: str) -> float:
    cell = [r for r in rows if r["arm"] == "linear" and r["coeff"] == coeff]
    assert len(cell) == 3
    return float(np.mean([r[key] for r in cell]))


def _none(rows: list[dict[str, Any]]) -> dict[str, Any]:
    (row,) = [r for r in rows if r["arm"] == "none"]
    return row


def _fluent(rows: list[dict[str, Any]], coeff: float) -> bool:
    ratio = _linear(rows, coeff, "perplexity") / _none(rows)["perplexity"]
    return (
        ratio <= MAX_RATIO
        and _linear(rows, coeff, "distinct_1") >= MIN_DISTINCT
    )


def _has(claim: str, value: float) -> bool:
    text = str(int(value)) if value == int(value) else f"{value:.3f}"
    pattern = rf"(?<![\d.]){re.escape(text)}(?![\d])"
    return re.search(pattern, claim) is not None


def test_every_c1_model_has_a_line_in_each_panel(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for model in _c1():
        for series in ("score", "ppl_ratio", "distinct_1"):
            got = [
                r for r in rows if r["model"] == model and r["series"] == series
            ]
            assert len(got) == 7


def test_lines_are_seed_means_and_pass_through_the_unsteered_row(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for model, table in _c1().items():
        for series, key in (("score", "score"), ("distinct_1", "distinct_1")):
            got = {
                float(r["x"]): float(r["value"])
                for r in rows
                if r["model"] == model and r["series"] == series
            }
            assert got[0.0] == pytest.approx(_none(table)[key])
            for coeff in (-0.3, -0.1, 0.1, 0.3):
                assert got[coeff] == pytest.approx(_linear(table, coeff, key))
        ratio = {
            float(r["x"]): float(r["value"])
            for r in rows
            if r["model"] == model and r["series"] == "ppl_ratio"
        }
        assert ratio[0.0] == pytest.approx(1.0)
        assert ratio[0.1] == pytest.approx(
            _linear(table, 0.1, "perplexity") / _none(table)["perplexity"]
        )


def test_prompting_arms_are_drawn_beside_the_score_panels(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for model, table in _c1().items():
        for arm in ("prompting+1", "prompting-1"):
            (want,) = [r for r in table if r["arm"] == arm]
            (got,) = [
                r for r in rows if r["model"] == model and r["series"] == arm
            ]
            assert float(got["value"]) == pytest.approx(want["score"])


def test_claim_numbers_are_recomputed_from_the_tables(
    output: registry.FigureOutput,
) -> None:
    tables = _c1()
    models = list(tables)
    expected = {
        "models": len(models),
        "score_unsteered": float(
            np.mean([_none(t)["score"] for t in tables.values()])
        ),
        "score_minus": float(
            np.mean([_linear(t, -0.1, "score") for t in tables.values()])
        ),
        "score_plus": float(
            np.mean([_linear(t, 0.1, "score") for t in tables.values()])
        ),
        "fluent_at_0.1": sum(
            _fluent(t, 0.1) and _fluent(t, -0.1) for t in tables.values()
        ),
        "fluent_at_0.3": sum(
            _fluent(t, 0.3) and _fluent(t, -0.3) for t in tables.values()
        ),
    }
    assert output.numbers == pytest.approx(expected)
    assert all(_has(output.claim, v) for v in expected.values())


def test_caption_says_that_c1_has_no_random_direction_arm(
    output: registry.FigureOutput,
) -> None:
    assert "no random-direction arm" in output.detail
    assert "1.5" in output.detail
    assert "0.15" in output.detail


def _boxes(fig: Figure) -> list[tuple[str, tuple[float, float, float, float]]]:
    canvas = FigureCanvasAgg(fig)
    with paper.paper_style():
        canvas.draw()
    renderer = canvas.get_renderer()
    unique = {id(t): t for t in fig.findobj(Text)}
    out = []
    for text in unique.values():
        if text.get_text() and text.get_visible():
            box = text.get_window_extent(renderer)
            out.append((text.get_text(), (box.x0, box.y0, box.x1, box.y1)))
    return out


def test_figure_is_print_width_with_one_letter_per_panel(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    assert fig.get_size_inches()[0] == paper.FULL_WIDTH
    letters = [t for t, _ in _boxes(fig) if t.startswith(r"$\mathbf")]
    assert letters == [rf"$\mathbf{{({c})}}$" for c in "abcdef"]


def test_no_text_overlaps_or_leaves_the_figure(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    boxes = _boxes(fig)
    width, height = (fig.get_size_inches() * fig.dpi).tolist()
    for label, (x0, y0, x1, y1) in boxes:
        assert min(x0, y0) >= -0.5, label
        assert x1 <= width + 0.5, label
        assert y1 <= height + 0.5, label
    for (la, a), (lb, b) in itertools.combinations(boxes, 2):
        overlap = (
            a[0] < b[2] - 0.5
            and b[0] < a[2] - 0.5
            and a[1] < b[3] - 0.5
            and b[1] < a[3] - 0.5
        )
        assert not overlap, (la, lb)


def test_svg_and_pdf_bytes_are_the_same_for_the_same_input(
    output: registry.FigureOutput,
) -> None:
    again = registry.build_figure(SPEC, Manifest(RESULTS))
    with paper.paper_style():
        assert paper.svg_bytes(output.figure) == paper.svg_bytes(again.figure)
        assert paper.pdf_bytes(output.figure) == paper.pdf_bytes(again.figure)
