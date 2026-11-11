import csv
import io
from pathlib import Path
import re
import runpy
from typing import Any

from matplotlib.lines import Line2D
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from tomography.site import registry
from tomography.site import stats
from tomography.site.figs import a_knowledge_geometry
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
_L = runpy.run_path(str(Path(__file__).with_name("_legends.py")))
SPEC = "a_knowledge_geometry"
PANEL = "entity"
SHOWN = ("gpt2-xl", "qwen3-1.7b-base", "qwen3-1.7b")
VIEWS = ("layer 1", "reader layer", "last layer", "words only")
MAX_DRAWN = 40
SVG_BUDGET = 300_000


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(output.csv)))


def _with_panel() -> list[str]:
    return [
        m
        for m in _R["models"]("v1")
        if PANEL in _R["data"](_R["run"]("v1", m), "projections.json")["panels"]
    ]


def _groups(model: str) -> list[tuple[str, str, list[int]]]:
    """Reads the stored rows of the known and of each unknown group."""
    sidecar = _R["data"](_R["run"]("v1", model), "projections.json")
    panel = sidecar["panels"][PANEL]
    index = {uid: i for i, uid in enumerate(panel["uids"])}
    known: dict[str, None] = {}
    for name in panel["types"]:
        known.update(dict.fromkeys(sidecar["uids"][name]["positive"]))
    found = [("known", "known", [index[u] for u in known])]
    for name in panel["types"]:
        rows = [index[u] for u in sidecar["uids"][name]["negative"]]
        found.append(("unknown", name, rows))
    return found


def _views(model: str) -> dict[str, tuple[int | None, np.ndarray]]:
    """Reads the stored coordinates of each view, by the name of the view."""
    path = _R["run"]("v1", model)
    chosen = _R["data"](path, "summary.json")["layer"]
    pca = _R["npz"](path, "pca2d.npz")
    found: dict[str, tuple[int | None, np.ndarray]] = {}
    for v, layer in enumerate(pca["view_layers"].tolist()):
        if layer == chosen:
            name = "reader layer"
        else:
            name = "layer 1" if v == 0 else "last layer"
        found[name] = (layer, pca[f"coordinates_{PANEL}"][v])
    words = _R["npz"](path, "text2d.npz")[f"coordinates_{PANEL}"]
    found["words only"] = (None, words)
    return found


def _separation(model: str, view: str) -> float:
    groups = _groups(model)
    points = _views(model)[view][1]
    rows = [i for _, _, found in groups for i in found]
    labels = [int(s == "unknown") for s, _, found in groups for _ in found]
    best = 0.0
    for axis in (0, 1):
        area = float(roc_auc_score(labels, points[rows, axis]))
        best = max(best, area, 1.0 - area)
    return best


def _range(view: str) -> tuple[float, float]:
    found = [_separation(m, view) for m in SHOWN]
    return min(found), max(found)


def _axes(output: registry.FigureOutput, row: int, column: int) -> Any:
    return output.figure.axes[row * len(VIEWS) + column]


def _drawn(ax: Any, series: str) -> Line2D:
    (line,) = [ln for ln in ax.lines if ln.get_gid() == series]
    return line


def test_the_shown_checkpoints_are_one_per_role() -> None:
    assert a_knowledge_geometry.SHOWN == SHOWN
    assert [stats.role(m) for m in SHOWN] == list(stats.ROLES)
    assert set(SHOWN) <= set(_with_panel())


def test_the_claim_quotes_the_separation_of_each_view(
    output: registry.FigureOutput,
) -> None:
    reader, early, words = (
        _range("reader layer"),
        _range("layer 1"),
        _range("words only"),
    )
    last = _range("last layer")
    assert output.numbers["reader_min"] == pytest.approx(reader[0])
    assert output.numbers["reader_max"] == pytest.approx(reader[1])
    assert output.numbers["early_min"] == pytest.approx(early[0])
    assert output.numbers["early_max"] == pytest.approx(early[1])
    assert output.numbers["last_min"] == pytest.approx(last[0])
    assert output.numbers["last_max"] == pytest.approx(last[1])
    assert output.numbers["words_min"] == pytest.approx(words[0])
    assert output.numbers["words_max"] == pytest.approx(words[1])
    assert output.claim == (
        "On the three checkpoints shown, the better of the two leading "
        "principal components of the hidden states at the reader layer "
        "separates known from unknown entity questions with an AUROC of "
        f"{reader[0]:.2f} to {reader[1]:.2f}, against {early[0]:.2f} to "
        f"{early[1]:.2f} at layer 1 and {words[0]:.2f} to {words[1]:.2f} "
        "for the better of two components of the words alone."
    )
    assert reader[0] > words[1]


def test_the_counts_are_per_role_and_cover_the_runs_that_exist(
    output: registry.FigureOutput,
) -> None:
    models = _with_panel()
    assert len(models) == 7
    above = {
        m: _separation(m, "reader layer") > _separation(m, "words only")
        for m in models
    }
    assert output.counts["reader_above_words"] == stats.per_role(above)
    every = {m: m in models for m in _R["MANIFEST"].checkpoints()}
    assert output.counts["with_views"] == stats.per_role(every)
    assert output.numbers["per_group"] == MAX_DRAWN
    sizes = [sum(len(found) for _, _, found in _groups(m)) for m in SHOWN]
    assert output.numbers["stored_min"] == min(sizes)
    assert output.numbers["stored_max"] == max(sizes)


def test_size_guards_and_layout(output: registry.FigureOutput) -> None:
    fig = output.figure
    assert tuple(fig.get_size_inches()) == (
        paper.FULL_WIDTH,
        a_knowledge_geometry.HEIGHT,
    )
    assert len(fig.axes) == len(SHOWN) * len(VIEWS)
    assert checks.guard_problems(fig, appendix=True) == []
    assert checks.problems(fig) == []


def test_svg_and_pdf_bytes_are_the_same_for_the_same_input() -> None:
    first = registry.build_figure(SPEC, Manifest(RESULTS))
    second = registry.build_figure(SPEC, Manifest(RESULTS))
    with paper.paper_style():
        assert paper.svg_bytes(first.figure) == paper.svg_bytes(second.figure)
        assert paper.pdf_bytes(first.figure) == paper.pdf_bytes(second.figure)


def test_the_points_are_capped_so_the_svg_stays_small(
    output: registry.FigureOutput,
) -> None:
    assert a_knowledge_geometry.MAX_DRAWN == MAX_DRAWN
    with paper.paper_style():
        assert len(paper.svg_bytes(output.figure)) < SVG_BUDGET


@pytest.mark.parametrize("row", range(len(SHOWN)))
@pytest.mark.parametrize("column", range(len(VIEWS)))
def test_each_panel_draws_the_first_stored_questions_of_each_group(
    output: registry.FigureOutput, row: int, column: int
) -> None:
    model, view = SHOWN[row], VIEWS[column]
    points = _views(model)[view][1]
    want: dict[str, list[int]] = {"known": [], "unknown": []}
    for series, _, found in _groups(model):
        want[series] += found[:MAX_DRAWN]
    ax = _axes(output, row, column)
    for series, rows in want.items():
        line = _drawn(ax, series)
        assert np.asarray(line.get_xydata()) == pytest.approx(points[rows])
    assert len(want["known"]) == MAX_DRAWN
    assert len(want["unknown"]) == 2 * MAX_DRAWN


@pytest.mark.parametrize("row", range(len(SHOWN)))
def test_classes_differ_by_hue_and_marker_and_fill_keeps_the_role(
    output: registry.FigureOutput, row: int
) -> None:
    post = stats.role(SHOWN[row]) == "post"
    for column in range(len(VIEWS)):
        ax = _axes(output, row, column)
        for series in ("known", "unknown"):
            line = _drawn(ax, series)
            hue = paper.ENC[series]
            assert line.get_marker() == paper.QUESTION_SET_MARKERS[series]
            assert line.get_linestyle() == "None"
            low, high = paper.DENSE_MARKER_RANGE
            assert low <= line.get_markersize() <= high
            if post:
                assert line.get_markerfacecolor() == "white"
                assert line.get_markeredgecolor() == hue
            else:
                assert line.get_markerfacecolor() == hue
                assert line.get_markeredgecolor() == "white"
    assert paper.ENC["known"] != paper.ENC["unknown"]


@pytest.mark.parametrize("row", range(len(SHOWN)))
def test_each_panel_names_its_layer_and_its_separation(
    output: registry.FigureOutput, row: int
) -> None:
    model = SHOWN[row]
    for column, view in enumerate(VIEWS):
        ax = _axes(output, row, column)
        layer = _views(model)[view][0]
        named = [t.get_text() for t in ax.texts if t.get_gid() == "layer"]
        assert named == ([] if layer is None else [f"layer {layer}"])
        (area,) = [t.get_text() for t in ax.texts if t.get_gid() == "auroc"]
        assert area == f"AUROC {_separation(model, view):.2f}"


def test_columns_are_lettered_views_and_rows_are_named_checkpoints(
    output: registry.FigureOutput,
) -> None:
    checks.draw(output.figure)
    for column, view in enumerate(VIEWS):
        top = _axes(output, 0, column)
        letters = [
            t.get_text()
            for t in top.texts
            if t.get_text().startswith(r"$\mathbf")
        ]
        assert letters == [rf"$\mathbf{{({'abcd'[column]})}}$"]
        assert top.get_title(loc="left") == view
        bottom = _axes(output, len(SHOWN) - 1, column)
        assert bottom.get_xlabel() == "component 1"
    for row, model in enumerate(SHOWN):
        left = _axes(output, row, 0)
        assert left.get_ylabel() == (
            f"{paper.display_name(model)}\ncomponent 2"
        )
    for ax in output.figure.axes:
        assert list(ax.get_xticks()) == []
        assert list(ax.get_yticks()) == []


def test_the_legend_keys_both_classes_with_a_filled_and_an_open_marker(
    output: registry.FigureOutput,
) -> None:
    (legend,) = output.figure.legends
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == [
        "known question (filled: GPT-2 and base; open: post-trained)",
        "unknown question, same fill",
    ]
    keys = {
        (str(ln.get_marker()), str(ln.get_markerfacecolor()))
        for ln in legend.findobj(Line2D)
    }
    assert keys == {
        (paper.QUESTION_SET_MARKERS[series], face)
        for series in ("known", "unknown")
        for face in (paper.ENC[series], "white")
    }
    assert _L["marker_clashes"](output.figure) == []


def test_the_data_file_holds_every_stored_question_of_every_checkpoint(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    assert list(rows[0]) == a_knowledge_geometry.HEADER
    models = _with_panel()
    assert sorted({r["checkpoint"] for r in rows}) == sorted(models)
    assert "gpt2" not in models
    for model in models:
        views = _views(model)
        groups = _groups(model)
        mine = [r for r in rows if r["checkpoint"] == model]
        assert {r["view"] for r in mine} == set(views)
        for view, (layer, points) in views.items():
            here = [r for r in mine if r["view"] == view]
            assert {r["layer"] for r in here} == {
                "" if layer is None else str(layer)
            }
            (area,) = [r for r in here if r["series"] == "separation"]
            assert float(area["auroc"]) == pytest.approx(
                _separation(model, view), abs=1e-6
            )
            for series, name, found in groups:
                got = [
                    r
                    for r in here
                    if r["series"] == series and r["type"] == name
                ]
                assert len(got) == len(found)
                assert [int(r["drawn"]) for r in got] == [
                    int(i < MAX_DRAWN) for i in range(len(found))
                ]
                stored = np.array([[float(r["x"]), float(r["y"])] for r in got])
                assert stored == pytest.approx(
                    points[found], rel=1e-3, abs=1e-6
                )


def test_the_caption_names_the_views_the_controls_and_the_cap(
    output: registry.FigureOutput,
) -> None:
    detail = output.detail
    for name in ("GPT-2 XL", "Qwen3-1.7B-Base", "Qwen3-1.7B"):
        assert name in detail
    for phrase in (
        "principal components",
        "an early control",
        "a second control",
        "latent semantic analysis",
        "Layer 0 is not drawn",
        f"at most {MAX_DRAWN} questions",
        "PopQA",
        "better",
        stats.per_role_text(output.counts["with_views"]),
    ):
        assert phrase in detail, phrase
    text = f"{output.claim} {detail}"
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text
    assert "_" not in text
    assert "baseline" not in text.lower()
    assert not re.search(r"\b10 checkpoints\b", text)


def test_the_module_docstring_names_no_figure_number() -> None:
    doc = a_knowledge_geometry.__doc__ or ""
    assert doc
    assert not re.search(r"Figure A?\d", doc)
