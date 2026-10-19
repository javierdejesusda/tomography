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
import pytest

from tomography.site import params
from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[2] / "results"
SPEC = "truth"
NEGATED = ("neg_cities", "neg_sp_en_trans")
NOT_SETS = ("val", "test")


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(output.csv)))


def _data(experiment: str, model: str, name: str) -> Any:
    (run,) = [p for p in (RESULTS / experiment / model).iterdir() if p.is_dir()]
    payload = json.loads((run / name).read_text(encoding="utf-8"))["data"]
    if isinstance(payload, dict) and "columns" in payload:
        return [
            dict(zip(payload["columns"], row, strict=True))
            for row in payload["rows"]
        ]
    return payload


def _models(experiment: str) -> list[str]:
    return Manifest(RESULTS).models(experiment)


def _r6() -> dict[str, dict[str, dict[str, Any]]]:
    found = {}
    for model in _models("r6"):
        rows = _data("r6", model, "table.json")
        found[model] = {
            "reader": next(
                r
                for r in rows
                if r["method"] == "lat"
                and r["layer_rule"] == "validation-layer"
                and r["source"] == "truthfulqa"
            ),
            "standard": next(r for r in rows if r["method"] == "standard"),
        }
        for text in ("text_word", "text_char"):
            hit = [r for r in rows if r["method"] == text]
            if hit:
                found[model][text] = hit[0]
    return found


def _has(claim: str, value: float) -> bool:
    text = str(int(value)) if value == int(value) else f"{value:.3f}"
    pattern = rf"(?<![\d.]){re.escape(text)}(?![\d])"
    return re.search(pattern, claim) is not None


def test_panel_a_uses_the_validation_layer_rows_only(
    output: registry.FigureOutput,
) -> None:
    rows = [r for r in _rows(output) if r["panel"] == "a"]
    for model, parts in _r6().items():
        for series in ("reader", "standard"):
            (got,) = [
                r for r in rows if r["model"] == model and r["series"] == series
            ]
            assert float(got["value"]) == pytest.approx(parts[series]["mc1"])
        (reader,) = [
            r for r in rows if r["model"] == model and r["series"] == "reader"
        ]
        assert float(reader["lo"]) == pytest.approx(parts["reader"]["lo"])
        assert float(reader["hi"]) == pytest.approx(parts["reader"]["hi"])


def test_panel_a_draws_one_band_for_the_text_baselines(
    output: registry.FigureOutput,
) -> None:
    r6 = _r6()
    with_text = [m for m, p in r6.items() if "text_word" in p]
    assert with_text
    for name in ("text_word", "text_char"):
        values = {
            (p[name]["mc1"], p[name]["lo"], p[name]["hi"])
            for p in r6.values()
            if name in p
        }
        assert len(values) == 1
    rows = [
        r
        for r in _rows(output)
        if r["panel"] == "a" and r["series"] in ("text_word", "text_char")
    ]
    assert len(rows) == 2
    word = next(r for r in rows if r["series"] == "text_word")
    expected = r6[with_text[0]]["text_word"]
    assert float(word["lo"]) == pytest.approx(expected["lo"])
    assert float(word["hi"]) == pytest.approx(expected["hi"])


def test_panel_b_pairs_the_honesty_transfer_with_the_seed_agreement(
    output: registry.FigureOutput,
) -> None:
    rows = [r for r in _rows(output) if r["panel"] == "b"]
    assert {r["model"] for r in rows} == set(_models("r5"))
    for row in rows:
        model = row["model"]
        accuracy = _data("r5", model, "summary.json")["test_accuracy"]
        agreement = _data("a1", model, "summary.json")["agreement_at_layer"][
            "r5"
        ]
        assert float(row["value"]) == pytest.approx(accuracy)
        assert float(row["x"]) == pytest.approx(agreement)


def test_panel_c_holds_the_nine_out_of_distribution_sets_per_model(
    output: registry.FigureOutput,
) -> None:
    rows = [r for r in _rows(output) if r["panel"] == "c"]
    for model in _models("r7"):
        sets = _data("r7", model, "summary.json")["sets"]
        names = [n for n in sets if n not in NOT_SETS]
        assert len(names) == 9
        got = {r["x"]: float(r["value"]) for r in rows if r["model"] == model}
        assert list(got) == names
        for name in names:
            assert got[name] == pytest.approx(
                sets[name]["accuracy_at_val_layer"]
            )


def test_claim_numbers_are_recomputed_from_the_files(
    output: registry.FigureOutput,
) -> None:
    r6 = _r6()
    band = next(p["text_word"] for p in r6.values() if "text_word" in p)
    char = next(p["text_char"] for p in r6.values() if "text_char" in p)
    low, high = min(band["lo"], char["lo"]), max(band["hi"], char["hi"])
    inside = sum(low <= p["reader"]["mc1"] <= high for p in r6.values())
    below = 0
    cells = 0
    for model in _models("r7"):
        sets = _data("r7", model, "summary.json")["sets"]
        for name in NEGATED:
            cells += 1
            below += sets[name]["accuracy_at_val_layer"] < 0.5
    accuracy = [
        _data("r5", m, "summary.json")["test_accuracy"] for m in _models("r5")
    ]
    agreement = [
        _data("a1", m, "summary.json")["agreement_at_layer"]["r5"]
        for m in _models("r5")
    ]
    expected = {
        "models": len(r6),
        "inside_text_interval": inside,
        "negated_cells": cells,
        "negated_below_chance": below,
        "honesty_min": min(accuracy),
        "honesty_max": max(accuracy),
        "agreement_min": min(agreement),
    }
    assert output.numbers == pytest.approx(expected)
    assert all(_has(output.claim, v) for v in expected.values())


def test_the_withdrawn_validation_figure_is_never_quoted(
    output: registry.FigureOutput,
) -> None:
    text = output.claim + output.detail + output.csv
    assert "0.718" not in text


def test_caption_says_what_the_intervals_resample(
    output: registry.FigureOutput,
) -> None:
    assert "15 readers" in output.detail
    assert "Wilson" in output.detail
    assert "exploratory" in output.detail


def test_sizes_come_from_the_parameter_counts(
    output: registry.FigureOutput,
) -> None:
    rows = [
        r
        for r in _rows(output)
        if r["panel"] == "a" and r["series"] == "reader"
    ]
    for row in rows:
        assert float(row["x"]) == params.param_count(row["model"])


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
    assert letters == [rf"$\mathbf{{({c})}}$" for c in "abc"]


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
