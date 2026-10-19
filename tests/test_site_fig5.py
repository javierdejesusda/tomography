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

RESULTS = Path(__file__).resolve().parents[1] / "results"
SPEC = registry.FIGURES[4]
RISK_MODEL = "qwen3-4b-base"


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(output.csv)))


def _table(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))["data"]
    if isinstance(payload, dict):
        return [
            dict(zip(payload["columns"], row, strict=True))
            for row in payload["rows"]
        ]
    return list(payload)


def _models(experiment: str) -> list[str]:
    return Manifest(RESULTS).models(experiment)


def _run(experiment: str, model: str) -> Path:
    runs = [p for p in (RESULTS / experiment / model).iterdir() if p.is_dir()]
    assert len(runs) == 1
    return runs[0]


def _p9b() -> dict[str, dict[str, Any]]:
    found = {}
    for model in _models("p9b"):
        rows = [
            r
            for r in _table(_run("p9b", model) / "table.json")
            if r["type"] == "obscure_real"
        ]
        if rows:
            found[model] = rows[0]
    return found


def _p9c() -> dict[str, dict[str, dict[str, Any]]]:
    found = {}
    for model in _models("p9c"):
        rows = _table(_run("p9c", model) / "table.json")
        found[model] = {
            r["predictor"]: r for r in rows if r["dataset"] == "popqa"
        }
    return found


def _has(claim: str, value: float) -> bool:
    text = str(int(value)) if value == int(value) else f"{value:.3f}"
    pattern = rf"(?<![\d.]){re.escape(text)}(?![\d])"
    return re.search(pattern, claim) is not None


def test_panel_a_holds_every_model_with_a_known_unknown_contrast(
    output: registry.FigureOutput,
) -> None:
    expected = _p9b()
    assert "gpt2" not in expected
    rows = [r for r in _rows(output) if r["panel"] == "a"]
    for series, column in (
        ("reader", "logistic_within"),
        ("popularity", "popularity_within"),
        ("text_word", "text_word_within"),
        ("text_char", "text_char_within"),
    ):
        got = {
            r["model"]: float(r["value"]) for r in rows if r["series"] == series
        }
        assert got.keys() == expected.keys()
        for model, row in expected.items():
            assert got[model] == pytest.approx(row[column])


def test_panel_a_draws_no_interval_because_it_resamples_reader_draws(
    output: registry.FigureOutput,
) -> None:
    rows = [r for r in _rows(output) if r["panel"] == "a"]
    assert all(r["lo"] == "" and r["hi"] == "" for r in rows)


def test_panel_b_lists_the_four_predictors_and_leaves_gpt2_without_lat(
    output: registry.FigureOutput,
) -> None:
    expected = _p9c()
    rows = [r for r in _rows(output) if r["panel"] == "b"]
    for series in ("answer_logprob", "probe", "lat", "popularity"):
        got = {r["model"]: r for r in rows if r["series"] == series}
        want = {m: v[series] for m, v in expected.items() if series in v}
        assert got.keys() == want.keys()
        for model, row in want.items():
            assert float(got[model]["value"]) == pytest.approx(
                row["auroc_within"]
            )
            assert float(got[model]["lo"]) == pytest.approx(row["lo_within"])
            assert float(got[model]["hi"]) == pytest.approx(row["hi_within"])
    assert "gpt2" not in {r["model"] for r in rows if r["series"] == "lat"}


def test_panel_c_reads_the_risk_coverage_of_one_model(
    output: registry.FigureOutput,
) -> None:
    run = _run("p9c", RISK_MODEL)
    names = json.loads(
        (run / "riskcov_popqa_outputs.json").read_text(encoding="utf-8")
    )["data"]["series"]
    with np.load(run / "riskcov_popqa_outputs.npz") as arrays:
        coverage = arrays["x"]
        risk = arrays["mean"][names.index("answer_logprob")]
    rows = [
        r
        for r in _rows(output)
        if r["panel"] == "c" and r["series"] == "answer_logprob"
    ]
    assert {r["model"] for r in rows} == {RISK_MODEL}
    assert [float(r["x"]) for r in rows] == pytest.approx(list(coverage))
    assert [float(r["value"]) for r in rows] == pytest.approx(list(risk))
    base = _p9c()[RISK_MODEL]["answer_logprob"]
    chance = [r for r in _rows(output) if r["series"] == "random_order"]
    assert float(chance[0]["value"]) == pytest.approx(
        base["n_error"] / base["n"]
    )
    assert risk[-1] == pytest.approx(base["n_error"] / base["n"])


def test_claim_numbers_are_recomputed_from_the_files(
    output: registry.FigureOutput,
) -> None:
    a, c = _p9b(), _p9c()
    above_pop = sum(
        r["logistic_within"] > r["popularity_within"] for r in a.values()
    )
    above_text = sum(
        r["logistic_within"] > max(r["text_word_within"], r["text_char_within"])
        for r in a.values()
    )
    with_lat = {m: v for m, v in c.items() if "lat" in v}
    gaps = [
        v["answer_logprob"]["auroc_within"] - v["lat"]["auroc_within"]
        for v in with_lat.values()
    ]
    expected = {
        "models_reader": len(a),
        "reader_above_popularity": above_pop,
        "reader_above_text": above_text,
        "models_lat": len(with_lat),
        "lat_below_logprob": sum(g > 0 for g in gaps),
        "gap_min": min(gaps),
        "gap_max": max(gaps),
    }
    assert output.numbers == pytest.approx(expected)
    assert all(_has(output.claim, v) for v in expected.values())
    assert output.claim.endswith(".")


def test_caption_says_what_each_interval_resamples(
    output: registry.FigureOutput,
) -> None:
    assert "resample" in output.detail
    assert "risk" in output.detail.lower()


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
