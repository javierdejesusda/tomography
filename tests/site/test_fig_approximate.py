import csv
import io
import itertools
from pathlib import Path
import re
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[2] / "results"
SPEC = "approximate"
CELL = ("free", "fewshot", "range_last", "greedy")
TRAIN = "height+wingspan"


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(output.csv)))


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run(experiment: str, model: str) -> Path:
    (run,) = [p for p in (RESULTS / experiment / model).iterdir() if p.is_dir()]
    return run


def _models(experiment: str) -> list[str]:
    return Manifest(RESULTS).models(experiment)


def _behavior() -> dict[str, dict[str, Any]]:
    found = {}
    for model in _models("p10a"):
        run = _run("p10a", model)
        table = {
            r["kind"]: r
            for r in _read(run / "table.csv")
            if r["family"] == "all"
            and (r["condition"], r["format"], r["shot_order"], r["decoding"])
            == CELL
        }
        (diag,) = [
            r
            for r in _read(run / "diagnostic.csv")
            if (r["condition"], r["format"], r["shot_order"], r["decoding"])
            == CELL
        ]
        found[model] = {
            "range": table["range"],
            "exact": table["exact"],
            "people": diag,
        }
    return found


def _duration() -> dict[str, dict[str, Any]]:
    found = {}
    for model in _models("p10b"):
        run = _run("p10b", model)
        cell = [
            r
            for r in _read(run / "transfer.csv")
            if r["template"] == "concept"
            and r["train_set"] == TRAIN
            and r["test_family"] == "duration"
        ]
        (text,) = [
            r
            for r in _read(run / "baselines.csv")
            if r["baseline"] == "text_word"
            and r["train_set"] == TRAIN
            and r["test_family"] == "duration"
        ]
        people = [
            r
            for r in _read(run / "people.csv")
            if r["source"] == "reader"
            and r["template"] == "concept"
            and r["train_set"] == TRAIN
        ]
        found[model] = {
            "reader": next(r for r in cell if r["method"] == "logistic"),
            "random": next(r for r in cell if r["method"] == "random"),
            "text": text,
            "position": next(r for r in people if r["method"] == "logistic"),
            "random_position": next(
                r for r in people if r["method"] == "random"
            ),
        }
    return found


def _has(claim: str, value: float) -> bool:
    text = str(int(value)) if value == int(value) else f"{value:.3f}"
    pattern = rf"(?<![\d.]){re.escape(text)}(?![\d])"
    return re.search(pattern, claim) is not None


def test_panel_a_takes_wilson_whiskers_from_the_table_csv(
    output: registry.FigureOutput,
) -> None:
    rows = [r for r in _rows(output) if r["panel"] == "a"]
    for model, parts in _behavior().items():
        for series, row, prefix in (
            ("range_items", parts["range"], "range"),
            ("exact_items", parts["exact"], "range"),
            ("people", parts["people"], "people"),
        ):
            (got,) = [
                r for r in rows if r["model"] == model and r["series"] == series
            ]
            assert float(got["value"]) == pytest.approx(
                float(row[f"{prefix}_rate"])
            )
            assert float(got["lo"]) == pytest.approx(float(row[f"{prefix}_lo"]))
            assert float(got["hi"]) == pytest.approx(float(row[f"{prefix}_hi"]))


def test_panel_b_is_the_held_out_duration_cell_of_the_concept_template(
    output: registry.FigureOutput,
) -> None:
    rows = [r for r in _rows(output) if r["panel"] == "b"]
    for model, parts in _duration().items():
        for series, row in (
            ("reader", parts["reader"]),
            ("random", parts["random"]),
            ("text_word", parts["text"]),
        ):
            (got,) = [
                r for r in rows if r["model"] == model and r["series"] == series
            ]
            assert float(got["value"]) == pytest.approx(float(row["auroc"]))


def test_panel_c_places_the_people_between_buildings_and_breeds(
    output: registry.FigureOutput,
) -> None:
    rows = [r for r in _rows(output) if r["panel"] == "c"]
    for model, parts in _duration().items():
        for series, key in (
            ("position_reader", "position"),
            ("position_random", "random_position"),
        ):
            (got,) = [
                r for r in rows if r["model"] == model and r["series"] == series
            ]
            assert float(got["value"]) == pytest.approx(
                float(parts[key]["position"])
            )


def test_claim_numbers_are_recomputed_from_the_files(
    output: registry.FigureOutput,
) -> None:
    behavior = _behavior()
    qwen = [m for m in behavior if not m.startswith("gpt2")]
    range_rate = [float(behavior[m]["range"]["range_rate"]) for m in qwen]
    exact_rate = [float(behavior[m]["exact"]["range_rate"]) for m in qwen]
    people_rate = [float(behavior[m]["people"]["people_rate"]) for m in qwen]
    duration = _duration()
    text = float(next(iter(duration.values()))["text"]["auroc"])
    expected = {
        "qwen_models": len(qwen),
        "range_min": round(100 * min(range_rate)),
        "range_max": round(100 * max(range_rate)),
        "exact_max": round(100 * max(exact_rate)),
        "people_max": round(100 * max(people_rate)),
        "models": len(duration),
        "reader_above_text": sum(
            float(d["reader"]["auroc"]) > float(d["text"]["auroc"])
            for d in duration.values()
        ),
        "text_word": text,
        "random_high": sum(
            float(d["random"]["auroc"]) >= 0.9 for d in duration.values()
        ),
    }
    assert output.numbers == pytest.approx(expected)
    assert all(_has(output.claim, v) for v in expected.values())


def test_caption_names_the_intervals_and_the_people_diagnostic(
    output: registry.FigureOutput,
) -> None:
    assert "Wilson" in output.detail
    assert "resample" in output.detail
    assert "diagnostic" in output.detail


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
