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

from tomography.site import params
from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
SPEC = registry.FIGURES[5]


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(output.csv)))


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))["data"]


def _runs() -> list[dict[str, Any]]:
    found = []
    for run in sorted(RESULTS.glob("p9d/*/*")):
        summary = _json(run / "summary.json")
        found.append(
            {
                "model": run.parent.name,
                "template": summary["template"],
                "path": run,
                "headline": summary["headline"],
            }
        )
    return found


def _has(claim: str, value: float) -> bool:
    text = str(int(value)) if value == int(value) else f"{value:.3f}"
    pattern = rf"(?<![\d.]){re.escape(text)}(?![\d])"
    return re.search(pattern, claim) is not None


def _dose_runs() -> list[dict[str, Any]]:
    fewshot = [r for r in _runs() if r["template"] == "fewshot"]
    entropy = [r for r in fewshot if r["headline"]["metric"] == "entropy"]
    post = [r for r in fewshot if r["headline"]["metric"] == "abstention"]
    largest = max(post, key=lambda r: params.param_count(r["model"]))
    return [*entropy, largest]


def test_claim_counts_come_from_the_headline_of_every_run(
    output: registry.FigureOutput,
) -> None:
    runs = _runs()
    distinguishable = sum(bool(r["headline"]["distinguishable"]) for r in runs)
    selective = sum(bool(r["headline"]["selective"]) for r in runs)
    post = [
        r
        for r in runs
        if not r["headline"]["distinguishable"]
        and not r["model"].startswith("gpt2")
        and not r["model"].endswith("-base")
    ]
    expected = {
        "runs": len(runs),
        "distinguishable": distinguishable,
        "selective": selective,
        "not_distinguishable": len(runs) - distinguishable,
        "not_distinguishable_post_trained": len(post),
    }
    assert output.numbers == pytest.approx(expected)
    assert all(_has(output.claim, v) for v in expected.values())


def test_summary_panel_has_one_unit_free_ratio_per_run(
    output: registry.FigureOutput,
) -> None:
    rows = [r for r in _rows(output) if r["panel"] == "h"]
    runs = _runs()
    assert len(rows) == len(runs)
    ratios = sorted(float(r["value"]) for r in rows)
    expected = sorted(
        r["headline"]["lat_slope"] / r["headline"]["max_abs_random_slope"]
        for r in runs
    )
    assert ratios == pytest.approx(expected)


def test_dose_panels_hold_the_entropy_runs_and_the_largest_post_trained_run(
    output: registry.FigureOutput,
) -> None:
    expected = _dose_runs()
    rows = [r for r in _rows(output) if r["panel"] != "h"]
    assert {r["run"] for r in rows} == {
        f"{r['model']} {r['template']}" for r in expected
    }
    for run in expected:
        stem = (
            "curves_obscure_real_entropy"
            if run["headline"]["metric"] == "entropy"
            else "curves_obscure_real"
        )
        names = _json(run["path"] / f"{stem}.json")["series"]
        with np.load(run["path"] / f"{stem}.npz") as arrays:
            xs, mean = arrays["x"], arrays["mean"]
        key = f"{run['model']} {run['template']}"
        for name in names:
            got = [r for r in rows if r["run"] == key and r["series"] == name]
            assert [float(r["x"]) for r in got] == pytest.approx(list(xs))
            assert [float(r["value"]) for r in got] == pytest.approx(
                list(mean[names.index(name)])
            )


def test_panels_are_the_dose_runs_then_the_summary(
    output: registry.FigureOutput,
) -> None:
    panels = {r["panel"] for r in _rows(output)}
    assert panels == {*"abcdefg", "h"}


def test_caption_names_the_ratio_and_what_the_bands_resample(
    output: registry.FigureOutput,
) -> None:
    assert "largest" in output.detail
    assert "resample" in output.detail
    assert "random" in output.detail


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
    assert letters == [rf"$\mathbf{{({c})}}$" for c in "abcdefgh"]


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
