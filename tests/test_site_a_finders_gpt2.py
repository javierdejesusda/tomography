import json
from pathlib import Path

import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
SPEC = "a_finders_gpt2"
TASKS = ("sentiment", "truth", "utility")
BAND = 0.05


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _report(model: str, task: str, method: str) -> float:
    run = Manifest(RESULTS).run("p4", model)
    assert run is not None
    data = json.loads((run / "summary.json").read_text("utf-8"))["data"]
    pairing = data["reference"][task]["pairing"]
    table = json.loads((run / "table.json").read_text("utf-8"))["data"]
    rows = [
        r
        for r in table
        if (r["task"], r["method"], r["pairing"], r["preprocess"])
        == (task, method, pairing, "center")
    ]
    assert len(rows) == 1
    return float(rows[0]["report"])


def test_a_panel_per_p4_model_and_task(output: registry.FigureOutput) -> None:
    models = Manifest(RESULTS).models("p4")
    assert len(output.figure.axes) == len(models) * 3
    assert output.figure.get_figwidth() == pytest.approx(paper.FULL_WIDTH)


def test_all_six_finders_are_drawn_with_the_random_band(
    output: registry.FigureOutput,
) -> None:
    for ax in output.figure.axes:
        assert len(ax.collections) == 1
        long = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 2]
        assert len(long) == 6


def test_the_claim_numbers_match_the_stored_tables(
    output: registry.FigureOutput,
) -> None:
    models = Manifest(RESULTS).models("p4")
    cells = [(m, t) for m in models for t in TASKS]
    raw = sum(
        abs(_report(m, t, "pca_raw") - _report(m, t, "pca_diff")) <= BAND
        for m, t in cells
    )
    kmeans = sum(
        abs(_report(m, t, "kmeans") - _report(m, t, "pca_diff")) <= BAND
        for m, t in cells
    )
    assert output.numbers["cells"] == len(cells)
    assert output.numbers["pca_raw_within"] == raw
    assert output.numbers["kmeans_within"] == kmeans
    assert f"{raw} of {len(cells)}" in output.claim
    assert f"{kmeans} of {len(cells)}" in output.claim
    assert output.claim.endswith(".")


def test_no_layout_faults(output: registry.FigureOutput) -> None:
    assert checks.problems(output.figure) == []
