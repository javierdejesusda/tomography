import json
from pathlib import Path

import numpy as np
import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
SPEC = "wording"
DROPPED = "no_trailing_space"
MATRIX_MODEL = "qwen3-1.7b-base"


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(model: str) -> list[dict]:
    run = Manifest(RESULTS).run("p1", model)
    assert run is not None
    table = json.loads((run / "table.json").read_text("utf-8"))["data"]
    return [
        r for r in table if r["task"] == "sentiment" and r["variant"] != DROPPED
    ]


def test_rows_of_the_matrix_are_the_wording_the_reader_was_fitted_on() -> None:
    run = Manifest(RESULTS).run("p1", MATRIX_MODEL)
    assert run is not None
    name = "transfer_sentiment"
    labels = json.loads((run / f"{name}.json").read_text("utf-8"))["data"]
    with np.load(run / f"{name}.npz") as npz:
        values = np.asarray(npz["values"])
    paper_row = values[labels["rows"].index("paper")]
    for row in _rows(MATRIX_MODEL):
        column = labels["cols"].index(row["variant"])
        assert paper_row[column] == pytest.approx(row["from_paper"])


def test_the_figure_has_a_heatmap_a_colourbar_and_a_strip(
    output: registry.FigureOutput,
) -> None:
    assert len(output.figure.axes) == 4
    heat, other, _, strip = output.figure.axes
    assert len(heat.images) == len(other.images) == 1
    assert heat.images[0].get_array().shape == (16, 16)
    assert not heat.texts or all(
        t.get_text().startswith("$") for t in heat.texts
    )
    assert strip.get_xlim()[0] <= 0.5


def test_the_strip_has_a_row_per_logistic_model(
    output: registry.FigureOutput,
) -> None:
    manifest = Manifest(RESULTS)
    models = [m for m in manifest.models("p1") if manifest.run("p1", m)]
    strip = output.figure.axes[3]
    labels = [t.get_text() for t in strip.get_yticklabels()]
    assert labels == [paper.display_name(m) for m in models]


def test_the_claim_numbers_match_a_recomputation(
    output: registry.FigureOutput,
) -> None:
    manifest = Manifest(RESULTS)
    spans = []
    cosines = []
    for model in manifest.models("p1"):
        rows = _rows(model)
        values = [r["from_paper"] for r in rows if r["from_paper"] is not None]
        spans.append(max(values) - min(values))
        cosines += [r["cosine_paper"] for r in rows if r["variant"] != "paper"]
    assert output.numbers["span_min"] == pytest.approx(min(spans))
    assert output.numbers["span_max"] == pytest.approx(max(spans))
    assert output.numbers["cosine_min"] == pytest.approx(min(cosines))
    assert output.numbers["cosine_max"] == pytest.approx(max(cosines))
    for name in ("span_min", "span_max"):
        assert f"{output.numbers[name]:.3f}" in output.claim
    for name in ("cosine_min", "cosine_max"):
        assert f"{output.numbers[name]:.2f}" in output.claim
    assert output.claim.endswith(".")


def test_nearly_orthogonal_is_not_claimed_where_cosines_reach_one(
    output: registry.FigureOutput,
) -> None:
    assert output.numbers["cosine_max"] > 0.5
    assert "orthogonal" not in output.claim


def test_no_layout_faults(output: registry.FigureOutput) -> None:
    assert checks.problems(output.figure) == []
