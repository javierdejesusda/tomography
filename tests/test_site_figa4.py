import json
from pathlib import Path

import numpy as np
import pytest

from tomography.site import registry
from tomography.site import store
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
SPEC = registry.APPENDIX[3]
SLACK = 0.02


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _model() -> str:
    manifest = Manifest(RESULTS)
    bases = [m for m in manifest.models("p9b") if not store.is_post_trained(m)]
    return max(bases, key=store.model_rank)


def _table(model: str) -> list[dict]:
    run = Manifest(RESULTS).run("p9b", model)
    assert run is not None
    data = json.loads((run / "table.json").read_text("utf-8"))["data"]
    return [dict(zip(data["columns"], r, strict=True)) for r in data["rows"]]


def test_the_matrix_is_the_largest_base_model_and_is_not_annotated_away(
    output: registry.FigureOutput,
) -> None:
    ax = output.figure.axes[0]
    assert ax.images[0].get_array().shape == (14, 14)
    assert paper.display_name(_model()) in output.figure.axes[0].get_title(
        loc="left"
    )


def test_every_cell_shows_its_stored_auroc(
    output: registry.FigureOutput,
) -> None:
    ax = output.figure.axes[0]
    run = Manifest(RESULTS).run("p9b", _model())
    assert run is not None
    with np.load(run / "matrix.npz") as npz:
        values = np.asarray(npz["values"], dtype=float)
    cells = [t for t in ax.texts if not t.get_text().startswith("$")]
    assert len(cells) == 196
    shown = np.array([float(t.get_text()) for t in cells]).reshape(14, 14)
    assert shown == pytest.approx(values, abs=0.0051)


def test_flagged_types_are_those_a_text_classifier_nearly_matches(
    output: registry.FigureOutput,
) -> None:
    rows = _table(_model())
    flagged = [
        r["type"]
        for r in rows
        if max(r["text_word"] or 0.0, r["text_char"] or 0.0)
        >= r["logistic"] - SLACK
    ]
    labels = [t.get_text() for t in output.figure.axes[0].get_xticklabels()]
    starred = [lab for lab in labels if lab.endswith("*")]
    assert len(starred) == len(flagged)
    assert output.numbers["flagged"] == len(flagged)
    assert output.numbers["types"] == len(rows)
    assert f"{len(flagged)} of {len(rows)}" in output.claim
    assert output.claim.endswith(".")


def test_no_layout_faults(output: registry.FigureOutput) -> None:
    assert checks.problems(output.figure) == []
