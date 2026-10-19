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
SPEC = "a_dose_types"
TYPES = (
    "obscure_real",
    "fictional_generated",
    "unknowable_detail",
    "false_premise",
)
TEMPLATES = ("fewshot", "concept")


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _runs() -> list[tuple[str, str, Path]]:
    manifest = Manifest(RESULTS)
    found = []
    for model in manifest.models("p9d"):
        if not store.is_post_trained(model):
            continue
        for template in TEMPLATES:
            run = manifest.run("p9d", model, template=template)
            if run is not None:
                found.append((model, template, run))
    return found


def test_a_row_per_post_trained_run_and_a_column_per_type(
    output: registry.FigureOutput,
) -> None:
    runs = _runs()
    assert len(runs) >= 5
    assert len(output.figure.axes) == len(runs) * len(TYPES)
    assert output.figure.get_figwidth() == pytest.approx(paper.FULL_WIDTH)


def test_every_panel_draws_four_series_with_bands(
    output: registry.FigureOutput,
) -> None:
    for ax in output.figure.axes:
        assert len(ax.collections) == 4
        long = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 2]
        assert len(long) == 4


def test_the_claim_numbers_match_the_stored_curves(
    output: registry.FigureOutput,
) -> None:
    moved = panels = 0
    for _, _, run in _runs():
        for kind in TYPES:
            with np.load(run / f"curves_{kind}.npz") as npz:
                mean = np.asarray(npz["mean"], dtype=float)
            names = json.loads(
                (run / f"curves_{kind}.json").read_text("utf-8")
            )["data"]["series"]
            lat = mean[names.index("lat: unknown")]
            rand = mean[names.index("random: unknown")]
            panels += 1
            moved += np.ptp(lat) > np.ptp(rand)
    assert output.numbers["panels"] == panels
    assert output.numbers["lat_moves_more"] == moved
    assert f"{moved} of {panels}" in output.claim
    assert output.claim.endswith(".")


def test_no_layout_faults(output: registry.FigureOutput) -> None:
    assert checks.problems(output.figure) == []
