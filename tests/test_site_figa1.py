from pathlib import Path

import numpy as np
import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
SPEC = registry.APPENDIX[0]
CONCEPTS = ("r1", "r2", "r3", "r7")
MARGIN = 0.1


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _series(exp: str, model: str) -> np.ndarray:
    run = Manifest(RESULTS).run(exp, model)
    assert run is not None
    with np.load(run / "curves.npz") as npz:
        return np.asarray(npz["mean"], dtype=float)


def test_twelve_panels_one_per_concept_and_family(
    output: registry.FigureOutput,
) -> None:
    assert len(output.figure.axes) == 12
    assert output.figure.get_figwidth() == pytest.approx(paper.FULL_WIDTH)


def test_each_panel_draws_three_series_per_model(
    output: registry.FigureOutput,
) -> None:
    counts = [len(ax.get_lines()) for ax in output.figure.axes]
    assert counts[:3] == [13, 10, 10]
    assert len(set(counts[::3])) == 1


def test_the_claim_numbers_match_the_stored_curves(
    output: registry.FigureOutput,
) -> None:
    manifest = Manifest(RESULTS)
    above = cells = 0
    worst = 0.0
    for exp in CONCEPTS:
        for model in manifest.models(exp):
            mean = _series(exp, model)
            cells += 1
            above += (mean[0][-1] - mean[1][-1]) > MARGIN
            worst = max(worst, float(np.abs(mean[2] - 0.5).max()))
    assert output.numbers["cells"] == cells
    assert output.numbers["above_at_last_layer"] == above
    assert output.numbers["shuffled_max_gap"] == pytest.approx(worst)
    assert f"{above} of {cells}" in output.claim
    assert f"{worst:.3f}" in output.claim
    assert output.claim.endswith(".")


def test_the_csv_has_every_layer_of_every_series(
    output: registry.FigureOutput,
) -> None:
    lines = output.csv.strip().splitlines()
    assert lines[0] == "panel,model,x,y,lo,hi,baseline"
    manifest = Manifest(RESULTS)
    expected = sum(
        len(_series(e, m)[0]) * 3 for e in CONCEPTS for m in manifest.models(e)
    )
    assert len(lines) - 1 == expected


def test_no_layout_faults(output: registry.FigureOutput) -> None:
    assert checks.problems(output.figure) == []
