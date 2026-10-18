import csv
from pathlib import Path

import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
SPEC = registry.APPENDIX[5]


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(model: str, name: str) -> list[dict[str, str]]:
    run = Manifest(RESULTS).run("p10c", model)
    assert run is not None
    with (run / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _counts() -> tuple[int, int, int, int]:
    manifest = Manifest(RESULTS)
    cos_out = cos_all = auc_out = auc_all = 0
    for model in manifest.models("p10c"):
        for row in _rows(model, "cosine.csv"):
            cos_all += 1
            value = float(row["cosine"])
            cos_out += (
                not float(row["band_lo"]) <= value <= float(row["band_hi"])
            )
        for row in _rows(model, "knowledge.csv"):
            if row["items"] != "all" or row["test_family"] != "pooled":
                continue
            auc_all += 1
            auc_out += not (
                float(row["lo"]) <= float(row["rand_hi"])
                and float(row["hi"]) >= float(row["rand_lo"])
            )
    return cos_out, cos_all, auc_out, auc_all


def test_a_cosine_and_an_auroc_panel_per_model(
    output: registry.FigureOutput,
) -> None:
    models = Manifest(RESULTS).models("p10c")
    images = [ax for ax in output.figure.axes if ax.images]
    assert len(images) == 2 * len(models)
    assert output.figure.get_figwidth() == pytest.approx(paper.FULL_WIDTH)


def test_values_inside_the_random_band_are_masked(
    output: registry.FigureOutput,
) -> None:
    cos_out, cos_all, auc_out, auc_all = _counts()
    images = [ax for ax in output.figure.axes if ax.images]
    shown_cos = sum(
        int((~ax.images[0].get_array().mask).sum()) for ax in images[0::2]
    )
    shown_auc = sum(
        int((~ax.images[0].get_array().mask).sum()) for ax in images[1::2]
    )
    assert shown_cos == cos_out
    assert shown_auc == auc_out
    assert cos_all > 0
    assert auc_all > 0


def test_the_claim_numbers_match_a_recount_of_the_csv_files(
    output: registry.FigureOutput,
) -> None:
    cos_out, cos_all, auc_out, auc_all = _counts()
    assert output.numbers["cosine_outside"] == cos_out
    assert output.numbers["cosine_cells"] == cos_all
    assert output.numbers["auroc_outside"] == auc_out
    assert output.numbers["auroc_cells"] == auc_all
    assert f"{cos_out} of {cos_all}" in output.claim
    assert f"{auc_out} of {auc_all}" in output.claim
    assert output.claim.endswith(".")


def test_no_layout_faults(output: registry.FigureOutput) -> None:
    assert checks.problems(output.figure) == []
