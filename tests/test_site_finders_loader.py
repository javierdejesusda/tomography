import json
from pathlib import Path

import numpy as np
import pytest

from tomography.site import finders
from tomography.site.manifest import Manifest

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_the_reference_cell_of_gpt2_xl_utility_uses_matched_pairs() -> None:
    found = finders.load(Manifest(RESULTS), "gpt2-xl")
    assert found is not None
    cell = found["utility"]
    assert cell.pairing == "matched"
    row = cell.rows["pca_diff"]
    assert row.layer == 26
    assert row.report == pytest.approx(0.576, abs=5e-4)
    assert row.lo < row.report < row.hi


def test_the_curve_at_the_selected_layer_is_the_reported_accuracy() -> None:
    manifest = Manifest(RESULTS)
    for model in manifest.models("p4"):
        found = finders.load(manifest, model)
        assert found is not None
        for cell in found.values():
            for method, row in cell.rows.items():
                mean = cell.curves[method][0]
                assert mean[row.layer - 1] == pytest.approx(row.report)


def test_depth_is_layer_over_layer_count() -> None:
    found = finders.load(Manifest(RESULTS), "gpt2")
    assert found is not None
    cell = found["sentiment"]
    assert cell.depth[0] == pytest.approx(1 / 12)
    assert cell.depth[-1] == pytest.approx(1.0)
    assert cell.row_depth("pca_diff") == pytest.approx(
        cell.rows["pca_diff"].layer / 12
    )


def test_the_methods_follow_the_stored_series_order() -> None:
    found = finders.load(Manifest(RESULTS), "gpt2")
    assert found is not None
    run = Manifest(RESULTS).run("p4", "gpt2")
    assert run is not None
    stored = json.loads((run / "curves_truth_methods.json").read_text("utf-8"))[
        "data"
    ]["series"]
    assert list(found["truth"].curves) == stored
    with np.load(run / "curves_truth_methods.npz") as npz:
        assert found["truth"].curves["logistic"][1].shape == npz["lo"][3].shape


def test_a_model_without_p4_returns_none() -> None:
    assert finders.load(Manifest(RESULTS), "gpt2-nope") is None
