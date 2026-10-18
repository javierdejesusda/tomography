import json
from pathlib import Path

import numpy as np
import pytest

from tomography.site import reading
from tomography.site.manifest import Manifest

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _raw(exp: str, model: str) -> tuple[dict, dict, np.lib.npyio.NpzFile]:
    run = Manifest(RESULTS).run(exp, model)
    assert run is not None
    summary = json.loads((run / "summary.json").read_text("utf-8"))["data"]
    curves = json.loads((run / "curves.json").read_text("utf-8"))["data"]
    return summary, curves, np.load(run / "curves.npz")


def test_r1_reads_the_best_layer_and_its_interval() -> None:
    found = reading.load(Manifest(RESULTS), "r1", "gpt2")
    assert found is not None
    assert found.layer == 8
    assert found.accuracy == pytest.approx(0.771875)
    assert (found.lo, found.hi) == pytest.approx((0.758203125, 0.784375))
    assert found.n_layers == 12


def test_r7_reads_the_test_set_at_the_validation_layer() -> None:
    found = reading.load(Manifest(RESULTS), "r7", "gpt2")
    assert found is not None
    assert found.layer == 9
    assert found.accuracy == pytest.approx(0.6559322033898305)
    assert (found.lo, found.hi) == pytest.approx(
        (0.6288135593220339, 0.6813983050847455)
    )


@pytest.mark.parametrize("exp", ["r1", "r2", "r3", "r7"])
def test_the_curve_at_the_selected_layer_is_the_reported_accuracy(
    exp: str,
) -> None:
    manifest = Manifest(RESULTS)
    for model in manifest.models(exp):
        found = reading.load(manifest, exp, model)
        assert found is not None
        assert found.reader[found.layer - 1] == pytest.approx(found.accuracy)
        _, _, npz = _raw(exp, model)
        assert found.random[found.layer - 1] == pytest.approx(
            npz["mean"][1][found.layer - 1]
        )
        assert found.random_at_layer == pytest.approx(
            found.random[found.layer - 1]
        )


def test_selectivity_is_the_reader_minus_the_random_of_the_same_run() -> None:
    found = reading.load(Manifest(RESULTS), "r3", "qwen3-4b-base")
    assert found is not None
    _, _, npz = _raw("r3", "qwen3-4b-base")
    expected = npz["mean"][0] - npz["mean"][1]
    assert found.selectivity == pytest.approx(expected)
    assert found.depth[-1] == pytest.approx(1.0)
    assert len(found.depth) == found.n_layers


def test_first_depth_above_returns_the_earliest_relative_depth() -> None:
    found = reading.load(Manifest(RESULTS), "r1", "gpt2-medium")
    assert found is not None
    depth = reading.first_depth_above(found, 0.1)
    assert depth is not None
    index = next(i for i, s in enumerate(found.selectivity) if s > 0.1)
    assert depth == pytest.approx((index + 1) / found.n_layers)
    assert reading.first_depth_above(found, 10.0) is None


def test_a_missing_cell_returns_none() -> None:
    assert reading.load(Manifest(RESULTS), "r1", "gpt2-nope") is None
