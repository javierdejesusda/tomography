import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from tomography.site import reading
from tomography.site import stats
from tomography.site.manifest import Manifest

_RESULTS = runpy.run_path(str(Path(__file__).with_name("_results.py")))
MANIFEST = _RESULTS["MANIFEST"]
data = _RESULTS["data"]
models = _RESULTS["models"]
npz = _RESULTS["npz"]
run = _RESULTS["run"]

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


R5_RANDOM = [
    0.483,
    0.476,
    0.473,
    0.502,
    0.554,
    0.512,
    0.492,
    0.485,
    0.531,
    0.505,
]
HELD_READER = [
    0.758,
    0.754,
    0.805,
    0.800,
    0.919,
    0.834,
    0.937,
    0.888,
    0.913,
    0.781,
]
HELD_RANDOM = [
    0.537,
    0.518,
    0.542,
    0.533,
    0.547,
    0.565,
    0.602,
    0.574,
    0.570,
    0.523,
]


def test_r5_reads_the_transfer_curves_for_the_reader_and_the_random() -> None:
    found = [reading.load(MANIFEST, "r5", m) for m in models("r5")]
    assert len(found) == 10
    for model, cell in zip(models("r5"), found, strict=True):
        assert cell is not None
        path = run("r5", model)
        summary = data(path, "summary.json")
        series = data(path, "curves_transfer.json")["series"]
        mean = npz(path, "curves_transfer.npz")["mean"]
        layer = summary["best_layer"]
        assert cell.layer == layer
        assert cell.accuracy == pytest.approx(summary["test_accuracy"])
        assert cell.reader[layer - 1] == pytest.approx(summary["test_accuracy"])
        want = mean[series.index("random")][layer - 1]
        assert cell.random_at_layer == pytest.approx(want)
    assert [round(c.random_at_layer, 3) for c in found if c] == R5_RANDOM


def test_r5_in_template_block_uses_the_in_template_summary() -> None:
    for model in models("r5"):
        block = reading.in_template(MANIFEST, model)
        assert block is not None
        path = run("r5", model)
        summary = data(path, "summary.json")["in_template"]
        series = data(path, "curves.json")["series"]
        mean = npz(path, "curves.npz")["mean"]
        assert block.layer == summary["best_layer"]
        assert block.reader == pytest.approx(summary["test_accuracy"])
        want = mean[series.index("random")][summary["best_layer"] - 1]
        assert block.random == pytest.approx(want)
        assert (
            block.n
            == data(path, "summary.json")["n_eval_pairs"]["in_template"]["test"]
        )


def test_heldout_sentiment_reads_the_64_pair_sample_size_run() -> None:
    got = [reading.heldout_sentiment(MANIFEST, m) for m in models("p3")]
    assert len(got) == 10
    assert [round(g.reader, 3) for g in got if g] == HELD_READER
    assert [round(g.random, 3) for g in got if g] == HELD_RANDOM
    for model, score in zip(models("p3"), got, strict=True):
        assert score is not None
        path = run("p3", model)
        meta = data(path, "sizes_sentiment.json")
        arrays = npz(path, "sizes_sentiment.npz")
        k = int(np.flatnonzero(arrays["sizes"] == 64)[0])
        col = int(np.flatnonzero(arrays["layers"] == meta["layer"])[0])
        acc = arrays["accuracy"]
        assert score.layer == meta["layer"]
        assert score.reader == pytest.approx(acc[k, 1, 0, :, col].mean())
        assert score.random == pytest.approx(acc[k, 1, 1, :, col].mean())
        assert score.n == 214


def test_margin_is_reader_minus_random_with_the_conservative_interval() -> None:
    score = reading.Score(layer=3, reader=0.80, random=0.55, n=214)
    d, lo, hi = reading.margin(score)
    want = stats.diff_interval(0.80, 0.55, 214)
    assert (d, lo, hi) == pytest.approx(want)
    assert d == pytest.approx(0.25)


def test_the_p4_sentiment_reading_uses_the_report_split_logistic_curves() -> (
    None
):
    for model in models("p4"):
        found = reading.load_sentiment(MANIFEST, model)
        assert found is not None
        path = run("p4", model)
        meta = data(path, "curves_sentiment_methods.json")
        mean = npz(path, "curves_sentiment_methods.npz")["mean"]
        series = list(meta["series"])
        assert found.n_layers == meta["n_layers"]
        assert found.reader == pytest.approx(mean[series.index("logistic")])
        assert found.random == pytest.approx(mean[series.index("random")])
        table = data(path, "table.json")
        reference = data(path, "summary.json")["reference"]["sentiment"]
        (row,) = [
            r
            for r in table
            if r["task"] == "sentiment"
            and r["method"] == "logistic"
            and r["pairing"] == reference["pairing"]
            and r["preprocess"] == "center"
        ]
        assert found.layer == row["layer"]
        assert found.accuracy == pytest.approx(row["report"])
        assert (found.lo, found.hi) == pytest.approx((row["lo"], row["hi"]))
        assert found.depth == pytest.approx(
            np.arange(1, meta["n_layers"] + 1) / meta["n_layers"]
        )


def test_the_random_band_is_the_stored_interval_of_the_random_series() -> None:
    manifest = Manifest(RESULTS)
    _, curves, arrays = _raw("r2", "gpt2")
    row = list(curves["series"]).index("random")
    found = reading.load(manifest, "r2", "gpt2")
    assert found is not None
    assert found.random_lo == pytest.approx(arrays["lo"][row])
    assert found.random_hi == pytest.approx(arrays["hi"][row])
    sentiment = reading.load_sentiment(manifest, "gpt2")
    assert sentiment is not None
    assert sentiment.random_lo is not None
    assert np.all(sentiment.random_lo <= sentiment.random + 1e-9)
    assert np.all(sentiment.random_hi >= sentiment.random - 1e-9)
