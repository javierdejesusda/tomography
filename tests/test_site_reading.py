import math
from pathlib import Path
import runpy

from matplotlib.image import AxesImage
import numpy as np
import pytest
from scipy import stats as scipy_stats

from tomography.site import params
from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

_RESULTS = runpy.run_path(str(Path(__file__).with_name("_results.py")))
data = _RESULTS["data"]
models = _RESULTS["models"]
npz = _RESULTS["npz"]
run = _RESULTS["run"]

RESULTS = Path(__file__).resolve().parents[1] / "results"
Z = 1.959963984540054
CONCEPTS = {
    "sentiment": "r1",
    "utility": "r2",
    "morality": "r3",
    "truth": "r7",
}
GPT2 = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")


def _wilson(p: float, n: int) -> tuple[float, float]:
    denom = 1 + Z**2 / n
    centre = (p + Z**2 / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def _curves(exp: str, model: str) -> tuple[np.ndarray, np.ndarray]:
    path = run(exp, model)
    series = data(path, "curves.json")["series"]
    mean = npz(path, "curves.npz")["mean"]
    return mean[series.index("reader")], mean[series.index("random")]


def _read_at(concept: str, model: str) -> tuple[int, float, float, int]:
    exp = CONCEPTS[concept]
    path = run(exp if concept != "sentiment" else "p3", model)
    if concept == "sentiment":
        meta = data(path, "sizes_sentiment.json")
        arrays = npz(path, "sizes_sentiment.npz")
        k = int(np.flatnonzero(arrays["sizes"] == 64)[0])
        layer = int(meta["layer"])
        col = int(np.flatnonzero(arrays["layers"] == layer)[0])
        acc = arrays["accuracy"]
        return (
            layer,
            float(acc[k, 1, 0, :, col].mean()),
            float(acc[k, 1, 1, :, col].mean()),
            214,
        )
    summary = data(path, "summary.json")
    _, random = _curves(exp, model)
    if concept == "truth":
        layer = int(summary["val_layer"])
        test = summary["sets"]["test"]
        return (
            layer,
            float(test["accuracy_at_val_layer"]),
            float(random[layer - 1]),
            int(test["n_eval_pairs"]),
        )
    layer = int(summary["best_layer"])
    return (
        layer,
        float(summary["test_accuracy"]),
        float(random[layer - 1]),
        int(summary["n_eval_pairs"]["test"]),
    )


def _onset(concept: str, model: str) -> float | None:
    reader, random = _curves(CONCEPTS[concept], model)
    above = np.flatnonzero(reader - random > 0.1)
    return float((above[0] + 1) / len(reader)) if above.size else None


def _depths(concept: str) -> list[float]:
    values = [_onset(concept, m) for m in models(CONCEPTS[concept])]
    return [v for v in values if v is not None]


def _slope(concept: str) -> tuple[float, float, float]:
    x = [math.log10(params.param_count(m)) for m in GPT2]
    y = [_read_at(concept, m)[1] for m in GPT2]
    fit = scipy_stats.linregress(x, y)
    half = float(scipy_stats.t.ppf(0.975, 2)) * float(fit.stderr)
    return float(fit.slope), fit.slope - half, fit.slope + half


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure("reading", Manifest(RESULTS))


def _records(output: registry.FigureOutput, panel: str) -> list[dict[str, str]]:
    lines = output.csv.strip().splitlines()
    head = lines[0].split(",")
    assert head == [
        "panel",
        "concept",
        "model",
        "x",
        "y",
        "lo",
        "hi",
        "baseline",
    ]
    rows = [dict(zip(head, line.split(","), strict=True)) for line in lines[1:]]
    return [r for r in rows if r["panel"] == panel]


def test_every_depth_cell_is_the_reader_minus_random_curve(
    output: registry.FigureOutput,
) -> None:
    rows = _records(output, "depth")
    total = 0
    for concept, exp in CONCEPTS.items():
        for model in models(exp):
            reader, random = _curves(exp, model)
            mine = [
                r
                for r in rows
                if r["concept"] == concept and r["model"] == model
            ]
            assert len(mine) == len(reader)
            total += len(mine)
            for i, rec in enumerate(mine):
                assert float(rec["y"]) == pytest.approx(
                    reader[i] - random[i], abs=1e-9
                )
                assert float(rec["x"]) == pytest.approx(
                    (i + 1) / len(reader), abs=1e-9
                )
    assert total == len(rows)


def test_every_size_point_is_recomputed_with_its_wilson_interval(
    output: registry.FigureOutput,
) -> None:
    rows = _records(output, "size")
    assert len(rows) == sum(len(models(e)) for e in CONCEPTS.values())
    for rec in rows:
        _, acc, random, n = _read_at(rec["concept"], rec["model"])
        lo, hi = _wilson(acc, n)
        assert float(rec["y"]) == pytest.approx(acc, abs=1e-9)
        assert float(rec["lo"]) == pytest.approx(lo, abs=1e-9)
        assert float(rec["hi"]) == pytest.approx(hi, abs=1e-9)
        assert float(rec["baseline"]) == pytest.approx(random, abs=1e-9)
        assert float(rec["x"]) == params.param_count(rec["model"])


def test_the_sentiment_row_is_the_held_out_value_not_the_r1_value() -> None:
    differs = 0
    for model in models("p3"):
        _, acc, _, _ = _read_at("sentiment", model)
        best = data(run("r1", model), "summary.json")["best_accuracy"]
        differs += abs(acc - best) > 1e-6
    assert differs >= 1


def test_the_onset_depths_are_recomputed_for_all_forty_cells(
    output: registry.FigureOutput,
) -> None:
    cells = 0
    for concept in CONCEPTS:
        depths = _depths(concept)
        cells += len(depths)
        assert output.numbers[f"onset_{concept}_min"] == pytest.approx(
            min(depths)
        )
        assert output.numbers[f"onset_{concept}_max"] == pytest.approx(
            max(depths)
        )
    assert cells == 40


def test_the_gpt2_slopes_use_a_t_interval_on_four_points(
    output: registry.FigureOutput,
) -> None:
    for concept in CONCEPTS:
        slope, lo, hi = _slope(concept)
        assert output.numbers[f"slope_{concept}"] == pytest.approx(slope)
        assert output.numbers[f"slope_{concept}_lo"] == pytest.approx(lo)
        assert output.numbers[f"slope_{concept}_hi"] == pytest.approx(hi)
    assert output.numbers["truth_n"] == 59


def test_the_plan_values_hold_at_the_current_results() -> None:
    assert (
        round(min(_depths("sentiment")), 3),
        round(max(_depths("sentiment")), 3),
    ) == (0.021, 0.125)
    assert (
        round(min(_depths("utility")), 3),
        round(max(_depths("utility")), 3),
    ) == (0.25, 0.417)
    assert round(_slope("utility")[0], 3) == 0.034
    assert round(_slope("morality")[0], 3) == 0.132
    assert round(_slope("sentiment")[0], 3) == 0.048
    assert round(_slope("truth")[0], 3) == pytest.approx(-0.027)


def test_the_caption_quotes_the_onsets_slopes_and_sizes(
    output: registry.FigureOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    for concept in CONCEPTS:
        assert f"{output.numbers[f'onset_{concept}_min']:.3f}" in text
        assert f"{output.numbers[f'onset_{concept}_max']:.3f}" in text
    assert "0.125" in text
    assert "214 sentiment pairs" in text
    assert "4,808 utility, 983 morality and 59 truth pairs" in text
    assert "per tenfold" in text
    assert "includes zero" in text


def test_the_figure_has_two_rows_of_four_and_no_model_legend(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    assert fig.get_size_inches()[0] == pytest.approx(paper.FULL_WIDTH)
    assert fig.get_size_inches()[1] == pytest.approx(4.4)
    images = fig.findobj(AxesImage)
    assert len(images) == sum(len(models(e)) for e in CONCEPTS.values())
    labels = [t.get_text() for lg in fig.legends for t in lg.get_texts()]
    assert not any("GPT-2" in x or "Qwen3" in x for x in labels)


def test_the_guards_are_green(output: registry.FigureOutput) -> None:
    assert checks.guard_problems(output.figure) == []
    assert checks.problems(output.figure) == []
