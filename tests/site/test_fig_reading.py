import json
from pathlib import Path

import numpy as np
import pytest

from tomography.site import params
from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[2] / "results"
SPEC = "reading"
GPT2 = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
MARGIN = 0.1


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    with paper.paper_style():
        return registry.build_figure(SPEC, Manifest(RESULTS))


def _curves(exp: str, model: str) -> tuple[np.ndarray, np.ndarray]:
    run = Manifest(RESULTS).run(exp, model)
    assert run is not None
    with np.load(run / "curves.npz") as npz:
        return np.array(npz["mean"][0]), np.array(npz["mean"][1])


def _first_depth(exp: str, model: str) -> float:
    reader, random = _curves(exp, model)
    index = int(np.flatnonzero(reader - random > MARGIN)[0])
    return (index + 1) / len(reader)


def _depths(exp: str) -> list[float]:
    return [_first_depth(exp, m) for m in Manifest(RESULTS).models(exp)]


def test_the_figure_is_seven_panels_in_two_rows(
    output: registry.FigureOutput,
) -> None:
    assert len(output.figure.axes) == 7
    assert output.figure.get_figwidth() == pytest.approx(paper.FULL_WIDTH)


def test_every_model_is_drawn_in_every_depth_panel(
    output: registry.FigureOutput,
) -> None:
    models = Manifest(RESULTS).models("r1")
    assert len(models) == 10
    for ax in output.figure.axes[:3]:
        assert len(ax.get_lines()) >= len(models)
        zero = [ln for ln in ax.get_lines() if list(ln.get_ydata()) == [0, 0]]
        assert zero


def test_the_size_panels_show_a_random_mark_per_model(
    output: registry.FigureOutput,
) -> None:
    for ax in output.figure.axes[3:]:
        crosses = [ln for ln in ax.get_lines() if ln.get_marker() == "x"]
        assert sum(len(ln.get_xdata()) for ln in crosses) == 10
        assert ax.get_xscale() == "log"


def test_the_dashed_fit_has_the_slope_stored_by_p6(
    output: registry.FigureOutput,
) -> None:
    stored = json.loads(
        (RESULTS / "p6" / "gpt2" / "25ee6de331d4" / "summary.json").read_text(
            "utf-8"
        )
    )["data"]["accuracy_per_decade"]
    for ax, exp in zip(
        output.figure.axes[3:], ["r1", "r2", "r3", "r7"], strict=True
    ):
        fits = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
        assert len(fits) == 1
        x = np.log10(np.asarray(fits[0].get_xdata(), dtype=float))
        y = np.asarray(fits[0].get_ydata(), dtype=float)
        assert np.polyfit(x, y, 1)[0] == pytest.approx(stored[exp])


def test_the_claim_numbers_match_a_recomputation_from_the_curves(
    output: registry.FigureOutput,
) -> None:
    r1, r3, r7 = _depths("r1"), _depths("r3"), _depths("r7")
    assert output.numbers["r1_depth_max"] == pytest.approx(max(r1))
    assert output.numbers["r3_depth_min"] == pytest.approx(min(r3))
    assert output.numbers["r3_depth_max"] == pytest.approx(max(r3))
    assert output.numbers["r7_depth_min"] == pytest.approx(min(r7))
    assert output.numbers["r7_depth_max"] == pytest.approx(max(r7))
    manifest = Manifest(RESULTS)
    for exp in ("r1", "r3", "r7"):
        accuracy = []
        for model in GPT2:
            run = manifest.run(exp, model)
            assert run is not None
            summary = json.loads((run / "summary.json").read_text("utf-8"))
            data = summary["data"]
            layer = data["val_layer"] if exp == "r7" else data["best_layer"]
            reader, _ = _curves(exp, model)
            accuracy.append(reader[layer - 1])
        decade = np.log10([params.param_count(m) for m in GPT2])
        slope = float(np.polyfit(decade, accuracy, 1)[0])
        assert output.numbers[f"slope_{exp}"] == pytest.approx(slope)


def test_every_claim_number_is_written_in_the_claim(
    output: registry.FigureOutput,
) -> None:
    for name, value in output.numbers.items():
        digits = 3 if name.startswith("slope") else 2
        assert f"{abs(value):.{digits}f}" in output.claim, name
    assert output.claim.endswith(".")
    assert "above GPT-2 on every" not in output.claim


def test_the_figure_has_no_layout_faults(
    output: registry.FigureOutput,
) -> None:
    assert checks.problems(output.figure) == []


def test_the_csv_lists_every_plotted_point(
    output: registry.FigureOutput,
) -> None:
    lines = output.csv.strip().splitlines()
    assert lines[0] == "panel,model,x,y,lo,hi,baseline"
    models = Manifest(RESULTS).models("r1")
    points = [ln for ln in lines if ln.startswith("size r1,")]
    assert len(points) == len(models)
