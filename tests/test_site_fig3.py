import json
from pathlib import Path

import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
SPEC = registry.FIGURES[2]
GPT2 = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _report(model: str, task: str, method: str) -> float:
    run = Manifest(RESULTS).run("p4", model)
    assert run is not None
    data = json.loads((run / "summary.json").read_text("utf-8"))["data"]
    pairing = data["reference"][task]["pairing"]
    table = json.loads((run / "table.json").read_text("utf-8"))["data"]
    rows = [
        r
        for r in table
        if (r["task"], r["method"], r["pairing"], r["preprocess"])
        == (task, method, pairing, "center")
    ]
    assert len(rows) == 1
    return float(rows[0]["report"])


def test_nine_panels_three_tasks_by_three_models(
    output: registry.FigureOutput,
) -> None:
    assert len(output.figure.axes) == 9
    assert output.figure.get_figwidth() == pytest.approx(paper.FULL_WIDTH)


def test_only_the_three_main_finders_are_drawn_with_a_random_band(
    output: registry.FigureOutput,
) -> None:
    for ax in output.figure.axes:
        assert len(ax.collections) == 1
        solid = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 2]
        assert len(solid) == 4
    assert "pca_raw" not in output.csv
    assert "kmeans" not in output.csv


def test_the_columns_are_gpt2_xl_and_two_qwen3_base_models(
    output: registry.FigureOutput,
) -> None:
    titles = [ax.get_title(loc="left") for ax in output.figure.axes[:3]]
    assert titles[0] == "GPT-2 XL"
    assert titles[1:] in (
        ["Qwen3-0.6B-Base", "Qwen3-1.7B-Base"],
        ["Qwen3-1.7B-Base", "Qwen3-4B-Base"],
    )


def test_the_claim_numbers_match_the_stored_tables(
    output: registry.FigureOutput,
) -> None:
    over = [
        _report(m, t, "pca_diff") - _report(m, t, "random")
        for m in GPT2
        for t in ("sentiment", "truth")
    ]
    logistic = [
        _report(m, t, "logistic") - _report(m, t, "random")
        for m in GPT2
        for t in ("sentiment", "truth")
    ]
    qwen = [
        _report("qwen3-1.7b-base", t, "pca_diff")
        - _report("qwen3-1.7b-base", t, "random")
        for t in ("sentiment", "truth", "utility")
    ]
    assert output.numbers["gpt2_pca_over_random_max"] == pytest.approx(
        max(over)
    )
    assert output.numbers["gpt2_logistic_over_random_min"] == pytest.approx(
        min(logistic)
    )
    assert output.numbers["qwen_pca_over_random_min"] == pytest.approx(
        min(qwen)
    )
    for name, value in output.numbers.items():
        assert f"{value:.3f}" in output.claim, name
    assert output.claim.endswith(".")


def test_the_claim_does_not_say_indistinguishable(
    output: registry.FigureOutput,
) -> None:
    assert "indistinguishable" not in output.claim


def test_no_layout_faults(output: registry.FigureOutput) -> None:
    assert checks.problems(output.figure) == []
