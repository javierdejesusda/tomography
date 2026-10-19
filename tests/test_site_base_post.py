import csv
import io
import math
from pathlib import Path
import re
import runpy
from typing import Any

import numpy as np
import pytest

from tomography.site import registry
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
SPEC = "base_post"
HEIGHT = 2.6
SIZES = {"06b": "0.6b", "17b": "1.7b", "4b": "4b"}
ROWS = (
    "sentiment",
    "utility",
    "morality",
    "truth",
    "honesty",
    "truthfulqa",
    "knowledge",
    "steering",
)
LABELS = {
    "sentiment": "sentiment, held out",
    "utility": "utility",
    "morality": "morality",
    "truth": "truth",
    "honesty": "honesty, transfer",
    "truthfulqa": "TruthfulQA",
    "knowledge": "knowledge",
    "steering": "steering",
}
OTHER = ("utility", "morality", "truth", "honesty", "truthfulqa")


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(output.csv)))


def _summary(exp: str, model: str) -> dict[str, Any]:
    out: dict[str, Any] = _R["data"](_R["run"](exp, model), "summary.json")
    return out


def _sentiment(model: str) -> float:
    path = _R["run"]("p3", model)
    layer = _R["data"](path, "sizes_sentiment.json")["layer"]
    arrays = _R["npz"](path, "sizes_sentiment.npz")
    size = int(np.flatnonzero(arrays["sizes"] == 64)[0])
    col = int(np.flatnonzero(arrays["layers"] == layer)[0])
    return float(arrays["accuracy"][size, 1, 0, :, col].mean())


def _accuracy(row: str, model: str) -> float:
    if row == "sentiment":
        return _sentiment(model)
    if row == "utility":
        return float(_summary("r2", model)["test_accuracy"])
    if row == "morality":
        return float(_summary("r3", model)["test_accuracy"])
    if row == "truth":
        return float(
            _summary("r7", model)["sets"]["test"]["accuracy_at_val_layer"]
        )
    if row == "honesty":
        return float(_summary("r5", model)["test_accuracy"])
    raise KeyError(row)


def _size(row: str, model: str) -> int:
    if row == "sentiment":
        return 214
    if row == "utility":
        return int(_summary("r2", model)["n_eval_pairs"]["test"])
    if row == "morality":
        return int(_summary("r3", model)["n_eval_pairs"]["test"])
    if row == "truth":
        return int(_summary("r7", model)["sets"]["test"]["n_eval_pairs"])
    if row == "honesty":
        return int(_summary("r5", model)["n_eval_pairs"]["transfer"]["test"])
    raise KeyError(row)


def _table(exp: str, model: str) -> list[dict[str, Any]]:
    payload = _R["data"](_R["run"](exp, model), "table.json")
    if isinstance(payload, dict):
        return [
            dict(zip(payload["columns"], r, strict=True))
            for r in payload["rows"]
        ]
    out: list[dict[str, Any]] = payload
    return out


def _truthfulqa(model: str) -> tuple[float, int]:
    (row,) = [
        r
        for r in _table("r6", model)
        if r["method"] == "lat"
        and r["layer_rule"] == "validation-layer"
        and r["source"] == "truthfulqa"
    ]
    return float(row["mc1"]), int(_summary("r6", model)["n_test"])


def _knowledge(model: str) -> dict[str, Any]:
    (row,) = [r for r in _table("p9b", model) if r["type"] == "obscure_real"]
    return row


def _steering(model: str) -> dict[str, Any]:
    (row,) = [
        r
        for r in _table("m1", model)
        if r["task"] == "manipulation"
        and r["setting"] == "0.3"
        and r["method"] == "logistic"
    ]
    return row


def _expected(row: str, size: str) -> tuple[float, float, float]:
    base, post = f"qwen3-{size}-base", f"qwen3-{size}"
    if row == "truthfulqa":
        (a, n), (b, _) = _truthfulqa(post), _truthfulqa(base)
        return stats.diff_interval(a, b, n)
    if row == "knowledge":
        pa, ba = _knowledge(post), _knowledge(base)
        half = stats.Z * math.hypot(
            stats.hanley_mcneil_se(
                pa["logistic_within"],
                pa["n_positive_test"],
                pa["n_negative_test"],
            ),
            stats.hanley_mcneil_se(
                ba["logistic_within"],
                ba["n_positive_test"],
                ba["n_negative_test"],
            ),
        )
        d = pa["logistic_within"] - ba["logistic_within"]
        return d, d - half, d + half
    if row == "steering":
        pa, ba = _steering(post), _steering(base)
        return stats.diff_interval(pa["value"], ba["value"], pa["n"], ba["n"])
    return stats.diff_interval(
        _accuracy(row, post), _accuracy(row, base), _size(row, post)
    )


def test_every_row_and_size_is_the_post_trained_minus_the_base_value(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    assert len(rows) == 24
    for row in ROWS:
        for key, size in SIZES.items():
            want = _expected(row, size)
            (got,) = [
                r for r in rows if r["row"] == LABELS[row] and r["size"] == size
            ]
            assert float(got["value"]) == pytest.approx(want[0])
            assert float(got["lo"]) == pytest.approx(want[1])
            assert float(got["hi"]) == pytest.approx(want[2])
            assert output.numbers[f"{row}_{key}"] == pytest.approx(want[0])
            assert output.numbers[f"{row}_{key}_lo"] == pytest.approx(want[1])
            assert output.numbers[f"{row}_{key}_hi"] == pytest.approx(want[2])


def test_values_match_the_plan_at_this_commit(
    output: registry.FigureOutput,
) -> None:
    n = output.numbers
    assert [round(n[f"sentiment_{k}"], 3) for k in SIZES] == [
        -0.084,
        -0.049,
        -0.132,
    ]
    assert round(n["utility_4b"], 3) == 0.022
    assert round(n["honesty_4b"], 3) == 0.100
    assert round(n["knowledge_17b_lo"], 3) == -0.111


def test_summary_numbers_are_recomputed(output: registry.FigureOutput) -> None:
    n = output.numbers
    sentiment = [_expected("sentiment", s) for s in SIZES.values()]
    assert n["sent_below"] == sum(hi < 0 for _, _, hi in sentiment)
    moves = [abs(_expected(row, s)[0]) for row in OTHER for s in SIZES.values()]
    assert n["max_abs_other"] == pytest.approx(max(moves))
    excluding = [
        (row, s)
        for row in OTHER
        for s in SIZES.values()
        if _expected(row, s)[1] > 0 or _expected(row, s)[2] < 0
    ]
    assert n["other_excluding"] == len(excluding)
    assert ("utility", "4b") in excluding


def test_caption_is_the_template_filled_from_the_files(
    output: registry.FigureOutput,
) -> None:
    n = output.numbers
    assert output.claim == (
        "Post-training lowers held-out sentiment reading and leaves most "
        "other readings within their intervals."
    )
    sentiment = sorted(abs(n[f"sentiment_{k}"]) for k in SIZES)
    assert (
        "Post-trained minus base checkpoint for each Qwen3 size (0.6B, "
        "1.7B and 4B, light to dark), with conservative 95 percent "
        "intervals for the test items. Sentiment falls by "
        f"{sentiment[0]:.3f} to {sentiment[-1]:.3f}, with an interval "
        f"below zero on {int(n['sent_below'])} of 3 sizes; "
        "utility, morality, truth, honesty transfer and TruthfulQA move by "
        f"at most {n['max_abs_other']:.3f}, with an interval excluding zero "
        "only for utility on 4B (+0.022); the knowledge AUROC and the "
        "steering win rate fall with an interval below zero on 1.7B only."
    ) in output.detail
    assert "[[sec:limitations]]" in output.detail
    assert "[[table:base_post]]" in output.detail
    assert "Section 6" not in output.detail
    assert "Table A5" not in output.detail
    assert "prompt format" in output.detail


def test_sentiment_falls_on_every_size_and_the_claim_holds(
    output: registry.FigureOutput,
) -> None:
    n = output.numbers
    assert all(n[f"sentiment_{k}"] < 0 for k in SIZES)
    assert n["sent_below"] == 2


def test_size_guards_and_layout(output: registry.FigureOutput) -> None:
    fig = output.figure
    assert tuple(fig.get_size_inches()) == (paper.FULL_WIDTH, HEIGHT)
    assert checks.guard_problems(fig) == []
    assert checks.problems(fig) == []


def test_markers_are_left_filled_diamonds_in_the_size_colors(
    output: registry.FigureOutput,
) -> None:
    (ax,) = output.figure.axes
    marked = [line for line in ax.lines if line.get_marker() == "D"]
    assert len(marked) == 24
    assert {line.get_fillstyle() for line in marked} == {"left"}
    assert len({line.get_color() for line in marked}) == 3
    assert len(ax.patches) == 0


def test_no_other_figure_uses_half_filled_markers() -> None:
    manifest = Manifest(RESULTS)
    for key in ("steering", "knowledge_reading", "approximate"):
        fig = registry.build_figure(key, manifest).figure
        for ax in fig.axes:
            assert all(line.get_fillstyle() != "left" for line in ax.lines)


def test_rows_are_labelled_and_the_zero_line_is_drawn(
    output: registry.FigureOutput,
) -> None:
    (ax,) = output.figure.axes
    checks.draw(output.figure)
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        "sentiment, held out",
        "utility",
        "morality",
        "truth",
        "honesty, transfer",
        "TruthfulQA",
        "knowledge",
        "steering",
    ]
    assert ax.get_xlim() == (-0.3, 0.3)
    text = f"{output.claim} {output.detail} {output.csv}"
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text
    assert "_" not in re.sub(r"\[\[[^\]]*\]\]", "", text)


def test_svg_and_pdf_bytes_are_the_same_for_the_same_input() -> None:
    first = registry.build_figure(SPEC, Manifest(RESULTS))
    second = registry.build_figure(SPEC, Manifest(RESULTS))
    with paper.paper_style():
        assert paper.svg_bytes(first.figure) == paper.svg_bytes(second.figure)
        assert paper.pdf_bytes(first.figure) == paper.pdf_bytes(second.figure)
