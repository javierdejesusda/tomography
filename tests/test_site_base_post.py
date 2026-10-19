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
from tomography.viz import axes
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
    "knowledge": "knowledge, AUROC",
    "steering": "steering, win rate",
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


def test_no_other_figure_uses_half_filled_markers() -> None:
    manifest = Manifest(RESULTS)
    for key in ("steering", "knowledge_reading", "approximate"):
        fig = registry.build_figure(key, manifest).figure
        for ax in fig.axes:
            assert all(line.get_fillstyle() != "left" for line in ax.lines)


def test_svg_and_pdf_bytes_are_the_same_for_the_same_input() -> None:
    first = registry.build_figure(SPEC, Manifest(RESULTS))
    second = registry.build_figure(SPEC, Manifest(RESULTS))
    with paper.paper_style():
        assert paper.svg_bytes(first.figure) == paper.svg_bytes(second.figure)
        assert paper.pdf_bytes(first.figure) == paper.pdf_bytes(second.figure)


def test_seed_paired_differences_that_exclude_zero_are_named_and_tokenised(
    output: registry.FigureOutput,
) -> None:
    names = {"r2": "utility", "r3": "morality", "r5": "honesty", "r7": "truth"}
    found = []
    for row in _table("p8", "qwen3-1.7b"):
        if row["source"] in names and (row["lo"] > 0 or row["hi"] < 0):
            found.append((names[row["source"]], row["tuned"]))
    assert sorted(found) == [
        ("honesty", "qwen3-0.6b"),
        ("morality", "qwen3-0.6b"),
        ("utility", "qwen3-1.7b"),
    ]
    assert output.numbers["paired_excluding_zero"] == 3
    for name, tuned in found:
        size = tuned.split("-")[1].replace(".", "")
        assert output.numbers[f"paired_{name}_{size}"] == 1
    assert output.claim == (
        "Post-training lowers held-out sentiment reading; on the other "
        "readings most intervals include zero, which is not evidence of no "
        "change."
    )


def test_the_paired_total_counts_the_rows_the_exclusion_count_ranges_over(
    output: registry.FigureOutput,
) -> None:
    rows = [
        r
        for r in _table("p8", "qwen3-1.7b")
        if r["source"] in ("r2", "r3", "r5", "r7")
    ]
    assert output.numbers["paired_total"] == len(rows) == 8
    assert (
        output.numbers["paired_excluding_zero"]
        <= output.numbers["paired_total"]
    )


PAIRED_SOURCE = {
    "utility": "r2",
    "morality": "r3",
    "truth": "r7",
    "honesty": "r5",
}
NARROW = 0.15
WIDE = 0.3
NEWLINE = chr(10)


def _paired(row: str, size: str) -> dict[str, Any] | None:
    source = PAIRED_SOURCE.get(row)
    if source is None or size == "4b":
        return None
    (found,) = [
        r
        for r in _table("p8", "qwen3-1.7b")
        if r["source"] == source and r["tuned"] == f"qwen3-{size}"
    ]
    return found


def _final(row: str, size: str) -> tuple[float, float, float]:
    stored = _paired(row, size)
    if stored is not None:
        return float(stored["delta"]), float(stored["lo"]), float(stored["hi"])
    return _expected(row, size)


def _excludes(interval: tuple[float, float, float]) -> bool:
    return interval[1] > 0 or interval[2] < 0


def _panels(output: registry.FigureOutput) -> tuple[Any, Any]:
    first, second = sorted(
        output.figure.axes, key=lambda a: -a.get_position().y1
    )
    return first, second


def test_every_row_and_size_uses_the_paired_interval_where_it_is_stored(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    assert len(rows) == 24
    for row in ROWS:
        for key, size in SIZES.items():
            want = _final(row, size)
            (got,) = [
                r for r in rows if r["row"] == LABELS[row] and r["size"] == size
            ]
            assert float(got["value"]) == pytest.approx(want[0])
            assert float(got["lo"]) == pytest.approx(want[1])
            assert float(got["hi"]) == pytest.approx(want[2])
            kind = "paired" if _paired(row, size) else "conservative"
            assert got["interval"] == kind
            assert output.numbers[f"{row}_{key}"] == pytest.approx(want[0])
            assert output.numbers[f"{row}_{key}_lo"] == pytest.approx(want[1])
            assert output.numbers[f"{row}_{key}_hi"] == pytest.approx(want[2])
    assert sum(r["interval"] == "paired" for r in rows) == 8


def test_the_paired_point_estimate_matches_the_conservative_one() -> None:
    for row in PAIRED_SOURCE:
        for size in ("0.6b", "1.7b"):
            assert _final(row, size)[0] == pytest.approx(
                _expected(row, size)[0]
            )


def test_summary_numbers_are_recomputed(output: registry.FigureOutput) -> None:
    n = output.numbers
    sentiment = [_final("sentiment", s) for s in SIZES.values()]
    assert n["sent_below"] == sum(hi < 0 for _, _, hi in sentiment)
    moves = [abs(_final(row, s)[0]) for row in OTHER for s in SIZES.values()]
    assert n["max_abs_other"] == pytest.approx(max(moves))
    excluding = [
        (row, s)
        for row in OTHER
        for s in SIZES.values()
        if _excludes(_final(row, s))
    ]
    assert n["other_excluding"] == len(excluding)
    assert ("utility", "4b") in excluding
    assert ("morality", "0.6b") in excluding


def test_caption_states_the_intervals_the_axes_and_the_fill(
    output: registry.FigureOutput,
) -> None:
    n = output.numbers
    sentiment = sorted(abs(n[f"sentiment_{k}"]) for k in SIZES)
    detail = output.detail
    assert "seed-paired" in detail
    assert "conservative" in detail
    assert "solid" in detail
    assert "dashed" in detail
    assert "filled" in detail
    assert "axes differ" in detail
    assert (
        f"Sentiment falls by {sentiment[0]:.3f} to {sentiment[-1]:.3f}, with "
        f"an interval below zero on {int(n['sent_below'])} of 3 sizes;"
    ) in detail
    assert (
        "an interval excluding zero only for utility on 1.7B "
        f"({chr(0x2212)}0.012), utility on 4B (+0.022), morality on 0.6B "
        f"({chr(0x2212)}0.016) and honesty transfer on 0.6B (+0.073)"
    ) in detail
    assert "the knowledge AUROC and the steering win rate fall" in detail
    assert "[[sec:limitations]]" in detail
    assert "[[table:base_post]]" in detail
    assert "prompt format" in detail
    assert "Section 6" not in detail
    assert "Table A5" not in detail


def test_sentiment_falls_on_every_size_and_the_claim_holds(
    output: registry.FigureOutput,
) -> None:
    n = output.numbers
    assert all(n[f"sentiment_{k}"] < 0 for k in SIZES)
    assert n["sent_below"] == 2


def test_two_panels_split_the_rows_by_how_wide_their_intervals_are(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    assert tuple(fig.get_size_inches()) == (paper.FULL_WIDTH, steering_height())
    assert checks.guard_problems(fig) == []
    assert checks.problems(fig) == []
    narrow, wide = _panels(output)
    assert narrow.get_xlim() == (-NARROW, NARROW)
    assert wide.get_xlim() == (-WIDE, WIDE)
    checks.draw(fig)
    names = {v: k for k, v in LABELS.items()}
    narrow_rows = [names[t.get_text()] for t in narrow.get_yticklabels()]
    wide_rows = [names[t.get_text()] for t in wide.get_yticklabels()]
    assert sorted(narrow_rows + wide_rows) == sorted(ROWS)
    for row in narrow_rows:
        bounds = [b for s in SIZES.values() for b in _final(row, s)[1:]]
        assert max(abs(b) for b in bounds) <= NARROW
    for row in wide_rows:
        bounds = [b for s in SIZES.values() for b in _final(row, s)[1:]]
        assert max(abs(b) for b in bounds) > NARROW
    order = {r: i for i, r in enumerate(ROWS)}
    assert narrow_rows == sorted(narrow_rows, key=order.get)
    assert wide_rows == sorted(wide_rows, key=order.get)
    text = f"{output.claim} {output.detail} {output.csv}"
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text
    assert "_" not in re.sub(r"\[\[[^\]]*\]\]", "", text)


def steering_height() -> float:
    from tomography.site.figs import base_post

    return float(base_post.HEIGHT)


def test_marker_fill_says_whether_the_interval_excludes_zero(
    output: registry.FigureOutput,
) -> None:
    narrow, wide = _panels(output)
    marked = [
        ln
        for ax in (narrow, wide)
        for ln in ax.lines
        if ln.get_gid() == "point"
    ]
    assert len(marked) == 24
    assert {ln.get_marker() for ln in marked} == {paper.SHAPES["direction"]}
    assert {ln.get_fillstyle() for ln in marked} == {"full"}
    filled = sum(ln.get_markerfacecolor() != "white" for ln in marked)
    want = sum(
        _excludes(_final(row, s)) for row in ROWS for s in SIZES.values()
    )
    assert filled == want
    assert len({ln.get_color() for ln in marked}) == 3


def test_paired_whiskers_are_solid_and_conservative_ones_dashed(
    output: registry.FigureOutput,
) -> None:
    whiskers = [
        ln
        for ax in output.figure.axes
        for ln in ax.lines
        if ln.get_gid() == "whisker"
    ]
    assert len(whiskers) == 24
    dashed = [ln for ln in whiskers if ln.get_linestyle() != "-"]
    assert len(dashed) == 24 - 8


def test_sizes_are_in_fixed_order_and_labelled_on_the_first_row(
    output: registry.FigureOutput,
) -> None:
    checks.draw(output.figure)
    for ax in _panels(output):
        points = [ln for ln in ax.lines if ln.get_gid() == "point"]
        by_row: dict[int, list[Any]] = {}
        for ln in points:
            y = float(ln.get_ydata()[0])
            by_row.setdefault(round(y), []).append(ln)
        for lines in by_row.values():
            lines.sort(key=lambda ln: float(ln.get_ydata()[0]))
            colors = [ln.get_color() for ln in lines]
            assert colors == [
                axes.model_style(f"qwen3-{s}").color
                for s in ("0.6b", "1.7b", "4b")
            ]
        labels = sorted(
            t.get_text()
            for t in ax.texts
            if t.get_text() in ("0.6B", "1.7B", "4B")
        )
        assert labels == ["0.6B", "1.7B", "4B"]


def test_zero_line_is_a_solid_half_point_line_between_whiskers_and_markers(
    output: registry.FigureOutput,
) -> None:
    for ax in _panels(output):
        (zero,) = [ln for ln in ax.lines if ln.get_gid() == "zero"]
        assert zero.get_linewidth() == pytest.approx(0.5)
        assert zero.get_linestyle() == "-"
        assert list(zero.get_xdata()) == [0.0, 0.0]
        points = [ln for ln in ax.lines if ln.get_gid() == "point"]
        whiskers = [ln for ln in ax.lines if ln.get_gid() == "whisker"]
        assert zero.get_zorder() < min(p.get_zorder() for p in points)
        assert zero.get_zorder() > max(w.get_zorder() for w in whiskers)
        texts = [t.get_text() for t in ax.texts]
        assert "base higher" in texts
        assert "post-trained higher" in texts


def test_the_key_names_the_fill_and_the_two_interval_kinds(
    output: registry.FigureOutput,
) -> None:
    (legend,) = output.figure.legends
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == [
        "interval excludes zero",
        "interval includes zero",
        "seed-paired interval",
        "conservative interval",
        "0.6B",
        "1.7B",
        "4B",
    ]


SECTIONS = {
    "sentiment": "reading accuracy",
    "utility": "reading accuracy",
    "morality": "reading accuracy",
    "truth": "reading accuracy",
    "honesty": "reading accuracy",
    "truthfulqa": "reading accuracy",
    "knowledge": "knowledge",
    "steering": "steering",
}


def test_rows_are_grouped_by_section_with_a_label_and_a_separator(
    output: registry.FigureOutput,
) -> None:
    checks.draw(output.figure)
    names = {v: k for k, v in LABELS.items()}
    for ax in _panels(output):
        rows = [names[t.get_text()] for t in ax.get_yticklabels()]
        order = []
        for row in rows:
            if SECTIONS[row] not in order:
                order.append(SECTIONS[row])
        want = [s for s in order if s == "reading accuracy"]
        groups = sorted(
            (t for t in ax.texts if t.get_gid() == "group"),
            key=lambda t: t.get_position()[1],
        )
        assert [t.get_text() for t in groups] == want
        lines = [ln for ln in ax.lines if ln.get_gid() == "separator"]
        assert len(lines) == len(order) - 1
        for ln in lines:
            ys = list(ln.get_ydata())
            assert ys[0] == ys[1]
            assert ln.get_linewidth() <= 0.5
        ticks = list(ax.get_yticks())
        firsts = {}
        for tick, row in zip(ticks, rows, strict=True):
            firsts.setdefault(SECTIONS[row], tick)
        for text in groups:
            assert text.get_position()[1] < firsts[text.get_text()] - 0.5
        assert [t.get_text() for t in groups] == want


def test_the_caption_has_at_most_200_words_and_leads_with_the_conclusion(
    output: registry.FigureOutput,
) -> None:
    assert len(f"{output.claim} {output.detail}".split()) <= 200
    assert output.detail.startswith("Sentiment falls by")


def test_a_value_that_rounds_to_zero_prints_without_a_sign() -> None:
    from tomography.site.figs import base_post

    minus = chr(0x2212)
    assert base_post._signed(0.0004) == "0.000"
    assert base_post._signed(-0.0004) == "0.000"
    assert base_post._signed(0.05) == "+0.050"
    assert base_post._signed(-0.05) == f"{minus}0.050"


def test_both_panels_carry_the_x_label(output: registry.FigureOutput) -> None:
    for ax in _panels(output):
        assert ax.get_xlabel() == "post-trained minus base"


def test_the_conservative_dash_is_plain_and_long_enough_to_read(
    output: registry.FigureOutput,
) -> None:
    whiskers = [
        ln
        for ax in output.figure.axes
        for ln in ax.lines
        if ln.get_gid() == "whisker" and ln.get_linestyle() != "-"
    ]
    assert len(whiskers) == 16
    for ln in whiskers:
        _, pattern = ln._unscaled_dash_pattern
        assert len(pattern) == 2
        assert pattern[0] >= 5.0


def test_the_size_key_runs_light_to_dark(output: registry.FigureOutput) -> None:
    (legend,) = output.figure.legends
    pairs = [
        (t.get_text(), h.get_markerfacecolor())
        for h, t in zip(legend.legend_handles, legend.get_texts(), strict=True)
        if t.get_text() in ("0.6B", "1.7B", "4B")
    ]
    assert pairs == [
        (label, axes.model_style(f"qwen3-{label.lower()}").color)
        for label in ("0.6B", "1.7B", "4B")
    ]


def _ends_on_a_dash(length: float, pattern: tuple[float, ...]) -> bool:
    dash, gap = pattern
    cycles = (length - dash) / (dash + gap)
    return cycles >= 1 and abs(cycles - round(cycles)) * (dash + gap) < 0.15


def test_every_conservative_interval_and_its_key_end_on_a_whole_dash(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    fig.canvas.draw()
    seen = 0
    for ax in fig.axes:
        for ln in ax.lines:
            if ln.get_gid() != "whisker" or ln.get_linestyle() == "-":
                continue
            x0, x1 = ln.get_xdata()
            px = abs(ax.transData.transform((x1, 0))[0])
            px -= abs(ax.transData.transform((x0, 0))[0])
            length = abs(px) * 72 / fig.dpi
            _, pattern = ln._dash_pattern
            assert len(pattern) == 2
            assert _ends_on_a_dash(length, pattern), (length, pattern)
            seen += 1
    assert seen == 16
    (legend,) = fig.legends
    (handle,) = [
        h
        for h, t in zip(legend.legend_handles, legend.get_texts(), strict=True)
        if t.get_text() == "conservative interval"
    ]
    _, pattern = handle._dash_pattern
    length = legend.handlelength * legend.get_texts()[0].get_fontsize()
    assert len(pattern) == 2
    assert _ends_on_a_dash(length, pattern), (length, pattern)
