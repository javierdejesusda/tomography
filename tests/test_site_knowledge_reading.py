import csv
import io
import itertools
import math
from pathlib import Path
import runpy
from typing import Any

import pytest

from tomography.site import registry
from tomography.site import stats
from tomography.site.figs import knowledge_reading
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
_L = runpy.run_path(str(Path(__file__).with_name("_legends.py")))
SPEC = "knowledge_reading"
Z = 1.959963984540054
MINUS = chr(0x2212)
HEIGHT = 4.2
READER = ("logistic_within", "popularity_within")
TEXT_CONTROLS = ("text_word_within", "text_char_within")


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(output.csv)))


def _models(exp: str) -> list[str]:
    models: list[str] = _R["models"](exp)
    return models


def _table(
    exp: str, model: str, name: str = "table.json"
) -> list[dict[str, Any]]:
    payload = _R["data"](_R["run"](exp, model), name)
    if isinstance(payload, dict):
        return [
            dict(zip(payload["columns"], r, strict=True))
            for r in payload["rows"]
        ]
    rows: list[dict[str, Any]] = payload
    return rows


def _contrast() -> dict[str, dict[str, Any]]:
    found = {}
    for model in _models("p9b"):
        rows = [r for r in _table("p9b", model) if r["type"] == "obscure_real"]
        if rows:
            found[model] = rows[0]
    return found


def _increments() -> dict[str, dict[str, dict[str, Any]]]:
    return {
        m: {
            r["features"]: r
            for r in _table("p9c", m, "table_incremental.json")
            if r["dataset"] == "popqa"
        }
        for m in _models("p9c")
    }


def _se(auc: float, n_pos: int, n_neg: int) -> float:
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    var = (
        auc * (1 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    return math.sqrt(var)


def _diff(row: dict[str, Any]) -> tuple[float, float, float]:
    a, b = row["logistic_within"], row["popularity_within"]
    n_pos, n_neg = row["n_positive_test"], row["n_negative_test"]
    half = Z * math.sqrt(_se(a, n_pos, n_neg) ** 2 + _se(b, n_pos, n_neg) ** 2)
    return a - b, a - b - half, a - b + half


def _text(phrase: stats.RoleCount | dict[str, stats.RoleCount]) -> str:
    assert isinstance(phrase, dict)
    return stats.per_role_text(phrase)


def _flags(rule: Any) -> dict[str, bool]:
    return {m: bool(rule(r)) for m, r in _contrast().items()}


def _by_role(flags: dict[str, bool]) -> dict[str, stats.RoleCount]:
    return stats.per_role(flags)


def test_every_checkpoint_with_a_contrast_has_the_four_panel_a_values(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    contrast = _contrast()
    assert "gpt2" not in contrast
    for model, row in contrast.items():
        for series, column in (
            ("direction", "logistic_within"),
            ("popularity", "popularity_within"),
            ("words", "text_word_within"),
            ("characters", "text_char_within"),
        ):
            (got,) = [
                r
                for r in rows
                if r["panel"] == "a"
                and r["model"] == model
                and r["series"] == series
            ]
            assert float(got["value"]) == pytest.approx(row[column])
    assert not [r for r in rows if r["panel"] == "a" and r["model"] == "gpt2"]


def test_direction_and_popularity_carry_hanley_mcneil_intervals(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for model, row in _contrast().items():
        for series, column in (
            ("direction", "logistic_within"),
            ("popularity", "popularity_within"),
        ):
            (got,) = [
                r
                for r in rows
                if r["panel"] == "a"
                and r["model"] == model
                and r["series"] == series
            ]
            value = row[column]
            half = Z * _se(
                value, row["n_positive_test"], row["n_negative_test"]
            )
            assert float(got["lo"]) == pytest.approx(max(0.0, value - half))
            assert float(got["hi"]) == pytest.approx(min(1.0, value + half))
        for series in ("words", "characters"):
            (got,) = [
                r
                for r in rows
                if r["panel"] == "a"
                and r["model"] == model
                and r["series"] == series
            ]
            assert got["lo"] == ""
            assert got["hi"] == ""


def test_the_arbiter_difference_for_qwen3_0_6b_base(
    output: registry.FigureOutput,
) -> None:
    d, lo, hi = _diff(_contrast()["qwen3-0.6b-base"])
    assert (round(d, 3), round(lo, 3), round(hi, 3)) == (0.120, 0.050, 0.189)


def test_panel_c_has_the_stored_paired_increments(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for model, found in _increments().items():
        for series, feature in (
            ("direction", "confidence+lat"),
            ("probe", "confidence+probe"),
        ):
            got = [
                r
                for r in rows
                if r["panel"] == "c"
                and r["model"] == model
                and r["series"] == series
            ]
            if feature not in found:
                assert not got
                continue
            (cell,) = got
            assert float(cell["value"]) == pytest.approx(
                found[feature]["diff_vs_confidence"]
            )
            assert float(cell["lo"]) == pytest.approx(found[feature]["lo"])
            assert float(cell["hi"]) == pytest.approx(found[feature]["hi"])
    assert "gpt2" not in {
        r["model"]
        for r in rows
        if r["panel"] == "c" and r["series"] == "direction"
    }


def test_counts_are_per_role_and_use_the_interval_criterion(
    output: registry.FigureOutput,
) -> None:
    point = _by_role(
        _flags(lambda r: r["logistic_within"] > r["popularity_within"])
    )
    interval = _by_role(_flags(lambda r: _diff(r)[1] > 0))
    probe = _by_role(
        {m: f["confidence+probe"]["lo"] > 0 for m, f in _increments().items()}
    )
    for name, want in (
        ("above_best_point", point),
        ("above_best", interval),
        ("probe_above", probe),
    ):
        for role in stats.ROLES:
            got = output.counts[name][role]
            assert (got.hits, got.total) == (want[role].hits, want[role].total)
            assert output.numbers[f"{name}_{role}"] == want[role].hits
    assert output.counts["above_best"]["gpt2"].total == 3
    assert "above_pop" not in output.counts


POOLED = (
    "dir_inc_min",
    "dir_inc_max",
    "dir_inc_n",
    "dir_inc_hi_max",
    "dir_inc_cover_all",
    "probe_inc_min",
    "probe_inc_max",
)


def test_increment_numbers_are_per_role_and_none_is_pooled(
    output: registry.FigureOutput,
) -> None:
    for key in POOLED:
        assert key not in output.numbers, key
    for role in stats.ROLES:
        lat = [
            f["confidence+lat"]
            for m, f in _increments().items()
            if "confidence+lat" in f and stats.role(m) == role
        ]
        probe = [
            f["confidence+probe"]
            for m, f in _increments().items()
            if stats.role(m) == role
        ]
        diffs = [r["diff_vs_confidence"] for r in lat]
        gains = [r["diff_vs_confidence"] for r in probe]
        n = output.numbers
        assert n[f"dir_inc_min_{role}"] == pytest.approx(min(diffs))
        assert n[f"dir_inc_max_{role}"] == pytest.approx(max(diffs))
        assert n[f"dir_inc_hi_max_{role}"] == pytest.approx(
            max(r["hi"] for r in lat)
        )
        assert n[f"probe_inc_min_{role}"] == pytest.approx(min(gains))
        assert n[f"probe_inc_max_{role}"] == pytest.approx(max(gains))
    cover = output.counts["dir_cover"]
    assert sum(c.total for c in cover.values()) == 9


def test_the_difference_is_taken_against_the_strongest_control() -> None:
    row = {
        "logistic_within": 0.80,
        "popularity_within": 0.70,
        "text_word_within": 0.75,
        "text_char_within": 0.60,
        "n_positive_test": 120,
        "n_negative_test": 130,
    }
    got = knowledge_reading._difference(row)
    want = stats.auc_diff_interval(0.80, 0.75, 120, 130)
    assert got == pytest.approx(want)
    row["text_char_within"] = 0.78
    assert knowledge_reading._difference(row) == pytest.approx(
        stats.auc_diff_interval(0.80, 0.78, 120, 130)
    )
    row["text_char_within"] = 0.5
    row["text_word_within"] = 0.5
    assert knowledge_reading._difference(row) == pytest.approx(
        stats.auc_diff_interval(0.80, 0.70, 120, 130)
    )


def test_panel_b_names_its_comparator(output: registry.FigureOutput) -> None:
    ax = output.figure.axes[1]
    assert ax.get_xlabel() == "direction minus\nstrongest control"


def test_the_probe_column_is_explained_in_the_caption(
    output: registry.FigureOutput,
) -> None:
    assert "a probe fitted on correct and wrong labels" in output.detail


def test_the_caption_names_the_task_of_each_panel(
    output: registry.FigureOutput,
) -> None:
    assert output.detail.startswith(
        "(a) Within-relation AUROC, known against unknown questions: "
        "direction, popularity and the two text classifiers (green bar: "
        "their range)."
    )
    assert (
        "(c) A second task, wrong against correct answers: change in pooled "
        "AUROC when the direction, or a probe fitted on correct and wrong "
        "labels, is added to the model's confidence."
    ) in output.detail
    assert "pentagon" not in output.detail


def test_popularity_is_the_strongest_control_where_the_caption_says_so() -> (
    None
):
    for model, row in _contrast().items():
        assert row["popularity_within"] > max(row[c] for c in TEXT_CONTROLS), (
            model
        )


def test_caption_is_at_most_90_words_and_opens_with_what_is_plotted(
    output: registry.FigureOutput,
) -> None:
    caption = f"{output.claim} {output.detail}"
    assert len(caption.split()) <= 90, len(caption.split())
    assert output.claim == (
        "A knowledge direction beside its controls and the model's "
        "confidence on PopQA."
    )
    assert "known-against-unknown" not in caption
    assert "small amount" not in caption


def test_caption_leaves_the_counts_and_levels_to_the_text(
    output: registry.FigureOutput,
) -> None:
    caption = f"{output.claim} {output.detail}"
    assert "[[" not in caption
    assert not any(c.isdigit() for c in caption)
    assert "all 9 checkpoints" not in caption
    assert "Hanley and McNeil" not in caption
    cover = _by_role(
        {
            m: f["confidence+lat"]["lo"] <= 0 <= f["confidence+lat"]["hi"]
            for m, f in _increments().items()
            if "confidence+lat" in f
        }
    )
    assert output.counts["dir_cover"] == cover
    interval = _by_role(_flags(lambda r: _diff(r)[1] > 0))
    assert output.counts["above_best"] == interval


def test_size_guards_and_layout(output: registry.FigureOutput) -> None:
    fig = output.figure
    assert tuple(fig.get_size_inches()) == (paper.FULL_WIDTH, HEIGHT)
    assert checks.guard_problems(fig) == []
    assert checks.problems(fig) == []


def test_dot_plot_rows_have_no_connecting_lines(
    output: registry.FigureOutput,
) -> None:
    for ax in output.figure.axes:
        for line in ax.get_lines():
            assert len(line.get_xdata()) <= 2


def test_svg_and_pdf_bytes_are_the_same_for_the_same_input() -> None:
    first = registry.build_figure(SPEC, Manifest(RESULTS))
    second = registry.build_figure(SPEC, Manifest(RESULTS))
    with paper.paper_style():
        assert paper.svg_bytes(first.figure) == paper.svg_bytes(second.figure)
        assert paper.pdf_bytes(first.figure) == paper.pdf_bytes(second.figure)


MODELS_A = [m for m in _models("p9b") if m in _contrast()]
LABELS = ["medium", "large", "XL", "0.6B", "1.7B", "4B", "0.6B", "1.7B", "4B"]
GROUPS = ["GPT-2", "Qwen3\nbase", "Qwen3\npost-trained"]


def _points(ax: Any, marker: str) -> list[Any]:
    return [ln for ln in ax.get_lines() if ln.get_marker() == marker]


def test_four_axes_share_block_rows_without_the_empty_gpt2_small_row(
    output: registry.FigureOutput,
) -> None:
    axes_ = output.figure.axes
    assert len(axes_) == 4
    for ax in axes_[1:]:
        assert axes_[0].get_shared_y_axes().joined(axes_[0], ax)
    assert [t.get_text() for t in axes_[0].get_yticklabels()] == LABELS
    drawn = {r["model"] for r in _rows(output) if r["panel"] != "c"}
    assert "gpt2" not in drawn


def test_rows_sit_in_role_blocks_named_on_the_left(
    output: registry.FigureOutput,
) -> None:
    ax = output.figure.axes[0]
    names = [t for t in ax.texts if t.get_text() in GROUPS]
    assert [t.get_text() for t in names] == GROUPS
    ys = [t.get_position()[1] for t in names]
    assert ys == sorted(ys)
    ticks = list(ax.get_yticks())
    gaps = [b - a for a, b in itertools.pairwise(ticks)]
    assert sorted(gaps)[-2:] == pytest.approx([max(gaps), max(gaps)])
    assert max(gaps) > 1.0


def test_panel_a_holds_only_the_known_against_unknown_task(
    output: registry.FigureOutput,
) -> None:
    series = {r["series"] for r in _rows(output) if r["panel"] == "a"}
    assert series == {"direction", "popularity", "words", "characters"}
    ax = output.figure.axes[0]
    marker = paper.MEASURE_MARKERS["answer_probability"]
    assert _points(ax, marker) == []


def test_panel_a_draws_one_green_range_for_the_two_text_classifiers(
    output: registry.FigureOutput,
) -> None:
    ax = output.figure.axes[0]
    green = [
        ln
        for ln in ax.get_lines()
        if ln.get_color() == paper.TEXT_GREEN and ln.get_marker() == "None"
    ]
    assert len(green) == len(MODELS_A)
    for model, line in zip(MODELS_A, green, strict=True):
        row = _contrast()[model]
        lo = min(row[c] for c in TEXT_CONTROLS)
        hi = max(row[c] for c in TEXT_CONTROLS)
        assert min(line.get_xdata()) == pytest.approx(lo)
        assert max(line.get_xdata()) == pytest.approx(hi)
    assert not _points(ax, "^")
    assert not _points(ax, "v")


def test_panel_a_direction_is_a_circle_and_popularity_a_square(
    output: registry.FigureOutput,
) -> None:
    ax = output.figure.axes[0]
    circles = _points(ax, "o")
    squares = _points(ax, "s")
    assert len(circles) == len(squares) == len(MODELS_A)
    for model, line in zip(MODELS_A, circles, strict=True):
        post = stats.role(model) == "post"
        assert (line.get_markerfacecolor() == "white") is post
    assert {s.get_color() for s in squares} == {paper.ENC["popularity"]}


def test_panel_b_is_the_difference_with_its_hanley_mcneil_interval(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for model in MODELS_A:
        d, lo, hi = _diff(_contrast()[model])
        (got,) = [
            r
            for r in rows
            if r["panel"] == "b"
            and r["model"] == model
            and r["series"] == "difference"
        ]
        assert float(got["value"]) == pytest.approx(d)
        assert float(got["lo"]) == pytest.approx(lo)
        assert float(got["hi"]) == pytest.approx(hi)
    ax = output.figure.axes[1]
    zero = [ln for ln in ax.get_lines() if list(ln.get_xdata()) == [0, 0]]
    assert zero
    assert len(_points(ax, "o")) == len(MODELS_A)


def test_panel_c_has_a_narrow_direction_axis_and_a_wide_probe_axis(
    output: registry.FigureOutput,
) -> None:
    narrow, wide = output.figure.axes[2:]
    low, high = narrow.get_xlim()
    assert (low, high) == (-0.035, 0.035)
    lat = [
        f["confidence+lat"]
        for f in _increments().values()
        if "confidence+lat" in f
    ]
    assert all(low < r["lo"] and r["hi"] < high for r in lat)
    assert wide.get_xlim() == (0.0, 0.08)
    assert len(_points(narrow, "o")) == len(MODELS_A)
    assert len(_points(wide, paper.MEASURE_MARKERS["probe"])) == len(MODELS_A)


def test_the_probe_only_gpt2_small_value_is_exported(
    output: registry.FigureOutput,
) -> None:
    cell = _increments()["gpt2"]["confidence+probe"]
    assert output.numbers["gpt2_small_probe_gain"] == pytest.approx(
        cell["diff_vs_confidence"]
    )


def test_legend_has_three_entries_and_no_role_hexagon(
    output: registry.FigureOutput,
) -> None:
    assert _L["marker_clashes"](output.figure) == []
    (legend,) = output.figure.legends
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == [
        "knowledge direction",
        "popularity",
        "text classifiers",
    ]
    markers = [
        str(h.get_marker()) if hasattr(h, "get_marker") else "None"
        for h in legend.legend_handles
    ]
    assert "h" not in markers
    assert markers[:2] == ["o", "s"]


def test_three_panel_letters_and_no_dashes(
    output: registry.FigureOutput,
) -> None:
    checks.draw(output.figure)
    letters = [
        t.get_text()
        for ax in output.figure.axes
        for t in ax.texts
        if t.get_text().startswith(r"$\mathbf")
    ]
    assert sorted(letters) == [rf"$\mathbf{{({x})}}$" for x in "abc"]
    text = f"{output.claim} {output.detail} {output.csv}"
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text


def test_caption_states_the_criteria(
    output: registry.FigureOutput,
) -> None:
    text = output.detail
    assert "Open markers: post-trained." in text
    assert "(b) Direction minus the strongest control, conservative" in text
    assert "the two text classifiers (green bar: their range)" in text


def test_zero_lines_stop_below_the_column_headers(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    checks.draw(fig)
    for ax in fig.axes[1:3]:
        (head,) = [
            t
            for t in ax.texts
            if t.get_text() in ("difference", "direction" + chr(10) + "added")
        ]
        top = head.get_window_extent().y0
        zero = [
            ln
            for ln in ax.get_lines()
            if ln.get_color() == paper.ENC["reference"]
        ]
        assert zero
        for line in zero:
            end = ax.transAxes.transform((0.0, max(line.get_ydata())))[1]
            assert end <= top


def test_legend_markers_are_plain_and_well_spaced(
    output: registry.FigureOutput,
) -> None:
    (legend,) = output.figure.legends
    handles = list(legend.legend_handles)
    for handle in handles:
        if handle.get_marker() != "None":
            assert handle.get_linestyle() == "None"
    checks.draw(output.figure)
    boxes = [t.get_window_extent() for t in legend.get_texts()]
    for left, right in itertools.pairwise(boxes):
        assert right.x0 - left.x1 >= 14
