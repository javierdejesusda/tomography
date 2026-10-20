import json
from pathlib import Path
import runpy
from typing import Any

from matplotlib.text import Text
import numpy as np
import pytest

from tomography.site import registry
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

_RESULTS = runpy.run_path(str(Path(__file__).with_name("_results.py")))
data = _RESULTS["data"]
models = _RESULTS["models"]
run = _RESULTS["run"]

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODELS = models("r6")
ROLES = ("gpt2", "base", "post")
SETS = ("cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans")
NEGATED = {"neg_cities": "cities", "neg_sp_en_trans": "sp_en_trans"}
EM = chr(0x2014)
EN = chr(0x2013)


def _role(model: str) -> str:
    if not model.startswith("qwen3"):
        return "gpt2"
    return "base" if model.endswith("-base") else "post"


def _table(model: str) -> list[dict[str, Any]]:
    table = data(run("r6", model), "table.json")
    return [dict(zip(table["columns"], r, strict=True)) for r in table["rows"]]


def _lat(model: str, source: str) -> dict[str, Any]:
    (row,) = [
        r
        for r in _table(model)
        if r["method"] == "lat"
        and r["layer_rule"] == "validation-layer"
        and r["source"] == source
    ]
    return row


def _method(model: str, method: str) -> dict[str, Any]:
    (row,) = [r for r in _table(model) if r["method"] == method]
    return row


def _summary(model: str) -> dict[str, Any]:
    return data(run("r7", model), "summary.json")


def _set(model: str, name: str) -> tuple[float, int]:
    cell = _summary(model)["sets"][name]
    return cell["accuracy_at_val_layer"], cell["n_eval_pairs"]


def _random(model: str, name: str) -> float:
    path = run("r7", model)
    series = json.loads((path / f"curves_{name}.json").read_text("utf-8"))[
        "data"
    ]["series"]
    layer = _summary(model)["val_layer"]
    z = np.load(path / f"curves_{name}.npz")
    x = list(z["x"])
    return float(z["mean"][series.index("random"), x.index(layer)])


def _n_questions(model: str) -> int:
    return int(data(run("r6", model), "summary.json")["n_test"])


def _band() -> tuple[float, float]:
    rows = [_method(MODELS[0], m) for m in ("text_word", "text_char")]
    return min(r["lo"] for r in rows), max(r["hi"] for r in rows)


def _any_below_half(model: str) -> bool:
    return any(_below_half(model, name) for name in NEGATED)


def _inside(model: str) -> bool:
    lo, hi = _band()
    return lo <= _lat(model, "truthfulqa")["mc1"] <= hi


def _below_half(model: str, name: str) -> bool:
    acc, n = _set(model, name)
    return stats.wilson(acc, n)[1] < 0.5


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure("truth", Manifest(RESULTS))


def _csv_rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    lines = output.csv.strip().splitlines()
    head = lines[0].split(",")
    assert head == ["panel", "model", "role", "series", "value", "lo", "hi"]
    return [dict(zip(head, ln.split(","), strict=True)) for ln in lines[1:]]


def _find(
    rows: list[dict[str, str]], panel: str, model: str, series: str
) -> dict[str, str]:
    (row,) = [
        r
        for r in rows
        if r["panel"] == panel and r["model"] == model and r["series"] == series
    ]
    return row


def test_panel_a_rows_hold_the_readers_and_the_log_probability_row(
    output: registry.FigureOutput,
) -> None:
    rows = _csv_rows(output)
    for model in MODELS:
        for series, source in (
            ("truthfulqa_reader", "truthfulqa"),
            ("persona_reader", "honesty"),
        ):
            row = _find(rows, "a", model, series)
            mc1 = _lat(model, source)["mc1"]
            assert float(row["value"]) == pytest.approx(mc1)
            lo, hi = stats.wilson(mc1, _n_questions(model))
            assert float(row["lo"]) == pytest.approx(lo, abs=1e-9)
            assert float(row["hi"]) == pytest.approx(hi, abs=1e-9)
            assert row["role"] == _role(model)
        standard = _method(model, "standard")
        row = _find(rows, "a", model, "log_probability")
        assert float(row["value"]) == pytest.approx(standard["mc1"])
        lo, hi = stats.wilson(standard["mc1"], _n_questions(model))
        assert float(row["lo"]) == pytest.approx(lo, abs=1e-9)
        assert float(row["hi"]) == pytest.approx(hi, abs=1e-9)


def test_numbers_are_recomputed_per_role(
    output: registry.FigureOutput,
) -> None:
    n = output.numbers
    lo, hi = _band()
    for role in ROLES:
        persona = [
            _lat(m, "honesty")["mc1"] for m in MODELS if _role(m) == role
        ]
        assert n[f"persona_{role}_min"] == pytest.approx(min(persona))
        assert n[f"persona_{role}_max"] == pytest.approx(max(persona))
    assert "persona_min" not in n
    assert "persona_max" not in n
    assert n["text_lo"] == pytest.approx(lo)
    assert n["text_hi"] == pytest.approx(hi)
    chance = json.loads(
        (run("r6", MODELS[0]) / "curves_truthfulqa.json").read_text("utf-8")
    )["data"]["chance"]
    assert n["chance"] == pytest.approx(chance)
    for role in ROLES:
        group = [m for m in MODELS if _role(m) == role]
        assert n[f"tqa_inside_{role}"] == sum(_inside(m) for m in group)
        words = max(
            _method(group[0], k)["mc1"] for k in ("text_word", "text_char")
        )
        assert n[f"tqa_above_words_{role}"] == sum(
            _lat(m, "truthfulqa")["mc1"] > words for m in group
        )
        below = sum(_any_below_half(m) for m in group)
        assert n[f"neg_below_half_{role}"] == below
        assert n[f"neg_below_half_n_{role}"] == len(group)
    gpt2 = [m for m in MODELS if _role(m) == "gpt2"]
    assert n["negcities_below_random_gpt2"] == sum(
        _set(m, "neg_cities")[0] < _random(m, "neg_cities") for m in gpt2
    )


def test_counts_are_per_role(output: registry.FigureOutput) -> None:
    for role in ROLES:
        group = [m for m in MODELS if _role(m) == role]
        inside = output.counts["tqa_inside"][role]
        assert (inside.hits, inside.total) == (
            sum(_inside(m) for m in group),
            len(group),
        )
        below = output.counts["neg_below_half"][role]
        assert (below.hits, below.total) == (
            sum(_any_below_half(m) for m in group),
            len(group),
        )


def test_the_plan_values_hold_at_the_current_results(
    output: registry.FigureOutput,
) -> None:
    counts = output.counts
    assert [counts["tqa_inside"][r].hits for r in ROLES] == [4, 2, 3]
    assert [counts["tqa_inside"][r].total for r in ROLES] == [4, 3, 3]
    assert [counts["neg_below_half"][r].hits for r in ROLES] == [3, 0, 2]
    assert output.numbers["persona_gpt2_min"] == pytest.approx(0.270, abs=5e-4)
    assert output.numbers["persona_post_max"] == pytest.approx(0.575, abs=5e-4)


def test_the_caption_states_the_counts_and_the_numbers(
    output: registry.FigureOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    n = output.numbers
    assert stats.per_role_text(output.counts["tqa_inside"]) in text
    names = ["text_lo", "text_hi", "chance"]
    for role in ROLES:
        names += [f"persona_{role}_min", f"persona_{role}_max"]
    for name in names:
        assert f"{n[name]:.3f}" in text, name
    for role in ROLES:
        lo, hi = n[f"persona_{role}_min"], n[f"persona_{role}_max"]
        assert f"{lo:.3f} to {hi:.3f}" in text
    assert "pooled" not in text
    assert EM not in text
    assert EN not in text
    assert "[[fig:a_truth_cities]]" in text
    assert "_" not in text.replace("a_truth_cities", "")


def test_the_guards_are_green(output: registry.FigureOutput) -> None:
    assert checks.guard_problems(output.figure) == []
    assert checks.problems(output.figure) == []


def test_the_claim_does_not_say_negation_breaks_the_reader(
    output: registry.FigureOutput,
) -> None:
    assert "breaks" not in output.claim
    n = _n_questions(MODELS[0])
    assert f"over {n} questions" in output.detail


def test_the_caption_separates_the_classifier_accuracies_from_their_intervals(
    output: registry.FigureOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    n = output.numbers
    assert "intervals" in output.claim
    assert f"{n['text_word']:.3f}" in text
    assert f"{n['text_char']:.3f}" in text


_L = runpy.run_path(str(Path(__file__).with_name("_legends.py")))


def test_no_legend_symbol_means_two_things_and_the_role_key_is_drawn(
    output: registry.FigureOutput,
) -> None:
    assert _L["marker_clashes"](output.figure) == []
    labels = [
        t.get_text() for leg in output.figure.legends for t in leg.get_texts()
    ]
    assert "GPT-2 and Qwen3 base" in labels
    assert "post-trained" in labels
    assert "truth reader" not in labels


def test_post_trained_rows_are_open_markers_in_every_panel(
    output: registry.FigureOutput,
) -> None:
    for gid, key in (
        ("levels", "truthfulqa_reader"),
        ("levels", "persona_reader"),
        ("difference", "diff"),
        ("logprob", "log_probability"),
    ):
        ax = _ax(output, gid)
        for model in MODELS:
            (dot,) = [
                ln
                for ln in _gid(ax, f"{key}:{model}")
                if ln.get_marker() != "None"
            ]
            post = model.startswith("qwen3") and not model.endswith("-base")
            assert (dot.get_markerfacecolor() == "white") is post, (key, model)


def test_the_levels_panel_has_the_dotted_chance_line(
    output: registry.FigureOutput,
) -> None:
    ax = _ax(output, "levels")
    (chance,) = _gid(ax, "chance")
    assert chance.get_xdata()[0] == pytest.approx(output.numbers["chance"])
    (other,) = _gid(_ax(output, "logprob"), "chance")
    assert chance.get_linestyle() == other.get_linestyle()
    assert chance.get_color() == paper.ENC["reference"]


def test_the_zero_line_is_the_dotted_grey_reference(
    output: registry.FigureOutput,
) -> None:
    (zero,) = _gid(_ax(output, "difference"), "zero")
    assert zero.get_color() == paper.ENC["reference"]
    (chance,) = _gid(_ax(output, "logprob"), "chance")
    assert zero.get_linestyle() == chance.get_linestyle()


def test_the_text_lines_sit_in_front_of_the_green_bands(
    output: registry.FigureOutput,
) -> None:
    ax = _ax(output, "levels")
    for method in ("text_word", "text_char"):
        (line,) = _gid(ax, f"text:{method}")
        (band,) = [p for p in ax.patches if p.get_gid() == f"textband:{method}"]
        assert line.get_zorder() > band.get_zorder()
        assert line.get_linewidth() >= 1.0


def _ax(output: registry.FigureOutput, gid: str) -> Any:
    (ax,) = [a for a in output.figure.axes if a.get_gid() == gid]
    return ax


def _gid(ax: Any, gid: str) -> list[Any]:
    return [ln for ln in ax.get_lines() if ln.get_gid() == gid]


def test_the_layout_is_three_panels_at_the_planned_size(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    assert sorted(a.get_gid() for a in fig.axes) == sorted(
        ["levels", "difference", "logprob"]
    )
    width, height = fig.get_size_inches()
    assert width == pytest.approx(paper.FULL_WIDTH)
    assert height == pytest.approx(3.6)
    assert len(f"{output.claim} {output.detail}".split()) <= 200


def test_the_two_text_classifiers_are_separate_green_lines_with_own_bands(
    output: registry.FigureOutput,
) -> None:
    ax = _ax(output, "levels")
    for method in ("text_word", "text_char"):
        row = _method(MODELS[0], method)
        (line,) = _gid(ax, f"text:{method}")
        assert line.get_xdata()[0] == pytest.approx(row["mc1"])
        assert line.get_color() == paper.ENC["text"]
        (band,) = [p for p in ax.patches if p.get_gid() == f"textband:{method}"]
        assert band.get_x() == pytest.approx(row["lo"])
        assert band.get_x() + band.get_width() == pytest.approx(row["hi"])
    union = _band()
    for p in ax.patches:
        assert not (
            p.get_x() == pytest.approx(union[0])
            and p.get_x() + p.get_width() == pytest.approx(union[1])
        )


def test_the_persona_reader_is_not_pink(
    output: registry.FigureOutput,
) -> None:
    ax = _ax(output, "levels")
    for model in MODELS:
        dots = [
            ln
            for ln in _gid(ax, f"persona_reader:{model}")
            if ln.get_marker() != "None"
        ]
        assert len(dots) == 1
        assert dots[0].get_color() != paper.ENC["paper"]


def test_the_log_probability_panel_has_wilson_whiskers_and_the_chance_line(
    output: registry.FigureOutput,
) -> None:
    ax = _ax(output, "logprob")
    (chance,) = _gid(ax, "chance")
    assert chance.get_xdata()[0] == pytest.approx(output.numbers["chance"])
    for model in MODELS:
        mc1 = _method(model, "standard")["mc1"]
        lo, hi = stats.wilson(mc1, _n_questions(model))
        (whisker,) = _gid(ax, f"whisker:log_probability:{model}")
        assert list(whisker.get_xdata()) == pytest.approx([lo, hi])
        low, high = ax.get_xlim()
        assert low <= lo
        assert hi <= high


def test_the_difference_panel_draws_reader_minus_words_with_a_zero_line(
    output: registry.FigureOutput,
) -> None:
    ax = _ax(output, "difference")
    assert _gid(ax, "zero")
    excluded = {r: 0 for r in ROLES}
    for model in MODELS:
        reader = _lat(model, "truthfulqa")["mc1"]
        words = max(
            _method(model, k)["mc1"] for k in ("text_word", "text_char")
        )
        d, lo, hi = stats.diff_interval(reader, words, _n_questions(model))
        (whisker,) = _gid(ax, f"whisker:diff:{model}")
        assert list(whisker.get_xdata()) == pytest.approx([lo, hi])
        dots = [
            ln for ln in _gid(ax, f"diff:{model}") if ln.get_marker() == "o"
        ]
        assert dots[0].get_xdata()[0] == pytest.approx(d)
        excluded[_role(model)] += not (lo <= 0.0 <= hi)
    for role in ROLES:
        assert output.numbers[f"diff_excludes_zero_{role}"] == excluded[role]
        assert output.counts["diff_excludes_zero"][role].hits == excluded[role]
    assert "1.96" in output.detail


def test_no_star_triangle_hexagon_or_cross_is_drawn(
    output: registry.FigureOutput,
) -> None:
    marks = {
        str(ln.get_marker())
        for ax in output.figure.axes
        for ln in ax.get_lines()
        if str(ln.get_marker()) not in ("None", "")
    }
    assert marks <= {"o", "p"}
    assert marks.isdisjoint(set(paper.CATEGORY_MARKERS) | {"h", "x"})


def test_the_text_is_at_least_the_glossary_floor_and_never_bold(
    output: registry.FigureOutput,
) -> None:
    for t in output.figure.findobj(Text):
        if t.get_visible() and t.get_text():
            assert t.get_fontsize() >= paper.MIN_TEXT, t.get_text()
            assert t.get_fontweight() in ("normal", 400), t.get_text()


def test_every_drawn_panel_letter_is_named_in_the_caption_and_vice_versa(
    output: registry.FigureOutput,
) -> None:
    drawn = sorted(
        t.get_text()[-4]
        for ax in output.figure.axes
        for t in ax.texts
        if t.get_text().startswith(r"$\mathbf{(")
    )
    assert drawn == list("abc")
    for letter in drawn:
        assert f"({letter})" in output.detail
    assert "(d)" not in output.detail
    assert "(e)" not in output.detail
    assert "plus)" not in output.detail
    assert "dark grey circle" in output.detail
    order = [output.detail.index(f"({x})") for x in "abc"]
    assert order == sorted(order)


def test_the_legend_pluralises_the_classifiers(
    output: registry.FigureOutput,
) -> None:
    (legend,) = output.figure.legends
    labels = [t.get_text() for t in legend.get_texts()]
    assert not any("(d, e)" in x for x in labels)
    assert any(
        x.startswith("text-only classifiers on words and") for x in labels
    )


def test_the_title_names_the_truth_reader_and_no_row_carries_a_label(
    output: registry.FigureOutput,
) -> None:
    ax = _ax(output, "levels")
    assert ax.get_title() == "truth reader and text"
    assert not [
        t for t in ax.texts if str(t.get_gid()).startswith("reader_label")
    ]
    texts = {t.get_gid(): t.get_text() for t in ax.texts}
    assert " ".join(texts["band_label"].split()) == "words, characters"


def test_the_three_panels_share_the_row_limits_with_headroom_for_labels(
    output: registry.FigureOutput,
) -> None:
    limits = {
        _ax(output, g).get_ylim() for g in ("levels", "difference", "logprob")
    }
    assert len(limits) == 1
    assert next(iter(limits))[1] < -1.0


def test_the_persona_reader_is_named_in_the_legend_not_beside_the_rows(
    output: registry.FigureOutput,
) -> None:
    ax = _ax(output, "levels")
    assert not [
        t for t in ax.texts if t.get_gid() == "reader_label:persona_reader"
    ]
    (legend,) = output.figure.legends
    labels = [t.get_text() for t in legend.get_texts()]
    assert "honesty persona reader" in labels
    handle = legend.legend_handles[labels.index("honesty persona reader")]
    assert handle.get_markerfacecolor() == handle.get_color()
    assert handle.get_color() != paper.INK


def test_the_text_baseline_is_drawn_in_the_log_probability_panel(
    output: registry.FigureOutput,
) -> None:
    ax = _ax(output, "logprob")
    for method, key in (("text_word", "text_word"), ("text_char", "text_char")):
        (line,) = _gid(ax, f"text:{method}")
        assert line.get_xdata()[0] == pytest.approx(output.numbers[key])
    assert ax.get_xlim()[1] >= output.numbers["text_hi"]


def test_the_band_label_sits_right_of_the_band_clear_of_the_line(
    output: registry.FigureOutput,
) -> None:
    ax = _ax(output, "levels")
    output.figure.draw_without_rendering()
    (label,) = [t for t in ax.texts if t.get_gid() == "band_label"]
    box = label.get_window_extent(output.figure.canvas.get_renderer())
    left, right = ax.transData.inverted().transform([[box.x0, 0], [box.x1, 0]])[
        :, 0
    ]
    bands = [p for p in ax.patches if str(p.get_gid()).startswith("textband")]
    edge = max(p.get_x() + p.get_width() for p in bands)
    lines = [ln for ln in _gid_prefix(ax, "text:")]
    assert left > edge
    assert right <= ax.get_xlim()[1]
    for line in lines:
        assert not left <= float(line.get_xdata()[0]) <= right
    assert label.get_fontsize() >= paper.MIN_TEXT


def _gid_prefix(ax: Any, prefix: str) -> list[Any]:
    return [ln for ln in ax.get_lines() if str(ln.get_gid()).startswith(prefix)]


def test_panel_a_ticks_reach_the_upper_limit(
    output: registry.FigureOutput,
) -> None:
    ax = _ax(output, "levels")
    ticks = [float(t) for t in ax.get_xticks()]
    assert ticks[-1] == ax.get_xlim()[1]
    output.figure.draw_without_rendering()
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert "1.0" in labels[-1]


MM = 72 / 25.4


def _marks(ax: Any) -> list[Any]:
    return [
        ln
        for ln in ax.get_lines()
        if ln.get_gid() and not str(ln.get_gid()).startswith("textband")
    ]


def _gap_pt(a: Any, b: Any, dpi: float) -> float:
    dx = max(a.x0 - b.x1, b.x0 - a.x1, 0.0)
    dy = max(a.y0 - b.y1, b.y0 - a.y1, 0.0)
    return float(np.hypot(dx, dy)) * 72 / dpi


def test_the_chance_line_is_labelled_clear_of_text_and_marks(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    for gid in ("levels", "logprob"):
        ax = _ax(output, gid)
        (label,) = [t for t in ax.texts if t.get_gid() == "chance_label"]
        assert label.get_text() == "chance"
        assert label.get_fontsize() >= paper.MIN_TEXT
        box = label.get_window_extent(renderer)
        inside = ax.get_window_extent(renderer)
        assert inside.x0 <= box.x0
        assert box.x1 <= inside.x1
        assert box.y1 <= inside.y1
        for other in ax.texts:
            if other is not label:
                far = _gap_pt(box, other.get_window_extent(renderer), fig.dpi)
                assert far >= 1.0, (gid, other.get_text())
        for mark in _marks(ax):
            if mark.get_gid() == "chance":
                continue
            gap = _gap_pt(box, mark.get_window_extent(renderer), fig.dpi)
            assert gap >= 1.0, (gid, mark.get_gid())
