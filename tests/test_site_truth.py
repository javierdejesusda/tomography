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


def test_panel_b_rows_hold_every_set_and_the_random_direction(
    output: registry.FigureOutput,
) -> None:
    rows = _csv_rows(output)
    for model in MODELS:
        for name in SETS:
            acc, n = _set(model, name)
            row = _find(rows, "b", model, name)
            assert float(row["value"]) == pytest.approx(acc)
            lo, hi = stats.wilson(acc, n)
            assert float(row["lo"]) == pytest.approx(lo, abs=1e-9)
            assert float(row["hi"]) == pytest.approx(hi, abs=1e-9)
        for name in NEGATED:
            row = _find(rows, "b", model, f"random_{name}")
            assert float(row["value"]) == pytest.approx(
                _random(model, name), abs=1e-9
            )
    assert len(rows) == len(MODELS) * (4 + len(SETS) + len(NEGATED))


def test_numbers_are_recomputed_per_role(
    output: registry.FigureOutput,
) -> None:
    n = output.numbers
    persona = [_lat(m, "honesty")["mc1"] for m in MODELS]
    lo, hi = _band()
    assert n["persona_min"] == pytest.approx(min(persona))
    assert n["persona_max"] == pytest.approx(max(persona))
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
    assert [counts["neg_below_half"][r].total for r in ROLES] == [4, 3, 3]
    assert output.numbers["persona_min"] == pytest.approx(0.270, abs=5e-4)
    assert output.numbers["persona_max"] == pytest.approx(0.575, abs=5e-4)


def test_the_caption_states_the_counts_and_the_numbers(
    output: registry.FigureOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    n = output.numbers
    assert stats.per_role_text(output.counts["tqa_inside"]) in text
    assert stats.per_role_text(output.counts["neg_below_half"]) in text
    for name in ("persona_min", "persona_max", "text_lo", "text_hi", "chance"):
        assert f"{n[name]:.3f}" in text, name
    assert "pooled" not in text
    assert EM not in text
    assert EN not in text
    assert "_" not in text


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
    assert "at least one negated set" in text
    assert "whiskers in (d) and (e)" in text
    assert "two negated sets per checkpoint" not in text


_L = runpy.run_path(str(Path(__file__).with_name("_legends.py")))


def test_no_legend_symbol_means_two_things_and_no_role_key_is_drawn(
    output: registry.FigureOutput,
) -> None:
    assert _L["marker_clashes"](output.figure) == []
    labels = [
        t.get_text() for leg in output.figure.legends for t in leg.get_texts()
    ]
    assert "GPT-2 and base" not in labels


def test_the_below_random_count_is_labelled_a_point_estimate(
    output: registry.FigureOutput,
) -> None:
    text = output.detail
    assert "below the random direction by point estimate on" in text


def _ax(output: registry.FigureOutput, gid: str) -> Any:
    (ax,) = [a for a in output.figure.axes if a.get_gid() == gid]
    return ax


def _gid(ax: Any, gid: str) -> list[Any]:
    return [ln for ln in ax.get_lines() if ln.get_gid() == gid]


def test_the_layout_is_five_panels_at_the_planned_size(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    assert sorted(a.get_gid() for a in fig.axes) == sorted(
        ["levels", "difference", "logprob", "cities", "translations"]
    )
    width, height = fig.get_size_inches()
    assert width == pytest.approx(paper.FULL_WIDTH)
    assert height == pytest.approx(5.8)


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


def test_the_persona_reader_is_not_pink_and_levels_has_no_chance_line(
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
    assert not _gid(ax, "chance")


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


def test_each_dumbbell_joins_the_sets_and_fills_the_negated_end_by_state(
    output: registry.FigureOutput,
) -> None:
    for gid, (plain, neg) in (
        ("cities", ("cities", "neg_cities")),
        ("translations", ("sp_en_trans", "neg_sp_en_trans")),
    ):
        ax = _ax(output, gid)
        for model in MODELS:
            acc, _ = _set(model, plain)
            nacc, nn = _set(model, neg)
            (bell,) = _gid(ax, f"dumbbell:{plain}:{model}")
            assert list(bell.get_xdata()) == pytest.approx([acc, nacc])
            (end,) = _gid(ax, f"neg:{plain}:{model}")
            filled = end.get_markerfacecolor() != "white"
            assert filled == (stats.wilson(nacc, nn)[1] < 0.5)
            (tick,) = _gid(ax, f"random:{plain}:{model}")
            assert tick.get_xdata()[0] == pytest.approx(_random(model, neg))
        (chance,) = _gid(ax, "chance")
        assert chance.get_xdata()[0] == pytest.approx(0.5)
        (shade,) = [p for p in ax.patches if p.get_gid() == "below_chance"]
        assert shade.get_x() == pytest.approx(ax.get_xlim()[0])
        assert shade.get_x() + shade.get_width() == pytest.approx(0.5)


def test_no_star_triangle_hexagon_or_cross_is_drawn(
    output: registry.FigureOutput,
) -> None:
    marks = {
        str(ln.get_marker())
        for ax in output.figure.axes
        for ln in ax.get_lines()
        if str(ln.get_marker()) not in ("None", "")
    }
    assert marks <= {"o", "p", "|"}
    assert marks.isdisjoint(set(paper.CATEGORY_MARKERS) | {"h", "x"})


def test_the_text_is_at_least_the_glossary_floor_and_never_bold(
    output: registry.FigureOutput,
) -> None:
    for t in output.figure.findobj(Text):
        if t.get_visible() and t.get_text():
            assert t.get_fontsize() >= paper.MIN_TEXT, t.get_text()
            assert t.get_fontweight() in ("normal", 400), t.get_text()


def test_the_caption_notes_the_weak_gpt2_cities_reader_and_the_criterion(
    output: registry.FigureOutput,
) -> None:
    n = output.numbers
    low = f"{n['gpt2_cities_min']:.2f} to {n['gpt2_cities_max']:.2f}"
    assert low in output.detail
    assert "not inversions of a strong reader" in output.detail
    assert "wholly below 0.5" in output.detail


def test_every_drawn_panel_letter_is_named_in_the_caption_and_vice_versa(
    output: registry.FigureOutput,
) -> None:
    drawn = sorted(
        t.get_text()[-4]
        for ax in output.figure.axes
        for t in ax.texts
        if t.get_text().startswith(r"$\mathbf{(")
    )
    assert drawn == list("abcde")
    for letter in drawn:
        assert f"({letter}" in output.detail or f", {letter})" in output.detail
    assert "(f" not in output.detail
    assert "plus)" not in output.detail
    assert "dark grey circle" in output.detail


def test_the_legend_names_the_dumbbell_panels_and_pluralises_the_classifiers(
    output: registry.FigureOutput,
) -> None:
    (legend,) = output.figure.legends
    labels = [t.get_text() for t in legend.get_texts()]
    assert any("(d, e)" in x for x in labels)
    assert not any("(b)" in x for x in labels)
    assert any(
        x.startswith("text-only classifiers on words and") for x in labels
    )


def test_the_readers_and_the_text_band_are_labelled_directly(
    output: registry.FigureOutput,
) -> None:
    ax = _ax(output, "levels")
    texts = {t.get_gid(): t.get_text() for t in ax.texts}
    assert texts["reader_label:truthfulqa_reader"] == "truth reader"
    assert texts["reader_label:persona_reader"] == "persona reader"
    assert texts["band_label"] == "words, characters"
