import json
from pathlib import Path
import runpy
from typing import Any

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
        assert float(row["lo"]) == pytest.approx(standard["lo"], abs=1e-9)
        assert float(row["hi"]) == pytest.approx(standard["hi"], abs=1e-9)


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
    assert len(rows) == len(MODELS) * (3 + len(SETS) + len(NEGATED))


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


def test_the_figure_is_two_panels_at_the_planned_size(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    assert len(fig.axes) == 2
    width, height = fig.get_size_inches()
    assert width == pytest.approx(paper.FULL_WIDTH)
    assert height == pytest.approx(3.0)


def test_the_text_band_and_chance_lines_are_drawn(
    output: registry.FigureOutput,
) -> None:
    left, right = output.figure.axes
    lo, hi = _band()
    spans = [
        (p.get_x(), p.get_x() + p.get_width())
        for p in left.patches
        if abs(p.get_x() - lo) < 1e-9
    ]
    polys = [
        c.get_paths()[0].get_extents()
        for c in left.collections
        if c.get_paths()
    ]
    drawn = bool(spans) or any(
        abs(e.x0 - lo) < 1e-9 and abs(e.x1 - hi) < 1e-9 for e in polys
    )
    assert drawn
    n = output.numbers
    for ax, value in ((left, n["chance"]), (right, 0.5)):
        assert any(
            np.allclose(np.asarray(line.get_xdata(), float), value)
            for line in ax.get_lines()
            if len(line.get_xdata()) == 2
        )


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
    assert "(b) whiskers" in text or "whiskers in (b)" in text
    assert "two negated sets per checkpoint" not in text


_L = runpy.run_path(str(Path(__file__).with_name("_legends.py")))


def test_no_legend_symbol_means_two_things_and_the_role_key_is_present(
    output: registry.FigureOutput,
) -> None:
    assert _L["marker_clashes"](output.figure) == []
    labels = [
        t.get_text() for leg in output.figure.legends for t in leg.get_texts()
    ]
    assert "GPT-2 and base" in labels
    assert "post-trained" in labels


def test_the_below_random_count_is_labelled_a_point_estimate(
    output: registry.FigureOutput,
) -> None:
    text = output.detail
    assert "below the random direction by point estimate on" in text
