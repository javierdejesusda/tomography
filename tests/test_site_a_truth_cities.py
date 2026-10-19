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
    return registry.build_figure("a_truth_cities", Manifest(RESULTS))


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
    assert len(rows) == len(MODELS) * (len(SETS) + len(NEGATED))


def test_the_guards_are_green(output: registry.FigureOutput) -> None:
    assert checks.guard_problems(output.figure, appendix=True) == []
    assert checks.problems(output.figure) == []


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


def test_the_gpt2_cities_range_has_its_random_direction_beside_it(
    output: registry.FigureOutput,
) -> None:
    gpt2 = [m for m in MODELS if _role(m) == "gpt2"]
    random = [_random(m, "cities") for m in gpt2]
    n = output.numbers
    assert n["gpt2_cities_random_min"] == pytest.approx(min(random))
    assert n["gpt2_cities_random_max"] == pytest.approx(max(random))
    low = f"{n['gpt2_cities_min']:.2f} to {n['gpt2_cities_max']:.2f}"
    rand = f"{min(random):.2f} to {max(random):.2f} for its random direction"
    assert f"{low}, against {rand}" in output.detail


def test_the_figure_is_two_dumbbell_panels_lettered_a_and_b(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    assert sorted(a.get_gid() for a in fig.axes) == ["cities", "translations"]
    width, height = fig.get_size_inches()
    assert width == pytest.approx(paper.FULL_WIDTH)
    assert height == pytest.approx(3.2)
    drawn = sorted(
        t.get_text()[-4]
        for ax in fig.axes
        for t in ax.texts
        if t.get_text().startswith(r"$\mathbf{(")
    )
    assert drawn == ["a", "b"]
    for ax in fig.axes:
        assert ax.get_ylim()[1] == pytest.approx(-0.7)


def test_the_claim_counts_the_checkpoints_below_chance_per_role(
    output: registry.FigureOutput,
) -> None:
    counts = output.counts["neg_below_half"]
    assert [counts[r].hits for r in ROLES] == [3, 0, 2]
    assert [counts[r].total for r in ROLES] == [4, 3, 3]
    assert stats.per_role_text(counts) in output.claim
    for role in ROLES:
        group = [m for m in MODELS if _role(m) == role]
        assert output.numbers[f"neg_below_half_{role}"] == sum(
            _any_below_half(m) for m in group
        )
    gpt2 = [m for m in MODELS if _role(m) == "gpt2"]
    assert output.numbers["negcities_below_random_gpt2"] == sum(
        _set(m, "neg_cities")[0] < _random(m, "neg_cities") for m in gpt2
    )


def test_the_caption_points_to_the_main_figure_and_is_self_contained(
    output: registry.FigureOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    assert "[[fig:truth]]" in text
    assert "Figure" not in text
    assert "whiskers are not drawn" in text.lower()
    assert "wholly below 0.5" in text
    for banned in (EM, EN, "_"):
        assert banned not in text
