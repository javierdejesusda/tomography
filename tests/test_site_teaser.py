from itertools import pairwise
import math
from pathlib import Path
import re
import runpy
from typing import Any

from matplotlib.lines import Line2D
import pytest

from tomography.site import content
from tomography.site import registry
from tomography.site import stats
from tomography.site.figs import teaser
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
ROWS = (
    "utility",
    "morality",
    "truth",
    "neg_cities",
    "honesty_transfer",
    "paper_finder",
    "truthfulqa",
    "knowledge",
    "knowledge_confidence",
    "steering",
)
ROLES = ("gpt2", "base", "post")
Cell = tuple[float, float, float, float, int]


def _role(model: str) -> str:
    if not model.startswith("qwen3"):
        return "gpt2"
    return "base" if model.endswith("-base") else "post"


def _diff(p1: float, p2: float, n1: int, n2: int | None = None) -> Cell:
    n2 = n1 if n2 is None else n2
    half = Z * math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return p1 - p2, p1 - p2 - half, p1 - p2 + half, p2, n1


def _curve_random(path: Path, stem: str, layer: int) -> float:
    series = data(path, f"{stem}.json")["series"]
    mean = npz(path, f"{stem}.npz")["mean"]
    return float(mean[series.index("random")][layer - 1])


def _records(path: Path, name: str) -> list[dict[str, Any]]:
    table = data(path, name)
    if isinstance(table, dict):
        return [
            dict(zip(table["columns"], r, strict=True)) for r in table["rows"]
        ]
    return list(table)


def _reading(exp: str, model: str) -> Cell:
    path = run(exp, model)
    d = data(path, "summary.json")
    layer = int(d["best_layer"])
    random = _curve_random(path, "curves", layer)
    return _diff(
        float(d["test_accuracy"]), random, int(d["n_eval_pairs"]["test"])
    )


def _truth(model: str) -> Cell:
    path = run("r7", model)
    d = data(path, "summary.json")
    test = d["sets"]["test"]
    random = _curve_random(path, "curves", int(d["val_layer"]))
    return _diff(
        float(test["accuracy_at_val_layer"]),
        random,
        int(test["n_eval_pairs"]),
    )


def _neg_cities(model: str) -> Cell:
    path = run("r7", model)
    d = data(path, "summary.json")
    block = d["sets"]["neg_cities"]
    random = _curve_random(path, "curves_neg_cities", int(d["val_layer"]))
    return _diff(
        float(block["accuracy_at_val_layer"]),
        random,
        int(block["n_eval_pairs"]),
    )


def _honesty(model: str) -> Cell:
    path = run("r5", model)
    d = data(path, "summary.json")
    layer = int(d["best_layer"])
    random = _curve_random(path, "curves_transfer", layer)
    return _diff(
        float(d["test_accuracy"]),
        random,
        int(d["n_eval_pairs"]["transfer"]["test"]),
    )


def _paper_finder(model: str) -> Cell:
    rows = _records(run("p4", model), "table.json")

    def pick(method: str) -> float:
        (found,) = [
            r
            for r in rows
            if r["task"] == "sentiment"
            and r["method"] == method
            and r["pairing"] == "label_contrastive"
            and r["preprocess"] == "center"
        ]
        return float(found["report"])

    return _diff(pick("pca_diff"), pick("random"), 214)


def _truthfulqa(model: str) -> Cell:
    d = data(run("r6", model), "summary.json")
    reader = d["lat"]["truthfulqa"]["validation-layer"]["mc1"]
    words = d["text_baselines"]["text_word"]["mc1"]
    return _diff(float(reader), float(words), int(d["n_test"]))


def _auc_se(a: float, n_pos: int, n_neg: int) -> float:
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (
        a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)
    ) / (n_pos * n_neg)
    return math.sqrt(var)


def _knowledge(model: str) -> Cell | None:
    rows = _records(run("p9b", model), "table.json")
    found = [r for r in rows if r["type"] == "obscure_real"]
    if not found:
        return None
    r = found[0]
    best = max(
        float(r["popularity_within"]),
        float(r["text_word_within"]),
        float(r["text_char_within"]),
    )
    a = float(r["logistic_within"])
    n_pos, n_neg = int(r["n_positive_test"]), int(r["n_negative_test"])
    half = Z * math.sqrt(
        _auc_se(a, n_pos, n_neg) ** 2 + _auc_se(best, n_pos, n_neg) ** 2
    )
    return a - best, a - best - half, a - best + half, best, n_pos + n_neg


def _confidence(model: str) -> Cell | None:
    rows = _records(run("p9c", model), "table_incremental.json")
    found = [
        r
        for r in rows
        if r["dataset"] == "popqa" and r["features"] == "confidence+lat"
    ]
    if not found:
        return None
    r = found[0]
    base = next(
        x["auroc"]
        for x in rows
        if x["dataset"] == "popqa" and x["features"] == "confidence"
    )
    return (
        float(r["diff_vs_confidence"]),
        float(r["lo"]),
        float(r["hi"]),
        float(base),
        0,
    )


def _seed_rates(model: str) -> tuple[list[float], list[float]]:
    rows = _records(run("m1", model), "table_seeds.json")
    cell = [r for r in rows if r["strength"] == 0.3 and r["judge"] == "primary"]
    logistic = [r["rate"] for r in cell if r["method"] == "logistic"]
    random = [r["rate"] for r in cell if r["method"] == "random"]
    assert len(logistic) == len(random) == 3
    return logistic, random


def _steering(model: str) -> Cell:
    logistic, random = _seed_rates(model)
    gap = min(logistic) - max(random)
    return gap, gap, gap, max(random), 0


def _fluent(model: str) -> bool | None:
    path = run("m1", model) / "table_effects.json"
    if not path.exists():
        return None
    rows = _records(run("m1", model), "table_effects.json")
    none = next(r for r in rows if r["method"] == "none")
    cell = [r for r in rows if r["method"] == "logistic" and r["coeff"] == 0.3]
    ratio = sum(r["perplexity"] for r in cell) / len(cell) / none["perplexity"]
    distinct = sum(r["distinct_1"] for r in cell) / len(cell)
    return ratio <= 1.5 and distinct >= 0.15


def _expected() -> dict[str, dict[str, Cell]]:
    out: dict[str, dict[str, Cell]] = {}
    sources: dict[str, Any] = {
        "utility": ("r2", lambda m: _reading("r2", m)),
        "morality": ("r3", lambda m: _reading("r3", m)),
        "truth": ("r7", _truth),
        "neg_cities": ("r7", _neg_cities),
        "honesty_transfer": ("r5", _honesty),
        "paper_finder": ("p4", _paper_finder),
        "truthfulqa": ("r6", _truthfulqa),
        "knowledge": ("p9b", _knowledge),
        "knowledge_confidence": ("p9c", _confidence),
        "steering": ("m1", _steering),
    }
    for row, (exp, fn) in sources.items():
        cells = {m: fn(m) for m in models(exp)}
        out[row] = {m: c for m, c in cells.items() if c is not None}
    return out


EXPECTED = _expected()


@pytest.fixture(scope="module")
def built() -> registry.FigureOutput:
    return registry.build_figure("teaser", Manifest(RESULTS))


def _csv(output: registry.FigureOutput) -> list[dict[str, str]]:
    lines = output.csv.strip().splitlines()
    head = lines[0].split(",")
    assert head == [
        "row",
        "model",
        "role",
        "value",
        "lo",
        "hi",
        "baseline",
        "n",
    ]
    return [dict(zip(head, line.split(","), strict=True)) for line in lines[1:]]


def test_every_plotted_value_equals_an_independent_recomputation(
    built: registry.FigureOutput,
) -> None:
    records = _csv(built)
    assert len(records) == sum(len(v) for v in EXPECTED.values())
    for rec in records:
        value, lo, hi, base, _ = EXPECTED[rec["row"]][rec["model"]]
        assert float(rec["value"]) == pytest.approx(value, abs=1e-9)
        assert float(rec["lo"]) == pytest.approx(lo, abs=1e-9)
        assert float(rec["hi"]) == pytest.approx(hi, abs=1e-9)
        assert float(rec["baseline"]) == pytest.approx(base, abs=1e-9)
        assert rec["role"] == _role(rec["model"])


def test_each_row_has_the_checkpoints_that_the_plan_expects() -> None:
    sizes = {row: len(cells) for row, cells in EXPECTED.items()}
    assert sizes["utility"] == sizes["morality"] == sizes["truth"] == 10
    assert sizes["knowledge"] == sizes["knowledge_confidence"] == 9
    assert "gpt2" not in EXPECTED["knowledge"]
    assert "gpt2" not in EXPECTED["knowledge_confidence"]


def test_the_counts_of_checkpoints_above_zero_match_the_plan() -> None:
    plan = {
        "utility": ((4, 4), (3, 3), (3, 3)),
        "morality": ((4, 4), (3, 3), (3, 3)),
        "truth": ((0, 4), (3, 3), (3, 3)),
        "neg_cities": ((0, 4), (3, 3), (2, 3)),
        "honesty_transfer": ((0, 4), (1, 3), (2, 3)),
        "paper_finder": ((0, 4), (1, 3), (0, 3)),
        "truthfulqa": ((0, 4), (0, 3), (0, 3)),
        "knowledge": ((0, 3), (3, 3), (2, 3)),
        "knowledge_confidence": ((0, 3), (0, 3), (0, 3)),
        "steering": ((1, 4), (2, 3), (2, 3)),
    }
    for row, per_role in plan.items():
        for role, (hits, total) in zip(ROLES, per_role, strict=True):
            cells = [c for m, c in EXPECTED[row].items() if _role(m) == role]
            assert (sum(c[1] > 0 for c in cells), len(cells)) == (hits, total)


def test_counts_and_numbers_expose_all_thirty_role_counts(
    built: registry.FigureOutput,
) -> None:
    for row in ROWS:
        assert set(built.counts[row]) == set(ROLES)
        for role in ROLES:
            cells = [c for m, c in EXPECTED[row].items() if _role(m) == role]
            count = built.counts[row][role]
            assert (count.hits, count.total) == (
                sum(c[1] > 0 for c in cells),
                len(cells),
            )
            assert built.numbers[f"{row}_{role}_above"] == count.hits
            assert built.numbers[f"{row}_{role}_n"] == count.total
            assert built.numbers[f"{row}_{role}_below"] == sum(
                c[2] < 0 for c in cells
            )
        margins = [c[0] for c in EXPECTED[row].values()]
        assert built.numbers[f"{row}_min"] == pytest.approx(min(margins))
        assert built.numbers[f"{row}_max"] == pytest.approx(max(margins))
    assert built.numbers["knowledge_confidence_n"] == 9


def test_the_knowledge_row_takes_popularity_as_the_strongest_control() -> None:
    for model in EXPECTED["knowledge"]:
        r = next(
            x
            for x in _records(run("p9b", model), "table.json")
            if x["type"] == "obscure_real"
        )
        strongest = max(
            r["popularity_within"], r["text_word_within"], r["text_char_within"]
        )
        assert EXPECTED["knowledge"][model][3] == pytest.approx(strongest)
        assert strongest == r["popularity_within"]


def _full(built: registry.FigureOutput) -> str:
    (section,) = [s for s in content.APPENDICES if s.key == "appendix_reading"]
    appendix = re.sub(
        r"\[\[count:fig:teaser\.(\w+)\]\]",
        lambda m: stats.per_role_text(built.counts[m.group(1)]),
        " ".join(section.paragraphs),
    )
    return " ".join([built.claim, built.detail, appendix])


def test_the_caption_is_at_most_120_words_and_leads_with_the_conclusion(
    built: registry.FigureOutput,
) -> None:
    text = f"{built.claim} {built.detail}"
    assert len(text.split()) <= 120
    assert built.claim.startswith("A direction reads utility and morality")


def test_the_method_detail_moved_to_the_appendix_text() -> None:
    assert not hasattr(teaser, "method_detail")


def test_the_claim_states_the_per_role_counts_without_pooling(
    built: registry.FigureOutput,
) -> None:
    text = _full(built)
    assert "all 10 checkpoints" not in text
    for concept in ("utility", "morality"):
        assert stats.per_role_text(built.counts[concept]) in text
    assert (
        "above a random direction on "
        + stats.per_role_text(built.counts["utility"])
        in text
    )
    assert "truth on " + stats.per_role_text(built.counts["truth"]) in text
    assert "Qwen3 checkpoints" not in text
    assert not re.search(r"of 10 checkpoints", text)
    assert "0 of 4 GPT-2, 1 of 3 Qwen3 base and 0 of 3 post-trained" in text
    assert stats.per_role_text(built.counts["knowledge_confidence"]) in text
    assert "on any of" not in text
    assert "1 of 4 GPT-2, 2 of 3 Qwen3 base and 2 of 3 post-trained" in text
    for banned in (chr(0x2014), chr(0x2013)):
        assert banned not in text


def test_figure_is_5_5_by_4_3_in_with_two_axes_with_the_guards_green(
    built: registry.FigureOutput,
) -> None:
    fig = built.figure
    assert fig.get_size_inches()[0] == pytest.approx(paper.FULL_WIDTH)
    assert fig.get_size_inches()[1] == pytest.approx(4.3)
    assert checks.guard_problems(fig) == []
    assert checks.problems(fig) == []


def test_the_bar_is_registered_and_the_module_builds_from_the_registry() -> (
    None
):
    assert "teaser" in registry.MAIN_FIGURES
    assert ("figure", "teaser") not in registry.missing()


def test_the_claim_makes_no_blanket_statement_about_stronger_controls(
    built: registry.FigureOutput,
) -> None:
    assert "rarely" not in built.claim
    assert not re.search(r"\bfirst\b", f"{built.claim} {built.detail}")


def test_the_scorecard_names_controls_and_the_confidence_baseline(
    built: registry.FigureOutput,
) -> None:
    ax = _ax(built, "scorecard")
    blocks = [t.get_text() for t in ax.texts]
    assert "against a stronger control or baseline" in blocks
    assert ax.get_xlabel() == (
        "margin over the control or baseline of its row, "
        "in the unit under each row"
    )
    assert "checkpoint margins over the row's control or baseline;" in (
        built.detail
    )


def test_the_legend_names_the_sizes_the_figure_holds(
    built: registry.FigureOutput,
) -> None:
    labels = [
        t.get_text() for leg in built.figure.legends for t in leg.get_texts()
    ]
    assert "GPT-2 small to XL" in labels
    assert "Qwen3 0.6B to 4B" in labels


def test_the_teaser_caption_has_no_signed_zero(
    built: registry.FigureOutput,
) -> None:
    text = _full(built)
    assert "-0.000" not in text


def test_the_pairing_is_read_from_the_run_summary_not_typed_in() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/tomography/site/figs/teaser.py"
    ).read_text("utf-8")
    assert "label_contrastive" not in source
    assert '"matched"' not in source


def test_the_steering_row_counts_checkpoints_whose_every_vector_wins(
    built: registry.FigureOutput,
) -> None:
    passed = {
        m: min(_seed_rates(m)[0]) > max(_seed_rates(m)[1]) for m in models("m1")
    }
    assert sorted(m for m, ok in passed.items() if ok) == [
        "gpt2",
        "qwen3-1.7b",
        "qwen3-1.7b-base",
        "qwen3-4b",
        "qwen3-4b-base",
    ]
    for role in ROLES:
        mine = [ok for m, ok in passed.items() if _role(m) == role]
        assert built.numbers[f"steering_{role}_above"] == sum(mine)
        assert built.numbers[f"steering_{role}_n"] == len(mine)


def test_the_steering_row_names_the_unit_the_judge_and_the_fluency_rule(
    built: registry.FigureOutput,
) -> None:
    text = _full(built)
    assert "every logistic steering vector" in text
    assert "utility direction" in text
    assert "sentiment judge" in text
    assert "relative coefficient 0.3" in text
    assert "0.05" in text
    flags = {m: _fluent(m) for m in models("m1")}
    for role in ROLES:
        known = [
            f for m, f in flags.items() if _role(m) == role and f is not None
        ]
        assert built.numbers[f"steering_fluent_{role}_above"] == sum(known)
        assert built.numbers[f"steering_fluent_{role}_n"] == len(known)
    assert "no checkpoint passes" not in text


def test_every_row_names_the_unit_its_margin_is_measured_in(
    built: registry.FigureOutput,
) -> None:
    ax = _ax(built, "scorecard")
    units = [t for t in ax.texts if t.get_gid() == "row_unit"]
    units += [
        t for t in _ax(built, "steering").texts if t.get_gid() == "row_unit"
    ]
    by_y = sorted(units, key=lambda t: t.xy[1])
    assert [t.get_text() for t in by_y] == [
        "accuracy",
        "accuracy",
        "accuracy",
        "accuracy",
        "accuracy",
        "accuracy",
        "MC1 accuracy",
        "AUROC",
        "AUROC",
        "win rate",
    ]
    ticks = [*ax.get_yticks(), *_ax(built, "steering").get_yticks()]
    assert len(ticks) == len(ROWS)
    for tick, unit in zip(ticks, by_y, strict=True):
        assert unit.xy[1] == pytest.approx(tick)
    assert "unit" in ax.get_xlabel()


def _ax(built: registry.FigureOutput, gid: str) -> Any:
    (ax,) = [a for a in built.figure.axes if a.get_gid() == gid]
    return ax


def test_the_steering_row_has_its_own_panel_and_axis_label(
    built: registry.FigureOutput,
) -> None:
    top, bottom = _ax(built, "scorecard"), _ax(built, "steering")
    assert bottom.get_position().y1 < top.get_position().y0
    assert "win rate" in bottom.get_xlabel()
    assert "unit" in top.get_xlabel()
    assert len(bottom.get_yticks()) == 1
    assert len(top.get_yticks()) == len(ROWS) - 1
    zero = [ln for ln in bottom.get_lines() if ln.get_xdata()[0] == 0.0]
    assert zero
    text = f"{built.claim} {built.detail}"
    assert "no interval is stored" in text


def test_the_role_key_is_a_filled_and_an_open_circle_in_neutral_ink(
    built: registry.FigureOutput,
) -> None:
    (legend,) = built.figure.legends
    flat = legend.findobj(Line2D)
    assert "h" not in {str(h.get_marker()) for h in flat}
    circles = [
        h for h in flat if h.get_marker() == "o" and h.get_color() == paper.INK
    ]
    assert {h.get_markerfacecolor() == "white" for h in circles} == {
        True,
        False,
    }


def test_the_teaser_exports_the_truth_half_widths_per_role(
    built: registry.FigureOutput,
) -> None:
    rows = [r for r in _csv(built) if r["row"] == "truth"]
    half: dict[str, list[float]] = {"gpt2": [], "base": [], "post": []}
    for r in rows:
        half[r["role"]].append((float(r["hi"]) - float(r["lo"])) / 2)
    assert all(half.values())
    for role, values in half.items():
        assert built.numbers[f"truth_half_{role}_min"] == pytest.approx(
            min(values)
        )
        assert built.numbers[f"truth_half_{role}_max"] == pytest.approx(
            max(values)
        )
    assert "truth_half_qwen3_min" not in built.numbers
    assert stats.ROLES == ("gpt2", "base", "post")


def test_the_size_markers_of_the_legend_are_one_and_a_half_diameters_apart(
    built: registry.FigureOutput,
) -> None:
    fig = built.figure
    fig.canvas.draw()
    (legend,) = fig.legends
    sizes = [
        h
        for h in legend.findobj(Line2D)
        if h.get_marker() == "o" and h.get_color() not in (paper.INK, "none")
    ]
    assert len(sizes) == 7
    xs = [float(h.get_xydata()[0][0]) for h in sizes]
    for group in (xs[:4], xs[4:]):
        gaps = [b - a for a, b in pairwise(group)]
        assert min(gaps) >= 1.5 * teaser.MARKER
