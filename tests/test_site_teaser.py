import math
from pathlib import Path
import re
import runpy
from typing import Any

import pytest

from tomography.site import registry
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


def _steering(model: str) -> Cell:
    rows = _records(run("m1", model), "table.json")

    def pick(method: str) -> dict[str, Any]:
        (found,) = [
            r
            for r in rows
            if r["task"] == "manipulation"
            and r["setting"] == "0.3"
            and r["method"] == method
        ]
        return found

    a, b = pick("logistic"), pick("random")
    return _diff(float(a["value"]), float(b["value"]), int(a["n"]), int(b["n"]))


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
        "steering": ((3, 4), (3, 3), (3, 3)),
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


def test_the_knowledge_row_takes_popularity_as_the_strongest_baseline() -> None:
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


def test_the_claim_states_the_per_role_counts_without_pooling(
    built: registry.FigureOutput,
) -> None:
    text = f"{built.claim} {built.detail}"
    assert "better than a random direction on all 10 checkpoints" in text
    assert "all 6 Qwen3 checkpoints but on none of the 4 GPT-2" in text
    assert not re.search(r"of 10 checkpoints", text)
    assert "0 of 4 GPT-2, 1 of 3 Qwen3 base and 0 of 3 post-trained" in text
    assert "on any of 9 checkpoints" in text
    assert "3 of 4 GPT-2, 3 of 3 Qwen3 base and 3 of 3 post-trained" in text
    for banned in (chr(0x2014), chr(0x2013)):
        assert banned not in text


def test_figure_is_a_single_5_5_by_3_7_in_axes_with_the_guards_green(
    built: registry.FigureOutput,
) -> None:
    fig = built.figure
    assert fig.get_size_inches()[0] == pytest.approx(paper.FULL_WIDTH)
    assert fig.get_size_inches()[1] == pytest.approx(3.7)
    assert checks.guard_problems(fig) == []
    assert checks.problems(fig) == []


def test_the_bar_is_registered_and_the_module_builds_from_the_registry() -> (
    None
):
    assert "teaser" in registry.MAIN_FIGURES
    assert ("figure", "teaser") not in registry.missing()


def test_the_claim_makes_no_blanket_statement_about_stronger_baselines(
    built: registry.FigureOutput,
) -> None:
    assert "rarely" not in built.claim
    assert not re.search(r"\bfirst\b", f"{built.claim} {built.detail}")


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
    text = f"{built.claim} {built.detail}"
    assert "-0.000" not in text
