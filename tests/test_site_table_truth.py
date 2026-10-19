from pathlib import Path
import re
import runpy

import numpy as np
import pytest

from tomography.site import registry
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.viz import paper

_HERE = Path(__file__).parent
_R = runpy.run_path(str(_HERE / "_results.py"))
_M = runpy.run_path(str(_HERE / "_site_math.py"))
ROOT, data, models, npz, run = (
    _R["ROOT"],
    _R["data"],
    _R["models"],
    _R["npz"],
    _R["run"],
)
fmt, groups, wilson = _M["fmt"], _M["groups"], _M["wilson"]
DASHES = (chr(0x2014), chr(0x2013))


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table("truth", Manifest(ROOT))


def _mc1(model: str, method: str, source: str | None = None) -> float:
    rows = data(run("r6", model), "table.json")
    rows = [dict(zip(rows["columns"], r, strict=True)) for r in rows["rows"]]
    hit = [
        r
        for r in rows
        if r["method"] == method
        and (
            method != "lat"
            or (r["layer_rule"] == "validation-layer" and r["source"] == source)
        )
    ]
    assert len(hit) == 1
    return float(hit[0]["mc1"])


def _transfer_random(model: str, layer: int) -> float:
    path = run("r5", model)
    series = data(path, "curves_transfer.json")["series"]
    mean = npz(path, "curves_transfer.npz")["mean"]
    return float(mean[series.index("random")][layer - 1])


def test_columns_and_header_spans(output: registry.TableOutput) -> None:
    assert output.columns == [
        "Model",
        "TruthfulQA reader",
        "ARC reader",
        "Honesty persona reader",
        "Answer log-probability",
        "Accuracy",
        "Layer",
        "Random",
        "Accuracy",
        "Random",
        "Seed agreement",
    ]
    assert output.align == "l" + "r" * 10
    assert [(g[0], g[1]) for g in output.groups] == [
        ("TruthfulQA MC1", 4),
        ("Honesty in template", 3),
        ("Honesty transfer", 2),
        ("Seed agreement", 1),
    ]
    assert not any(g[2] for g in output.groups)


def test_rows_follow_paper_order_with_a_rule_before_qwen3(
    output: registry.TableOutput,
) -> None:
    names = models("r6")
    n = len(names)
    assert [r[0] for r in output.rows[:n]] == [
        paper.display_name(m) for m in names
    ]
    assert output.row_groups == (*groups(names), n)


def test_baseline_rows_follow_the_checkpoints_with_values_in_the_table(
    output: registry.TableOutput,
) -> None:
    n = len(models("r6"))
    summary = data(run("r6", "gpt2"), "summary.json")["text_baselines"]
    chance = data(run("r6", "gpt2"), "curves_truthfulqa.json")["chance"]
    rows = output.rows[n:]
    assert [r[0] for r in rows] == [
        "Words classifier",
        "Characters classifier",
        "Chance",
    ]
    assert rows[0][1] == fmt(summary["text_word"]["mc1"])
    assert rows[1][1] == fmt(summary["text_char"]["mc1"])
    assert rows[2][1] == fmt(chance)
    assert all(len(r) == len(output.columns) for r in rows)
    assert all(c == "" for r in rows for c in r[2:])
    assert "each of the four TruthfulQA columns" in output.detail


def test_every_cell_matches_a_recomputation(
    output: registry.TableOutput,
) -> None:
    for row, model in zip(output.rows, models("r6"), strict=False):
        assert row[1] == fmt(_mc1(model, "lat", "truthfulqa"))
        assert row[2] == fmt(_mc1(model, "lat", "arc"))
        assert row[3] == fmt(_mc1(model, "lat", "honesty"))
        assert row[4] == fmt(_mc1(model, "standard"))
        summary = data(run("r5", model), "summary.json")
        block = summary["in_template"]
        assert row[5] == fmt(block["test_accuracy"])
        assert row[6] == str(block["best_layer"])
        r5 = run("r5", model)
        meta = data(r5, "curves.json")
        with np.load(r5 / "curves.npz") as arrays:
            mean = np.asarray(arrays["mean"], dtype=float)
        layer = block["best_layer"]
        series = list(meta["series"]).index("random")
        assert row[7] == fmt(mean[series][layer - 1])
        assert row[8] == fmt(summary["test_accuracy"])
        assert row[9] == fmt(_transfer_random(model, summary["best_layer"]))
        agree = data(run("a1", model), "summary.json")["agreement_at_layer"]
        assert row[10] == fmt(agree["r5"])


def test_the_caption_quotes_the_text_baselines_chance_and_the_wilson_width(
    output: registry.TableOutput,
) -> None:
    summary = data(run("r6", "gpt2"), "summary.json")
    word = summary["text_baselines"]["text_word"]
    char = summary["text_baselines"]["text_char"]
    n = summary["n_test"]
    chance = data(run("r6", "gpt2"), "curves_truthfulqa.json")["chance"]
    lo, hi = wilson(0.5, n)
    text = output.claim + " " + output.detail
    assert f"{fmt(word['mc1'])} [{fmt(word['lo'])}, {fmt(word['hi'])}]" in text
    assert f"{fmt(char['mc1'])} [{fmt(char['lo'])}, {fmt(char['hi'])}]" in text
    assert f"chance is {fmt(chance)}" in text
    assert f"\u00b1{fmt((hi - lo) / 2)}" in text
    assert f"{n} test questions" in text
    assert "0.718" not in text


def test_the_in_template_sentence_names_the_checkpoints_by_layer(
    output: registry.TableOutput,
) -> None:
    layers = {
        m: data(run("r5", m), "summary.json")["in_template"]["best_layer"]
        for m in models("r6")
    }
    odd = [m for m, v in layers.items() if v != 1]
    text = output.detail
    assert "not a held-out measure of honesty" in text
    for model in odd:
        assert paper.display_name(model) in text
        assert f"layer {layers[model]}" in text
    assert output.numbers["in_template_layer1"] == sum(
        v == 1 for v in layers.values()
    )


def test_the_seed_agreement_is_read_at_half_depth_and_the_transfer_has_n(
    output: registry.TableOutput,
) -> None:
    text = output.detail
    assert "relative depth 0.5" in text
    assert "not at the transfer layer" in text
    n = data(run("r5", "gpt2"), "summary.json")["n_eval_pairs"]["transfer"]
    assert f"{n['test']} test pairs" in text
    assert f"{n['val']} validation pairs" in text


def test_no_dash_underscore_or_run_code(output: registry.TableOutput) -> None:
    texts = [output.claim, output.detail, *output.columns]
    texts += [c for r in output.rows for c in r]
    texts += [g[0] for g in output.groups]
    texts = [re.sub(r"\[\[.*?\]\]", "", t) for t in texts]
    assert not any(ch in t for t in texts for ch in DASHES)
    assert not any("_" in t for t in texts)
    assert not any(re.search(r"\b[RPMCA][0-9]+[a-d]?\b", t) for t in texts)


def test_the_layer_one_count_is_per_role(
    output: registry.TableOutput,
) -> None:
    layers = {
        m: data(run("r5", m), "summary.json")["in_template"]["best_layer"]
        for m in models("r6")
    }
    expected = stats.per_role_text(
        stats.per_role({m: v == 1 for m, v in layers.items()})
    )
    assert f"at layer 1 on {expected}" in output.detail
    assert "on 9 checkpoints" not in output.detail


def test_the_in_template_random_direction_is_quoted_per_role(
    output: registry.TableOutput,
) -> None:
    randoms: dict[str, list[float]] = {r: [] for r in stats.ROLES}
    for model in models("r6"):
        r5 = run("r5", model)
        layer = data(r5, "summary.json")["in_template"]["best_layer"]
        meta = data(r5, "curves.json")
        with np.load(r5 / "curves.npz") as npz:
            mean = np.asarray(npz["mean"], dtype=float)
        value = mean[list(meta["series"]).index("random")][layer - 1]
        randoms[stats.role(model)].append(value)
    text = output.detail
    for pooled in ("template_random_min", "template_random_max"):
        assert pooled not in output.numbers
        assert pooled not in text
    for role in stats.ROLES:
        low = output.numbers[f"template_random_{role}_min"]
        assert low == pytest.approx(min(randoms[role]))
        high = output.numbers[f"template_random_{role}_max"]
        assert high == pytest.approx(max(randoms[role]))
    for end in ("min", "max"):
        found = [text.find(f"template_random_{r}_{end}") for r in stats.ROLES]
        assert min(found) >= 0
        assert found == sorted(found)
    assert "(GPT-2)" in text
    assert "(Qwen3 base)" in text
    assert "(post-trained)" in text
    assert "[[table:reading_full]]" in text
