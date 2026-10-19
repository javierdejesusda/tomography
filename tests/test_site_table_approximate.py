import csv
from pathlib import Path
import re
import runpy

import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import paper

_HERE = Path(__file__).parent
_R = runpy.run_path(str(_HERE / "_results.py"))
_M = runpy.run_path(str(_HERE / "_site_math.py"))
ROOT, models, run = _R["ROOT"], _R["models"], _R["run"]
fmt, groups = _M["fmt"], _M["groups"]
PLUS_MINUS = chr(0x00B1)
DASHES = (chr(0x2014), chr(0x2013))
Z = 1.959963984540054
TRAIN = "height+wingspan"


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table("approximate", Manifest(ROOT))


def _csv(model: str, exp: str, name: str) -> list[dict[str, str]]:
    with (run(exp, model) / name).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _hm_se(a: float, n_pos: int, n_neg: int) -> float:
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    var = (
        a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)
    ) / (n_pos * n_neg)
    return var**0.5


def _share(model: str) -> list[str]:
    cell = {
        "condition": "free",
        "format": "fewshot",
        "shot_order": "range_last",
        "decoding": "greedy",
    }
    table = [
        r
        for r in _csv(model, "p10a", "table.csv")
        if r["family"] == "all" and all(r[k] == v for k, v in cell.items())
    ]
    by_kind = {r["kind"]: float(r["range_rate"]) for r in table}
    diag = [
        r
        for r in _csv(model, "p10a", "diagnostic.csv")
        if all(r[k] == v for k, v in cell.items())
    ]
    assert len(diag) == 1
    rates = [by_kind["range"], by_kind["exact"], float(diag[0]["people_rate"])]
    return [str(round(100 * v)) for v in rates]


def _transfer(model: str, method: str) -> dict[str, str]:
    hit = [
        r
        for r in _csv(model, "p10b", "transfer.csv")
        if r["template"] == "concept"
        and r["method"] == method
        and r["train_set"] == TRAIN
        and r["test_family"] == "duration"
    ]
    assert len(hit) == 1
    return hit[0]


def _words(model: str) -> float:
    hit = [
        r
        for r in _csv(model, "p10b", "baselines.csv")
        if r["baseline"] == "text_word"
        and r["train_set"] == TRAIN
        and r["test_family"] == "duration"
    ]
    assert len(hit) == 1
    return float(hit[0]["auroc"])


def _people(model: str) -> tuple[float, float, float]:
    rows = [
        r
        for r in _csv(model, "p10b", "people.csv")
        if r["source"] == "reader" and r["template"] == "concept"
    ]
    reader = [
        r for r in rows if r["method"] == "logistic" and r["train_set"] == TRAIN
    ]
    random = [float(r["position"]) for r in rows if r["method"] == "random"]
    assert len(reader) == 1
    assert len(random) == 7
    return float(reader[0]["position"]), min(random), max(random)


def test_columns_and_header_spans(output: registry.TableOutput) -> None:
    assert output.columns == [
        "Model",
        "Range items",
        "Exact items",
        "People",
        "Reader",
        "Random control",
        f"Reader {chr(0x2212)} words",
        "Position",
        "Random band",
        "Inside band",
    ]
    assert [(g[0], g[1]) for g in output.groups] == [
        ("Greedy answers written as a range (percent)", 3),
        ("AUROC on held-out durations", 3),
        ("People between buildings (0) and breeds (1)", 3),
    ]
    assert output.align == "l" + "r" * 9


def test_rows_follow_paper_order_with_a_family_rule(
    output: registry.TableOutput,
) -> None:
    names = models("p10b")
    assert [r[0] for r in output.rows] == [paper.display_name(m) for m in names]
    assert output.row_groups == groups(names)


def test_every_cell_matches_a_recomputation(
    output: registry.TableOutput,
) -> None:
    for row, model in zip(output.rows, models("p10b"), strict=True):
        assert row[1:4] == _share(model), model
        reader, random = (
            _transfer(model, "logistic"),
            _transfer(model, "random"),
        )
        assert row[4] == fmt(float(reader["auroc"]))
        assert row[5] == fmt(float(random["auroc"]))
        a, w = float(reader["auroc"]), _words(model)
        n_pos, n_neg = int(reader["n_range"]), int(reader["n_exact"])
        half = (
            Z
            * (_hm_se(a, n_pos, n_neg) ** 2 + _hm_se(w, n_pos, n_neg) ** 2)
            ** 0.5
        )
        d = a - w
        star = "*" if d - half > 0 or d + half < 0 else ""
        assert row[6] == f"{fmt(d)} {PLUS_MINUS} {fmt(half)}{star}"
        position, lo, hi = _people(model)
        assert row[7] == fmt(position)
        assert row[8] == f"[{fmt(lo)}, {fmt(hi)}]"
        assert row[9] == ("yes" if lo <= position <= hi else "no")


def test_counts_of_the_reader_above_words_are_per_role(
    output: registry.TableOutput,
) -> None:
    names = models("p10b")
    count = output.counts["above_words"]
    assert [count[r].total for r in ("gpt2", "base", "post")] == [4, 3, 3]
    point = sum(
        float(_transfer(m, "logistic")["auroc"]) > _words(m) for m in names
    )
    assert (
        sum(c.hits for c in output.counts["above_words_point"].values())
        == point
    )


def test_the_caption_carries_the_people_diagnostic_and_the_random_band(
    output: registry.TableOutput,
) -> None:
    text = output.claim + " " + output.detail
    assert "120 people" in text
    assert "random control" in text
    assert "5 random directions" in text
    assert "7 training sets" in text
    assert "0.929" in text
    assert not any(ch in text for ch in DASHES)


def test_no_underscore_backtick_or_run_code(
    output: registry.TableOutput,
) -> None:
    texts = [output.claim, output.detail, *output.columns]
    texts += [c for r in output.rows for c in r]
    texts += [g[0] for g in output.groups]
    texts = [re.sub(r"\[\[.*?\]\]", "", t) for t in texts]
    assert not any("_" in t or "`" in t for t in texts)
    assert not any(re.search(r"\b[RPMCA][0-9]+[a-d]?\b", t) for t in texts)


def test_the_claim_uses_the_figure_framing_of_the_random_control(
    output: registry.TableOutput,
) -> None:
    claim = output.claim
    assert "matched" not in claim
    assert "[[count:table:approximate.above_words]]" in claim
    assert "[[count:fig:approximate.reader_above_random]]" in claim
    assert "[[count:fig:approximate.control_below_half]]" in claim
    assert "[[count:fig:approximate.control_above_95]]" in claim
    assert "reader_in_random" not in claim
    assert "people_outside_band" not in claim
    assert "random_high" not in output.counts
    assert "high" not in output.numbers


def test_the_range_reader_layer_is_counted_per_role(
    output: registry.TableOutput,
) -> None:
    names = models("p10b")
    at_one = sum(float(_transfer(m, "logistic")["layer"]) == 1 for m in names)
    count = output.counts["layer_one"]
    assert sum(c.hits for c in count.values()) == at_one == 9
    assert [count[r].total for r in ("gpt2", "base", "post")] == [4, 3, 3]


def test_the_claim_names_the_interval_criterion(
    output: registry.TableOutput,
) -> None:
    assert (
        "above a classifier on the question words, with the interval of the "
        "difference above zero, on "
    ) in output.claim
