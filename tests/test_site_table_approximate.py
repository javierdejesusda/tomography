import csv
import json
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


def _sizes() -> tuple[int, int, int]:
    people, seeds, sets = set(), set(), set()
    for model in models("p10b"):
        rows = [
            r
            for r in _csv(model, "p10b", "people.csv")
            if r["source"] == "reader" and r["template"] == "concept"
        ]
        people.update(int(float(r["n_people"])) for r in rows)
        sets.add(sum(r["method"] == "random" for r in rows))
        config = json.loads(
            (run("p10b", model) / "config.json").read_text("utf-8")
        )
        seeds.add(int(config["n_seeds"]))
    (n_people,), (n_seeds,), (n_sets,) = people, seeds, sets
    return n_people, n_seeds, n_sets


def test_the_caption_says_what_each_column_group_holds(
    output: registry.TableOutput,
) -> None:
    n_people, n_seeds, n_sets = _sizes()
    assert (n_people, n_seeds, n_sets) == (120, 5, 7)
    assert output.claim == "The values behind [[fig:approximate]]."
    assert output.detail == (
        "Left: percentage of greedy few-shot answers written as a range on "
        f"range items, exact items and the heights of {n_people} people. "
        "Reader: AUROC on held-out durations of a range reader fitted on "
        "heights and wingspans with the concept prompt, beside the mean of "
        f"{n_seeds} random directions oriented on the training labels and "
        "its difference from a classifier on the question words "
        f"({fmt(_words('gpt2'))}), with the half-width of its interval "
        "[[cite:hanley1982meaning]] and an asterisk where it excludes zero. "
        "Position: the people between the test buildings (0) and the breeds "
        "(1), beside the range of the control's position over its "
        f"{n_sets} training sets."
    )
    text = f"{output.claim} {output.detail}"
    assert "0.929" in text
    assert len(text.split()) <= 170, len(text.split())
    assert not any(ch in text for ch in DASHES)
    assert "Hanley" not in text


def test_no_underscore_backtick_or_run_code(
    output: registry.TableOutput,
) -> None:
    texts = [output.claim, output.detail, *output.columns]
    texts += [c for r in output.rows for c in r]
    texts += [g[0] for g in output.groups]
    texts = [re.sub(r"\[\[.*?\]\]", "", t) for t in texts]
    assert not any("_" in t or "`" in t for t in texts)
    assert not any(re.search(r"\b[RPMCA][0-9]+[a-d]?\b", t) for t in texts)


def test_the_superseded_counts_are_gone_and_the_kept_ones_stay(
    output: registry.TableOutput,
) -> None:
    assert "random_high" not in output.counts
    assert "high" not in output.numbers
    assert set(output.counts) == {
        "above_words",
        "above_words_point",
        "inside_band",
        "layer_one",
    }
    assert set(output.numbers) == {"text_word", "inside_band"}


def test_the_range_reader_layer_is_counted_per_role(
    output: registry.TableOutput,
) -> None:
    names = models("p10b")
    at_one = sum(float(_transfer(m, "logistic")["layer"]) == 1 for m in names)
    count = output.counts["layer_one"]
    assert sum(c.hits for c in count.values()) == at_one == 9
    assert [count[r].total for r in ("gpt2", "base", "post")] == [4, 3, 3]


def test_the_reader_above_the_words_is_counted_by_the_interval(
    output: registry.TableOutput,
) -> None:
    above = 0
    for model in models("p10b"):
        reader = _transfer(model, "logistic")
        a, b = float(reader["auroc"]), _words(model)
        n_pos, n_neg = int(reader["n_range"]), int(reader["n_exact"])
        half = (
            Z
            * (_hm_se(a, n_pos, n_neg) ** 2 + _hm_se(b, n_pos, n_neg) ** 2)
            ** 0.5
        )
        above += a - b - half > 0
    assert sum(c.hits for c in output.counts["above_words"].values()) == above


def test_the_table_prints_in_the_compact_type(
    output: registry.TableOutput,
) -> None:
    assert output.compact_print
