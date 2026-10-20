import glob
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
ROOT, data, models, run = _R["ROOT"], _R["data"], _R["models"], _R["run"]
fmt, groups = _M["fmt"], _M["groups"]
MINUS = chr(0x2212)
PLUS_MINUS = chr(0x00B1)
DASHES = (chr(0x2014), chr(0x2013))
Z = 1.959963984540054
NO_CONTRAST = "no contrast"


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table("knowledge", Manifest(ROOT))


def _table(exp: str, model: str, name: str) -> list[dict]:
    table = data(run(exp, model), name)
    return [dict(zip(table["columns"], r, strict=True)) for r in table["rows"]]


def _hm_se(a: float, n_pos: int, n_neg: int) -> float:
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    var = (
        a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)
    ) / (n_pos * n_neg)
    return var**0.5


def _steering(model: str) -> tuple[int, int, int]:
    found = hits = selective = 0
    for path in glob.glob(str(ROOT / "p9d" / model / "*")):
        head = json.loads(Path(path, "summary.json").read_text("utf-8"))[
            "data"
        ]["headline"]
        found += 1
        hits += bool(head["distinguishable"])
        selective += bool(head["selective"])
    return found, hits, selective


def test_columns_and_header_spans(output: registry.TableOutput) -> None:
    assert output.columns == [
        "Model",
        "Direction",
        "Popularity",
        "Words",
        "Characters",
        f"Direction {MINUS} popularity",
        "+ direction",
        "+ probe",
        "Distinguishable",
        "Selective",
    ]
    assert [(g[0], g[1]) for g in output.groups] == [
        ("Within-relation AUROC", 4),
        ("Difference", 1),
        ("Added to the model's confidence", 2),
        ("Steering runs", 2),
    ]
    assert output.align == "l" + "r" * 9


def test_rows_in_paper_order_with_a_gpt2_qwen3_rule(
    output: registry.TableOutput,
) -> None:
    names = models("p9a")
    assert [r[0] for r in output.rows] == [paper.display_name(m) for m in names]
    assert output.row_groups == groups(names)


def test_reading_cells_match_a_recomputation(
    output: registry.TableOutput,
) -> None:
    for row, model in zip(output.rows, models("p9a"), strict=True):
        found = [
            r
            for r in _table("p9b", model, "table.json")
            if r["type"] == "obscure_real"
        ]
        if not found:
            assert row[1:6] == [NO_CONTRAST] * 5, model
            continue
        r = found[0]
        assert row[1] == fmt(r["logistic_within"])
        assert row[2] == fmt(r["popularity_within"])
        assert row[3] == fmt(r["text_word_within"])
        assert row[4] == fmt(r["text_char_within"])
        n_pos, n_neg = r["n_positive_test"], r["n_negative_test"]
        d = r["logistic_within"] - r["popularity_within"]
        half = (
            Z
            * (
                _hm_se(r["logistic_within"], n_pos, n_neg) ** 2
                + _hm_se(r["popularity_within"], n_pos, n_neg) ** 2
            )
            ** 0.5
        )
        star = "*" if d - half > 0 or d + half < 0 else ""
        assert row[5] == f"{fmt(d)} {PLUS_MINUS} {fmt(half)}{star}"


def test_increment_cells_carry_the_stored_paired_interval(
    output: registry.TableOutput,
) -> None:
    for row, model in zip(output.rows, models("p9a"), strict=True):
        inc = {
            r["features"]: r
            for r in _table("p9c", model, "table_incremental.json")
            if r["dataset"] == "popqa"
        }

        def cell(key: str, inc: dict = inc) -> str:
            r = inc[key]
            d, lo, hi = r["diff_vs_confidence"], r["lo"], r["hi"]
            return f"{fmt(d)} [{fmt(lo)}, {fmt(hi)}]"

        if "confidence+lat" in inc:
            assert row[6] == cell("confidence+lat")
        else:
            assert row[6] == NO_CONTRAST
        assert row[7] == cell("confidence+probe")


def test_steering_columns_count_runs_per_checkpoint(
    output: registry.TableOutput,
) -> None:
    for row, model in zip(output.rows, models("p9a"), strict=True):
        found, hits, selective = _steering(model)
        if not found:
            assert row[8:] == ["none", "none"]
        else:
            assert row[8] == f"{hits} of {found}"
            assert row[9] == f"{selective} of {found}"
    count = output.counts["distinguishable"]
    assert [count[r].total for r in ("gpt2", "base", "post")] == [3, 3, 6]
    assert [count[r].hits for r in ("gpt2", "base", "post")] == [3, 3, 3]
    count = output.counts["selective"]
    assert [count[r].hits for r in ("gpt2", "base", "post")] == [3, 3, 4]


def test_counts_of_the_direction_above_popularity_are_per_role(
    output: registry.TableOutput,
) -> None:
    count = output.counts["above_popularity"]
    assert [count[r].total for r in ("gpt2", "base", "post")] == [3, 3, 3]
    assert count["gpt2"].hits == 0


def test_the_caption_defines_popularity_and_the_missing_contrast(
    output: registry.TableOutput,
) -> None:
    text = output.claim + " " + output.detail
    assert "monthly Wikipedia page views" in text
    assert "[[cite:mallen2023when]]" in text
    assert "Hanley and McNeil" in text
    assert "paired bootstrap" in text
    assert "GPT-2 small has too few known questions" in text
    assert "3 random directions" in text
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


def test_the_largest_upper_bound_of_the_direction_gain_is_a_number(
    output: registry.TableOutput,
) -> None:
    upper = []
    for model in models("p9a"):
        for r in _table("p9c", model, "table_incremental.json"):
            if r["dataset"] == "popqa" and r["features"] == "confidence+lat":
                upper.append(r["hi"])
    assert output.numbers["gain_upper_max"] == max(upper)
    assert fmt(max(upper)) == "0.032"


def test_the_added_columns_name_the_confidence_not_one_probability(
    output: registry.TableOutput,
) -> None:
    text = output.claim + " " + output.detail
    assert "answer probability" not in text
    assert "confidence of the model, which stacks five output signals" in text


def test_the_table_prints_compact_because_it_overflows_at_the_normal_size(
    output: registry.TableOutput,
) -> None:
    assert output.compact_print is True
