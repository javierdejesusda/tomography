import json
import math
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
DASHES = (chr(0x2014), chr(0x2013))


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table("steering", Manifest(ROOT))


def _rows(model: str) -> list[dict]:
    table = data(run("c1", model), "table.json")
    return [dict(zip(table["columns"], r, strict=True)) for r in table["rows"]]


def _mean(rows: list[dict], arm: str, coeff: float | None, key: str) -> float:
    hit = [
        r
        for r in rows
        if r["arm"] == arm and (coeff is None or r["coeff"] == coeff)
    ]
    assert hit
    return sum(r[key] for r in hit) / len(hit)


def _expected(model: str) -> list[str]:
    rows = _rows(model)
    score = [_mean(rows, "linear", c, "score") for c in (-0.1, 0.1)]
    zero = _mean(rows, "none", 0.0, "score")
    out = [fmt(score[0]), fmt(zero), fmt(score[1])]
    out += [
        fmt(_mean(rows, "prompting-1", None, "score")),
        fmt(_mean(rows, "prompting+1", None, "score")),
    ]
    p0 = _mean(rows, "none", 0.0, "perplexity")
    out += [
        fmt(_mean(rows, "linear", c, "perplexity") / p0, 2) for c in (-0.1, 0.1)
    ]
    f0 = _mean(rows, "none", 0.0, "fluency")
    out += [
        fmt(math.exp(f0 - _mean(rows, "linear", c, "fluency")), 2)
        for c in (-0.1, 0.1)
    ]
    out += [fmt(_mean(rows, "linear", c, "distinct_1")) for c in (-0.1, 0.1)]
    return out


def test_columns_and_header_spans(output: registry.TableOutput) -> None:
    assert output.columns == [
        "Model",
        f"{MINUS}0.1",
        "0",
        "+0.1",
        "Negative",
        "Positive",
        f"{MINUS}0.1",
        "+0.1",
        f"{MINUS}0.1",
        "+0.1",
        f"{MINUS}0.1",
        "+0.1",
    ]
    assert [(g[0], g[1]) for g in output.groups] == [
        ("Judged sentiment at relative coefficient", 3),
        ("Judged sentiment under an instruction", 2),
        ("Neutral-text perplexity ratio", 2),
        ("Completion perplexity ratio", 2),
        ("Distinct-1", 2),
    ]
    assert output.align == "l" + "r" * 11


def test_ten_rows_split_between_gpt2_and_qwen3(
    output: registry.TableOutput,
) -> None:
    names = models("c1")
    assert len(names) == 10
    assert [r[0] for r in output.rows] == [paper.display_name(m) for m in names]
    assert output.row_groups == groups(names)


def test_all_110_cells_match_a_recomputation(
    output: registry.TableOutput,
) -> None:
    for row, model in zip(output.rows, models("c1"), strict=True):
        assert row[1:] == _expected(model), model
    assert sum(len(r) - 1 for r in output.rows) == 110


def test_the_caption_defines_each_column_and_the_fluency_rule(
    output: registry.TableOutput,
) -> None:
    text = output.claim + " " + output.detail
    assert "SiEBERT" in text
    assert "200 held-out WikiText-103 passages" in text
    assert "at most 1.5" in text
    assert "at least 0.15" in text
    assert "distinct-1" in text
    assert "[[fig:steering]]" in text
    assert "DistilBERT" in text
    assert not any(ch in text for ch in DASHES)


def test_the_neutral_passage_count_comes_from_the_run_config(
    output: registry.TableOutput,
) -> None:
    config = run("c1", "gpt2") / "config.json"
    n_neutral = json.loads(config.read_text("utf-8"))["n_neutral"]
    assert f"{n_neutral} held-out WikiText-103 passages" in output.detail
    assert output.numbers["n_neutral"] == n_neutral


def test_no_run_code_underscore_or_backtick(
    output: registry.TableOutput,
) -> None:
    texts = [output.claim, output.detail, *output.columns]
    texts += [c for r in output.rows for c in r]
    texts += [g[0] for g in output.groups]
    texts = [re.sub(r"\[\[.*?\]\]", "", t) for t in texts]
    assert not any("_" in t or "`" in t for t in texts)
    assert not any(re.search(r"\b[RPMCA][0-9]+[a-d]?\b", t) for t in texts)
