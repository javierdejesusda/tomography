from pathlib import Path
import re
import runpy

import pytest

from tomography.site import params
from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import paper

_HERE = Path(__file__).parent
_R = runpy.run_path(str(_HERE / "_results.py"))
_M = runpy.run_path(str(_HERE / "_site_math.py"))
ROOT, data, models, run = _R["ROOT"], _R["data"], _R["models"], _R["run"]
diff, excludes_zero = _M["diff"], _M["excludes_zero"]
fmt, groups = _M["fmt"], _M["groups"]

TASKS = ("sentiment", "truth", "utility")
METHODS = ("random", "pca_diff", "logistic")
DASHES = (chr(0x2014), chr(0x2013))


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table("finders", Manifest(ROOT))


def _report(model: str, task: str, method: str) -> float:
    path = run("p4", model)
    pairing = data(path, "summary.json")["reference"][task]["pairing"]
    hits = [
        r
        for r in data(path, "table.json")
        if (r["task"], r["method"], r["pairing"], r["preprocess"])
        == (task, method, pairing, "center")
    ]
    assert len(hits) == 1
    return float(hits[0]["report"])


def test_columns_are_three_per_task_under_header_spans(
    output: registry.TableOutput,
) -> None:
    assert output.columns == ["Model"] + [
        "Random",
        "PCA of paired differences (paper)",
        "Logistic regression",
    ] * 3
    assert output.align == "l" + "r" * 9
    assert [g[0] for g in output.groups] == ["Sentiment", "Truth", "Utility"]
    assert [g[1] for g in output.groups] == [3, 3, 3]
    assert not any(g[2] for g in output.groups)


def test_one_row_per_checkpoint_with_a_rule_between_gpt2_and_qwen3(
    output: registry.TableOutput,
) -> None:
    names = models("p4")
    assert [r[0] for r in output.rows] == [paper.display_name(m) for m in names]
    assert output.row_groups == groups(names)


def test_every_cell_is_the_random_accuracy_or_the_margin_with_a_star(
    output: registry.TableOutput,
) -> None:
    names = models("p4")
    stars = 0
    for row, model in zip(output.rows, names, strict=True):
        for i, task in enumerate(TASKS):
            n = params.REPORT_PAIRS[task]
            rand = _report(model, task, "random")
            cells = row[1 + 3 * i : 4 + 3 * i]
            assert cells[0] == fmt(rand)
            for method, cell in zip(METHODS[1:], cells[1:], strict=True):
                d, lo, hi = diff(_report(model, task, method), rand, n)
                star = "*" if excludes_zero(lo, hi) else ""
                stars += bool(star)
                assert cell == fmt(d) + star
    assert output.numbers["starred"] == stars


def test_the_caption_names_the_test_sizes_and_points_to_the_appendix(
    output: registry.TableOutput,
) -> None:
    text = output.claim + " " + output.detail
    for n in (214, 59, 256):
        assert str(n) in text
    assert "random direction" in text
    assert "asterisk" in text
    assert "matched scenarios" in text
    assert "Table A2" in text or "[[table:finders_all]]" in text


def test_no_dash_run_code_or_underscore_in_the_table(
    output: registry.TableOutput,
) -> None:
    texts = [output.claim, output.detail, *output.columns]
    texts += [c for r in output.rows for c in r]
    texts += [g[0] for g in output.groups]
    texts = [re.sub(r"\[\[.*?\]\]", "", t) for t in texts]
    assert not any(ch in t for t in texts for ch in DASHES)
    assert not any(re.search(r"\b[RPMCA][0-9]+[a-d]?\b", t) for t in texts)
    assert not any("_" in t for t in texts)
