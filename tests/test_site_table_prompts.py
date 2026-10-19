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
fmt, groups, wilson, diff = (
    _M["fmt"],
    _M["groups"],
    _M["wilson"],
    _M["diff"],
)
DASHES = (chr(0x2014), chr(0x2013))
N = 214
PM = chr(0x00B1)


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table("prompts", Manifest(ROOT))


def _accuracy(model: str) -> dict[str, float]:
    rows = data(run("p2", model), "table.json")
    return {
        r["condition"]: r["accuracy"]
        for r in rows
        if r["task"] == "sentiment" and r["position"] == "last"
    }


def test_columns_and_groups(output: registry.TableOutput) -> None:
    assert output.columns == [
        "Model",
        "Original",
        "Random decoy",
        "Against decoy",
    ]
    assert output.align == "lrrr"
    assert output.row_groups == groups(models("p2"))
    assert len(output.rows) == 10


def test_cells_are_accuracy_with_the_wilson_half_width(
    output: registry.TableOutput,
) -> None:
    for row, model in zip(output.rows, models("p2"), strict=True):
        acc = _accuracy(model)
        assert row[0] == paper.display_name(model)
        for col, cond in enumerate(("original", "random", "against"), 1):
            lo, hi = wilson(acc[cond], N)
            assert row[col] == f"{fmt(acc[cond])} {PM} {fmt((hi - lo) / 2)}"


def test_count_uses_the_conservative_difference_interval(
    output: registry.TableOutput,
) -> None:
    hits = {"gpt2": 0, "base": 0, "post": 0}
    totals = {"gpt2": 0, "base": 0, "post": 0}
    for model in models("p2"):
        role = (
            "gpt2"
            if model.startswith("gpt2")
            else "base"
            if model.endswith("base")
            else "post"
        )
        acc = _accuracy(model)
        _, _, hi = diff(acc["against"], acc["original"], N)
        hits[role] += hi < 0
        totals[role] += 1
    got = output.counts["against_lowers"]
    for role in hits:
        assert (got[role].hits, got[role].total) == (hits[role], totals[role])


def test_caption_names_both_decoys_and_the_training_prompts(
    output: registry.TableOutput,
) -> None:
    text = output.claim + " " + output.detail
    assert "drawn at random" in text
    assert "always contradicting the label" in text
    assert "trained on the original prompts" in text
    assert "214" in text
    assert not any(ch in text for ch in DASHES)


def test_no_underscore_or_run_code(output: registry.TableOutput) -> None:
    texts = [output.claim, output.detail, *output.columns]
    texts += [c for r in output.rows for c in r]
    assert not any("_" in t or "`" in t for t in texts)
    assert not any(re.search(r"\b[RPMCA][0-9]+[a-d]?\b", t) for t in texts)


def test_caption_says_what_the_contrast_is_and_where_the_baseline_is(
    output: registry.TableOutput,
) -> None:
    text = output.detail
    assert "within-run contrast across prompt variants" in text
    assert "not the held-out split of [[table:reading]]" in text
    assert "no random direction is stored for these runs" in text
    assert "[[num:table:prompts.chance]]" in text
    assert output.numbers["chance"] == 0.5


def test_caption_explains_why_table_a3_random_values_do_not_apply(
    output: registry.TableOutput,
) -> None:
    text = output.detail
    assert "[[table:reading_full]]" in text
    assert "other runs with their own training sample and layer" in text
    assert "not a baseline for these accuracies" in text
