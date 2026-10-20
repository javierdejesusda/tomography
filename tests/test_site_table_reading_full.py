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
_S = runpy.run_path(str(_HERE / "_site_scores.py"))
ROOT, models = _R["ROOT"], _R["models"]
diff, fmt = _M["diff"], _M["fmt"]
score, n_layers = _S["score"], _S["n_layers"]
PLUS_MINUS = chr(0x00B1)
DASHES = (chr(0x2014), chr(0x2013))
BLOCKS = (
    (
        "sentiment_r1",
        "Sentiment, layer chosen on the scored pairs (optimistic)",
    ),
    ("sentiment", "Sentiment, held out"),
    ("utility", "Utility"),
    ("morality", "Morality"),
    ("truth", "Truth"),
    ("template", "Honesty in template"),
    ("honesty", "Honesty transfer"),
)


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table("reading_full", Manifest(ROOT))


def test_columns_and_one_group_per_concept(
    output: registry.TableOutput,
) -> None:
    assert output.columns == [
        "Concept",
        "Model",
        "Layer",
        "Relative depth",
        "Reader",
        "Random",
        "Margin",
        "Test pairs",
    ]
    assert output.align == "llrrrrrr"
    assert output.row_groups == tuple(10 * i for i in range(7))
    assert len(output.rows) == 70
    for i, (_, label) in enumerate(BLOCKS):
        assert output.rows[10 * i][0] == label
        assert all(output.rows[10 * i + j][0] == "" for j in range(1, 10))


def test_every_row_matches_a_recomputation(
    output: registry.TableOutput,
) -> None:
    names = models("p3")
    assert len(names) == 10
    for i, (concept, _) in enumerate(BLOCKS):
        for j, model in enumerate(names):
            row = output.rows[10 * i + j]
            layer, reader, random, n = score(concept, model)
            d, lo, hi = diff(reader, random, n)
            half = (hi - lo) / 2
            assert row[1] == paper.display_name(model)
            assert row[2] == str(layer)
            assert row[3] == fmt(layer / n_layers(model))
            assert row[4] == fmt(reader)
            assert row[5] == fmt(random)
            assert row[6] == f"{fmt(d)} {PLUS_MINUS} {fmt(half)}"
            assert row[7] == f"{n:,}"


def test_the_caption_flags_the_optimistic_block_and_the_interval_unit(
    output: registry.TableOutput,
) -> None:
    text = output.claim + " " + output.detail
    assert "optimistic" in text
    assert "256 SST-2 validation pairs" in text
    assert "214 pairs" in text
    assert "resample the readers only" in text
    assert not any(ch in text for ch in DASHES)


def test_no_underscore_or_run_code_outside_the_optimistic_label(
    output: registry.TableOutput,
) -> None:
    texts = [output.claim, output.detail, *output.columns]
    texts += [c for r in output.rows for c in r]
    assert not any("_" in t or "`" in t for t in texts)
    assert not any(re.search(r"\b[RPMCA][0-9]+[a-d]?\b", t) for t in texts)
