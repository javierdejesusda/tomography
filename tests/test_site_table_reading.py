from pathlib import Path
import re
import runpy

import numpy as np
import pytest

from tomography.site import registry
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
diff, excludes_zero = _M["diff"], _M["excludes_zero"]
fmt, groups = _M["fmt"], _M["groups"]

DASHES = (chr(0x2014), chr(0x2013))
CONCEPTS = ("sentiment", "utility", "morality", "truth", "honesty")
SENTIMENT_PAIRS = 214


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table("reading", Manifest(ROOT))


def _random_at(path: Path, stem: str, layer: int) -> float:
    series = data(path, f"{stem}.json")["series"]
    mean = npz(path, f"{stem}.npz")["mean"]
    return float(mean[series.index("random")][layer - 1])


def _score(concept: str, model: str) -> tuple[float, float, int]:
    if concept == "sentiment":
        path = run("p3", model)
        meta = data(path, "sizes_sentiment.json")
        arrays = npz(path, "sizes_sentiment.npz")
        k = int(np.flatnonzero(arrays["sizes"] == 64)[0])
        col = int(np.flatnonzero(arrays["layers"] == meta["layer"])[0])
        acc = arrays["accuracy"]
        return (
            float(acc[k, 1, 0, :, col].mean()),
            float(acc[k, 1, 1, :, col].mean()),
            SENTIMENT_PAIRS,
        )
    exp = {"utility": "r2", "morality": "r3", "truth": "r7", "honesty": "r5"}[
        concept
    ]
    path = run(exp, model)
    summary = data(path, "summary.json")
    if exp == "r7":
        layer = summary["val_layer"]
        reader = summary["sets"]["test"]["accuracy_at_val_layer"]
        n = summary["sets"]["test"]["n_eval_pairs"]
    elif exp == "r5":
        layer = summary["best_layer"]
        reader = summary["test_accuracy"]
        n = summary["n_eval_pairs"]["transfer"]["test"]
    else:
        layer = summary["best_layer"]
        reader = summary["test_accuracy"]
        n = summary["n_eval_pairs"]["test"]
    stem = "curves_transfer" if exp == "r5" else "curves"
    return float(reader), _random_at(path, stem, layer), int(n)


def test_columns_are_a_reader_and_random_pair_per_concept(
    output: registry.TableOutput,
) -> None:
    assert output.columns == ["Model"] + ["Reader", "Random"] * 5
    assert output.align == "l" + "r" * 10
    assert [g[0] for g in output.groups] == [
        "Sentiment (held out)",
        "Utility",
        "Morality",
        "Truth",
        "Honesty (transfer)",
    ]
    assert [g[1] for g in output.groups] == [2] * 5
    assert not any(g[2] for g in output.groups)


def test_one_row_per_checkpoint_split_between_gpt2_and_qwen3(
    output: registry.TableOutput,
) -> None:
    names = models("p3")
    assert len(names) == 10
    assert [r[0] for r in output.rows] == [paper.display_name(m) for m in names]
    assert output.row_groups == groups(names)


def test_every_cell_matches_a_recomputation_and_the_stars_agree(
    output: registry.TableOutput,
) -> None:
    names = models("p3")
    stars = {c: 0 for c in CONCEPTS}
    for row, model in zip(output.rows, names, strict=True):
        for i, concept in enumerate(CONCEPTS):
            reader, random, n = _score(concept, model)
            lo = diff(reader, random, n)[1]
            star = "*" if lo > 0 else ""
            stars[concept] += bool(star)
            assert row[1 + 2 * i] == fmt(reader) + star, (model, concept)
            assert row[2 + 2 * i] == fmt(random), (model, concept)
    for concept in CONCEPTS:
        assert output.numbers[f"above_{concept}"] == stars[concept]


def test_sentiment_is_the_held_out_value_not_the_r1_layer_value(
    output: registry.TableOutput,
) -> None:
    differs = 0
    for row, model in zip(output.rows, models("p3"), strict=True):
        best = data(run("r1", model), "summary.json")["best_accuracy"]
        differs += row[1].rstrip("*") != fmt(best)
    assert differs >= 1


def test_no_cell_reads_not_available_and_no_header_is_a_run_code(
    output: registry.TableOutput,
) -> None:
    cells = [c for r in output.rows for c in r]
    assert "n/a" not in cells
    heads = [*output.columns, *(g[0] for g in output.groups)]
    assert not any(re.search(r"\b[RPMCA][0-9]+[a-d]?\b", h) for h in heads)
    assert not any("LAT" in h for h in heads)


def test_the_counts_are_per_role(output: registry.TableOutput) -> None:
    for concept in CONCEPTS:
        count = output.counts[f"above_{concept}"]
        total = sum(c.total for c in count.values())
        hits = sum(c.hits for c in count.values())
        assert total == 10
        assert hits == output.numbers[f"above_{concept}"]
        assert (count["gpt2"].total, count["base"].total) == (4, 3)


def test_the_caption_names_every_test_size_and_the_honesty_direction(
    output: registry.TableOutput,
) -> None:
    text = output.claim + " " + output.detail
    for concept in ("utility", "morality"):
        n = _score(concept, "gpt2")[2]
        assert f"{n:,}" in text
    assert "214 sentiment pairs" in text
    assert "59 truth" in text or "59 plain true and false statements" in text
    assert "fitted on the paper's honesty persona prompts" in text
    assert "no refit" in text
    assert "[[table:reading_full]]" in text
    assert not any(ch in text for ch in DASHES)
