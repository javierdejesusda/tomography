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
diff, excludes_zero, fmt = _M["diff"], _M["excludes_zero"], _M["fmt"]
PLUS_MINUS = chr(0x00B1)
DASHES = (chr(0x2014), chr(0x2013))
METHODS = ("pca_diff", "pca_raw", "mean_diff", "logistic", "kmeans", "random")
TASKS = ("sentiment", "truth", "utility")


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table("finders_all", Manifest(ROOT))


def _report(model: str, task: str, method: str) -> float:
    path = run("p4", model)
    pairing = data(path, "summary.json")["reference"][task]["pairing"]
    hit = [
        r
        for r in data(path, "table.json")
        if (r["task"], r["method"], r["pairing"], r["preprocess"])
        == (task, method, pairing, "center")
    ]
    assert len(hit) == 1
    return float(hit[0]["report"])


def _win(model: str, method: str) -> list:
    table = data(run("m1", model), "table.json")
    hit = [
        r
        for r in table["rows"]
        if r[0] == method and r[1] == "manipulation" and r[2] == "0.3"
    ]
    assert len(hit) == 1
    return hit[0]


def test_columns_are_the_checkpoints_and_rows_are_finders_in_groups(
    output: registry.TableOutput,
) -> None:
    names = models("p4")
    assert output.columns == ["Group", "Finder"] + [
        paper.display_name(m) for m in names
    ]
    assert output.align == "ll" + "r" * len(names)
    assert len(output.rows) == 24
    assert output.row_groups == (0, 6, 12, 18)
    labels = [output.rows[i][0] for i in output.row_groups]
    assert labels[:3] == ["Sentiment", "Truth", "Utility"]
    assert labels[3].startswith("Win rate at 0.3")
    blank = [r[0] for i, r in enumerate(output.rows) if i not in (0, 6, 12, 18)]
    assert set(blank) == {""}


def test_reading_cells_are_margins_with_stars_and_random_is_its_accuracy(
    output: registry.TableOutput,
) -> None:
    names = models("p4")
    for t, task in enumerate(TASKS):
        n = params.REPORT_PAIRS[task]
        for k, method in enumerate(METHODS):
            row = output.rows[6 * t + k]
            assert row[1] == params.display(method)
            for col, model in enumerate(names):
                rand = _report(model, task, "random")
                if method == "random":
                    assert row[2 + col] == fmt(rand)
                    continue
                d, lo, hi = diff(_report(model, task, method), rand, n)
                star = "*" if excludes_zero(lo, hi) else ""
                assert row[2 + col] == fmt(d) + star, (task, method, model)


def test_steering_cells_are_win_rates_with_the_stored_wilson_half_width(
    output: registry.TableOutput,
) -> None:
    names = models("p4")
    for k, method in enumerate(METHODS):
        row = output.rows[18 + k]
        assert row[1] == params.display(method)
        for col, model in enumerate(names):
            _, _, _, value, lo, hi, _ = _win(model, method)
            want = f"{fmt(value)} {PLUS_MINUS} {fmt((hi - lo) / 2)}"
            assert row[2 + col] == want, (method, model)


def test_the_caption_names_the_win_rate_and_the_random_reference(
    output: registry.TableOutput,
) -> None:
    text = output.claim + " " + output.detail
    assert "accuracy minus the random direction" in text
    assert "+0.3" in text
    assert "decisive prompt pairs" in text
    assert "Three random directions of the same norm" in text
    assert "0.5 is not" in text
    assert (
        "Mean difference is the mass-mean probe [[cite:marks2024geometry]] "
        "without whitening and logistic regression the linear probe "
        "[[cite:alain2016understanding]]; both are published baselines for "
        "the finder of the paper. The random direction is the control."
    ) in text
    assert not any(ch in text for ch in DASHES)


def test_no_underscore_backtick_or_run_code(
    output: registry.TableOutput,
) -> None:
    texts = [output.claim, output.detail, *output.columns]
    texts += [c for r in output.rows for c in r]
    texts = [re.sub(r"\[\[.*?\]\]", "", t) for t in texts]
    assert not any("_" in t or "`" in t for t in texts)
    assert not any(re.search(r"\b[RPMCA][0-9]+[a-d]?\b", t) for t in texts)


def test_the_table_prints_compact_because_it_overflows_at_the_normal_size(
    output: registry.TableOutput,
) -> None:
    assert output.compact_print is True
