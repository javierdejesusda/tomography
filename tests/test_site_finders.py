import math
from pathlib import Path
import runpy
from typing import Any

from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
import pytest

from tomography.site import params
from tomography.site import registry
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

_RESULTS = runpy.run_path(str(Path(__file__).with_name("_results.py")))
data = _RESULTS["data"]
models = _RESULTS["models"]
run = _RESULTS["run"]

RESULTS = Path(__file__).resolve().parents[1] / "results"
Z = 1.959963984540054
PAIRS = {"sentiment": 214, "truth": 59, "utility": 256}
PAIRING = {
    "sentiment": "label_contrastive",
    "truth": "label_contrastive",
    "utility": "matched",
}
TASKS = ("sentiment", "truth", "utility")
FINDERS = ("logistic", "pca_diff")
ROLES = ("gpt2", "base", "post")


def _role(model: str) -> str:
    if not model.startswith("qwen3"):
        return "gpt2"
    return "base" if model.endswith("-base") else "post"


def _report(table: list[dict[str, Any]], task: str, method: str) -> float:
    (found,) = [
        r
        for r in table
        if r["task"] == task
        and r["method"] == method
        and r["pairing"] == PAIRING[task]
        and r["preprocess"] == "center"
    ]
    return float(found["report"])


def _expected() -> dict[
    tuple[str, str, str], tuple[float, float, float, float]
]:
    out = {}
    for model in models("p4"):
        raw = data(run("p4", model), "table.json")
        table = (
            [dict(zip(raw["columns"], r, strict=True)) for r in raw["rows"]]
            if isinstance(raw, dict)
            else raw
        )
        for task in TASKS:
            n = PAIRS[task]
            rand = _report(table, task, "random")
            for finder in FINDERS:
                p = _report(table, task, finder)
                half = Z * math.sqrt(p * (1 - p) / n + rand * (1 - rand) / n)
                out[(model, task, finder)] = (
                    p - rand,
                    p - rand - half,
                    p - rand + half,
                    rand,
                )
    return out


EXPECTED = _expected()
MODELS = models("p4")


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure("finders", Manifest(RESULTS))


def _count(task: str, finder: str, test: Any) -> dict[str, tuple[int, int]]:
    result = {}
    for role in ROLES:
        cells = [
            EXPECTED[(m, task, finder)] for m in MODELS if _role(m) == role
        ]
        result[role] = (sum(bool(test(c)) for c in cells), len(cells))
    return result


def test_the_csv_holds_every_margin_with_its_interval(
    output: registry.FigureOutput,
) -> None:
    lines = output.csv.strip().splitlines()
    head = lines[0].split(",")
    assert head == [
        "task",
        "model",
        "role",
        "finder",
        "margin",
        "lo",
        "hi",
        "random",
        "n",
    ]
    assert len(lines) - 1 == len(MODELS) * len(TASKS) * len(FINDERS)
    for line in lines[1:]:
        rec = dict(zip(head, line.split(","), strict=True))
        margin, lo, hi, rand = EXPECTED[
            (rec["model"], rec["task"], rec["finder"])
        ]
        assert float(rec["margin"]) == pytest.approx(margin, abs=1e-9)
        assert float(rec["lo"]) == pytest.approx(lo, abs=1e-9)
        assert float(rec["hi"]) == pytest.approx(hi, abs=1e-9)
        assert float(rec["random"]) == pytest.approx(rand, abs=1e-9)
        assert int(rec["n"]) == PAIRS[rec["task"]]


def test_the_counts_follow_the_interval_and_the_point_estimate(
    output: registry.FigureOutput,
) -> None:
    cases = {
        "pca_sentiment": _count("sentiment", "pca_diff", lambda c: c[1] > 0),
        "logistic_sentiment": _count(
            "sentiment", "logistic", lambda c: c[1] > 0
        ),
        "pca_utility_point": _count("utility", "pca_diff", lambda c: c[0] > 0),
    }
    for name, roles in cases.items():
        for role, (hits, total) in roles.items():
            count = output.counts[name][role]
            assert (count.hits, count.total) == (hits, total)
    for role, (hits, _) in cases["pca_sentiment"].items():
        assert output.numbers[f"pca_sentiment_above_{role}"] == hits
    for role, (hits, _) in _count(
        "sentiment", "pca_diff", lambda c: c[0] > 0
    ).items():
        assert output.numbers[f"pca_sentiment_point_above_{role}"] == hits
    for role, (hits, _) in cases["logistic_sentiment"].items():
        assert output.numbers[f"logistic_sentiment_above_{role}"] == hits
    for role, (hits, _) in cases["pca_utility_point"].items():
        assert output.numbers[f"pca_utility_point_above_{role}"] == hits


def test_the_numbers_quoted_by_the_claim_match_the_recomputation(
    output: registry.FigureOutput,
) -> None:
    gpt2 = [m for m in MODELS if _role(m) == "gpt2"]
    truth = max(EXPECTED[(m, "truth", "pca_diff")][0] for m in gpt2)
    assert output.numbers["pca_truth_gpt2_max"] == pytest.approx(truth)
    utility = [EXPECTED[(m, "utility", "pca_diff")][0] for m in MODELS]
    assert output.numbers["pca_utility_min"] == pytest.approx(min(utility))
    assert output.numbers["pca_utility_max"] == pytest.approx(max(utility))
    randoms = [
        c[3] for (_, _, finder), c in EXPECTED.items() if finder == "pca_diff"
    ]
    assert output.numbers["random_max"] == pytest.approx(max(randoms))


def test_the_findings_of_the_plan_hold_at_the_current_results() -> None:
    sentiment = _count("sentiment", "pca_diff", lambda c: c[1] > 0)
    assert sentiment == {"gpt2": (0, 4), "base": (1, 3), "post": (0, 3)}
    point = _count("sentiment", "pca_diff", lambda c: c[0] > 0)
    assert point["gpt2"][0] == 0
    assert point["post"][0] == 0
    winner = [
        m
        for m in MODELS
        if _role(m) == "base" and EXPECTED[(m, "sentiment", "pca_diff")][1] > 0
    ]
    assert winner == ["qwen3-1.7b-base"]
    logistic = _count("sentiment", "logistic", lambda c: c[1] > 0)
    assert logistic == {"gpt2": (4, 4), "base": (3, 3), "post": (3, 3)}
    truth = [
        EXPECTED[(m, "truth", "pca_diff")][0]
        for m in MODELS
        if _role(m) == "gpt2"
    ]
    assert max(truth) <= 1e-12
    utility = [EXPECTED[(m, "utility", "pca_diff")][0] for m in MODELS]
    assert (round(min(utility), 3), round(max(utility), 3)) == (0.025, 0.154)


def test_the_caption_quotes_every_number_and_the_sample_sizes(
    output: registry.FigureOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    gpt2 = [
        EXPECTED[(m, "utility", "pca_diff")][0]
        for m in MODELS
        if _role(m) == "gpt2"
    ]
    assert f"{min(gpt2):.3f} to {max(gpt2):.3f}" in text
    assert "0.154" not in text
    assert "214 sentiment, 59 truth and 256 utility pairs" in text
    assert "[[fig:a_finders_gpt2]], [[fig:a_finders_qwen3]]" in text
    assert "[[table:finders_all]]" in text
    assert "random" in text
    assert "of 10 checkpoints" not in text


def test_the_caption_has_at_most_200_words(
    output: registry.FigureOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    assert len(text.split()) <= 200


def test_the_figure_has_three_panels_with_shared_rows_and_no_heat_map(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    assert len(fig.axes) == 3
    assert fig.get_size_inches()[0] == pytest.approx(paper.FULL_WIDTH)
    assert fig.get_size_inches()[1] == pytest.approx(3.0)
    assert not [i for i in fig.findobj(AxesImage)]
    checks.draw(fig)
    names = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert names == [paper.display_name(m) for m in MODELS]
    for ax in fig.axes[1:]:
        assert not ax.yaxis.get_tick_params()["labelleft"]
    assert [ax.get_title(loc="left") for ax in fig.axes] == [
        "Sentiment",
        "Truth",
        "Utility",
    ]


def test_the_guards_are_green(output: registry.FigureOutput) -> None:
    assert checks.guard_problems(output.figure) == []
    assert checks.problems(output.figure) == []


def test_the_caption_has_no_first_and_the_pair_counts_come_from_the_runs(
    output: registry.FigureOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    assert "first" not in text.lower()
    pairs = params.REPORT_PAIRS
    assert (
        f"{pairs['sentiment']} sentiment, {pairs['truth']} truth and "
        f"{pairs['utility']} utility pairs"
    ) in text


def test_a_rounded_zero_is_printed_without_a_sign(
    output: registry.FigureOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    assert "-0.000" not in text
    assert chr(0x2212) + "0.000" not in text


def test_the_pairing_is_read_from_the_run_summary_not_typed_in() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/tomography/site/figs/finders.py"
    ).read_text("utf-8")
    assert "label_contrastive" not in source
    assert '"matched"' not in source


def test_the_interval_counts_for_truth_and_utility_are_per_role(
    output: registry.FigureOutput,
) -> None:
    for task in ("truth", "utility"):
        expected = _count(task, "pca_diff", lambda c: c[1] > 0)
        for role, (hits, total) in expected.items():
            count = output.counts[f"pca_{task}"][role]
            assert (count.hits, count.total) == (hits, total)
            assert output.numbers[f"pca_{task}_above_{role}"] == hits
    truth = _count("truth", "pca_diff", lambda c: c[1] > 0)
    assert truth == {"gpt2": (0, 4), "base": (2, 3), "post": (2, 3)}
    utility = _count("utility", "pca_diff", lambda c: c[1] > 0)
    assert utility["base"] == (3, 3)
    assert utility["post"] == (3, 3)


def test_the_claim_describes_the_pairings_without_a_causal_title(
    output: registry.FigureOutput,
) -> None:
    text = output.claim
    assert "needs matched pairs" not in text
    assert "confounded" in text
    assert "by point estimate only" in text
    assert "for truth on 0 of 4 GPT-2, 2 of 3 Qwen3 base" in text


def _panels(output: registry.FigureOutput) -> list[Any]:
    return list(output.figure.axes)


def test_the_zero_reference_is_drawn_in_front_of_the_grey_band(
    output: registry.FigureOutput,
) -> None:
    for ax in _panels(output):
        (zero,) = [
            ln
            for ln in ax.get_lines()
            if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] == 0.0
        ]
        bands = [c.get_zorder() for c in ax.collections]
        assert bands
        assert zero.get_zorder() > max(bands)


def test_each_panel_counts_the_checkpoints_above_zero_per_role(
    output: registry.FigureOutput,
) -> None:
    for ax, task in zip(_panels(output), TASKS, strict=True):
        (note,) = [t for t in ax.texts if t.get_gid() == "counts"]
        total = {r: 0 for r in ROLES}
        for model in models("p4"):
            total[_role(model)] += 1
        hits = [int(output.numbers[f"pca_{task}_above_{r}"]) for r in ROLES]
        expected = " ".join(
            f"{h}/{total[r]}" for h, r in zip(hits, ROLES, strict=True)
        )
        assert note.get_text() == f"PCA above 0: {expected}"


def test_the_role_key_is_a_filled_and_an_open_circle_in_neutral_ink(
    output: registry.FigureOutput,
) -> None:
    (legend,) = output.figure.legends
    flat = legend.findobj(Line2D)
    assert "h" not in {str(h.get_marker()) for h in flat}
    circles = [
        h for h in flat if h.get_marker() == "o" and h.get_color() == paper.INK
    ]
    assert {h.get_markerfacecolor() == "white" for h in circles} == {
        True,
        False,
    }


def test_the_utility_gpt2_range_comes_from_the_gpt2_rows_only(
    output: registry.FigureOutput,
) -> None:
    gpt2 = [
        EXPECTED[(m, "utility", "pca_diff")][0]
        for m in MODELS
        if _role(m) == "gpt2"
    ]
    assert output.numbers["pca_utility_gpt2_min"] == pytest.approx(min(gpt2))
    assert output.numbers["pca_utility_gpt2_max"] == pytest.approx(max(gpt2))
    text = f"{output.claim} {output.detail}"
    assert f"by {min(gpt2):.3f} to {max(gpt2):.3f})" in text
    assert "over all ten" not in text


def test_the_utility_interval_count_is_stated_per_role_not_pooled(
    output: registry.FigureOutput,
) -> None:
    hits = {
        r: sum(
            EXPECTED[(m, "utility", "pca_diff")][1] > 0
            for m in MODELS
            if _role(m) == r
        )
        for r in ROLES
    }
    totals = {r: sum(_role(m) == r for m in MODELS) for r in ROLES}
    expected = (
        f"on {hits['base']} of {totals['base']} Qwen3 base and "
        f"{hits['post']} of {totals['post']} post-trained checkpoints "
        "by interval"
    )
    assert expected in output.claim
    assert "all 6 Qwen3" not in output.claim


def test_the_logistic_clause_is_counted_per_role_not_as_all_ten(
    output: registry.FigureOutput,
) -> None:
    assert "all 10 checkpoints" not in output.claim
    where = stats.per_role_text(output.counts["logistic_sentiment"])
    assert f"for sentiment on {where}" in output.claim
