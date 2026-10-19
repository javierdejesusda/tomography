import math
from pathlib import Path
import runpy
from typing import Any

from matplotlib.image import AxesImage
import pytest

from tomography.site import registry
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
    assert "Qwen3-1.7B-Base" in text
    assert "0.025 to 0.154" in text
    assert "214 sentiment, 59 truth and 256 utility pairs" in text
    assert "all 10 checkpoints" in text
    assert "Figures A2 and A3" in text
    assert "Table A2" in text
    assert "random" in text
    assert "of 10 checkpoints" not in text


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
