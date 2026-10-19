import math
from pathlib import Path
import runpy
from typing import Any

from matplotlib.image import AxesImage
import numpy as np
import pytest

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
MODELS = models("p1")
ROLES = ("gpt2", "base", "post")
DROPPED = "no_trailing_space"


def _role(model: str) -> str:
    if not model.startswith("qwen3"):
        return "gpt2"
    return "base" if model.endswith("-base") else "post"


def _rows(model: str) -> list[dict[str, Any]]:
    table = data(run("p1", model), "table.json")
    return [
        r for r in table if r["task"] == "sentiment" and r["variant"] != DROPPED
    ]


def _n_eval(model: str) -> int:
    summary = data(run("p1", model), "summary.json")
    return int(summary["n_eval_pairs"]["sentiment"]["report"])


def _span(model: str) -> float:
    values = [r["from_paper"] for r in _rows(model)]
    return max(values) - min(values)


def _noise(model: str) -> float:
    rows = _rows(model)
    p = sum(r["from_paper"] for r in rows) / len(rows)
    return stats.expected_range(len(rows)) * math.sqrt(
        p * (1 - p) / _n_eval(model)
    )


def _bare(model: str) -> dict[str, Any]:
    (row,) = [r for r in _rows(model) if r["variant"] == "bare"]
    return row


def _paper(model: str) -> dict[str, Any]:
    (row,) = [r for r in _rows(model) if r["variant"] == "paper"]
    return row


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure("wording", Manifest(RESULTS))


def test_the_csv_holds_every_wording_row_of_every_checkpoint(
    output: registry.FigureOutput,
) -> None:
    lines = output.csv.strip().splitlines()
    head = lines[0].split(",")
    assert head == [
        "model",
        "role",
        "variant",
        "kind",
        "family",
        "from_paper",
        "to_paper",
        "cosine_paper",
        "random",
        "stability",
    ]
    assert len(lines) - 1 == sum(len(_rows(m)) for m in MODELS)
    for line in lines[1:]:
        rec = dict(zip(head, line.split(","), strict=True))
        (row,) = [
            r for r in _rows(rec["model"]) if r["variant"] == rec["variant"]
        ]
        for column in ("from_paper", "to_paper", "cosine_paper", "random"):
            assert float(rec[column]) == pytest.approx(row[column], abs=1e-9)
        assert rec["role"] == _role(rec["model"])


def test_spans_are_per_role_and_noise_comes_from_the_expected_range(
    output: registry.FigureOutput,
) -> None:
    for role in ROLES:
        spans = [_span(m) for m in MODELS if _role(m) == role]
        if not spans:
            continue
        assert output.numbers[f"span_{role}_min"] == pytest.approx(min(spans))
        assert output.numbers[f"span_{role}_max"] == pytest.approx(max(spans))
    assert output.numbers["noise_max"] == pytest.approx(
        max(_noise(m) for m in MODELS)
    )
    assert output.numbers["n_checkpoints"] == len(MODELS)


def test_the_bare_wording_numbers_are_recomputed(
    output: registry.FigureOutput,
) -> None:
    cos = [_bare(m)["cosine_paper"] for m in MODELS]
    reads = [_bare(m)["to_paper"] for m in MODELS]
    random = [_paper(m)["random"] for m in MODELS]
    assert output.numbers["bare_cos_min"] == pytest.approx(min(cos))
    assert output.numbers["bare_cos_max"] == pytest.approx(max(cos))
    assert output.numbers["bare_to_paper_min"] == pytest.approx(min(reads))
    assert output.numbers["bare_to_paper_max"] == pytest.approx(max(reads))
    assert output.numbers["bare_random_min"] == pytest.approx(min(random))
    assert output.numbers["bare_random_max"] == pytest.approx(max(random))


def test_the_plan_values_hold_for_the_eight_checkpoints_it_checked() -> None:
    cos = {
        "gpt2": 0.179,
        "gpt2-medium": 0.279,
        "gpt2-large": 0.274,
        "gpt2-xl": 0.276,
        "qwen3-0.6b-base": 0.110,
        "qwen3-0.6b": 0.075,
        "qwen3-1.7b-base": 0.137,
        "qwen3-1.7b": 0.121,
    }
    for model in cos:
        assert round(_bare(model)["cosine_paper"], 3) == pytest.approx(
            cos[model]
        )
    assert stats.expected_range(16) == pytest.approx(3.532, abs=1e-3)


def test_the_caption_quotes_the_spans_the_noise_and_the_bare_numbers(
    output: registry.FigureOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    n = output.numbers
    for role in ROLES:
        if f"span_{role}_min" in n:
            assert f"{n[f'span_{role}_min']:.3f}" in text
            assert f"{n[f'span_{role}_max']:.3f}" in text
    for name in (
        "noise_max",
        "bare_cos_min",
        "bare_cos_max",
        "bare_to_paper_min",
        "bare_to_paper_max",
        "bare_random_min",
        "bare_random_max",
    ):
        assert f"{n[name]:.3f}" in text, name
    assert "Figure A4" in text
    assert "Table A6" in text


def test_the_guards_are_green(output: registry.FigureOutput) -> None:
    assert checks.guard_problems(output.figure) == []
    assert checks.problems(output.figure) == []


def test_the_bare_claim_is_counted_per_role_without_a_pooled_count(
    output: registry.FigureOutput,
) -> None:
    for role in ROLES:
        group = [m for m in MODELS if _role(m) == role]
        hits = sum(_bare(m)["to_paper"] > _paper(m)["random"] for m in group)
        assert output.numbers[f"bare_above_{role}"] == hits
        assert output.numbers[f"bare_n_{role}"] == len(group)
        count = output.counts["bare_above"][role]
        assert (count.hits, count.total) == (hits, len(group))
    counts = output.counts["bare_above"]
    assert stats.per_role_text(counts) in output.claim
    assert f"of {len(MODELS)} checkpoints" not in output.claim
    for model in MODELS:
        if _bare(model)["to_paper"] <= _paper(model)["random"]:
            assert paper.display_name(model) in output.claim
            assert f"{_bare(model)['to_paper']:.3f}" in output.claim
            assert f"{_paper(model)['random']:.3f}" in output.claim


def test_the_dropped_wording_is_absent_and_the_noise_uses_the_report_pairs(
    output: registry.FigureOutput,
) -> None:
    assert DROPPED not in output.csv
    assert output.numbers["noise_max"] == pytest.approx(
        max(_noise(m) for m in MODELS)
    )
    assert {_n_eval(m) for m in MODELS} == {214}
    assert "by point estimate" in output.claim


def test_the_opening_sentence_agrees_with_the_per_role_counts(
    output: registry.FigureOutput,
) -> None:
    first = output.claim.split(". ")[0]
    assert "orthogonal" not in output.claim
    assert "same concept" not in first
    assert "large angle" in first
    assert "mostly" in first


def test_the_span_above_noise_count_follows_the_right_hand_columns(
    output: registry.FigureOutput,
) -> None:
    flags = {m: _span(m) > _noise(m) for m in MODELS}
    for role in ROLES:
        mine = [f for m, f in flags.items() if _role(m) == role]
        count = output.counts["span_above_noise"][role]
        assert (count.hits, count.total) == (sum(mine), len(mine))
        assert output.numbers[f"span_above_noise_{role}"] == sum(mine)
        assert output.numbers[f"span_above_noise_n_{role}"] == len(mine)
    assert [output.counts["span_above_noise"][r].hits for r in ROLES] == [
        2,
        3,
        3,
    ]


def test_the_claim_states_the_span_count_and_does_not_overstate_gpt2(
    output: registry.FigureOutput,
) -> None:
    first = output.claim.split(". ")[0]
    counts = output.counts["span_above_noise"]
    assert "beyond test-pair noise on most checkpoints" not in first
    assert stats.per_role_text(counts) in output.claim
    assert "point comparison" in output.claim


FAMILIES = ("frame", "wording", "concept")


def _gid_lines(ax: Any, prefix: str) -> dict[str, Any]:
    return {
        ln.get_gid(): ln
        for ln in ax.get_lines()
        if (ln.get_gid() or "").startswith(prefix)
    }


def _strip(output: registry.FigureOutput) -> Any:
    (ax,) = [a for a in output.figure.axes if a.get_gid() == "wordings"]
    return ax


def _facets(output: registry.FigureOutput) -> dict[str, Any]:
    return {
        a.get_gid().removeprefix("cosine_"): a
        for a in output.figure.axes
        if (a.get_gid() or "").startswith("cosine_")
    }


def _rank(model: str) -> int:
    values = sorted((r["from_paper"] for r in _rows(model)), reverse=True)
    (paper_row,) = [r for r in _rows(model) if r["variant"] == "paper"]
    return values.index(paper_row["from_paper"]) + 1


def test_the_layout_is_one_strip_over_three_role_facets_and_no_heat_map(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    assert not fig.findobj(AxesImage)
    width, height = fig.get_size_inches()
    assert width == pytest.approx(paper.FULL_WIDTH)
    assert height == pytest.approx(6.6)
    assert sorted(_facets(output)) == sorted(ROLES)
    assert len(_facets(output)) == 3
    assert _strip(output).get_xlim() == pytest.approx((0.45, 1.0))


def test_every_row_has_fifteen_identical_circles_coloured_by_three_families(
    output: registry.FigureOutput,
) -> None:
    ax = _strip(output)
    colours: dict[str, set[Any]] = {f: set() for f in FAMILIES}
    for model in MODELS:
        rows = _rows(model)
        drawn = 0
        for family in FAMILIES:
            line = _gid_lines(ax, f"words:{model}:{family}")[
                f"words:{model}:{family}"
            ]
            want = sorted(
                r["from_paper"]
                for r in rows
                if r["variant"] != "paper"
                and {
                    "frame": "frame",
                    "wording": "wording",
                    "concept": "concept",
                }[r["family"]]
                == family
            )
            assert sorted(line.get_xdata()) == pytest.approx(want)
            assert line.get_marker() == "o"
            assert line.get_markersize() == pytest.approx(3.0)
            drawn += len(want)
            colours[family].add(line.get_markerfacecolor())
        assert drawn == 15
    assert all(len(c) == 1 for c in colours.values())
    assert len({next(iter(c)) for c in colours.values()}) == 3


def test_the_paper_wording_is_a_diamond_with_its_rank_beside_it(
    output: registry.FigureOutput,
) -> None:
    ax = _strip(output)
    ranks = {}
    for model in MODELS:
        (row,) = [r for r in _rows(model) if r["variant"] == "paper"]
        line = _gid_lines(ax, f"paper:{model}")[f"paper:{model}"]
        assert line.get_marker() == "D"
        assert line.get_xdata()[0] == pytest.approx(row["from_paper"])
        assert line.get_markersize() > 3.0
        (text,) = [t for t in ax.texts if t.get_gid() == f"rank:{model}"]
        assert text.get_text() == str(_rank(model))
        ranks[model] = _rank(model)
    assert output.numbers["paper_rank_min"] == min(ranks.values())
    assert output.numbers["paper_rank_max"] == max(ranks.values())
    assert 1 <= min(ranks.values()) <= max(ranks.values()) <= 16


def test_the_median_tick_the_noise_band_and_the_random_tick_are_drawn(
    output: registry.FigureOutput,
) -> None:
    ax = _strip(output)
    for model in MODELS:
        rows = _rows(model)
        values = [r["from_paper"] for r in rows]
        (row,) = [r for r in rows if r["variant"] == "paper"]
        median = _gid_lines(ax, f"median:{model}")[f"median:{model}"]
        assert median.get_xdata()[0] == pytest.approx(float(np.median(values)))
        tick = _gid_lines(ax, f"random:{model}")[f"random:{model}"]
        assert tick.get_xdata()[0] == pytest.approx(row["random"])
        (band,) = [p for p in ax.patches if p.get_gid() == f"noise:{model}"]
        assert band.get_width() == pytest.approx(_noise(model))
        assert band.get_x() + band.get_width() / 2 == pytest.approx(
            row["from_paper"]
        )


def test_the_right_column_marks_the_checkpoints_whose_span_exceeds_noise(
    output: registry.FigureOutput,
) -> None:
    ax = _strip(output)
    marked = [t for t in ax.texts if t.get_gid() == "above_noise"]
    want = sum(_span(m) > _noise(m) for m in MODELS)
    assert len(marked) == want == 8
    spans = [t for t in ax.texts if t.get_gid() == "span"]
    assert len(spans) == len(MODELS)
    assert sorted(t.get_text() for t in spans) == sorted(
        f"{_span(m):.3f}" for m in MODELS
    )


def test_the_rows_keep_the_roles_apart_in_two_headed_blocks(
    output: registry.FigureOutput,
) -> None:
    ax = _strip(output)
    heads = [t.get_text() for t in ax.texts if t.get_gid() == "block"]
    assert heads == ["GPT-2", "Qwen3, base then post-trained"]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == len(MODELS)
    assert len(set(labels)) == len(MODELS)


def test_each_facet_holds_only_its_role_with_the_paper_direction_at_one(
    output: registry.FigureOutput,
) -> None:
    for role, ax in _facets(output).items():
        mine = [m for m in MODELS if _role(m) == role]
        drawn = _gid_lines(ax, "cosine:")
        assert sorted(drawn) == sorted(f"cosine:{m}" for m in mine)
        for model in mine:
            rows = [r for r in _rows(model) if r["variant"] != "paper"]
            line = drawn[f"cosine:{model}"]
            assert sorted(line.get_xdata()) == pytest.approx(
                sorted(r["cosine_paper"] for r in rows)
            )
            assert sorted(line.get_ydata()) == pytest.approx(
                sorted(r["to_paper"] for r in rows)
            )
            (diamond,) = [
                ln
                for ln in ax.get_lines()
                if ln.get_gid() == f"paperdir:{model}"
            ]
            assert diamond.get_xdata()[0] == pytest.approx(1.0)
            assert diamond.get_marker() == "D"
        assert ax.get_xlim() == pytest.approx((0.0, 1.05))
        assert ax.get_ylim() == pytest.approx((0.45, 1.0))


def test_the_caption_states_the_band_the_conservative_noise_and_the_rank(
    output: registry.FigureOutput,
) -> None:
    text = output.detail
    n = output.numbers
    assert "15 circles" in text
    assert "wordings share the same test pairs" in text
    assert "conservative" in text
    assert "excludes fitting noise" in text
    assert "diamond" in text
    assert "median" in text
    assert (
        f"{int(n['paper_rank_min'])} to {int(n['paper_rank_max'])} of the 16"
        in output.claim
    )
