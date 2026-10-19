import math
from pathlib import Path
import runpy
from typing import Any

from matplotlib.image import AxesImage
from matplotlib.text import Text
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
    assert "[[fig:a_wording]]" in text
    assert "[[table:wordings]]" in text


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
LANES = {
    "frame": "frame",
    "wording": "paraphrase, scaffold",
    "concept": "concept swap",
}
COLUMN_HEADERS = ["paper", "noise", "span", "rank of 16", "above noise"]


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
    return values.index(_paper(model)["from_paper"]) + 1


def _table_name(row: dict[str, Any]) -> str:
    base = row["kind"].replace("_", " ")
    name = row["variant"]
    if "_" not in name or name == row["kind"]:
        return base
    return f"{base} {int(name.rsplit('_', 1)[1]) + 1}"


def _row_y(ax: Any, model: str) -> float:
    return float(
        _gid_lines(ax, f"paper:{model}")[f"paper:{model}"].get_ydata()[0]
    )


def _side(output: registry.FigureOutput) -> Any:
    (ax,) = [a for a in output.figure.axes if a.get_gid() == "table"]
    return ax


def _column(ax: Any, gid: str) -> list[Any]:
    return sorted(
        (t for t in ax.texts if t.get_gid() == gid),
        key=lambda t: t.get_position()[1],
    )


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
    ax = _strip(output)
    assert ax.get_xlim()[0] == pytest.approx(0.45)
    assert ax.get_xticks().tolist()[:3] == pytest.approx([0.5, 0.75, 1.0])


def test_every_row_has_fifteen_identical_ink_circles_in_three_lanes(
    output: registry.FigureOutput,
) -> None:
    ax = _strip(output)
    colours = set()
    lane_offsets: dict[str, set[float]] = {f: set() for f in FAMILIES}
    for model in MODELS:
        rows = _rows(model)
        centre = _row_y(ax, model)
        drawn = 0
        ys = {}
        for family in FAMILIES:
            line = _gid_lines(ax, f"words:{model}:{family}")[
                f"words:{model}:{family}"
            ]
            want = sorted(
                r["from_paper"]
                for r in rows
                if r["variant"] != "paper" and r["family"] == family
            )
            assert sorted(line.get_xdata()) == pytest.approx(want)
            assert line.get_marker() == "o"
            assert line.get_markersize() == pytest.approx(3.0)
            assert len(set(line.get_ydata())) == 1
            ys[family] = float(line.get_ydata()[0])
            lane_offsets[family].add(round(ys[family] - centre, 6))
            drawn += len(want)
            colours.add(line.get_markerfacecolor())
        assert drawn == 15
        assert ys["frame"] < ys["wording"] < ys["concept"]
    assert len(colours) == 1
    assert all(len(v) == 1 for v in lane_offsets.values())


def test_the_lanes_are_named_once_at_the_top_of_the_first_block(
    output: registry.FigureOutput,
) -> None:
    ax = _strip(output)
    lanes = [t for t in ax.texts if t.get_gid() == "lane"]
    assert [t.get_text() for t in lanes] == list(LANES.values())
    first = MODELS[0]
    ys = sorted(float(t.get_position()[1]) for t in lanes)
    for family, text in zip(FAMILIES, ys, strict=True):
        line = _gid_lines(ax, f"words:{first}:{family}")[
            f"words:{first}:{family}"
        ]
        assert text == pytest.approx(float(line.get_ydata()[0]))


def test_no_colour_key_names_the_wording_families(
    output: registry.FigureOutput,
) -> None:
    (legend,) = output.figure.legends
    texts = " ".join(t.get_text() for t in legend.get_texts())
    assert "frame" not in texts
    assert "concept swap" not in texts


def test_the_paper_wording_is_a_diamond_and_the_rank_is_in_the_right_column(
    output: registry.FigureOutput,
) -> None:
    ax = _strip(output)
    assert not [t for t in ax.texts if (t.get_gid() or "").startswith("rank:")]
    ranks = _column(_side(output), "rank")
    assert len(ranks) == len(MODELS)
    by_row = {round(_row_y(ax, m), 6): m for m in MODELS}
    for text in ranks:
        model = by_row[round(float(text.get_position()[1]), 6)]
        assert text.get_text() == str(_rank(model))
    for model in MODELS:
        line = _gid_lines(ax, f"paper:{model}")[f"paper:{model}"]
        assert line.get_marker() == "D"
        assert line.get_xdata()[0] == pytest.approx(_paper(model)["from_paper"])
        assert line.get_markersize() > 3.0
    ranked = [_rank(m) for m in MODELS]
    assert output.numbers["paper_rank_min"] == min(ranked)
    assert output.numbers["paper_rank_max"] == max(ranked)


def test_the_median_tick_the_noise_band_and_the_random_tick_are_drawn(
    output: registry.FigureOutput,
) -> None:
    ax = _strip(output)
    for model in MODELS:
        rows = _rows(model)
        values = [r["from_paper"] for r in rows]
        row = _paper(model)
        median = _gid_lines(ax, f"median:{model}")[f"median:{model}"]
        assert median.get_xdata()[0] == pytest.approx(float(np.median(values)))
        tick = _gid_lines(ax, f"random:{model}")[f"random:{model}"]
        assert tick.get_xdata()[0] == pytest.approx(row["random"])
        (band,) = [p for p in ax.patches if p.get_gid() == f"noise:{model}"]
        assert band.get_width() == pytest.approx(_noise(model))
        assert band.get_x() + band.get_width() / 2 == pytest.approx(
            row["from_paper"]
        )


def test_the_right_columns_are_paper_noise_span_rank_and_the_mark(
    output: registry.FigureOutput,
) -> None:
    ax = _side(output)
    heads = [
        t.get_text().replace("\n", " ")
        for t in ax.texts
        if t.get_gid() == "column"
    ]
    assert heads == COLUMN_HEADERS
    for gid, want in (
        ("paper_acc", sorted(f"{_paper(m)['from_paper']:.3f}" for m in MODELS)),
        ("noise_value", sorted(f"{_noise(m):.3f}" for m in MODELS)),
        ("span", sorted(f"{_span(m):.3f}" for m in MODELS)),
    ):
        assert (
            sorted(t.get_text() for t in ax.texts if t.get_gid() == gid) == want
        )
    marked = [t for t in ax.texts if t.get_gid() == "above_noise"]
    assert len(marked) == sum(_span(m) > _noise(m) for m in MODELS) == 8


def test_the_extreme_wordings_are_named_on_their_own_row_centre_line(
    output: registry.FigureOutput,
) -> None:
    ax = _strip(output)
    named = [t for t in ax.texts if t.get_gid() == "named"]
    expected = 0
    for model in MODELS:
        rows = sorted(
            (r for r in _rows(model) if r["variant"] != "paper"),
            key=lambda r: r["from_paper"],
        )
        centre = _row_y(ax, model)
        half = _noise(model) / 2
        paper_acc = _paper(model)["from_paper"]
        mine = [
            t for t in named if t.get_position()[1] == pytest.approx(centre)
        ]
        for pick, side in ((rows[0], "right"), (rows[-1], "left")):
            if abs(pick["from_paper"] - paper_acc) <= half:
                continue
            expected += 1
            (text,) = [t for t in mine if t.get_text() == _table_name(pick)]
            assert text.get_ha() == side
            x = text.get_position()[0]
            if side == "right":
                assert x < rows[0]["from_paper"]
            else:
                assert x > rows[-1]["from_paper"]
    assert len(named) == expected


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
    titles = {
        "gpt2": "GPT-2",
        "base": "Qwen3 base",
        "post": "Qwen3 post-trained",
    }
    for role, ax in _facets(output).items():
        assert ax.get_title() == titles[role]
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


def test_each_facet_draws_the_random_direction_as_lines_and_one_chance_line(
    output: registry.FigureOutput,
) -> None:
    for role, ax in _facets(output).items():
        mine = [m for m in MODELS if _role(m) == role]
        lines = _gid_lines(ax, "random_line:")
        assert sorted(lines) == sorted(f"random_line:{m}" for m in mine)
        for model in mine:
            ln = lines[f"random_line:{model}"]
            assert list(ln.get_ydata()) == pytest.approx(
                [_paper(model)["random"]] * 2
            )
            assert list(ln.get_xdata()) == pytest.approx(list(ax.get_xlim()))
        chance = [ln for ln in ax.get_lines() if ln.get_gid() == "chance"]
        assert len(chance) == 1
        assert list(chance[0].get_xdata()) == pytest.approx(list(ax.get_xlim()))


def test_the_key_names_the_diamond_the_seed_ticks_and_the_random_direction(
    output: registry.FigureOutput,
) -> None:
    (legend,) = output.figure.legends
    texts = " ".join(t.get_text() for t in legend.get_texts())
    assert "paper direction" in texts
    assert "across seeds" in texts
    assert "random direction" in texts


def test_the_text_is_at_least_the_glossary_floor_and_never_bold(
    output: registry.FigureOutput,
) -> None:
    for t in output.figure.findobj(Text):
        if t.get_visible() and t.get_text():
            assert t.get_fontsize() >= paper.MIN_TEXT, t.get_text()
            assert t.get_fontweight() in ("normal", 400), t.get_text()


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
    assert "names as in [[table:wordings]]" in text
    assert "three lanes" in text
    assert (
        f"{int(n['paper_rank_min'])} to {int(n['paper_rank_max'])} of the 16"
        in output.claim
    )
