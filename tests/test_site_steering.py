import csv
import io
import itertools
import json
from pathlib import Path
import re
import runpy
from typing import Any

from matplotlib.collections import PolyCollection
import numpy as np
import pytest

from tomography.site import registry
from tomography.site import stats
from tomography.site.figs import steering
from tomography.site.manifest import Manifest
from tomography.viz import axes
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
_T = runpy.run_path(str(Path(__file__).with_name("_titles.py")))
_L = runpy.run_path(str(Path(__file__).with_name("_legends.py")))
SPEC = "steering"
MAX_RATIO = 1.5
MIN_DISTINCT = 0.15
COEFFS = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
MINUS = chr(0x2212)
ROLE_NAMES = {"gpt2": "GPT-2", "base": "Qwen3 base", "post": "post-trained"}


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(output.csv)))


def _models() -> list[str]:
    models: list[str] = _R["models"]("c1")
    return models


def _table(model: str) -> list[dict[str, Any]]:
    path = _R["run"]("c1", model)
    payload = _R["data"](path, "table.json")
    return [
        dict(zip(payload["columns"], r, strict=True)) for r in payload["rows"]
    ]


def _linear(rows: list[dict[str, Any]], coeff: float) -> list[dict[str, Any]]:
    cell = [r for r in rows if r["arm"] == "linear" and r["coeff"] == coeff]
    assert len(cell) == 3
    return cell


def _none(rows: list[dict[str, Any]]) -> dict[str, Any]:
    (row,) = [r for r in rows if r["arm"] == "none"]
    return row


def _score(rows: list[dict[str, Any]], coeff: float) -> float:
    if coeff == 0.0:
        return float(_none(rows)["score"])
    return float(np.mean([r["score"] for r in _linear(rows, coeff)]))


def _ratio(rows: list[dict[str, Any]], coeff: float) -> float:
    if coeff == 0.0:
        return 1.0
    mean = float(np.mean([r["perplexity"] for r in _linear(rows, coeff)]))
    return mean / float(_none(rows)["perplexity"])


def _distinct(rows: list[dict[str, Any]], coeff: float) -> float:
    if coeff == 0.0:
        return float(_none(rows)["distinct_1"])
    return float(np.mean([r["distinct_1"] for r in _linear(rows, coeff)]))


def _fluent(rows: list[dict[str, Any]], coeff: float) -> bool:
    return (
        _ratio(rows, coeff) <= MAX_RATIO
        and _distinct(rows, coeff) >= MIN_DISTINCT
    )


def _tile(rows: list[dict[str, Any]], coeff: float) -> str:
    ratio = _ratio(rows, coeff)
    text = f"{ratio:.2f}" if ratio < 10 else f"{ratio:.0f}"
    return text + ("*" if _distinct(rows, coeff) < MIN_DISTINCT else "")


def _has(text: str, value: float) -> bool:
    shown = str(int(value)) if value == int(value) else f"{value:.3f}"
    pattern = rf"(?<![\d.]){re.escape(shown)}(?![\d])"
    return re.search(pattern, text) is not None


def _by_role() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {"gpt2": [], "base": [], "post": []}
    for model in _models():
        found[stats.role(model)].append(model)
    return found


def _role_mean(coeff: float) -> dict[str, float]:
    return {
        role: float(np.mean([_score(_table(m), coeff) for m in members]))
        for role, members in _by_role().items()
    }


def _texts(output: registry.FigureOutput) -> list[str]:
    checks.draw(output.figure)
    return [
        t.get_text()
        for ax in output.figure.axes
        for t in ax.texts
        if t.get_text()
    ]


def test_score_lines_are_seed_means_with_the_range_over_the_three_vectors(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for model in _models():
        table = _table(model)
        got = {
            float(r["x"]): r
            for r in rows
            if r["model"] == model and r["series"] == "score"
        }
        for coeff in COEFFS:
            assert float(got[coeff]["value"]) == pytest.approx(
                _score(table, coeff)
            )
            if coeff == 0.0:
                assert float(got[coeff]["lo"]) == float(got[coeff]["hi"])
                continue
            scores = [r["score"] for r in _linear(table, coeff)]
            assert float(got[coeff]["lo"]) == pytest.approx(min(scores))
            assert float(got[coeff]["hi"]) == pytest.approx(max(scores))


def test_prompting_triangles_are_the_prompting_scores(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for model in _models():
        for arm in ("prompting+1", "prompting-1"):
            (want,) = [r for r in _table(model) if r["arm"] == arm]
            (got,) = [
                r for r in rows if r["model"] == model and r["series"] == arm
            ]
            assert float(got["value"]) == pytest.approx(want["score"])


def test_fluency_counts_are_per_role(output: registry.FigureOutput) -> None:
    for name, size in (("fluent01", 0.1), ("fluent03", 0.3)):
        for role, members in _by_role().items():
            if name == "fluent01":
                hits = sum(
                    _fluent(_table(m), size) and _fluent(_table(m), -size)
                    for m in members
                )
            else:
                hits = sum(
                    _fluent(_table(m), size) or _fluent(_table(m), -size)
                    for m in members
                )
            count = output.counts[name][role]
            assert (count.hits, count.total) == (hits, len(members))
            assert output.numbers[f"{name}_{role}"] == hits
            assert output.numbers[f"n_{role}"] == len(members)


def test_score_means_are_taken_within_a_role_only(
    output: registry.FigureOutput,
) -> None:
    for role, value in _role_mean(0.0).items():
        assert output.numbers[f"unsteered_{role}"] == pytest.approx(value)
    for role, value in _role_mean(-0.1).items():
        assert output.numbers[f"minus_{role}"] == pytest.approx(value)
    for role, value in _role_mean(0.1).items():
        assert output.numbers[f"plus_{role}"] == pytest.approx(value)
    for key in output.numbers:
        assert key == "n_neutral" or key.rsplit("_", 1)[-1] in ROLE_NAMES, key


def test_caption_is_the_template_filled_from_the_files(
    output: registry.FigureOutput,
) -> None:
    by_role = _by_role()
    flags01 = {
        m: _fluent(_table(m), 0.1) and _fluent(_table(m), -0.1)
        for m in _models()
    }
    flags03 = {
        m: _fluent(_table(m), 0.3) or _fluent(_table(m), -0.3)
        for m in _models()
    }
    held = stats.per_role_text(stats.per_role(flags01))
    assert not any(flags03.values())
    fails = f"none of the {len(_models())} checkpoints"
    assert sum(len(v) for v in by_role.values()) == len(_models())
    config = json.loads(
        (_R["run"]("c1", "gpt2") / "config.json").read_text(encoding="utf-8")
    )
    n_neutral = int(config["n_neutral"])
    caption = f"{output.claim} {output.detail}"
    assert output.claim == (
        "Steering moves the judged sentiment, less on post-trained checkpoints."
    )
    assert (
        f"Perplexity ratio, steered to unsteered model, on {n_neutral} "
        "held-out WikiText-103 passages"
    ) in caption
    assert f"The fluency rule holds at both signs of 0.1 on {held}" in caption
    assert f"and at 0.3 or {MINUS}0.3 on {fails}." in caption
    assert "no random-direction arm" in caption
    assert "range over the 3 steering vectors" in caption
    assert output.numbers["n_neutral"] == n_neutral


def test_the_perplexity_is_not_called_the_perplexity_of_the_completions(
    output: registry.FigureOutput,
) -> None:
    assert "unsteered text" not in output.detail
    assert "not of the completions" in output.detail


def test_negative_numbers_use_the_true_minus(
    output: registry.FigureOutput,
) -> None:
    checks.draw(output.figure)
    shown = [
        t.get_text()
        for ax in output.figure.axes
        for t in (*ax.get_xticklabels(), *ax.get_yticklabels())
    ]
    assert not [t for t in shown if re.match(r"-\d", t)]
    assert any(MINUS in t or "{-" in t for t in shown)


def test_no_dash_characters_in_the_caption_or_csv(
    output: registry.FigureOutput,
) -> None:
    text = f"{output.claim} {output.detail} {output.csv}"
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text


def test_svg_and_pdf_bytes_are_the_same_for_the_same_input() -> None:
    first = registry.build_figure(SPEC, Manifest(RESULTS))
    second = registry.build_figure(SPEC, Manifest(RESULTS))
    with paper.paper_style():
        assert paper.svg_bytes(first.figure) == paper.svg_bytes(second.figure)
        assert paper.pdf_bytes(first.figure) == paper.pdf_bytes(second.figure)


def test_the_layout_is_frozen_so_two_builds_place_every_axes_alike() -> None:
    first = registry.build_figure(SPEC, Manifest(RESULTS)).figure
    second = registry.build_figure(SPEC, Manifest(RESULTS)).figure
    for fig in (first, second):
        for ax in fig.axes:
            assert all(
                abs(v - round(v, 6)) < 1e-12 for v in ax.get_position().bounds
            )
    assert [ax.get_position().bounds for ax in first.axes] == [
        ax.get_position().bounds for ax in second.axes
    ]


def test_the_fluency_rule_in_the_caption_comes_from_the_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tomography.site.figs import steering

    monkeypatch.setattr(steering, "MAX_RATIO", 1.25)
    monkeypatch.setattr(steering, "MIN_DISTINCT", 0.2)
    detail = registry.build_figure(SPEC, Manifest(RESULTS)).detail
    assert "(perplexity ratio at most 1.25, distinct-1 at least 0.2)" in detail


def test_cross_references_are_tokens_and_not_literal_numbers(
    output: registry.FigureOutput,
) -> None:
    assert "[[fig:teaser]]" in output.detail
    assert "[[table:steering]]" in output.detail
    assert not re.search(r"\b(Figure|Table) \d", output.detail)


def _moves(rows: list[dict[str, Any]], column: str) -> bool:
    base = float(_none(rows)[column])
    return all(
        sign * (float(r[column]) - base) > 0
        for coeff, sign in ((0.1, 1), (-0.1, -1))
        for r in _linear(rows, coeff)
    )


def test_moves_counts_need_every_vector_at_both_signs_on_both_judges(
    output: registry.FigureOutput,
) -> None:
    want = {
        "moves01": lambda r: _moves(r, "score") and _moves(r, "score_check"),
        "moves01_primary": lambda r: _moves(r, "score"),
        "moves01_check": lambda r: _moves(r, "score_check"),
        "judges_agree": lambda r: (
            _moves(r, "score") == _moves(r, "score_check")
        ),
    }
    for name, rule in want.items():
        for role, members in _by_role().items():
            hits = sum(rule(_table(m)) for m in members)
            count = output.counts[name][role]
            assert (count.hits, count.total) == (hits, len(members))
            assert output.numbers[f"{name}_{role}"] == hits
    assert [
        output.counts["moves01"][r].hits for r in ("gpt2", "base", "post")
    ] == [
        4,
        3,
        2,
    ]


def test_caption_states_the_moves_rule_and_the_fluency_inputs(
    output: registry.FigureOutput,
) -> None:
    moves = stats.per_role_text(output.counts["moves01"])
    assert f"two sentiment judges: {moves}." in output.detail
    assert "all 3 steering vectors" in output.detail
    assert "judge agreement" in output.detail
    assert "not of the completions" in output.detail


ORDER = ("gpt2", "base", "post")
NEWLINE = chr(10)


def _grid(output: registry.FigureOutput) -> list[list[Any]]:
    axs = output.figure.axes
    assert len(axs) == 9
    ordered = sorted(
        axs, key=lambda a: (-round(a.get_position().y1, 3), a.get_position().x0)
    )
    return [ordered[0:3], ordered[3:6], ordered[6:9]]


def _points(ax: Any) -> list[Any]:
    return [ln for ln in ax.lines if ln.get_gid() == "point"]


def _filled(line: Any) -> bool:
    return line.get_markerfacecolor() != "white"


def _flags(model: str) -> dict[float, bool]:
    table = _table(model)
    return {c: _fluent(table, c) for c in COEFFS}


def test_the_figure_is_three_columns_by_three_rows(
    output: registry.FigureOutput,
) -> None:
    grid = _grid(output)
    assert tuple(output.figure.get_size_inches()) == (
        paper.FULL_WIDTH,
        steering.HEIGHT,
    )
    for row in grid:
        assert len({round(a.get_position().y0, 3) for a in row}) == 1
    for col in range(3):
        lefts = {round(grid[r][col].get_position().x0, 3) for r in range(3)}
        assert len(lefts) == 1
    assert [a.get_title() for a in grid[0]] == [
        "GPT-2",
        "Qwen3 base",
        "Qwen3 post-trained",
    ]
    assert checks.guard_problems(output.figure) == []
    assert checks.problems(output.figure) == []


def test_axes_are_shared_and_labelled_on_the_outer_panels_only(
    output: registry.FigureOutput,
) -> None:
    checks.draw(output.figure)
    grid = _grid(output)
    for r, row in enumerate(grid):
        for c, ax in enumerate(row):
            assert ax.get_xlim() == grid[2][0].get_xlim()
            assert ax.get_ylim() == grid[r][0].get_ylim()
            shown_x = any(t.get_text() for t in ax.get_xticklabels())
            shown_y = any(t.get_text() for t in ax.get_yticklabels())
            assert ax.xaxis.get_tick_params()["labelbottom"] == (r == 2)
            assert ax.yaxis.get_tick_params()["labelleft"] == (c == 0)
            assert shown_x or r < 2
            assert shown_y or c > 0
    left, right = grid[2][0].get_xlim()
    assert left <= -0.3
    assert right >= 0.3
    assert grid[2][0].get_xticks().tolist() == [-0.3, 0.0, 0.3]
    assert grid[0][0].get_ylim() == (0.0, 1.0)
    assert grid[1][0].get_yscale() == "log"
    assert grid[1][0].get_ylim() == (1.0, 100.0)
    assert grid[2][0].get_yscale() == "linear"
    assert grid[2][0].get_ylim() == (0.0, 0.35)


def test_the_limits_are_labelled_reference_lines(
    output: registry.FigureOutput,
) -> None:
    checks.draw(output.figure)
    grid = _grid(output)
    for row, value, label in (
        (1, MAX_RATIO, f"limit {MAX_RATIO:g}"),
        (2, MIN_DISTINCT, f"limit {MIN_DISTINCT:g}"),
    ):
        for ax in grid[row]:
            refs = [ln for ln in ax.lines if ln.get_gid() == "limit"]
            assert len(refs) == 1
            assert list(refs[0].get_ydata()) == [value, value]
            assert refs[0].get_linewidth() == pytest.approx(0.5)
        for ax in grid[row]:
            texts = [t.get_text().replace(NEWLINE, " ") for t in ax.texts]
            assert label in texts


def test_a_marker_per_checkpoint_and_coefficient_filled_when_fluent(
    output: registry.FigureOutput,
) -> None:
    grid = _grid(output)
    roles = _by_role()
    for col, role in enumerate(ORDER):
        for row in range(3):
            points = _points(grid[row][col])
            assert len(points) == len(COEFFS) * len(roles[role])
        want = sum(_flags(m)[c] for m in roles[role] for c in COEFFS)
        for row in range(3):
            filled = sum(_filled(p) for p in _points(grid[row][col]))
            assert filled == want
    flags = [r for r in _rows(output) if r["series"] == "fluent"]
    assert len(flags) == 70
    assert sum(float(r["value"]) for r in flags) == sum(
        _flags(m)[c] for m in _models() for c in COEFFS
    )


def test_segments_touching_a_non_fluent_point_are_dashed(
    output: registry.FigureOutput,
) -> None:
    grid = _grid(output)
    roles = _by_role()
    for col, role in enumerate(ORDER):
        for row in range(3):
            segments = [
                ln for ln in grid[row][col].lines if ln.get_gid() == "segment"
            ]
            assert len(segments) == 6 * len(roles[role])
            dashed = [ln for ln in segments if ln.get_linestyle() != "-"]
            want = 0
            for model in roles[role]:
                flags = _flags(model)
                want += sum(
                    not (flags[a] and flags[b])
                    for a, b in itertools.pairwise(COEFFS)
                )
            assert len(dashed) == want


def test_instruction_scores_are_bars_left_of_the_curves_by_checkpoint_colour(
    output: registry.FigureOutput,
) -> None:
    grid = _grid(output)
    roles = _by_role()
    for col, role in enumerate(ORDER):
        ax = grid[0][col]
        assert not [ln for ln in ax.lines if ln.get_linestyle() == ":"]
        bars = [ln for ln in ax.lines if ln.get_gid() == "instruction"]
        first = min(COEFFS) - 0.02
        assert all(max(ln.get_xdata()) < first for ln in bars)
        rows = axes.model_rows(roles[role])
        for model in sorted(roles[role], key=rows.get):
            own = [
                ln
                for ln in bars
                if ln.get_color() == axes.model_style(model).color
            ]
            want = sorted(
                float(r["score"])
                for r in _table(model)
                if r["arm"] in ("prompting+1", "prompting-1")
            )
            assert sorted(float(ln.get_ydata()[0]) for ln in own) == (
                pytest.approx(want)
            )
            by_arm = {r["arm"]: r["score"] for r in _table(model)}
            assert by_arm["prompting+1"] > by_arm["prompting-1"]
        slots = {}
        for model in roles[role]:
            xs = {
                float(np.mean(ln.get_xdata()))
                for ln in bars
                if ln.get_color() == axes.model_style(model).color
            }
            assert len(xs) == 1
            slots[model] = xs.pop()
        assert sorted(slots, key=slots.get) == sorted(roles[role], key=rows.get)


def test_the_range_over_the_vectors_is_a_low_alpha_band_and_no_heat_map(
    output: registry.FigureOutput,
) -> None:
    grid = _grid(output)
    for ax in grid[0]:
        bands = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert bands
        assert all(c.get_alpha() <= 0.2 for c in bands)
    assert not [
        c
        for ax in output.figure.axes
        for c in ax.collections
        if c.__class__.__name__ == "QuadMesh"
    ]


def test_checkpoints_are_labelled_at_the_line_ends_and_not_in_a_legend(
    output: registry.FigureOutput,
) -> None:
    checks.draw(output.figure)
    grid = _grid(output)
    roles = _by_role()
    for col, role in enumerate(ORDER):
        texts = [t.get_text() for t in grid[0][col].texts]
        for model in roles[role]:
            assert steering.end_label(model) in texts
    (legend,) = output.figure.legends
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == [
        "fluent",
        "not fluent",
        "instruction score (bars at left)",
    ]


def test_end_labels_keep_size_order_from_top_to_bottom_and_do_not_overlap(
    output: registry.FigureOutput,
) -> None:
    checks.draw(output.figure)
    grid = _grid(output)
    roles = _by_role()
    for col, role in enumerate(ORDER):
        rows = axes.model_rows(roles[role])
        names = {steering.end_label(m): m for m in roles[role]}
        found = [t for t in grid[0][col].texts if t.get_text() in names]
        found.sort(key=lambda t: -t.get_position()[1])
        got = [names[t.get_text()] for t in found]
        assert got == sorted(roles[role], key=rows.get)
        boxes = [t.get_window_extent() for t in found]
        for upper, lower in itertools.pairwise(boxes):
            assert upper.y0 >= lower.y1


def test_each_column_states_how_many_checkpoints_stay_fluent(
    output: registry.FigureOutput,
) -> None:
    checks.draw(output.figure)
    grid = _grid(output)
    for col, role in enumerate(ORDER):
        count = output.counts["fluent01"][role]
        want = f"fluent at both signs of 0.1: {count.hits} of {count.total}"
        assert want in [
            t.get_text().replace(NEWLINE, " ") for t in grid[0][col].texts
        ]


def test_the_three_rows_carry_letters_a_to_c(
    output: registry.FigureOutput,
) -> None:
    letters = [t for t in _texts(output) if t.startswith(r"$\mathbf")]
    assert sorted(letters) == [rf"$\mathbf{{({c})}}$" for c in "abc"]


def test_only_the_direction_circle_is_drawn_as_a_marker(
    output: registry.FigureOutput,
) -> None:
    drawn = {
        ln.get_marker()
        for ax in output.figure.axes
        for ln in ax.lines
        if ln.get_marker() not in ("None", "", None)
    }
    assert drawn == {paper.SHAPES["direction"]}
    assert _L["marker_clashes"](output.figure) == []


def test_the_caption_keys_the_drawing_and_has_no_heat_map_panel(
    output: registry.FigureOutput,
) -> None:
    detail = output.detail
    for word in ("Filled", "open", "dashed", "bars", "ratio"):
        assert word in detail
    for letter in "abc":
        assert f"({letter}" in detail
    assert "(d)" not in detail
    assert "[[table:steering]]" in detail


def test_the_caption_has_at_most_200_words_and_leads_with_the_conclusion(
    output: registry.FigureOutput,
) -> None:
    assert len(f"{output.claim} {output.detail}".split()) <= 200
    assert output.detail.startswith("A checkpoint moves when all 3 steering")


def test_the_key_and_the_limit_labels_read_at_the_tick_size(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    sizes = [
        text.get_fontsize()
        for legend in fig.legends
        for text in legend.get_texts()
    ]
    for ax in fig.axes:
        sizes += [
            text.get_fontsize()
            for text in ax.texts
            if text.get_text().startswith("limit")
        ]
    assert sizes
    assert min(sizes) >= paper.TICK
