from collections import Counter
import csv
import io
import json
from pathlib import Path
import re
import runpy
from typing import Any

import matplotlib as mpl
from matplotlib.collections import QuadMesh
from matplotlib.patches import Rectangle
import numpy as np
import pytest

from tomography.site import registry
from tomography.site import stats
from tomography.site.manifest import Manifest
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


def test_every_checkpoint_has_a_score_line_and_a_prompting_strip(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for model in _models():
        got = [
            r for r in rows if r["model"] == model and r["series"] == "score"
        ]
        assert sorted(float(r["x"]) for r in got) == list(COEFFS)
        for arm in ("prompting+1", "prompting-1"):
            assert (
                len(
                    [
                        r
                        for r in rows
                        if r["model"] == model and r["series"] == arm
                    ]
                )
                == 1
            )


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


def test_tile_matrix_has_seventy_tiles_with_the_perplexity_ratio(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    tiles = [r for r in rows if r["series"] == "ppl_ratio"]
    assert len(tiles) == 70
    for row in tiles:
        table = _table(row["model"])
        assert float(row["value"]) == pytest.approx(
            _ratio(table, float(row["x"]))
        )
    distinct = [r for r in rows if r["series"] == "distinct_1"]
    assert len(distinct) == 70
    for row in distinct:
        table = _table(row["model"])
        assert float(row["value"]) == pytest.approx(
            _distinct(table, float(row["x"]))
        )


def test_tile_labels_show_the_ratio_and_the_distinct_1_asterisk(
    output: registry.FigureOutput,
) -> None:
    want = Counter(_tile(_table(m), c) for m in _models() for c in COEFFS)
    have = Counter(_texts(output))
    assert not want - have


def test_framed_tiles_are_exactly_those_that_keep_the_fluency_rule(
    output: registry.FigureOutput,
) -> None:
    frames = [
        p
        for ax in output.figure.axes
        for p in ax.patches
        if isinstance(p, Rectangle)
        and not p.get_fill()
        and p.get_linewidth() == pytest.approx(0.9)
    ]
    want = sum(_fluent(_table(m), c) for m in _models() for c in COEFFS)
    assert len(frames) == want
    rows = _rows(output)
    flags = [r for r in rows if r["series"] == "fluent"]
    assert len(flags) == 70
    assert sum(float(r["value"]) for r in flags) == want


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
    unsteered, minus, plus = _role_mean(0.0), _role_mean(-0.1), _role_mean(0.1)
    sentences = []
    for role, label in (
        ("gpt2", "on GPT-2"),
        ("base", "on Qwen3 base"),
        ("post", "on post-trained checkpoints"),
    ):
        sentences.append(
            f"from {unsteered[role]:.3f} to {minus[role]:.3f} and "
            f"{plus[role]:.3f} {label}"
        )
    moves = f"{sentences[0]}, {sentences[1]} and {sentences[2]}"
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
        f"At relative coefficients {MINUS}0.1 and +0.1 the mean score moves "
        f"{moves}."
    ) in caption
    assert (
        f"Perplexity of {n_neutral} held-out WikiText-103 passages" in caption
    )
    assert f"The rule holds at both signs of 0.1 on {held}" in caption
    assert f"and at 0.3 or {MINUS}0.3 on {fails}." in caption
    assert "no random-direction arm" in caption
    assert "range over the 3 steering vectors" in caption
    assert output.numbers["n_neutral"] == n_neutral


def test_the_perplexity_is_not_called_the_perplexity_of_the_completions(
    output: registry.FigureOutput,
) -> None:
    assert "unsteered text" not in output.detail.split("Perplexity of")[0]
    assert "not the perplexity of the completions" in output.detail


def test_size_guards_and_layout(output: registry.FigureOutput) -> None:
    fig = output.figure
    assert tuple(fig.get_size_inches()) == (paper.FULL_WIDTH, 3.6)
    assert checks.guard_problems(fig) == []
    assert checks.problems(fig) == []


def test_one_panel_letter_each_for_the_four_panels(
    output: registry.FigureOutput,
) -> None:
    letters = [t for t in _texts(output) if t.startswith(r"$\mathbf")]
    assert sorted(letters) == [rf"$\mathbf{{({c})}}$" for c in "abcd"]


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


def test_tiles_use_the_full_colormap_and_the_ink_that_contrasts_most(
    output: registry.FigureOutput,
) -> None:
    ax = next(
        a
        for a in output.figure.axes
        if any(isinstance(c, QuadMesh) for c in a.collections)
        and len([c for c in a.collections if isinstance(c, QuadMesh)]) > 2
    )
    meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
    labels = [t for t in ax.texts if t.get_gid() == paper.TILE_LABEL]
    assert len(labels) == 7 * (len(meshes))
    cmap = meshes[0].get_cmap()
    assert cmap(0.0) == mpl.colormaps["cividis_r"](0.0)
    assert cmap(1.0) == mpl.colormaps["cividis_r"](1.0)
    for i, mesh in enumerate(meshes):
        mesh.update_scalarmappable()
        for j, face in enumerate(mesh.get_facecolor()):
            text = labels[7 * i + j]
            assert text.get_color() == paper.tile_ink(face)
    inks = {t.get_color() for t in labels}
    assert inks == {paper.INK, paper.WHITE}


def test_letters_do_not_touch_their_titles(
    output: registry.FigureOutput,
) -> None:
    gaps = _T["letter_title_gaps"](output.figure)
    assert len(gaps) == 3
    assert min(gaps) >= 2.0


def test_the_fluency_rule_in_the_caption_comes_from_the_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tomography.site.figs import steering

    monkeypatch.setattr(steering, "MAX_RATIO", 1.25)
    monkeypatch.setattr(steering, "MIN_DISTINCT", 0.2)
    detail = registry.build_figure(SPEC, Manifest(RESULTS)).detail
    assert "distinct-1 of the completions below 0.2" in detail
    assert "(ratio at most 1.25 and distinct-1 at least 0.2)" in detail


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
    agree = stats.per_role_text(output.counts["judges_agree"])
    assert f"on {moves}" in output.detail
    assert "all 3 steering vectors" in output.detail
    assert "second judge" in output.detail
    assert f"the two judges agree on {agree}" in output.detail
    assert "not the perplexity of the completions" in output.detail


def test_markers_do_not_encode_the_checkpoint_and_prompting_has_its_own(
    output: registry.FigureOutput,
) -> None:
    drawn = {
        ln.get_marker()
        for ax in output.figure.axes
        for ln in ax.lines
        if ln.get_marker() not in ("None", "", None)
    }
    direction = paper.MEASURE_MARKERS["direction"]
    prompts = set(paper.PROMPT_MARKERS.values())
    assert drawn == {direction} | prompts
    assert _L["marker_clashes"](output.figure) == []
    (legend,) = output.figure.legends
    labels = [t.get_text() for t in legend.get_texts()]
    assert "positive instruction" in labels
    assert "negative instruction" in labels
    assert "triangles" not in output.detail
