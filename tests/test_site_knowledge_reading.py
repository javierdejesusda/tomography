import csv
import io
import math
from pathlib import Path
import runpy
from typing import Any

import pytest

from tomography.site import registry
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
SPEC = "knowledge_reading"
Z = 1.959963984540054
MINUS = chr(0x2212)
HEIGHT = 3.4
READER = ("logistic_within", "popularity_within")
BASELINES = ("text_word_within", "text_char_within")


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(output.csv)))


def _models(exp: str) -> list[str]:
    models: list[str] = _R["models"](exp)
    return models


def _table(
    exp: str, model: str, name: str = "table.json"
) -> list[dict[str, Any]]:
    payload = _R["data"](_R["run"](exp, model), name)
    if isinstance(payload, dict):
        return [
            dict(zip(payload["columns"], r, strict=True))
            for r in payload["rows"]
        ]
    rows: list[dict[str, Any]] = payload
    return rows


def _contrast() -> dict[str, dict[str, Any]]:
    found = {}
    for model in _models("p9b"):
        rows = [r for r in _table("p9b", model) if r["type"] == "obscure_real"]
        if rows:
            found[model] = rows[0]
    return found


def _increments() -> dict[str, dict[str, dict[str, Any]]]:
    return {
        m: {
            r["features"]: r
            for r in _table("p9c", m, "table_incremental.json")
            if r["dataset"] == "popqa"
        }
        for m in _models("p9c")
    }


def _se(auc: float, n_pos: int, n_neg: int) -> float:
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    var = (
        auc * (1 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    return math.sqrt(var)


def _diff(row: dict[str, Any]) -> tuple[float, float, float]:
    a, b = row["logistic_within"], row["popularity_within"]
    n_pos, n_neg = row["n_positive_test"], row["n_negative_test"]
    half = Z * math.sqrt(_se(a, n_pos, n_neg) ** 2 + _se(b, n_pos, n_neg) ** 2)
    return a - b, a - b - half, a - b + half


def _text(phrase: stats.RoleCount | dict[str, stats.RoleCount]) -> str:
    assert isinstance(phrase, dict)
    return stats.per_role_text(phrase)


def _flags(rule: Any) -> dict[str, bool]:
    return {m: bool(rule(r)) for m, r in _contrast().items()}


def _by_role(flags: dict[str, bool]) -> dict[str, stats.RoleCount]:
    return stats.per_role(flags)


def test_every_checkpoint_with_a_contrast_has_the_four_panel_a_values(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    contrast = _contrast()
    assert "gpt2" not in contrast
    for model, row in contrast.items():
        for series, column in (
            ("direction", "logistic_within"),
            ("popularity", "popularity_within"),
            ("words", "text_word_within"),
            ("characters", "text_char_within"),
        ):
            (got,) = [
                r
                for r in rows
                if r["panel"] == "a"
                and r["model"] == model
                and r["series"] == series
            ]
            assert float(got["value"]) == pytest.approx(row[column])
    assert not [r for r in rows if r["panel"] == "a" and r["model"] == "gpt2"]


def test_direction_and_popularity_carry_hanley_mcneil_intervals(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for model, row in _contrast().items():
        for series, column in (
            ("direction", "logistic_within"),
            ("popularity", "popularity_within"),
        ):
            (got,) = [
                r
                for r in rows
                if r["panel"] == "a"
                and r["model"] == model
                and r["series"] == series
            ]
            value = row[column]
            half = Z * _se(
                value, row["n_positive_test"], row["n_negative_test"]
            )
            assert float(got["lo"]) == pytest.approx(max(0.0, value - half))
            assert float(got["hi"]) == pytest.approx(min(1.0, value + half))
        for series in ("words", "characters"):
            (got,) = [
                r
                for r in rows
                if r["panel"] == "a"
                and r["model"] == model
                and r["series"] == series
            ]
            assert got["lo"] == ""
            assert got["hi"] == ""


def test_the_arbiter_difference_for_qwen3_0_6b_base(
    output: registry.FigureOutput,
) -> None:
    d, lo, hi = _diff(_contrast()["qwen3-0.6b-base"])
    assert (round(d, 3), round(lo, 3), round(hi, 3)) == (0.120, 0.050, 0.189)


def test_panel_b_has_the_stored_paired_increments(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for model, found in _increments().items():
        for series, feature in (
            ("direction", "confidence+lat"),
            ("probe", "confidence+probe"),
        ):
            got = [
                r
                for r in rows
                if r["panel"] == "b"
                and r["model"] == model
                and r["series"] == series
            ]
            if feature not in found:
                assert not got
                continue
            (cell,) = got
            assert float(cell["value"]) == pytest.approx(
                found[feature]["diff_vs_confidence"]
            )
            assert float(cell["lo"]) == pytest.approx(found[feature]["lo"])
            assert float(cell["hi"]) == pytest.approx(found[feature]["hi"])
    assert "gpt2" not in {
        r["model"]
        for r in rows
        if r["panel"] == "b" and r["series"] == "direction"
    }


def test_counts_are_per_role_and_use_the_interval_criterion(
    output: registry.FigureOutput,
) -> None:
    point = _by_role(
        _flags(lambda r: r["logistic_within"] > r["popularity_within"])
    )
    interval = _by_role(_flags(lambda r: _diff(r)[1] > 0))
    probe = _by_role(
        {m: f["confidence+probe"]["lo"] > 0 for m, f in _increments().items()}
    )
    for name, want in (
        ("above_pop_point", point),
        ("above_pop", interval),
        ("probe_above", probe),
    ):
        for role in stats.ROLES:
            got = output.counts[name][role]
            assert (got.hits, got.total) == (want[role].hits, want[role].total)
            assert output.numbers[f"{name}_{role}"] == want[role].hits
    assert output.counts["above_pop"]["gpt2"].total == 3


def test_increment_range_numbers(output: registry.FigureOutput) -> None:
    lat = [
        f["confidence+lat"]
        for f in _increments().values()
        if "confidence+lat" in f
    ]
    probe = [f["confidence+probe"] for f in _increments().values()]
    diffs = [r["diff_vs_confidence"] for r in lat]
    assert output.numbers["dir_inc_min"] == pytest.approx(min(diffs))
    assert output.numbers["dir_inc_max"] == pytest.approx(max(diffs))
    assert output.numbers["dir_inc_n"] == len(lat)
    assert output.numbers["dir_inc_cover_all"] == sum(
        r["lo"] <= 0 <= r["hi"] for r in lat
    )
    gains = [r["diff_vs_confidence"] for r in probe]
    assert output.numbers["probe_inc_min"] == pytest.approx(min(gains))
    assert output.numbers["probe_inc_max"] == pytest.approx(max(gains))


def test_popularity_is_the_strongest_baseline_where_the_caption_says_so() -> (
    None
):
    for model, row in _contrast().items():
        assert row["popularity_within"] > max(row[c] for c in BASELINES), model


def test_caption_is_the_template_filled_from_the_files(
    output: registry.FigureOutput,
) -> None:
    point = _by_role(
        _flags(lambda r: r["logistic_within"] > r["popularity_within"])
    )
    interval = _by_role(_flags(lambda r: _diff(r)[1] > 0))
    lat = [
        f["confidence+lat"]
        for f in _increments().values()
        if "confidence+lat" in f
    ]
    probe = [f["confidence+probe"] for f in _increments().values()]
    diffs = [r["diff_vs_confidence"] for r in lat]
    gains = [r["diff_vs_confidence"] for r in probe]

    def signed(value: float) -> str:
        text = f"{value:.3f}"
        return text.replace("-", MINUS)

    assert all(c.hits == c.total for c in point.values() if c.total)
    assert all(r["lo"] <= 0 <= r["hi"] for r in lat)
    assert all(r["lo"] > 0 for r in probe)
    caption = f"{output.claim} {output.detail}"
    assert output.claim == (
        "A known-against-unknown direction ranks questions but adds nothing "
        "to the model's confidence."
    )
    assert (
        f"on all {len(lat)} checkpoints by point estimate, and the interval "
        f"of the difference lies above zero on {stats.per_role_text(interval)}"
    ) in caption
    assert (
        f"the direction changes it by {signed(min(diffs))} to "
        f"{signed(max(diffs))} with an interval covering zero on all "
        f"{len(lat)} checkpoints, while the probe adds {signed(min(gains))} "
        f"to {signed(max(gains))} with an interval above zero on all "
        f"{len(probe)}"
    ) in caption
    assert "monthly Wikipedia page views" in caption
    assert "Hanley and McNeil" in caption


def test_size_guards_and_layout(output: registry.FigureOutput) -> None:
    fig = output.figure
    assert tuple(fig.get_size_inches()) == (paper.FULL_WIDTH, HEIGHT)
    assert checks.guard_problems(fig) == []
    assert checks.problems(fig) == []


def test_two_panels_share_the_checkpoint_rows(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    assert len(fig.axes) == 2
    first, second = fig.axes
    assert first.get_shared_y_axes().joined(first, second)
    labels = [t.get_text() for t in first.get_yticklabels()]
    assert labels == [paper.display_name(m) for m in _models("p9b")]


def test_the_gpt2_small_row_says_it_has_no_contrast(
    output: registry.FigureOutput,
) -> None:
    texts = [t.get_text() for t in output.figure.axes[0].texts]
    assert "no known-against-unknown contrast" in texts


def test_dot_plot_rows_have_no_connecting_lines(
    output: registry.FigureOutput,
) -> None:
    for ax in output.figure.axes:
        for line in ax.get_lines():
            assert len(line.get_xdata()) <= 2


def test_panel_letters_and_dashes(output: registry.FigureOutput) -> None:
    checks.draw(output.figure)
    letters = [
        t.get_text()
        for ax in output.figure.axes
        for t in ax.texts
        if t.get_text().startswith(r"$\mathbf")
    ]
    assert sorted(letters) == [r"$\mathbf{(a)}$", r"$\mathbf{(b)}$"]
    text = f"{output.claim} {output.detail} {output.csv}"
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text


def test_svg_and_pdf_bytes_are_the_same_for_the_same_input() -> None:
    first = registry.build_figure(SPEC, Manifest(RESULTS))
    second = registry.build_figure(SPEC, Manifest(RESULTS))
    with paper.paper_style():
        assert paper.svg_bytes(first.figure) == paper.svg_bytes(second.figure)
        assert paper.pdf_bytes(first.figure) == paper.pdf_bytes(second.figure)
