import csv
import io
import json
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
SPEC = "approximate"
CELL = ("free", "fewshot", "range_last", "greedy")
TRAIN = "height+wingspan"
HEIGHT = 3.4


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(output.csv)))


def _models(exp: str) -> list[str]:
    models: list[str] = _R["models"](exp)
    return models


def _read(exp: str, model: str, name: str) -> list[dict[str, Any]]:
    out = []
    for row in _R["csv_rows"](_R["run"](exp, model), name):
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            try:
                parsed[key] = float(value) if value != "" else None
            except ValueError:
                parsed[key] = value
        out.append(parsed)
    return out


def _behavior(model: str) -> dict[str, dict[str, Any]]:
    table = {
        r["kind"]: r
        for r in _read("p10a", model, "table.csv")
        if r["family"] == "all"
        and (r["condition"], r["format"], r["shot_order"], r["decoding"])
        == CELL
    }
    (diag,) = [
        r
        for r in _read("p10a", model, "diagnostic.csv")
        if (r["condition"], r["format"], r["shot_order"], r["decoding"]) == CELL
    ]
    return {
        "range_items": {
            "value": table["range"]["range_rate"],
            "lo": table["range"]["range_lo"],
            "hi": table["range"]["range_hi"],
        },
        "exact_items": {
            "value": table["exact"]["range_rate"],
            "lo": table["exact"]["range_lo"],
            "hi": table["exact"]["range_hi"],
        },
        "people": {
            "value": diag["people_rate"],
            "lo": diag["people_lo"],
            "hi": diag["people_hi"],
        },
    }


def _cell(model: str, method: str) -> dict[str, Any]:
    (row,) = [
        r
        for r in _read("p10b", model, "transfer.csv")
        if r["template"] == "concept"
        and r["method"] == method
        and r["train_set"] == TRAIN
        and r["test_family"] == "duration"
    ]
    return row


def _text(model: str) -> dict[str, Any]:
    (row,) = [
        r
        for r in _read("p10b", model, "baselines.csv")
        if r["baseline"] == "text_word"
        and r["train_set"] == TRAIN
        and r["test_family"] == "duration"
    ]
    return row


def _people(model: str, method: str) -> list[dict[str, Any]]:
    return [
        r
        for r in _read("p10b", model, "people.csv")
        if r["source"] == "reader"
        and r["template"] == "concept"
        and r["method"] == method
    ]


def _position(model: str) -> float:
    (row,) = [r for r in _people(model, "logistic") if r["train_set"] == TRAIN]
    return float(row["position"])


def _percent(value: float) -> int:
    return math.floor(value * 100 + 0.5 + 1e-9)


def _by_role(flags: dict[str, bool]) -> dict[str, stats.RoleCount]:
    return stats.per_role(flags)


def _qwen(rows: dict[str, float], base: bool) -> list[float]:
    return [
        v
        for m, v in rows.items()
        if m.startswith("qwen3") and m.endswith("-base") == base
    ]


def test_panel_a_has_range_exact_and_people_for_every_checkpoint(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for model in _models("p10a"):
        want = _behavior(model)
        for series, cell in want.items():
            (got,) = [
                r
                for r in rows
                if r["panel"] == "a"
                and r["model"] == model
                and r["series"] == series
            ]
            assert float(got["value"]) == pytest.approx(cell["value"])
            assert float(got["lo"]) == pytest.approx(cell["lo"])
            assert float(got["hi"]) == pytest.approx(cell["hi"])


def test_panel_b_has_reader_random_and_one_text_value(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for model in _models("p10b"):
        for series, method in (("reader", "logistic"), ("random", "random")):
            (got,) = [
                r
                for r in rows
                if r["panel"] == "b"
                and r["model"] == model
                and r["series"] == series
            ]
            cell = _cell(model, method)
            assert float(got["value"]) == pytest.approx(cell["auroc"])
            assert float(got["lo"]) == pytest.approx(cell["lo"])
            assert float(got["hi"]) == pytest.approx(cell["hi"])
    words = [r for r in rows if r["panel"] == "b" and r["series"] == "words"]
    assert len(words) == 1
    text = _text("gpt2")
    assert float(words[0]["value"]) == pytest.approx(text["auroc"])
    assert float(words[0]["lo"]) == pytest.approx(text["lo"])
    assert float(words[0]["hi"]) == pytest.approx(text["hi"])
    for model in _models("p10b"):
        assert _text(model)["auroc"] == text["auroc"]


def test_panel_c_has_the_reader_position_and_the_random_range(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for model in _models("p10b"):
        (got,) = [
            r
            for r in rows
            if r["panel"] == "c"
            and r["model"] == model
            and r["series"] == "reader"
        ]
        assert float(got["value"]) == pytest.approx(_position(model))
        (band,) = [
            r
            for r in rows
            if r["panel"] == "c"
            and r["model"] == model
            and r["series"] == "random"
        ]
        draws = _people(model, "random")
        assert len(draws) == 7
        assert float(band["lo"]) == pytest.approx(
            min(r["position"] for r in draws)
        )
        assert float(band["hi"]) == pytest.approx(
            max(r["position"] for r in draws)
        )


def test_numbers_are_recomputed_from_the_files(
    output: registry.FigureOutput,
) -> None:
    models = _models("p10a")
    rate = {m: _behavior(m)["range_items"]["value"] for m in models}
    exact = {m: _behavior(m)["exact_items"]["value"] for m in models}
    people = {m: _behavior(m)["people"]["value"] for m in models}
    numbers = output.numbers
    assert numbers["range_base_min"] == _percent(min(_qwen(rate, True)))
    assert numbers["range_base_max"] == _percent(max(_qwen(rate, True)))
    assert numbers["range_post_min"] == _percent(min(_qwen(rate, False)))
    assert numbers["range_post_max"] == _percent(max(_qwen(rate, False)))
    qwen = [m for m in models if m.startswith("qwen3")]
    assert numbers["exact_qwen_max"] == _percent(max(exact[m] for m in qwen))
    assert numbers["people_qwen_max"] == _percent(max(people[m] for m in qwen))
    assert numbers["gpt2_range"] == _percent(rate["gpt2"])
    assert numbers["gpt2_exact"] == _percent(exact["gpt2"])
    text = _text("gpt2")
    assert numbers["text_word"] == pytest.approx(text["auroc"])
    assert numbers["text_lo"] == pytest.approx(text["lo"])
    assert numbers["text_hi"] == pytest.approx(text["hi"])
    positions = [_position(m) for m in _models("p10b")]
    assert numbers["position_min"] == pytest.approx(min(positions))
    assert numbers["position_max"] == pytest.approx(max(positions))
    reader = {m: _cell(m, "logistic") for m in _models("p10b")}
    above = _by_role({m: c["auroc"] > text["auroc"] for m, c in reader.items()})
    inside = _by_role(
        {
            m: _cell(m, "random")["lo"]
            <= c["auroc"]
            <= _cell(m, "random")["hi"]
            for m, c in reader.items()
        }
    )
    outside = _by_role(
        {
            m: not (
                min(r["position"] for r in _people(m, "random"))
                <= _position(m)
                <= max(r["position"] for r in _people(m, "random"))
            )
            for m in _models("p10b")
        }
    )
    for name, want in (
        ("above_text", above),
        ("reader_in_random", inside),
        ("people_outside_band", outside),
    ):
        for role in stats.ROLES:
            assert numbers[f"{name}_{role}"] == want[role].hits
            got = output.counts[name][role]
            assert (got.hits, got.total) == (want[role].hits, want[role].total)
    assert all(
        c["hi"] >= text["lo"] and c["lo"] <= text["hi"] for c in reader.values()
    )
    assert numbers["n_people"] == 120
    config = json.loads(
        (_R["run"]("p10b", "gpt2") / "config.json").read_text(encoding="utf-8")
    )
    assert numbers["n_random_draws"] == config["n_seeds"]


def test_percent_values_match_the_plan_at_this_commit(
    output: registry.FigureOutput,
) -> None:
    numbers = output.numbers
    assert (numbers["range_base_min"], numbers["range_base_max"]) == (83, 99)
    assert (numbers["range_post_min"], numbers["range_post_max"]) == (28, 68)
    assert numbers["exact_qwen_max"] == 15
    assert numbers["people_qwen_max"] == 0


def test_caption_is_the_template_filled_from_the_files(
    output: registry.FigureOutput,
) -> None:
    n = output.numbers
    caption = f"{output.claim} {output.detail}"
    assert output.claim == (
        "Qwen3 writes ranges where the sourced answer is a range, but a "
        "range reader beats its random control only on some checkpoints."
    )
    assert (
        f"Qwen3 base checkpoints write a range on {int(n['range_base_min'])} "
        f"to {int(n['range_base_max'])} percent of the range items and "
        f"post-trained ones on {int(n['range_post_min'])} to "
        f"{int(n['range_post_max'])} percent, on at most "
        f"{int(n['exact_qwen_max'])} percent of the exact items and on none "
        "of the heights"
    ) in caption
    assert (
        f"GPT-2 small writes a range on {int(n['gpt2_range'])} percent of "
        f"the range items and {int(n['gpt2_exact'])} percent of the exact "
        "items"
    ) in caption
    above = stats.per_role_text(
        {r: output.counts["above_text"][r] for r in stats.ROLES}
    )
    outside = stats.per_role_text(
        {r: output.counts["people_outside_band"][r] for r in stats.ROLES}
    )
    assert f"The reader is above the classifier's value on {above}" in caption
    assert f"outside the random band on {outside}" in caption
    assert "reaches 0.9" not in caption
    assert f"{n['text_word']:.3f}" in caption
    assert f"{n['text_lo']:.3f} to {n['text_hi']:.3f}" in caption
    assert (
        f"at {n['position_min']:.3f} to {n['position_max']:.3f}, next to the "
        "breeds"
    ) in caption


def test_size_guards_and_layout(output: registry.FigureOutput) -> None:
    fig = output.figure
    assert tuple(fig.get_size_inches()) == (paper.FULL_WIDTH, HEIGHT)
    assert checks.guard_problems(fig) == []
    assert checks.problems(fig) == []


def test_patches_are_the_text_band_and_the_random_bars_only(
    output: registry.FigureOutput,
) -> None:
    a, b, c = output.figure.axes
    assert len(a.patches) == 0
    assert len(b.patches) == 1
    assert len(c.patches) == len(_models("p10b"))


def test_three_panels_share_rows_and_carry_letters(
    output: registry.FigureOutput,
) -> None:
    a, b, c = output.figure.axes
    assert a.get_shared_y_axes().joined(a, b)
    assert a.get_shared_y_axes().joined(a, c)
    checks.draw(output.figure)
    letters = [
        t.get_text()
        for ax in output.figure.axes
        for t in ax.texts
        if t.get_text().startswith(r"$\mathbf")
    ]
    assert sorted(letters) == [rf"$\mathbf{{({x})}}$" for x in "abc"]
    text = f"{output.claim} {output.detail} {output.csv}"
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text


def test_svg_and_pdf_bytes_are_the_same_for_the_same_input() -> None:
    first = registry.build_figure(SPEC, Manifest(RESULTS))
    second = registry.build_figure(SPEC, Manifest(RESULTS))
    with paper.paper_style():
        assert paper.svg_bytes(first.figure) == paper.svg_bytes(second.figure)
        assert paper.pdf_bytes(first.figure) == paper.pdf_bytes(second.figure)


def test_no_two_legend_keys_share_a_marker_and_fill(
    output: registry.FigureOutput,
) -> None:
    (legend,) = output.figure.legends
    seen: dict[tuple[str, str, str], str] = {}
    for handle, text in zip(
        legend.legend_handles, legend.get_texts(), strict=True
    ):
        key = (
            str(handle.get_marker()),
            str(handle.get_markerfacecolor()),
            str(handle.get_markeredgecolor()),
        )
        if key[0] in ("None", ""):
            continue
        assert key not in seen, (seen[key], text.get_text())
        seen[key] = text.get_text()


def test_the_reader_is_drawn_with_the_marker_its_legend_key_shows(
    output: registry.FigureOutput,
) -> None:
    (legend,) = output.figure.legends
    labels = {
        t.get_text(): h
        for h, t in zip(legend.legend_handles, legend.get_texts(), strict=True)
    }
    reader = labels["range reader"].get_marker()
    _, ax_b, ax_c = output.figure.axes
    for ax in (ax_b, ax_c):
        assert any(line.get_marker() == reader for line in ax.lines)


def test_control_counts_are_per_role_and_use_the_stored_values(
    output: registry.FigureOutput,
) -> None:
    models = _models("p10b")
    want = {
        "reader_above_random": {
            m: _cell(m, "logistic")["auroc"] > _cell(m, "random")["hi"]
            for m in models
        },
        "control_below_half": {
            m: _cell(m, "random")["auroc"] < 0.5 for m in models
        },
        "control_above_95": {
            m: _cell(m, "random")["auroc"] > 0.95 for m in models
        },
    }
    for name, flags in want.items():
        counts = _by_role(flags)
        for role in stats.ROLES:
            got = output.counts[name][role]
            assert (got.hits, got.total) == (
                counts[role].hits,
                counts[role].total,
            )
            assert output.numbers[f"{name}_{role}"] == counts[role].hits
    draws = [r["position"] for m in models for r in _people(m, "random")]
    assert output.numbers["band_min"] == pytest.approx(min(draws))
    assert output.numbers["band_max"] == pytest.approx(max(draws))


def test_caption_defines_the_control_interval_and_the_clipped_bands(
    output: registry.FigureOutput,
) -> None:
    n = output.numbers
    for name in (
        "reader_above_random",
        "control_below_half",
        "control_above_95",
    ):
        text = stats.per_role_text(
            {r: output.counts[name][r] for r in stats.ROLES}
        )
        assert text in output.detail
    assert "95 percent bootstrap interval of the mean over the" in output.detail
    assert "above the random control's interval" in output.detail
    assert "below 0.5" in output.detail
    assert "above 0.95" in output.detail
    low = f"{n['band_min']:.1f}".replace("-", chr(0x2212))
    assert (
        f"the random bands span {low} to {n['band_max']:.1f}" in output.detail
    )
    assert "cannot separate the reader from random" in output.detail
    assert "inside the random control's interval" not in output.detail


def test_clipped_bands_are_marked_on_panel_c(
    output: registry.FigureOutput,
) -> None:
    ax_c = output.figure.axes[2]
    assert "band clipped" in ax_c.get_xlabel()
    markers = {line.get_marker() for line in ax_c.lines}
    assert {"<", ">"} & markers


def test_the_reader_is_the_direction_circle_and_no_symbol_repeats(
    output: registry.FigureOutput,
) -> None:
    (legend,) = output.figure.legends
    markers = {
        t.get_text(): h.get_marker()
        for h, t in zip(legend.legend_handles, legend.get_texts(), strict=True)
    }
    assert markers["range reader"] == paper.MEASURE_MARKERS["direction"]
    assert markers["random direction"] == paper.MEASURE_MARKERS["random"]
    assert markers["GPT-2 and base"] == paper.ROLE_KEY_MARKER
    labelled = [m for m in markers.values() if m not in ("None", "")]
    assert labelled.count("o") == 1
