import csv
import io
import json
import math
from pathlib import Path
import runpy
from typing import Any

from matplotlib import colors as mpl_colors
import pytest

from tomography.site import registry
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.viz import axes
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
SPEC = "approximate"
CELL = ("free", "fewshot", "range_last", "greedy")
FORCED = ("forced", "fewshot", "range_last", "greedy")
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
    for pooled in ("exact_qwen_max", "people_qwen_max", "wilson_max_points"):
        assert pooled not in numbers, pooled
    for role in stats.ROLES:
        own = [m for m in models if stats.role(m) == role]
        assert numbers[f"exact_{role}_max"] == _percent(
            max(exact[m] for m in own)
        )
        assert numbers[f"people_{role}_max"] == _percent(
            max(people[m] for m in own)
        )
    assert numbers["gpt2_range"] == _percent(rate["gpt2"])
    assert numbers["gpt2_exact"] == _percent(exact["gpt2"])
    text = _text("gpt2")
    assert numbers["text_word"] == pytest.approx(text["auroc"])
    assert numbers["text_lo"] == pytest.approx(text["lo"])
    assert numbers["text_hi"] == pytest.approx(text["hi"])
    positions = [_position(m) for m in _models("p10b")]
    for pooled in ("position_min", "position_max", "band_min", "band_max"):
        assert pooled not in numbers, pooled
    for role in stats.ROLES:
        own = [
            p
            for m, p in zip(_models("p10b"), positions, strict=True)
            if stats.role(m) == role
        ]
        assert numbers[f"position_{role}_min"] == pytest.approx(min(own))
        assert numbers[f"position_{role}_max"] == pytest.approx(max(own))
    reader = {m: _cell(m, "logistic") for m in _models("p10b")}
    above = _by_role({m: c["auroc"] > text["auroc"] for m, c in reader.items()})
    for name, want in (("above_text", above),):
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
    assert numbers["exact_base_max"] == 15
    assert numbers["exact_post_max"] == 0
    assert numbers["people_base_max"] == 0
    assert numbers["people_post_max"] == 0


def _forced(model: str) -> float:
    (row,) = [
        r
        for r in _read("p10a", model, "table.csv")
        if r["family"] == "all"
        and r["kind"] == "range"
        and (r["condition"], r["format"], r["shot_order"], r["decoding"])
        == FORCED
    ]
    return float(row["range_rate"])


def test_caption_is_at_most_200_words_and_leads_with_the_conclusion(
    output: registry.FigureOutput,
) -> None:
    caption = f"{output.claim} {output.detail}"
    assert len(caption.split()) <= 200, len(caption.split())
    assert output.claim.startswith("Qwen3 writes a range for "), output.claim


def test_the_claim_gives_the_range_shares_and_the_control_count_per_role(
    output: registry.FigureOutput,
) -> None:
    n = output.numbers
    above = stats.per_role_text(output.counts["reader_above_random"])
    assert output.claim == (
        f"Qwen3 writes a range for {int(n['range_base_min'])} to "
        f"{int(n['range_base_max'])} percent of range items (base) and "
        f"{int(n['range_post_min'])} to {int(n['range_post_max'])} percent "
        "(post-trained), and a range reader is above the interval of "
        f"its random control on {above}."
    )


def test_the_detail_gives_the_baselines_of_each_panel(
    output: registry.FigureOutput,
) -> None:
    n = output.numbers
    detail = output.detail
    assert (
        f"at most {int(n['exact_base_max'])} and "
        f"{int(n['people_base_max'])} percent (Qwen3 base), "
        f"{int(n['exact_post_max'])} and {int(n['people_post_max'])} "
        "percent (post-trained)"
    ) in detail
    assert (
        f"GPT-2 small {int(n['gpt2_range'])} and "
        f"{int(n['gpt2_exact'])} percent on range and exact items"
    ) in detail
    assert f"{n['text_word']:.3f}" in detail
    assert f"{n['text_lo']:.3f} to {n['text_hi']:.3f}" in detail
    assert f"mean of {int(n['n_random_draws'])} random directions" in detail
    assert "[[sec:appendix_approximate]]" in detail
    for gone in ("sign flip", "Hanley", "cannot separate"):
        assert gone not in detail


def test_the_wilson_intervals_lie_inside_the_markers_and_the_caption_says_so(
    output: registry.FigureOutput,
) -> None:
    widths = [
        float(r["hi"]) - float(r["lo"])
        for r in _rows(output)
        if r["panel"] == "a"
    ]
    widest = max(widths)
    ax = output.figure.axes[0]
    output.figure.canvas.draw()
    points = ax.get_window_extent().width / output.figure.dpi * 72
    low, high = ax.get_xlim()
    assert widest / (high - low) * points < paper.MARKER_SIZE
    by_role: dict[str, float] = {}
    for r in _rows(output):
        if r["panel"] == "a":
            role = stats.role(r["model"])
            width = float(r["hi"]) - float(r["lo"])
            by_role[role] = max(by_role.get(role, 0.0), width)
    wide = {}
    for role in stats.ROLES:
        wide[role] = math.ceil(by_role[role] * 1000 - 1e-9) / 10
        assert output.numbers[f"wilson_{role}_points"] == wide[role]
        assert wide[role] >= by_role[role] * 100
    assert (
        "Wilson 95 percent intervals lie inside the markers, at most "
        f"{wide['gpt2']:.1f} (GPT-2), {wide['base']:.1f} (Qwen3 base) and "
        f"{wide['post']:.1f} (post-trained) points wide"
    ) in output.detail


def test_the_forced_range_shares_are_exported_per_role(
    output: registry.FigureOutput,
) -> None:
    qwen = [m for m in _models("p10a") if m.startswith("qwen3")]
    for role, base in (("base", True), ("post", False)):
        values = [_forced(m) for m in qwen if m.endswith("-base") == base]
        assert output.numbers[f"forced_{role}_max"] == _percent(max(values))
    assert output.numbers["forced_gpt2_max"] == _percent(
        max(_forced(m) for m in _models("p10a") if m.startswith("gpt2"))
    )


def test_the_people_outside_the_random_band_are_counted_per_role(
    output: registry.FigureOutput,
) -> None:
    want = _by_role(
        {
            m: not (
                min(r["position"] for r in _people(m, "random"))
                <= _position(m)
                <= max(r["position"] for r in _people(m, "random"))
            )
            for m in _models("p10b")
        }
    )
    for role in stats.ROLES:
        got = output.counts["outside"][role]
        assert (got.hits, got.total) == (want[role].hits, want[role].total)


def test_size_guards_and_layout(output: registry.FigureOutput) -> None:
    fig = output.figure
    assert tuple(fig.get_size_inches()) == (paper.FULL_WIDTH, HEIGHT)
    assert checks.guard_problems(fig) == []
    assert checks.problems(fig) == []


def test_svg_and_pdf_bytes_are_the_same_for_the_same_input() -> None:
    first = registry.build_figure(SPEC, Manifest(RESULTS))
    second = registry.build_figure(SPEC, Manifest(RESULTS))
    with paper.paper_style():
        assert paper.svg_bytes(first.figure) == paper.svg_bytes(second.figure)
        assert paper.pdf_bytes(first.figure) == paper.pdf_bytes(second.figure)


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
    for role in stats.ROLES:
        draws = [
            r["position"]
            for m in models
            if stats.role(m) == role
            for r in _people(m, "random")
        ]
        assert output.numbers[f"band_{role}_min"] == pytest.approx(min(draws))
        assert output.numbers[f"band_{role}_max"] == pytest.approx(max(draws))


def test_the_superseded_reader_and_band_counts_are_gone(
    output: registry.FigureOutput,
) -> None:
    assert "reader_in_random" not in output.counts
    assert "people_outside_band" not in output.counts
    for role in stats.ROLES:
        assert f"reader_in_random_{role}" not in output.numbers
        assert f"people_outside_band_{role}" not in output.numbers


def _state(model: str) -> str:
    reader, control = _cell(model, "logistic"), _cell(model, "random")
    if reader["auroc"] > control["hi"]:
        return "above"
    if reader["auroc"] < control["lo"]:
        return "below"
    return "inside"


def _by_gid(ax: Any, prefix: str) -> list[Any]:
    return [ln for ln in ax.get_lines() if ln.get_gid() == prefix]


def test_five_panels_share_rows_and_carry_three_letters(
    output: registry.FigureOutput,
) -> None:
    axes_ = output.figure.axes
    assert len(axes_) == 5
    for ax in axes_[1:]:
        assert axes_[0].get_shared_y_axes().joined(axes_[0], ax)
    checks.draw(output.figure)
    letters = [
        t.get_text()
        for ax in axes_
        for t in ax.texts
        if t.get_text().startswith(r"$\mathbf")
    ]
    assert sorted(letters) == [rf"$\mathbf{{({x})}}$" for x in "abc"]
    text = f"{output.claim} {output.detail} {output.csv}"
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text


def test_panel_a_is_three_item_columns_of_whole_circles(
    output: registry.FigureOutput,
) -> None:
    columns = output.figure.axes[:3]
    heads = [
        " ".join(t.get_text() for t in ax.texts if "mathbf" not in t.get_text())
        for ax in columns
    ]
    assert [h.replace("\n", " ") for h in heads] == [
        "range items",
        "exact items",
        "people's heights",
    ]
    for ax in columns:
        markers = {str(ln.get_marker()) for ln in ax.get_lines()}
        assert markers - {"None", " ", ""} <= {"o"}
        points = [ln for ln in ax.get_lines() if ln.get_marker() == "o"]
        assert len(points) == len(_models("p10a"))
        assert not any(ln.get_clip_on() for ln in ax.get_lines())
        low, high = ax.get_xlim()
        assert low < 0 < 1 < high


def test_panel_a_fill_carries_the_role_in_the_family_colour(
    output: registry.FigureOutput,
) -> None:
    ax = output.figure.axes[0]
    models = _models("p10a")
    points = [ln for ln in ax.get_lines() if ln.get_marker() == "o"]
    for model, line in zip(models, points, strict=True):
        colour = axes.model_style(model).color
        post = stats.role(model) == "post"
        assert line.get_markeredgecolor() == colour
        assert (line.get_markerfacecolor() == "white") is post


def test_panel_b_marker_state_follows_the_interval_rule(
    output: registry.FigureOutput,
) -> None:
    ax = output.figure.axes[3]
    for model in _models("p10b"):
        (line,) = _by_gid(ax, f"reader-{model}")
        state = _state(model)
        assert line.get_marker() == "h"
        filled = line.get_markerfacecolor() == paper.INK
        assert filled is (state == "above")
        if state != "above":
            assert line.get_markeredgecolor() == paper.RANDOM_GREY
        ticks = _by_gid(ax, f"below-tick-{model}")
        assert len(ticks) == (1 if state == "below" else 0)


def test_panel_b_control_is_a_grey_bar_on_the_readers_row(
    output: registry.FigureOutput,
) -> None:
    ax = output.figure.axes[3]
    for model in _models("p10b"):
        control = _cell(model, "random")
        (bar,) = [p for p in ax.patches if p.get_gid() == f"control-{model}"]
        assert bar.get_x() == pytest.approx(control["lo"])
        assert bar.get_x() + bar.get_width() == pytest.approx(control["hi"])
        (reader,) = _by_gid(ax, f"reader-{model}")
        y_bar = bar.get_y() + bar.get_height() / 2
        assert y_bar == pytest.approx(reader.get_ydata()[0])


def test_panel_b_words_classifier_is_a_line_and_a_top_whisker_not_a_band(
    output: registry.FigureOutput,
) -> None:
    ax = output.figure.axes[3]
    text = _text("gpt2")
    assert all(str(p.get_gid()).startswith("control-") for p in ax.patches)
    (line,) = _by_gid(ax, "words-line")
    assert line.get_xdata()[0] == pytest.approx(text["auroc"])
    assert line.get_color() == paper.TEXT_GREEN
    (whisker,) = _by_gid(ax, "words-whisker")
    assert min(whisker.get_xdata()) == pytest.approx(text["lo"])
    assert max(whisker.get_xdata()) == pytest.approx(text["hi"])
    assert whisker.get_ydata()[0] < 0
    names = {t.get_text() for t in ax.texts}
    assert {"words", "chance"} <= names


def test_panel_b_draws_every_control_above_its_marker(
    output: registry.FigureOutput,
) -> None:
    ax = output.figure.axes[3]
    for model in _models("p10b"):
        (marker,) = _by_gid(ax, f"reader-{model}")
        (bar,) = [p for p in ax.patches if p.get_gid() == f"control-{model}"]
        (tick,) = _by_gid(ax, f"control-tick-{model}")
        assert bar.get_zorder() > marker.get_zorder(), model
        assert tick.get_zorder() > marker.get_zorder(), model


def test_panel_b_control_tick_stands_out_of_the_marker(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    checks.draw(fig)
    ax = fig.axes[3]
    radius = _rows_marker_radius(fig)
    for model in _models("p10b"):
        (tick,) = _by_gid(ax, f"control-tick-{model}")
        box = tick.get_window_extent()
        assert box.height / 2 > radius, model


def _rows_marker_radius(fig: Any) -> float:
    from tomography.site.figs import _rows

    return _rows.MARKER_SIZE / 2 * fig.dpi / 72


def test_panel_b_has_no_ceiling_annotation_and_a_clear_words_label(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    checks.draw(fig)
    ax = fig.axes[3]
    assert not [t for t in ax.texts if "control at" in t.get_text()]
    (label,) = [t for t in ax.texts if t.get_text() == "words"]
    box = label.get_window_extent()
    (whisker,) = _by_gid(ax, "words-whisker")
    wbox = whisker.get_window_extent()
    apart = box.y0 - wbox.y1 >= 2 or wbox.y0 - box.y1 >= 2
    assert apart or box.x1 + 2 <= wbox.x0 or wbox.x1 + 2 <= box.x0


def test_panel_c_states_no_range_over_all_roles(
    output: registry.FigureOutput,
) -> None:
    ax_c = output.figure.axes[4]
    label = ax_c.get_xlabel()
    assert "clipped" in label
    assert " to " not in label
    assert "band:" not in label
    arrows = [t for t in ax_c.texts if t.get_text() == ""]
    assert arrows


def test_the_legends_have_three_entries_a_role_key_and_a_state_key(
    output: registry.FigureOutput,
) -> None:
    main, states = output.figure.legends
    labels = [t.get_text() for t in main.get_texts()]
    assert labels == [
        "random control",
        "word classifier",
        "(a, c) GPT-2 and Qwen3 base",
        "(a, c) post-trained",
    ]
    assert [t.get_text() for t in states.get_texts()] == [
        "above the control",
        "inside the control",
        "below the control",
    ]
    handles = dict(zip(labels, main.legend_handles, strict=True))
    assert handles["(a, c) post-trained"].get_marker() == "o"
    assert handles["(a, c) GPT-2 and Qwen3 base"].get_marker() == "o"
    markers = {
        str(h.get_marker())
        for h in states.legend_handles
        if hasattr(h, "get_marker")
    }
    assert markers - {"None"} == {"h"}


def test_only_glossary_shapes_and_inks_are_drawn(
    output: registry.FigureOutput,
) -> None:
    allowed = {"o", "h", "None", " ", ""}
    fig = output.figure
    drawn = {str(ln.get_marker()) for ax in fig.axes for ln in ax.get_lines()}
    for legend in fig.legends:
        drawn |= {
            str(h.get_marker())
            for h in legend.legend_handles
            if hasattr(h, "get_marker")
        }
    assert drawn <= allowed
    inks = {
        paper.INK,
        paper.RANDOM_GREY,
        paper.RANDOM_BAND,
        paper.TEXT_GREEN,
        paper.CHANCE,
        paper.WHITE,
        *paper.ENC["gpt2"],
        *paper.ENC["qwen3"],
        paper.ENC["secondary_text"],
    }
    colours = {
        mpl_colors.to_hex(ln.get_color())
        for ax in fig.axes
        for ln in ax.get_lines()
    }
    assert colours <= {mpl_colors.to_hex(c) for c in inks}


def test_caption_states_the_state_rule_and_the_roles(
    output: registry.FigureOutput,
) -> None:
    text = output.detail
    assert "a filled hexagon is above the control's interval" in text
    assert "an open hexagon inside it" in text
    assert "a tick below it" in text


def test_panel_a_columns_label_only_zero_and_one(
    output: registry.FigureOutput,
) -> None:
    for ax in output.figure.axes[:3]:
        assert list(ax.get_xticks()) == [0.0, 1.0]


def test_the_random_band_is_stated_as_the_range_over_the_training_sets(
    output: registry.FigureOutput,
) -> None:
    sets = {len(_people(m, "random")) for m in _models("p10b")}
    assert sets == {7}
    assert output.numbers["n_band_sets"] == 7
    assert (
        "the range of the control's position over its 7 training sets"
        in output.detail
    )


def test_control_layer_counts_are_per_role(
    output: registry.FigureOutput,
) -> None:
    flags = {
        m: _cell(m, "random")["layer"] != _cell(m, "logistic")["layer"]
        for m in _models("p10b")
    }
    counts = _by_role(flags)
    for role in stats.ROLES:
        got = output.counts["control_other_layer"][role]
        assert (got.hits, got.total) == (counts[role].hits, counts[role].total)
        assert output.numbers[f"control_other_layer_{role}"] == (
            counts[role].hits
        )


def test_an_open_state_marker_shows_the_control_bar_through_it(
    output: registry.FigureOutput,
) -> None:
    ax = output.figure.axes[3]
    for model in _models("p10b"):
        (line,) = _by_gid(ax, f"reader-{model}")
        if _state(model) != "above":
            assert line.get_markerfacecolor() == "none", model
