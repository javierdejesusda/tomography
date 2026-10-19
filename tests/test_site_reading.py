import math
from pathlib import Path
import runpy
from typing import Any

from matplotlib.collections import LineCollection
from matplotlib.collections import PolyCollection
from matplotlib.image import AxesImage
from matplotlib.text import Text
import numpy as np
import pytest
from scipy import stats as scipy_stats

from tomography.site import params
from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

_RESULTS = runpy.run_path(str(Path(__file__).with_name("_results.py")))
data = _RESULTS["data"]
models = _RESULTS["models"]
npz = _RESULTS["npz"]
run = _RESULTS["run"]

RESULTS = Path(__file__).resolve().parents[1] / "results"
Z = 1.959963984540054
CONCEPTS = {
    "sentiment": "p4",
    "utility": "r2",
    "morality": "r3",
    "truth": "r7",
}
GPT2 = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")


def _wilson(p: float, n: int) -> tuple[float, float]:
    denom = 1 + Z**2 / n
    centre = (p + Z**2 / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def _curves(exp: str, model: str) -> tuple[np.ndarray, np.ndarray]:
    path = run(exp, model)
    if exp == "p4":
        series = data(path, "curves_sentiment_methods.json")["series"]
        mean = npz(path, "curves_sentiment_methods.npz")["mean"]
        return mean[series.index("logistic")], mean[series.index("random")]
    series = data(path, "curves.json")["series"]
    mean = npz(path, "curves.npz")["mean"]
    return mean[series.index("reader")], mean[series.index("random")]


def _p4_layer(model: str) -> int:
    """Reads the layer the finder comparison chose, from its table."""
    rows = data(run("p4", model), "table.json")
    (row,) = [
        r
        for r in rows
        if r["task"] == "sentiment"
        and r["method"] == "logistic"
        and r["pairing"] == "label_contrastive"
        and r["preprocess"] == "center"
    ]
    return int(row["layer"])


def _curve_layer(concept: str, model: str) -> int:
    """Gives the layer at which the curve dot of a concept is drawn."""
    if concept == "sentiment":
        return _p4_layer(model)
    return _read_at(concept, model)[0]


def _read_at(concept: str, model: str) -> tuple[int, float, float, int]:
    exp = CONCEPTS[concept]
    path = run(exp if concept != "sentiment" else "p3", model)
    if concept == "sentiment":
        meta = data(path, "sizes_sentiment.json")
        arrays = npz(path, "sizes_sentiment.npz")
        k = int(np.flatnonzero(arrays["sizes"] == 64)[0])
        layer = int(meta["layer"])
        col = int(np.flatnonzero(arrays["layers"] == layer)[0])
        acc = arrays["accuracy"]
        return (
            layer,
            float(acc[k, 1, 0, :, col].mean()),
            float(acc[k, 1, 1, :, col].mean()),
            214,
        )
    summary = data(path, "summary.json")
    _, random = _curves(exp, model)
    if concept == "truth":
        layer = int(summary["val_layer"])
        test = summary["sets"]["test"]
        return (
            layer,
            float(test["accuracy_at_val_layer"]),
            float(random[layer - 1]),
            int(test["n_eval_pairs"]),
        )
    layer = int(summary["best_layer"])
    return (
        layer,
        float(summary["test_accuracy"]),
        float(random[layer - 1]),
        int(summary["n_eval_pairs"]["test"]),
    )


def _onset(concept: str, model: str) -> float | None:
    reader, random = _curves(CONCEPTS[concept], model)
    above = np.flatnonzero(reader - random > 0.1)
    return float((above[0] + 1) / len(reader)) if above.size else None


def _depths(concept: str) -> list[float]:
    values = [_onset(concept, m) for m in models(CONCEPTS[concept])]
    return [v for v in values if v is not None]


def _slope(concept: str) -> tuple[float, float, float]:
    x = [math.log10(params.param_count(m)) for m in GPT2]
    y = [_read_at(concept, m)[1] for m in GPT2]
    fit = scipy_stats.linregress(x, y)
    half = float(scipy_stats.t.ppf(0.975, 2)) * float(fit.stderr)
    return float(fit.slope), fit.slope - half, fit.slope + half


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure("reading", Manifest(RESULTS))


def _records(output: registry.FigureOutput, panel: str) -> list[dict[str, str]]:
    lines = output.csv.strip().splitlines()
    head = lines[0].split(",")
    assert head == [
        "panel",
        "concept",
        "model",
        "x",
        "y",
        "lo",
        "hi",
        "baseline",
    ]
    rows = [dict(zip(head, line.split(","), strict=True)) for line in lines[1:]]
    return [r for r in rows if r["panel"] == panel]


def test_every_depth_cell_is_the_reader_minus_random_curve(
    output: registry.FigureOutput,
) -> None:
    rows = _records(output, "depth")
    total = 0
    for concept, exp in CONCEPTS.items():
        for model in models(exp):
            reader, random = _curves(exp, model)
            mine = [
                r
                for r in rows
                if r["concept"] == concept and r["model"] == model
            ]
            assert len(mine) == len(reader)
            total += len(mine)
            for i, rec in enumerate(mine):
                assert float(rec["y"]) == pytest.approx(
                    reader[i] - random[i], abs=1e-9
                )
                assert float(rec["x"]) == pytest.approx(
                    (i + 1) / len(reader), abs=1e-9
                )
    assert total == len(rows)


def test_every_size_point_is_recomputed_with_its_wilson_interval(
    output: registry.FigureOutput,
) -> None:
    rows = _records(output, "size")
    assert len(rows) == sum(len(models(e)) for e in CONCEPTS.values())
    for rec in rows:
        _, acc, random, n = _read_at(rec["concept"], rec["model"])
        lo, hi = _wilson(acc, n)
        assert float(rec["y"]) == pytest.approx(acc, abs=1e-9)
        assert float(rec["lo"]) == pytest.approx(lo, abs=1e-9)
        assert float(rec["hi"]) == pytest.approx(hi, abs=1e-9)
        assert float(rec["baseline"]) == pytest.approx(random, abs=1e-9)
        assert float(rec["x"]) == params.param_count(rec["model"])


def test_the_sentiment_row_is_the_held_out_value_not_the_r1_value() -> None:
    differs = 0
    for model in models("p3"):
        _, acc, _, _ = _read_at("sentiment", model)
        best = data(run("r1", model), "summary.json")["best_accuracy"]
        differs += abs(acc - best) > 1e-6
    assert differs >= 1


def test_the_onset_depths_are_recomputed_for_all_forty_cells(
    output: registry.FigureOutput,
) -> None:
    cells = 0
    for concept in CONCEPTS:
        depths = _depths(concept)
        cells += len(depths)
        assert output.numbers[f"onset_{concept}_min"] == pytest.approx(
            min(depths)
        )
        assert output.numbers[f"onset_{concept}_max"] == pytest.approx(
            max(depths)
        )
    assert cells == 40


def test_the_gpt2_slopes_use_a_t_interval_on_four_points(
    output: registry.FigureOutput,
) -> None:
    for concept in CONCEPTS:
        slope, lo, hi = _slope(concept)
        assert output.numbers[f"slope_{concept}"] == pytest.approx(slope)
        assert output.numbers[f"slope_{concept}_lo"] == pytest.approx(lo)
        assert output.numbers[f"slope_{concept}_hi"] == pytest.approx(hi)
    assert output.numbers["truth_n"] == 59


def test_the_plan_values_hold_at_the_current_results() -> None:
    assert (
        round(min(_depths("sentiment")), 3),
        round(max(_depths("sentiment")), 3),
    ) == (0.021, 0.167)
    assert (
        round(min(_depths("utility")), 3),
        round(max(_depths("utility")), 3),
    ) == (0.25, 0.417)
    assert round(_slope("utility")[0], 3) == 0.034
    assert round(_slope("morality")[0], 3) == 0.132
    assert round(_slope("sentiment")[0], 3) == 0.048
    assert round(_slope("truth")[0], 3) == pytest.approx(-0.027)


def test_the_caption_quotes_the_onsets_slopes_and_sizes(
    output: registry.FigureOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    for concept in CONCEPTS:
        assert f"{output.numbers[f'onset_{concept}_min']:.3f}" in text
        assert f"{output.numbers[f'onset_{concept}_max']:.3f}" in text
    assert "0.167" in text
    assert "214 sentiment pairs" in text
    assert "4,808 utility, 983 morality and 59 truth pairs" in text
    assert "per tenfold" in text
    assert "include zero" in text


def test_the_guards_are_green(output: registry.FigureOutput) -> None:
    assert checks.guard_problems(output.figure) == []
    assert checks.problems(output.figure) == []


def test_the_caption_keeps_names_and_grammar_and_avoids_first(
    output: registry.FigureOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    assert "gpt-2" not in text
    assert "GPT-2 sizes" in text
    assert "intervals that includes" not in text
    assert "first" not in text.lower()


def test_the_truth_onset_is_called_a_crossing_on_its_pair_count(
    output: registry.FigureOutput,
) -> None:
    text = output.detail
    n = int(output.numbers["truth_n"])
    assert n == 59
    assert "earliest layer at which the margin curve crosses 0.1" in text
    assert f"truth the curve rests on {n} truth pairs" in text


ROLE_ROWS = ("gpt2", "base", "post")
SIZE_ORDER = {
    "gpt2": ["S", "M", "L", "XL"],
    "base": ["0.6B", "1.7B", "4B"],
    "post": ["0.6B", "1.7B", "4B"],
}


def _role(model: str) -> str:
    if not model.startswith("qwen3"):
        return "gpt2"
    return "base" if model.endswith("-base") else "post"


def _axes(output: registry.FigureOutput, prefix: str) -> dict[str, object]:
    return {
        ax.get_gid(): ax
        for ax in output.figure.axes
        if (ax.get_gid() or "").startswith(prefix)
    }


def test_the_heat_maps_and_the_colour_bar_are_gone_and_the_grid_is_four_by_four(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    assert not fig.findobj(AxesImage)
    width, height = fig.get_size_inches()
    assert width == pytest.approx(paper.FULL_WIDTH)
    assert height == pytest.approx(6.0)
    assert len(_axes(output, "curve_")) == 12
    assert len(_axes(output, "size_")) == 4
    assert not [a for a in fig.axes if a.get_label() == "<colorbar>"]


def test_each_role_row_holds_only_its_own_checkpoints_never_pooled(
    output: registry.FigureOutput,
) -> None:
    for concept, exp in CONCEPTS.items():
        for role in ROLE_ROWS:
            ax = _axes(output, f"curve_{role}_{concept}")[
                f"curve_{role}_{concept}"
            ]
            drawn = {
                ln.get_gid(): np.asarray(ln.get_ydata())
                for ln in ax.get_lines()
                if (ln.get_gid() or "").startswith("curve:")
            }
            want = [m for m in models(exp) if _role(m) == role]
            assert sorted(drawn) == sorted(f"curve:{m}" for m in want)
            for m in want:
                reader, random = _curves(exp, m)
                assert drawn[f"curve:{m}"] == pytest.approx(reader - random)
                assert ax.get_ylim() == pytest.approx((-0.1, 0.5))


def test_every_curve_panel_has_a_band_a_zero_line_and_a_dot_per_checkpoint(
    output: registry.FigureOutput,
) -> None:
    for concept, exp in CONCEPTS.items():
        for role in ROLE_ROWS:
            gid = f"curve_{role}_{concept}"
            ax = _axes(output, gid)[gid]
            bands = [
                c
                for c in ax.collections
                if isinstance(c, PolyCollection)
                and c.get_gid() == "random_band"
            ]
            assert len(bands) == 1
            dots = [
                ln
                for ln in ax.get_lines()
                if (ln.get_gid() or "").startswith("read:")
            ]
            want = [m for m in models(exp) if _role(m) == role]
            assert len(dots) == len(want)
            for ln in dots:
                model = ln.get_gid().removeprefix("read:")
                layer = _curve_layer(concept, model)
                reader, random = _curves(exp, model)
                assert ln.get_xdata()[0] == pytest.approx(layer / len(reader))
                assert ln.get_ydata()[0] == pytest.approx(
                    reader[layer - 1] - random[layer - 1]
                )
            zero = [ln for ln in ax.get_lines() if ln.get_gid() == "zero"]
            assert len(zero) == 1


def test_the_onset_is_a_tick_at_the_earliest_layer_above_the_margin(
    output: registry.FigureOutput,
) -> None:
    for concept, exp in CONCEPTS.items():
        for role in ROLE_ROWS:
            gid = f"curve_{role}_{concept}"
            ax = _axes(output, gid)[gid]
            ticks = [
                c
                for c in ax.collections
                if isinstance(c, LineCollection) and c.get_gid() == "onset"
            ]
            want = sorted(
                d
                for m in models(exp)
                if _role(m) == role and (d := _onset(concept, m)) is not None
            )
            got = sorted(
                float(seg[0][0]) for c in ticks for seg in c.get_segments()
            )
            assert got == pytest.approx(want)


def test_the_end_labels_sit_in_the_label_column_in_size_order(
    output: registry.FigureOutput,
) -> None:
    labels = _axes(output, "labels_")
    assert sorted(labels) == ["labels_base", "labels_gpt2", "labels_post"]
    for role, ax in labels.items():
        texts = [t.get_text() for t in ax.texts if t.get_gid() == "end_label"]
        assert texts == SIZE_ORDER[role.removeprefix("labels_")]
    for gid, ax in _axes(output, "curve_").items():
        assert not [t for t in ax.texts if t.get_gid() == "end_label"], gid


def test_the_size_row_has_a_reader_whisker_a_random_tick_and_no_trend_line(
    output: registry.FigureOutput,
) -> None:
    for concept, exp in CONCEPTS.items():
        ax = _axes(output, f"size_{concept}")[f"size_{concept}"]
        readers = [
            ln
            for ln in ax.get_lines()
            if (ln.get_gid() or "").startswith("reader:")
        ]
        assert len(readers) == len(models(exp))
        for ln in readers:
            model = ln.get_gid().removeprefix("reader:")
            _, acc, _, _ = _read_at(concept, model)
            assert ln.get_ydata()[0] == pytest.approx(acc)
            filled = _role(model) != "post"
            assert (ln.get_markerfacecolor() != "white") == filled
        randoms = [
            ln
            for ln in ax.get_lines()
            if (ln.get_gid() or "").startswith("random:")
        ]
        assert len(randoms) == len(models(exp))
        assert not [ln for ln in ax.get_lines() if ln.get_gid() == "trend"]
        assert ax.get_ylim() == pytest.approx((0.4, 1.0))


def test_the_detail_describes_the_curves_the_band_the_onset_and_the_size_row(
    output: registry.FigureOutput,
) -> None:
    text = output.detail
    assert "one line per checkpoint" in text
    assert "grey band" in text
    assert "95 percent interval over seeds" in text
    assert "tick on the x axis" in text
    assert "darker lines are larger checkpoints" in text
    assert "no fitted line" in text
    assert "sentiment curves" in text.lower()
    assert "does not include the sampling error" in text


def _fig_texts(fig: Any) -> list[Any]:
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]


def test_the_curve_rows_share_one_quantity_label_and_titled_roles(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    (shared,) = [t for t in fig.texts if t.get_gid() == "curve_ylabel"]
    assert shared.get_text() == "reader minus random accuracy"
    weights = set()
    for role, name in (
        ("gpt2", "GPT-2"),
        ("base", "Qwen3 base"),
        ("post", "Qwen3 post-trained"),
    ):
        (ax,) = [
            a for a in fig.axes if a.get_gid() == f"curve_{role}_sentiment"
        ]
        assert ax.get_ylabel() == name
        assert ax.yaxis.label.get_fontsize() >= 8
        weights.add(ax.yaxis.label.get_fontweight())
    assert len(weights) == 1


def test_the_key_uses_circles_like_the_data_not_hexagons(
    output: registry.FigureOutput,
) -> None:
    (legend,) = output.figure.legends
    texts = [t.get_text() for t in legend.get_texts()]
    assert any("GPT-2 and Qwen3 base" in x for x in texts)
    assert any("post-trained" in x for x in texts)
    marks = {h.get_marker() for h in legend.legend_handles}
    assert "h" not in marks
    assert "o" in marks
    fills = {
        h.get_markerfacecolor() == "white"
        for h in legend.legend_handles
        if h.get_marker() == "o"
    }
    assert fills == {True, False}


def test_every_size_panel_has_one_full_width_chance_line(
    output: registry.FigureOutput,
) -> None:
    for concept in CONCEPTS:
        (ax,) = [
            a for a in output.figure.axes if a.get_gid() == f"size_{concept}"
        ]
        lines = [ln for ln in ax.get_lines() if ln.get_gid() == "chance"]
        assert len(lines) == 1
        assert list(lines[0].get_ydata()) == [0.5, 0.5]
        assert list(lines[0].get_xdata()) == pytest.approx(list(ax.get_xlim()))


def test_the_size_axis_holds_every_whisker_end(
    output: registry.FigureOutput,
) -> None:
    for concept in CONCEPTS:
        (ax,) = [
            a for a in output.figure.axes if a.get_gid() == f"size_{concept}"
        ]
        low, high = ax.get_ylim()
        for ln in ax.get_lines():
            if (ln.get_gid() or "").startswith("whisker:"):
                assert low <= min(ln.get_ydata())
                assert max(ln.get_ydata()) <= high
    (ax,) = [a for a in output.figure.axes if a.get_gid() == "size_truth"]
    assert ax.get_ylim()[0] <= 0.44


def test_the_text_is_at_least_the_glossary_floor_and_never_bold(
    output: registry.FigureOutput,
) -> None:
    for t in _fig_texts(output.figure):
        assert t.get_fontsize() >= paper.MIN_TEXT, t.get_text()
        assert t.get_fontweight() in ("normal", 400), t.get_text()


def test_the_caption_says_where_the_sentiment_size_dot_is_read(
    output: registry.FigureOutput,
) -> None:
    assert "its own layer" in output.detail
    assert "for sentiment" in output.detail.lower()


@pytest.mark.parametrize("model", ["gpt2-xl", "qwen3-0.6b-base"])
def test_the_sentiment_curve_dot_sits_at_the_finder_comparison_layer(
    output: registry.FigureOutput, model: str
) -> None:
    layer = _p4_layer(model)
    assert layer != _read_at("sentiment", model)[0]
    role = _role(model)
    ax = _axes(output, f"curve_{role}_sentiment")[f"curve_{role}_sentiment"]
    (dot,) = [ln for ln in ax.get_lines() if ln.get_gid() == f"read:{model}"]
    reader, random = _curves("p4", model)
    assert dot.get_xdata()[0] == pytest.approx(layer / len(reader))
    assert dot.get_ydata()[0] == pytest.approx(
        reader[layer - 1] - random[layer - 1]
    )
    rows = [
        r
        for r in _records(output, "read")
        if r["concept"] == "sentiment" and r["model"] == model
    ]
    (row,) = rows
    assert float(row["x"]) == pytest.approx(layer / len(reader))
    assert float(row["y"]) == pytest.approx(
        reader[layer - 1] - random[layer - 1]
    )
