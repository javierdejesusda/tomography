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
    return registry.build_figure("a_reading_curves", Manifest(RESULTS))


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


def test_the_guards_are_green(output: registry.FigureOutput) -> None:
    assert checks.guard_problems(output.figure) == []
    assert checks.problems(output.figure) == []


def test_the_caption_avoids_first_and_points_to_the_main_figure(
    output: registry.FigureOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    assert "gpt-2" not in text
    assert "first" not in text.lower()
    assert "[[fig:reading]]" in text
    assert "darker lines are larger checkpoints" in text
    assert "95 percent interval over seeds" in text


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


def test_the_grid_is_three_role_rows_by_four_concepts_and_has_no_size_row(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    assert not fig.findobj(AxesImage)
    width, height = fig.get_size_inches()
    assert width == pytest.approx(paper.FULL_WIDTH)
    assert height == pytest.approx(4.6)
    assert len(_axes(output, "curve_")) == 12
    assert not _axes(output, "size_")
    letters = sorted(
        t.get_text()
        for ax in fig.axes
        for t in ax.texts
        if t.get_text().startswith(r"$\mathbf{(")
    )
    assert len(letters) == 4


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


def test_the_text_is_at_least_the_glossary_floor_and_never_bold(
    output: registry.FigureOutput,
) -> None:
    for t in _fig_texts(output.figure):
        assert t.get_fontsize() >= paper.MIN_TEXT, t.get_text()
        assert t.get_fontweight() in ("normal", 400), t.get_text()


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
