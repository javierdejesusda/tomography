import json
from pathlib import Path
import re
import runpy

import numpy as np
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
SPEC = "a_finders_gpt2"
TASKS = ("sentiment", "truth", "utility")
FINDERS = ("pca_diff", "logistic")
CAPTION_START = (
    "Report-split accuracy at every layer of the paper's finder (purple) "
    "and logistic regression (black) on GPT-2"
)
MODELS = [m for m in models("p4") if not m.startswith("qwen3")]
HEIGHT = 4.6
EM = chr(0x2014)
EN = chr(0x2013)


def _curves(model: str, task: str) -> tuple[list[str], dict[str, np.ndarray]]:
    path = run("p4", model)
    meta = json.loads(
        (path / f"curves_{task}_methods.json").read_text("utf-8")
    )["data"]
    with np.load(path / f"curves_{task}_methods.npz") as z:
        arrays = {
            k: np.asarray(z[k], dtype=float) for k in ("mean", "lo", "hi")
        }
    return list(meta["series"]), arrays


def _selected(model: str, task: str, finder: str) -> tuple[int, float]:
    path = run("p4", model)
    reference = data(path, "summary.json")["reference"][task]
    records = data(path, "table.json")
    (row,) = [
        r
        for r in records
        if r["task"] == task
        and r["method"] == finder
        and r["pairing"] == reference["pairing"]
        and r["preprocess"] == "center"
    ]
    return int(row["layer"]), float(row["report"])


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    lines = output.csv.strip().splitlines()
    head = lines[0].split(",")
    assert head == ["task", "model", "series", "depth", "value", "lo", "hi"]
    return [dict(zip(head, ln.split(","), strict=True)) for ln in lines[1:]]


def test_the_grid_is_checkpoints_by_tasks_at_the_planned_size(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    assert len(fig.axes) == len(MODELS) * len(TASKS)
    assert tuple(fig.get_size_inches()) == pytest.approx(
        (paper.FULL_WIDTH, HEIGHT)
    )


def test_the_csv_holds_the_two_finders_and_the_random_curve(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for model in MODELS:
        for task in TASKS:
            series, arrays = _curves(model, task)
            n = arrays["mean"].shape[1]
            for name in (*FINDERS, "random"):
                got = [
                    r
                    for r in rows
                    if r["model"] == model
                    and r["task"] == task
                    and r["series"] == name
                ]
                index = series.index(name)
                assert len(got) == n
                for key, column in (
                    ("value", "mean"),
                    ("lo", "lo"),
                    ("hi", "hi"),
                ):
                    assert [float(r[key]) for r in got] == pytest.approx(
                        list(arrays[column][index]), abs=1e-5
                    )
    assert {r["series"] for r in rows} == {
        *FINDERS,
        "random",
        *(f"selected_{f}" for f in FINDERS),
    }


def test_the_dots_sit_at_the_selected_layers(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for model in MODELS:
        for task in TASKS:
            n = _curves(model, task)[1]["mean"].shape[1]
            for finder in FINDERS:
                layer, report = _selected(model, task, finder)
                (row,) = [
                    r
                    for r in rows
                    if r["model"] == model
                    and r["task"] == task
                    and r["series"] == f"selected_{finder}"
                ]
                assert float(row["depth"]) == pytest.approx(layer / n, abs=1e-5)
                assert float(row["value"]) == pytest.approx(report, abs=1e-5)


def test_each_panel_has_a_seed_band_a_chance_line_and_two_finder_curves(
    output: registry.FigureOutput,
) -> None:
    for ax in output.figure.axes:
        assert len(ax.collections) >= 1
        assert any(
            np.allclose(np.asarray(line.get_ydata(), float), 0.5)
            for line in ax.get_lines()
        )
        colours = {
            line.get_color()
            for line in ax.get_lines()
            if len(line.get_xdata()) > 2
        }
        assert paper.ENC["paper"] in colours
        assert paper.ENC["direction"] in colours


def test_letters_are_per_row_and_ticks_are_on_the_bottom_row_only(
    output: registry.FigureOutput,
) -> None:
    axes = output.figure.axes
    for row in range(len(MODELS)):
        letter = "abcdefgh"[row]
        texts = [t.get_text() for t in axes[row * 3].texts]
        assert any(f"({letter})" in t for t in texts)
    for index, ax in enumerate(axes):
        bottom = index // 3 == len(MODELS) - 1
        labels = [t.get_text() for t in ax.get_xticklabels() if t.get_visible()]
        assert bool(ax.xaxis.get_tick_params()["labelbottom"]) == bottom
        if bottom:
            assert labels


def test_the_caption_matches_the_plan(output: registry.FigureOutput) -> None:
    text = f"{output.claim} {output.detail}"
    assert CAPTION_START in text
    assert "against the random direction of the same run (grey)" in text
    assert "select split chose" in text
    assert "[[table:finders_all]]" in text
    assert EM not in text
    assert EN not in text
    assert "_" not in re.sub(r"\[\[[^\]]*\]\]", "", text)


def test_the_guards_are_green(output: registry.FigureOutput) -> None:
    assert checks.guard_problems(output.figure, appendix=True) == []
    assert checks.problems(output.figure) == []


def test_every_row_has_the_same_axis_style_and_a_visible_chance_line(
    output: registry.FigureOutput,
) -> None:
    widths = set()
    for ax in output.figure.axes:
        for side in ("bottom", "left"):
            assert ax.spines[side].get_visible()
            widths.add(ax.spines[side].get_linewidth())
        chance = [
            ln
            for ln in ax.get_lines()
            if np.allclose(np.asarray(ln.get_ydata(), float), 0.5)
            and len(ln.get_xdata()) == 2
        ]
        assert chance
        assert all(ln.get_linewidth() >= 0.8 for ln in chance)
    assert len(widths) == 1
    assert widths.pop() >= 0.8
