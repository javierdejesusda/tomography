import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

_RESULTS = runpy.run_path(str(Path(__file__).with_name("_results.py")))
models = _RESULTS["models"]
run = _RESULTS["run"]

RESULTS = Path(__file__).resolve().parents[1] / "results"
CONCEPTS = ("r1", "r2", "r3", "r7")
TITLES = ("sentiment", "utility", "morality", "truth")
GROUPS = ("gpt2", "base", "post")
GRID = np.linspace(0.0, 1.0, 101)
LABELS = {
    "gpt2": "S",
    "gpt2-medium": "M",
    "gpt2-large": "L",
    "gpt2-xl": "XL",
    "qwen3-0.6b-base": "0.6B",
    "qwen3-0.6b": "0.6B",
    "qwen3-1.7b-base": "1.7B",
    "qwen3-1.7b": "1.7B",
    "qwen3-4b-base": "4B",
    "qwen3-4b": "4B",
}
EM = chr(0x2014)
EN = chr(0x2013)


def _group(model: str) -> str:
    if not model.startswith("qwen3"):
        return "gpt2"
    return "base" if model.endswith("-base") else "post"


def _curves(exp: str, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = run(exp, model)
    series = json.loads((path / "curves.json").read_text("utf-8"))["data"][
        "series"
    ]
    with np.load(path / "curves.npz") as z:
        mean = np.asarray(z["mean"], dtype=float)
    reader, random = mean[series.index("reader")], mean[series.index("random")]
    depth = np.arange(1, len(reader) + 1) / len(reader)
    return depth, reader, random


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure("a_layers", Manifest(RESULTS))


def _rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    lines = output.csv.strip().splitlines()
    head = lines[0].split(",")
    assert head == [
        "panel",
        "concept",
        "group",
        "model",
        "series",
        "depth",
        "value",
    ]
    return [dict(zip(head, ln.split(","), strict=True)) for ln in lines[1:]]


def test_the_figure_is_twelve_panels_at_the_planned_size(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    assert len(fig.axes) == 12
    assert tuple(fig.get_size_inches()) == pytest.approx(
        (paper.FULL_WIDTH, 6.0)
    )


def test_the_csv_has_no_shuffled_series_and_every_reader_value(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    assert {r["series"] for r in rows} == {
        "reader",
        "random",
        "band_lo",
        "band_hi",
    }
    assert "shuffled" not in output.csv
    for exp, title in zip(CONCEPTS, TITLES, strict=True):
        for model in models(exp):
            depth, reader, random = _curves(exp, model)
            for name, values in (("reader", reader), ("random", random)):
                got = [
                    r
                    for r in rows
                    if r["concept"] == title
                    and r["model"] == model
                    and r["series"] == name
                ]
                assert len(got) == len(depth)
                assert [float(r["depth"]) for r in got] == pytest.approx(
                    list(depth), abs=1e-5
                )
                assert [float(r["value"]) for r in got] == pytest.approx(
                    list(values), abs=1e-5
                )


def test_the_grey_band_is_the_range_of_the_random_curves_per_panel(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for exp, title in zip(CONCEPTS, TITLES, strict=True):
        for group in GROUPS:
            members = [m for m in models(exp) if _group(m) == group]
            curves = [
                np.interp(GRID, d, r)
                for d, _, r in (_curves(exp, m) for m in members)
            ]
            for name, expected in (
                ("band_lo", np.min(curves, axis=0)),
                ("band_hi", np.max(curves, axis=0)),
            ):
                got = [
                    float(r["value"])
                    for r in rows
                    if r["concept"] == title
                    and r["group"] == group
                    and r["series"] == name
                ]
                assert got == pytest.approx(list(expected), abs=1e-5)


def test_every_panel_draws_one_grey_band_and_the_chance_line(
    output: registry.FigureOutput,
) -> None:
    for ax in output.figure.axes:
        bands = [
            c
            for c in ax.collections
            if any(
                np.allclose(f[:3], _rgb(paper.RANDOM_BAND))
                for f in c.get_facecolor()
            )
        ]
        assert len(bands) == 1
        assert any(
            np.allclose(np.asarray(line.get_ydata(), float), 0.5)
            for line in ax.get_lines()
        )


def _rgb(colour: str) -> np.ndarray:
    return np.array(
        [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)], dtype=float
    )


def test_each_row_has_a_letter_and_each_line_is_labelled_at_its_end(
    output: registry.FigureOutput,
) -> None:
    axes = output.figure.axes
    for row, letter in enumerate("abcd"):
        texts = [t.get_text() for t in axes[row * 3].texts]
        assert any(f"({letter})" in t for t in texts)
    for index, ax in enumerate(axes):
        exp, group = CONCEPTS[index // 3], GROUPS[index % 3]
        texts = {t.get_text() for t in ax.texts}
        for model in models(exp):
            if _group(model) == group:
                assert LABELS[model] in texts


def test_the_caption_matches_the_plan(output: registry.FigureOutput) -> None:
    text = f"{output.claim} {output.detail}"
    assert (
        "against the range of the random directions of the same panel" in text
    )
    assert "256 SST-2 validation pairs" in text
    assert "shuffled" not in text.lower()
    assert EM not in text
    assert EN not in text


def test_the_guards_are_green(output: registry.FigureOutput) -> None:
    assert checks.guard_problems(output.figure, appendix=True) == []
    assert checks.problems(output.figure) == []
