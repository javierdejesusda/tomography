import csv
import io
from pathlib import Path
import runpy
from typing import Any

import numpy as np
import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
_T = runpy.run_path(str(Path(__file__).with_name("_titles.py")))
_L = runpy.run_path(str(Path(__file__).with_name("_legends.py")))
SPEC = "a_dose_runs"
HEIGHT = 5.6
SERIES = ("lat: known", "lat: unknown", "random: unknown")


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(output.csv)))


def _runs() -> list[tuple[str, str]]:
    found = []
    for model in _R["models"]("p9d"):
        templates = sorted(
            {
                str(r.config["template"])
                for r in Manifest(RESULTS).runs("p9d", model)
            },
            key=("fewshot", "concept").index,
        )
        found.extend((model, t) for t in templates)
    return sorted(found, key=lambda run: not _entropy(*run))


def _path(model: str, template: str) -> Path:
    path: Path = _R["run"]("p9d", model, template=template)
    return path


def _entropy(model: str, template: str) -> bool:
    head: dict[str, Any] = _R["data"](_path(model, template), "summary.json")[
        "headline"
    ]
    return bool(head["metric"] == "entropy")


def _curves(model: str, template: str) -> dict[str, np.ndarray]:
    path = _path(model, template)
    stem = (
        "curves_obscure_real_entropy"
        if _entropy(model, template)
        else "curves_obscure_real"
    )
    names = _R["data"](path, f"{stem}.json")["series"]
    arrays = _R["npz"](path, f"{stem}.npz")
    return {
        "x": arrays["x"],
        **{
            f"{kind}:{name}": arrays[kind][i]
            for i, name in enumerate(names)
            for kind in ("mean", "lo", "hi")
        },
    }


def _panels(output: registry.FigureOutput) -> list[Any]:
    return [ax for ax in output.figure.axes if ax.get_label() != "<legend>"]


def test_there_is_one_panel_per_run_in_four_rows_of_three(
    output: registry.FigureOutput,
) -> None:
    runs = _runs()
    assert len(runs) == 12
    panels = _panels(output)
    assert len(panels) == 12
    boxes = [ax.get_position() for ax in panels]
    rows = sorted({round(b.y0, 2) for b in boxes})
    cols = sorted({round(b.x0, 2) for b in boxes})
    assert (len(rows), len(cols)) == (4, 3)


def test_every_panel_draws_the_stored_curves(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for letter, run in zip("abcdefghijkl", _runs(), strict=True):
        curves = _curves(*run)
        for series in SERIES:
            got = [
                r
                for r in rows
                if r["panel"] == letter
                and r["run"] == " ".join(run)
                and r["series"] == series
            ]
            assert [float(r["x"]) for r in got] == list(curves["x"])
            assert [float(r["value"]) for r in got] == pytest.approx(
                list(curves[f"mean:{series}"])
            )


def test_panels_of_one_unit_share_a_vertical_range_that_holds_the_curves(
    output: registry.FigureOutput,
) -> None:
    panels = _panels(output)
    for plain in (True, False):
        chosen = [
            (ax, run)
            for ax, run in zip(panels, _runs(), strict=True)
            if _entropy(*run) == plain
        ]
        limits = {ax.get_ylim() for ax, _ in chosen}
        assert len(limits) == 1
        low, high = limits.pop()
        for _, run in chosen:
            curves = _curves(*run)
            for series in SERIES:
                assert low <= curves[f"lo:{series}"].min()
                assert curves[f"hi:{series}"].max() <= high


def test_caption_names_the_runs_and_the_grey_band(
    output: registry.FigureOutput,
) -> None:
    assert output.claim == (
        "Dose curves of every knowledge-steering run on unknown and known "
        "questions."
    )
    assert (
        "the mean over the 3 random directions on unknown questions with "
        "its 95 percent interval that resamples the questions (grey band)"
    ) in output.detail
    assert "Figure 9" in output.detail
    assert output.numbers["runs"] == 12


def test_size_guards_and_layout(output: registry.FigureOutput) -> None:
    fig = output.figure
    assert tuple(fig.get_size_inches()) == (paper.FULL_WIDTH, HEIGHT)
    assert checks.guard_problems(fig, appendix=True) == []
    assert checks.problems(fig) == []


def test_twelve_panel_letters_and_no_dashes(
    output: registry.FigureOutput,
) -> None:
    letters = [
        t.get_text()
        for ax in output.figure.axes
        for t in ax.texts
        if t.get_text().startswith(r"$\mathbf")
    ]
    assert sorted(letters) == sorted(
        rf"$\mathbf{{({c})}}$" for c in "abcdefghijkl"
    )
    text = f"{output.claim} {output.detail} {output.csv}"
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text


def test_svg_and_pdf_bytes_are_the_same_for_the_same_input() -> None:
    first = registry.build_figure(SPEC, Manifest(RESULTS))
    second = registry.build_figure(SPEC, Manifest(RESULTS))
    with paper.paper_style():
        assert paper.svg_bytes(first.figure) == paper.svg_bytes(second.figure)
        assert paper.pdf_bytes(first.figure) == paper.pdf_bytes(second.figure)


def test_legend_names_the_grey_band_as_the_mean_of_the_random_arms(
    output: registry.FigureOutput,
) -> None:
    legend = output.figure.legends[0]
    labels = [t.get_text() for t in legend.get_texts()]
    assert "mean of 3 random directions" in labels
    assert not any("pooled" in label for label in labels)


def test_letters_do_not_touch_their_titles(
    output: registry.FigureOutput,
) -> None:
    gaps = _T["letter_title_gaps"](output.figure)
    assert len(gaps) == 12
    assert min(gaps) >= 2.0


def test_every_row_has_one_measure_and_each_block_names_it_once(
    output: registry.FigureOutput,
) -> None:
    panels = _panels(output)
    runs = _runs()
    rows = [
        list(zip(panels[i : i + 3], runs[i : i + 3], strict=True))
        for i in (0, 3, 6, 9)
    ]
    for row in rows:
        assert len({_entropy(*run) for _, run in row}) == 1
    names = {True: "entropy shift", False: "abstention rate"}
    for block in (rows[:2], rows[2:]):
        measure = _entropy(*block[0][0][1])
        labels = [
            ax.get_ylabel() for row in block for ax, _ in row if ax.get_ylabel()
        ]
        assert labels == [names[measure]]
    assert [_entropy(*run) for run in runs] == [True] * 6 + [False] * 6


def test_no_two_legend_keys_draw_the_same_symbol(
    output: registry.FigureOutput,
) -> None:
    assert _L["marker_clashes"](output.figure) == []
