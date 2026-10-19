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
SPEC = "a_dose_types"
HEIGHT = 7.2
TYPES = (
    "obscure_real",
    "fictional_generated",
    "unknowable_detail",
    "false_premise",
)
SERIES = ("lat: known", "lat: unknown", "random: unknown")
ARMS = ("random0", "random1", "random2")


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(output.csv)))


def _post_runs() -> list[tuple[str, str]]:
    found = []
    for model in _R["models"]("p9d"):
        if model.startswith("gpt2") or model.endswith("-base"):
            continue
        templates = sorted(
            {
                str(r.config["template"])
                for r in Manifest(RESULTS).runs("p9d", model)
            },
            key=("fewshot", "concept").index,
        )
        found.extend((model, t) for t in templates)
    return found


def _path(model: str, template: str) -> Path:
    path: Path = _R["run"]("p9d", model, template=template)
    return path


def _curves(model: str, template: str, kind: str) -> dict[str, np.ndarray]:
    path = _path(model, template)
    names = _R["data"](path, f"curves_{kind}.json")["series"]
    arrays = _R["npz"](path, f"curves_{kind}.npz")
    return {
        "x": arrays["x"],
        **{
            f"{stat}:{name}": arrays[stat][i]
            for i, name in enumerate(names)
            for stat in ("mean", "lo", "hi")
        },
    }


def _moves_more(model: str, template: str, kind: str) -> bool:
    payload = _R["data"](_path(model, template), "table_dose.json")
    rows: list[dict[str, Any]] = [
        dict(zip(payload["columns"], r, strict=True)) for r in payload["rows"]
    ]
    arms = [
        r
        for r in rows
        if r["type"] == kind
        and r["metric"] == "abstention"
        and r["side"] == "unknown"
        and r["arm"] == "lat0"
        and r["vs"] in ARMS
    ]
    assert len(arms) == len(ARMS)
    return all(r["abs_lo"] > 0 for r in arms)


def _panels(output: registry.FigureOutput) -> list[Any]:
    return [ax for ax in output.figure.axes if ax.get_label() != "<legend>"]


def test_six_post_trained_runs_by_four_question_types(
    output: registry.FigureOutput,
) -> None:
    runs = _post_runs()
    assert len(runs) == 6
    panels = _panels(output)
    assert len(panels) == len(runs) * len(TYPES)
    boxes = [ax.get_position() for ax in panels]
    assert len({round(b.y0, 2) for b in boxes}) == len(runs)
    assert len({round(b.x0, 2) for b in boxes}) == len(TYPES)


def test_every_panel_draws_the_stored_curves(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for letter, run in zip("abcdef", _post_runs(), strict=True):
        for kind in TYPES:
            curves = _curves(*run, kind)
            for series in SERIES:
                got = [
                    r
                    for r in rows
                    if r["panel"] == letter
                    and r["run"] == " ".join(run)
                    and r["type"] == kind
                    and r["series"] == series
                ]
                assert [float(r["x"]) for r in got] == list(curves["x"])
                assert [float(r["value"]) for r in got] == pytest.approx(
                    list(curves[f"mean:{series}"])
                )
                assert [float(r["lo"]) for r in got] == pytest.approx(
                    list(curves[f"lo:{series}"])
                )


def test_the_count_is_the_stored_per_arm_interval_criterion(
    output: registry.FigureOutput,
) -> None:
    flags = [_moves_more(*run, kind) for run in _post_runs() for kind in TYPES]
    assert len(flags) == 24
    assert sum(flags) == 11
    assert output.numbers["panels"] == 24
    assert output.numbers["moves_more"] == 11
    assert output.claim == (
        "The direction moves the abstention rate of unknown questions more "
        "than all 3 random directions in 11 of 24 panels."
    )


def test_all_panels_share_one_vertical_range_that_holds_the_curves(
    output: registry.FigureOutput,
) -> None:
    panels = _panels(output)
    limits = {ax.get_ylim() for ax in panels}
    assert len(limits) == 1
    low, high = limits.pop()
    for run in _post_runs():
        for kind in TYPES:
            curves = _curves(*run, kind)
            for series in SERIES:
                assert low <= curves[f"lo:{series}"].min()
                assert curves[f"hi:{series}"].max() <= high


def test_size_guards_and_layout(output: registry.FigureOutput) -> None:
    fig = output.figure
    assert tuple(fig.get_size_inches()) == (paper.FULL_WIDTH, HEIGHT)
    assert fig.get_size_inches()[1] <= 8.0
    assert checks.guard_problems(fig, appendix=True) == []
    assert checks.problems(fig) == []


def test_one_letter_per_row_and_no_dashes(
    output: registry.FigureOutput,
) -> None:
    letters = [
        t.get_text()
        for ax in output.figure.axes
        for t in ax.texts
        if t.get_text().startswith(r"$\mathbf")
    ]
    assert sorted(letters) == [rf"$\mathbf{{({c})}}$" for c in "abcdef"]
    text = f"{output.claim} {output.detail} {output.csv}"
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text
    assert "_" not in f"{output.claim} {output.detail}".replace(
        "[[fig:knowledge_steering]]", ""
    )


def test_svg_and_pdf_bytes_are_the_same_for_the_same_input() -> None:
    first = registry.build_figure(SPEC, Manifest(RESULTS))
    second = registry.build_figure(SPEC, Manifest(RESULTS))
    with paper.paper_style():
        assert paper.svg_bytes(first.figure) == paper.svg_bytes(second.figure)
        assert paper.pdf_bytes(first.figure) == paper.pdf_bytes(second.figure)


def test_caption_describes_the_grey_band_as_the_mean_of_the_random_arms(
    output: registry.FigureOutput,
) -> None:
    assert (
        "the mean over the 3 random directions on unknown questions with its "
        "95 percent interval that resamples the questions (grey band)"
    ) in output.detail


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
    assert len(gaps) == 1
    assert min(gaps) >= 2.0


def test_no_two_legend_keys_draw_the_same_symbol(
    output: registry.FigureOutput,
) -> None:
    assert _L["marker_clashes"](output.figure) == []


def test_the_encoding_is_cross_referenced_by_token(
    output: registry.FigureOutput,
) -> None:
    assert "[[fig:knowledge_steering]]" in output.detail
    assert "Figure 9" not in output.detail


def test_the_panels_that_pass_are_starred_and_the_bands_are_keyed(
    output: registry.FigureOutput,
) -> None:
    starred = [
        ax
        for ax in _panels(output)
        if any(line.get_marker() == "*" for line in ax.lines)
    ]
    assert len(starred) == output.numbers["moves_more"]
    assert len(starred) > 0
    (legend,) = output.figure.legends
    labels = [t.get_text() for t in legend.get_texts()]
    assert "interval of the direction" in labels
    assert "interval of the random mean" in labels
    assert any("beyond all 3 random directions" in label for label in labels)
