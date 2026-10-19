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


def _ordered() -> list[tuple[str, str]]:
    runs = _post_runs()
    passes = [sum(_moves_more(*run, kind) for kind in TYPES) for run in runs]
    order = sorted(range(len(runs)), key=lambda i: -passes[i])
    return [runs[i] for i in order]


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
    for letter, run in zip("abcdef", _ordered(), strict=True):
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


def _grid(output: registry.FigureOutput) -> list[list[Any]]:
    panels = sorted(
        _panels(output),
        key=lambda a: (-round(a.get_position().y0, 2), a.get_position().x0),
    )
    return [panels[i : i + 4] for i in range(0, 24, 4)]


def test_passing_panels_are_framed_in_the_run_colour_and_no_star_is_drawn(
    output: registry.FigureOutput,
) -> None:
    from tomography.viz import axes

    grid = _grid(output)
    framed = 0
    for row, run in zip(grid, _ordered(), strict=True):
        color = axes.model_style(run[0]).color
        for ax, kind in zip(row, sorted(TYPES), strict=True):
            frames = [p for p in ax.patches if p.get_gid() == "pass-frame"]
            passed = _moves_more(*run, kind)
            assert len(frames) == int(passed)
            if passed:
                assert frames[0].get_edgecolor()[:3] == pytest.approx(
                    mpl_rgb(color)
                )
                assert frames[0].get_linewidth() >= 1.5
            framed += passed
            assert not [ln for ln in ax.lines if ln.get_marker() == "*"]
    assert framed == output.numbers["moves_more"] == 11


def mpl_rgb(color: str) -> tuple[float, float, float]:
    import matplotlib.colors as mcolors

    return mcolors.to_rgb(color)


def test_rows_are_ordered_by_how_many_panels_pass(
    output: registry.FigureOutput,
) -> None:
    counts = [
        sum(_moves_more(*run, kind) for kind in TYPES) for run in _ordered()
    ]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]
    runs_by_letter = {r["panel"]: r["run"] for r in _rows(output)}
    assert [runs_by_letter[c] for c in "abcdef"] == [
        " ".join(run) for run in _ordered()
    ]


def test_random_is_one_grey_and_the_direction_band_has_the_run_colour(
    output: registry.FigureOutput,
) -> None:
    from matplotlib.collections import PolyCollection

    from tomography.viz import axes

    grid = _grid(output)
    for row, run in zip(grid, _ordered(), strict=True):
        color = mpl_rgb(axes.model_style(run[0]).color)
        for ax in row:
            bands = [c for c in ax.collections if isinstance(c, PolyCollection)]
            faces = [tuple(c.get_facecolor()[0][:3]) for c in bands]
            grey = mpl_rgb(paper.RANDOM_GREY)
            assert sum(f == pytest.approx(grey) for f in faces) == 1
            assert sum(f == pytest.approx(color) for f in faces) == 2
            lines = [
                ln.get_color()
                for ln in ax.lines
                if ln.get_color() in (paper.RANDOM_GREY, paper.RANDOM_BAND)
            ]
            assert set(lines) == {paper.RANDOM_GREY}


def test_the_caption_says_what_the_frame_and_the_test_are(
    output: registry.FigureOutput,
) -> None:
    detail = output.detail
    assert "framed" in detail
    assert "star" not in detail
    assert "uses each of the 3 random directions" in detail
    assert "plot shows only their mean" in detail
    assert "probability at most 1/4" in detail
    assert "ordered by the number of framed panels" in detail
    (legend,) = output.figure.legends
    labels = [t.get_text() for t in legend.get_texts()]
    assert "interval of the direction" in labels
    assert "interval of the random mean" in labels
    assert any("framed" in label for label in labels)
    assert not any("star" in label for label in labels)


def test_the_marker_keys_are_open_like_the_post_trained_data(
    output: registry.FigureOutput,
) -> None:
    (legend,) = output.figure.legends
    marked = [
        h
        for h in legend.legend_handles
        if hasattr(h, "get_marker") and h.get_marker() not in ("None", "")
    ]
    assert len(marked) == 2
    assert {h.get_markerfacecolor() for h in marked} == {"white"}


def test_the_caption_reads_without_the_main_figure(
    output: registry.FigureOutput,
) -> None:
    assert "Open markers and dashed lines are post-trained" in output.detail
    assert "The encoding is that of" not in output.detail


def test_the_pass_frame_lies_wholly_inside_its_panel(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    fig.canvas.draw()
    pixels = fig.dpi / 72
    checked = 0
    for ax in _panels(output):
        for frame in ax.patches:
            if frame.get_gid() != "pass-frame":
                continue
            half = frame.get_linewidth() * pixels / 2
            outer = frame.get_window_extent().expanded(
                1 + 2 * half / frame.get_window_extent().width,
                1 + 2 * half / frame.get_window_extent().height,
            )
            box = ax.get_window_extent()
            inside = (
                outer.x0 >= box.x0 - 0.5,
                outer.x1 <= box.x1 + 0.5,
                outer.y0 >= box.y0 - 0.5,
                outer.y1 <= box.y1 + 0.5,
            )
            assert all(inside)
            checked += 1
    assert checked == 11
