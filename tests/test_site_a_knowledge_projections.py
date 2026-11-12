import csv
import io
from pathlib import Path
import re
import runpy
from typing import Any

from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patches import StepPatch
import numpy as np
import pytest

from tomography.site import registry
from tomography.site import stats
from tomography.site.figs import a_knowledge_projections
from tomography.site.manifest import Manifest
from tomography.viz import axes
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
_L = runpy.run_path(str(Path(__file__).with_name("_legends.py")))
SPEC = "a_knowledge_projections"
SHOWN = ("gpt2-xl", "qwen3-1.7b-base", "qwen3-1.7b")
BLOCKS = {"entity": "entity", "question": "question", "unknown": "unknown type"}
LIMIT = 3.0
BINS = 18
HALF = 0.42
FOREST = len(SHOWN)


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(output.csv)))


def _sidecar(model: str) -> dict[str, Any]:
    found: dict[str, Any] = _R["data"](
        _R["run"]("v1", model), "projections.json"
    )
    return found


def _drawn(model: str) -> dict[str, str]:
    """Maps each contrast a figure draws to its panel."""
    return {
        name: panel
        for panel, entry in _sidecar(model)["panels"].items()
        for name in entry["types"]
    }


def _order() -> list[str]:
    """Lists the drawn contrasts of the shown checkpoints, panel by panel."""
    found: dict[str, None] = {}
    for panel in BLOCKS:
        for model in SHOWN:
            entry = _sidecar(model)["panels"].get(panel, {"types": []})
            found.update(dict.fromkeys(entry["types"]))
    return list(found)


def _sides(model: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Recomputes the share of each side in each bin of the z-score."""
    stored = _R["npz"](_R["run"]("v1", model), "projections.npz")
    scores = np.asarray(stored[f"projection_{name}"], dtype=np.float64)
    first = len(_sidecar(model)["uids"][name]["positive"])
    z = np.clip((scores - scores.mean()) / scores.std(), -LIMIT, LIMIT)
    edges = np.linspace(-LIMIT, LIMIT, BINS + 1)
    top, _ = np.histogram(z[:first], bins=edges)
    bottom, _ = np.histogram(z[first:], bins=edges)
    return top / top.sum(), bottom / bottom.sum()


def _auroc(model: str, name: str) -> dict[str, float]:
    path = _R["run"]("v1", model)
    stored = _R["npz"](path, "projections.npz")
    sidecar = _sidecar(model)
    t = sidecar["types"].index(name)
    k = (
        stored["layers"]
        .tolist()
        .index(_R["data"](path, "summary.json")["layer"])
    )
    random = stored["test"][sidecar["methods"].index("random"), t, :, k]
    return {
        "reader": float(stored["published"][t, k]),
        "random": float(random.mean()),
        "random_low": float(random.min()),
        "random_high": float(random.max()),
        "shuffled": float(stored["shuffled"][t, :, k].mean()),
    }


def _draws(model: str) -> int:
    stored = _R["npz"](_R["run"]("v1", model), "projections.npz")
    return int(stored["shuffled"].shape[1])


def _positions(output: registry.FigureOutput) -> dict[str, float]:
    first = output.figure.axes[0]
    labels = [t.get_text() for t in first.get_yticklabels()]
    return dict(zip(labels, first.get_yticks(), strict=True))


def _by_gid(ax: Any, kind: type, gid: str) -> Any:
    (found,) = [
        a for a in ax.findobj(kind) if a.get_gid() == gid and a.axes is ax
    ]
    return found


def test_rows_are_the_drawn_contrasts_in_three_named_blocks(
    output: registry.FigureOutput,
) -> None:
    order = _order()
    assert len(order) == 13
    positions = _positions(output)
    assert list(positions) == [name.replace("_", " ") for name in order]
    places = list(positions.values())
    assert places == sorted(places)
    first = output.figure.axes[0]
    blocks = [t.get_text() for t in first.texts if t.get_gid() == "block"]
    assert blocks == list(BLOCKS.values())
    for ax in output.figure.axes[1 : FOREST + 1]:
        assert first.get_shared_y_axes().joined(first, ax)


@pytest.mark.parametrize("column", range(len(SHOWN)))
def test_each_cell_is_a_back_to_back_histogram_of_the_z_scores(
    output: registry.FigureOutput, column: int
) -> None:
    model = SHOWN[column]
    ax = output.figure.axes[column]
    positions = _positions(output)
    drawn = _drawn(model)
    for name in _order():
        gids = (f"first:{name}", f"second:{name}")
        if name not in drawn:
            for gid in gids:
                assert [
                    p for p in ax.findobj(StepPatch) if p.get_gid() == gid
                ] == []
            continue
        base = positions[name.replace("_", " ")]
        top, bottom = _sides(model, name)
        scale = HALF / max(top.max(), bottom.max())
        upper = _by_gid(ax, StepPatch, gids[0]).get_data()
        lower = _by_gid(ax, StepPatch, gids[1]).get_data()
        assert upper.baseline == lower.baseline == base
        assert upper.values == pytest.approx(base - top * scale)
        assert lower.values == pytest.approx(base + bottom * scale)
        assert upper.edges == pytest.approx(
            np.linspace(-LIMIT, LIMIT, BINS + 1)
        )


@pytest.mark.parametrize("column", range(len(SHOWN)))
def test_hue_is_the_class_and_the_side_is_the_group_named_first(
    output: registry.FigureOutput, column: int
) -> None:
    ax = output.figure.axes[column]
    for name, panel in _drawn(SHOWN[column]).items():
        upper = _by_gid(ax, StepPatch, f"first:{name}")
        lower = _by_gid(ax, StepPatch, f"second:{name}")
        first = "unknown" if panel == "unknown" else "known"
        assert mcolors.to_hex(upper.get_facecolor()) == (
            paper.ENC[first].lower()
        )
        assert mcolors.to_hex(lower.get_facecolor()) == (
            paper.ENC["unknown"].lower()
        )
    assert ax.get_xlim() == (-LIMIT - 0.2, LIMIT + 0.2)
    assert ax.get_xlabel() == (
        "projection on\nthe reader (z-score\nwithin the cell)"
    )
    assert ax.get_title(loc="left") == paper.display_name(SHOWN[column])


def test_the_last_panel_sets_each_reader_beside_its_two_controls(
    output: registry.FigureOutput,
) -> None:
    ax = output.figure.axes[FOREST]
    positions = _positions(output)
    assert ax.get_xlabel() == "AUROC"
    means = [
        min(found["shuffled"], found["random"])
        for model in SHOWN
        for name in _drawn(model)
        for found in [_auroc(model, name)]
    ]
    low, high = ax.get_xlim()
    assert high == 1.02
    assert min(means) - 0.05 <= low < min(means)
    assert low <= 0.4
    assert output.numbers["forest_left"] == low
    lanes: dict[str, list[float]] = {}
    for model in SHOWN:
        style = axes.model_style(model)
        for name in _drawn(model):
            want = _auroc(model, name)
            key = f"{model}:{name}"
            reader = _by_gid(ax, Line2D, f"reader:{key}")
            lane = float(reader.get_ydata()[0])
            lanes.setdefault(name, []).append(lane)
            assert float(reader.get_xdata()[0]) == pytest.approx(want["reader"])
            assert reader.get_marker() == paper.MEASURE_MARKERS["direction"]
            assert reader.get_markeredgecolor() == style.color
            assert (reader.get_markerfacecolor() == "white") == (
                not style.filled
            )
            patch = _by_gid(ax, Rectangle, f"random:{key}")
            band = patch.get_bbox()
            assert band.x0 == pytest.approx(want["random_low"])
            assert band.x1 == pytest.approx(want["random_high"])
            assert band.y0 < lane < band.y1
            halo = _by_gid(ax, Line2D, f"halo:{key}")
            assert halo.get_xydata().tolist() == reader.get_xydata().tolist()
            assert halo.get_marker() == reader.get_marker()
            assert halo.get_markerfacecolor() == paper.WHITE
            assert halo.get_markeredgecolor() == paper.WHITE
            assert halo.get_markersize() >= reader.get_markersize() + 2.0
            assert patch.get_zorder() < halo.get_zorder() < reader.get_zorder()
            mean = _by_gid(ax, Line2D, f"random mean:{key}")
            assert float(mean.get_xdata()[0]) == pytest.approx(want["random"])
            assert mean.get_marker() == paper.MEASURE_MARKERS["random"]
            assert mean.get_color() == paper.RANDOM_GREY
            tick = _by_gid(ax, Line2D, f"shuffled:{key}")
            assert float(tick.get_xdata()[0]) == pytest.approx(want["shuffled"])
            assert tick.get_marker() == "|"
            assert tick.get_color() == paper.ENC["shuffled"]
            assert tick.get_zorder() < halo.get_zorder()
            assert float(tick.get_ydata()[0]) == lane
    for name, found in lanes.items():
        assert found == sorted(found)
        centre = positions[name.replace("_", " ")]
        assert min(found) < centre < max(found) or len(found) == 1


def test_the_claim_counts_the_readers_above_all_their_random_draws(
    output: registry.FigureOutput,
) -> None:
    parts = []
    readers, randoms = [], []
    for model in SHOWN:
        found = [_auroc(model, name) for name in _drawn(model)]
        above = sum(a["reader"] > a["random_high"] for a in found)
        role = stats.role(model)
        assert output.numbers[f"above_{role}"] == above
        assert output.numbers[f"contrasts_{role}"] == len(found)
        parts.append(
            f"{above} of {len(found)} drawn contrasts on "
            f"{paper.display_name(model)}"
        )
        readers += [a["reader"] for a in found]
        randoms += [a["random_high"] for a in found]
    assert output.numbers["reader_min"] == pytest.approx(min(readers))
    assert output.numbers["random_max"] == pytest.approx(max(randoms))
    draws = _draws(SHOWN[0])
    assert output.numbers["draws"] == draws == 5
    assert output.claim == (
        f"The all-pairs reader has a higher AUROC than each of its {draws} "
        f"random directions on {stats.join_and(parts)}; the lowest drawn "
        f"reader is at {min(readers):.2f} and the highest single random "
        f"direction at {max(randoms):.2f}."
    )


def test_the_figure_names_what_is_not_drawn_and_every_exception(
    output: registry.FigureOutput,
) -> None:
    undrawn: dict[str, None] = {}
    lowest = []
    for model in _R["models"]("v1"):
        left = [n for n in _sidecar(model)["types"] if n not in _drawn(model)]
        undrawn.update(dict.fromkeys(left))
        if model in SHOWN:
            lowest += [_auroc(model, name)["reader"] for name in left]
    assert list(undrawn) == ["obscure_real_pop", "triviaqa_known"]
    assert output.labels["undrawn"] == "obscure real pop and triviaqa known"
    assert output.numbers["undrawn_reader_min"] == pytest.approx(min(lowest))
    assert min(lowest) < output.numbers["reader_min"]
    below = {
        model: [
            name
            for name in _drawn(model)
            if _auroc(model, name)["reader"]
            <= _auroc(model, name)["random_high"]
        ]
        for model in _R["models"]("v1")
    }
    below = {model: names for model, names in below.items() if names}
    assert below == {"gpt2-medium": ["fictional_generated", "obscure_real"]}
    assert output.labels["exceptions"] == "GPT-2 medium"
    assert output.labels["exception_contrasts"] == (
        "fictional generated and obscure real"
    )
    for name in below["gpt2-medium"]:
        found = _auroc("gpt2-medium", name)
        assert (
            f"{name.replace('_', ' ')} ({found['random_high']:.3f} against "
            f"{found['reader']:.3f})"
        ) in output.detail


def test_checkpoints_are_counted_per_role(
    output: registry.FigureOutput,
) -> None:
    models = _R["models"]("v1")
    every = {m: m in models for m in _R["MANIFEST"].checkpoints()}
    assert output.counts["with_run"] == stats.per_role(every)
    above = {
        model: all(
            _auroc(model, name)["reader"] > _auroc(model, name)["random_high"]
            for name in _drawn(model)
        )
        for model in models
    }
    assert above["gpt2-xl"]
    assert not above["gpt2-medium"]
    assert output.counts["above_every_draw"] == stats.per_role(above)
    assert output.numbers["per_side"] == 300
    assert output.numbers["bins"] == BINS
    assert output.numbers["limit"] == LIMIT


def test_size_guards_and_layout(output: registry.FigureOutput) -> None:
    fig = output.figure
    assert tuple(fig.get_size_inches()) == (
        paper.FULL_WIDTH,
        a_knowledge_projections.HEIGHT,
    )
    assert checks.guard_problems(fig, appendix=True) == []
    assert checks.problems(fig) == []
    assert _L["marker_clashes"](fig) == []


def test_svg_and_pdf_bytes_are_the_same_for_the_same_input() -> None:
    first = registry.build_figure(SPEC, Manifest(RESULTS))
    second = registry.build_figure(SPEC, Manifest(RESULTS))
    with paper.paper_style():
        assert paper.svg_bytes(first.figure) == paper.svg_bytes(second.figure)
        assert paper.pdf_bytes(first.figure) == paper.pdf_bytes(second.figure)


def test_the_legend_keys_the_classes_the_reader_and_both_controls(
    output: registry.FigureOutput,
) -> None:
    (legend,) = output.figure.legends
    assert [t.get_text() for t in legend.get_texts()] == [
        "known questions",
        "unknown questions",
        "all-pairs reader (brown: GPT-2; blue: Qwen3; open: post-trained)",
        "random directions: range and mean",
        "shuffled-label refits: mean",
    ]


def test_the_data_file_holds_the_bins_and_the_values_of_every_checkpoint(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    assert list(rows[0]) == a_knowledge_projections.HEADER
    models = _R["models"]("v1")
    assert len(models) == 8
    assert sorted({r["checkpoint"] for r in rows}) == sorted(models)
    for model in models:
        for name, panel in _drawn(model).items():
            mine = [
                r
                for r in rows
                if r["checkpoint"] == model and r["contrast"] == name
            ]
            assert {r["panel"] for r in mine} == {panel}
            top, bottom = _sides(model, name)
            for series, shares in (
                ("first side", top),
                ("second side", bottom),
            ):
                got = [r for r in mine if r["series"] == series]
                assert [float(r["value"]) for r in got] == pytest.approx(
                    shares, abs=1e-6
                )
                assert float(got[0]["low"]) == -LIMIT
                assert float(got[-1]["high"]) == LIMIT
            want = _auroc(model, name)
            value = {
                r["series"]: r
                for r in mine
                if r["series"] in ("reader", "random", "shuffled labels")
            }
            assert float(value["reader"]["value"]) == pytest.approx(
                want["reader"], abs=1e-6
            )
            assert float(value["random"]["value"]) == pytest.approx(
                want["random"], abs=1e-6
            )
            assert float(value["random"]["low"]) == pytest.approx(
                want["random_low"], abs=1e-6
            )
            assert float(value["random"]["high"]) == pytest.approx(
                want["random_high"], abs=1e-6
            )
            assert float(value["shuffled labels"]["value"]) == pytest.approx(
                want["shuffled"], abs=1e-6
            )


def test_the_caption_says_what_is_drawn_and_what_the_controls_are(
    output: registry.FigureOutput,
) -> None:
    detail = output.detail
    assert "published reader" not in detail
    for phrase in (
        "all-pairs reader",
        "The views run, which reads the questions of the knowledge reading "
        "run again",
        "z-score",
        "upper side",
        "lower side",
        "grey band",
        "controls",
        "labels shuffled",
        "top to bottom",
        "at most 300",
        "does not read",
        "each row is a drawn contrast",
        "obscure real pop and triviaqa known, are scored and not drawn",
        "[[table:knowledge_projections]]",
        f"mean of the {_draws(SHOWN[0])} shuffled-label refits",
        "SyntHal [[cite:lu2026relational]]",
        "KUQ [[cite:amayuelas2024knowledge]]",
        "brown for GPT-2 and blue for Qwen3",
        "z-score over the questions of the cell",
        "continues below it",
        "GPT-2 medium",
    ):
        assert phrase in detail, phrase
    assert "stores the questions of two" not in detail
    text = re.sub(r"\[\[[^\]]*\]\]", "", f"{output.claim} {detail}")
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text
    assert "_" not in text
    assert "baseline" not in text.lower()
    doc = a_knowledge_projections.__doc__ or ""
    assert doc
    assert not re.search(r"Figure A?\d", doc)
