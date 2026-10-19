import csv
import io
import itertools
import json
import math
from pathlib import Path
import runpy
from typing import Any

import pytest

from tomography.site import params
from tomography.site import registry
from tomography.site import stats
from tomography.site.figs import a_links
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
_T = runpy.run_path(str(Path(__file__).with_name("_titles.py")))
SPEC = "a_links"
HEIGHT = 3.8


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _out(output: registry.FigureOutput) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(output.csv)))


def _read(model: str, name: str) -> list[dict[str, str]]:
    path = _R["run"]("p10c", model) / name
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cosine_outside(row: dict[str, str]) -> bool:
    return (
        not float(row["band_lo"])
        <= float(row["cosine"])
        <= float(row["band_hi"])
    )


def _auroc_outside(row: dict[str, str]) -> bool:
    return not (
        float(row["lo"]) <= float(row["rand_hi"])
        and float(row["hi"]) >= float(row["rand_lo"])
    )


def _pooled(model: str) -> list[dict[str, str]]:
    return [
        r
        for r in _read(model, "knowledge.csv")
        if r["items"] == "all" and r["test_family"] == "pooled"
    ]


def _counts(which: str) -> dict[str, tuple[int, int]]:
    found = {role: [0, 0] for role in stats.ROLES}
    for model in _R["models"]("p10c"):
        rows = _read(model, "cosine.csv") if which == "cos" else _pooled(model)
        flag = _cosine_outside if which == "cos" else _auroc_outside
        cell = found[stats.role(model)]
        cell[0] += sum(flag(r) for r in rows)
        cell[1] += len(rows)
    return {k: (v[0], v[1]) for k, v in found.items()}


def test_every_layer_share_is_recomputed_from_the_csv_files(
    output: registry.FigureOutput,
) -> None:
    rows = _out(output)
    for model in _R["models"]("p10c"):
        for panel, source, flag in (
            ("cosine", _read(model, "cosine.csv"), _cosine_outside),
            ("auroc", _pooled(model), _auroc_outside),
        ):
            layers = max(int(r["layer"]) for r in source)
            got = [r for r in rows if r["row"] == panel and r["model"] == model]
            assert [int(r["layer"]) for r in got] == list(range(1, layers + 1))
            for r in got:
                cells = [
                    x for x in source if int(x["layer"]) == int(r["layer"])
                ]
                assert float(r["share"]) == pytest.approx(
                    sum(flag(x) for x in cells) / len(cells)
                )
                assert float(r["depth"]) == pytest.approx(
                    int(r["layer"]) / layers
                )


def test_counts_per_role_and_numbers(output: registry.FigureOutput) -> None:
    for name, which in (("cos", "cos"), ("auc", "auc")):
        want = _counts(which)
        for role in stats.ROLES:
            got = output.counts[f"{name}_out"][role]
            assert (got.hits, got.total) == want[role]
            assert output.numbers[f"{name}_out_{role}"] == want[role][0]
            assert output.numbers[f"cells_{role}"] == want[role][1]
    cos, auc = _counts("cos"), _counts("auc")
    assert {r: v[1] for r, v in cos.items()} == {
        r: v[1] for r, v in auc.items()
    }


def test_layer_one_statement_uses_the_chosen_layer_of_every_checkpoint(
    output: registry.FigureOutput,
) -> None:
    layers = []
    biggest = 0.0
    for model in _R["models"]("p10c"):
        layer = int(
            json.loads(
                (_R["run"]("p10c", model) / "summary.json").read_text("utf-8")
            )["data"]["layer"]
        )
        layers.append(layer)
        biggest = max(
            biggest,
            max(
                abs(float(r["cosine"]))
                for r in _read(model, "cosine.csv")
                if int(r["layer"]) == layer
            ),
        )
    assert set(layers) == {1}
    assert output.numbers["layer1_cos_max"] == pytest.approx(biggest)
    assert (
        "At layer 1, the layer the plain-template range reader chose on every "
        "checkpoint because ties go to the first layer, every cosine is at "
        f"most {biggest:.3f} in absolute value; these values describe that "
        "layer and do not locate a link."
    ) in output.detail


def test_other_chosen_layers_get_a_general_sentence() -> None:
    text = a_links.layer_sentence([1, 3], 0.123)
    assert "layer 1" not in text
    assert "0.123" in text
    assert "At the layer the plain-template range reader chose on each" in text


def test_caption_counts_per_role(output: registry.FigureOutput) -> None:
    cos, auc = _counts("cos"), _counts("auc")
    assert output.claim == (
        "The range direction against the knowledge directions at every layer."
    )
    assert (
        "Over all layers the cosine lies outside its random band in "
        f"{cos['gpt2'][0]:,} of {cos['gpt2'][1]:,} GPT-2 cells, "
        f"{cos['base'][0]:,} of {cos['base'][1]:,} Qwen3 base cells and "
        f"{cos['post'][0]:,} of {cos['post'][1]:,} post-trained cells, and the "
        f"AUROC interval outside its band in {auc['gpt2'][0]:,}, "
        f"{auc['base'][0]:,} and {auc['post'][0]:,} of the same cells."
    ) in output.detail
    assert "0.05" in output.detail


def test_size_guards_and_layout(output: registry.FigureOutput) -> None:
    fig = output.figure
    assert tuple(fig.get_size_inches()) == (paper.FULL_WIDTH, HEIGHT)
    assert checks.guard_problems(fig, appendix=True) == []
    assert checks.problems(fig) == []


def test_svg_and_pdf_bytes_are_the_same_for_the_same_input() -> None:
    first = registry.build_figure(SPEC, Manifest(RESULTS))
    second = registry.build_figure(SPEC, Manifest(RESULTS))
    with paper.paper_style():
        assert paper.svg_bytes(first.figure) == paper.svg_bytes(second.figure)
        assert paper.pdf_bytes(first.figure) == paper.pdf_bytes(second.figure)


BINS = 6
GROUPS = ["GPT-2", "Qwen3\nbase", "Qwen3\npost-trained"]
SHORT = ["small", "medium", "large", "XL"] + ["0.6B", "1.7B", "4B"] * 2
ROLE_ORDER = ("gpt2", "base", "post")


def _ordered() -> list[str]:
    models: list[str] = _R["models"]("p10c")
    return [
        m
        for role in ROLE_ORDER
        for m in sorted(models, key=params.param_count)
        if stats.role(m) == role
    ]


def _flags(model: str, which: str) -> list[tuple[int, bool]]:
    rows = _read(model, "cosine.csv") if which == "cos" else _pooled(model)
    flag = _cosine_outside if which == "cos" else _auroc_outside
    return [(int(r["layer"]), bool(flag(r))) for r in rows]


def _share(model: str, which: str) -> tuple[int, int]:
    flags = _flags(model, which)
    return sum(f for _, f in flags), len(flags)


def _panels(output: registry.FigureOutput) -> list[Any]:
    return [ax for ax in output.figure.axes if ax.get_label() != "<legend>"]


def _dots(ax: Any) -> list[Any]:
    return [ln for ln in ax.get_lines() if ln.get_marker() == "o"]


def test_two_dot_panels_and_a_heat_strip_share_the_role_block_rows(
    output: registry.FigureOutput,
) -> None:
    ax_a, ax_b, ax_c, bar = _panels(output)
    for ax in (ax_b, ax_c):
        assert ax_a.get_shared_y_axes().joined(ax_a, ax)
    labels = [t.get_text() for t in ax_a.get_yticklabels()]
    assert labels == SHORT
    names = [t.get_text() for t in ax_a.texts if t.get_text() in GROUPS]
    assert names == GROUPS
    ticks = list(ax_a.get_yticks())
    gaps = [b - a for a, b in itertools.pairwise(ticks)]
    assert max(gaps) > 1.0
    assert bar is not None


def test_each_dot_is_the_share_of_cells_outside_the_band_of_its_checkpoint(
    output: registry.FigureOutput,
) -> None:
    ax_a, ax_b = _panels(output)[:2]
    for ax, which in ((ax_a, "cos"), (ax_b, "auc")):
        dots = _dots(ax)
        assert len(dots) == len(_ordered())
        for model, dot in zip(_ordered(), dots, strict=True):
            hits, total = _share(model, which)
            assert dot.get_xdata()[0] == pytest.approx(hits / total)
            post = stats.role(model) == "post"
            assert (dot.get_markerfacecolor() == "white") is post
        assert len(set(dot.get_ydata()[0] for dot in dots)) == len(dots)


def test_the_expected_share_is_a_labelled_reference_in_both_dot_panels(
    output: registry.FigureOutput,
) -> None:
    for ax in _panels(output)[:2]:
        refs = [
            ln
            for ln in ax.get_lines()
            if list(ln.get_xdata()) == [a_links.CHANCE, a_links.CHANCE]
        ]
        assert refs
        texts = [t.get_text() for t in ax.texts]
        assert "expected for\nrandom vectors" in texts


def test_cell_counts_are_printed_at_the_right_of_every_row(
    output: registry.FigureOutput,
) -> None:
    ax_a, ax_b = _panels(output)[:2]
    for ax, which in ((ax_a, "cos"), (ax_b, "auc")):
        shown = [t.get_text() for t in ax.texts if " of " in t.get_text()]
        want = []
        for model in _ordered():
            hits, total = _share(model, which)
            want.append(f"{hits:,} of {total:,}")
        assert shown == want


def test_heat_strip_is_the_cosine_share_in_six_depth_bins(
    output: registry.FigureOutput,
) -> None:
    ax_c = _panels(output)[2]
    meshes = ax_c.collections
    assert len(meshes) == 3
    assert not ax_c.images
    got = []
    for mesh in meshes:
        got.extend(mesh.get_array().reshape(-1, BINS).tolist())
    assert len(got) == len(_ordered())
    for model, row in zip(_ordered(), got, strict=True):
        cells = _flags(model, "cos")
        layers = max(k for k, _ in cells)
        for b in range(BINS):
            members = [
                f
                for k, f in cells
                if min(math.ceil(k / layers * BINS) - 1, BINS - 1) == b
            ]
            assert row[b] == pytest.approx(sum(members) / len(members))


def test_the_colour_bar_marks_the_expected_share(
    output: registry.FigureOutput,
) -> None:
    bar = _panels(output)[3]
    ticks = [round(float(t), 2) for t in bar.get_yticks()]
    assert a_links.CHANCE in ticks


def test_the_csv_exports_the_per_checkpoint_totals(
    output: registry.FigureOutput,
) -> None:
    rows = _out(output)
    for panel, which in (("cosine_all", "cos"), ("auroc_all", "auc")):
        for model in _ordered():
            (got,) = [
                r for r in rows if r["row"] == panel and r["model"] == model
            ]
            hits, total = _share(model, which)
            assert int(got["outside"]) == hits
            assert int(got["types"]) == total
            assert float(got["share"]) == pytest.approx(hits / total)


def test_the_role_key_uses_neutral_circles(
    output: registry.FigureOutput,
) -> None:
    (legend,) = output.figure.legends
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == ["GPT-2 and Qwen3 base", "post-trained"]
    markers = {h.get_marker() for h in legend.legend_handles}
    assert markers == {"o"}
    colours = {h.get_markeredgecolor() for h in legend.legend_handles}
    assert colours == {paper.ENC["reference"]}


def test_caption_warns_about_steps_and_dependent_layers(
    output: registry.FigureOutput,
) -> None:
    text = output.detail
    assert "moves in steps" in text
    assert "adjacent layers share a residual stream" in text
    assert "rough reference" in text
    assert "six bins" in text
    assert "At layer 1" in text


def test_three_panel_letters_and_no_dashes(
    output: registry.FigureOutput,
) -> None:
    letters = [
        t.get_text()
        for ax in output.figure.axes
        for t in ax.texts
        if t.get_text().startswith(r"$\mathbf")
    ]
    assert sorted(letters) == [rf"$\mathbf{{({c})}}$" for c in "abc"]
    text = f"{output.claim} {output.detail} {output.csv}"
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text
    assert "_" not in f"{output.claim} {output.detail}"


def test_the_heat_strip_and_its_bar_use_the_glossary_sequential_map(
    output: registry.FigureOutput,
) -> None:
    ax_c = _panels(output)[2]
    for mesh in ax_c.collections:
        assert mesh.get_cmap().name == paper.SEQUENTIAL_CMAP


def test_the_reference_line_does_not_cross_its_label(
    output: registry.FigureOutput,
) -> None:
    checks.draw(output.figure)
    for ax in _panels(output)[:2]:
        (label,) = [t for t in ax.texts if "expected" in t.get_text()]
        line_x = ax.transData.transform((a_links.CHANCE, 0.0))[0]
        box = label.get_window_extent()
        assert box.x0 > line_x or box.x1 < line_x


def test_the_heat_strip_names_its_rows(output: registry.FigureOutput) -> None:
    ax_c = _panels(output)[2]
    checks.draw(output.figure)
    shown = [
        t.get_text()
        for t in ax_c.get_yticklabels()
        if t.get_visible() and t.get_text()
    ]
    assert shown == SHORT
    assert ax_c.yaxis.get_tick_params()["labelleft"]


def test_the_colour_bar_has_a_title_and_labels_the_expected_share(
    output: registry.FigureOutput,
) -> None:
    bar = _panels(output)[3]
    checks.draw(output.figure)
    assert bar.get_ylabel() == "share of cells\noutside the band"
    labels = [t.get_text() for t in bar.get_yticklabels()]
    assert "0.05 random" in labels
