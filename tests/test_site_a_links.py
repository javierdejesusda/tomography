import csv
import io
import json
from pathlib import Path
import runpy

import pytest

from tomography.site import registry
from tomography.site import stats
from tomography.site.figs import a_links
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
SPEC = "a_links"
HEIGHT = 3.4


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
        "At layer 1, the layer the range reader chose on every checkpoint "
        "because ties go to the first layer, every cosine is at most "
        f"{biggest:.3f} in absolute value; these values describe that layer "
        "and do not locate a link."
    ) in output.detail


def test_other_chosen_layers_get_a_general_sentence() -> None:
    text = a_links.layer_sentence([1, 3], 0.123)
    assert "layer 1" not in text
    assert "0.123" in text
    assert "At the layer the range reader chose on each checkpoint" in text


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


def test_two_by_two_with_one_line_per_checkpoint(
    output: registry.FigureOutput,
) -> None:
    panels = [ax for ax in output.figure.axes if ax.get_label() != "<legend>"]
    assert len(panels) == 4
    models = _R["models"]("p10c")
    gpt2 = [m for m in models if m.startswith("gpt2")]
    qwen = [m for m in models if not m.startswith("gpt2")]
    for ax in panels:
        assert not ax.images
        assert not ax.collections
    for ax, count in zip(panels, (len(gpt2), len(qwen)) * 2, strict=True):
        long = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 2]
        assert len(long) == count
    dashed = [
        ln.get_linestyle()
        for ln in panels[1].get_lines()
        if len(ln.get_xdata()) > 2
    ]
    assert dashed.count("--") == len(qwen) // 2


def test_size_guards_and_layout(output: registry.FigureOutput) -> None:
    fig = output.figure
    assert tuple(fig.get_size_inches()) == (paper.FULL_WIDTH, HEIGHT)
    assert checks.guard_problems(fig, appendix=True) == []
    assert checks.problems(fig) == []


def test_four_panel_letters_and_no_dashes(
    output: registry.FigureOutput,
) -> None:
    letters = [
        t.get_text()
        for ax in output.figure.axes
        for t in ax.texts
        if t.get_text().startswith(r"$\mathbf")
    ]
    assert sorted(letters) == [rf"$\mathbf{{({c})}}$" for c in "abcd"]
    text = f"{output.claim} {output.detail} {output.csv}"
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text
    assert "_" not in f"{output.claim} {output.detail}"


def test_svg_and_pdf_bytes_are_the_same_for_the_same_input() -> None:
    first = registry.build_figure(SPEC, Manifest(RESULTS))
    second = registry.build_figure(SPEC, Manifest(RESULTS))
    with paper.paper_style():
        assert paper.svg_bytes(first.figure) == paper.svg_bytes(second.figure)
        assert paper.pdf_bytes(first.figure) == paper.pdf_bytes(second.figure)
