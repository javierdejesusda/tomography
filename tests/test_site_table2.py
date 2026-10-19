import json
from pathlib import Path

import numpy as np
import pytest

from tomography.site import registry
from tomography.site import store
from tomography.site.manifest import Manifest
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
SPEC = registry.TABLES[1]
MARGIN = 0.1
CONCEPTS = ("r1", "r2", "r3", "r7")
ALL = (*CONCEPTS, "r5")


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table(SPEC, Manifest(RESULTS))


def _summary(exp: str, model: str) -> dict:
    run = Manifest(RESULTS).run(exp, model)
    assert run is not None
    return json.loads((run / "summary.json").read_text("utf-8"))["data"]


def _selected(exp: str, model: str) -> tuple[int, float]:
    data = _summary(exp, model)
    if exp == "r7":
        test = data["sets"]["test"]
        return data["val_layer"], test["accuracy_at_val_layer"]
    key = "best_accuracy" if exp == "r1" else "test_accuracy"
    return data["best_layer"], data[key]


def _random(exp: str, model: str, layer: int) -> float:
    run = Manifest(RESULTS).run(exp, model)
    assert run is not None
    with np.load(run / "curves.npz") as npz:
        return float(npz["mean"][1][layer - 1])


def test_the_table_is_wide_with_a_pair_of_columns_per_concept(
    output: registry.TableOutput,
) -> None:
    assert output.columns == ["Model"] + ["LAT", "Rand."] * 5
    assert output.align == "l" + "rr" * 5
    assert [g[0] for g in output.groups] == [
        "R1 sentiment",
        "R2 utility",
        "R3 morality",
        "R7 truth",
        "R5 honesty (exploratory)",
    ]
    assert [g[1] for g in output.groups] == [2] * 5
    assert [g[2] for g in output.groups] == [False] * 4 + [True]
    assert all(len(row) == 11 for row in output.rows)


def test_there_is_one_row_per_model_in_paper_order(
    output: registry.TableOutput,
) -> None:
    models = {m for e in ALL for m in Manifest(RESULTS).models(e)}
    expected = [
        paper.display_name(m) for m in sorted(models, key=store.model_rank)
    ]
    assert [r[0] for r in output.rows] == expected
    assert len(expected) == 10


def test_every_cell_holds_the_stored_accuracy_and_random_value(
    output: registry.TableOutput,
) -> None:
    manifest = Manifest(RESULTS)
    rows = {r[0]: r for r in output.rows}
    for i, exp in enumerate(ALL):
        have = set(manifest.models(exp))
        for model in {m for e in ALL for m in manifest.models(e)}:
            row = rows[paper.display_name(model)]
            reader, random = row[1 + 2 * i], row[2 + 2 * i]
            if model not in have:
                assert (reader, random) == ("n/a", "n/a")
                continue
            layer, acc = _selected(exp, model)
            assert reader == f"{acc:.3f}"
            if exp == "r5":
                assert random == "n/a"
            else:
                assert random == f"{_random(exp, model, layer):.3f}"


def test_the_caption_points_to_the_appendix_for_intervals_and_layers(
    output: registry.TableOutput,
) -> None:
    assert "Table A1" in output.detail
    assert "interval" in output.detail
    assert "exploratory" in output.detail


def test_the_claim_numbers_match_a_recomputation(
    output: registry.TableOutput,
) -> None:
    manifest = Manifest(RESULTS)
    margins = []
    for exp in CONCEPTS:
        for model in manifest.models(exp):
            layer, acc = _selected(exp, model)
            margins.append(acc - _random(exp, model, layer))
    assert output.numbers["cells"] == len(margins)
    assert output.numbers["above"] == sum(m > MARGIN for m in margins)
    assert output.numbers["smallest_margin"] == pytest.approx(min(margins))
    assert f"{int(output.numbers['above'])} of " in output.claim
    assert f"{output.numbers['smallest_margin']:.3f}" in output.claim


def test_the_footer_gives_the_p6_slopes(output: registry.TableOutput) -> None:
    run = Manifest(RESULTS).run("p6", "gpt2")
    assert run is not None
    stored = json.loads((run / "summary.json").read_text("utf-8"))["data"][
        "accuracy_per_decade"
    ]
    text = " ".join(output.notes)
    for exp, value in stored.items():
        assert f"{exp.upper()} {value:.3f}" in text


def test_no_cell_has_a_dash_character(output: registry.TableOutput) -> None:
    banned = {chr(0x2014), chr(0x2013)}
    texts = [output.claim, output.detail, *output.notes]
    texts += [c for row in output.rows for c in row]
    texts += [g[0] for g in output.groups]
    assert not any(ch in text for text in texts for ch in banned)
