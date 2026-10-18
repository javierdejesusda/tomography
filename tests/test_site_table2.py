import json
from pathlib import Path

import numpy as np
import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest

RESULTS = Path(__file__).resolve().parents[1] / "results"
SPEC = registry.TABLES[1]
MARGIN = 0.1
CONCEPTS = ("r1", "r2", "r3", "r7")


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table(SPEC, Manifest(RESULTS))


def _summary(exp: str, model: str) -> dict:
    run = Manifest(RESULTS).run(exp, model)
    assert run is not None
    return json.loads((run / "summary.json").read_text("utf-8"))["data"]


def _selected(exp: str, model: str) -> tuple[int, float, list[float]]:
    data = _summary(exp, model)
    if exp == "r7":
        test = data["sets"]["test"]
        return (
            data["val_layer"],
            test["accuracy_at_val_layer"],
            test["ci_at_val_layer"],
        )
    key = "best_accuracy" if exp == "r1" else "test_accuracy"
    return data["best_layer"], data[key], data["ci"]


def _random(exp: str, model: str, layer: int) -> float:
    run = Manifest(RESULTS).run(exp, model)
    assert run is not None
    with np.load(run / "curves.npz") as npz:
        return float(npz["mean"][1][layer - 1])


def test_columns_and_alignment(output: registry.TableOutput) -> None:
    assert output.columns == [
        "Concept",
        "Model",
        "Reader [95% interval]",
        "Random",
        "Layer",
    ]
    assert output.align == "llrrr"
    assert all(len(row) == 5 for row in output.rows)


def test_every_model_and_concept_has_a_row_with_the_stored_numbers(
    output: registry.TableOutput,
) -> None:
    manifest = Manifest(RESULTS)
    by_cell = {}
    concept = ""
    for row in output.rows:
        concept = row[0] or concept
        by_cell[(concept, row[1])] = row
    names = {"r1": "R1", "r2": "R2", "r3": "R3", "r7": "R7"}
    for exp in CONCEPTS:
        for model in manifest.models(exp):
            layer, acc, ci = _selected(exp, model)
            cell = next(
                row
                for (c, m), row in by_cell.items()
                if c.startswith(names[exp]) and m == _display(model)
            )
            assert cell[2] == f"{acc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
            assert cell[3] == f"{_random(exp, model, layer):.3f}"
            assert cell[4] == str(layer)


def _display(model: str) -> str:
    from tomography.viz import paper

    return paper.display_name(model)


def test_r5_is_a_labelled_exploratory_group_without_a_random_value(
    output: registry.TableOutput,
) -> None:
    rows = [r for r in output.rows if r[0].startswith("R5")]
    assert len(rows) == 1
    assert "exploratory" in rows[0][0]
    group = output.rows[output.rows.index(rows[0]) :][:10]
    for row, model in zip(group, Manifest(RESULTS).models("r5"), strict=True):
        data = _summary("r5", model)
        low, high = data["ci"]
        assert row[2] == f"{data['test_accuracy']:.3f} [{low:.3f}, {high:.3f}]"
        assert row[3] == "n/a"


def test_the_claim_numbers_match_a_recomputation(
    output: registry.TableOutput,
) -> None:
    manifest = Manifest(RESULTS)
    margins = []
    for exp in CONCEPTS:
        for model in manifest.models(exp):
            layer, acc, _ = _selected(exp, model)
            margins.append(acc - _random(exp, model, layer))
    assert output.numbers["cells"] == len(margins)
    assert output.numbers["above"] == sum(m > MARGIN for m in margins)
    assert output.numbers["smallest_margin"] == pytest.approx(min(margins))
    assert f"{int(output.numbers['above'])} of " in output.claim
    assert f"{output.numbers['smallest_margin']:.3f}" in output.claim
    assert output.claim.endswith(".")


def test_the_footer_gives_the_p6_slopes(output: registry.TableOutput) -> None:
    stored = json.loads(
        (RESULTS / "p6" / "gpt2" / "25ee6de331d4" / "summary.json").read_text(
            "utf-8"
        )
    )["data"]["accuracy_per_decade"]
    text = " ".join(output.notes)
    for exp, value in stored.items():
        assert f"{exp.upper()} {value:.3f}" in text


def test_no_cell_has_a_dash_character(output: registry.TableOutput) -> None:
    banned = {chr(0x2014), chr(0x2013)}
    texts = [output.claim, output.detail, *output.notes]
    texts += [c for row in output.rows for c in row]
    assert not any(ch in text for text in texts for ch in banned)
