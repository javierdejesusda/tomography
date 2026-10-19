import json
from pathlib import Path
import re
from typing import Any

import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.site.tables import _cells
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
SPEC = registry.TABLES[6]


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table(SPEC, Manifest(RESULTS))


def _run(experiment: str, model: str) -> Path:
    (run,) = [p for p in (RESULTS / experiment / model).iterdir() if p.is_dir()]
    return run


def _models() -> list[str]:
    return Manifest(RESULTS).models("r6")


def _r6(model: str) -> list[dict[str, Any]]:
    return _cells.read_table(_run("r6", model) / "table.json")


def _summary(experiment: str, model: str) -> Any:
    path = _run(experiment, model) / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))["data"]


def _reader(model: str) -> dict[str, Any]:
    return next(
        r
        for r in _r6(model)
        if r["method"] == "lat"
        and r["layer_rule"] == "validation-layer"
        and r["source"] == "truthfulqa"
    )


def _text(name: str) -> dict[str, Any]:
    for model in _models():
        hit = [r for r in _r6(model) if r["method"] == name]
        if hit:
            return hit[0]
    raise AssertionError(name)


def _has(claim: str, value: float) -> bool:
    text = str(int(value)) if value == int(value) else f"{value:.3f}"
    pattern = rf"(?<![\d.]){re.escape(text)}(?![\d])"
    return re.search(pattern, claim) is not None


def test_there_is_one_row_per_r6_model(output: registry.TableOutput) -> None:
    assert [r[0] for r in output.rows] == [
        paper.display_name(m) for m in _models()
    ]
    assert all(len(r) == len(output.columns) for r in output.rows)
    assert len(output.align) == len(output.columns)


def test_reader_and_standard_cells_use_the_validation_layer_row(
    output: registry.TableOutput,
) -> None:
    for model, row in zip(_models(), output.rows, strict=True):
        reader = _reader(model)
        standard = next(r for r in _r6(model) if r["method"] == "standard")
        assert row[output.columns.index("Reader")] == _cells.interval(
            reader["mc1"], reader["lo"], reader["hi"]
        )
        assert row[output.columns.index("Standard")] == _cells.interval(
            standard["mc1"], standard["lo"], standard["hi"]
        )


def test_text_cells_are_the_same_model_independent_row_for_every_model(
    output: registry.TableOutput,
) -> None:
    for name, column in (
        ("text_word", "Text words"),
        ("text_char", "Text chars"),
    ):
        row = _text(name)
        want = _cells.interval(row["mc1"], row["lo"], row["hi"])
        index = output.columns.index(column)
        assert {r[index] for r in output.rows} == {want}
    assert any("model independent" in note for note in output.notes)


def test_honesty_cells_pair_r5_transfer_with_a1_agreement(
    output: registry.TableOutput,
) -> None:
    for model, row in zip(_models(), output.rows, strict=True):
        r5 = _summary("r5", model)
        a1 = _summary("a1", model)["agreement_at_layer"]["r5"]
        lo, hi = r5["ci"]
        assert row[output.columns.index("Honesty transfer")] == (
            _cells.interval(r5["test_accuracy"], lo, hi)
        )
        assert row[output.columns.index("Seed agreement")] == _cells.num(a1)


def test_claim_numbers_are_recomputed_from_the_files(
    output: registry.TableOutput,
) -> None:
    models = _models()
    word, char = _text("text_word"), _text("text_char")
    low, high = min(word["lo"], char["lo"]), max(word["hi"], char["hi"])
    standard = {
        m: next(r for r in _r6(m) if r["method"] == "standard") for m in models
    }
    accuracy = [_summary("r5", m)["test_accuracy"] for m in models]
    agreement = [_summary("a1", m)["agreement_at_layer"]["r5"] for m in models]
    expected = {
        "models": len(models),
        "inside_text_interval": sum(
            low <= _reader(m)["mc1"] <= high for m in models
        ),
        "above_standard": sum(
            _reader(m)["mc1"] > standard[m]["mc1"] for m in models
        ),
        "honesty_min": min(accuracy),
        "honesty_max": max(accuracy),
        "agreement_min": min(agreement),
    }
    assert output.numbers == pytest.approx(expected)
    assert all(_has(output.claim, v) for v in expected.values())
    assert output.claim.endswith(".")


def test_the_withdrawn_validation_figure_is_never_quoted(
    output: registry.TableOutput,
) -> None:
    text = " ".join([output.claim, output.detail, *output.notes])
    text += " ".join(c for row in output.rows for c in row)
    assert "0.718" not in text


def test_detail_names_what_each_interval_resamples(
    output: registry.TableOutput,
) -> None:
    assert "15 readers" in output.detail
    assert "Wilson" in output.detail
    assert "exploratory" in output.detail


def test_no_cell_uses_a_dash_character(output: registry.TableOutput) -> None:
    text = " ".join(
        [output.claim, output.detail, *output.columns, *output.notes]
        + [c for row in output.rows for c in row]
    )
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text


def test_build_is_deterministic() -> None:
    first = registry.build_table(SPEC, Manifest(RESULTS))
    second = registry.build_table(SPEC, Manifest(RESULTS))
    assert first == second
