from pathlib import Path
import re
from typing import Any

import numpy as np
import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.site.tables import _cells
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[2] / "results"
SPEC = "steering"
MAX_RATIO = 1.5
MIN_DISTINCT = 0.15


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table(SPEC, Manifest(RESULTS))


def _run(experiment: str, model: str) -> Path:
    (run,) = [p for p in (RESULTS / experiment / model).iterdir() if p.is_dir()]
    return run


def _models() -> list[str]:
    return Manifest(RESULTS).models("c1")


def _c1(model: str) -> list[dict[str, Any]]:
    return _cells.read_table(_run("c1", model) / "table.json")


def _none(rows: list[dict[str, Any]]) -> dict[str, Any]:
    (row,) = [r for r in rows if r["arm"] == "none"]
    return row


def _mean(rows: list[dict[str, Any]], coeff: float, key: str) -> float:
    cell = [
        r[key] for r in rows if r["arm"] == "linear" and r["coeff"] == coeff
    ]
    assert len(cell) == 3
    return float(np.mean(cell))


def _triple(rows: list[dict[str, Any]], coeff: float) -> str:
    ratio = _mean(rows, coeff, "perplexity") / _none(rows)["perplexity"]
    return (
        f"{_mean(rows, coeff, 'score'):.2f} / {ratio:.2f} / "
        f"{_mean(rows, coeff, 'distinct_1'):.2f}"
    )


def _fluent(rows: list[dict[str, Any]], size: float) -> bool:
    for coeff in (size, -size):
        ratio = _mean(rows, coeff, "perplexity") / _none(rows)["perplexity"]
        if ratio > MAX_RATIO or _mean(rows, coeff, "distinct_1") < MIN_DISTINCT:
            return False
    return True


def _m1(model: str) -> dict[str, dict[str, Any]]:
    run = Manifest(RESULTS).run("m1", model)
    if run is None:
        return {}
    rows = _cells.read_table(run / "table.json")
    return {
        r["method"]: r
        for r in rows
        if r["task"] == "manipulation" and r["setting"] == "0.3"
    }


def _has(claim: str, value: float) -> bool:
    text = str(int(value)) if value == int(value) else f"{value:.3f}"
    pattern = rf"(?<![\d.]){re.escape(text)}(?![\d])"
    return re.search(pattern, claim) is not None


def test_there_is_one_row_per_c1_model(output: registry.TableOutput) -> None:
    assert [r[0] for r in output.rows] == [
        paper.display_name(m) for m in _models()
    ]
    assert all(len(r) == len(output.columns) for r in output.rows)
    assert len(output.align) == len(output.columns)


def test_steering_cells_hold_score_ratio_and_distinct_1(
    output: registry.TableOutput,
) -> None:
    for model, row in zip(_models(), output.rows, strict=True):
        rows = _c1(model)
        none = _none(rows)
        assert row[output.columns.index("Unsteered")] == (
            f"{none['score']:.2f} / 1.00 / {none['distinct_1']:.2f}"
        )
        for name, coeff in (("-0.1", -0.1), ("+0.1", 0.1), ("+0.3", 0.3)):
            assert row[output.columns.index(name)] == _triple(rows, coeff)
        (prompt,) = [r for r in rows if r["arm"] == "prompting+1"]
        assert row[output.columns.index("Prompting +1")] == (
            f"{prompt['score']:.2f} / n/a / {prompt['distinct_1']:.2f}"
        )


def test_m1_columns_show_the_random_arm_where_it_is_stored(
    output: registry.TableOutput,
) -> None:
    for model, row in zip(_models(), output.rows, strict=True):
        arms = _m1(model)
        for name, method in (
            ("M1 random", "random"),
            ("M1 mean_diff", "mean_diff"),
        ):
            hit = arms.get(method)
            want = (
                _cells.interval(hit["value"], hit["lo"], hit["hi"], 2)
                if hit
                else "n/a"
            )
            assert row[output.columns.index(name)] == want


def test_claim_numbers_are_recomputed_from_the_files(
    output: registry.TableOutput,
) -> None:
    models = _models()
    tables = {m: _c1(m) for m in models}
    arms = {m: _m1(m) for m in models if _m1(m)}
    expected = {
        "models": len(models),
        "score_unsteered": float(
            np.mean([_none(t)["score"] for t in tables.values()])
        ),
        "score_minus": float(
            np.mean([_mean(t, -0.1, "score") for t in tables.values()])
        ),
        "score_plus": float(
            np.mean([_mean(t, 0.1, "score") for t in tables.values()])
        ),
        "fluent_at_0.1": sum(_fluent(t, 0.1) for t in tables.values()),
        "fluent_at_0.3": sum(_fluent(t, 0.3) for t in tables.values()),
        "m1_models": len(arms),
        "m1_random_max": max(a["random"]["value"] for a in arms.values()),
        "m1_mean_diff_min": min(a["mean_diff"]["value"] for a in arms.values()),
    }
    assert output.numbers == pytest.approx(expected)
    assert all(_has(output.claim, v) for v in expected.values())
    assert output.claim.endswith(".")


def test_notes_say_that_c1_has_no_random_arm(
    output: registry.TableOutput,
) -> None:
    text = " ".join(output.notes) + output.detail
    assert "no random-direction arm" in text
    assert "1.5" in text
    assert "0.15" in text


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
