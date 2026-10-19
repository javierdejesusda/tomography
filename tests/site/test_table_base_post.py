from pathlib import Path
import re
from typing import Any

import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.site.tables import _cells
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[2] / "results"
SPEC = "base_post"
CLOSE = 0.02


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table(SPEC, Manifest(RESULTS))


def _manifest() -> Manifest:
    return Manifest(RESULTS)


def _table(experiment: str, model: str) -> list[dict[str, Any]]:
    run = _manifest().run(experiment, model)
    assert run is not None
    return _cells.read_table(run / "table.json")


def _p8() -> list[dict[str, Any]]:
    return [r for m in _manifest().models("p8") for r in _table("p8", m)]


def _p3(model: str) -> list[dict[str, Any]]:
    rows = [
        r
        for r in _table("p3", model)
        if r["task"] == "sentiment" and r["target"] == "sentiment"
    ]
    return sorted(rows, key=lambda r: r["size"])


def _p2(model: str) -> dict[str, dict[str, Any]]:
    return {
        r["condition"]: r
        for r in _table("p2", model)
        if r["task"] == "sentiment" and r["position"] == "last"
    }


def _close(rows: list[dict[str, Any]]) -> int:
    top = rows[-1]["accuracy"]
    return next(r["size"] for r in rows if r["accuracy"] >= top - CLOSE)


def _block(output: registry.TableOutput, name: str) -> list[list[str]]:
    return [r for r in output.rows if r[0] == name]


def _has(claim: str, value: float) -> bool:
    text = str(int(value)) if value == int(value) else f"{value:.3f}"
    pattern = rf"(?<![\d.]){re.escape(text)}(?![\d])"
    return re.search(pattern, claim) is not None


def test_rows_are_p8_then_p3_then_p2(output: registry.TableOutput) -> None:
    blocks = [r[0] for r in output.rows]
    assert blocks == sorted(blocks, key=["P8", "P3", "P2"].index)
    assert all(len(r) == len(output.columns) for r in output.rows)
    assert len(output.align) == len(output.columns)


def test_p8_rows_hold_every_pair_with_the_delta_and_its_interval(
    output: registry.TableOutput,
) -> None:
    rows = _block(output, "P8")
    expected = _p8()
    assert len(rows) == len(expected)
    for row, pair in zip(rows, expected, strict=True):
        assert row[1] == (
            f"{paper.display_name(pair['base'])} to "
            f"{paper.display_name(pair['tuned'])}"
        )
        assert pair["source"] in row[2]
        assert row[3] == _cells.interval(pair["delta"], pair["lo"], pair["hi"])


def test_p3_rows_give_the_smallest_the_largest_and_the_close_size(
    output: registry.TableOutput,
) -> None:
    rows = _block(output, "P3")
    models = _manifest().models("p3")
    assert len(rows) == 3 * len(models)
    for i, model in enumerate(models):
        table = _p3(model)
        small, large = table[0], table[-1]
        got = rows[3 * i : 3 * i + 3]
        assert {r[1] for r in got} == {paper.display_name(model)}
        assert got[0][3] == _cells.interval(
            small["accuracy"], small["lo"], small["hi"]
        )
        assert got[1][3] == _cells.interval(
            large["accuracy"], large["lo"], large["hi"]
        )
        assert got[2][3] == str(_close(table))


def test_p2_rows_compare_the_original_and_the_misleading_sentence(
    output: registry.TableOutput,
) -> None:
    rows = _block(output, "P2")
    models = _manifest().models("p2")
    assert len(rows) == 2 * len(models)
    for i, model in enumerate(models):
        conditions = _p2(model)
        for cell, key in zip(
            rows[2 * i : 2 * i + 2], ("original", "against"), strict=True
        ):
            hit = conditions[key]
            assert cell[3] == _cells.interval(
                hit["accuracy"], hit["lo"], hit["hi"]
            )


def test_claim_numbers_are_recomputed_from_the_files(
    output: registry.TableOutput,
) -> None:
    pairs = _p8()
    p3_models = _manifest().models("p3")
    p2_models = _manifest().models("p2")
    close = [_close(_p3(m)) for m in p3_models]
    expected = {
        "pairs": len(pairs),
        "post_lower": sum(p["hi"] < 0 for p in pairs),
        "post_higher": sum(p["lo"] > 0 for p in pairs),
        "close_min": min(close),
        "close_max": max(close),
        "p2_models": len(p2_models),
        "misleading_lowers": sum(
            _p2(m)["against"]["accuracy"] < _p2(m)["original"]["accuracy"]
            for m in p2_models
        ),
    }
    assert output.numbers == pytest.approx(expected)
    assert all(_has(output.claim, v) for v in expected.values())
    assert output.claim.endswith(".")


def test_detail_names_what_each_interval_resamples(
    output: registry.TableOutput,
) -> None:
    assert "resample" in output.detail
    assert "0.02" in output.detail


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
