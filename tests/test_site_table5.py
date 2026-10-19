import csv
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
SPEC = registry.TABLES[4]
CELL = ("free", "fewshot", "range_last", "greedy")


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table(SPEC, Manifest(RESULTS))


def _run(experiment: str, model: str) -> Path:
    (run,) = [p for p in (RESULTS / experiment / model).iterdir() if p.is_dir()]
    return run


def _models() -> list[str]:
    return Manifest(RESULTS).models("p10a")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _percent(text: str) -> str:
    return str(round(100 * float(text)))


def _cosines(model: str) -> list[dict[str, float]]:
    path = _run("p10c", model) / "summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))["data"]["cosine"]
    return list(data.values())


def _counts(model: str) -> dict[str, tuple[int, int]]:
    run = _run("p10b", model)
    transfer = _read(run / "transfer.csv")
    reader = [float(r["auroc"]) for r in transfer if r["method"] == "logistic"]
    random = [float(r["auroc"]) for r in transfer if r["method"] == "random"]
    text = [
        float(r["auroc"])
        for r in _read(run / "baselines.csv")
        if r["baseline"] == "text_word"
    ]
    return {
        "reader": (sum(round(v, 3) == 1.0 for v in reader), len(reader)),
        "text": (sum(round(v, 3) == 1.0 for v in text), len(text)),
        "random": (sum(v >= 0.899 for v in random), len(random)),
    }


def _has(claim: str, value: float) -> bool:
    text = str(int(value)) if value == int(value) else f"{value:.3f}"
    pattern = rf"(?<![\d.]){re.escape(text)}(?![\d])"
    return re.search(pattern, claim) is not None


def test_there_is_one_row_per_p10_model(output: registry.TableOutput) -> None:
    assert [r[0] for r in output.rows] == [
        paper.display_name(m) for m in _models()
    ]
    assert all(len(r) == len(output.columns) for r in output.rows)
    assert len(output.align) == len(output.columns)


def test_behavior_columns_are_the_greedy_range_last_cell(
    output: registry.TableOutput,
) -> None:
    for model, row in zip(_models(), output.rows, strict=True):
        run = _run("p10a", model)
        table = {
            r["kind"]: r
            for r in _read(run / "table.csv")
            if r["family"] == "all"
            and (r["condition"], r["format"], r["shot_order"], r["decoding"])
            == CELL
        }
        (diag,) = [
            r
            for r in _read(run / "diagnostic.csv")
            if (r["condition"], r["format"], r["shot_order"], r["decoding"])
            == CELL
        ]
        forced = [
            r
            for r in _read(run / "forced_in_range.csv")
            if (r["family"], r["kind"], r["format"], r["shot_order"])
            == ("all", "range", "fewshot", "range_last")
        ]
        assert row[output.columns.index("Range items (%)")] == _percent(
            table["range"]["range_rate"]
        )
        assert row[output.columns.index("Exact items (%)")] == _percent(
            table["exact"]["range_rate"]
        )
        assert row[output.columns.index("People (%)")] == _percent(
            diag["people_rate"]
        )
        assert row[output.columns.index("Forced in range (%)")] == _percent(
            forced[0]["inside_rate"]
        )


def test_reader_cell_counts_are_taken_over_every_stored_cell(
    output: registry.TableOutput,
) -> None:
    for model, row in zip(_models(), output.rows, strict=True):
        want = _counts(model)
        for name, key in (
            ("Reader 1.000", "reader"),
            ("Text words 1.000", "text"),
            ("Random 0.899 or more", "random"),
        ):
            assert row[output.columns.index(name)] == _cells.count(*want[key])


def test_link_column_counts_cosines_outside_the_random_band(
    output: registry.TableOutput,
) -> None:
    for model, row in zip(_models(), output.rows, strict=True):
        cosines = _cosines(model)
        outside = sum(
            not c["band_lo"] <= c["cosine"] <= c["band_hi"] for c in cosines
        )
        column = output.columns.index("Cosines outside band")
        assert row[column] == _cells.count(outside, len(cosines))


def test_claim_numbers_are_recomputed_from_the_files(
    output: registry.TableOutput,
) -> None:
    models = _models()
    counts = [_counts(m) for m in models]
    cosines = [c for m in models for c in _cosines(m)]
    expected = {
        "models": len(models),
        "reader_cells": sum(c["reader"][0] for c in counts),
        "reader_total": sum(c["reader"][1] for c in counts),
        "text_cells": sum(c["text"][0] for c in counts),
        "text_total": sum(c["text"][1] for c in counts),
        "random_cells": sum(c["random"][0] for c in counts),
        "random_total": sum(c["random"][1] for c in counts),
        "cosine_outside": sum(
            not c["band_lo"] <= c["cosine"] <= c["band_hi"] for c in cosines
        ),
        "cosine_total": len(cosines),
    }
    assert output.numbers == pytest.approx(expected)
    claim = output.claim.replace(",", "")
    assert all(_has(claim, v) for v in expected.values())
    assert output.claim.endswith(".")


def test_detail_defines_the_cells_and_the_sources(
    output: registry.TableOutput,
) -> None:
    assert "greedy" in output.detail
    assert "Wilson" not in output.detail or "resample" in output.detail
    assert "random" in output.detail


def test_no_cell_uses_a_dash_character(output: registry.TableOutput) -> None:
    text: Any = " ".join(
        [output.claim, output.detail, *output.columns, *output.notes]
        + [c for row in output.rows for c in row]
    )
    assert "\u2014" not in text
    assert "\u2013" not in text


def test_build_is_deterministic() -> None:
    first = registry.build_table(SPEC, Manifest(RESULTS))
    second = registry.build_table(SPEC, Manifest(RESULTS))
    assert first == second
