from pathlib import Path
import re
from typing import Any

import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.site.tables import _cells
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[2] / "results"
SPEC = "knowledge"


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table(SPEC, Manifest(RESULTS))


def _run(experiment: str, model: str) -> Path:
    (run,) = [p for p in (RESULTS / experiment / model).iterdir() if p.is_dir()]
    return run


def _models() -> list[str]:
    return Manifest(RESULTS).models("p9a")


def _json(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))["data"]


def _reader(model: str) -> dict[str, Any] | None:
    rows = _cells.read_table(_run("p9b", model) / "table.json")
    hit = [r for r in rows if r["type"] == "obscure_real"]
    return hit[0] if hit else None


def _increment(model: str, features: str) -> dict[str, Any] | None:
    rows = _cells.read_table(_run("p9c", model) / "table_incremental.json")
    hit = [
        r for r in rows if r["dataset"] == "popqa" and r["features"] == features
    ]
    return hit[0] if hit else None


def _has(claim: str, value: float) -> bool:
    text = str(int(value)) if value == int(value) else f"{value:.3f}"
    pattern = rf"(?<![\d.]){re.escape(text)}(?![\d])"
    return re.search(pattern, claim) is not None


def test_there_is_one_row_per_model_with_a_p9a_run(
    output: registry.TableOutput,
) -> None:
    assert [r[0] for r in output.rows] == [
        paper.display_name(m) for m in _models()
    ]
    assert all(len(r) == len(output.columns) for r in output.rows)
    assert len(output.align) == len(output.columns)


def test_known_questions_come_from_the_guess_controlled_count(
    output: registry.TableOutput,
) -> None:
    column = output.columns.index("Known questions")
    for model, row in zip(_models(), output.rows, strict=True):
        known = _json(_run("p9a", model) / "summary.json")["guess"]["n_known"]
        assert row[column] == f"{known:,}"


def test_reader_and_baseline_cells_are_the_within_relation_auroc(
    output: registry.TableOutput,
) -> None:
    for model, row in zip(_models(), output.rows, strict=True):
        found = _reader(model)
        for name, key in (
            ("Reader", "logistic_within"),
            ("Popularity", "popularity_within"),
            ("Text words", "text_word_within"),
            ("Text chars", "text_char_within"),
        ):
            cell = row[output.columns.index(name)]
            want = found[key] if found else None
            assert cell == _cells.num(want)
        pair = row[output.columns.index("Test known / unknown")]
        if found:
            assert pair == _cells.count(
                found["n_positive_test"], found["n_negative_test"]
            )
        else:
            assert pair == "n/a"


def test_increment_cells_carry_their_bootstrap_interval(
    output: registry.TableOutput,
) -> None:
    for model, row in zip(_models(), output.rows, strict=True):
        for name, features in (
            ("Confidence + lat", "confidence+lat"),
            ("Confidence + probe", "confidence+probe"),
        ):
            hit = _increment(model, features)
            want = (
                _cells.interval(hit["diff_vs_confidence"], hit["lo"], hit["hi"])
                if hit
                else "n/a"
            )
            assert row[output.columns.index(name)] == want


def test_claim_numbers_are_recomputed_from_the_files(
    output: registry.TableOutput,
) -> None:
    models = _models()
    readers = {m: _reader(m) for m in models if _reader(m)}
    lat = {m: _increment(m, "confidence+lat") for m in models}
    probe = {m: _increment(m, "confidence+probe") for m in models}
    expected = {
        "models": len(models),
        "with_contrast": len(readers),
        "reader_above_popularity": sum(
            r["logistic_within"] > r["popularity_within"]
            for r in readers.values()
        ),
        "reader_above_text": sum(
            r["logistic_within"]
            > max(r["text_word_within"], r["text_char_within"])
            for r in readers.values()
        ),
        "models_with_lat": sum(v is not None for v in lat.values()),
        "lat_above_zero": sum(
            v is not None and v["lo"] > 0 for v in lat.values()
        ),
        "probe_above_zero": sum(
            v is not None and v["lo"] > 0 for v in probe.values()
        ),
    }
    assert output.numbers == pytest.approx(expected)
    assert all(_has(output.claim, v) for v in expected.values())
    assert output.claim.endswith(".")


def test_detail_says_why_the_reader_has_no_interval(
    output: registry.TableOutput,
) -> None:
    assert "reader draws" in output.detail
    assert "resample" in output.detail


def test_no_cell_uses_a_dash_character(output: registry.TableOutput) -> None:
    text = " ".join(
        [output.claim, output.detail, *output.columns, *output.notes]
        + [c for row in output.rows for c in row]
    )
    assert "\u2014" not in text
    assert "\u2013" not in text


def test_build_is_deterministic() -> None:
    first = registry.build_table(SPEC, Manifest(RESULTS))
    second = registry.build_table(SPEC, Manifest(RESULTS))
    assert first == second
