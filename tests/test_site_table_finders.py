import json
from pathlib import Path

import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
SPEC = "finders"
TASKS = ("sentiment", "truth", "utility")


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table(SPEC, Manifest(RESULTS))


def _cell(model: str, task: str, method: str) -> dict:
    run = Manifest(RESULTS).run("p4", model)
    assert run is not None
    data = json.loads((run / "summary.json").read_text("utf-8"))["data"]
    pairing = data["reference"][task]["pairing"]
    table = json.loads((run / "table.json").read_text("utf-8"))["data"]
    hits = [
        r
        for r in table
        if (r["task"], r["method"], r["pairing"], r["preprocess"])
        == (task, method, pairing, "center")
    ]
    assert len(hits) == 1
    return hits[0]


def _text(row: dict) -> str:
    return f"{row['report']:.3f} [{row['lo']:.3f}, {row['hi']:.3f}]"


def test_columns(output: registry.TableOutput) -> None:
    assert output.columns == [
        "Task",
        "Model",
        "Random",
        "PCA of differences (paper)",
        "Logistic regression",
    ]
    assert output.align == "llrrr"


def test_every_p4_model_has_a_row_per_task_with_the_stored_cells(
    output: registry.TableOutput,
) -> None:
    models = Manifest(RESULTS).models("p4")
    assert len(output.rows) == 3 * len(models)
    expected = []
    for task in TASKS:
        for model in models:
            expected.append(
                [
                    _text(_cell(model, task, "random")),
                    _text(_cell(model, task, "pca_diff")),
                    _text(_cell(model, task, "logistic")),
                ]
            )
    assert [r[2:] for r in output.rows] == expected
    assert [r[1] for r in output.rows[: len(models)]] == [
        paper.display_name(m) for m in models
    ]


def test_the_claim_numbers_match_a_recomputation(
    output: registry.TableOutput,
) -> None:
    models = Manifest(RESULTS).models("p4")
    cells = [(m, t) for t in TASKS for m in models]
    logistic_wins = sum(
        _cell(m, t, "logistic")["report"] >= _cell(m, t, "pca_diff")["report"]
        for m, t in cells
    )
    pca_reads = sum(
        _cell(m, t, "pca_diff")["report"] - _cell(m, t, "random")["report"]
        > 0.1
        for m, t in cells
    )
    assert output.numbers["cells"] == len(cells)
    assert output.numbers["logistic_at_least_pca"] == logistic_wins
    assert output.numbers["pca_above_random"] == pca_reads
    assert f"{logistic_wins} of {len(cells)}" in output.claim
    assert f"{pca_reads} of {len(cells)}" in output.claim
    assert output.claim.endswith(".")


def test_the_footer_gives_the_m1_win_rates_of_every_m1_model(
    output: registry.TableOutput,
) -> None:
    manifest = Manifest(RESULTS)
    text = " ".join(output.notes)
    for model in manifest.models("m1"):
        run = manifest.run("m1", model)
        assert run is not None
        rows = json.loads((run / "table.json").read_text("utf-8"))["data"][
            "rows"
        ]
        wins = {
            r[0]: r[3] for r in rows if r[1] == "manipulation" and r[2] == "0.3"
        }
        part = " / ".join(
            f"{wins[m]:.3f}"
            for m in ("pca_diff", "mean_diff", "logistic", "random")
        )
        assert f"{paper.display_name(model)} {part}" in text, model


def test_no_dash_characters(output: registry.TableOutput) -> None:
    banned = {chr(0x2014), chr(0x2013)}
    texts = [output.claim, output.detail, *output.notes]
    texts += [c for row in output.rows for c in row]
    assert not any(ch in t for t in texts for ch in banned)
