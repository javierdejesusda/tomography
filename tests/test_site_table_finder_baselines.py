import json
import math
from pathlib import Path
import runpy
from typing import Any

import pytest

from tomography.site import cells
from tomography.site import params
from tomography.site import registry
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.viz import paper

_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
ROOT, data, models, run = _R["ROOT"], _R["data"], _R["models"], _R["run"]
SPEC = "finder_baselines"
DASHES = (chr(0x2014), chr(0x2013))
READER = "pca_diff"
BASELINES = ("mean_diff", "whitened_mean_diff", "logistic", "ccs")
CONTROLS = ("random", "shuffled", "text_word", "text_char")
COLUMNS = [
    "Model",
    "Dataset",
    "PCA of differences",
    "Mass-mean",
    "Whitened mass-mean",
    "Logistic regression",
    "CCS",
    "Random direction",
    "Shuffled labels",
    "Words",
    "Characters",
]
Z = 1.959963984540054


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table(SPEC, Manifest(ROOT))


def _table(model: str) -> dict[str, dict[str, dict[str, Any]]]:
    """Reads the rows of a run as dataset to method to row."""
    payload = data(run("b2", model), "table.json")
    found: dict[str, dict[str, dict[str, Any]]] = {}
    for values in payload["rows"]:
        row = dict(zip(payload["columns"], values, strict=True))
        found.setdefault(row["dataset"], {})[row["method"]] = row
    return found


def _differs(row: dict[str, Any], reader: dict[str, Any]) -> tuple[bool, bool]:
    """Recomputes the conservative interval of a difference of accuracies.

    Returns:
        Whether the interval excludes zero, and whether it lies above it.
    """
    p, q, n = row["accuracy"], reader["accuracy"], reader["n_test_pairs"]
    half = Z * math.sqrt(p * (1 - p) / n + q * (1 - q) / n)
    return abs(p - q) > half, p - q > half


def _expected(model: str) -> list[list[str]]:
    rows = []
    for place, (dataset, by) in enumerate(_table(model).items()):
        reader = by[READER]
        row = [
            paper.display_name(model) if place == 0 else "",
            params.display(dataset),
            cells.num(reader["accuracy"]),
        ]
        for method in (*BASELINES, *CONTROLS):
            star = "*" if _differs(by[method], reader)[0] else ""
            row.append(cells.num(by[method]["accuracy"]) + star)
        rows.append(row)
    return rows


def test_columns_sit_in_groups_by_the_kind_the_runs_give(
    output: registry.TableOutput,
) -> None:
    assert output.columns == COLUMNS
    assert output.align == "ll" + "r" * 9
    assert output.groups == (
        ("", 1, False),
        ("Finder of the paper", 1, False),
        ("Published baselines", 4, False),
        ("Controls", 4, False),
    )
    assert output.compact_print
    for model in models("b2"):
        kinds = data(run("b2", model), "summary.json")["kinds"]
        assert kinds == {
            READER: "reader",
            **dict.fromkeys(BASELINES, "baseline"),
            **dict.fromkeys(CONTROLS, "control"),
        }


def test_there_is_one_row_per_checkpoint_and_dataset_from_the_files(
    output: registry.TableOutput,
) -> None:
    want: list[list[str]] = []
    starts = []
    for model in models("b2"):
        starts.append(len(want))
        want += _expected(model)
    assert len(models("b2")) == 8
    assert len(want) == 32
    assert output.rows == want
    assert output.row_groups == tuple(starts)
    assert output.numbers["rows"] == len(want)
    assert {row[1] for row in output.rows} == {
        "sentiment",
        "cities",
        "larger than",
        "Spanish to English",
    }
    assert any(cell.endswith("*") for row in output.rows for cell in row)
    assert all("_" not in cell for row in output.rows for cell in row)


def test_the_sizes_of_the_runs_are_quoted_from_their_configs(
    output: registry.TableOutput,
) -> None:
    for model in models("b2"):
        path = run("b2", model) / "config.json"
        stored = json.loads(path.read_text("utf-8"))
        assert stored["n_seeds"] == output.numbers["seeds"] == 10
        assert stored["n_pairs"] == output.numbers["train_pairs"] == 64
    assert "10 training sets of 64 pairs" in output.detail


def test_the_widest_wilson_interval_is_quoted(
    output: registry.TableOutput,
) -> None:
    reach = max(
        max(
            row["accuracy"] - row["wilson_lo"],
            row["wilson_hi"] - row["accuracy"],
        )
        for model in models("b2")
        for by in _table(model).values()
        for row in by.values()
    )
    assert 0.05 < reach < 0.15
    assert output.numbers["wilson_reach"] == pytest.approx(reach)
    assert (
        "Wilson 95 percent interval of a value over the test pairs reaches "
        f"at most {cells.num(reach)} from it"
    ) in output.detail


def test_checkpoints_are_counted_per_role(
    output: registry.TableOutput,
) -> None:
    every = {m: m in models("b2") for m in _R["MANIFEST"].checkpoints()}
    assert output.counts["with_run"] == stats.per_role(every)
    above_random = {}
    baselines_above = {}
    for model in models("b2"):
        table = _table(model)
        above_random[model] = any(
            _differs(by[READER], by["random"])[1] for by in table.values()
        )
        baselines_above[model] = all(
            any(_differs(by[method], by[READER])[1] for by in table.values())
            for method in BASELINES
        )
    assert not above_random["gpt2-xl"]
    assert above_random["qwen3-1.7b-base"]
    assert output.counts["finder_above_random"] == stats.per_role(above_random)
    assert output.counts["every_baseline_above"] == stats.per_role(
        baselines_above
    )


def test_the_caption_names_baselines_and_controls_apart(
    output: registry.TableOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    for phrase in (
        "test accuracy",
        "layer",
        "validation",
        "Wilson",
        "asterisk",
        "conservative 95 percent interval",
        f"{params.display('mean_diff')} [[cite:marks2024geometry]]",
        f"{params.display('whitened_mean_diff')} [[cite:marks2024geometry]]",
        f"{params.display('logistic')} [[cite:alain2016understanding]]",
        f"{params.display('ccs')} [[cite:burns2023discovering]]",
        "published baselines",
        "The controls are",
        "labels shuffled",
        "Ledoit-Wolf",
        "one prompt",
        stats.per_role_text(output.counts["with_run"]),
    ):
        assert phrase in text, phrase
    controls = text[text.index("The controls are") :].split(". ")[0]
    assert "baseline" not in controls
    assert not any(d in text for d in DASHES)
    assert "_" not in text.replace("[[cite:", "")
