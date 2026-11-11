from pathlib import Path
import runpy
from typing import Any

import pytest

from tomography.site import cells
from tomography.site import registry
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.viz import paper

_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
ROOT, data, models, run = _R["ROOT"], _R["data"], _R["models"], _R["run"]
SPEC = "detection_baselines"
DASHES = (chr(0x2014), chr(0x2013))
READER = "lat"
WITH_INTERVAL = ("answer_probe", "semantic_entropy", "eigenscore")
PLAIN = ("p_true", "first_token_prob", "probe")
CONTROLS = ("popularity", "length")
COLUMNS = [
    "Model",
    "Direction",
    "Answer probe",
    "Semantic entropy",
    "EigenScore",
    "P(True)",
    "Token probability",
    "Prompt probe",
    "Popularity",
    "Length",
]


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table(SPEC, Manifest(ROOT))


def _popqa(model: str) -> dict[str, dict[str, Any]]:
    """Reads the PopQA rows of a run as predictor to row."""
    payload = data(run("b1", model), "table.json")
    rows = [
        dict(zip(payload["columns"], r, strict=True)) for r in payload["rows"]
    ]
    return {r["predictor"]: r for r in rows if r["dataset"] == "popqa"}


def _within(row: dict[str, Any]) -> str:
    return cells.interval(
        row["auroc_within"], row["lo_within"], row["hi_within"]
    )


def _expected(model: str) -> list[str]:
    by = _popqa(model)
    return [
        paper.display_name(model),
        _within(by[READER]) if READER in by else "no contrast",
        *(_within(by[name]) for name in WITH_INTERVAL),
        *(cells.num(by[name]["auroc_within"]) for name in (*PLAIN, *CONTROLS)),
    ]


def test_columns_sit_in_groups_of_reader_baselines_and_controls(
    output: registry.TableOutput,
) -> None:
    assert output.columns == COLUMNS
    assert output.align == "l" + "r" * 9
    assert output.groups == (
        ("Reader", 1, False),
        ("Published baselines", 6, False),
        ("Controls", 2, False),
    )
    assert output.compact_print


def test_there_is_one_row_per_checkpoint_from_the_files(
    output: registry.TableOutput,
) -> None:
    found = models("b1")
    assert len(found) == 8
    assert output.rows == [_expected(model) for model in found]
    assert output.row_groups == (0, 4)
    assert output.rows[0][1] == "no contrast"
    assert "lat" not in _popqa("gpt2")
    assert data(run("b1", "gpt2"), "summary.json")["missing_references"][
        "popqa"
    ] == ["lat"]


def test_the_number_of_questions_is_quoted_from_the_runs(
    output: registry.TableOutput,
) -> None:
    sizes = {
        row["n"] for model in models("b1") for row in _popqa(model).values()
    }
    assert sizes == {1937}
    assert output.numbers["questions"] == 1937
    assert "1,937 test questions" in output.detail


def test_checkpoints_are_counted_per_role(
    output: registry.TableOutput,
) -> None:
    every = {m: m in models("b1") for m in _R["MANIFEST"].checkpoints()}
    assert output.counts["with_run"] == stats.per_role(every)
    probe_above = {}
    above_controls = {}
    for model in models("b1"):
        by = _popqa(model)
        if READER not in by:
            continue
        ours = by[READER]["auroc_within"]
        probe_above[model] = by["answer_probe"]["auroc_within"] > ours
        above_controls[model] = all(
            ours > by[name]["auroc_within"] for name in ("p_true", *CONTROLS)
        )
    assert len(probe_above) == 7
    assert output.counts["answer_probe_above"] == stats.per_role(probe_above)
    assert output.counts["above_ptrue_and_controls"] == stats.per_role(
        above_controls
    )


def test_the_caption_names_baselines_and_controls_apart(
    output: registry.TableOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    for phrase in (
        "AUROC",
        "inside each PopQA relation",
        "greedy",
        "bootstrap interval",
        "[[cite:orgad2025llms]]",
        "[[cite:farquhar2024detecting]]",
        "[[cite:chen2024inside]]",
        "[[cite:kadavath2022language]]",
        "[[cite:alain2016understanding]]",
        "[[cite:mallen2023when]]",
        "five answers",
        "published baselines",
        "are controls",
        "no direction",
        "TriviaQA and NQ-open",
        stats.per_role_text(output.counts["with_run"]),
    ):
        assert phrase in text, phrase
    controls = [s for s in text.split(". ") if "are controls" in s]
    assert len(controls) == 1
    assert "baseline" not in controls[0]
    assert not any(d in text for d in DASHES)
    assert "_" not in text.replace("[[cite:", "")
