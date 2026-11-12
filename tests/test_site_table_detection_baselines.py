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
CONTROLS = ("popularity", "length")
COLUMNS = [
    "Model",
    "Direction",
    "Answer probe",
    "Prompt probe",
    "Semantic entropy",
    "EigenScore",
    "P(True)",
    "Token probability",
    "Popularity",
    "Length",
]
LEFT_OUT = {
    "neg_entropy": "Entropy of the first token",
    "answer_logprob": "Answer log-probability",
    "mean_token_logprob": "Mean token log-probability",
    "consistency": "Agreement of samples",
    "template": "Relation accuracy",
    "name_logprob": "Name log-probability",
}


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
        _within(by["answer_probe"]),
        cells.num(by["probe"]["auroc_within"]),
        _within(by["semantic_entropy"]),
        _within(by["eigenscore"]),
        *(
            cells.num(by[name]["auroc_within"])
            for name in ("p_true", "first_token_prob", *CONTROLS)
        ),
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
    assert output.roomy


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


def test_the_predictors_left_out_are_in_the_data_file(
    output: registry.TableOutput,
) -> None:
    shown = {
        READER,
        "answer_probe",
        "probe",
        "semantic_entropy",
        "eigenscore",
        "p_true",
        "first_token_prob",
        *CONTROLS,
    }
    for model in models("b1"):
        assert set(_popqa(model)) - shown == set(LEFT_OUT)
    assert output.data_columns == tuple(LEFT_OUT.values())
    assert list(output.data_cells) == [
        tuple(cells.num(_popqa(m)[name]["auroc_within"]) for name in LEFT_OUT)
        for m in models("b1")
    ]
    assert (
        "The runs also score six signals that the table leaves out and the "
        "data file holds: the entropy of the first token of the answer, the "
        "log-probability of the greedy answer and its mean per token, the "
        "share of sampled answers equal to the greedy one, the training "
        "accuracy of the relation, which is constant inside a relation, and "
        "the log-probability of the name of the subject."
    ) in output.detail


def test_the_claim_states_the_finding_by_point_estimate(
    output: registry.TableOutput,
) -> None:
    assert output.claim == (
        "By point estimate, the answer probe tells correct from wrong greedy "
        "answers on held-out PopQA questions better than the knowledge "
        "direction on "
        f"{stats.per_role_text(output.counts['answer_probe_above'])}, and "
        "the direction does better than P(True) and both controls on "
        f"{stats.per_role_text(output.counts['above_ptrue_and_controls'])}."
    )


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
        "pooled over the relations by their number of pairs of a correct "
        "and a wrong answer",
        "oriented so that a higher score means a more likely correct answer",
        "The direction, the answer probe, semantic entropy and EigenScore "
        "carry the percentile bootstrap interval",
        "with no step that locates the exact answer",
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
    assert "did not have" not in text
    assert "_" not in text.replace("[[cite:", "")
