import json
from pathlib import Path
import runpy
import shutil
from typing import Any

import pytest

from tomography.site import registry
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.viz import paper

_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
ROOT, data, models, run = _R["ROOT"], _R["data"], _R["models"], _R["run"]
SPEC = "steering_baselines"
DASHES = (chr(0x2014), chr(0x2013))
ARMS = (
    "reader",
    "prompt",
    "caa",
    "mean_centring",
    "actadd",
    "diffmean",
    "none",
    "random/0",
)
LABELS = {
    "reader": "Logistic regression",
    "prompt": "Prompt prepending",
    "caa": "CAA",
    "mean_centring": f"Mean{chr(0x2011)}centring",
    "actadd": "ActAdd",
    "diffmean": "DiffMean",
    "none": "Unsteered",
    "random/0": "Random direction",
}
KINDS = [
    "Reader",
    "Published baselines",
    "",
    "",
    "",
    "",
    "Controls",
    "",
]
ROLES = {"base": "base", "post_trained": "post-trained"}
COEFFS = {0.05: "0.05", 0.1: "0.10", 0.2: "0.20", 0.4: "0.40", 0.8: "0.80"}
COLUMNS = [
    "Kind",
    "Arm",
    "Coefficient",
    "Concept",
    "Instruction",
    "Fluency",
    "Overall",
    "Perplexity",
    "Distinct-1",
]


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table(SPEC, Manifest(ROOT))


def _table(model: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Reads the held-out rows of a run as (concept, arm) to row."""
    payload = data(run("b3", model), "table.json")
    found = {}
    for values in payload["rows"]:
        row = dict(zip(payload["columns"], values, strict=True))
        found[(row["concept"], row["arm"])] = row
    return found


def _three(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _expected(model: str) -> tuple[list[str], list[list[str]]]:
    """Recomputes the headings and rows of one checkpoint from its files."""
    summary = data(run("b3", model), "summary.json")
    table = _table(model)
    headings = []
    rows = []
    for concept, description in summary["concepts"].items():
        headings.append(
            f"{paper.display_name(model)} ({ROLES[summary['role']]}), "
            f"steered toward {description}"
        )
        for kind, arm in zip(KINDS, ARMS, strict=True):
            row = table[(concept, arm)]
            overall = (
                f"{row['overall']:.3f} "
                f"[{row['overall_lo']:.3f}, {row['overall_hi']:.3f}]"
            )
            rows.append(
                [
                    kind,
                    LABELS[arm],
                    "" if row["coeff"] is None else COEFFS[row["coeff"]],
                    _three(row["concept_score"]),
                    _three(row.get("instruction_score")),
                    _three(row["fluency_score"]),
                    overall,
                    ""
                    if row["perplexity"] is None
                    else f"{row['perplexity']:.1f}",
                    _three(row["distinct_1"]),
                ]
            )
    return headings, rows


def test_columns_group_the_judge_scores(output: registry.TableOutput) -> None:
    assert output.columns == COLUMNS
    assert output.align == "ll" + "r" * 7
    assert output.groups == (
        ("", 2, False),
        ("Judge score", 4, False),
        ("", 2, False),
    )
    assert output.span_labels
    assert output.wide
    assert output.roomy
    assert output.compact_print


def test_rows_are_concept_by_arm_per_checkpoint_from_the_files(
    output: registry.TableOutput,
) -> None:
    want_rows: list[list[str]] = []
    want_headings: list[tuple[int, str]] = []
    for model in models("b3"):
        headings, rows = _expected(model)
        for place, text in enumerate(headings):
            want_headings.append((len(want_rows) + 8 * place, text))
        want_rows += rows
    assert models("b3") == ["gpt2-xl", "qwen3-1.7b-base", "qwen3-1.7b"]
    assert len(want_rows) == 72
    assert output.rows == want_rows
    assert output.headings == tuple(want_headings)
    assert output.numbers["rows"] == 72


def test_the_kinds_sit_in_separate_ruled_groups(
    output: registry.TableOutput,
) -> None:
    starts = [place for place, _ in output.headings]
    assert starts == list(range(0, 72, 8))
    for start in starts:
        block = output.rows[start : start + 8]
        assert [row[0] for row in block] == KINDS
        assert [row[1] for row in block] == [LABELS[a] for a in ARMS]
    assert output.row_groups == tuple(
        place for start in starts for place in (start + 1, start + 6)
    )


def test_each_role_has_its_own_rows_and_none_are_pooled(
    output: registry.TableOutput,
) -> None:
    per_role: dict[str, int] = {}
    owners: list[str] = []
    for model in models("b3"):
        role = data(run("b3", model), "summary.json")["role"]
        per_role[role] = per_role.get(role, 0) + 24
        owners += [paper.display_name(model)] * 24
    assert per_role == {"base": 48, "post_trained": 24}
    texts = [text for _, text in output.headings]
    assert sum("(base)" in t for t in texts) == 6
    assert sum("(post-trained)" in t for t in texts) == 3
    assert [cells[0] for cells in output.data_cells] == owners
    for place, text in output.headings:
        assert text.startswith(owners[place])


def test_instruction_is_shown_on_the_post_trained_checkpoint_only(
    output: registry.TableOutput,
) -> None:
    instruction = [row[4] for row in output.rows]
    assert instruction[:48] == [""] * 48
    assert all(cell.replace(".", "").isdigit() for cell in instruction[48:])
    for model in models("b3"):
        criteria = data(run("b3", model), "summary.json")["criteria"]
        role = data(run("b3", model), "summary.json")["role"]
        assert ("instruction" in criteria) == (role == "post_trained")


def test_cells_are_formatted_with_their_interval_and_blanks(
    output: registry.TableOutput,
) -> None:
    first = output.rows[0]
    assert first[:3] == ["Reader", "Logistic regression", "0.40"]
    assert first[6] == "0.500 [0.350, 0.650]"
    assert first[7] == "84.2"
    happiness = output.rows[8:16]
    assert happiness[3][1] == f"Mean{chr(0x2011)}centring"
    assert happiness[3][7] == "994.9"
    post_happiness = output.rows[56]
    assert post_happiness[6] == "0.975 [0.753, 1.202]"
    for start in range(0, 72, 8):
        prompt, none = output.rows[start + 1], output.rows[start + 6]
        assert prompt[2] == ""
        assert prompt[7] == ""
        assert none[2] == ""
        assert none[7] != ""
    assert all(
        "_" not in cell and "/" not in cell
        for row in output.rows
        for cell in row
    )


def test_the_data_file_keeps_the_checkpoint_concept_and_classifier(
    output: registry.TableOutput,
) -> None:
    assert output.data_columns == ("Model", "Target concept", "Classifier")
    want = []
    for model in models("b3"):
        table = _table(model)
        summary = data(run("b3", model), "summary.json")
        for concept in summary["concepts"]:
            for arm in ARMS:
                want.append(
                    (
                        paper.display_name(model),
                        concept,
                        f"{table[(concept, arm)]['classifier']:.3f}",
                    )
                )
    assert list(output.data_cells) == want


def test_the_sizes_are_quoted_from_the_runs(
    output: registry.TableOutput,
) -> None:
    for model in models("b3"):
        path = run("b3", model) / "config.json"
        stored = json.loads(path.read_text("utf-8"))
        assert stored["n_tune"] == output.numbers["tune"] == 20
        assert stored["n_held_out"] == output.numbers["held_out"] == 40
        assert stored["max_new_tokens"] == output.numbers["new_tokens"] == 128
        assert stored["n_pairs"] == output.numbers["train_pairs"] == 64
        assert stored["n_neutral"] == output.numbers["neutral"] == 200
        assert stored["n_boot"] == output.numbers["resamples"] == 2000
        assert all(row["n_prompts"] == 40 for row in _table(model).values())
    for phrase in (
        "40 held-out prompts",
        "20 tuning prompts",
        "128 new tokens",
        "64 training pairs",
        "200 held-out WikiText-103 passages",
        "2,000 resamples",
    ):
        assert phrase in output.detail, phrase


def test_the_caption_says_how_the_table_was_judged_and_cites_each_source(
    output: registry.TableOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    for phrase in (
        "LLM judge",
        "a second judge slot",
        "same model",
        "harmonic mean",
        "the coefficient that the arm chose on the",
        "share no prompt",
        "95 percent percentile bootstrap interval over the held-out prompts",
        "published baselines",
        "The controls are",
        "[[cite:wu2025axbench]]",
        "[[cite:marks2024geometry]]",
        "[[cite:rimsky2024steering]]",
        "[[cite:turner2023steering]]",
        "[[cite:jorgensen2023improving]]",
        "[[cite:dubois2023alpacafarm]]",
        stats.per_role_text(output.counts["with_run"]),
    ):
        assert phrase in text, phrase
    controls = text[text.index("The controls are") :].split(". ")[0]
    assert "baseline" not in controls
    assert not any(d in text for d in DASHES)
    for verdict in ("better", "beats", "outperform", "worse", "apart"):
        assert verdict not in text, verdict


def test_checkpoints_are_counted_per_role(
    output: registry.TableOutput,
) -> None:
    every = {m: m in models("b3") for m in _R["MANIFEST"].checkpoints()}
    assert output.counts["with_run"] == stats.per_role(every)
    with_run = output.counts["with_run"]
    assert [with_run[r].hits for r in stats.ROLES] == [1, 1, 1]


def _copy(tmp_path: Path) -> Path:
    """Copies the files the table reads from every b3 run."""
    for model in models("b3"):
        source = run("b3", model)
        target = tmp_path / "b3" / model / source.name
        target.mkdir(parents=True)
        for name in ("config.json", "meta.json", "summary.json", "table.json"):
            shutil.copyfile(source / name, target / name)
    return tmp_path


def _edit(root: Path, model: str, name: str, change: Any) -> None:
    """Rewrites one file of a copied run through `change`."""
    (path,) = (root / "b3" / model).glob(f"*/{name}")
    stored = json.loads(path.read_text("utf-8"))
    change(stored if name == "config.json" else stored["data"])
    path.write_text(json.dumps(stored), encoding="utf-8")


def _extra_arm(table: dict[str, Any]) -> None:
    table["rows"].append([*table["rows"][-1]])
    table["rows"][-1][1] = "random/1"


def _fewer_prompts(table: dict[str, Any]) -> None:
    table["rows"][0][table["columns"].index("n_prompts")] = 39


def _no_instruction(summary: dict[str, Any]) -> None:
    summary["criteria"].remove("instruction")


def _one_concept_less(summary: dict[str, Any]) -> None:
    del summary["concepts"]["anger"]


def _other_tuning_half(config: dict[str, Any]) -> None:
    config["n_tune"] = 30


@pytest.mark.parametrize(
    ("model", "name", "change", "message"),
    [
        ("gpt2-xl", "table.json", _extra_arm, "b3/gpt2-xl holds other arms"),
        ("gpt2-xl", "table.json", _fewer_prompts, "scores other prompts"),
        ("qwen3-1.7b", "summary.json", _no_instruction, "other criteria"),
        ("gpt2-xl", "summary.json", _one_concept_less, "other concepts"),
        ("gpt2-xl", "config.json", _other_tuning_half, "differ in their"),
    ],
)
def test_a_run_that_breaks_the_layout_is_refused(
    tmp_path: Path, model: str, name: str, change: Any, message: str
) -> None:
    root = _copy(tmp_path)
    registry.build_table(SPEC, Manifest(root))
    _edit(root, model, name, change)
    with pytest.raises(ValueError, match=message):
        registry.build_table(SPEC, Manifest(root))
