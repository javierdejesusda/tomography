import json
from pathlib import Path
import runpy
import shutil
from typing import Any

import pytest

from tomography.site import registry
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.tables import steering_baselines
from tomography.viz import paper

_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
ROOT, data, models, run = _R["ROOT"], _R["data"], _R["models"], _R["run"]
SPEC = "steering_baselines"
DASHES = (chr(0x2014), chr(0x2013))
LABELS = {
    "reader": "Logistic regression",
    "prompt": "Prompt prepending",
    "diffmean": "DiffMean",
    "caa": "CAA",
    "actadd": "ActAdd",
    "mean_centring": f"Mean{chr(0x2011)}centring",
    "none": "Unsteered",
    "random/0": "Random direction",
}
ARMS = tuple(LABELS)
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
COLUMNS = ["Kind", "Arm", "Coefficient", "Overall", "Perplexity"]
DATA_COLUMNS = (
    "Model",
    "Target concept",
    "Classifier",
    "Judge score: Concept",
    "Judge score: Instruction",
    "Judge score: Fluency",
    "Distinct-1",
)


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


def _blank(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


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
                    _blank(row["coeff"], 2),
                    overall,
                    _blank(row["perplexity"], 1),
                ]
            )
    return headings, rows


def test_the_page_shows_coefficient_overall_and_perplexity(
    output: registry.TableOutput,
) -> None:
    assert output.columns == COLUMNS
    assert output.align == "llrrr"
    assert not output.groups
    assert output.span_labels
    assert not output.wide
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


def test_instruction_is_scored_on_the_post_trained_checkpoint_only(
    output: registry.TableOutput,
) -> None:
    place = DATA_COLUMNS.index("Judge score: Instruction")
    instruction = [cells[place] for cells in output.data_cells]
    assert instruction[:48] == [""] * 48
    assert all(cell.replace(".", "").isdigit() for cell in instruction[48:])
    for model in models("b3"):
        summary = data(run("b3", model), "summary.json")
        post_trained = summary["role"] == "post_trained"
        assert ("instruction" in summary["criteria"]) == post_trained


def test_cells_are_formatted_with_their_interval_and_blanks(
    output: registry.TableOutput,
) -> None:
    first = output.rows[0]
    assert first[:3] == ["Reader", "Logistic regression", "0.40"]
    assert first[3] == "0.500 [0.350, 0.650]"
    assert first[4] == "84.2"
    happiness = output.rows[8:16]
    assert happiness[5][1] == f"Mean{chr(0x2011)}centring"
    assert happiness[5][4] == "994.9"
    post_happiness = output.rows[56]
    assert post_happiness[3] == "0.975 [0.753, 1.202]"
    for start in range(0, 72, 8):
        prompt, none = output.rows[start + 1], output.rows[start + 6]
        assert prompt[2] == ""
        assert prompt[4] == ""
        assert none[2] == ""
        assert none[4] != ""
    assert all(
        "_" not in cell and "/" not in cell
        for row in output.rows
        for cell in row
    )


def test_the_data_file_keeps_the_columns_the_page_dropped(
    output: registry.TableOutput,
) -> None:
    assert output.data_columns == DATA_COLUMNS
    want = []
    for model in models("b3"):
        table = _table(model)
        summary = data(run("b3", model), "summary.json")
        for concept in summary["concepts"]:
            for arm in ARMS:
                row = table[(concept, arm)]
                want.append(
                    (
                        paper.display_name(model),
                        concept,
                        f"{row['classifier']:.3f}",
                        _blank(row["concept_score"]),
                        _blank(row.get("instruction_score")),
                        _blank(row["fluency_score"]),
                        _blank(row["distinct_1"]),
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
        "averaged over the 40 held-out prompts of the concept",
        "it chose on 20 separate tuning prompts",
        "at most 128 new tokens, cut at the end-of-sequence token",
        "that of 200 held-out WikiText-103 passages under the steered model",
    ):
        assert phrase in output.detail, phrase


def test_the_caption_says_how_the_table_was_judged_and_cites_each_source(
    output: registry.TableOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    concepts = data(run("b3", "gpt2-xl"), "summary.json")["concepts"]
    assert output.claim == (
        "The reader beside five published steering methods and two controls, "
        f"steering toward {stats.join_and(list(concepts.values()))} on one "
        "checkpoint per role."
    )
    assert "positive sentiment, happiness and anger" in output.claim
    for phrase in (
        "An LLM judge rates each generation 0, 1 or 2 on concept and fluency",
        "and on the post-trained checkpoint on how closely its topic relates "
        "to the instruction;",
        "Overall is the harmonic mean of the ratings, 0 when any is 0",
        "with a 95 percent bootstrap interval per arm",
        "Each arm uses the relative coefficient it chose on",
        "Baselines: prompt prepending and DiffMean",
        "whose protocol we adapt to each role",
        "The unsteered model and prompt prepending have no coefficient; "
        "prompt prepending has no perplexity.",
        "[[cite:wu2025axbench]]",
        "[[cite:marks2024geometry]]",
        "[[cite:rimsky2024steering]]",
        "[[cite:turner2023steering]]",
        "[[cite:jorgensen2023improving]]",
        "[[cite:dubois2023alpacafarm]]",
    ):
        assert phrase in text, phrase
    assert "following the instruction" not in text
    assert "[[table:" not in text
    assert "[[fig:" not in text
    assert "[[sec:" not in text
    assert len(text.split()) <= 170
    kinds = [kind for kind, _ in steering_baselines.ARMS.values()]
    assert kinds.count(steering_baselines.BASELINE) == 5
    assert kinds.count(steering_baselines.CONTROL) == 2
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


def _other_description(summary: dict[str, Any]) -> None:
    summary["concepts"]["anger"] = "rage"


def _other_tuning_half(config: dict[str, Any]) -> None:
    config["n_tune"] = 30


@pytest.mark.parametrize(
    ("model", "name", "change", "message"),
    [
        ("gpt2-xl", "table.json", _extra_arm, "b3/gpt2-xl holds other arms"),
        ("gpt2-xl", "table.json", _fewer_prompts, "scores other prompts"),
        ("qwen3-1.7b", "summary.json", _no_instruction, "other criteria"),
        ("gpt2-xl", "summary.json", _one_concept_less, "other concepts"),
        ("gpt2-xl", "summary.json", _other_description, "their concepts"),
        ("gpt2-xl", "config.json", _other_tuning_half, "in their sizes"),
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
