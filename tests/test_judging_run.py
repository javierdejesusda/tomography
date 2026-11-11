"""Tests for exporting, importing and scoring the judgments of a run."""

from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Sequence
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tomography import judges
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.judging import run as run_module
from tomography.judging.batches import Export
from tomography.judging.judgments import JudgmentError
from tomography.judging.rubric import rubric_sha256
from tomography.judging.rubric import rubric_text
from tomography.judging.run import Judged
from tomography.judging.run import Record
from tomography.judging.run import agreement
from tomography.judging.run import export_stage
from tomography.judging.run import held_out_rows
from tomography.judging.run import import_judgments
from tomography.judging.run import judge_scores
from tomography.judging.run import read_selection
from tomography.judging.run import select_coefficients
from tomography.judging.run import tuning_rows
from tomography.judging.scores import SELECTION_RULE
from tomography.judging.scores import weighted_kappa
from tomography.reading.metrics import bootstrap_ci

CRITERIA = ("concept", "fluency")
DESCRIPTION = "positive sentiment"
PROMPTS = {"tune": ("t0", "t1", "t2"), "held_out": ("h0", "h1", "h2")}
ARMS: dict[str, tuple[float | None, ...]] = {
    "caa": (0.1, 0.2),
    "none": (None,),
}
# The second sample of this tuning prompt came back empty.
EMPTY = "sentiment/tune/caa/0.2/1/1"

Scoring = Callable[[str], tuple[int, int]]

# The folder each stage of a run was last exported to.
EXPORTED: dict[tuple[Path, str], Path] = {}


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Config(ExperimentConfig):
    experiment: str = "t1"
    model: str = "gpt2"


def _judged(empty: Collection[str] = (EMPTY,)) -> Judged:
    records = []
    for split, prompts in PROMPTS.items():
        for arm, coeffs in ARMS.items():
            for coeff in coeffs:
                for p, prompt in enumerate(prompts):
                    for sample in range(2):
                        label = "none" if coeff is None else f"{coeff:g}"
                        key = f"sentiment/{split}/{arm}/{label}/{p}/{sample}"
                        records.append(
                            Record(
                                key=key,
                                concept="sentiment",
                                description=DESCRIPTION,
                                arm=arm,
                                coeff=coeff,
                                split=split,
                                prompt_index=p,
                                prompt=prompt,
                                text="" if key in empty else f"text of {key}",
                            )
                        )
    return Judged(criteria=CRITERIA, records=tuple(records))


@pytest.fixture
def run_dir(tmp_path: Path) -> RunDir:
    run = RunDir.create(_Config(seed=3), out_root=tmp_path / "results")
    run.write_json("stored", {"value": 1})
    return run


def _digest(run: RunDir) -> str:
    """Hashes every file of a run directory with its relative path."""
    digest = hashlib.sha256()
    for path in sorted(run.path.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(run.path).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _plan(run: RunDir, stage: str) -> Export:
    stored = run.read_json(f"judge_export_{stage}")
    assert isinstance(stored, dict)
    return Export.from_dict(stored)


def _by_key(key: str) -> tuple[int, int]:
    """Scores a generation from its key: 0.2 beats 0.1 on the concept."""
    _, _, arm, coeff, prompt, _ = key.split("/")
    if arm == "none":
        return 0, 2
    return (2 if coeff == "0.2" else 1), (1 if prompt == "0" else 2)


def _answer(
    run: RunDir, stage: str, scoring: Scoring, *, judge: int | None = None
) -> list[str]:
    """Imports judgments for the batches of a stage and returns their ids."""
    done = []
    for batch in _plan(run, stage).batches:
        if judge is not None and batch.judge != judge:
            continue
        lines = []
        for item_id, key in zip(batch.item_ids, batch.keys, strict=True):
            concept, fluency = scoring(key)
            lines.append(
                json.dumps(
                    {"item_id": item_id, "concept": concept, "fluency": fluency}
                )
            )
        import_judgments(
            run,
            batch.batch_id,
            "\n".join(lines),
            judge_model="judge-model-1",
            export_dir=EXPORTED[run.path, stage],
        )
        done.append(batch.batch_id)
    return done


def _export(
    run: RunDir,
    stage: str,
    out: Path,
    *,
    judged: Judged | None = None,
    **options: object,
) -> list[Path]:
    arguments: dict[str, object] = {"batch_size": 5, "second_fraction": 0.25}
    arguments.update(options)
    EXPORTED[run.path, stage] = out
    return export_stage(run, judged or _judged(), stage, out, **arguments)  # type: ignore[arg-type]


def test_the_tuning_export_holds_the_non_empty_tuning_generations(
    run_dir: RunDir, tmp_path: Path
) -> None:
    paths = _export(run_dir, "tune", tmp_path / "out")

    plan = _plan(run_dir, "tune")
    wanted = {r.key for r in _judged().records if r.split == "tune" and r.text}
    first = [k for b in plan.batches if b.judge == 1 for k in b.keys]
    assert sorted(first) == sorted(wanted)
    assert len(wanted) == 17
    assert EMPTY not in first
    assert plan.criteria == CRITERIA
    assert plan.rubric_sha256 == rubric_sha256(rubric_text())
    assert plan.seed == derive_seed(3, "judge", "tune")
    assert sorted(p.name for p in paths) == sorted(
        [
            *(f"{b.batch_id}.jsonl" for b in plan.batches),
            "criteria.json",
            "rubric.txt",
        ]
    )
    assert (tmp_path / "out" / "rubric.txt").read_text("utf-8") == rubric_text()
    row = json.loads(paths[0].read_text("utf-8").splitlines()[0])
    assert row["concept"] == DESCRIPTION
    assert row["prompt"] in PROMPTS["tune"]


def test_an_export_can_be_repeated_byte_for_byte(
    run_dir: RunDir, tmp_path: Path
) -> None:
    first = _export(run_dir, "tune", tmp_path / "a")
    stored = _digest(run_dir)

    second = _export(run_dir, "tune", tmp_path / "b")

    assert _digest(run_dir) == stored
    assert [p.name for p in first] == [p.name for p in second]
    for one, other in zip(first, second, strict=True):
        assert one.read_bytes() == other.read_bytes()


def test_a_given_seed_replaces_the_seed_derived_from_the_run(
    run_dir: RunDir, tmp_path: Path
) -> None:
    _export(run_dir, "tune", tmp_path / "a")
    derived = _plan(run_dir, "tune")

    _export(run_dir, "tune", tmp_path / "b", seed=99)

    assert _plan(run_dir, "tune").seed == 99
    assert _plan(run_dir, "tune").batches != derived.batches


def test_an_import_stores_the_judgments_with_the_identity_of_the_judge(
    run_dir: RunDir, tmp_path: Path
) -> None:
    _export(run_dir, "tune", tmp_path / "out")
    batch = _plan(run_dir, "tune").batches[0]
    lines = [
        json.dumps({"item_id": item_id, "fluency": 2, "concept": i % 3})
        for i, item_id in enumerate(batch.item_ids)
    ]

    path = import_judgments(
        run_dir,
        batch.batch_id,
        "\n".join(reversed(lines)),
        judge_model=" judge-model-1 ",
        export_dir=tmp_path / "out",
    )

    assert path == run_dir.path / f"judgments_{batch.batch_id}.json"
    assert run_dir.read_json(f"judgments_{batch.batch_id}") == {
        "batch_id": batch.batch_id,
        "stage": "tune",
        "judge": 1,
        "judge_model": "judge-model-1",
        "rubric_sha256": rubric_sha256(rubric_text()),
        "batch_sha256": hashlib.sha256(
            (tmp_path / "out" / f"{batch.batch_id}.jsonl").read_bytes()
        ).hexdigest(),
        "criteria": ["concept", "fluency"],
        "judgments": [
            {"item_id": item_id, "concept": i % 3, "fluency": 2}
            for i, item_id in enumerate(batch.item_ids)
        ],
    }


def _valid(run: RunDir, batch_id: str) -> str:
    batch = _plan(run, "tune").batch(batch_id)
    return "\n".join(
        json.dumps({"item_id": item_id, "concept": 1, "fluency": 1})
        for item_id in batch.item_ids
    )


@pytest.mark.parametrize(
    ("corrupt", "error", "message"),
    [
        (lambda text: text.rsplit("\n", 1)[0], JudgmentError, "missing 1 item"),
        (
            lambda text: text + "\n" + text.splitlines()[0],
            JudgmentError,
            "duplicate item",
        ),
        (
            lambda text: text.replace('"concept": 1', '"concept": 3', 1),
            JudgmentError,
            "concept must be 0, 1 or 2, got 3",
        ),
        (
            lambda text: text.replace('"fluency"', '"instruction"', 1),
            JudgmentError,
            "unexpected fields: instruction",
        ),
        (
            lambda text: (
                text + '\n{"item_id": "ff", "concept": 1, "fluency": 1}'
            ),
            JudgmentError,
            "unknown item 'ff'",
        ),
    ],
    ids=["missing", "duplicate", "out_of_range", "wrong_criterion", "unknown"],
)
def test_a_refused_import_leaves_the_stored_data_untouched(
    run_dir: RunDir,
    tmp_path: Path,
    corrupt: Callable[[str], str],
    error: type[Exception],
    message: str,
) -> None:
    _export(run_dir, "tune", tmp_path / "out")
    batch_id = _plan(run_dir, "tune").batches[0].batch_id
    before = _digest(run_dir)

    with pytest.raises(error, match=message):
        import_judgments(
            run_dir,
            batch_id,
            corrupt(_valid(run_dir, batch_id)),
            judge_model="judge-model-1",
            export_dir=tmp_path / "out",
        )

    assert _digest(run_dir) == before
    assert not (run_dir.path / f"judgments_{batch_id}.json").exists()


def test_an_import_needs_a_known_batch_a_judge_and_no_earlier_import(
    run_dir: RunDir, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="no export"):
        import_judgments(
            run_dir, "0a0a0a0a", "", judge_model="m", export_dir=tmp_path
        )
    out = tmp_path / "out"
    _export(run_dir, "tune", out)
    batch_id = _plan(run_dir, "tune").batches[0].batch_id
    text = _valid(run_dir, batch_id)
    before = _digest(run_dir)

    with pytest.raises(ValueError, match="unknown batch '0a0a0a0a'"):
        import_judgments(
            run_dir, "0a0a0a0a", text, judge_model="m", export_dir=out
        )
    with pytest.raises(ValueError, match="judge_model"):
        import_judgments(
            run_dir, batch_id, text, judge_model="  ", export_dir=out
        )
    assert _digest(run_dir) == before

    import_judgments(run_dir, batch_id, text, judge_model="m", export_dir=out)
    stored = _digest(run_dir)
    with pytest.raises(ValueError, match="already imported"):
        import_judgments(
            run_dir, batch_id, text, judge_model="other", export_dir=out
        )
    assert _digest(run_dir) == stored


def test_scores_come_back_per_generation_and_an_empty_one_scores_zero(
    run_dir: RunDir, tmp_path: Path
) -> None:
    _export(run_dir, "tune", tmp_path / "out")
    _answer(run_dir, "tune", _by_key)

    scores = judge_scores(run_dir, _judged(), stages=("tune",))

    assert set(scores) == {
        r.key for r in _judged().records if r.split == "tune"
    }
    assert scores["sentiment/tune/caa/0.2/0/0"].tolist() == [2, 1]
    assert scores["sentiment/tune/none/none/2/1"].tolist() == [0, 2]
    assert scores[EMPTY].tolist() == [0, 0]
    assert scores[EMPTY].dtype == np.int64


def test_scores_need_every_batch_of_the_first_judge(
    run_dir: RunDir, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="stage 'tune' has no export"):
        judge_scores(run_dir, _judged(), stages=("tune",))
    _export(run_dir, "tune", tmp_path / "out")
    _answer(run_dir, "tune", _by_key, judge=2)
    first = [b.batch_id for b in _plan(run_dir, "tune").batches if b.judge == 1]

    with pytest.raises(ValueError, match="not imported") as error:
        judge_scores(run_dir, _judged(), stages=("tune",))

    assert f"not imported: {', '.join(first)}" in str(error.value)


def test_the_coefficient_with_the_highest_tuning_mean_is_selected(
    run_dir: RunDir, tmp_path: Path
) -> None:
    _export(run_dir, "tune", tmp_path / "out")
    _answer(run_dir, "tune", _by_key)
    scores = judge_scores(run_dir, _judged(), stages=("tune",))

    rows = tuning_rows(_judged(), scores)
    selected = select_coefficients(_judged(), scores)

    # Per item: (1, 1) gives 1, (1, 2) gives 4/3, (2, 1) gives 4/3 and
    # (2, 2) gives 2; the empty generation of 0.2 gives 0.
    assert rows == [
        {
            "concept": "sentiment",
            "arm": "caa",
            "coeff": 0.1,
            "n_items": 6,
            "n_empty": 0,
            "overall": pytest.approx((2 * 1 + 4 * 4 / 3) / 6),
        },
        {
            "concept": "sentiment",
            "arm": "caa",
            "coeff": 0.2,
            "n_items": 6,
            "n_empty": 1,
            "overall": pytest.approx((2 * 4 / 3 + 3 * 2 + 0) / 6),
        },
        {
            "concept": "sentiment",
            "arm": "none",
            "coeff": None,
            "n_items": 6,
            "n_empty": 0,
            "overall": 0.0,
        },
    ]
    assert selected == {("sentiment", "caa"): 0.2, ("sentiment", "none"): None}


def test_the_held_out_export_waits_for_the_tuning_judgments(
    run_dir: RunDir, tmp_path: Path
) -> None:
    _export(run_dir, "tune", tmp_path / "out")
    before = _digest(run_dir)

    with pytest.raises(ValueError, match="not imported"):
        _export(run_dir, "held_out", tmp_path / "held")

    assert _digest(run_dir) == before
    assert not (tmp_path / "held").exists()


def test_the_held_out_export_holds_only_the_selected_coefficient(
    run_dir: RunDir, tmp_path: Path
) -> None:
    _export(run_dir, "tune", tmp_path / "out")
    _answer(run_dir, "tune", _by_key)

    _export(run_dir, "held_out", tmp_path / "out")

    plan = _plan(run_dir, "held_out")
    first = sorted(k for b in plan.batches if b.judge == 1 for k in b.keys)
    assert first == sorted(
        f"sentiment/held_out/{arm}/{p}/{sample}"
        for arm in ("caa/0.2", "none/none")
        for p in range(3)
        for sample in range(2)
    )
    assert plan.seed == derive_seed(3, "judge", "held_out")
    tune = _plan(run_dir, "tune")
    assert not {i for b in tune.batches for i in b.item_ids} & {
        i for b in plan.batches for i in b.item_ids
    }


def test_the_held_out_export_needs_every_tuning_batch_of_both_judges(
    run_dir: RunDir, tmp_path: Path
) -> None:
    _export(run_dir, "tune", tmp_path / "out")
    _answer(run_dir, "tune", _by_key, judge=1)
    second = [
        b.batch_id for b in _plan(run_dir, "tune").batches if b.judge == 2
    ]
    before = _digest(run_dir)

    with pytest.raises(ValueError, match="needs every batch") as error:
        _export(run_dir, "held_out", tmp_path / "held")

    assert second
    assert f"not imported: {', '.join(second)}" in str(error.value)
    assert _digest(run_dir) == before
    assert not (tmp_path / "held").exists()


def test_the_held_out_export_stores_the_selection_with_its_rule(
    run_dir: RunDir, tmp_path: Path
) -> None:
    _export(run_dir, "tune", tmp_path / "out")
    _answer(run_dir, "tune", _by_key)
    assert read_selection(run_dir) is None

    _export(run_dir, "held_out", tmp_path / "out")

    assert run_dir.read_json("judge_selection") == {
        "split": "tune",
        "rule": SELECTION_RULE,
        "selected": [
            {
                "concept": "sentiment",
                "arm": "caa",
                "coeff": 0.2,
                "overall": pytest.approx((2 * 4 / 3 + 3 * 2 + 0) / 6),
            },
            {
                "concept": "sentiment",
                "arm": "none",
                "coeff": None,
                "overall": 0.0,
            },
        ],
    }
    assert "highest mean overall score on the tuning half" in SELECTION_RULE
    assert "tie goes to the smallest magnitude" in SELECTION_RULE
    assert read_selection(run_dir) == {
        ("sentiment", "caa"): 0.2,
        ("sentiment", "none"): None,
    }
    stored = _digest(run_dir)
    _export(run_dir, "held_out", tmp_path / "again")
    assert _digest(run_dir) == stored


def test_an_empty_held_out_generation_is_counted_and_scores_zero(
    run_dir: RunDir, tmp_path: Path
) -> None:
    held = "sentiment/held_out/caa/0.2/1/1"
    judged = _judged(empty=(EMPTY, held))
    _export(run_dir, "tune", tmp_path / "out", judged=judged)
    _answer(run_dir, "tune", _by_key)
    _export(run_dir, "held_out", tmp_path / "out", judged=judged)
    exported = {k for b in _plan(run_dir, "held_out").batches for k in b.keys}
    _answer(run_dir, "held_out", _by_key)
    scores = judge_scores(run_dir, judged)

    rows = held_out_rows(
        judged, scores, select_coefficients(judged, scores), n_boot=50, seed=5
    )

    assert held not in exported
    assert scores[held].tolist() == [0, 0]
    assert (rows[0]["arm"], rows[0]["coeff"]) == ("caa", 0.2)
    assert (rows[0]["n_items"], rows[0]["n_empty"]) == (6, 1)
    # Prompt 0 scores (2, 1) twice, prompt 1 holds one (2, 2) and the empty
    # generation, prompt 2 scores (2, 2) twice.
    assert rows[0]["overall"] == pytest.approx((4 / 3 + 1.0 + 2.0) / 3)
    assert rows[0]["concept_score"] == pytest.approx((2.0 + 1.0 + 2.0) / 3)
    assert rows[1]["n_empty"] == 0


def test_the_unsteered_arm_is_exported_in_both_stages_like_any_other(
    run_dir: RunDir, tmp_path: Path
) -> None:
    _export(run_dir, "tune", tmp_path / "out")
    _answer(run_dir, "tune", _by_key)
    _export(run_dir, "held_out", tmp_path / "out")

    for stage in ("tune", "held_out"):
        plan = _plan(run_dir, stage)
        arm_of = {
            item_id: key.split("/")[2]
            for b in plan.batches
            for item_id, key in zip(b.item_ids, b.keys, strict=True)
        }
        lines = [
            json.loads(line)
            for b in plan.batches
            for line in (tmp_path / "out" / f"{b.batch_id}.jsonl")
            .read_text("utf-8")
            .splitlines()
        ]
        first = [k for b in plan.batches if b.judge == 1 for k in b.keys]
        assert sum("/none/" in key for key in first) == 6
        assert {arm_of[row["item_id"]] for row in lines} == {"none", "caa"}
        for row in lines:
            assert sorted(row) == ["concept", "generation", "item_id", "prompt"]
            assert row["concept"] == DESCRIPTION
            assert row["prompt"] in PROMPTS[stage]


def test_each_stage_has_a_seeded_second_subset_under_other_item_ids(
    run_dir: RunDir, tmp_path: Path
) -> None:
    other = RunDir.create(_Config(seed=3), out_root=tmp_path / "other")
    reseeded = RunDir.create(_Config(seed=4), out_root=tmp_path / "reseeded")
    for run in (run_dir, other, reseeded):
        _export(run, "tune", tmp_path / "out")
        _answer(run, "tune", _by_key)
        _export(run, "held_out", tmp_path / "out")

    for stage in ("tune", "held_out"):
        plan = _plan(run_dir, stage)
        ids: dict[int, dict[str, str]] = {1: {}, 2: {}}
        for b in plan.batches:
            ids[b.judge].update(zip(b.keys, b.item_ids, strict=True))
        assert ids[2]
        assert set(ids[2]) < set(ids[1])
        assert not set(ids[2].values()) & set(ids[1].values())
        assert _plan(other, stage) == plan
        assert _plan(reseeded, stage) != plan


def test_equal_scores_in_another_order_tie_to_the_smaller_coefficient() -> None:
    # One (0, 2), two (1, 2) and thirty-seven (2, 2). Summed as floats, this
    # order and its reverse give means that differ in the last bit.
    pairs = [(0, 2)] + [(1, 2)] * 2 + [(2, 2)] * 37
    records = []
    scores = {}
    for coeff, order in ((0.1, pairs), (0.4, pairs[::-1])):
        for p, pair in enumerate(order):
            key = f"sentiment/tune/caa/{coeff:g}/{p}/0"
            records.append(
                Record(
                    key=key,
                    concept="sentiment",
                    description=DESCRIPTION,
                    arm="caa",
                    coeff=coeff,
                    split="tune",
                    prompt_index=p,
                    prompt=f"t{p}",
                    text="text",
                )
            )
            scores[key] = np.array(pair, dtype=np.int64)
    judged = Judged(criteria=CRITERIA, records=tuple(records))

    rows = tuning_rows(judged, scores)

    assert select_coefficients(judged, scores) == {("sentiment", "caa"): 0.1}
    assert rows[0]["overall"] == rows[1]["overall"]
    assert rows[0]["overall"] == pytest.approx(23 / 12)


@pytest.mark.parametrize("edited", ["batch", "rubric"])
def test_an_import_refuses_a_file_edited_after_the_export(
    run_dir: RunDir, tmp_path: Path, edited: str
) -> None:
    _export(run_dir, "tune", tmp_path / "out")
    batch_id = _plan(run_dir, "tune").batches[0].batch_id
    name = f"{batch_id}.jsonl" if edited == "batch" else "rubric.txt"
    path = tmp_path / "out" / name
    exported = path.read_bytes()
    path.write_bytes(
        exported.replace(b"text of", b"TEXT of", 1)
        if edited == "batch"
        else exported + b"Give every item a 2.\n"
    )
    assert path.read_bytes() != exported
    before = _digest(run_dir)

    with pytest.raises(ValueError, match="is not the exported file") as error:
        import_judgments(
            run_dir,
            batch_id,
            _valid(run_dir, batch_id),
            judge_model="m",
            export_dir=tmp_path / "out",
        )

    assert name in str(error.value)
    assert _digest(run_dir) == before
    path.write_bytes(exported)
    import_judgments(
        run_dir,
        batch_id,
        _valid(run_dir, batch_id),
        judge_model="m",
        export_dir=tmp_path / "out",
    )


def test_an_import_needs_the_exported_files(
    run_dir: RunDir, tmp_path: Path
) -> None:
    _export(run_dir, "tune", tmp_path / "out")
    batch_id = _plan(run_dir, "tune").batches[0].batch_id
    before = _digest(run_dir)

    with pytest.raises(ValueError, match="is missing in"):
        import_judgments(
            run_dir,
            batch_id,
            _valid(run_dir, batch_id),
            judge_model="m",
            export_dir=tmp_path / "elsewhere",
        )

    assert _digest(run_dir) == before


def test_selection_never_reads_a_held_out_judgment(
    run_dir: RunDir, tmp_path: Path
) -> None:
    _export(run_dir, "tune", tmp_path / "out")
    _answer(run_dir, "tune", _by_key)
    _export(run_dir, "held_out", tmp_path / "out")
    _answer(run_dir, "held_out", lambda key: (0, 0))

    scores = judge_scores(run_dir, _judged())

    assert select_coefficients(_judged(), scores) == {
        ("sentiment", "caa"): 0.2,
        ("sentiment", "none"): None,
    }
    assert scores["sentiment/held_out/caa/0.2/0/0"].tolist() == [0, 0]
    assert "sentiment/held_out/caa/0.1/0/0" not in scores


def test_the_held_out_rows_are_hand_computed_with_bootstrap_intervals(
    run_dir: RunDir, tmp_path: Path
) -> None:
    _export(run_dir, "tune", tmp_path / "out")
    _answer(run_dir, "tune", _by_key)
    _export(run_dir, "held_out", tmp_path / "out")
    _answer(run_dir, "held_out", _by_key)
    judged = _judged()
    scores = judge_scores(run_dir, judged)
    selected = select_coefficients(judged, scores)

    rows = held_out_rows(judged, scores, selected, n_boot=200, seed=5)

    # Prompt 0 scores (2, 1) twice, prompts 1 and 2 score (2, 2) twice.
    per_prompt = np.array([4 / 3, 2.0, 2.0])
    lo, hi = bootstrap_ci(
        per_prompt, n_boot=200, seed=derive_seed(5, "sentiment", "caa")
    )
    assert rows[0] == {
        "concept": "sentiment",
        "arm": "caa",
        "coeff": 0.2,
        "n_prompts": 3,
        "n_items": 6,
        "n_empty": 0,
        "concept_score": pytest.approx(2.0),
        "fluency_score": pytest.approx(5 / 3),
        "overall": pytest.approx(per_prompt.mean()),
        "overall_lo": pytest.approx(float(lo)),
        "overall_hi": pytest.approx(float(hi)),
    }
    assert rows[1]["arm"] == "none"
    assert rows[1]["coeff"] is None
    assert rows[1]["overall"] == 0.0
    assert (rows[1]["overall_lo"], rows[1]["overall_hi"]) == (0.0, 0.0)
    assert len(rows) == 2


def test_the_held_out_rows_need_the_held_out_judgments(
    run_dir: RunDir, tmp_path: Path
) -> None:
    _export(run_dir, "tune", tmp_path / "out")
    _answer(run_dir, "tune", _by_key)
    judged = _judged()
    scores = judge_scores(run_dir, judged, stages=("tune",))

    with pytest.raises(ValueError, match="no judgment for"):
        held_out_rows(
            judged,
            scores,
            select_coefficients(judged, scores),
            n_boot=10,
            seed=0,
        )


def test_agreement_is_cohens_kappa_per_criterion_on_the_second_subset(
    run_dir: RunDir, tmp_path: Path
) -> None:
    _export(run_dir, "tune", tmp_path / "out", second_fraction=1.0)
    _answer(run_dir, "tune", _by_key, judge=1)

    def other(key: str) -> tuple[int, int]:
        concept, fluency = _by_key(key)
        return concept, 2 if key.endswith("/0") else fluency

    _answer(run_dir, "tune", other, judge=2)

    result = agreement(run_dir, _judged(), stages=("tune",))

    every = sorted(
        r.key for r in _judged().records if r.split == "tune" and r.text
    )
    first = [_by_key(key)[1] for key in every]
    second = [other(key)[1] for key in every]
    expected = judges.label_agreement(first, second)
    assert result == {
        "concept": {
            "n": 17,
            "rate": 1.0,
            "kappa": 1.0,
            "weighted_kappa": 1.0,
        },
        "fluency": {
            "n": 17,
            "rate": pytest.approx(expected.rate),
            "kappa": pytest.approx(expected.kappa),
            "weighted_kappa": pytest.approx(weighted_kappa(first, second)),
        },
    }
    assert expected.rate < 1.0


def test_agreement_is_empty_without_a_second_judge(
    run_dir: RunDir, tmp_path: Path
) -> None:
    _export(run_dir, "tune", tmp_path / "out", second_fraction=0.0)
    _answer(run_dir, "tune", _by_key)

    assert agreement(run_dir, _judged(), stages=("tune",)) == {}


def test_agreement_needs_the_judgments_of_the_second_judge(
    run_dir: RunDir, tmp_path: Path
) -> None:
    _export(run_dir, "tune", tmp_path / "out")
    _answer(run_dir, "tune", _by_key, judge=1)

    with pytest.raises(ValueError, match="not imported"):
        agreement(run_dir, _judged(), stages=("tune",))


def test_a_changed_plan_is_refused_once_judgments_are_stored(
    run_dir: RunDir, tmp_path: Path
) -> None:
    _export(run_dir, "tune", tmp_path / "a")
    _export(run_dir, "tune", tmp_path / "b", batch_size=3)
    assert len(_plan(run_dir, "tune").batches[0].keys) == 3
    _answer(run_dir, "tune", _by_key)
    before = _digest(run_dir)

    with pytest.raises(ValueError, match="already has judgments"):
        _export(run_dir, "tune", tmp_path / "c", batch_size=4)

    assert _digest(run_dir) == before
    assert not (tmp_path / "c").exists()
    _export(run_dir, "tune", tmp_path / "d", batch_size=3)
    assert _digest(run_dir) == before


@pytest.mark.parametrize("shared", ["batch", "item"])
def test_a_stage_never_shares_an_id_with_the_other_stage(
    run_dir: RunDir,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shared: str,
) -> None:
    _export(run_dir, "tune", tmp_path / "tune")
    _answer(run_dir, "tune", _by_key)
    taken = _plan(run_dir, "tune").batches[0]
    plan_export = run_module.plan_export

    def clashing(keys: Sequence[str], **options: Any) -> Export:
        plan = plan_export(keys, **options)
        first = plan.batches[0]
        if shared == "batch":
            first = dataclasses.replace(first, batch_id=taken.batch_id)
        else:
            first = dataclasses.replace(
                first, item_ids=(taken.item_ids[0], *first.item_ids[1:])
            )
        return dataclasses.replace(plan, batches=(first, *plan.batches[1:]))

    monkeypatch.setattr(run_module, "plan_export", clashing)
    before = _digest(run_dir)

    with pytest.raises(
        ValueError, match=f"a {shared} id of stage 'held_out' repeats"
    ):
        _export(run_dir, "held_out", tmp_path / "held_out")

    assert _digest(run_dir) == before
    assert not (tmp_path / "held_out").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"stage": "test"}, "stage must be one of"),
        ({"criteria": ("fluency", "concept")}, "criteria"),
    ],
    ids=["stage", "criteria"],
)
def test_an_export_rejects_what_it_cannot_plan(
    run_dir: RunDir, tmp_path: Path, change: dict[str, object], message: str
) -> None:
    judged = _judged()
    if "criteria" in change:
        judged = dataclasses.replace(judged, criteria=change["criteria"])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        export_stage(
            run_dir, judged, str(change.get("stage", "tune")), tmp_path / "out"
        )


def test_records_must_have_distinct_keys_and_a_known_split() -> None:
    record = _judged().records[0]

    with pytest.raises(ValueError, match="duplicate"):
        Judged(criteria=CRITERIA, records=(record, record))
    with pytest.raises(ValueError, match="split"):
        Judged(
            criteria=CRITERIA,
            records=(dataclasses.replace(record, split="test"),),
        )
