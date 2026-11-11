"""Judging the stored generations of a run: export, import and scores.

A run that stores generations describes them as `Record` rows. The tuning
half is exported first; once its judgments are imported, the held-out half
is exported at the coefficient each arm won on the tuning half, so a held-out
judgment can never reach the selection. Plans and judgments are stored in
the run directory: `judge_export_<stage>.json` and
`judgments_<batch_id>.json`.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.judges import label_agreement
from tomography.judging.batches import Export
from tomography.judging.batches import Generation
from tomography.judging.batches import plan_export
from tomography.judging.batches import write_export
from tomography.judging.judgments import parse_judgments
from tomography.judging.rubric import rubric_sha256
from tomography.judging.rubric import rubric_text
from tomography.judging.scores import harmonic_mean
from tomography.judging.scores import select_coefficient
from tomography.reading.metrics import bootstrap_ci

TUNE = "tune"
HELD_OUT = "held_out"
STAGES: tuple[str, ...] = (TUNE, HELD_OUT)

Scores = Mapping[str, NDArray[np.int64]]
Selection = Mapping[tuple[str, str], float | None]


@dataclass(frozen=True)
class Record:
    """One stored generation and where it sits in the design.

    Attributes:
        key: Identifier of the generation, unique in its run.
        concept: Name of the concept the arm steers to.
        description: Description of the concept, as a judge reads it.
        arm: Name of the arm.
        coeff: Coefficient of the arm, or None for an arm without a grid.
        split: `tune` for a tuning prompt, `held_out` for a reported one.
        prompt_index: Index of the prompt inside its split.
        prompt: The prompt as a judge reads it.
        text: The generated text.
    """

    key: str
    concept: str
    description: str
    arm: str
    coeff: float | None
    split: str
    prompt_index: int
    prompt: str
    text: str

    @property
    def empty(self) -> bool:
        """Whether the generation holds nothing but whitespace."""
        return not self.text.strip()


@dataclass(frozen=True)
class Judged:
    """The generations of a run and the criteria they are judged on.

    Attributes:
        criteria: The criteria of the role of the checkpoint.
        records: Every stored generation.
    """

    criteria: tuple[str, ...]
    records: tuple[Record, ...]

    def __post_init__(self) -> None:
        """Checks the keys and the splits.

        Raises:
            ValueError: If a key repeats or a split is not a stage.
        """
        if len({record.key for record in self.records}) != len(self.records):
            raise ValueError("duplicate generation keys")
        for record in self.records:
            if record.split not in STAGES:
                raise ValueError(
                    f"{record.key}: split must be one of {STAGES}, got "
                    f"{record.split!r}"
                )

    def cells(self) -> list[tuple[str, str]]:
        """Returns each `(concept, arm)` once, in order of first appearance."""
        return list(dict.fromkeys((r.concept, r.arm) for r in self.records))


def _stem(stage: str) -> str:
    return f"judge_export_{stage}"


def read_export(run_dir: RunDir, stage: str) -> Export | None:
    """Reads the stored plan of a stage.

    Args:
        run_dir: The run.
        stage: `tune` or `held_out`.

    Returns:
        The plan, or None if the stage was never exported.
    """
    if not (run_dir.path / f"{_stem(stage)}.json").is_file():
        return None
    stored = run_dir.read_json(_stem(stage))
    assert isinstance(stored, dict)
    return Export.from_dict(stored)


def _judgments_stem(batch_id: str) -> str:
    return f"judgments_{batch_id}"


def _is_imported(run_dir: RunDir, batch_id: str) -> bool:
    return (run_dir.path / f"{_judgments_stem(batch_id)}.json").is_file()


def _stage_keys(run_dir: RunDir, judged: Judged, stage: str) -> list[str]:
    """Lists the non-empty generations a stage sends to the judges."""
    if stage == TUNE:
        return [
            r.key for r in judged.records if r.split == TUNE and not r.empty
        ]
    selected = select_coefficients(
        judged, judge_scores(run_dir, judged, stages=(TUNE,))
    )
    return [
        r.key
        for r in judged.records
        if r.split == HELD_OUT
        and not r.empty
        and r.coeff == selected[r.concept, r.arm]
    ]


def export_stage(
    run_dir: RunDir,
    judged: Judged,
    stage: str,
    out_dir: Path,
    *,
    seed: int | None = None,
    batch_size: int = 50,
    second_fraction: float = 0.1,
) -> list[Path]:
    """Plans a stage, stores the plan in the run and writes the batch files.

    The tuning stage holds every non-empty generation of the tuning prompts.
    The held-out stage needs the tuning judgments of the first judge and
    holds the non-empty generations of the held-out prompts at the selected
    coefficient of each arm, and every one of an arm without a grid. An
    empty generation is never exported: it scores 0 on every criterion.

    Repeating an export with the same arguments rewrites the same bytes. A
    different plan replaces the stored one only while no judgment of the
    stage is stored.

    Args:
        run_dir: The run that holds the generations.
        judged: The generations of the run.
        stage: `tune` or `held_out`.
        out_dir: Folder for the batch files, the rubric and the manifest.
        seed: Seed of the plan; None derives it from the seed of the run and
            the stage.
        batch_size: Largest number of items per batch file.
        second_fraction: Share of the generations a second judge rates again.

    Returns:
        The written paths, as `write_export` returns them.

    Raises:
        ValueError: If the stage is unknown, the tuning judgments that the
            held-out stage needs are not imported, the plan differs from a
            stored plan that already has judgments, an identifier collides
            with one of the other stage, or `plan_export` rejects its
            arguments. Nothing is written then.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    keys = _stage_keys(run_dir, judged, stage)
    rubric = rubric_text()
    plan = plan_export(
        keys,
        stage=stage,
        seed=(
            derive_seed(int(run_dir.config["seed"]), "judge", stage)
            if seed is None
            else seed
        ),
        criteria=judged.criteria,
        rubric_sha256=rubric_sha256(rubric),
        batch_size=batch_size,
        second_fraction=second_fraction,
    )
    stored = read_export(run_dir, stage)
    if stored != plan:
        if stored is not None and any(
            _is_imported(run_dir, batch.batch_id) for batch in stored.batches
        ):
            raise ValueError(
                f"stage {stage!r} already has judgments for another plan; "
                "export it with the arguments of the first export"
            )
        _check_disjoint(run_dir, plan)
        run_dir.write_json(_stem(stage), plan.to_dict())
    by_key = {r.key: r for r in judged.records}
    return write_export(
        plan,
        {
            key: Generation(
                key=key,
                prompt=by_key[key].prompt,
                concept=by_key[key].description,
                text=by_key[key].text,
            )
            for key in keys
        },
        out_dir,
        rubric=rubric,
    )


def _ids(plan: Export) -> dict[str, set[str]]:
    """Returns the batch identifiers and the item identifiers of a plan."""
    return {
        "batch": {b.batch_id for b in plan.batches},
        "item": {i for b in plan.batches for i in b.item_ids},
    }


def _check_disjoint(run_dir: RunDir, plan: Export) -> None:
    """Refuses a plan that shares an identifier with another stage."""
    for stage in STAGES:
        other = None if stage == plan.stage else read_export(run_dir, stage)
        if other is None:
            continue
        for label, ids in _ids(plan).items():
            if ids & _ids(other)[label]:
                raise ValueError(
                    f"a {label} id of stage {plan.stage!r} repeats one of "
                    f"stage {stage!r}; export with another seed"
                )


def import_judgments(
    run_dir: RunDir, batch_id: str, text: str, *, judge_model: str
) -> Path:
    """Checks the judgments of one batch and stores them in the run.

    `judgments_<batch_id>.json` holds the batch id, its stage and judge
    slot, the model id of the judge, the digest of the rubric the batch was
    exported with, the criteria and the raw judgments in batch order.

    Args:
        run_dir: The run the batch was exported from.
        batch_id: Identifier of the exported batch.
        text: Content of the judgments file.
        judge_model: Model id of the judge, as the judge reports it.

    Returns:
        The path of the stored judgments.

    Raises:
        JudgmentError: If the file does not hold exactly one valid judgment
            per exported item.
        ValueError: If the run has no export, the batch is unknown or
            already imported, or `judge_model` is blank. Nothing is written
            when an error is raised.
    """
    plans = [plan for stage in STAGES if (plan := read_export(run_dir, stage))]
    if not plans:
        raise ValueError(f"{run_dir.path} has no export to import into")
    plan = next(
        (p for p in plans if any(b.batch_id == batch_id for b in p.batches)),
        None,
    )
    if plan is None:
        raise ValueError(f"unknown batch {batch_id!r} for {run_dir.path}")
    model = judge_model.strip()
    if not model:
        raise ValueError("judge_model must name the judge")
    if _is_imported(run_dir, batch_id):
        raise ValueError(f"batch {batch_id} is already imported")
    batch = plan.batch(batch_id)
    scores = parse_judgments(text, batch, plan.criteria)
    return run_dir.write_json(
        _judgments_stem(batch_id),
        {
            "batch_id": batch_id,
            "stage": plan.stage,
            "judge": batch.judge,
            "judge_model": model,
            "rubric_sha256": plan.rubric_sha256,
            "criteria": list(plan.criteria),
            "judgments": [
                {
                    "item_id": item_id,
                    **dict(zip(plan.criteria, row.tolist(), strict=True)),
                }
                for item_id, row in zip(batch.item_ids, scores, strict=True)
            ],
        },
    )


def _stored_scores(
    run_dir: RunDir, stages: Sequence[str], judge: int
) -> dict[str, NDArray[np.int64]]:
    """Reads the stored scores of one judge, keyed by generation."""
    scores: dict[str, NDArray[np.int64]] = {}
    for stage in stages:
        plan = read_export(run_dir, stage)
        if plan is None:
            raise ValueError(f"stage {stage!r} has no export")
        batches = [b for b in plan.batches if b.judge == judge]
        pending = [
            b.batch_id for b in batches if not _is_imported(run_dir, b.batch_id)
        ]
        if pending:
            raise ValueError(
                f"stage {stage!r} has batches of judge {judge} that are not "
                f"imported: {', '.join(pending)}"
            )
        for batch in batches:
            stored: Any = run_dir.read_json(_judgments_stem(batch.batch_id))
            for key, row in zip(batch.keys, stored["judgments"], strict=True):
                scores[key] = np.array(
                    [row[name] for name in plan.criteria], dtype=np.int64
                )
    return scores


def judge_scores(
    run_dir: RunDir, judged: Judged, *, stages: Sequence[str] = STAGES
) -> dict[str, NDArray[np.int64]]:
    """Reads the scores the first judge gave the generations of some stages.

    Args:
        run_dir: The run.
        judged: The generations of the run.
        stages: The stages to read.

    Returns:
        Maps the key of every judged generation to its scores in criteria
        order, and every empty generation of the stages to zeros.

    Raises:
        ValueError: If a stage has no export or a batch of the first judge
            is not imported.
    """
    scores = _stored_scores(run_dir, stages, 1)
    zeros = np.zeros(len(judged.criteria), dtype=np.int64)
    for record in judged.records:
        if record.split in stages and record.empty:
            scores[record.key] = zeros
    return scores


def _overall(records: Sequence[Record], scores: Scores) -> NDArray[np.float64]:
    missing = [r.key for r in records if r.key not in scores]
    if missing:
        raise ValueError(
            f"no judgment for {len(missing)} generations, first {missing[0]!r}"
        )
    return harmonic_mean(np.stack([scores[r.key] for r in records]))


def tuning_rows(judged: Judged, scores: Scores) -> list[dict[str, object]]:
    """Averages the overall score of each arm and coefficient on tuning.

    Args:
        judged: The generations of the run.
        scores: The scores of the tuning generations.

    Returns:
        One row per concept, arm and coefficient, in order of first
        appearance, with `concept`, `arm`, `coeff`, `n_items` and `overall`,
        the mean harmonic mean over the tuning generations.

    Raises:
        ValueError: If a tuning generation has no judgment.
    """
    groups: dict[tuple[str, str, float | None], list[Record]] = {}
    for record in judged.records:
        if record.split == TUNE:
            groups.setdefault(
                (record.concept, record.arm, record.coeff), []
            ).append(record)
    return [
        {
            "concept": concept,
            "arm": arm,
            "coeff": coeff,
            "n_items": len(records),
            "overall": float(_overall(records, scores).mean()),
        }
        for (concept, arm, coeff), records in groups.items()
    ]


def select_coefficients(
    judged: Judged, scores: Scores
) -> dict[tuple[str, str], float | None]:
    """Selects the coefficient of each arm on the tuning prompts only.

    Args:
        judged: The generations of the run.
        scores: The scores of the tuning generations.

    Returns:
        Maps `(concept, arm)` to the coefficient with the highest mean
        overall score on the tuning prompts (see `select_coefficient` for
        ties), or to None for an arm without a grid.

    Raises:
        ValueError: If a tuning generation has no judgment.
    """
    grids: dict[tuple[str, str], list[tuple[float, float]]] = {
        cell: [] for cell in judged.cells()
    }
    for row in tuning_rows(judged, scores):
        if row["coeff"] is not None:
            grids[str(row["concept"]), str(row["arm"])].append(
                (float(row["coeff"]), float(row["overall"]))  # type: ignore[arg-type]
            )
    return {
        cell: (
            select_coefficient([c for c, _ in grid], [m for _, m in grid])
            if grid
            else None
        )
        for cell, grid in grids.items()
    }


def held_out_rows(
    judged: Judged,
    scores: Scores,
    selected: Selection,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> list[dict[str, object]]:
    """Scores each arm on the held-out prompts at its selected coefficient.

    The samples of a prompt are averaged first, and the interval of the
    overall score is a percentile bootstrap over the held-out prompts
    (`bootstrap_ci`), seeded per concept and arm.

    Args:
        judged: The generations of the run.
        scores: The scores of the held-out generations.
        selected: The coefficient of each `(concept, arm)`.
        n_boot: Number of bootstrap resamples.
        seed: Base seed of the resampling.
        alpha: One minus the coverage of the interval.

    Returns:
        One row per concept and arm, in order of first appearance, with
        `concept`, `arm`, `coeff`, `n_prompts`, `n_items`, one
        `<criterion>_score` mean per criterion, and `overall` with
        `overall_lo` and `overall_hi`.

    Raises:
        ValueError: If a held-out generation at a selected coefficient has
            no judgment, or an arm has no held-out generation.
    """
    rows: list[dict[str, object]] = []
    for concept, arm in judged.cells():
        coeff = selected[concept, arm]
        records = [
            r
            for r in judged.records
            if (r.concept, r.arm, r.split, r.coeff)
            == (concept, arm, HELD_OUT, coeff)
        ]
        if not records:
            raise ValueError(
                f"{concept}/{arm} has no held-out generation at {coeff}"
            )
        overall = _overall(records, scores)
        values = np.column_stack(
            [np.stack([scores[r.key] for r in records]), overall]
        )
        prompts = sorted({r.prompt_index for r in records})
        index = np.array([prompts.index(r.prompt_index) for r in records])
        per_prompt = np.stack(
            [values[index == p].mean(axis=0) for p in range(len(prompts))]
        )
        lo, hi = bootstrap_ci(
            per_prompt[:, -1],
            alpha=alpha,
            n_boot=n_boot,
            seed=derive_seed(seed, concept, arm),
        )
        means = per_prompt.mean(axis=0)
        rows.append(
            {
                "concept": concept,
                "arm": arm,
                "coeff": coeff,
                "n_prompts": len(prompts),
                "n_items": len(records),
                **{
                    f"{name}_score": float(means[column])
                    for column, name in enumerate(judged.criteria)
                },
                "overall": float(means[-1]),
                "overall_lo": float(lo),
                "overall_hi": float(hi),
            }
        )
    return rows


def agreement(
    run_dir: RunDir, judged: Judged, *, stages: Sequence[str] = STAGES
) -> dict[str, dict[str, float]]:
    """Compares the two judges on the generations both of them rated.

    Args:
        run_dir: The run.
        judged: The generations of the run.
        stages: The stages to pool.

    Returns:
        Per criterion, the number of generations `n`, the agreement `rate`
        and Cohen's `kappa` of the two score sequences. Empty when no stage
        exported anything for a second judge.

    Raises:
        ValueError: If a stage has no export or a batch of either judge is
            not imported.
    """
    first = _stored_scores(run_dir, stages, 1)
    second = _stored_scores(run_dir, stages, 2)
    if not second:
        return {}
    keys = sorted(second)
    result: dict[str, dict[str, float]] = {}
    for column, name in enumerate(judged.criteria):
        found = label_agreement(
            [int(first[key][column]) for key in keys],
            [int(second[key][column]) for key in keys],
        )
        result[name] = {
            "n": len(keys),
            "rate": found.rate,
            "kappa": found.kappa,
        }
    return result
