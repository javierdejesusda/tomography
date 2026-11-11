"""Blinded, shuffled batches of generations for an external judge.

A plan assigns every generation an opaque item id and a batch. The plan stays
with the run and is the only link from an item id back to the arm, the
coefficient and the split of a generation; the exported files hold nothing
but the item id, the prompt, the concept description and the generation.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any
from typing import Self

import numpy as np

from tomography.judging.rubric import CRITERIA
from tomography.judging.rubric import SCORES

RUBRIC_FILE = "rubric.txt"

_STAGE = re.compile(r"[a-z][a-z0-9_]*")
_ITEM_BYTES = 8
_BATCH_BYTES = 4


@dataclass(frozen=True)
class Generation:
    """One stored generation, as a judge sees it.

    Attributes:
        key: Identifier of the generation inside its run. It may name the
            arm, the coefficient and the split, so it is never exported.
        prompt: The prompt, without any chat markup of the checkpoint.
        concept: Description of the concept the generation was steered to.
        text: The generated text.
    """

    key: str
    prompt: str
    concept: str
    text: str


@dataclass(frozen=True)
class Batch:
    """The items one judge rates in one file.

    Attributes:
        batch_id: Opaque identifier, the stem of the batch file.
        judge: 1 for the judge that rates every generation, 2 for the
            independent judge of the agreement subset.
        item_ids: Opaque identifier of each item, in file order.
        keys: The generation behind each item.
    """

    batch_id: str
    judge: int
    item_ids: tuple[str, ...]
    keys: tuple[str, ...]


@dataclass(frozen=True)
class Export:
    """The plan of one export: which item of which batch is which generation.

    Attributes:
        stage: Name of the export, such as `tune`.
        seed: Seed of the shuffle and of the identifiers.
        rubric_sha256: Digest of the rubric the judges are given.
        criteria: The criteria the judges rate, in `CRITERIA` order.
        batches: The batches of the first judge, then of the second.
    """

    stage: str
    seed: int
    rubric_sha256: str
    criteria: tuple[str, ...]
    batches: tuple[Batch, ...]

    def batch(self, batch_id: str) -> Batch:
        """Finds a batch by its identifier.

        Args:
            batch_id: The identifier.

        Returns:
            The batch.

        Raises:
            ValueError: If the export has no such batch.
        """
        for batch in self.batches:
            if batch.batch_id == batch_id:
                return batch
        raise ValueError(f"unknown batch {batch_id!r} in stage {self.stage!r}")

    def to_dict(self) -> dict[str, Any]:
        """Returns the plan as JSON values."""
        return {
            "stage": self.stage,
            "seed": self.seed,
            "rubric_sha256": self.rubric_sha256,
            "criteria": list(self.criteria),
            "batches": [
                {
                    "batch_id": batch.batch_id,
                    "judge": batch.judge,
                    "items": [
                        [item_id, key]
                        for item_id, key in zip(
                            batch.item_ids, batch.keys, strict=True
                        )
                    ],
                }
                for batch in self.batches
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Rebuilds a plan from the values of `to_dict`.

        Args:
            data: The stored plan.

        Returns:
            The plan.
        """
        return cls(
            stage=str(data["stage"]),
            seed=int(data["seed"]),
            rubric_sha256=str(data["rubric_sha256"]),
            criteria=tuple(data["criteria"]),
            batches=tuple(
                Batch(
                    batch_id=str(batch["batch_id"]),
                    judge=int(batch["judge"]),
                    item_ids=tuple(item for item, _ in batch["items"]),
                    keys=tuple(key for _, key in batch["items"]),
                )
                for batch in data["batches"]
            ),
        )


def _draw_ids(
    rng: np.random.Generator, count: int, n_bytes: int, taken: set[str]
) -> list[str]:
    """Draws distinct hex identifiers, none of them in `taken`."""
    drawn: list[str] = []
    while len(drawn) < count:
        candidate = rng.bytes(n_bytes).hex()
        if candidate not in taken:
            taken.add(candidate)
            drawn.append(candidate)
    return drawn


def _check_criteria(criteria: Sequence[str]) -> tuple[str, ...]:
    wanted = tuple(criteria)
    if not wanted or wanted != tuple(c for c in CRITERIA if c in wanted):
        raise ValueError(
            f"criteria must be a non-empty selection of {CRITERIA} in that "
            f"order, got {wanted}"
        )
    return wanted


def plan_export(
    keys: Sequence[str],
    *,
    stage: str,
    seed: int,
    criteria: Sequence[str],
    rubric_sha256: str,
    batch_size: int = 50,
    second_fraction: float = 0.1,
) -> Export:
    """Shuffles generations into blinded batches for one or two judges.

    The keys are sorted first, so the plan depends on the set of keys and on
    the seed only. One generator seeded with `seed` shuffles the keys of the
    first judge, draws the subset of the second judge and its order, and draws
    every identifier.

    Args:
        keys: The key of every generation to export.
        stage: Name of the export: lowercase letters, digits and underscores.
        seed: Seed of the plan.
        criteria: The criteria to rate, a selection of `CRITERIA` in order.
        rubric_sha256: Digest of the rubric the judges are given.
        batch_size: Largest number of items per batch file.
        second_fraction: Share of the generations a second judge rates again
            under other item ids, rounded up; 0 exports none.

    Returns:
        The plan.

    Raises:
        ValueError: If there are no generations, a key repeats, the stage name
            is not usable, the criteria are not a selection of `CRITERIA` in
            order, `batch_size` is below 1 or `second_fraction` is outside
            [0, 1].
    """
    ordered = sorted(keys)
    if not ordered:
        raise ValueError("no generations to export")
    if len(set(ordered)) != len(ordered):
        raise ValueError("duplicate generation keys")
    if not _STAGE.fullmatch(stage):
        raise ValueError(f"stage {stage!r} must match {_STAGE.pattern}")
    wanted = _check_criteria(criteria)
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if not 0.0 <= second_fraction <= 1.0:
        raise ValueError(
            f"second_fraction must be in [0, 1], got {second_fraction}"
        )
    rng = np.random.default_rng(seed)
    n_second = math.ceil(second_fraction * len(ordered))
    passes = (
        (1, rng.permutation(len(ordered))),
        (2, rng.permutation(len(ordered))[:n_second]),
    )
    item_ids: set[str] = set()
    batch_ids: set[str] = set()
    batches: list[Batch] = []
    for judge, order in passes:
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            (batch_id,) = _draw_ids(rng, 1, _BATCH_BYTES, batch_ids)
            batches.append(
                Batch(
                    batch_id=batch_id,
                    judge=judge,
                    item_ids=tuple(
                        _draw_ids(rng, len(chunk), _ITEM_BYTES, item_ids)
                    ),
                    keys=tuple(ordered[i] for i in chunk),
                )
            )
    return Export(
        stage=stage,
        seed=seed,
        rubric_sha256=rubric_sha256,
        criteria=wanted,
        batches=tuple(batches),
    )


def _json_line(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")


def write_export(
    export: Export,
    generations: Mapping[str, Generation],
    out_dir: Path,
    *,
    rubric: str,
) -> list[Path]:
    """Writes the batch files, the rubric and the manifest of an export.

    Every file is UTF-8 with LF line ends on every platform, so two
    exports of one plan are byte-identical. `<batch_id>.jsonl` holds one JSON
    object per item with the fields `item_id`, `prompt`, `concept` and
    `generation`. `manifest_<stage>.json` lists the batch files with the judge
    each one goes to, and names the criteria, the scores and the rubric.

    Args:
        export: The plan.
        generations: The generation of every planned key.
        out_dir: Folder to write into; created if missing.
        rubric: The rubric text the judges are given.

    Returns:
        The written paths: the batch files in plan order, the rubric, then
        the manifest.

    Raises:
        ValueError: If a planned key has no generation. Nothing is written
            then.
    """
    missing = [
        key
        for batch in export.batches
        for key in batch.keys
        if key not in generations
    ]
    if missing:
        raise ValueError(
            f"no generation for {len(missing)} planned keys, first "
            f"{missing[0]!r}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for batch in export.batches:
        path = out_dir / f"{batch.batch_id}.jsonl"
        path.write_bytes(
            b"".join(
                _json_line(
                    {
                        "item_id": item_id,
                        "prompt": generations[key].prompt,
                        "concept": generations[key].concept,
                        "generation": generations[key].text,
                    }
                )
                for item_id, key in zip(batch.item_ids, batch.keys, strict=True)
            )
        )
        paths.append(path)
    rubric_path = out_dir / RUBRIC_FILE
    rubric_path.write_bytes(rubric.replace("\r\n", "\n").encode("utf-8"))
    manifest = {
        "stage": export.stage,
        "criteria": list(export.criteria),
        "scores": list(SCORES),
        "rubric_file": RUBRIC_FILE,
        "rubric_sha256": export.rubric_sha256,
        "batches": [
            {
                "batch_id": batch.batch_id,
                "file": f"{batch.batch_id}.jsonl",
                "judge": batch.judge,
                "n_items": len(batch.item_ids),
            }
            for batch in export.batches
        ],
    }
    manifest_path = out_dir / f"manifest_{export.stage}.json"
    manifest_path.write_bytes(
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    )
    return [*paths, rubric_path, manifest_path]
