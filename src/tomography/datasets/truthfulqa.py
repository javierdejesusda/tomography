"""TruthfulQA multiple-choice questions (Lin et al., 2022), MC1 targets.

In the published MC1 data the correct choice is always at index 0, so the
choices are shuffled with a seed: a scorer that favors early positions would
otherwise look perfect.
"""

from collections.abc import Iterable
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from tomography.datasets.records import ChoiceExample
from tomography.datasets.records import assign_splits
from tomography.datasets.sources import PinnedSource
from tomography.datasets.sources import fetch

SOURCE = PinnedSource(
    name="truthfulqa",
    kind="hub",
    location="truthfulqa/truthful_qa",
    revision="741b8276f2d1982aa3d5b832d3ee81ed3b896490",
    files=("multiple_choice/validation-00000-of-00001.parquet",),
    license="Apache-2.0",
    sha256={},
)


def parse_truthfulqa(
    rows: Iterable[Mapping[str, Any]], *, seed: int = 0
) -> list[ChoiceExample]:
    """Turns TruthfulQA rows into multiple-choice items with shuffled choices.

    Args:
        rows: Mappings with `question` and `mc1_targets`, a mapping of
            `choices` and `labels` where exactly one label is 1.
        seed: Seed of the choice shuffle.

    Returns:
        One item per row with the uid "truthfulqa-<row>" and the choices in
        a seeded random order, `correct` pointing at the right one. The
        splits come from `assign_splits`, because the source has a single
        split.

    Raises:
        ValueError: If a row does not have exactly one correct choice, or the
            number of labels differs from the number of choices.
    """
    rng = np.random.default_rng(seed)
    items = []
    for row_index, row in enumerate(rows):
        uid = f"truthfulqa-{row_index}"
        choices = [str(c).strip() for c in row["mc1_targets"]["choices"]]
        labels = list(row["mc1_targets"]["labels"])
        if len(labels) != len(choices) or sum(labels) != 1:
            raise ValueError(
                f"{uid}: expected one correct choice among {len(choices)} "
                f"choices, got labels {labels}"
            )
        order = rng.permutation(len(choices))
        items.append(
            (
                uid,
                str(row["question"]).strip(),
                tuple(choices[i] for i in order),
                int(np.flatnonzero(order == labels.index(1))[0]),
            )
        )
    splits = assign_splits([item[0] for item in items], dataset="truthfulqa")
    return [
        ChoiceExample(
            uid=uid,
            question=question,
            choices=choices,
            correct=correct,
            split=split,
        )
        for (uid, question, choices, correct), split in zip(
            items, splits, strict=True
        )
    ]


def load_truthfulqa(
    *, seed: int = 0, cache: Path | None = None, offline: bool = False
) -> list[ChoiceExample]:
    """Loads the 817 TruthfulQA validation questions.

    Args:
        seed: Seed of the choice shuffle.
        cache: Cache directory; defaults to `sources.cache_dir()`.
        offline: If True, read only from the cache.

    Returns:
        The items of `parse_truthfulqa` for the pinned file.
    """
    path = fetch(SOURCE, cache=cache, offline=offline)
    return parse_truthfulqa(pq.read_table(path).to_pylist(), seed=seed)
