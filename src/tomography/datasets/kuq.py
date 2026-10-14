"""KUQ: questions that cannot be answered, in six categories.

The knowledge-boundary study uses the unknown rows as natural transfer sets
for its reading. They are unanswerable by nature, not merely beyond one model.
"""

from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from tomography.datasets.records import QAExample
from tomography.datasets.records import assign_splits
from tomography.datasets.sources import PinnedSource
from tomography.datasets.sources import fetch

SOURCE = PinnedSource(
    name="kuq",
    kind="hub",
    location="amayuelas/KUQ",
    revision="f99b53aa226dbb0d1b086db3ec352b0da0aa8f41",
    files=("knowns_unknowns.jsonl",),
    license="MIT",
    sha256={},
)


def load_kuq(
    *,
    cache: Path | None = None,
    offline: bool = False,
    categories: Collection[str] | None = None,
    drop_duplicates: bool = True,
) -> list[QAExample]:
    """Loads the unknown questions of KUQ from the pinned Hub revision.

    Args:
        cache: Cache directory; defaults to the shared tomography cache.
        offline: If True, read only what is already cached.
        categories: Category names to keep, such as "future unknown"; None
            keeps all six.
        drop_duplicates: Whether to keep only the first copy of a question.

    Returns:
        The items `parse_kuq` builds from the pinned `knowns_unknowns.jsonl`.

    Raises:
        OfflineCacheMissError: If `offline` is True and the file is not
            cached.
    """
    path = fetch(SOURCE, cache=cache, offline=offline)
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return parse_kuq(
        rows, categories=categories, drop_duplicates=drop_duplicates
    )


def parse_kuq(
    rows: Iterable[Mapping[str, Any]],
    *,
    categories: Collection[str] | None = None,
    drop_duplicates: bool = True,
) -> list[QAExample]:
    """Turns KUQ rows into unanswerable question items.

    Only rows flagged unknown are kept: their "answer" field holds crowd
    explanations, not gold answers, so the items carry no answers. The
    dataset lists 955 unknown questions twice, always in the same category;
    the first copy wins. The uid is the row position in the file, so it does
    not depend on the filters.

    Args:
        rows: Mappings with "question", "unknown" and, for unknown rows,
            "category".
        categories: Category names to keep; None keeps all.
        drop_duplicates: Whether to keep only the first copy of a question.

    Returns:
        One item per kept row, with kind "kuq:<category>" (spaces become
        underscores).
    """
    kept: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not row["unknown"]:
            continue
        if categories is not None and row["category"] not in categories:
            continue
        if drop_duplicates:
            if row["question"] in seen:
                continue
            seen.add(row["question"])
        kept.append((f"kuq:{index}", row))
    splits = assign_splits([uid for uid, _ in kept], dataset="kuq")
    return [
        QAExample(
            uid=uid,
            question=row["question"],
            answers=(),
            kind="kuq:" + row["category"].replace(" ", "_"),
            split=split,
            meta={"category": row["category"], "source": "kuq"},
        )
        for (uid, row), split in zip(kept, splits, strict=True)
    ]
