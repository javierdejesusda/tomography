"""True and false statements from the reference repository (facts topic).

The file has 612 rows, 14 of which repeat an earlier statement, and one
statement appears with both labels. So that no statement can sit in two
splits, repeats are collapsed to their first row and the statement with
conflicting labels is dropped, which leaves 597 examples.
"""

import csv
import io
from pathlib import Path

from tomography.datasets.records import Example
from tomography.datasets.records import assign_splits
from tomography.datasets.sources import PinnedSource
from tomography.datasets.sources import fetch

SOURCE = PinnedSource(
    name="facts",
    kind="url",
    location=(
        "https://raw.githubusercontent.com/andyzoujm/"
        "representation-engineering/"
        "5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e/"
    ),
    revision=None,
    files=("data/facts/facts_true_false.csv",),
    license="MIT (granted by the representation-engineering repository)",
    sha256={
        "data/facts/facts_true_false.csv": (
            "151b6163289460eb8b37f7c6a1998e23aed8e05b5e557bf1a2297079bf0e5af9"
        )
    },
)


def parse_facts(text: str) -> list[Example]:
    """Parses the facts CSV into examples.

    Args:
        text: File content with the header `statement,label`, where label 1
            means the statement is true.

    Returns:
        One example per distinct statement, in file order, with the uid
        "facts-<row>" of its first row, label 1 for true statements and the
        surrounding whitespace stripped. Statements that carry both labels
        are dropped.

    Raises:
        ValueError: If a column is missing or a label is not 0 or 1.
    """
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not {"statement", "label"} <= set(reader.fieldnames or ()):
        raise ValueError(
            f"facts: expected the columns statement and label, got "
            f"{reader.fieldnames}"
        )
    first: dict[str, tuple[str, int]] = {}
    conflicting: set[str] = set()
    for row_index, row in enumerate(reader):
        uid = f"facts-{row_index}"
        if row["label"] not in ("0", "1"):
            raise ValueError(f"{uid}: label {row['label']!r} is not 0 or 1")
        statement = row["statement"].strip()
        label = int(row["label"])
        if statement not in first:
            first[statement] = (uid, label)
        elif first[statement][1] != label:
            conflicting.add(statement)
    kept = [(s, uid, label) for s, (uid, label) in first.items()]
    kept = [item for item in kept if item[0] not in conflicting]
    splits = assign_splits([uid for _, uid, _ in kept], dataset="facts")
    return [
        Example(uid=uid, text=statement, label=label, split=split)
        for (statement, uid, label), split in zip(kept, splits, strict=True)
    ]


def load_facts(
    *, cache: Path | None = None, offline: bool = False
) -> list[Example]:
    """Loads the true and false statements.

    Args:
        cache: Cache directory; defaults to `sources.cache_dir()`.
        offline: If True, read only from the cache.

    Returns:
        The examples of `parse_facts` for the pinned file.
    """
    path = fetch(SOURCE, cache=cache, offline=offline)
    return parse_facts(path.read_bytes().decode("utf-8"))
