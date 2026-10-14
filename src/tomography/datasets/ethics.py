"""ETHICS utilitarianism and commonsense morality (Hendrycks et al., 2021).

Both tasks are read from the canonical tar in memory: members are looked up
by name and nothing is extracted. The `test_hard` files and the other tasks
are not used.

The source train file is split into `train` (75%) and `val` (25%) by group of
linked scenarios: rows that share a scenario text, directly or through a chain
of rows, always land in the same split. The source test file is the `test`
split.
"""

from collections.abc import Mapping
import csv
import io
from pathlib import Path
import tarfile
from typing import Literal

from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.datasets.records import assign_splits
from tomography.datasets.sources import PinnedSource
from tomography.datasets.sources import fetch

SourceSplit = Literal["train", "test"]

SOURCE = PinnedSource(
    name="ethics",
    kind="url",
    location="https://people.eecs.berkeley.edu/~hendrycks/",
    revision=None,
    files=("ethics.tar",),
    license="MIT",
    sha256={
        "ethics.tar": (
            "40acbf1ac0da79a2aabef394d58889136b8d38b05be09482006de2453fb06333"
        )
    },
)

_UTILITY_MEMBERS: Mapping[SourceSplit, str] = {
    "train": "ethics/utilitarianism/util_train.csv",
    "test": "ethics/utilitarianism/util_test.csv",
}
_MORALITY_MEMBERS: Mapping[SourceSplit, str] = {
    "train": "ethics/commonsense/cm_train.csv",
    "test": "ethics/commonsense/cm_test.csv",
}
_MORALITY_COLUMNS = ("label", "input", "is_short", "edited")
_TRAIN_VAL_FRACTIONS = (0.75, 0.25, 0.0)


def parse_ethics_utility(
    text: str, *, source_split: SourceSplit
) -> list[Example]:
    """Parses a utilitarianism CSV, which has no header.

    Args:
        text: File content; column 0 is the more pleasant scenario and
            column 1 the less pleasant one.
        source_split: The source file the text comes from.

    Returns:
        Two examples per row: "<row uid>:a" with label 1 (the more pleasant)
        and "<row uid>:b" with label 0, where the row uid is
        "ethics-util-<source_split>-<row>". Both carry the row uid in
        `meta["pair"]` and share a split, and rows that share a scenario text,
        directly or through a chain of rows, always share a split. Texts are
        stripped.

    Raises:
        ValueError: If a row does not have exactly two columns.
    """
    rows: list[tuple[str, str, str]] = []
    for row_index, row in enumerate(csv.reader(io.StringIO(text, newline=""))):
        row_uid = f"ethics-util-{source_split}-{row_index}"
        if len(row) != 2:
            raise ValueError(f"{row_uid}: expected 2 columns, got {len(row)}")
        rows.append((row_uid, row[0].strip(), row[1].strip()))
    splits = _assign_group_splits(rows, source_split)
    examples = []
    for (row_uid, more_pleasant, less_pleasant), split in zip(
        rows, splits, strict=True
    ):
        examples.append(
            Example(
                uid=f"{row_uid}:a",
                text=more_pleasant,
                label=1,
                split=split,
                meta={"pair": row_uid},
            )
        )
        examples.append(
            Example(
                uid=f"{row_uid}:b",
                text=less_pleasant,
                label=0,
                split=split,
                meta={"pair": row_uid},
            )
        )
    return examples


def parse_ethics_morality(
    text: str, *, source_split: SourceSplit
) -> list[Example]:
    """Parses a commonsense morality CSV.

    Args:
        text: File content with the header `label,input,is_short,edited`,
            where the source label 1 means the scenario is morally wrong.
        source_split: The source file the text comes from.

    Returns:
        One example per row with the uid "ethics-cm-<source_split>-<row>".
        The label is flipped so that 1 means acceptable. `meta` holds
        `is_short` (a one-sentence scenario, else a long post) and `edited`
        as 0 or 1. Texts are stripped.

    Raises:
        ValueError: If a column is missing or a value is not 0 or 1 (label)
            or True or False (the flags).
    """
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not set(_MORALITY_COLUMNS) <= set(reader.fieldnames or ()):
        raise ValueError(
            f"ethics morality: expected the columns {list(_MORALITY_COLUMNS)}"
            f", got {reader.fieldnames}"
        )
    rows = []
    for row_index, row in enumerate(reader):
        uid = f"ethics-cm-{source_split}-{row_index}"
        if row["label"] not in ("0", "1"):
            raise ValueError(f"{uid}: label {row['label']!r} is not 0 or 1")
        for flag in ("is_short", "edited"):
            if row[flag] not in ("True", "False"):
                raise ValueError(
                    f"{uid}: {flag} {row[flag]!r} is not True or False"
                )
        rows.append((uid, row))
    splits = _assign_row_splits(
        [uid for uid, _ in rows], source_split, "ethics_morality"
    )
    return [
        Example(
            uid=uid,
            text=row["input"].strip(),
            label=1 - int(row["label"]),
            split=split,
            meta={
                "is_short": int(row["is_short"] == "True"),
                "edited": int(row["edited"] == "True"),
            },
        )
        for (uid, row), split in zip(rows, splits, strict=True)
    ]


def load_ethics_utility(
    *, cache: Path | None = None, offline: bool = False
) -> list[Example]:
    """Loads the utilitarianism pairs from the ETHICS tar.

    Args:
        cache: Cache directory; defaults to `sources.cache_dir()`.
        offline: If True, read only from the cache.

    Returns:
        The examples of `parse_ethics_utility` for the source train file
        followed by the source test file.
    """
    texts = _read_members(
        fetch(SOURCE, cache=cache, offline=offline), _UTILITY_MEMBERS
    )
    return [
        example
        for source_split, text in texts.items()
        for example in parse_ethics_utility(text, source_split=source_split)
    ]


def load_ethics_morality(
    *, cache: Path | None = None, offline: bool = False
) -> list[Example]:
    """Loads the commonsense morality scenarios from the ETHICS tar.

    Args:
        cache: Cache directory; defaults to `sources.cache_dir()`.
        offline: If True, read only from the cache.

    Returns:
        The examples of `parse_ethics_morality` for the source train file
        followed by the source test file.
    """
    texts = _read_members(
        fetch(SOURCE, cache=cache, offline=offline), _MORALITY_MEMBERS
    )
    return [
        example
        for source_split, text in texts.items()
        for example in parse_ethics_morality(text, source_split=source_split)
    ]


def _assign_row_splits(
    uids: list[str], source_split: SourceSplit, dataset: str
) -> list[Split]:
    if source_split == "test":
        return ["test"] * len(uids)
    return assign_splits(uids, dataset=dataset, fractions=_TRAIN_VAL_FRACTIONS)


def _assign_group_splits(
    rows: list[tuple[str, str, str]], source_split: SourceSplit
) -> list[Split]:
    if source_split == "test":
        return ["test"] * len(rows)
    parent: dict[str, str] = {}

    def find(text: str) -> str:
        root = parent.setdefault(text, text)
        while parent[root] != root:
            root = parent[root]
        while parent[text] != root:
            parent[text], text = root, parent[text]
        return root

    for _, more_pleasant, less_pleasant in rows:
        parent[find(more_pleasant)] = find(less_pleasant)
    first_uid: dict[str, str] = {}
    for row_uid, more_pleasant, _ in rows:
        first_uid.setdefault(find(more_pleasant), row_uid)
    group_uids = list(first_uid.values())
    group_splits = dict(
        zip(
            first_uid,
            assign_splits(
                group_uids,
                dataset="ethics_utility",
                fractions=_TRAIN_VAL_FRACTIONS,
            ),
            strict=True,
        )
    )
    return [group_splits[find(more_pleasant)] for _, more_pleasant, _ in rows]


def _read_members(
    archive_path: Path, members: Mapping[SourceSplit, str]
) -> dict[SourceSplit, str]:
    texts: dict[SourceSplit, str] = {}
    with tarfile.open(archive_path, "r:") as archive:
        for source_split, name in members.items():
            try:
                info = archive.getmember(name)
            except KeyError as error:
                raise ValueError(
                    f"{name} is not in {archive_path.name}"
                ) from error
            handle = archive.extractfile(info) if info.isfile() else None
            if handle is None:
                raise ValueError(
                    f"{name} is not a regular file in {archive_path.name}"
                )
            texts[source_split] = handle.read().decode("utf-8")
    return texts
