"""Geometry of Truth statement sets (Marks and Tegmark, 2024).

Nine of the published CSV files are used as out-of-distribution truth sets.
The repository states no license, so the files are downloaded at runtime and
never vendored. Rows are kept as published, including the few repeated
statements of the Spanish translation sets.
"""

from collections.abc import Sequence
import csv
import io
from pathlib import Path

from tomography.datasets.records import Example
from tomography.datasets.records import assign_splits
from tomography.datasets.sources import PinnedSource
from tomography.datasets.sources import fetch

SUBSETS = (
    "cities",
    "neg_cities",
    "sp_en_trans",
    "neg_sp_en_trans",
    "larger_than",
    "smaller_than",
    "cities_cities_conj",
    "cities_cities_disj",
    "common_claim_true_false",
)

SOURCE = PinnedSource(
    name="geometry_of_truth",
    kind="url",
    location=(
        "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/"
        "5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb/"
    ),
    revision=None,
    files=tuple(f"datasets/{name}.csv" for name in SUBSETS),
    license="None stated (runtime download only)",
    sha256={
        "datasets/cities.csv": (
            "f2560a0c1758c69935b50125111b574b71896bb75930dd87b1d8c973c1fa086e"
        ),
        "datasets/neg_cities.csv": (
            "66c5c84de98ca15570138c205ab0e3907b57750083fac72a4eacafc3d9ebcea7"
        ),
        "datasets/sp_en_trans.csv": (
            "9afa97c1b8af539a05f47655a80a5c7832c6fdb914885fd24cb4bcccbe2795b5"
        ),
        "datasets/neg_sp_en_trans.csv": (
            "bec33e37047135b15cdd18c0cb0c66b65fb435ace08910cbcc220ce1ef610c96"
        ),
        "datasets/larger_than.csv": (
            "406d6872e97910f3af5a031600d4932b5a567b8666e7545c8b446057471ad723"
        ),
        "datasets/smaller_than.csv": (
            "ee6b7cf668d7e0fb3f49a84ca8d510cb7e484ae029ef86ed238362e26ef61a44"
        ),
        "datasets/cities_cities_conj.csv": (
            "80e9987e8abcdd56ac8256d22b8ad6c2d6211e06a6c925c32cfcb0bc93d9d56d"
        ),
        "datasets/cities_cities_disj.csv": (
            "5f4883382bb3f8664fa9b0a51cfadec507e130bf21c94f3d04776ca923646565"
        ),
        "datasets/common_claim_true_false.csv": (
            "1889b4793c2ebd13c767eea8b65ebd4a5c2da700567677d512097b495a262576"
        ),
    },
)


def parse_geometry_of_truth(text: str, *, subset: str) -> list[Example]:
    """Parses one Geometry of Truth CSV into examples.

    Args:
        text: File content with at least the columns `statement` and
            `label`, where label 1 means the statement is true. Other
            columns are ignored.
        subset: The subset name, used in the uids and in `meta["subset"]`.

    Returns:
        One example per row with the uid "got-<subset>-<row>" and the
        surrounding whitespace of the statement stripped.

    Raises:
        ValueError: If a column is missing or a label is not 0 or 1.
    """
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not {"statement", "label"} <= set(reader.fieldnames or ()):
        raise ValueError(
            f"{subset}: expected the columns statement and label, got "
            f"{reader.fieldnames}"
        )
    rows = []
    for row_index, row in enumerate(reader):
        uid = f"got-{subset}-{row_index}"
        if row["label"] not in ("0", "1"):
            raise ValueError(f"{uid}: label {row['label']!r} is not 0 or 1")
        rows.append((uid, row["statement"].strip(), int(row["label"])))
    splits = assign_splits([uid for uid, _, _ in rows], dataset=SOURCE.name)
    return [
        Example(
            uid=uid,
            text=statement,
            label=label,
            split=split,
            meta={"subset": subset},
        )
        for (uid, statement, label), split in zip(rows, splits, strict=True)
    ]


def load_geometry_of_truth(
    *,
    subsets: Sequence[str] | None = None,
    cache: Path | None = None,
    offline: bool = False,
) -> list[Example]:
    """Loads Geometry of Truth statement sets.

    Args:
        subsets: Names from `SUBSETS` to load, in the order wanted; all nine
            in `SUBSETS` order when None.
        cache: Cache directory; defaults to `sources.cache_dir()`.
        offline: If True, read only from the cache.

    Returns:
        The examples of `parse_geometry_of_truth` for each subset in turn.

    Raises:
        ValueError: If a name is not in `SUBSETS`.
    """
    names = SUBSETS if subsets is None else tuple(subsets)
    unknown = [name for name in names if name not in SUBSETS]
    if unknown:
        raise ValueError(f"unknown subsets {unknown}; choose from {SUBSETS}")
    examples: list[Example] = []
    for name in names:
        path = fetch(
            SOURCE, file=f"datasets/{name}.csv", cache=cache, offline=offline
        )
        text = path.read_bytes().decode("utf-8")
        examples.extend(parse_geometry_of_truth(text, subset=name))
    return examples
