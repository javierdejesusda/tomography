"""NQ-open: Natural Questions as open-domain questions (Lee et al., 2019).

P9 uses the validation split as a reference set for error prediction.
"""

from collections.abc import Iterable
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tomography.datasets.records import QAExample
from tomography.datasets.records import assign_splits
from tomography.datasets.sources import PinnedSource
from tomography.datasets.sources import fetch

SOURCE = PinnedSource(
    name="nq_open",
    kind="hub",
    location="google-research-datasets/nq_open",
    revision="5dd9790a83002ad084ddeb7c420dc716852c6f28",
    files=("nq_open/validation-00000-of-00001.parquet",),
    license="CC-BY-SA-3.0",
    sha256={},
)


def load_nq_open(
    *, cache: Path | None = None, offline: bool = False
) -> list[QAExample]:
    """Loads the NQ-open validation questions from the pinned Hub revision.

    Args:
        cache: Cache directory; defaults to the shared tomography cache.
        offline: If True, read only what is already cached.

    Returns:
        The items `parse_nq_open` builds from the pinned parquet file.

    Raises:
        OfflineCacheMissError: If `offline` is True and the file is not
            cached.
    """
    path = fetch(SOURCE, cache=cache, offline=offline)
    return parse_nq_open(pq.read_table(path).to_pylist())


def parse_nq_open(rows: Iterable[Mapping[str, Any]]) -> list[QAExample]:
    """Turns NQ-open rows into open question items.

    The questions are lowercase and have no question mark, as in the source.
    The dataset has no ids, so the uid is the row position in the file.

    Args:
        rows: Mappings with "question" and "answer" (a list of strings).

    Returns:
        One "open" item per row, in row order.
    """
    kept = list(rows)
    uids = [f"nq_open:{index}" for index in range(len(kept))]
    splits = assign_splits(uids, dataset="nq_open")
    return [
        QAExample(
            uid=uid,
            question=row["question"],
            answers=tuple(dict.fromkeys(row["answer"])),
            kind="open",
            split=split,
            meta={"source": "nq_open"},
        )
        for uid, row, split in zip(uids, kept, splits, strict=True)
    ]
