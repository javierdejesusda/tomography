"""FalseQA: natural questions built on a false premise (Hu et al., 2023).

P9 uses the false-premise half of the test split as one more type of question
that has no answer.
"""

from collections.abc import Iterable
from collections.abc import Mapping
import csv
from pathlib import Path

from tomography.datasets.records import QAExample
from tomography.datasets.records import assign_splits
from tomography.datasets.sources import PinnedSource
from tomography.datasets.sources import fetch

_BASE = "https://raw.githubusercontent.com/thunlp/FalseQA/"
_COMMIT = "9a9729974d8c5cfbfc6929e5b830111c0ed7c0c7"
_FILE = "dataset/test.csv"
_DIGEST = "8983aa5054982b88be37cbc7ce3fd9c9faeb56fd1481da01f51a806dac110624"

SOURCE = PinnedSource(
    name="falseqa",
    kind="url",
    location=f"{_BASE}{_COMMIT}/",
    revision=None,
    files=(_FILE,),
    license="no license: runtime download only",
    sha256={_FILE: _DIGEST},
)


def load_falseqa(
    *, cache: Path | None = None, offline: bool = False
) -> list[QAExample]:
    """Loads the FalseQA test split from the pinned GitHub commit.

    Args:
        cache: Cache directory; defaults to the shared tomography cache.
        offline: If True, read only what is already cached.

    Returns:
        The items `parse_falseqa` builds from the pinned `test.csv`.

    Raises:
        ChecksumError: If the downloaded file does not match its digest.
        OfflineCacheMissError: If `offline` is True and the file is not
            cached.
    """
    path = fetch(SOURCE, cache=cache, offline=offline)
    with path.open(encoding="utf-8", newline="") as handle:
        return parse_falseqa(csv.DictReader(handle))


def parse_falseqa(rows: Iterable[Mapping[str, str]]) -> list[QAExample]:
    """Turns FalseQA rows into false-premise question items.

    Label 1 marks a false premise; label 0 rows are valid questions with a
    written answer and are skipped. A false-premise row has no gold answer,
    so the items carry none. The uid is the row position in the file.

    Args:
        rows: Mappings with the columns "question" and "label".

    Returns:
        One "false_premise" item per label-1 row, in row order.
    """
    kept = [
        (f"falseqa:{index}", row["question"])
        for index, row in enumerate(rows)
        if row["label"] == "1"
    ]
    splits = assign_splits([uid for uid, _ in kept], dataset="falseqa")
    return [
        QAExample(
            uid=uid,
            question=question,
            answers=(),
            kind="false_premise",
            split=split,
            meta={"source": "falseqa"},
        )
        for (uid, question), split in zip(kept, splits, strict=True)
    ]
