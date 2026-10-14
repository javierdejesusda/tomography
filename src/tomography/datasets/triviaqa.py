"""TriviaQA: trivia questions with answer aliases (Joshi et al., 2017).

The knowledge-boundary study uses the validation split of the
reading-comprehension config, without the evidence documents, as a reference
set for error prediction.
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

_FILE = "rc.nocontext/validation-00000-of-00001.parquet"

SOURCE = PinnedSource(
    name="triviaqa",
    kind="hub",
    location="mandarjoshi/trivia_qa",
    revision="0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
    files=(_FILE,),
    license="unknown on the dataset card, Apache-2.0 on the GitHub repository",
    sha256={},
)


def load_triviaqa(
    *, cache: Path | None = None, offline: bool = False
) -> list[QAExample]:
    """Loads the TriviaQA validation questions from the pinned Hub revision.

    Only the question and answer columns are read, not the evidence.

    Args:
        cache: Cache directory; defaults to the shared tomography cache.
        offline: If True, read only what is already cached.

    Returns:
        The items `parse_triviaqa` builds from the pinned parquet file.

    Raises:
        OfflineCacheMissError: If `offline` is True and the file is not
            cached.
    """
    path = fetch(SOURCE, cache=cache, offline=offline)
    table = pq.read_table(path, columns=["question", "question_id", "answer"])
    return parse_triviaqa(table.to_pylist())


def parse_triviaqa(rows: Iterable[Mapping[str, Any]]) -> list[QAExample]:
    """Turns TriviaQA rows into open question items.

    The reading-comprehension config has one row per question and evidence
    document, so a question id repeats; the first row wins.

    Args:
        rows: Mappings with "question", "question_id" and "answer" (a
            mapping with "value" and "aliases").

    Returns:
        One "open" item per question id, in first-seen order, with the value
        first and the aliases after it.
    """
    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        unique.setdefault(row["question_id"], row)
    uids = [f"triviaqa:{question_id}" for question_id in unique]
    splits = assign_splits(uids, dataset="triviaqa")
    return [
        QAExample(
            uid=uid,
            question=row["question"],
            answers=tuple(
                dict.fromkeys(
                    [row["answer"]["value"], *row["answer"]["aliases"]]
                )
            ),
            kind="open",
            split=split,
            meta={"source": "triviaqa"},
        )
        for uid, row, split in zip(uids, unique.values(), splits, strict=True)
    ]
