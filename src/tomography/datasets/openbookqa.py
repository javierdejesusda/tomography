"""OpenBookQA multiple-choice science questions (Mihaylov et al., 2018).

The main configuration's three source splits are kept: train, validation (as
`val`) and test.
"""

from collections.abc import Iterable
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import Literal

import pyarrow.parquet as pq

from tomography.datasets.records import ChoiceExample
from tomography.datasets.records import Split
from tomography.datasets.sources import PinnedSource
from tomography.datasets.sources import fetch

SourceSplit = Literal["train", "validation", "test"]

SOURCE = PinnedSource(
    name="openbookqa",
    kind="hub",
    location="allenai/openbookqa",
    revision="388097ea7776314e93a529163e0fea805b8a6454",
    files=(
        "main/train-00000-of-00001.parquet",
        "main/validation-00000-of-00001.parquet",
        "main/test-00000-of-00001.parquet",
    ),
    license="unknown on the dataset card, Apache-2.0 on the GitHub repository",
    sha256={},
)

_FILES: Mapping[SourceSplit, str] = dict(
    zip(("train", "validation", "test"), SOURCE.files, strict=True)
)
_SPLITS: Mapping[SourceSplit, Split] = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


def parse_openbookqa(
    rows: Iterable[Mapping[str, Any]], *, source_split: SourceSplit
) -> list[ChoiceExample]:
    """Turns OpenBookQA rows into multiple-choice items.

    Args:
        rows: Mappings with `id`, `question_stem`, `choices` (a mapping of
            `text` and `label` lists) and `answerKey`, which names one label.
        source_split: The source split the rows come from; "validation"
            becomes the "val" split.

    Returns:
        One item per row with the uid "openbookqa-<id>", the stripped stem
        as the question, the stripped choices in source order and `correct`
        the index of the answer key among the labels.

    Raises:
        ValueError: If the choice texts and labels differ in number, or the
            answer key is not one of the labels.
    """
    items = []
    for row in rows:
        uid = f"openbookqa-{row['id']}"
        texts = [str(t).strip() for t in row["choices"]["text"]]
        labels = list(row["choices"]["label"])
        if len(texts) != len(labels) or row["answerKey"] not in labels:
            raise ValueError(
                f"{uid}: answer key {row['answerKey']!r} does not name one of "
                f"the {len(texts)} choices with labels {labels}"
            )
        items.append(
            ChoiceExample(
                uid=uid,
                question=str(row["question_stem"]).strip(),
                choices=tuple(texts),
                correct=labels.index(row["answerKey"]),
                split=_SPLITS[source_split],
            )
        )
    return items


def load_openbookqa(
    *, cache: Path | None = None, offline: bool = False
) -> list[ChoiceExample]:
    """Loads OpenBookQA, all three source splits.

    Args:
        cache: Cache directory; defaults to `sources.cache_dir()`.
        offline: If True, read only from the cache.

    Returns:
        The items of `parse_openbookqa` for the train, validation and test
        files in that order.
    """
    items: list[ChoiceExample] = []
    for source_split, file in _FILES.items():
        path = fetch(SOURCE, file=file, cache=cache, offline=offline)
        rows = pq.read_table(path).to_pylist()
        items.extend(parse_openbookqa(rows, source_split=source_split))
    return items
