"""CommonsenseQA multiple-choice questions (Talmor et al., 2019).

The train and validation splits (as `val`) are kept; the test split has no
answer keys.
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

SourceSplit = Literal["train", "validation"]

SOURCE = PinnedSource(
    name="commonsenseqa",
    kind="hub",
    location="tau/commonsense_qa",
    revision="94630fe30dad47192a8546eb75f094926d47e155",
    files=(
        "data/train-00000-of-00001.parquet",
        "data/validation-00000-of-00001.parquet",
    ),
    license="MIT",
    sha256={},
)

_FILES: Mapping[SourceSplit, str] = dict(
    zip(("train", "validation"), SOURCE.files, strict=True)
)
_SPLITS: Mapping[SourceSplit, Split] = {"train": "train", "validation": "val"}


def parse_commonsenseqa(
    rows: Iterable[Mapping[str, Any]], *, source_split: SourceSplit
) -> list[ChoiceExample]:
    """Turns CommonsenseQA rows into multiple-choice items.

    Args:
        rows: Mappings with `id`, `question`, `choices` (a mapping of `label`
            and `text` lists) and `answerKey`, which names one label.
        source_split: The source split the rows come from; "validation"
            becomes the "val" split.

    Returns:
        One item per row with the uid "commonsenseqa-<id>", the stripped
        question, the stripped choices in source order and `correct` the
        index of the answer key among the labels.

    Raises:
        ValueError: If the choice texts and labels differ in number, or the
            answer key is not one of the labels (the unlabeled test split).
    """
    items = []
    for row in rows:
        uid = f"commonsenseqa-{row['id']}"
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
                question=str(row["question"]).strip(),
                choices=tuple(texts),
                correct=labels.index(row["answerKey"]),
                split=_SPLITS[source_split],
            )
        )
    return items


def load_commonsenseqa(
    *, cache: Path | None = None, offline: bool = False
) -> list[ChoiceExample]:
    """Loads the labeled CommonsenseQA splits.

    Args:
        cache: Cache directory; defaults to `sources.cache_dir()`.
        offline: If True, read only from the cache.

    Returns:
        The items of `parse_commonsenseqa` for the train and validation
        files in that order.
    """
    items: list[ChoiceExample] = []
    for source_split, file in _FILES.items():
        path = fetch(SOURCE, file=file, cache=cache, offline=offline)
        rows = pq.read_table(path).to_pylist()
        items.extend(parse_commonsenseqa(rows, source_split=source_split))
    return items
