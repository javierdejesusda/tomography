"""SST-2 sentiment phrases (Socher et al., 2013) as labeled examples.

The train split holds phrase-level snippets of the source training sentences
and the validation split the full sentences of the source dev split. The
source test split has no labels (all -1) and is not loaded, so there is no
`test` split.
"""

from collections.abc import Iterable
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import Literal

import pyarrow.parquet as pq

from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.datasets.sources import PinnedSource
from tomography.datasets.sources import fetch

SourceSplit = Literal["train", "validation"]

SOURCE = PinnedSource(
    name="sst2",
    kind="hub",
    location="stanfordnlp/sst2",
    revision="8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb",
    files=(
        "data/train-00000-of-00001.parquet",
        "data/validation-00000-of-00001.parquet",
    ),
    license="Unknown (cite Socher et al., 2013)",
    sha256={},
)

_FILES: Mapping[SourceSplit, str] = {
    "train": SOURCE.files[0],
    "validation": SOURCE.files[1],
}
_SPLITS: Mapping[SourceSplit, Split] = {"train": "train", "validation": "val"}


def parse_sst2(
    rows: Iterable[Mapping[str, Any]], *, source_split: SourceSplit
) -> list[Example]:
    """Turns SST-2 rows into examples.

    Args:
        rows: Mappings with the source columns `idx`, `sentence` and `label`.
        source_split: The source split the rows come from; "validation"
            becomes the "val" split.

    Returns:
        One example per row, label 1 for positive sentiment, with the padding
        around the phrase stripped.

    Raises:
        ValueError: If a label is not 0 or 1, as in the unlabeled test split.
    """
    examples = []
    for row in rows:
        uid = f"sst2-{source_split}-{row['idx']}"
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"{uid}: label {label} is not 0 or 1")
        examples.append(
            Example(
                uid=uid,
                text=str(row["sentence"]).strip(),
                label=label,
                split=_SPLITS[source_split],
            )
        )
    return examples


def load_sst2(
    *, cache: Path | None = None, offline: bool = False
) -> list[Example]:
    """Loads the labeled SST-2 train and validation splits.

    Args:
        cache: Cache directory; defaults to `sources.cache_dir()`.
        offline: If True, read only from the cache.

    Returns:
        The train examples followed by the validation examples.
    """
    examples: list[Example] = []
    for source_split, file in _FILES.items():
        path = fetch(SOURCE, file=file, cache=cache, offline=offline)
        rows = pq.read_table(path).to_pylist()
        examples.extend(parse_sst2(rows, source_split=source_split))
    return examples
