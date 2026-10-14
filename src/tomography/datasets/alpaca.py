"""Alpaca instructions without an input (Taori et al., 2023).

The instruction-tuning arm of the control experiments uses the rows whose
`input` field is empty, as the reference code does. The data is licensed
CC BY-NC 4.0 and was generated with an OpenAI model, so it is downloaded at
runtime only and never vendored.
"""

from collections.abc import Iterable
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tomography.datasets.records import Example
from tomography.datasets.records import assign_splits
from tomography.datasets.sources import PinnedSource
from tomography.datasets.sources import fetch

SOURCE = PinnedSource(
    name="alpaca",
    kind="hub",
    location="tatsu-lab/alpaca",
    revision="dce01c9b08f87459cf36a430d809084718273017",
    files=("data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",),
    license="CC BY-NC 4.0 plus OpenAI terms (runtime download only)",
    sha256={},
)


def parse_alpaca(rows: Iterable[Mapping[str, Any]]) -> list[Example]:
    """Keeps the instructions that have no input and a non-empty response.

    Args:
        rows: Mappings with the source columns `instruction`, `input` and
            `output`.

    Returns:
        One example per kept row with the uid "alpaca-<row in the source
        file>". `text` is the response, `meta["instruction"]` the prompt
        that asks for it, and label is -1 because the rows are unlabeled.
        Texts are stripped.
    """
    kept = []
    for row_index, row in enumerate(rows):
        response = str(row["output"]).strip()
        if str(row["input"]).strip() or not response:
            continue
        kept.append(
            (f"alpaca-{row_index}", response, str(row["instruction"]).strip())
        )
    splits = assign_splits([uid for uid, _, _ in kept], dataset="alpaca")
    return [
        Example(
            uid=uid,
            text=response,
            label=-1,
            split=split,
            meta={"instruction": instruction},
        )
        for (uid, response, instruction), split in zip(
            kept, splits, strict=True
        )
    ]


def load_alpaca(
    *, cache: Path | None = None, offline: bool = False
) -> list[Example]:
    """Loads the Alpaca instructions that have no input.

    Args:
        cache: Cache directory; defaults to `sources.cache_dir()`.
        offline: If True, read only from the cache.

    Returns:
        The examples of `parse_alpaca` for the pinned file.
    """
    path = fetch(SOURCE, cache=cache, offline=offline)
    return parse_alpaca(pq.read_table(path).to_pylist())
