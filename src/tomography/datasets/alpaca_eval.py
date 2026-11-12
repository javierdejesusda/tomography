"""AlpacaEval instructions, the prompts of the steering benchmark protocol.

The file holds 805 instructions drawn from five instruction sets. Only the
instructions are read; the reference outputs of the file are ignored. The
data is licensed CC BY-NC 4.0, so it is downloaded at runtime only and never
vendored.
"""

from collections.abc import Iterable
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from tomography.datasets.records import Example
from tomography.datasets.sources import PinnedSource
from tomography.datasets.sources import fetch

SOURCE = PinnedSource(
    name="alpaca_eval",
    kind="hub",
    location="tatsu-lab/alpaca_eval",
    revision="2edc6fad8be6b14ea7230aabfd08188da6b8b814",
    files=("alpaca_eval.json",),
    license="CC-BY-NC-4.0 (runtime download only)",
    sha256={
        "alpaca_eval.json": (
            "d92b92c51e8f1962a21193abe74e6f727c2bc8286035f4041505ff38a7c3ae51"
        )
    },
)


def parse_alpaca_eval(rows: Iterable[Mapping[str, Any]]) -> list[Example]:
    """Keeps the non-blank instructions, in file order.

    Args:
        rows: Mappings with the source fields `instruction` and `dataset`.

    Returns:
        One example per kept row with the uid "alpaca_eval-<row in the
        source file>". `text` is the stripped instruction, the label is -1
        because the rows are unlabeled, `meta["dataset"]` names the
        instruction set the row comes from and every split is "test": the
        file is an evaluation set, and an experiment splits it itself.
    """
    examples = []
    for row_index, row in enumerate(rows):
        instruction = str(row["instruction"]).strip()
        if not instruction:
            continue
        examples.append(
            Example(
                uid=f"alpaca_eval-{row_index}",
                text=instruction,
                label=-1,
                split="test",
                meta={"dataset": str(row["dataset"])},
            )
        )
    return examples


def load_alpaca_eval(
    *, cache: Path | None = None, offline: bool = False
) -> list[Example]:
    """Loads the AlpacaEval instructions.

    Args:
        cache: Cache directory; defaults to `sources.cache_dir()`.
        offline: If True, read only from the cache.

    Returns:
        The examples of `parse_alpaca_eval` for the pinned file.
    """
    path = fetch(SOURCE, cache=cache, offline=offline)
    return parse_alpaca_eval(json.loads(path.read_text("utf-8")))
