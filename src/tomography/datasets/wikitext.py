"""WikiText-103 test lines as a neutral corpus for perplexity.

Only the test shard of the raw (untokenized) variant is fetched, so the large
train shards are never downloaded. Each non-empty line of the file, a heading
or a paragraph, is one text.
"""

from collections.abc import Iterable
from pathlib import Path

import pyarrow.parquet as pq

from tomography.datasets.records import Example
from tomography.datasets.sources import PinnedSource
from tomography.datasets.sources import fetch

SOURCE = PinnedSource(
    name="wikitext",
    kind="hub",
    location="Salesforce/wikitext",
    revision="b08601e04326c79dfdd32d625aee71d232d685c3",
    files=("wikitext-103-raw-v1/test-00000-of-00001.parquet",),
    license="CC-BY-SA-3.0 and GFDL",
    sha256={},
)


def parse_wikitext(lines: Iterable[str]) -> list[Example]:
    """Turns the lines of the test file into unlabeled examples.

    Args:
        lines: The lines in file order.

    Returns:
        One example per non-blank line with the uid "wikitext-test-<line in
        the source file>", the whitespace around the line stripped, label -1
        and the `test` split.
    """
    return [
        Example(
            uid=f"wikitext-test-{line_index}",
            text=line.strip(),
            label=-1,
            split="test",
        )
        for line_index, line in enumerate(lines)
        if line.strip()
    ]


def load_wikitext(
    *, cache: Path | None = None, offline: bool = False
) -> list[Example]:
    """Loads the non-empty lines of the WikiText-103 test split.

    Args:
        cache: Cache directory; defaults to `sources.cache_dir()`.
        offline: If True, read only from the cache.

    Returns:
        The examples of `parse_wikitext` for the pinned file.
    """
    path = fetch(SOURCE, cache=cache, offline=offline)
    column = pq.read_table(path, columns=["text"]).column("text")
    return parse_wikitext(column.to_pylist())
