"""RACE reading comprehension, high-school level (Lai et al., 2017).

The Hub files hold one row per question. The loader groups them into one
passage per article, as the GPT-3 paper and the lm-evaluation-harness RACE
task do, keeping the article's last question (the item both the reading and
the harness score) and its earlier questions (solved examples in the harness
prompt). The three source splits are kept: train, validation (as `val`) and
test.
"""

from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
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
    name="race",
    kind="hub",
    location="ehovy/race",
    revision="2fec9fd81f1dc971569a9b729c43f2f0e6436637",
    files=(
        "high/train-00000-of-00001.parquet",
        "high/validation-00000-of-00001.parquet",
        "high/test-00000-of-00001.parquet",
    ),
    license="non-commercial research purpose only",
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
_LETTERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class RacePassage:
    """One RACE article with the questions the protocols read.

    Attributes:
        article: The article text as in the source.
        last: The article's last question as a 4-way item.
        earlier: The question and gold answer text of each earlier question
            of the article, in source order.
    """

    article: str
    last: ChoiceExample
    earlier: tuple[tuple[str, str], ...]


def parse_race(
    rows: Iterable[Mapping[str, Any]], *, source_split: SourceSplit
) -> list[RacePassage]:
    """Groups RACE question rows into one passage per article.

    Articles keep their order of first appearance and each article keeps the
    source order of its questions, as the harness's `_collate_data`, which
    the reference code copies, does.

    Every option is kept even when options repeat a text, so validation
    article 444 (high3189.txt), whose last question has four "_ & _" options,
    stays one 4-way group. The reference drops the options equal to the gold
    text, which leaves that article one stimulus with four labels and
    misaligns every later validation group.

    Args:
        rows: Mappings with `example_id`, `article`, `question`, `options`
            (four texts) and `answer`, the letter of the gold option.
        source_split: The source split the rows come from; "validation"
            becomes the "val" split.

    Returns:
        One passage per article. Its last question has the uid
        "race-<example_id>", the stripped question and options in source
        order and `correct` the index of the answer letter.

    Raises:
        ValueError: If a question does not have four options or its answer
            is not one of the letters A to D.
    """
    articles: dict[str, list[ChoiceExample]] = {}
    for row in rows:
        item = _question(row, _SPLITS[source_split])
        articles.setdefault(str(row["article"]), []).append(item)
    passages = []
    for article, questions in articles.items():
        *earlier, last = questions
        passages.append(
            RacePassage(
                article=article,
                last=last,
                earlier=tuple(
                    (q.question, q.choices[q.correct]) for q in earlier
                ),
            )
        )
    return passages


def load_race(
    *, cache: Path | None = None, offline: bool = False
) -> list[RacePassage]:
    """Loads RACE-high, all three source splits.

    Args:
        cache: Cache directory; defaults to `sources.cache_dir()`.
        offline: If True, read only from the cache.

    Returns:
        The passages of `parse_race` for the train, validation and test files
        in that order.
    """
    passages: list[RacePassage] = []
    for source_split, file in _FILES.items():
        path = fetch(SOURCE, file=file, cache=cache, offline=offline)
        rows = pq.read_table(path).to_pylist()
        passages.extend(parse_race(rows, source_split=source_split))
    return passages


def _question(row: Mapping[str, Any], split: Split) -> ChoiceExample:
    uid = f"race-{row['example_id']}"
    options = [str(o).strip() for o in row["options"]]
    if len(options) != len(_LETTERS) or row["answer"] not in _LETTERS:
        raise ValueError(
            f"{uid}: answer {row['answer']!r} does not name one of "
            f"{len(options)} options; RACE questions have four, A to D"
        )
    return ChoiceExample(
        uid=uid,
        question=str(row["question"]).strip(),
        choices=tuple(options),
        correct=_LETTERS.index(row["answer"]),
        split=split,
    )
