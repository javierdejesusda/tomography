"""PopQA: entity questions with a popularity score (Mallen et al., 2023).

P9 reads the four relations that SyntHal shares (father, mother, occupation,
sport), so real and fictional subjects can be asked with one template.
"""

from collections import Counter
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
import csv
import json
from pathlib import Path

from tomography.datasets.records import QAExample
from tomography.datasets.records import assign_splits
from tomography.datasets.sources import PinnedSource
from tomography.datasets.sources import fetch

SOURCE = PinnedSource(
    name="popqa",
    kind="hub",
    location="akariasai/PopQA",
    revision="098765c79ea10a2cb19c828324e33281b8336ec0",
    files=("test.tsv",),
    license="no license on the dataset card (code repository: MIT)",
    sha256={},
)

TEMPLATES: Mapping[str, str] = {
    "occupation": "What is {}'s occupation?",
    "place of birth": "In what city was {} born?",
    "genre": "What genre is {}?",
    "father": "Who is the father of {}?",
    "country": "In what country is {}?",
    "producer": "Who was the producer of {}?",
    "director": "Who was the director of {}?",
    "capital of": "What is {} the capital of?",
    "screenwriter": "Who was the screenwriter for {}?",
    "composer": "Who was the composer of {}?",
    "color": "What color is {}?",
    "religion": "What is the religion of {}?",
    "sport": "What sport does {} play?",
    "author": "Who is the author of {}?",
    "mother": "Who is the mother of {}?",
    "capital": "What is the capital of {}?",
}

SHARED_RELATIONS: tuple[str, ...] = ("father", "mother", "occupation", "sport")


def load_popqa(
    *,
    cache: Path | None = None,
    offline: bool = False,
    relations: Collection[str] | None = SHARED_RELATIONS,
    min_alias_chars: int = 4,
    min_subject_chars: int = 3,
    drop_ambiguous: bool = True,
) -> list[QAExample]:
    """Loads PopQA from the pinned Hub revision.

    Args:
        cache: Cache directory; defaults to the shared tomography cache.
        offline: If True, read only what is already cached.
        relations: The "prop" values to keep, None for all sixteen.
        min_alias_chars: Shortest alias kept.
        min_subject_chars: Shortest subject kept.
        drop_ambiguous: Whether to drop questions that occur more than once.

    Returns:
        The items `parse_popqa` builds from the pinned `test.tsv`.

    Raises:
        OfflineCacheMissError: If `offline` is True and the file is not
            cached.
    """
    path = fetch(SOURCE, cache=cache, offline=offline)
    with path.open(encoding="utf-8", newline="") as handle:
        return parse_popqa(
            csv.DictReader(handle, delimiter="	"),
            relations=relations,
            min_alias_chars=min_alias_chars,
            min_subject_chars=min_subject_chars,
            drop_ambiguous=drop_ambiguous,
        )


def parse_popqa(
    rows: Iterable[Mapping[str, str]],
    *,
    relations: Collection[str] | None = SHARED_RELATIONS,
    min_alias_chars: int = 4,
    min_subject_chars: int = 3,
    drop_ambiguous: bool = True,
) -> list[QAExample]:
    """Turns PopQA rows into question items.

    Aliases shorter than `min_alias_chars` match too much text and are
    dropped. A row left without an alias, or with a subject shorter than
    `min_subject_chars`, is dropped. A question string that occurs in more
    than one row names several entities with different golds, so every copy
    is dropped unless `drop_ambiguous` is False.

    Args:
        rows: Mappings with the PopQA columns "id", "subj", "prop", "s_pop",
            "question" and "possible_answers" (a JSON list of strings).
        relations: The "prop" values to keep, None for all sixteen.
        min_alias_chars: Shortest alias kept.
        min_subject_chars: Shortest subject kept.
        drop_ambiguous: Whether to drop questions that occur more than once.

    Returns:
        One "real" item per kept row, in row order.

    Raises:
        ValueError: If a kept row's "possible_answers" is not a JSON list of
            strings.
    """
    candidates = [
        row for row in rows if relations is None or row["prop"] in relations
    ]
    seen = Counter(row["question"] for row in candidates)
    kept: list[tuple[str, Mapping[str, str], tuple[str, ...]]] = []
    for row in candidates:
        if len(row["subj"].strip()) < min_subject_chars:
            continue
        if drop_ambiguous and seen[row["question"]] > 1:
            continue
        uid = f"popqa:{row['id']}"
        answers = _clean_aliases(uid, row["possible_answers"], min_alias_chars)
        if answers:
            kept.append((uid, row, answers))
    splits = assign_splits([uid for uid, _, _ in kept], dataset="popqa")
    return [
        QAExample(
            uid=uid,
            question=row["question"],
            answers=answers,
            kind="real",
            split=split,
            meta={
                "relation": row["prop"],
                "subject": row["subj"],
                "popularity": int(row["s_pop"]),
                "source": "popqa",
            },
        )
        for (uid, row, answers), split in zip(kept, splits, strict=True)
    ]


def _clean_aliases(uid: str, encoded: str, min_chars: int) -> tuple[str, ...]:
    try:
        aliases = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError(f"{uid}: possible_answers is not JSON") from error
    if not isinstance(aliases, list) or not all(
        isinstance(alias, str) for alias in aliases
    ):
        raise ValueError(f"{uid}: possible_answers is not a list of strings")
    stripped = (alias.strip() for alias in aliases)
    return tuple(dict.fromkeys(a for a in stripped if len(a) >= min_chars))
