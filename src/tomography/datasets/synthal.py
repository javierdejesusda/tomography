"""SyntHal: prompts about synthetic subjects invented for the benchmark.

The knowledge-boundary study uses them as the fictional items. The four
relations PopQA shares are asked with one template per relation, the same one
PopQA's real subjects get.
"""

from collections.abc import Iterable
from collections.abc import Mapping
import json
from pathlib import Path

from tomography.datasets import popqa
from tomography.datasets.records import QAExample
from tomography.datasets.records import assign_splits
from tomography.datasets.sources import PinnedSource
from tomography.datasets.sources import fetch

_BASE = "https://raw.githubusercontent.com/yuetianluNLP/SyntHal/"
_COMMIT = "c4fe4cecf41bc70ad65edcba0d0dab7252cfabac"
_FILE = "data/00_dataset_SyntHal/SyntHal_prompts.jsonl"
_DIGEST = "7391085698049413bfd1e2fc9351d350c444a4179fbf04a66d0e5ae171eb0def"

SOURCE = PinnedSource(
    name="synthal",
    kind="url",
    location=f"{_BASE}{_COMMIT}/",
    revision=None,
    files=(_FILE,),
    license="research and review terms, no permissive license",
    sha256={_FILE: _DIGEST},
)

RELATIONS: Mapping[str, str] = {
    "person_father": "father",
    "person_mother": "mother",
    "person_occupation": "occupation",
    "person_plays_pro_sport": "sport",
}


def load_synthal(
    *,
    cache: Path | None = None,
    offline: bool = False,
    relations: Mapping[str, str] = RELATIONS,
    templates: Mapping[str, str] = popqa.TEMPLATES,
) -> list[QAExample]:
    """Loads SyntHal from the pinned GitHub commit.

    Args:
        cache: Cache directory; defaults to the shared tomography cache.
        offline: If True, read only what is already cached.
        relations: SyntHal "relation_key" to the relation name to keep.
        templates: Question template per relation name.

    Returns:
        The items `parse_synthal` builds from the pinned prompt file.

    Raises:
        ChecksumError: If the downloaded file does not match its digest.
        OfflineCacheMissError: If `offline` is True and the file is not
            cached.
        ValueError: If the template used for a row has no '{}' slot.
    """
    path = fetch(SOURCE, cache=cache, offline=offline)
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return parse_synthal(rows, relations=relations, templates=templates)


def parse_synthal(
    rows: Iterable[Mapping[str, str]],
    *,
    relations: Mapping[str, str] = RELATIONS,
    templates: Mapping[str, str] = popqa.TEMPLATES,
) -> list[QAExample]:
    """Turns SyntHal prompt rows into fictional question items.

    The question is rebuilt from the relation's shared template so real and
    fictional subjects are asked the same way. A relation with no entry in
    `templates` keeps the template of its own row. The split is a function
    of the subject, so every question about one subject lands in one split.

    Args:
        rows: Mappings with the columns "relation_key", "subject",
            "template" (one "{}" slot) and "example_id".
        relations: SyntHal "relation_key" to the relation name to keep.
        templates: Question template per relation name.

    Returns:
        One "fictional" item per row of a kept relation, in row order.

    Raises:
        ValueError: If the template used for a row has no '{}' slot.
    """
    kept = [row for row in rows if row["relation_key"] in relations]
    uids = [f"synthal:{row['example_id']}" for row in kept]
    splits = assign_splits([row["subject"] for row in kept], dataset="synthal")
    items = []
    for row, uid, split in zip(kept, uids, splits, strict=True):
        relation = relations[row["relation_key"]]
        template = templates.get(relation, row["template"])
        if "{}" not in template:
            raise ValueError(f"{uid}: template {template!r} has no slot")
        items.append(
            QAExample(
                uid=uid,
                question=template.replace("{}", row["subject"], 1),
                answers=(),
                kind="fictional",
                split=split,
                meta={
                    "relation": relation,
                    "subject": row["subject"],
                    "popularity": None,
                    "source": "synthal",
                },
            )
        )
    return items
