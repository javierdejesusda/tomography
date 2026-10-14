"""Emotion scenarios and replies from the reference repository.

Each of the six emotion files holds short first-person scenarios; the replies
file holds 582 truncated model replies (one is empty) used as the response
text of the emotion function template.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import json
from pathlib import Path

from tomography.datasets.records import Example
from tomography.datasets.records import assign_splits
from tomography.datasets.sources import PinnedSource
from tomography.datasets.sources import fetch

EMOTIONS = ("happiness", "sadness", "anger", "fear", "disgust", "surprise")

_REPLIES_FILE = "data/emotions/all_truncated_outputs.json"

SOURCE = PinnedSource(
    name="emotions",
    kind="url",
    location=(
        "https://raw.githubusercontent.com/andyzoujm/"
        "representation-engineering/"
        "5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e/"
    ),
    revision=None,
    files=(*(f"data/emotions/{name}.json" for name in EMOTIONS), _REPLIES_FILE),
    license="MIT (granted by the representation-engineering repository)",
    sha256={
        "data/emotions/happiness.json": (
            "3dedefdc819007074a1a53c7a606d1e5661ac0cc19984212aa809c3a9f6bb646"
        ),
        "data/emotions/sadness.json": (
            "999065615aac1bfb95c5cec054965f9abf17f36d59c7a779916eb28eb3f4ea24"
        ),
        "data/emotions/anger.json": (
            "4577af8b0c62b18e3f06fb96883e95ed7b776f0809f605bfe792e438fe1832df"
        ),
        "data/emotions/fear.json": (
            "cf791978f94f62ec617480a646f3a16d38aaf594a385c698bec08bf32747ee2b"
        ),
        "data/emotions/disgust.json": (
            "3532b5d3c3701ad23c799dd436a86db81b716ecd41b35e890e2fb2ae4a80dfff"
        ),
        "data/emotions/surprise.json": (
            "43d90c0bbf755c789373286e35bc522461f9fe960fb5e3c5586a1749e19c7812"
        ),
        _REPLIES_FILE: (
            "c87990913b839fa792b751b532a8f9abc372770bf4d34a1de358a3a233edb6e5"
        ),
    },
)


def parse_emotions(scenarios: Mapping[str, Sequence[str]]) -> list[Example]:
    """Turns the per-emotion scenario lists into examples.

    Args:
        scenarios: Scenario texts keyed by emotion name; must have exactly
            the keys of `EMOTIONS`.

    Returns:
        Examples grouped by emotion in `EMOTIONS` order. The label is the
        index of the emotion in `EMOTIONS`, `meta["emotion"]` its name, and
        the uid "emotion-<name>-<position in the source list>". Texts are
        stripped, and empty or repeated scenarios within an emotion are
        dropped.

    Raises:
        ValueError: If an emotion is missing or an unknown one is present.
    """
    unknown = sorted(set(scenarios) - set(EMOTIONS))
    missing = [name for name in EMOTIONS if name not in scenarios]
    if unknown or missing:
        raise ValueError(
            f"emotions: missing {missing}, unknown {unknown}; "
            f"expected {list(EMOTIONS)}"
        )
    rows = [
        (f"emotion-{name}-{position}", text, label, name)
        for label, name in enumerate(EMOTIONS)
        for position, text in _distinct_texts(scenarios[name])
    ]
    splits = assign_splits([row[0] for row in rows], dataset="emotions")
    return [
        Example(
            uid=uid,
            text=text,
            label=label,
            split=split,
            meta={"emotion": name},
        )
        for (uid, text, label, name), split in zip(rows, splits, strict=True)
    ]


def parse_emotion_replies(replies: Sequence[str]) -> list[Example]:
    """Turns the truncated replies into unlabeled examples.

    Args:
        replies: The reply texts in file order.

    Returns:
        One example per non-empty reply with the uid "emotion-reply-<position
        in the source list>" and label -1, because a reply carries no emotion
        of its own.
    """
    rows = _distinct_texts(replies)
    uids = [f"emotion-reply-{position}" for position, _ in rows]
    splits = assign_splits(uids, dataset="emotion_replies")
    return [
        Example(uid=uid, text=text, label=-1, split=split)
        for uid, (_, text), split in zip(uids, rows, splits, strict=True)
    ]


def load_emotions(
    *, cache: Path | None = None, offline: bool = False
) -> list[Example]:
    """Loads the emotion scenarios of all six emotions.

    Args:
        cache: Cache directory; defaults to `sources.cache_dir()`.
        offline: If True, read only from the cache.

    Returns:
        The examples of `parse_emotions` for the pinned files.
    """
    scenarios = {
        name: _read_texts(
            fetch(
                SOURCE,
                file=f"data/emotions/{name}.json",
                cache=cache,
                offline=offline,
            )
        )
        for name in EMOTIONS
    }
    return parse_emotions(scenarios)


def load_emotion_replies(
    *, cache: Path | None = None, offline: bool = False
) -> list[Example]:
    """Loads the truncated replies for the emotion function template.

    Args:
        cache: Cache directory; defaults to `sources.cache_dir()`.
        offline: If True, read only from the cache.

    Returns:
        The examples of `parse_emotion_replies` for the pinned file.
    """
    path = fetch(SOURCE, file=_REPLIES_FILE, cache=cache, offline=offline)
    return parse_emotion_replies(_read_texts(path))


def _distinct_texts(texts: Sequence[str]) -> list[tuple[int, str]]:
    seen: set[str] = set()
    kept: list[tuple[int, str]] = []
    for position, text in enumerate(texts):
        text = text.strip()
        if text and text not in seen:
            seen.add(text)
            kept.append((position, text))
    return kept


def _read_texts(path: Path) -> list[str]:
    data = json.loads(path.read_bytes().decode("utf-8"))
    if not isinstance(data, list) or not all(isinstance(t, str) for t in data):
        raise ValueError(f"{path.name}: expected a JSON list of strings")
    return data
