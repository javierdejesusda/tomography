import collections
import json
from pathlib import Path

import pytest

from tomography.datasets import emotions
from tomography.datasets.emotions import EMOTIONS
from tomography.datasets.emotions import SOURCE
from tomography.datasets.emotions import load_emotion_replies
from tomography.datasets.emotions import load_emotions
from tomography.datasets.emotions import parse_emotion_replies
from tomography.datasets.emotions import parse_emotions
from tomography.datasets.records import assign_splits

REPLIES_FILE = "data/emotions/all_truncated_outputs.json"


def _scenarios(**overrides: list[str]) -> dict[str, list[str]]:
    base = {
        name: [f"{name} scenario one", f"{name} scenario two"]
        for name in EMOTIONS
    }
    base.update(overrides)
    return base


def test_emotions_are_in_the_documented_class_order() -> None:
    assert EMOTIONS == (
        "happiness",
        "sadness",
        "anger",
        "fear",
        "disgust",
        "surprise",
    )


def test_source_pins_the_reference_repository_commit() -> None:
    assert SOURCE.name == "emotions"
    assert SOURCE.kind == "url"
    assert SOURCE.location == (
        "https://raw.githubusercontent.com/andyzoujm/"
        "representation-engineering/"
        "5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e/"
    )
    assert SOURCE.files == (
        *(f"data/emotions/{name}.json" for name in EMOTIONS),
        REPLIES_FILE,
    )
    assert SOURCE.sha256[REPLIES_FILE] == (
        "c87990913b839fa792b751b532a8f9abc372770bf4d34a1de358a3a233edb6e5"
    )
    assert SOURCE.sha256["data/emotions/happiness.json"] == (
        "3dedefdc819007074a1a53c7a606d1e5661ac0cc19984212aa809c3a9f6bb646"
    )


def test_parse_labels_follow_the_emotion_order() -> None:
    examples = parse_emotions(_scenarios())

    by_emotion = {e.meta["emotion"]: e.label for e in examples}
    assert by_emotion == {name: i for i, name in enumerate(EMOTIONS)}


def test_parse_lists_the_emotions_in_class_order_not_mapping_order() -> None:
    scenarios = dict(reversed(list(_scenarios().items())))

    examples = parse_emotions(scenarios)

    assert [e.meta["emotion"] for e in examples][::2] == list(EMOTIONS)


def test_parse_builds_uids_from_the_emotion_and_source_position() -> None:
    examples = parse_emotions(_scenarios())

    assert examples[0].uid == "emotion-happiness-0"
    assert examples[1].uid == "emotion-happiness-1"
    assert examples[2].uid == "emotion-sadness-0"


def test_parse_strips_scenarios_and_drops_empty_ones() -> None:
    scenarios = _scenarios(anger=["  Cut off in traffic. ", "", "  "])

    anger = [e for e in parse_emotions(scenarios) if e.label == 2]

    assert [(e.uid, e.text) for e in anger] == [
        ("emotion-anger-0", "Cut off in traffic.")
    ]


def test_parse_keeps_the_first_copy_of_a_repeated_scenario() -> None:
    scenarios = _scenarios(disgust=["Mold.", "Slime.", "Mold."])

    disgust = [e for e in parse_emotions(scenarios) if e.label == 4]

    assert [(e.uid, e.text) for e in disgust] == [
        ("emotion-disgust-0", "Mold."),
        ("emotion-disgust-1", "Slime."),
    ]


def test_parse_assigns_splits_by_uid() -> None:
    scenarios = _scenarios(
        happiness=[f"Happy scenario {i}." for i in range(300)]
    )

    examples = parse_emotions(scenarios)

    expected = assign_splits([e.uid for e in examples], dataset="emotions")
    assert [e.split for e in examples] == expected


def test_parse_requires_every_emotion() -> None:
    scenarios = _scenarios()
    del scenarios["fear"]

    with pytest.raises(ValueError, match="fear"):
        parse_emotions(scenarios)


def test_parse_rejects_an_unknown_emotion() -> None:
    with pytest.raises(ValueError, match="joy"):
        parse_emotions(_scenarios(joy=["Cake."]))


def test_parse_replies_are_unlabeled_and_skip_empty_ones() -> None:
    examples = parse_emotion_replies(["", " That game I can see ", "Sure."])

    assert [(e.uid, e.text, e.label) for e in examples] == [
        ("emotion-reply-1", "That game I can see", -1),
        ("emotion-reply-2", "Sure.", -1),
    ]
    assert examples[0].meta == {}


def test_parse_replies_keeps_the_first_copy_of_a_duplicate_reply() -> None:
    examples = parse_emotion_replies(["Sure.", "Sure.", "Fine."])

    assert [e.uid for e in examples] == ["emotion-reply-0", "emotion-reply-2"]
    assert [e.text for e in examples] == ["Sure.", "Fine."]


def test_parse_replies_assigns_splits_by_uid() -> None:
    replies = [f"Reply number {i}." for i in range(300)]

    examples = parse_emotion_replies(replies)

    expected = assign_splits(
        [e.uid for e in examples], dataset="emotion_replies"
    )
    assert [e.split for e in examples] == expected


@pytest.fixture
def served(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_fetch(source: object, *, file: str, **kwargs: object) -> Path:
        calls.append({"source": source, "file": file, **kwargs})
        path = tmp_path / Path(file).name
        if file == REPLIES_FILE:
            payload = ["", "A reply."]
        else:
            name = Path(file).stem
            payload = [f"{name} one", f"{name} two"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    monkeypatch.setattr(emotions, "fetch", fake_fetch)
    return calls


def test_load_emotions_reads_one_file_per_emotion(
    served: list[dict[str, object]], tmp_path: Path
) -> None:
    examples = load_emotions(cache=tmp_path, offline=True)

    assert [c["file"] for c in served] == [
        f"data/emotions/{name}.json" for name in EMOTIONS
    ]
    assert all(c["source"] is SOURCE for c in served)
    assert all(c["cache"] == tmp_path for c in served)
    assert all(c["offline"] is True for c in served)
    assert len(examples) == 12
    assert examples[-1].text == "surprise two"


def test_load_emotion_replies_reads_the_replies_file(
    served: list[dict[str, object]],
) -> None:
    examples = load_emotion_replies()

    assert [c["file"] for c in served] == [REPLIES_FILE]
    assert [e.text for e in examples] == ["A reply."]


def test_load_rejects_a_file_that_is_not_a_list_of_strings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    monkeypatch.setattr(emotions, "fetch", lambda *a, **k: path)

    with pytest.raises(ValueError, match="list of strings"):
        load_emotion_replies()


@pytest.mark.network
def test_load_emotions_reads_the_pinned_files() -> None:
    examples = load_emotions()

    counts = collections.Counter(e.meta["emotion"] for e in examples)
    assert counts == {
        "happiness": 203,
        "sadness": 228,
        "anger": 203,
        "fear": 203,
        "disgust": 204,
        "surprise": 208,
    }
    assert len({e.uid for e in examples}) == len(examples)
    assert len({e.text for e in examples}) == len(examples)


@pytest.mark.network
def test_load_emotion_replies_reads_the_pinned_file() -> None:
    examples = load_emotion_replies()

    assert len(examples) == 581
    assert len({e.uid for e in examples}) == 581
