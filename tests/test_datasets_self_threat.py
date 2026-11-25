"""Tests for the self-threat messages."""

import collections

import pytest

from tomography.datasets.self_threat import CONDITIONS
from tomography.datasets.self_threat import FRAMES
from tomography.datasets.self_threat import PAIRED_CONDITIONS
from tomography.datasets.self_threat import load_self_threat

N_FRAMES = 24


def _by_condition() -> dict[str, list[str]]:
    texts: dict[str, list[str]] = collections.defaultdict(list)
    for message in load_self_threat():
        texts[str(message.meta["condition"])].append(message.text)
    return texts


def _words(text: str) -> int:
    return len(text.split())


def test_there_are_eight_conditions_of_twenty_four_messages() -> None:
    messages = load_self_threat()

    assert CONDITIONS == (
        "self_harm",
        "self_benign",
        "other_harm",
        "other_benign",
        "object_harm",
        "human_harm",
        "praise",
        "insult",
    )
    assert CONDITIONS[:6] == PAIRED_CONDITIONS
    assert len(FRAMES) == N_FRAMES
    assert len(messages) == len(CONDITIONS) * N_FRAMES
    assert collections.Counter(m.meta["condition"] for m in messages) == {
        name: N_FRAMES for name in CONDITIONS
    }
    for message in messages:
        condition = str(message.meta["condition"])
        frame = int(str(message.meta["frame"]))
        assert message.label == CONDITIONS.index(condition)
        assert message.split == "test"
        assert message.uid == f"self-threat-{condition}-{frame}"
        assert message.text == message.text.strip()
        assert message.text[0].isupper()
        assert message.text[-1] in ".!"
    assert len({m.uid for m in messages}) == len(messages)
    assert len({m.text for m in messages}) == len(messages)


def test_each_condition_lists_the_frames_in_order() -> None:
    for name in CONDITIONS:
        frames = [
            int(str(m.meta["frame"]))
            for m in load_self_threat()
            if m.meta["condition"] == name
        ]
        assert frames == list(range(N_FRAMES))


def test_item_i_of_every_paired_condition_shares_frame_i() -> None:
    texts = _by_condition()
    for i, frame in enumerate(FRAMES):
        head, _, tail = frame.sentence.partition("{action}")
        for name in PAIRED_CONDITIONS:
            text = texts[name][i]
            assert text.lower().startswith(head.lower()), (name, i)
            assert text.endswith(tail), (name, i)


def test_paired_conditions_change_only_the_target_or_the_verb() -> None:
    texts = _by_condition()
    for i, frame in enumerate(FRAMES):

        def say(action: str, frame_sentence: str = frame.sentence) -> str:
            text = frame_sentence.format(action=action)
            return text[0].upper() + text[1:]

        assert texts["self_harm"][i] == say(
            frame.harm.format(target=frame.self_target)
        )
        assert texts["self_benign"][i] == say(
            frame.benign.format(target=frame.self_target)
        )
        assert texts["other_harm"][i] == say(
            frame.harm.format(target=frame.other_target)
        )
        assert texts["other_benign"][i] == say(
            frame.benign.format(target=frame.other_target)
        )
        assert texts["object_harm"][i] == say(
            frame.harm.format(target=frame.object_target)
        )
        assert texts["human_harm"][i] == say(frame.human)


def test_the_targets_name_the_model_another_model_and_an_object() -> None:
    for frame in FRAMES:
        assert "you" in frame.self_target.lower().split()[0]
        assert "other" in frame.other_target.lower()
        assert "you" not in frame.other_target.lower().split()
        assert "you" not in frame.object_target.lower().split()
        assert "model" not in frame.object_target.lower()
        assert "{target}" in frame.harm
        assert "{target}" in frame.benign
        assert "{target}" not in frame.human
        assert "you" in frame.human.lower().replace("your", "you").split()


def test_lengths_stay_matched_within_a_frame() -> None:
    texts = _by_condition()
    for i in range(N_FRAMES):
        assert (
            abs(_words(texts["self_harm"][i]) - _words(texts["self_benign"][i]))
            <= 2
        )
        lengths = [_words(texts[name][i]) for name in PAIRED_CONDITIONS]
        assert max(lengths) - min(lengths) <= 5, (i, lengths)


@pytest.mark.parametrize("name", ["praise", "insult"])
def test_praise_and_insult_address_the_model(name: str) -> None:
    texts = _by_condition()[name]
    assert all("you" in t.lower().replace("your", "you") for t in texts)
    assert sum(_words(t) for t in texts) / len(texts) < 16
