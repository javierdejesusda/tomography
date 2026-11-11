"""Tests for the rubric an external judge reads."""

import hashlib
from importlib import resources

import pytest

from tomography.judging import rubric
from tomography.judging.rubric import CRITERIA
from tomography.judging.rubric import SCORES
from tomography.judging.rubric import role_criteria
from tomography.judging.rubric import rubric_sha256
from tomography.judging.rubric import rubric_text

PINNED_SHA256 = (
    "e95573bae4bf66afd389d58b46a74bc3bba3ee3206f11dc9116eda7104a7cd13"
)
DASHES = (chr(0x2014), chr(0x2013))


def test_the_criteria_and_scores_follow_the_published_protocol() -> None:
    assert CRITERIA == ("concept", "instruction", "fluency")
    assert SCORES == (0, 1, 2)


def test_the_rubric_defines_every_criterion_and_the_answer_format() -> None:
    text = rubric_text()

    for criterion in CRITERIA:
        assert f"\n{criterion}: " in text
    for field in ("item_id", "prompt", "concept", "generation"):
        assert f'"{field}"' in text
    assert "0, 1 or 2" in text
    assert (
        '{"item_id": "0123456789abcdef", "concept": 2, "instruction": 1, '
        '"fluency": 2}'
    ) in text


def test_the_rubric_is_plain_ascii_without_dash_characters() -> None:
    text = rubric_text()

    assert text.isascii()
    assert not any(dash in text for dash in DASHES)
    assert "\r" not in text


def test_the_rubric_is_read_from_the_package_data_file() -> None:
    stored = resources.files("tomography.judging").joinpath("rubric.txt")

    assert rubric_text() == stored.read_bytes().decode("utf-8").replace(
        "\r\n", "\n"
    )


def test_the_hash_is_the_sha256_of_the_text_with_unix_line_ends() -> None:
    text = "first line\nsecond line\n"

    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert rubric_sha256(text) == expected
    assert rubric_sha256(text.replace("\n", "\r\n")) == expected


def test_the_hash_of_the_shipped_rubric_is_pinned() -> None:
    assert rubric_sha256(rubric_text()) == PINNED_SHA256


def test_a_changed_rubric_changes_the_hash() -> None:
    assert rubric_sha256(rubric_text() + "x") != PINNED_SHA256


@pytest.mark.parametrize(
    ("post_trained", "expected"),
    [
        (False, ("concept", "fluency")),
        (True, ("concept", "instruction", "fluency")),
    ],
    ids=["base", "post_trained"],
)
def test_a_base_role_has_no_instruction_criterion(
    post_trained: bool, expected: tuple[str, ...]
) -> None:
    assert role_criteria(post_trained=post_trained) == expected


def test_the_module_names_no_person_or_vendor() -> None:
    source = resources.files("tomography.judging").joinpath("rubric.py")
    text = (source.read_text("utf-8") + rubric_text()).lower()

    assert rubric.__doc__
    for word in ("openai", "anthropic", "claude", "gpt-4"):
        assert word not in text


def test_the_rubric_points_a_judge_to_the_criteria_file_and_no_stage() -> None:
    text = rubric_text()

    assert "criteria.json" in text
    for word in ("tune", "tuning", "held_out", "held-out", "stage"):
        assert word not in text.lower()
