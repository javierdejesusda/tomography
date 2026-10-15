"""Glue test: the three copies of the emotion names agree."""

from tomography import judges
from tomography import templates
from tomography.datasets import emotions


def test_the_three_emotion_copies_are_equal() -> None:
    assert emotions.EMOTIONS == judges.EMOTIONS == templates.EMOTIONS
