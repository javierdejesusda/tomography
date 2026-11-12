"""The rubric an external judge reads, and the criteria of each role.

The rubric text is package data (`rubric.txt`), so the exact words a judge was
given are versioned with the code, and its SHA-256 is stored with every
judgment. The three criteria and the 0 to 2 scale follow the judge prompts of
AxBench (Wu et al., 2025).
"""

import hashlib
from importlib import resources

CRITERIA: tuple[str, ...] = ("concept", "instruction", "fluency")
SCORES: tuple[int, ...] = (0, 1, 2)

_RUBRIC_FILE = "rubric.txt"


def rubric_text() -> str:
    """Reads the rubric shipped with the package.

    Returns:
        The text with LF line ends, whatever the line ends of the checkout.
    """
    stored = resources.files("tomography.judging").joinpath(_RUBRIC_FILE)
    return stored.read_bytes().decode("utf-8").replace("\r\n", "\n")


def rubric_sha256(text: str) -> str:
    """Hashes a rubric text.

    Args:
        text: The rubric.

    Returns:
        The hex SHA-256 of the UTF-8 text with LF line ends, so a checkout
        with other line ends gives the same digest.
    """
    normalized = text.replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def role_criteria(*, post_trained: bool) -> tuple[str, ...]:
    """Names the criteria a checkpoint of a role is judged on.

    Args:
        post_trained: Whether the checkpoint is instruction tuned.

    Returns:
        `CRITERIA` for a post-trained checkpoint. A base checkpoint continues
        its prompt and follows no instruction, so it is judged on the concept
        and the fluency only.
    """
    if post_trained:
        return CRITERIA
    return tuple(name for name in CRITERIA if name != "instruction")
