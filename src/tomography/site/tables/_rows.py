"""Row layout helpers shared by the tables.

Rows are checkpoints in paper order; a rule separates GPT-2 from Qwen3 and
nothing separates the Qwen3 pairs. A star marks an interval that excludes
zero.
"""

from collections.abc import Sequence


def role_groups(models: Sequence[str]) -> tuple[int, ...]:
    """Finds the first row of each model family.

    Args:
        models: Checkpoint keys in row order.

    Returns:
        Row indices of the first GPT-2 row and the first Qwen3 row; a family
        without rows is left out.
    """
    starts = []
    for family in ("gpt2", "qwen3"):
        for i, model in enumerate(models):
            if model.startswith(family):
                starts.append(i)
                break
    return tuple(starts)


def star(lo: float, hi: float) -> str:
    """Marks an interval that lies on one side of zero.

    Args:
        lo: Lower limit.
        hi: Upper limit.

    Returns:
        An asterisk when the interval excludes zero, else an empty string.
    """
    return "*" if lo > 0 or hi < 0 else ""
