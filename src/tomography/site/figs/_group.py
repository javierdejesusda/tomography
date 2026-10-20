"""Drawing and bookkeeping shared by Figures 5 to 9.

Lines connect checkpoints only within one family and one role (GPT-2, Qwen3
base, Qwen3 post-trained), ordered by parameter count, so no line crosses
from GPT-2 to Qwen3 or from a base to a post-trained model.
"""

from collections.abc import Sequence

from matplotlib.axes import Axes

from tomography.site import cells
from tomography.site import params
from tomography.site import store

ROLES = ("GPT-2", "Qwen3 base", "Qwen3 post-trained")


def role(model: str) -> str:
    """Names the family and role of a checkpoint.

    Args:
        model: Checkpoint key.

    Returns:
        `GPT-2`, `Qwen3 base` or `Qwen3 post-trained`.
    """
    if model.startswith("gpt2"):
        return "GPT-2"
    return (
        "Qwen3 post-trained" if store.is_post_trained(model) else "Qwen3 base"
    )


def lines_by_role(models: Sequence[str]) -> dict[str, list[str]]:
    """Groups checkpoints into the lines that may be drawn.

    Args:
        models: Checkpoint keys.

    Returns:
        Role to the checkpoints of that role in increasing size, without
        empty roles.
    """
    found: dict[str, list[str]] = {}
    for name in ROLES:
        members = [m for m in models if role(m) == name]
        if members:
            found[name] = sorted(members, key=params.param_count)
    return found


csv_text = cells.csv_text


def clip_ticks(ax: Axes) -> None:
    """Drops ticks that lie outside the current view of both axes.

    Matplotlib keeps labels for ticks beyond the limits, which then count as
    text outside the figure; fixing the ticks to the visible ones avoids it.

    Args:
        ax: The axes, with its limits already set.
    """
    low, high = ax.get_xlim()
    ax.set_xticks([t for t in ax.get_xticks() if low <= t <= high])
    low, high = ax.get_ylim()
    ax.set_yticks([t for t in ax.get_yticks() if low <= t <= high])
