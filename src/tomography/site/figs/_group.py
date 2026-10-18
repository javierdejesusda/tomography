"""Drawing and bookkeeping shared by Figures 5 to 9.

Lines connect checkpoints only within one family and one role (GPT-2, Qwen3
base, Qwen3 post-trained), ordered by parameter count, so no line crosses
from GPT-2 to Qwen3 or from a base to a post-trained model.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import csv
import io
import math
from typing import Any

from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from tomography.site import params
from tomography.site import store
from tomography.viz import axes
from tomography.viz import paper

ROLES = ("GPT-2", "Qwen3 base", "Qwen3 post-trained")
Z95 = 1.959963984540054


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


def wilson(successes: int, total: int) -> tuple[float, float]:
    """Computes the 95 percent Wilson score interval of a proportion.

    Args:
        successes: Number of successes.
        total: Number of trials.

    Returns:
        The lower and upper ends.

    Raises:
        ValueError: If the number of trials is not positive.
    """
    if total <= 0:
        raise ValueError("the number of trials must be positive")
    p = successes / total
    z2 = Z95**2
    centre = (p + z2 / (2 * total)) / (1 + z2 / total)
    half = (
        Z95
        * math.sqrt(p * (1 - p) / total + z2 / (4 * total**2))
        / (1 + z2 / total)
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def csv_text(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Writes rows as CSV text; None becomes an empty field.

    Args:
        header: Column names.
        rows: One sequence of cells per row.

    Returns:
        The CSV document.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(["" if cell is None else cell for cell in row])
    return buffer.getvalue()


def draw_series(
    ax: Axes,
    values: Mapping[str, float],
    color: str,
    *,
    lo: Mapping[str, float] | None = None,
    hi: Mapping[str, float] | None = None,
    label: str | None = None,
) -> None:
    """Draws one quantity against parameter count, one line per role.

    Args:
        ax: The axes, with a size axis.
        values: Checkpoint key to value.
        color: Line color.
        lo: Optional lower whisker end per checkpoint.
        hi: Optional upper whisker end per checkpoint.
        label: Legend label, set on the first line only.
    """
    first = True
    for members in lines_by_role(list(values)).values():
        style = axes.model_style(members[0])
        xs = [params.param_count(m) for m in members]
        ys = [values[m] for m in members]
        ax.plot(
            xs,
            ys,
            color=color,
            ls=style.linestyle,
            marker=style.marker,
            ms=3.2,
            mfc=color if style.filled else "white",
            mec=color,
            mew=0.8,
            lw=1.0,
            zorder=3,
            label=label if first else None,
        )
        first = False
        if lo is not None and hi is not None:
            for m, x in zip(members, xs, strict=True):
                ax.plot([x, x], [lo[m], hi[m]], color=color, lw=0.6, zorder=2)


def role_handles() -> list[Line2D]:
    """Builds legend handles that explain the three line styles.

    Returns:
        One handle per role, drawn in ink color.
    """
    handles = []
    for name, model in zip(
        ROLES, ("gpt2", "qwen3-1.7b-base", "qwen3-1.7b"), strict=True
    ):
        style = axes.model_style(model)
        handles.append(
            Line2D(
                [],
                [],
                color=paper.INK,
                ls=style.linestyle,
                marker=style.marker,
                ms=3.2,
                mfc=paper.INK if style.filled else "white",
                mec=paper.INK,
                mew=0.8,
                lw=1.0,
                label=name,
            )
        )
    return handles
