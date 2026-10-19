"""Measures the gap between a panel letter and the left title beside it."""

from matplotlib.figure import Figure
from matplotlib.text import Text

from tomography.viz import checks


def letter_title_gaps(fig: Figure) -> list[float]:
    """Gives the gap in points from each panel letter to its left title.

    Args:
        fig: The figure.

    Returns:
        One gap per panel that has both a bold letter and a left title.
    """
    checks.draw(fig)
    gaps = []
    for ax in fig.axes:
        letters = [t for t in ax.texts if t.get_text().startswith(r"$\mathbf")]
        title = ax.get_title(loc="left")
        if not letters or not title:
            continue
        (name,) = [
            c
            for c in ax.get_children()
            if isinstance(c, Text) and c.get_text() == title
        ]
        edge = name.get_window_extent().x0 - letters[0].get_window_extent().x1
        gaps.append(edge * 72.0 / fig.dpi)
    return gaps
