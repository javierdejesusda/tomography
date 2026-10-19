"""Appendix figure: The paper's finder and logistic regression, Qwen3.

The Qwen3 counterpart of the GPT-2 finder figure: one row per base or
post-trained checkpoint and one column per task.
"""

from tomography.site import stats
from tomography.site.figs import a_finders_gpt2
from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput

ROW_HEIGHT = 1.0
BASE_HEIGHT = 0.6
MAX_HEIGHT = 8.0


def _height(rows: int) -> float:
    return min(MAX_HEIGHT, BASE_HEIGHT + ROW_HEIGHT * rows)


def build(manifest: Manifest) -> FigureOutput:
    """Builds the Qwen3 figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    return a_finders_gpt2.split(
        manifest, lambda m: stats.role(m) != "gpt2", _height, "Qwen3"
    )
