"""Figure 2: Where and how well a concept can be read.

Stub: `build` is implemented together with its test in
`tests/test_site_fig2.py`.
"""

from tomography.site.manifest import Manifest
from tomography.site.registry import FigureOutput


def build(manifest: Manifest) -> FigureOutput:
    """Builds the figure from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.

    Raises:
        NotImplementedError: Until the figure is built.
    """
    raise NotImplementedError("Figure 2")
