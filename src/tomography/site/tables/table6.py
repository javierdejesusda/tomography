"""Table 6: Steering and fluency.

Stub: `build` is implemented together with its test in
`tests/test_site_table6.py`.
"""

from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.

    Raises:
        NotImplementedError: Until the table is built.
    """
    raise NotImplementedError("Table 6")
