"""Table 7: TruthfulQA and honesty.

Stub: `build` is implemented together with its test in
`tests/test_site_table7.py`.
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
    raise NotImplementedError("Table 7")
