"""Table A1: Models, judges and datasets with pins and licenses."""

from tomography.site import assets
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput


def _cite(keys: tuple[str, ...]) -> str:
    return f"[[cite:{','.join(keys)}]]"


def _license(text: str) -> str:
    return text[:1].upper() + text[1:]


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the code pins.

    Args:
        manifest: The run resolver; unused, the rows are static.

    Returns:
        The table and its caption parts.
    """
    del manifest
    rows = [
        [
            a.name,
            a.used,
            _cite(a.cite),
            a.location,
            a.revision,
            _license(a.license),
        ]
        for a in assets.ASSETS
    ]
    claim = (
        "Models, judges and datasets with their pinned revisions and their "
        "licenses as stated by their sources."
    )
    detail = (
        "Revision is the leading 12 characters of the pinned commit of the "
        "Hugging Face Hub repository or Git repository, or the SHA-256 "
        "digest of the pinned file. Sources without a permissive license are "
        "downloaded at run time and are not redistributed in the repository."
    )
    return TableOutput(
        claim,
        detail,
        [
            "Asset",
            "Used for",
            "Citation",
            "Hub ID or source",
            "Revision",
            "License as stated",
        ],
        rows,
        "llllll",
        [],
        {"assets": float(len(rows))},
        row_groups=(),
    )
