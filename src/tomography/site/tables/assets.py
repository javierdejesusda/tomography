"""Table A1: Models, judges and datasets with pins and licenses."""

from tomography.site import assets
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.viz import paper


def _cite(keys: tuple[str, ...]) -> str:
    return f"[[cite:{','.join(keys)}]]"


def _license(text: str, keys: tuple[str, ...] = ()) -> str:
    """Capitalises a license and marks the reference a license tells to cite.

    Args:
        text: The license as stated by its source.
        keys: The citation keys of the asset.

    Returns:
        The cell text.
    """
    text = text[:1].upper() + text[1:]
    if keys and "(cite " in text:
        return text.replace(")", f" {_cite(keys)})")
    return text


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the code pins.

    Args:
        manifest: The run resolver; unused, the rows are static.

    Returns:
        The table and its caption parts.
    """
    del manifest
    rows = []
    for a in assets.ASSETS:
        if a.key in assets.FAMILIES:
            rows += [
                [
                    paper.display_name(key),
                    f"{a.name} checkpoint",
                    _cite(a.cite),
                    pin.repo_id,
                    pin.revision[:12],
                    _license(pin.license),
                ]
                for key, pin in assets.CHECKPOINTS.items()
                if key.startswith(a.key)
            ]
        else:
            rows.append(
                [
                    a.name,
                    a.used,
                    _cite(a.cite),
                    a.location,
                    a.revision,
                    _license(a.license, a.cite),
                ]
            )
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
