"""Table A6: The wordings of the sentiment template.

One row per wording that the transfer matrices score, in the order of the
matrix axes, with the change it makes to the paper's wording and the template
text with the stimulus slot.
"""

from tomography.experiments import variants
from tomography.site import runfiles
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _cells
from tomography.templates import ChatFormat

DROPPED = "no_trailing_space"
COLUMNS = ["No.", "Wording", "Kind of change", "Template"]
CHANGES = {
    "paper": "none, the wording of the paper",
    "no_template": "template removed, chat tags kept",
    "bare": "template and chat tags removed",
    "no_chat_tags": "chat tags removed, template kept",
    "neutral_frame": "frame that does not name the concept",
    "no_trailing_question": "closing question removed",
    "paraphrase": "paraphrase of the instruction",
    "scaffold": "closing sentence unrelated to the concept",
    "concept_swap": "concept word replaced by an unrelated one",
}


def _wording(variant: variants.Variant) -> str:
    base = variant.kind.replace("_", " ")
    if "_" not in variant.name or variant.name == variant.kind:
        return base
    return f"{base} {int(variant.name.rsplit('_', 1)[1]) + 1}"


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the template definitions and the matrix axes.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    model = manifest.models("p1")[0]
    run = manifest.run("p1", model)
    assert run is not None
    names = [
        n
        for n in runfiles.data(run, "transfer_sentiment.json")["rows"]
        if n != DROPPED
    ]
    made, _ = variants.template_variants(
        variants.READING_TASKS["sentiment"], ChatFormat("plain")
    )
    by_name = {v.name: v for v in made}
    rows = [
        [
            str(i),
            _wording(by_name[name]),
            CHANGES[by_name[name].kind],
            by_name[name].template.user.replace("\n", "\\n"),
        ]
        for i, name in enumerate(names, 1)
    ]
    claim = "The wordings of the sentiment template."
    detail = (
        "Index numbers are those of [[fig:wording]] and [[fig:a_wording]]. "
        "The template is the user text, with the stimulus in place of "
        "{stimulus} and the concept word in place of {concept}, and the "
        "symbol \\n marks a line break. The wording that differs from the "
        "paper's only in a space token after the assistant tag is left out "
        "of the figures and of this table."
    )
    return TableOutput(
        claim,
        detail,
        COLUMNS,
        rows,
        _cells.check_align("rlll", len(COLUMNS)),
        [],
        {"wordings": float(len(rows))},
        row_groups=(),
    )
