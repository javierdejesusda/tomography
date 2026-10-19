"""Table A1: Reading accuracy with intervals and selected layers.

One block per concept; each row is a checkpoint with the reader's accuracy
and its 95 percent interval at the layer chosen on validation data, the
random direction of the same run at that layer, and that layer. Table 2 of
the main text shows the same accuracies in a wide form.
"""

from tomography.site import reading
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.viz import paper

BLOCKS = (
    ("r1", "R1 sentiment"),
    ("r2", "R2 utility"),
    ("r3", "R3 morality"),
    ("r7", "R7 truth"),
)
EXPLORATORY = ("r5", "R5 honesty transfer (exploratory)")


def _interval(found: reading.Reading) -> str:
    return f"{found.accuracy:.3f} [{found.lo:.3f}, {found.hi:.3f}]"


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    rows: list[list[str]] = []
    for exp, title in (*BLOCKS, EXPLORATORY):
        first = True
        for model in manifest.models(exp):
            found = reading.load(manifest, exp, model)
            if found is None:
                continue
            random = "n/a" if exp == "r5" else f"{found.random_at_layer:.3f}"
            rows.append(
                [
                    title if first else "",
                    paper.display_name(model),
                    _interval(found),
                    random,
                    str(found.layer),
                ]
            )
            first = False
    claim = (
        "Accuracy of the reader with its 95 percent interval, the random "
        "direction of the same run and the selected layer, for every "
        "concept and checkpoint of Table 2."
    )
    detail = (
        "Accuracy is read at the layer chosen on validation data, on the "
        "test pairs; intervals are 95 percent intervals over the evaluation "
        "pairs. Random is the accuracy of a random direction of the same "
        "run at the same layer. R5 is exploratory and has no random "
        "direction at its layer."
    )
    return TableOutput(
        claim,
        detail,
        ["Concept", "Model", "Reader [95% interval]", "Random", "Layer"],
        rows,
        "llrrr",
        [],
        {},
    )
