"""Table 2: Notation and terms used on the page.

A static table: it holds no result, only the meaning of each symbol and term,
so the page can name experiments by what they test and keep the run codes of
the repository here.
"""

from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput

ELL = chr(0x2113)
MINUS = chr(0x2212)

ROWS = (
    (
        "pair",
        "Two inputs that differ in the concept: a positive member (label 1) "
        "and a negative member (label 0).",
    ),
    (
        f"h(x, {ELL})",
        f"Residual-stream activation of the last token of input x after "
        f"layer {ELL}.",
    ),
    (
        f"v({ELL})",
        f"Unit direction at layer {ELL} found by a finder on the training "
        "pairs and oriented with their labels.",
    ),
    (
        f"u({ELL})",
        f"Random unit direction at layer {ELL}, oriented on the same "
        "training pairs.",
    ),
    (
        "finder",
        "Method that turns training pairs into a direction: PCA of paired "
        "differences (the paper's), PCA of activations, mean difference, "
        "logistic regression or k-means.",
    ),
    (
        "accuracy",
        "Share of held-out pairs whose positive member projects higher on "
        "the direction; ties count one half; 0.5 for a direction with no "
        "information.",
    ),
    (
        "margin",
        "Accuracy of v minus accuracy of u on the same test pairs at the "
        "same layer.",
    ),
    ("relative depth", "Layer index divided by the number of layers."),
    (
        "layer chosen on validation",
        "Layer with the highest accuracy on validation or select pairs that "
        "the reported score does not use.",
    ),
    (
        "AUROC",
        "Probability that a positive item scores above a negative one; 0.5 "
        "is chance.",
    ),
    (
        "relative coefficient c",
        f"Steering strength as a multiple of r, the mean residual norm at "
        f"layer {ELL} on calibration text.",
    ),
    (
        "win rate",
        "Share of decisive prompt pairs in which the completion steered at "
        f"+c is judged more positive than the one steered at {MINUS}c.",
    ),
    (
        "role",
        "GPT-2, Qwen3 base or Qwen3 post-trained; every count is given per "
        "role.",
    ),
    (
        "run codes",
        "The directories of the results folder are named by an experiment "
        "code, a letter and a number; the page names experiments by what "
        "they test and the README lists the codes.",
    ),
)


def build(manifest: Manifest) -> TableOutput:
    """Builds the static notation table.

    Args:
        manifest: The run resolver; unused, the rows are static.

    Returns:
        The table and its caption parts.
    """
    del manifest
    return TableOutput(
        "Symbols and terms used in the text.",
        "Experiment names in the repository are given in the last row.",
        ["Symbol or term", "Meaning"],
        [list(row) for row in ROWS],
        "ll",
        [],
        {},
        row_groups=(),
    )
