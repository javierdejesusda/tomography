"""Appendix table: The knowledge readers recomputed beside their controls.

One row per checkpoint and contrast: the AUROC of the committed reader, the
same draws refitted in float32, the published reader, and the controls of
the contrast, which are the random direction and the shuffled-label reader
of each draw and the committed classifiers on the words, on the characters
and on the popularity of the subject. The page gives the mean over the draws
of each control and the data file adds its lowest and highest draw.
"""

from tomography.site import cells
from tomography.site import stats
from tomography.site import views
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.viz import paper

COLUMNS = [
    "Model",
    "Contrast",
    "Committed",
    "Recomputed",
    "Published",
    "Random",
    "Shuffled labels",
    "Words",
    "Characters",
    "Popularity",
]
DATA_COLUMNS = (
    "Random: lowest draw",
    "Random: highest draw",
    "Shuffled labels: lowest draw",
    "Shuffled labels: highest draw",
)
METHOD_NAMES = {
    "logistic": "reader",
    "pca_diff": "PCA direction",
    "mean_diff": "difference of means",
    "random": "random direction",
}


def _name(value: str) -> str:
    return value.replace("_", " ")


def _ends(found: views.Spread) -> tuple[str, str]:
    return cells.num(found.low), cells.num(found.high)


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    rows: list[list[str]] = []
    ends: list[tuple[str, ...]] = []
    perfect: dict[str, None] = {}
    starts: list[int] = []
    within: dict[str, bool] = {}
    misses: list[str] = []
    reader_gaps: list[float] = []
    tolerance = 0.0
    for model in manifest.models(views.EXPERIMENT):
        found = views.load(manifest, model)
        if found is None:
            continue
        tolerance = found.tolerance
        starts.append(len(rows))
        within[model] = True
        for place, contrast in enumerate(found.contrasts):
            rows.append(
                [
                    paper.display_name(model) if place == 0 else "",
                    _name(contrast.name),
                    cells.num(contrast.committed),
                    cells.num(contrast.recomputed),
                    cells.num(contrast.published),
                    cells.num(contrast.random.mean),
                    cells.num(contrast.shuffled.mean),
                    cells.num(contrast.text_word),
                    cells.num(contrast.text_char),
                    cells.num(contrast.popularity),
                ]
            )
            ends.append((*_ends(contrast.random), *_ends(contrast.shuffled)))
            if "1.000" in (
                cells.num(contrast.text_word),
                cells.num(contrast.text_char),
            ):
                perfect[_name(contrast.name)] = None
            reader_gaps.append(contrast.gaps[views.READER])
            for method, gap in contrast.gaps.items():
                if gap > found.tolerance:
                    within[model] = False
                    misses.append(
                        f"the {METHOD_NAMES[method]} of "
                        f"{_name(contrast.name)} on "
                        f"{paper.display_name(model)}, whose mean over the "
                        f"five draws moves by {cells.num(gap)}"
                    )
    counts = {
        "with_run": stats.per_role(
            {model: model in within for model in manifest.checkpoints()}
        ),
        "within_tolerance": stats.per_role(within),
    }
    numbers = {
        "rows": float(len(rows)),
        "tolerance": tolerance,
        "misses": float(len(misses)),
        "reader_gap_max": max(reader_gaps),
    }
    claim = (
        "The AUROC of every knowledge reader, recomputed beside its "
        "committed value and beside the controls of its contrast."
    )
    detail = (
        "Every value is an AUROC on the test questions of the contrast at "
        "the layer of the reader, for "
        f"{stats.per_role_text(counts['with_run'])}. Committed is the mean "
        "over the five draws of the committed reading run, which ran in the "
        "default precision of the checkpoint; recomputed refits the same "
        "five draws on hidden states extracted again in float32, and "
        f"differs from it by at most {cells.num(max(reader_gaps))}. "
        "Published is the published reader, fitted on every training pair "
        "and not refitted, so it is another estimate and has no committed "
        "value. The other columns are controls. Random is the random "
        "direction of each draw and shuffled labels is the reader of each "
        "draw refitted with its labels shuffled inside each stratum: both "
        "give the mean over the five draws, and the data file adds the "
        "lowest and the highest draw. Words, characters and popularity are "
        "the committed classifiers on the question text and on the page "
        "views of the subject, which the recomputation reproduces exactly; "
        f"a contrast without a popularity control reads {cells.MISSING}. "
        "The classifier on the words or the one on the characters reads "
        f"1.000 for {stats.join_and(list(perfect))} on at least one "
        "checkpoint each: there the text of the questions alone separates "
        "the two sides. The recomputed mean over the draws of every method, "
        "the random direction included, stays within "
        f"{cells.num(tolerance)} of the committed one on "
        f"{stats.per_role_text(counts['within_tolerance'])} with stored "
        "views."
    )
    notes = (
        [
            f"Outside the tolerance of {cells.num(tolerance)}: "
            f"{stats.join_and(misses)}."
        ]
        if misses
        else []
    )
    return TableOutput(
        claim,
        detail,
        COLUMNS,
        rows,
        "ll" + "r" * 8,
        notes,
        numbers,
        row_groups=tuple(starts),
        counts=counts,
        compact_print=True,
        data_columns=DATA_COLUMNS,
        data_cells=tuple(ends),
    )
