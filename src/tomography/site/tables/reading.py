"""Table 4: Reader accuracy next to the random direction.

One row per checkpoint. For each concept a pair of columns gives the reader's
accuracy at the layer chosen on validation data and the random direction of
the same run at that layer. Sentiment is the held-out value of the
sample-size run, never the optimistic main run. A star marks a reader whose
margin over the random direction has a conservative interval above zero.
"""

from tomography.site import params
from tomography.site import reading
from tomography.site import stats
from tomography.site import store
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _cells
from tomography.site.tables import _rows
from tomography.viz import paper

CONCEPTS = (
    ("sentiment", "Sentiment (held out)"),
    ("utility", "Utility"),
    ("morality", "Morality"),
    ("truth", "Truth"),
    ("honesty", "Honesty (transfer)"),
)


def _score(
    manifest: Manifest, concept: str, model: str
) -> reading.Score | None:
    if concept == "sentiment":
        return reading.heldout_sentiment(manifest, model)
    exp = {"utility": "r2", "morality": "r3", "truth": "r7", "honesty": "r5"}
    found = reading.load(manifest, exp[concept], model)
    if found is None:
        return None
    n = params.eval_pairs(manifest, exp[concept], model)
    return reading.Score(found.layer, found.accuracy, found.random_at_layer, n)


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    models = sorted(
        {m for e in ("p3", "r2", "r3", "r7", "r5") for m in manifest.models(e)},
        key=store.model_rank,
    )
    rows: list[list[str]] = []
    flags: dict[str, dict[str, bool]] = {c: {} for c, _ in CONCEPTS}
    sizes: dict[str, int] = {}
    for model in models:
        row = [paper.display_name(model)]
        for concept, _ in CONCEPTS:
            score = _score(manifest, concept, model)
            if score is None:
                row += [_cells.MISSING, _cells.MISSING]
                continue
            sizes[concept] = score.n
            _, lo, _ = reading.margin(score)
            flags[concept][model] = lo > 0
            row += [
                _cells.num(score.reader) + ("*" if lo > 0 else ""),
                _cells.num(score.random),
            ]
        rows.append(row)
    numbers = {f"above_{c}": float(sum(flags[c].values())) for c, _ in CONCEPTS}
    numbers |= {f"n_{c}": float(sizes[c]) for c, _ in CONCEPTS}
    counts = {f"above_{c}": stats.per_role(flags[c]) for c, _ in CONCEPTS}
    claim = (
        "Reader accuracy at the layer chosen on validation, next to the "
        "random direction of the same run at that layer."
    )
    detail = (
        "An asterisk marks a reader whose margin over the random direction "
        "has a conservative 95 percent interval above zero for the test "
        f"pairs: {sizes['sentiment']} sentiment pairs (sample-size run, 64 "
        f"training pairs), {sizes['utility']:,} utility, "
        f"{sizes['morality']:,} morality and {sizes['truth']} truth pairs, "
        f"and {sizes['honesty']} plain true and false statements for "
        "honesty, read by a direction fitted on the paper's honesty persona "
        "prompts with no refit. Margins with their intervals, the layers and "
        "the in-template honesty values are in [[table:reading_full]]."
    )
    return TableOutput(
        claim,
        detail,
        ["Model", *["Reader", "Random"] * len(CONCEPTS)],
        rows,
        "l" + "rr" * len(CONCEPTS),
        [],
        numbers,
        groups=tuple((title, 2, False) for _, title in CONCEPTS),
        row_groups=_rows.role_groups(models),
        counts=counts,
    )
