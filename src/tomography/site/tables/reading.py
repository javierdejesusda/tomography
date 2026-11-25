"""Table A4: Reader accuracy next to the random direction.

One row per checkpoint. For each concept a pair of columns gives the reader's
accuracy at the layer chosen on validation data and the random direction of
the same run at that layer. Sentiment is the held-out value of the
sample-size run, never the optimistic main run. A star marks a reader whose
margin over the random direction has a conservative interval above zero.
"""

import json

from tomography.site import cells
from tomography.site import params
from tomography.site import reading
from tomography.site import stats
from tomography.site import store
from tomography.site.manifest import MODEL_FREE
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _rows
from tomography.viz import paper

CONCEPTS = (
    ("sentiment", "Sentiment (held out)"),
    ("utility", "Utility"),
    ("morality", "Morality"),
    ("truth", "Truth"),
    ("honesty", "Honesty (transfer)"),
)


CONTROLS = (
    ("Words alone (word n-grams)", "word"),
    ("Words alone (character n-grams)", "char"),
)


def _controls(manifest: Manifest) -> dict[tuple[str, str], list[float]]:
    """Reads the model-free text controls of the reading concepts.

    Args:
        manifest: The run resolver.

    Returns:
        For each concept and analyzer, accuracy, lower and upper limit, the
        number of test pairs and the number of training sets.

    Raises:
        FileNotFoundError: If the controls have no run.
    """
    run = manifest.run("tb1", MODEL_FREE)
    if run is None:
        raise FileNotFoundError("results/tb1 has no run")
    table = json.loads((run / "table.json").read_text("utf-8"))["data"]
    column = {name: i for i, name in enumerate(table["columns"])}
    found: dict[tuple[str, str], list[float]] = {}
    for row in table["rows"]:
        key = (row[column["concept"]], row[column["analyzer"]])
        found[key] = [float(row[column[c]]) for c in ("accuracy", "lo", "hi")]
        found[key] += [
            float(row[column["n_pairs"]]),
            float(row[column["n_seeds"]]),
        ]
    return found


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
                row += [cells.MISSING, cells.MISSING]
                continue
            sizes[concept] = score.n
            _, lo, _ = reading.margin(score)
            flags[concept][model] = lo > 0
            row += [
                cells.num(score.reader) + ("*" if lo > 0 else ""),
                cells.num(score.random),
            ]
        rows.append(row)
    numbers = {f"above_{c}": float(sum(flags[c].values())) for c, _ in CONCEPTS}
    numbers |= {f"n_{c}": float(sizes[c]) for c, _ in CONCEPTS}
    counts = {f"above_{c}": stats.per_role(flags[c]) for c, _ in CONCEPTS}
    controls = _controls(manifest)
    for concept, _ in CONCEPTS:
        for label, analyzer in CONTROLS:
            n = controls[(concept, analyzer)][3]
            if int(n) != sizes[concept]:
                raise ValueError(f"{label} uses {int(n)} pairs for {concept}")
    for label, analyzer in CONTROLS:
        row = [label]
        for concept, _ in CONCEPTS:
            value, lo, hi, *_ = controls[(concept, analyzer)]
            row += [cells.interval(value, lo, hi), ""]
            numbers[f"{analyzer}_{concept}"] = value
            numbers[f"{analyzer}_{concept}_lo"] = lo
            numbers[f"{analyzer}_{concept}_hi"] = hi
            numbers[f"{analyzer}_{concept}_above"] = float(lo > 0.5)
        rows.append(row)
    claim = (
        "Reader accuracy at the layer chosen on validation, beside the "
        "random direction of the same run at that layer."
    )
    detail = (
        "An asterisk marks a margin over the random direction with a "
        "conservative 95 percent interval above zero (test pairs as in "
        "[[fig:reading]]). Honesty is a transfer: a direction fitted on "
        f"honesty persona prompts reads {sizes['honesty']} plain true and "
        "false statements with no refit. The last two rows are the words "
        "alone, as means over training sets with intervals for the test "
        "pairs only. In the honesty column they are fitted on the true and "
        "false labels, not the persona labels of the reader, so they control "
        "the transfer pairs only."
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
        row_groups=(*_rows.role_groups(models), len(models)),
        counts=counts,
        wrap_labels=True,
        compact_print=True,
    )
