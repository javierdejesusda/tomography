"""Table 4: Reader accuracy next to the random direction.

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


BASELINES = (
    ("Words alone (word n-grams)", "word"),
    ("Words alone (character n-grams)", "char"),
)


def _baselines(manifest: Manifest) -> dict[tuple[str, str], list[float]]:
    """Reads the model-free text baselines of the reading concepts.

    Args:
        manifest: The run resolver.

    Returns:
        For each concept and analyzer, accuracy, lower and upper limit, the
        number of test pairs and the number of training sets.

    Raises:
        FileNotFoundError: If the baselines have no run.
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
    baselines = _baselines(manifest)
    for concept, _ in CONCEPTS:
        for label, analyzer in BASELINES:
            n = baselines[(concept, analyzer)][3]
            if int(n) != sizes[concept]:
                raise ValueError(f"{label} uses {int(n)} pairs for {concept}")
    for label, analyzer in BASELINES:
        row = [label]
        for concept, _ in CONCEPTS:
            value, lo, hi, *_ = baselines[(concept, analyzer)]
            row += [cells.interval(value, lo, hi), ""]
            numbers[f"{analyzer}_{concept}"] = value
            numbers[f"{analyzer}_{concept}_lo"] = lo
            numbers[f"{analyzer}_{concept}_hi"] = hi
            numbers[f"{analyzer}_{concept}_above"] = float(lo > 0.5)
        rows.append(row)
    seeds = {
        concept: {int(baselines[(concept, a)][4]) for _, a in BASELINES}
        for concept, _ in CONCEPTS
    }
    others = {
        n for c, found in seeds.items() if c != "sentiment" for n in found
    }
    if len(seeds["sentiment"]) != 1 or len(others) != 1:
        raise ValueError("the baselines differ in their training sets")
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
        "the in-template honesty values are in [[table:reading_full]]. The "
        "last two rows give the words alone, a text classifier on word or "
        "character n-grams of the pair texts, with no model: each value is "
        "the mean over the training sets "
        f"({next(iter(seeds['sentiment']))} for the held-out sentiment "
        f"split and {next(iter(others))} for the others) and its interval "
        "covers the sampling of the test pairs only, not the spread over "
        "training sets. The "
        "honesty baseline is a classifier fitted on the true and false "
        "labels of the facts, not on the persona labels of the reader, so "
        "it is a baseline for the transfer pairs only."
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
    )
