"""Table A3: Every reading at its reported layer.

One block per concept with a row per checkpoint: the layer, its relative
depth, the reader, the random direction of the same run at that layer, the
margin with its conservative interval for the test pairs, and the test size.
The main sentiment run chooses its layer on the pairs it is scored on and is
labelled optimistic; the held-out block is what the main text uses.
"""

from tomography.site import params
from tomography.site import reading
from tomography.site import runfiles
from tomography.site import store
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _cells
from tomography.viz import paper

BLOCKS = (
    (
        "sentiment_r1",
        "Sentiment, layer chosen on the scored pairs (optimistic)",
    ),
    ("sentiment", "Sentiment, held out"),
    ("utility", "Utility"),
    ("morality", "Morality"),
    ("truth", "Truth"),
    ("template", "Honesty in template"),
    ("honesty", "Honesty transfer"),
)
EXPERIMENT = {
    "sentiment_r1": "r1",
    "sentiment": "p3",
    "utility": "r2",
    "morality": "r3",
    "truth": "r7",
    "template": "r5",
    "honesty": "r5",
}


def _score(
    manifest: Manifest, concept: str, model: str
) -> reading.Score | None:
    if concept == "sentiment":
        return reading.heldout_sentiment(manifest, model)
    if concept == "template":
        return reading.in_template(manifest, model)
    if concept == "sentiment_r1":
        found = reading.load(manifest, "r1", model)
        run = manifest.run("r1", model)
        if found is None or run is None:
            return None
        n = int(runfiles.summary(run)["n_eval_pairs"])
        return reading.Score(
            found.layer, found.accuracy, found.random_at_layer, n
        )
    exp = EXPERIMENT[concept]
    found = reading.load(manifest, exp, model)
    if found is None:
        return None
    n = params.eval_pairs(manifest, exp, model)
    return reading.Score(found.layer, found.accuracy, found.random_at_layer, n)


def _depth(model: str, layer: int) -> float:
    config = params.model_config(model)
    count = config.get("n_layer") or config["num_hidden_layers"]
    return layer / int(count)


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    models = sorted(
        {
            m
            for e in ("p3", "r1", "r2", "r3", "r5", "r7")
            for m in manifest.models(e)
        },
        key=store.model_rank,
    )
    rows: list[list[str]] = []
    sizes: dict[str, int] = {}
    starts: list[int] = []
    for concept, label in BLOCKS:
        starts.append(len(rows))
        for k, model in enumerate(models):
            found = _score(manifest, concept, model)
            head = [label if k == 0 else "", paper.display_name(model)]
            if found is None:
                rows.append(head + [_cells.MISSING] * 6)
                continue
            sizes[concept] = found.n
            d, lo, hi = reading.margin(found)
            rows.append(
                [
                    *head,
                    str(found.layer),
                    _cells.num(_depth(model, found.layer)),
                    _cells.num(found.reader),
                    _cells.num(found.random),
                    _cells.pm(d, (hi - lo) / 2),
                    f"{found.n:,}",
                ]
            )
    claim = (
        "Every reading at its reported layer, with its margin over the "
        "random direction."
    )
    detail = (
        "The first sentiment block chooses the layer on the same "
        f"{sizes['sentiment_r1']} SST-2 validation pairs it is scored on and "
        "is optimistic; the held-out block chooses it on the select split of "
        f"the sample-size run and scores {sizes['sentiment']} pairs. "
        "Margins carry conservative 95 percent intervals "
        "for the test pairs; the intervals stored with the runs resample the "
        "readers only and are in the data files. Honesty in template is "
        "scored on held-out persona prompts and honesty transfer on plain "
        "true and false statements."
    )
    return TableOutput(
        claim,
        detail,
        [
            "Concept",
            "Model",
            "Layer",
            "Relative depth",
            "Reader",
            "Random",
            "Margin",
            "Test pairs",
        ],
        rows,
        "llrrrrrr",
        [],
        {"blocks": float(len(BLOCKS))},
        row_groups=tuple(starts),
    )
