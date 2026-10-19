"""Table 6: Sentiment steering at relative coefficient 0.1.

One row per checkpoint: the judged sentiment of the completions at both signs
of the coefficient, unsteered and under an instruction, the perplexity of
held-out neutral text under the steered model as a ratio to the unsteered
model, the perplexity of the completions themselves as a ratio, and
distinct-1. Every value is the mean over the steering vectors of the run.
"""

import json
import math
from typing import Any

from tomography.site import runfiles
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _cells
from tomography.site.tables import _rows
from tomography.viz import paper

COEFF = 0.1
MAX_RATIO = 1.5
MIN_DISTINCT = 0.15
SIGNS = (-COEFF, COEFF)
SIGN_NAMES = (_cells.MINUS + "0.1", "+0.1")
GROUPS = (
    ("Judged sentiment at relative coefficient", 3, False),
    ("Judged sentiment under an instruction", 2, False),
    ("Neutral-text perplexity ratio", 2, False),
    ("Completion perplexity ratio", 2, False),
    ("Distinct-1", 2, False),
)


def _mean(
    rows: list[dict[str, Any]], arm: str, coeff: float | None, key: str
) -> float:
    hit = [
        float(r[key])
        for r in rows
        if r["arm"] == arm and (coeff is None or r["coeff"] == coeff)
    ]
    return sum(hit) / len(hit)


def _cells_of(rows: list[dict[str, Any]]) -> list[str]:
    zero_perplexity = _mean(rows, "none", 0.0, "perplexity")
    zero_fluency = _mean(rows, "none", 0.0, "fluency")
    return [
        _cells.num(_mean(rows, "linear", SIGNS[0], "score")),
        _cells.num(_mean(rows, "none", 0.0, "score")),
        _cells.num(_mean(rows, "linear", SIGNS[1], "score")),
        _cells.num(_mean(rows, "prompting-1", None, "score")),
        _cells.num(_mean(rows, "prompting+1", None, "score")),
        *(
            _cells.num(
                _mean(rows, "linear", c, "perplexity") / zero_perplexity, 2
            )
            for c in SIGNS
        ),
        *(
            _cells.num(
                math.exp(zero_fluency - _mean(rows, "linear", c, "fluency")), 2
            )
            for c in SIGNS
        ),
        *(_cells.num(_mean(rows, "linear", c, "distinct_1")) for c in SIGNS),
    ]


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    models = manifest.models("c1")
    rows: list[list[str]] = []
    n_neutral = n_vectors = 0
    for model in models:
        run = manifest.run("c1", model)
        assert run is not None
        config = json.loads((run / "config.json").read_text("utf-8"))
        n_neutral = int(config["n_neutral"])
        n_vectors = int(config["n_steer_seeds"])
        rows.append(
            [
                paper.display_name(model),
                *_cells_of(runfiles.rows(run, "table.json")),
            ]
        )
    claim = "Sentiment steering at relative coefficient 0.1."
    detail = (
        "Judged sentiment is the mean positive-class probability of SiEBERT "
        "over the completions, with a DistilBERT SST-2 classifier as a check "
        "in the data file; the instruction columns give the score under a "
        "negative or a positive instruction instead of steering. "
        f"Neutral-text perplexity is the perplexity of {n_neutral} held-out "
        "WikiText-103 passages under the steered model as a ratio to the "
        "unsteered model; completion perplexity is the perplexity of the "
        "completions themselves as a ratio to that of the unsteered "
        "completions; distinct-1 is the share of distinct tokens in the "
        "completions. Every value is the mean over the "
        f"{n_vectors} steering vectors. "
        f"The fluency rule of [[fig:steering]] uses the neutral-text ratio "
        f"(at most {MAX_RATIO}) and distinct-1 (at least {MIN_DISTINCT})."
    )
    return TableOutput(
        claim,
        detail,
        [
            "Model",
            *SIGN_NAMES[:1],
            "0",
            SIGN_NAMES[1],
            "Negative",
            "Positive",
            *SIGN_NAMES,
            *SIGN_NAMES,
            *SIGN_NAMES,
        ],
        rows,
        "l" + "r" * 11,
        [],
        {"n_neutral": float(n_neutral)},
        groups=GROUPS,
        row_groups=_rows.role_groups(models),
    )
