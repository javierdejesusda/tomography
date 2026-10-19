"""Table 5: Truthfulness and honesty readings.

One row per checkpoint: TruthfulQA MC1 of three readers and of the model's
own answer log-probability on the same test questions, the honesty reader on
held-out persona prompts and on plain true and false statements next to its
random direction, and the agreement between seeds.
"""

import collections
from typing import Any

import numpy as np

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _cells
from tomography.site.tables import _rows
from tomography.viz import paper

READERS = ("truthfulqa", "arc", "honesty")
COLUMNS = [
    "Model",
    "TruthfulQA reader",
    "ARC reader",
    "Honesty persona reader",
    "Answer log-probability",
    "Accuracy",
    "Layer",
    "Accuracy",
    "Random",
    "Seed agreement",
]
GROUPS = (
    ("TruthfulQA MC1", 4, False),
    ("Honesty in template", 2, False),
    ("Honesty transfer", 2, False),
    ("Seed agreement", 1, False),
)


def _r6(manifest: Manifest, model: str) -> list[dict[str, Any]]:
    run = manifest.run("r6", model)
    assert run is not None
    return runfiles.rows(run, "table.json")


def _mc1(
    rows: list[dict[str, Any]], method: str, source: str = ""
) -> dict[str, Any]:
    hit = [
        r
        for r in rows
        if r["method"] == method
        and (
            method != "lat"
            or (r["layer_rule"] == "validation-layer" and r["source"] == source)
        )
    ]
    return hit[0]


def _transfer_random(manifest: Manifest, model: str, layer: int) -> float:
    run = manifest.run("r5", model)
    assert run is not None
    series = runfiles.data(run, "curves_transfer.json")["series"]
    mean = runfiles.arrays(run, "curves_transfer.npz")["mean"]
    return float(np.asarray(mean)[series.index("random")][layer - 1])


def _template_sentence(accuracy: list[float], layers: dict[str, int]) -> str:
    by_layer: dict[int, list[str]] = collections.defaultdict(list)
    for model, layer in layers.items():
        by_layer[layer].append(model)
    parts = []
    for layer in sorted(by_layer, key=lambda k: -len(by_layer[k])):
        names = [paper.display_name(m) for m in by_layer[layer]]
        who = (
            f"{len(names)} checkpoints"
            if len(names) > 3
            else " and ".join(names)
        )
        parts.append(f"at layer {layer} on {who}")
    low, high = min(accuracy), max(accuracy)
    value = (
        _cells.num(low)
        if low == high
        else f"{_cells.num(low)} to {_cells.num(high)}"
    )
    return (
        "Honesty in template is the accuracy on held-out persona prompts, "
        f"{value} {' and '.join(parts)} because the members of a pair "
        "differ in the persona instruction; it is not a held-out measure of "
        "honesty."
    )


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.

    Raises:
        ValueError: If two runs store different text-only rows.
    """
    models = manifest.models("r6")
    rows: list[list[str]] = []
    in_acc: list[float] = []
    in_layer: dict[str, int] = {}
    persona: list[float] = []
    text: dict[str, dict[str, Any]] = {}
    n_test = n_val = 0
    for model in models:
        table = _r6(manifest, model)
        for name in ("text_word", "text_char"):
            row = _mc1(table, name)
            if name in text and text[name]["mc1"] != row["mc1"]:
                raise ValueError(f"the {name} row differs between models")
            text[name] = row
        run5, run_a1 = manifest.run("r5", model), manifest.run("a1", model)
        assert run5 is not None
        assert run_a1 is not None
        s5 = runfiles.summary(run5)
        block = s5["in_template"]
        in_acc.append(float(block["test_accuracy"]))
        in_layer[model] = int(block["best_layer"])
        n_test = int(s5["n_eval_pairs"]["transfer"]["test"])
        n_val = int(s5["n_eval_pairs"]["transfer"]["val"])
        mc1 = [_mc1(table, "lat", s)["mc1"] for s in READERS]
        persona.append(float(mc1[2]))
        rows.append(
            [
                paper.display_name(model),
                *(_cells.num(v) for v in mc1),
                _cells.num(_mc1(table, "standard")["mc1"]),
                _cells.num(block["test_accuracy"]),
                str(block["best_layer"]),
                _cells.num(s5["test_accuracy"]),
                _cells.num(
                    _transfer_random(manifest, model, int(s5["best_layer"]))
                ),
                _cells.num(
                    runfiles.summary(run_a1)["agreement_at_layer"]["r5"]
                ),
            ]
        )
    run6 = manifest.run("r6", models[0])
    assert run6 is not None
    n_questions = int(runfiles.summary(run6)["n_test"])
    chance = float(runfiles.data(run6, "curves_truthfulqa.json")["chance"])
    lo, hi = stats.wilson(0.5, n_questions)
    half = (hi - lo) / 2
    word, char = text["text_word"], text["text_char"]
    claim = (
        "Truthfulness and honesty readings on every checkpoint, with the "
        "baselines of the same questions."
    )
    detail = (
        f"The four leftmost columns are TruthfulQA MC1 on {n_questions} test "
        "questions for readers fitted on TruthfulQA, on ARC and on the "
        "paper's honesty persona prompts, each at the layer chosen on "
        "validation questions, and for the model's own answer "
        f"log-probability; a 95 percent Wilson interval at this size is at "
        f"most {chr(0x00B1)}{_cells.num(half)} wide. A classifier on the "
        f"question and answer words reaches "
        f"{_cells.interval(word['mc1'], word['lo'], word['hi'])}, one on the "
        "characters "
        f"{_cells.interval(char['mc1'], char['lo'], char['hi'])}, and chance "
        f"is {_cells.num(chance)}. "
        + _template_sentence(in_acc, in_layer)
        + " Honesty transfer scores the honesty reader on plain true and "
        f"false statements ({n_test} test pairs, layer chosen on {n_val} "
        "validation pairs) next to the random direction at the same layer. "
        "Seed agreement is the mean absolute cosine between the honesty "
        "directions of different seeds at relative depth 0.5, not at the "
        "transfer layer."
    )
    numbers = {
        "n_questions": float(n_questions),
        "wilson_half": half,
        "persona_min": min(persona),
        "persona_max": max(persona),
        "in_template_layer1": float(sum(v == 1 for v in in_layer.values())),
        "text_word": float(word["mc1"]),
        "text_char": float(char["mc1"]),
        "chance": chance,
    }
    return TableOutput(
        claim,
        detail,
        COLUMNS,
        rows,
        _cells.check_align("l" + "r" * 9, len(COLUMNS)),
        [],
        numbers,
        groups=GROUPS,
        row_groups=_rows.role_groups(models),
    )
