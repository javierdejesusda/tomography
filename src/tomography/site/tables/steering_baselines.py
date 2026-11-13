"""Appendix table: The reader beside published steering methods.

One block per checkpoint and concept, under a heading row: the held-out
judge scores, perplexity and distinct-1 of the reader, five published
steering methods and two controls, each arm at the coefficient it chose on
the tuning prompts. The roles are never pooled. The data file adds the
checkpoint, the concept and the mean score of the pinned classifier.
"""

import json
from typing import Any

from tomography.site import cells
from tomography.site import runfiles
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.viz import paper

EXPERIMENT = "b3"
NO_BREAK = chr(0x2011)
BASELINE = "Published baselines"
CONTROL = "Controls"
ARMS = {
    "reader": ("Reader", "Logistic regression"),
    "prompt": (BASELINE, "Prompt prepending"),
    "diffmean": (BASELINE, "DiffMean"),
    "caa": (BASELINE, "CAA"),
    "actadd": (BASELINE, "ActAdd"),
    "mean_centring": (BASELINE, f"Mean{NO_BREAK}centring"),
    "none": (CONTROL, "Unsteered"),
    "random/0": (CONTROL, "Random direction"),
}
ROLES = {"base": "base", "post_trained": "post-trained"}
SIZES = {
    "tune": "n_tune",
    "held_out": "n_held_out",
    "new_tokens": "max_new_tokens",
    "train_pairs": "n_pairs",
    "neutral": "n_neutral",
    "resamples": "n_boot",
}
COLUMNS = [
    "Kind",
    "Arm",
    "Coefficient",
    "Concept",
    "Instruction",
    "Fluency",
    "Overall",
    "Perplexity",
    "Distinct-1",
]


def _blank(value: float | None, digits: int = 3) -> str:
    """Writes a value, or nothing where the arm or the role has none."""
    return "" if value is None else cells.num(value, digits)


def _row(row: dict[str, Any], kind: str, label: str) -> list[str]:
    """Writes the cells of one arm on one concept."""
    return [
        kind,
        label,
        _blank(row["coeff"], 2),
        cells.num(row["concept_score"]),
        _blank(row.get("instruction_score")),
        cells.num(row["fluency_score"]),
        cells.interval(row["overall"], row["overall_lo"], row["overall_hi"]),
        _blank(row["perplexity"], 1),
        cells.num(row["distinct_1"]),
    ]


def _block(
    name: str,
    table: list[dict[str, Any]],
    summary: dict[str, Any],
    concept: str,
) -> list[dict[str, Any]]:
    """Picks the rows of one concept in the order of `ARMS`.

    Args:
        name: Checkpoint key, for the errors.
        table: The held-out rows of the run.
        summary: The summary of the run.
        concept: The concept steered toward.

    Returns:
        One row per arm.

    Raises:
        ValueError: If the concept does not hold every arm exactly once, an
            arm scores another number of prompts, or the instruction score
            is present where the criteria of the run have none, or missing
            where they have it.
    """
    mine = [r for r in table if r["concept"] == concept]
    if sorted(r["arm"] for r in mine) != sorted(ARMS):
        raise ValueError(f"{EXPERIMENT}/{name} holds other arms on {concept}")
    judged = "instruction" in summary["criteria"]
    for row in mine:
        if row["n_prompts"] != summary["n_held_out"]:
            raise ValueError(f"{EXPERIMENT}/{name} scores other prompts")
        if ("instruction_score" in row) != judged:
            raise ValueError(f"{EXPERIMENT}/{name} has other criteria")
    by = {r["arm"]: r for r in mine}
    return [by[arm] for arm in ARMS]


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.

    Raises:
        ValueError: If a run holds other arms, scores other prompts or other
            criteria, or two runs differ in a size the caption quotes.
    """
    models = manifest.models(EXPERIMENT)
    rows: list[list[str]] = []
    headings: list[tuple[int, str]] = []
    extra: list[tuple[str, ...]] = []
    sizes: set[tuple[int, ...]] = set()
    for model in models:
        run = manifest.run(EXPERIMENT, model)
        assert run is not None
        name = paper.display_name(model)
        summary = runfiles.summary(run)
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        sizes.add(tuple(int(config[field]) for field in SIZES.values()))
        table = runfiles.rows(run, "table.json")
        if {r["concept"] for r in table} != set(summary["concepts"]):
            raise ValueError(f"{EXPERIMENT}/{model} holds other concepts")
        for concept, description in summary["concepts"].items():
            headings.append(
                (
                    len(rows),
                    f"{name} ({ROLES[summary['role']]}), steered toward "
                    f"{description}",
                )
            )
            shown = ""
            for row in _block(model, table, summary, concept):
                kind, label = ARMS[row["arm"]]
                rows.append(_row(row, "" if kind == shown else kind, label))
                extra.append((name, concept, cells.num(row["classifier"])))
                shown = kind
    if len(sizes) != 1:
        raise ValueError(f"{EXPERIMENT} runs differ in their sizes")
    n = dict(zip(SIZES, sizes.pop(), strict=True))
    opened = {place for place, _ in headings}
    rules = [i for i, line in enumerate(rows) if line[0] and i not in opened]
    counts = {
        "with_run": stats.per_role(
            {m: m in models for m in manifest.checkpoints()}
        ),
    }
    claim = (
        "Steering toward a concept along the reader, beside five published "
        "steering methods and two controls, scored by an LLM judge on "
        "held-out prompts, on one checkpoint per role."
    )
    detail = (
        "The roles are not pooled. An LLM judge rates each generation 0, 1 "
        "or 2 on the concept and on fluency, and on the post-trained "
        "checkpoint also on following the instruction; the overall score "
        "of a generation is the harmonic mean of its criteria, 0 when any "
        "of them is 0. A seeded share of the generations is rated again in "
        "a second judge slot by the same model. The base checkpoints "
        "continue the opening words of held-out WikiText-103 passages and "
        "the post-trained one answers AlpacaEval instructions "
        "[[cite:dubois2023alpacafarm]] in its chat template, "
        f"{n['new_tokens']} new tokens per prompt. Every value is the mean "
        f"over the {n['held_out']} held-out prompts of the concept, at the "
        "coefficient that the arm chose on the "
        f"{n['tune']} tuning prompts, which share no prompt with them; the "
        "coefficient is relative to the residual norm, and the overall "
        "score carries the 95 percent percentile bootstrap interval over "
        f"the held-out prompts ({n['resamples']:,} resamples), computed "
        "for each arm on its own. Perplexity is that of "
        f"{n['neutral']} held-out WikiText-103 passages under the steered "
        "model, and distinct-1 the share of distinct tokens in the "
        "generations. The published baselines are prompt prepending and "
        "DiffMean, the difference of class means [[cite:marks2024geometry]], "
        "both baselines of AxBench [[cite:wu2025axbench]], contrastive "
        "activation addition (CAA) [[cite:rimsky2024steering]], activation "
        "addition (ActAdd) [[cite:turner2023steering]] and mean-centring "
        "[[cite:jorgensen2023improving]]; the reader is the "
        "logistic-regression finder, as in the sentiment steering run. CAA, "
        "mean-centring, DiffMean and the reader are built from the same "
        f"{n['train_pairs']} training pairs per concept. The controls are "
        "the unsteered model and a random unit direction steered like the "
        "vector arms. The unsteered model and prompt prepending have no "
        "coefficient, and prompt prepending, which changes the prompt and "
        "not the model, has no perplexity. The runs cover "
        f"{stats.per_role_text(counts['with_run'])}."
    )
    return TableOutput(
        claim,
        detail,
        COLUMNS,
        rows,
        "ll" + "r" * (len(COLUMNS) - 2),
        [],
        {"rows": float(len(rows)), **{k: float(v) for k, v in n.items()}},
        groups=(("", 2, False), ("Judge score", 4, False), ("", 2, False)),
        row_groups=tuple(rules),
        counts=counts,
        compact_print=True,
        data_columns=("Model", "Target concept", "Classifier"),
        data_cells=tuple(extra),
        wide=True,
        roomy=True,
        headings=tuple(headings),
        span_labels=True,
    )
