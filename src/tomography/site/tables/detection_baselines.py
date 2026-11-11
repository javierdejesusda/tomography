"""Appendix table: The knowledge direction beside published error detectors.

One row per checkpoint: the within-relation AUROC for telling correct from
wrong greedy answers on held-out PopQA questions, for the knowledge
direction, six published baselines and two controls.
"""

from typing import Any

from tomography.site import cells
from tomography.site import runfiles
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _rows
from tomography.site.tables import knowledge
from tomography.viz import paper

EXPERIMENT = "b1"
DATASET = "popqa"
READER = "lat"
WITH_INTERVAL = {
    "answer_probe": "Answer probe",
    "semantic_entropy": "Semantic entropy",
    "eigenscore": "EigenScore",
}
PLAIN = {
    "p_true": "P(True)",
    "first_token_prob": "Token probability",
    "probe": "Prompt probe",
}
CONTROLS = {"popularity": "Popularity", "length": "Length"}
GROUPS = (
    ("Reader", 1, False),
    ("Published baselines", len(WITH_INTERVAL) + len(PLAIN), False),
    ("Controls", len(CONTROLS), False),
)
COLUMNS = [
    "Model",
    "Direction",
    *WITH_INTERVAL.values(),
    *PLAIN.values(),
    *CONTROLS.values(),
]


def _within(row: dict[str, Any]) -> str:
    return cells.interval(
        row["auroc_within"], row["lo_within"], row["hi_within"]
    )


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    models = manifest.models(EXPERIMENT)
    rows: list[list[str]] = []
    questions: set[int] = set()
    probe_above: dict[str, bool] = {}
    above_controls: dict[str, bool] = {}
    for model in models:
        run = manifest.run(EXPERIMENT, model)
        assert run is not None
        by = {
            r["predictor"]: r
            for r in runfiles.rows(run, "table.json")
            if r["dataset"] == DATASET
        }
        questions.update(r["n"] for r in by.values())
        ours = by.get(READER)
        rows.append(
            [
                paper.display_name(model),
                _within(ours) if ours else knowledge.NO_CONTRAST,
                *(_within(by[name]) for name in WITH_INTERVAL),
                *(
                    cells.num(by[name]["auroc_within"])
                    for name in (*PLAIN, *CONTROLS)
                ),
            ]
        )
        if ours is None:
            continue
        value = ours["auroc_within"]
        probe_above[model] = by["answer_probe"]["auroc_within"] > value
        above_controls[model] = all(
            value > by[name]["auroc_within"] for name in ("p_true", *CONTROLS)
        )
    (asked,) = questions
    counts = {
        "with_run": stats.per_role(
            {m: m in models for m in manifest.checkpoints()}
        ),
        "answer_probe_above": stats.per_role(probe_above),
        "above_ptrue_and_controls": stats.per_role(above_controls),
    }
    claim = (
        "Within-relation AUROC for telling correct from wrong greedy "
        "answers on held-out PopQA questions: the knowledge direction "
        "beside six published baselines and two controls."
    )
    detail = (
        "Every value is the AUROC of correct against wrong answers computed "
        f"inside each PopQA relation, on {asked:,} test questions per "
        "checkpoint. The direction and three of the published baselines, "
        "those the knowledge runs did not have, carry the percentile "
        "bootstrap interval over those questions, resampled inside each "
        "relation. The answer probe is a "
        "logistic-regression probe at the last token of the greedy answer "
        "[[cite:orgad2025llms]], at the layer chosen on validation "
        "questions, and the prompt probe is the same probe at the last "
        "token of the question [[cite:alain2016understanding]]; both are "
        "fitted on the correctness of training questions, which the "
        "direction never sees. Semantic entropy "
        "[[cite:farquhar2024detecting]], in its discrete form, and "
        "EigenScore [[cite:chen2024inside]] fit nothing and read the five "
        "answers sampled per question, where their papers sample 10. "
        "P(True) [[cite:kadavath2022language]] is the model grading its "
        "own answer, and the token probability is that of the first token "
        "of the answer. The popularity of the subject "
        "[[cite:mallen2023when]] and the length of the question in tokens "
        "are controls. GPT-2 small has too few known questions for a "
        "contrast, so it has no direction. The runs cover "
        f"{stats.per_role_text(counts['with_run'])}; they also score "
        "TriviaQA and NQ-open, which have no relations and are not shown."
    )
    return TableOutput(
        claim,
        detail,
        COLUMNS,
        rows,
        "l" + "r" * (len(COLUMNS) - 1),
        [],
        {"questions": float(asked), "rows": float(len(rows))},
        groups=GROUPS,
        row_groups=_rows.role_groups(models),
        counts=counts,
        compact_print=True,
    )
