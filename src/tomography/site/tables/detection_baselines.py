"""Appendix table: The knowledge direction beside published error detectors.

One row per checkpoint: the within-relation AUROC for telling correct from
wrong greedy answers on held-out PopQA questions, for the knowledge
direction, six published baselines and two controls. The data file adds the
six signals of the runs that the table leaves out.
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
BASELINES = {
    "answer_probe": "Answer probe",
    "probe": "Prompt probe",
    "semantic_entropy": "Semantic entropy",
    "eigenscore": "EigenScore",
    "p_true": "P(True)",
    "first_token_prob": "Token probability",
}
WITH_INTERVAL = ("answer_probe", "semantic_entropy", "eigenscore")
CONTROLS = {"popularity": "Popularity", "length": "Length"}
LEFT_OUT = {
    "neg_entropy": "Entropy of the first token",
    "answer_logprob": "Answer log-probability",
    "mean_token_logprob": "Mean token log-probability",
    "consistency": "Agreement of samples",
    "template": "Relation accuracy",
    "name_logprob": "Name log-probability",
}
GROUPS = (
    ("Reader", 1, False),
    ("Published baselines", len(BASELINES), False),
    ("Controls", len(CONTROLS), False),
)
COLUMNS = [
    "Model",
    "Direction",
    *BASELINES.values(),
    *CONTROLS.values(),
]


def _within(row: dict[str, Any]) -> str:
    return cells.interval(
        row["auroc_within"], row["lo_within"], row["hi_within"]
    )


def _cell(row: dict[str, Any], name: str) -> str:
    if name in WITH_INTERVAL:
        return _within(row)
    return cells.num(row["auroc_within"])


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.

    Raises:
        ValueError: If a run scores signals other than the shown ones and
            those of `LEFT_OUT`.
    """
    models = manifest.models(EXPERIMENT)
    rows: list[list[str]] = []
    left_out: list[tuple[str, ...]] = []
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
                *(_cell(by[name], name) for name in (*BASELINES, *CONTROLS)),
            ]
        )
        if set(by) - {READER, *BASELINES, *CONTROLS} != set(LEFT_OUT):
            raise ValueError(f"{EXPERIMENT}/{model} scores other signals")
        left_out.append(
            tuple(cells.num(by[name]["auroc_within"]) for name in LEFT_OUT)
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
        "By point estimate, the answer probe tells correct from wrong greedy "
        "answers on held-out PopQA questions better than the knowledge "
        "direction on "
        f"{stats.per_role_text(counts['answer_probe_above'])}, and the "
        "direction does better than P(True) and both controls on "
        f"{stats.per_role_text(counts['above_ptrue_and_controls'])}."
    )
    detail = (
        "The table sets the knowledge direction beside six published "
        "baselines and two controls. Every value is the AUROC of correct "
        "against wrong answers computed inside each PopQA relation and "
        "pooled over the relations by their number of pairs of a correct "
        f"and a wrong answer, on {asked:,} test questions per checkpoint. "
        "Every score is oriented so that a higher score means a more likely "
        "correct answer; semantic entropy and EigenScore, which rise with "
        "errors, are negated. The direction, the answer probe, semantic "
        "entropy and EigenScore carry the percentile bootstrap interval "
        "over those questions, resampled inside each relation. The answer "
        "probe is a logistic-regression probe at the last token of the "
        "greedy answer [[cite:orgad2025llms]], at the layer chosen on "
        "validation questions; where its reference reads the tokens of the "
        "exact answer inside a longer response, it reads the residual "
        "stream at the last token of the short greedy answer, with no step "
        "that locates the exact answer. The prompt probe is the same probe "
        "at the last token of the question [[cite:alain2016understanding]]; "
        "both are fitted on the correctness of training questions, which "
        "the direction never sees. Semantic entropy "
        "[[cite:farquhar2024detecting]], in its discrete form, and "
        "EigenScore [[cite:chen2024inside]] fit nothing and read the five "
        "answers sampled per question, where their papers sample 10. "
        "P(True) [[cite:kadavath2022language]] is the model grading its "
        "own answer, and the token probability is that of the first token "
        "of the answer. The popularity of the subject "
        "[[cite:mallen2023when]] and the length of the question in tokens "
        "are controls. GPT-2 small has too few known questions for a "
        "contrast, so it has no direction. The runs also score six signals "
        "that the table leaves out and the data file holds: the entropy of "
        "the first token of the answer, the log-probability of the greedy "
        "answer and its mean per token, the share of sampled answers equal "
        "to the greedy one, the training accuracy of the relation, which is "
        "constant inside a relation, and the log-probability of the name of "
        "the subject. The runs cover "
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
        data_columns=tuple(LEFT_OUT.values()),
        data_cells=tuple(left_out),
        roomy=True,
    )
