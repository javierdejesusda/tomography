"""Appendix table: The design of each experiment the main text uses.

A text table with one row per experiment under a heading row per study: its
stimuli and labels, splits, controls, published baselines and checkpoints.
The text is static, but every count inside it is read from the stored
configuration or summary of the runs, and the checkpoints are counted per
role from the runs that exist. The question and the metric of each
experiment, and the experiments that have no figure or table on the page,
are in the methodology document.
"""

from collections.abc import Mapping
import dataclasses
import json
import re
from typing import Any

from tomography.site import params
from tomography.site import runfiles
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput

COLUMNS = (
    "Experiment",
    "Stimuli and labels",
    "Splits",
    "Controls",
    "Published baselines",
    "Checkpoints",
)
SLOT = re.compile(r"\{([a-z_]+)\.([a-z_.]+)\}")
STANDARD = "Standard."
NONE = "None."
ABOVE = "As above."
SWEEP_CONTROLS = "Random direction, shuffled labels."
PROBE = (
    "None; the reader is itself a logistic-regression probe, made the "
    "default finder after pilot runs (post hoc)."
)
REPORTED = params.REPORT_PAIRS["sentiment"]
DRAWS = ("n_seeds", "n_pairs")

READING = "Reading"
FINDERS = "Finders"
TRUTH = "Truth"
KNOWLEDGE = "Knowledge"
STEERING = "Steering"
POST = "Post-training"

RUNS = {
    "sentiment": "r1",
    "training_size": "p3",
    "utility": "r2",
    "morality": "r3",
    "truth": "r7",
    "honesty": "r5",
    "wording": "p1",
    "pairing": "p4",
    "published_finders": "b2",
    "truthfulqa": "r6",
    "labels": "p9a",
    "knowledge_reading": "p9b",
    "error_prediction": "p9c",
    "error_detectors": "b1",
    "knowledge_steering": "p9d",
    "knowledge_views": "v1",
    "approximate_answers": "p10a",
    "approximate_reading": "p10b",
    "sentiment_steering": "c1",
    "finders_steering": "m1",
    "steering_baselines": "b3",
    "post_training": "p8",
}
PAIRED = "post_training"


@dataclasses.dataclass(frozen=True)
class Row:
    """The design of one experiment.

    A `{key.path}` inside a text is replaced by the count stored at that
    dotted path of the configuration, or else of the summary, of the runs of
    the experiment `key` of `RUNS`.

    Attributes:
        key: Name of the experiment in the numbers and counts of the table.
        study: Study the experiment belongs to.
        name: Name of the experiment on the page.
        stimuli: What the model is given and where the labels come from.
        splits: Which items fit, choose and score.
        controls: What uses no direction fitted for the concept.
        baselines: Published methods run on the same items.
        scope: Clause after the count of checkpoints.
        draws: Experiment whose number of readers and of training pairs
            this one shares, where its splits rely on them without printing
            them.
        also: Other stored counts of the experiment that the page quotes.
    """

    key: str
    study: str
    name: str
    stimuli: str
    splits: str
    controls: str
    baselines: str
    scope: str = ""
    draws: str = ""
    also: tuple[str, ...] = ()

    def texts(self) -> tuple[str, ...]:
        """Lists the static texts of the row, in the order of the columns."""
        return (self.stimuli, self.splits, self.controls, self.baselines)


CLAIM = "The design of each experiment the main text uses."
DETAIL = (
    "Standard splits are {utility.n_seeds} readers, each on its own draw of "
    "{utility.n_pairs} training pairs, with the layer chosen on validation "
    "items and the score on test items; a Splits cell gives the sizes and "
    "what differs. Standard controls are a random direction oriented on the "
    "training labels, the reader scored against shuffled labels, and the "
    "words alone. A control uses no direction fitted for the concept; a "
    "baseline is a published method run on the same items. Select pairs "
    "choose the layer, and report pairs, which that choice does not use, "
    "give the score. Post hoc marks a choice made after results were seen. "
    "Every count is read from the stored configuration or summary of the "
    "runs and is the same on every checkpoint of the experiment. Checkpoints "
    "are counted per role in the order GPT-2, Qwen3 base, Qwen3 "
    "post-trained; All stands for "
)

ROWS = (
    Row(
        "sentiment",
        READING,
        "Sentiment",
        "SST-2 phrases for training, validation sentences for scoring.",
        "Main run: the {sentiment.n_eval_pairs} validation pairs choose the "
        "layer and give the score (optimistic). Reported run: "
        "{training_size.n_seeds} draws; half of the validation sentences "
        f"select, the other half ({REPORTED} pairs) reports.",
        STANDARD,
        PROBE,
        draws="utility",
        also=("min_words",),
    ),
    Row(
        "utility",
        READING,
        "Utility",
        "ETHICS utilitarianism rows, which list the more pleasant scenario "
        "before the other.",
        "{utility.n_eval_pairs.val} validation and "
        "{utility.n_eval_pairs.test} test pairs.",
        STANDARD,
        ABOVE,
    ),
    Row(
        "morality",
        READING,
        "Morality",
        "Short ETHICS commonsense scenarios; the acceptable one is positive.",
        "{morality.n_eval_pairs.val} validation and "
        "{morality.n_eval_pairs.test} test pairs.",
        STANDARD,
        ABOVE,
        draws="utility",
    ),
    Row(
        "truth",
        READING,
        "Truth",
        "True and false statements; nine Geometry of Truth sets with no refit.",
        "{truth.sets.test.n_eval_pairs} test pairs.",
        STANDARD,
        ABOVE,
        draws="utility",
    ),
    Row(
        "honesty",
        READING,
        "Honesty transfer",
        "True statements under an honest and an untruthful persona, read at "
        "the response tokens; tested on plain statements.",
        "{honesty.n_eval_pairs.transfer.val} validation and "
        "{honesty.n_eval_pairs.transfer.test} test pairs.",
        STANDARD,
        ABOVE,
        draws="utility",
    ),
    Row(
        "wording",
        READING,
        "Prompt wording",
        "Sentiment, truth and emotion pairs in variants of the template; "
        "same items and draws.",
        STANDARD,
        SWEEP_CONTROLS,
        ABOVE,
        draws="utility",
    ),
    Row(
        "pairing",
        FINDERS,
        "Finder comparison",
        "Sentiment, truth and utility pairs; pairing, preprocessing and "
        "finder vary one at a time.",
        "Each series, the random direction included, has the layer of its "
        "own select split; scores on report pairs.",
        SWEEP_CONTROLS,
        "Logistic regression, mean difference (mass-mean probe).",
        draws="utility",
    ),
    Row(
        "published_finders",
        FINDERS,
        "Published finders",
        "Three Geometry of Truth sets, each statement paired with its "
        "negation; SST-2 texts.",
        STANDARD,
        STANDARD,
        "Mass-mean and whitened mass-mean probes, logistic-regression probe, "
        "contrast-consistent search.",
        draws="utility",
        also=("ccs_restarts",),
    ),
    Row(
        "truthfulqa",
        TRUTH,
        "TruthfulQA",
        "{truthfulqa.n_questions} multiple-choice questions; readers fitted "
        "on train questions, ARC items or persona statements.",
        "{truthfulqa.n_seeds} readers; {truthfulqa.n_train} train, "
        "{truthfulqa.n_val} validation, {truthfulqa.n_test} test questions.",
        STANDARD,
        "The two zero-shot scorers of the paper.",
    ),
    Row(
        "labels",
        KNOWLEDGE,
        "Labels",
        "Closed-book questions from seven sources. The PopQA part of the "
        "pool was widened from four to twelve relations after the guess "
        "filter of earlier labelling runs left too few known questions "
        "(post hoc). Known: the greedy answer and at least "
        "{labels.recipe.known_min_correct} of {labels.n_samples} sampled "
        "answers are correct; unknown: all are wrong, or no gold answer "
        "exists; the rest is left out.",
        "Hashed by subject or source.",
        NONE,
        NONE,
        also=("min_known", "n_shots"),
    ),
    Row(
        "knowledge_reading",
        KNOWLEDGE,
        "Knowledge reading",
        "Known against unknown questions of the same checkpoint.",
        "{knowledge_reading.n_seeds} draws of {knowledge_reading.n_pairs} "
        "training pairs.",
        "Random direction, the words alone, popularity.",
        "The finder of the paper, mean difference.",
        scope="too few known questions on GPT-2 small",
    ),
    Row(
        "error_prediction",
        KNOWLEDGE,
        "Error prediction",
        "PopQA, TriviaQA and NQ-open questions; labelled by the correctness "
        "of the greedy answer.",
        ABOVE,
        "Popularity, relation accuracy, question length, likelihood of the "
        "subject's name.",
        "The model's confidence.",
        scope="no direction on GPT-2 small",
        draws="knowledge_reading",
    ),
    Row(
        "error_detectors",
        KNOWLEDGE,
        "Published detectors",
        "The same questions and answers.",
        ABOVE,
        "Popularity, question length.",
        "Answer-token probe, semantic entropy, EigenScore, P(True).",
        scope="no direction on GPT-2 small",
    ),
    Row(
        "knowledge_steering",
        KNOWLEDGE,
        "Knowledge steering",
        "Known and unknown test questions over a grid of relative "
        "coefficients.",
        "Direction from train and validation questions.",
        "{knowledge_steering.n_random} random directions of the same norm.",
        NONE,
    ),
    Row(
        "knowledge_views",
        KNOWLEDGE,
        "Activation views",
        "The questions of knowledge reading, at every layer it read.",
        "As knowledge reading.",
        "Random direction, labels permuted inside each group, the text "
        "alone, popularity.",
        NONE,
    ),
    Row(
        "approximate_answers",
        KNOWLEDGE,
        "Approximate answers",
        "Wikidata quantity questions, {approximate_answers.n_per_side} per "
        "side and family, and people with one exact height; range or exact "
        "by the Wikidata bounds.",
        "One greedy and {approximate_answers.n_samples} sampled answers.",
        "The people.",
        NONE,
    ),
    Row(
        "approximate_reading",
        KNOWLEDGE,
        "Range reader",
        "The same questions.",
        "Hashed by family; {approximate_reading.n_seeds} draws; also scored "
        "on a held-out family.",
        "Random direction, the words alone, question length, popularity, "
        "{approximate_reading.n_people} people.",
        "As knowledge reading.",
    ),
    Row(
        "sentiment_steering",
        STEERING,
        "Sentiment steering",
        "Completions of {sentiment_steering.n_prompts} neutral prompts, "
        "{sentiment_steering.num_samples} samples each, scored by a "
        "sentiment classifier.",
        "One list of prompts chooses the layers, a second gives the results.",
        "No random direction: the unsteered model and a prompt instruction.",
        NONE,
        also=("n_neutral", "n_steer_seeds"),
    ),
    Row(
        "finders_steering",
        STEERING,
        "Utility test",
        "{finders_steering.n_prompts} truncated ETHICS scenarios continued "
        "under steering; the sentiment classifier judges.",
        "Paired by prompt and seed.",
        "A random finder; {finders_steering.n_steer_seeds} steering vectors "
        "per finder.",
        "Logistic regression, mean difference, among five finders.",
    ),
    Row(
        "steering_baselines",
        STEERING,
        "Published steering methods",
        "WikiText-103 openings (base) or AlpacaEval instructions "
        "(post-trained); an LLM judge rates concept, fluency and, on "
        "post-trained checkpoints, how closely the topic relates to the "
        "instruction.",
        "Coefficient on {steering_baselines.n_tune} tuning prompts, score on "
        "{steering_baselines.n_held_out} held-out prompts, per concept.",
        "Unsteered model, random unit direction.",
        "The five methods of [[table:steering_baselines]].",
        scope="GPT-2 XL is steered as a base checkpoint",
        also=("max_new_tokens", "n_neutral", "n_pairs"),
    ),
    Row(
        "post_training",
        POST,
        "Base against post-trained",
        "The finished readers, wording run and sentiment steering run of "
        "each pair.",
        "Paired by draw; bootstrap of {post_training.n_boot} resamples over "
        "the draws.",
        "Those of the runs read.",
        NONE,
    ),
)


def agreed(name: str, values: Mapping[str, float | None]) -> float:
    """Returns the one value that every checkpoint stores for a count.

    Args:
        name: What the count is, for the error.
        values: Checkpoint to its stored value, or None where it is absent.

    Returns:
        The shared value.

    Raises:
        ValueError: If a checkpoint lacks the value or two of them differ.
    """
    found = set(values.values())
    if len(found) != 1 or None in found:
        raise ValueError(f"{name} is not one stored value: {dict(values)}")
    (value,) = found
    assert value is not None
    return value


def _stored(document: Any, path: str) -> float | None:
    """Reads the number at a dotted path of a stored document."""
    for part in path.split("."):
        if not isinstance(document, dict) or part not in document:
            return None
        document = document[part]
    if isinstance(document, bool) or not isinstance(document, (int, float)):
        return None
    return document


def _wanted() -> dict[str, set[str]]:
    """Lists the stored paths that the texts ask of each experiment."""
    wanted: dict[str, set[str]] = {key: set() for key in RUNS}
    for text in (DETAIL, *(t for row in ROWS for t in row.texts())):
        for key, path in SLOT.findall(text):
            wanted[key].add(path)
    for row in ROWS:
        wanted[row.key].update(row.also)
        if row.draws:
            wanted[row.key].update(DRAWS)
            wanted[row.draws].update(DRAWS)
    return wanted


def _facts(manifest: Manifest) -> dict[str, float]:
    """Reads the counts that the texts ask for, named `key_path`."""
    facts: dict[str, float] = {}
    for key, paths in _wanted().items():
        code = RUNS[key]
        found: dict[str, dict[str, float | None]] = {p: {} for p in paths}
        for model in manifest.models(code) if paths else ():
            run = manifest.run(code, model)
            assert run is not None
            config = json.loads(
                (run / "config.json").read_text(encoding="utf-8")
            )
            summary = runfiles.summary(run)
            for path in paths:
                value = _stored(config, path)
                if value is None:
                    value = _stored(summary, path)
                found[path][model] = value
        for path in sorted(paths):
            facts[f"{key}_{path.replace('.', '_')}"] = agreed(
                f"{code} {path}", found[path]
            )
    for row in ROWS:
        for path in DRAWS if row.draws else ():
            agreed(
                f"{row.key} {path}",
                {
                    row.key: facts[f"{row.key}_{path}"],
                    row.draws: facts[f"{row.draws}_{path}"],
                },
            )
    return facts


def _filled(text: str, facts: Mapping[str, float]) -> str:
    """Writes the stored counts into the slots of a text."""

    def count(slot: re.Match[str]) -> str:
        name = f"{slot[1]}_{slot[2].replace('.', '_')}"
        return f"{int(facts[name]):,}"

    return SLOT.sub(count, text)


def _members(manifest: Manifest, key: str) -> set[str]:
    """Finds the checkpoints an experiment covers."""
    code = RUNS[key]
    models = manifest.models(code)
    if key != PAIRED:
        return set(models)
    run = manifest.run(code, models[0])
    assert run is not None
    return {m for pair in runfiles.summary(run)["pairs"] for m in pair}


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the static rows and the committed runs.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    facts = _facts(manifest)
    every = manifest.checkpoints()
    counts = {
        key: stats.per_role({m: m in _members(manifest, key) for m in every})
        for key in RUNS
    }
    totals = stats.per_role(dict.fromkeys(every, True))
    full = ", ".join(
        f"{totals[r].total}/{totals[r].total}" for r in stats.ROLES
    )
    rows: list[list[str]] = []
    headings: list[tuple[int, str]] = []
    for row in ROWS:
        if not headings or headings[-1][1] != row.study:
            headings.append((len(rows), row.study))
        covered = stats.per_role_compact(counts[row.key])
        checkpoints = f"{covered}; {row.scope}." if row.scope else f"{covered}."
        rows.append(
            [
                row.name,
                *(_filled(text, facts) for text in row.texts()),
                checkpoints,
            ]
        )
    numbers = {
        **facts,
        "experiments": float(len(ROWS)),
        "studies": float(len(headings)),
        "rows": float(len(rows)),
    }
    return TableOutput(
        CLAIM,
        f"{_filled(DETAIL, facts)}{full}.",
        list(COLUMNS),
        rows,
        "l" * len(COLUMNS),
        [],
        numbers,
        row_groups=(),
        counts=counts,
        compact_print=True,
        data_columns=("Study",),
        data_cells=tuple((row.study,) for row in ROWS),
        headings=tuple(headings),
    )
