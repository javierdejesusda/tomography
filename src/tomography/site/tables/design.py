"""Appendix table: The design of every experiment shown on the page.

A text table in blocks, one per experiment: the study it belongs to, its
question, stimuli, labels, splits, controls, published baselines, metric and
checkpoints. The text is static, but every count inside it is read from the
stored configuration or summary of the runs, and the checkpoints are counted
per role from the runs that exist.
"""

from collections.abc import Mapping
import dataclasses
import json
import re
from typing import Any

from tomography.site import runfiles
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput

ASPECTS = (
    "Study",
    "Question",
    "Stimuli",
    "Labels",
    "Splits",
    "Controls",
    "Published baselines",
    "Metric",
    "Checkpoints",
)
SLOT = re.compile(r"\{([a-z_.]+)\}")
NO_MODEL = "None: no model is loaded."
NONE = "None."
READING_CONTROLS = (
    "A random direction oriented on the training labels, the reader scored "
    "against shuffled labels, and classifiers on the words and on the "
    "characters of the texts."
)
SWEEP_CONTROLS = (
    "A random direction oriented on the training labels, and the reader "
    "scored against shuffled labels."
)
PROBE = "None; the reader is itself a logistic-regression probe."
SAME_SWEEP = (
    "Logistic regression (a linear probe) and the mean difference (the "
    "mass-mean probe), in the same sweep as the finder of the paper."
)
DRAWS = "{n_seeds} readers, each on its own draw of {n_pairs} training pairs; "


@dataclasses.dataclass(frozen=True)
class Card:
    """The design of one experiment.

    A `{path}` inside a text is replaced by the count stored at that dotted
    path of the configuration, or else of the summary, of the runs.

    Attributes:
        key: Name of the experiment in the numbers and counts of the table.
        study: Study the experiment belongs to.
        name: Name of the experiment on the page.
        codes: Result directories the experiment reads.
        question: What the experiment asks.
        stimuli: What the model is given.
        labels: Where the labels come from.
        splits: Which items fit, choose and score.
        controls: What uses no learned signal for the concept.
        baselines: Published methods run on the same items.
        metric: What is reported, with its interval.
        scope: Sentence after the count of checkpoints.
        paired: Whether one stored run names the checkpoints it compares.
        model_free: Whether the experiment loads no model.
    """

    key: str
    study: str
    name: str
    codes: tuple[str, ...]
    question: str
    stimuli: str
    labels: str
    splits: str
    controls: str
    baselines: str
    metric: str
    scope: str = ""
    paired: bool = False
    model_free: bool = False

    def texts(self) -> tuple[str, ...]:
        """Lists the static texts of the card, in the order of the rows."""
        return (
            self.question,
            self.stimuli,
            self.labels,
            self.splits,
            self.controls,
            self.baselines,
            self.metric,
        )


READING = "Reading concepts"
FINDERS = "Finders"
KNOWLEDGE = "Known and unknown questions"
APPROXIMATE = "Approximate answers"
TRUTH = "Truth"
STEERING = "Steering"
POST = "Post-training"

CARDS = (
    Card(
        "sentiment",
        READING,
        "Sentiment",
        ("r1",),
        "Does a direction separate positive from negative phrases?",
        "SST-2 phrases of at least {min_words} words for training and the "
        "full sentences of its validation split for scoring, in the template "
        "of the paper.",
        "The binary sentiment label of SST-2.",
        DRAWS + "the same {n_eval_pairs} validation pairs choose the layer "
        "and give the score, so the value is optimistic.",
        READING_CONTROLS,
        PROBE,
        "Pair accuracy at the chosen layer, as the mean over the readers.",
    ),
    Card(
        "utility",
        READING,
        "Utility",
        ("r2",),
        "Does a direction read which of two scenarios is more pleasant?",
        "ETHICS utilitarianism rows, two scenarios per row, in the template "
        "of the paper.",
        "The order of the source row, which lists the more pleasant scenario "
        "before the other.",
        DRAWS + "the layer is chosen on {n_eval_pairs.val} validation pairs "
        "and the score is on {n_eval_pairs.test} test pairs, the test file "
        "of the source.",
        READING_CONTROLS,
        PROBE,
        "Pair accuracy on the test pairs, as the mean over the readers.",
    ),
    Card(
        "morality",
        READING,
        "Morality",
        ("r3",),
        "Does a direction read whether a scenario is acceptable?",
        "ETHICS commonsense scenarios, the short ones only, in the template "
        "of the paper.",
        "The source label, flipped so that the positive member is the "
        "acceptable scenario.",
        DRAWS + "the layer is chosen on {n_eval_pairs.val} validation pairs "
        "and the score is on {n_eval_pairs.test} test pairs, the test file "
        "of the source.",
        READING_CONTROLS,
        PROBE,
        "Pair accuracy on the test pairs, as the mean over the readers.",
    ),
    Card(
        "text_only",
        READING,
        "Text-only classifiers",
        ("tb1",),
        "How much of the accuracy of a reader do the words and the "
        "characters of its own pairs reach with no model?",
        "The raw training and test texts of the sentiment, utility, "
        "morality, truth and honesty readers and of the training-set size "
        "run.",
        "Those of each reader.",
        "One fit per training set of the reader, scored on the test pairs "
        "of that reader.",
        "This run is the text-only control of those readers.",
        NONE,
        "Pair accuracy, ties sharing the credit, with a Wilson interval "
        "over the test pairs.",
        model_free=True,
    ),
    Card(
        "training_size",
        READING,
        "Training-set size",
        ("p3",),
        "How many training pairs does a direction need, and how much can "
        "one draw mislead?",
        "Sentiment, truth and utility pairs, with truth also read on the "
        "cities and the common-claim statements.",
        "Those of the source datasets.",
        "{n_seeds} resampled training sets per size; for sentiment the "
        "SST-2 validation sentences are halved by a seeded draw, one half "
        "chooses the layer and the other gives the score.",
        READING_CONTROLS,
        PROBE,
        "Pair accuracy on the report pairs at each training size, with a "
        "bootstrap interval over the resamples.",
    ),
    Card(
        "token_position",
        READING,
        "Token position and a misleading sentence",
        ("p2",),
        "Where in the prompt is a concept read, and what does a misleading "
        "sentence in the prompt cost?",
        "Sentiment, truth and utility pairs in the template of the paper, "
        "read at the last token, at the concept tokens and as the mean over "
        "tokens, with and without an added sentence that misleads about the "
        "concept.",
        "Those of the source datasets.",
        DRAWS + "the layer is chosen on pairs that the score does not use.",
        SWEEP_CONTROLS,
        "The zero-shot heuristic of the paper, which asks the model for the "
        "level in words.",
        "Pair accuracy per token position and per kind of prompt.",
    ),
    Card(
        "wording",
        READING,
        "Prompt wording",
        ("p1",),
        "How much does the wording of the template move a reading?",
        "Sentiment, truth and emotion pairs, each in variants of the "
        "template of the paper with the same items and the same draws.",
        "Those of the source datasets.",
        DRAWS + "the layer is chosen on pairs that the score does not use.",
        SWEEP_CONTROLS,
        PROBE,
        "Pair accuracy per variant and its spread across the variants.",
    ),
    Card(
        "pairing",
        FINDERS,
        "Pairing, preprocessing and finder",
        ("p4",),
        "Do the pairing, the preprocessing of the differences and the "
        "finder change what is read?",
        "Sentiment, truth and utility pairs; the pairing, the preprocessing "
        "and the finder are varied one at a time around the finder of the "
        "paper.",
        "Those of the source datasets.",
        DRAWS + "every series, the random direction included, is read at "
        "the layer chosen on its own select split and scored on report "
        "pairs.",
        "A random direction with a layer of its own, and the reader scored "
        "against shuffled labels.",
        SAME_SWEEP,
        "Pair accuracy, and its margin over the random direction.",
    ),
    Card(
        "finders_steering",
        FINDERS,
        "Six finders on utility",
        ("m1",),
        "Do six finders that read utility also steer it?",
        "ETHICS utility pairs for reading; {n_prompts} truncated scenarios "
        "that the model continues under steering.",
        "The order of the source row for reading; a sentiment classifier, "
        "with a second one as a check, for the continuations.",
        "Reading is scored on held-out pairs; for the same prompt and seed, "
        "the continuation steered at a positive coefficient is compared "
        "with the one steered at the negative coefficient.",
        "A random finder, read and steered like the others, with "
        "{n_steer_seeds} steering vectors per finder.",
        "Logistic regression (a linear probe) and the mean difference (the "
        "mass-mean probe), among the six finders.",
        "Pair accuracy; the win rate of the positive continuation on "
        "decisive pairs, with a Wilson interval.",
    ),
    Card(
        "published_finders",
        FINDERS,
        "Published finders",
        ("b2",),
        "Does the finder of the paper read truth and sentiment as well as "
        "published finders fitted on the same pairs?",
        "Statements of three Geometry of Truth sets, each paired with its "
        "negation, and SST-2 texts.",
        "The source labels; a pair holds exactly one true member.",
        "{n_seeds} training sets of {n_pairs} pairs; each series chooses "
        "its layer on the validation pairs and is scored on the test pairs.",
        "A random direction, the finder of the paper scored against labels "
        "shuffled inside each pair, and classifiers on the words and on the "
        "characters of the texts.",
        "Mass-mean and whitened mass-mean probes, a logistic-regression "
        "probe and CCS with the best of {ccs_restarts} restarts.",
        "Test accuracy, with a Wilson interval over the test pairs.",
    ),
    Card(
        "labels",
        KNOWLEDGE,
        "Labels from the behaviour of the model",
        ("p9a",),
        "Which questions does each model know?",
        "Closed-book questions in nine groups, built from PopQA, TriviaQA, "
        "KUQ, FalseQA, SyntHal, NQ-open and Wikidata, asked with a few-shot "
        "prompt of {n_shots} examples.",
        "Known when the greedy answer and at least "
        "{recipe.known_min_correct} of {n_samples} sampled answers contain "
        "a gold alias; unknown when the greedy answer and every sample are "
        "wrong; the rest is uncertain and left out. Questions about "
        "fictional subjects are unknown by construction.",
        "Questions are hashed into train, validation and test by subject or "
        "source, so one subject stays in one split.",
        "A group of questions that a constant guess answers often supplies "
        "no known question, and a model enters the known against unknown "
        "contrasts only with at least {min_known} known questions.",
        NONE,
        "The number of known, unknown and uncertain questions per group.",
    ),
    Card(
        "knowledge_reading",
        KNOWLEDGE,
        "Reading known against unknown",
        ("p9b",),
        "Does a direction separate questions a model answers from questions "
        "it cannot answer?",
        "The labelled questions in the few-shot prompt, as contrasts of a "
        "known group against an unknown group; one contrast keeps a single "
        "dataset on both sides.",
        "Those of the labelling run of the same checkpoint.",
        "{n_seeds} draws of {n_pairs} training pairs; the layer is chosen "
        "on the validation questions and the score is on the test questions.",
        "A random direction, classifiers on the words and on the characters "
        "of the question, and the popularity of the subject, each on the "
        "test questions of the reader.",
        SAME_SWEEP,
        "AUROC, inside each relation where the contrast holds PopQA "
        "questions; its stored interval resamples the draws, not the test "
        "questions.",
    ),
    Card(
        "knowledge_views",
        KNOWLEDGE,
        "Hidden states behind the readers",
        ("v1",),
        "What do the hidden states behind the knowledge readers look like, "
        "and do the stored scores reproduce?",
        "The train and test questions of the reading run, read again at "
        "the layer of that run.",
        "Those of the labelling run of the same checkpoint.",
        "Those of the reading run; nothing is chosen again.",
        "A random direction, readers refitted on labels permuted inside "
        "each group, a view of the text alone and the popularity of the "
        "subject.",
        "None of its own; it recomputes the readers of the reading run.",
        "The AUROC of each recomputed reader beside the stored one, and "
        "projections of the hidden states on two axes.",
    ),
    Card(
        "error_prediction",
        KNOWLEDGE,
        "Predicting wrong answers",
        ("p9c",),
        "Does the knowledge direction predict a wrong answer beyond the "
        "confidence of the model?",
        "PopQA, TriviaQA and NQ-open questions with the greedy answer of "
        "the model.",
        "Whether the greedy answer contains a gold alias.",
        "The direction is fitted on {n_seeds} draws of {n_pairs} training "
        "pairs; probes and layers are chosen on train and validation "
        "questions and every score is on the test questions.",
        "The popularity of the subject, the accuracy of the relation, the "
        "length of the question and the likelihood of the name of the "
        "subject.",
        "The confidence of the model: the probability of the most likely "
        "next token, the negative entropy of that distribution, the mean "
        "log-probability of the answer tokens, the share of sampled answers "
        "equal to the greedy answer, and P(True).",
        "AUROC of wrong against correct answers, inside each relation on "
        "PopQA, with a bootstrap interval over the test questions.",
    ),
    Card(
        "error_detectors",
        KNOWLEDGE,
        "Published error detectors",
        ("b1",),
        "Do published error detectors do better than the knowledge "
        "direction on the same test questions?",
        "The questions and the greedy and sampled answers of the "
        "error-prediction run.",
        "Whether the greedy answer contains a gold alias.",
        "Those of the error-prediction run; the probe at the answer token "
        "chooses its layer on validation questions.",
        "The popularity of the subject and the length of the question.",
        "A probe at the last token of the answer, semantic entropy and "
        "EigenScore, beside P(True) and the other signals of the "
        "error-prediction run.",
        "AUROC of wrong against correct answers, inside each relation on "
        "PopQA, with a bootstrap interval over the test questions.",
    ),
    Card(
        "knowledge_steering",
        KNOWLEDGE,
        "Steering along the knowledge direction",
        ("p9d",),
        "Does adding the knowledge direction change how a model answers?",
        "Known and unknown test questions, answered while the direction is "
        "added over a grid of relative coefficients.",
        "The outcome is the abstention rate, by phrase matching, on "
        "post-trained checkpoints and the entropy of the next-token "
        "distribution on the others.",
        "The direction and its layer come from train and validation "
        "questions; the steered questions are test questions.",
        "{n_random} random directions of the same norm, steered through "
        "the same grid.",
        NONE,
        "The mean slope of the outcome over the coefficients, with a "
        "bootstrap interval over the questions, against each random "
        "direction and between unknown and known questions.",
        scope="GPT-2 small has too few known questions for a contrast.",
    ),
    Card(
        "approximate_answers",
        APPROXIMATE,
        "Answers to range and exact questions",
        ("p10a",),
        "Does a model answer with a range when the sourced answer is a range?",
        "Wikidata quantity questions on height, wingspan and duration, "
        "{n_per_side} per side and family, and people with one exact "
        "height.",
        "The kind of the sourced answer, range or exact, from the bounds of "
        "the Wikidata statement and not from a model.",
        "Every question is asked, free and when told to give one number, "
        "with one greedy and {n_samples} sampled answers.",
        "The people, who are living things with an exact answer.",
        NONE,
        "The share of answers of each form and the range share on range "
        "questions minus that on exact questions, with Wilson and bootstrap "
        "intervals over the questions.",
    ),
    Card(
        "approximate_reading",
        APPROXIMATE,
        "Reading range against exact",
        ("p10b",),
        "Can a direction tell a range question from an exact one before "
        "the model answers?",
        "The same questions, bare and in the concept template of the paper, "
        "read at the last token.",
        "The kind of the sourced answer, as above.",
        "Each family is hashed into train, validation and test; readers "
        "are fitted on one, two or three families, {n_seeds} draws each, "
        "with the layer chosen on validation questions and the score on "
        "test questions, also of a family held out of training.",
        "A random direction with a layer of its own, classifiers on the "
        "words and on the characters, the length of the question, the "
        "popularity of the subject, and {n_people} people as test items.",
        SAME_SWEEP,
        "AUROC of the test questions with range as the positive class, "
        "with a bootstrap interval over the test questions.",
    ),
    Card(
        "approximate_links",
        APPROXIMATE,
        "Links to knowledge and to hedging",
        ("p10c",),
        "Is the range direction related to the knowledge directions and to "
        "how often the model answers with a range?",
        "The stored range reader, knowledge readers and answers of the "
        "same checkpoint; nothing is trained.",
        "The kind of the sourced answer, as above.",
        "The link to the answers uses the test questions, inside one kind "
        "of question.",
        "The band of {n_random} random unit directions at the same layer.",
        NONE,
        "The cosine between directions per layer, the AUROC of the "
        "knowledge readers on the quantity questions and a rank "
        "correlation, with bootstrap intervals over the questions.",
    ),
    Card(
        "truth",
        TRUTH,
        "True and false statements",
        ("r7",),
        "Does a direction read whether a statement is true, also on "
        "statements from other sources?",
        "The true and false statements of the representation engineering "
        "repository, and nine Geometry of Truth sets scored with no refit.",
        "The true or false label of each source file.",
        DRAWS + "the layer is chosen on validation pairs and the score is "
        "on test pairs; a source statement stays in one split.",
        READING_CONTROLS,
        PROBE,
        "Pair accuracy on the test pairs and on each of the other sets.",
    ),
    Card(
        "truthfulqa",
        TRUTH,
        "TruthfulQA",
        ("r6",),
        "Does a reader choose the true answer more often than the words alone?",
        "{n_questions} multiple-choice questions with shuffled choices; "
        "the readers are fitted on TruthfulQA train questions, on ARC "
        "items or on true statements under two personas.",
        "The correct choice of the source.",
        "{n_seeds} readers; {n_train} train, {n_val} validation and "
        "{n_test} test questions by hash, with the layer chosen on the "
        "validation questions.",
        "A random direction, shuffled labels, and classifiers on the words "
        "and on the characters of the text alone.",
        "The standard and the heuristic zero-shot scorers of the paper, on "
        "every question.",
        "The share of test questions on which the true choice scores highest.",
    ),
    Card(
        "honesty",
        TRUTH,
        "Honesty transfer",
        ("r5",),
        "Does a direction found from an honest and an untruthful persona "
        "read the truth of plain statements?",
        "True statements under an honest and an untruthful persona, read "
        "at the response tokens; plain true and false statements for the "
        "test.",
        "In training the persona, which the template sets; in the test the "
        "true or false label of the statements file.",
        DRAWS + "the layer is chosen on {n_eval_pairs.transfer.val} "
        "validation pairs and the score is on {n_eval_pairs.transfer.test} "
        "test pairs, both of plain statements.",
        READING_CONTROLS,
        PROBE,
        "Pair accuracy on the plain statements, beside the accuracy inside "
        "the persona template.",
    ),
    Card(
        "agreement",
        TRUTH,
        "Agreement of directions",
        ("a1",),
        "Do directions fitted on different draws of the training pairs agree?",
        "The stored directions of the sentiment, utility, morality, "
        "honesty and truth readers; no activation is read.",
        "None: it compares directions.",
        "The {n_seeds} draws of each reader, at the layer at half of the "
        "depth.",
        "The absolute cosine expected between random directions.",
        NONE,
        "The mean cosine between the directions of two draws, with a t "
        "interval on jackknife values.",
    ),
    Card(
        "sentiment_steering",
        STEERING,
        "Sentiment steering",
        ("c1",),
        "Does adding a sentiment direction move the judged sentiment of "
        "the text, and at what cost to the text?",
        "Completions of {n_prompts} short neutral prompts, {num_samples} "
        "samples each, steered over a grid of relative coefficients at "
        "three consecutive layers.",
        "The positive-class probability of a sentiment classifier, with a "
        "second classifier as a check.",
        "A layer sweep on one list of prompts chooses where to steer and "
        "the results are on a second list; {n_steer_seeds} steering "
        "vectors.",
        "None of the random or shuffled kind: the comparisons are the "
        "unsteered model and an instruction in the prompt.",
        NONE,
        "The mean judged score per coefficient, and the fluency rule on "
        "the perplexity of {n_neutral} held-out WikiText-103 passages and "
        "the distinct-1 of the completions.",
    ),
    Card(
        "post_training",
        POST,
        "Base against post-trained",
        ("p8",),
        "What does post-training change in reading and in steering?",
        "No new stimuli: the finished sentiment, utility, morality, truth "
        "and honesty readers, the wording run and the sentiment steering "
        "run of a base checkpoint and of its post-trained pair.",
        "Those of the runs it reads.",
        "The draw is the paired unit: one draw uses the same stimuli on "
        "both checkpoints of a pair.",
        "None of its own; every run it reads carries its controls.",
        NONE,
        "The difference between the post-trained and the base checkpoint "
        "per draw, with a bootstrap interval over the draws of {n_boot} "
        "resamples.",
        scope="One stored run reads the finished runs of each pair.",
        paired=True,
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


def _facts(manifest: Manifest, card: Card) -> dict[str, float]:
    """Reads the counts that the texts of a card ask for."""
    paths = sorted({p for text in card.texts() for p in SLOT.findall(text)})
    if not paths:
        return {}
    (code,) = card.codes
    found: dict[str, dict[str, float | None]] = {path: {} for path in paths}
    for model in manifest.models(code):
        run = manifest.run(code, model)
        assert run is not None
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        summary = runfiles.summary(run)
        for path in paths:
            value = _stored(config, path)
            if value is None:
                value = _stored(summary, path)
            found[path][model] = value
    return {
        path: agreed(f"{code} {path}", values) for path, values in found.items()
    }


def _filled(text: str, facts: Mapping[str, float]) -> str:
    """Writes the stored counts into the slots of a text."""
    return SLOT.sub(lambda slot: f"{int(facts[slot[1]]):,}", text)


def _members(manifest: Manifest, card: Card) -> set[str]:
    """Finds the checkpoints an experiment covers."""
    (code,) = card.codes
    models = manifest.models(code)
    if not card.paired:
        return set(models)
    run = manifest.run(code, models[0])
    assert run is not None
    return {m for pair in runfiles.summary(run)["pairs"] for m in pair}


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the static cards and the committed runs.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    rows: list[list[str]] = []
    numbers: dict[str, float] = {}
    counts: dict[str, dict[str, stats.RoleCount]] = {}
    for card in CARDS:
        facts = _facts(manifest, card)
        for path, value in facts.items():
            numbers[f"{card.key}_{path.replace('.', '_')}"] = value
        if card.model_free:
            checkpoints = NO_MODEL
        else:
            members = _members(manifest, card)
            counts[card.key] = stats.per_role(
                {m: m in members for m in manifest.checkpoints()}
            )
            checkpoints = " ".join(
                part
                for part in (
                    f"{stats.per_role_text(counts[card.key])}.",
                    card.scope,
                )
                if part
            )
        texts = [_filled(text, facts) for text in card.texts()]
        cells = [f"{card.study}.", *texts, checkpoints]
        for place, (aspect, cell) in enumerate(
            zip(ASPECTS, cells, strict=True)
        ):
            rows.append([card.name if place == 0 else "", aspect, cell])
    studies = list(dict.fromkeys(card.study for card in CARDS))
    numbers |= {
        "experiments": float(len(CARDS)),
        "studies": float(len(studies)),
        "rows": float(len(rows)),
    }
    return TableOutput(
        "The design of every experiment shown on this page: its question, "
        "stimuli, labels, splits, controls, published baselines, metric and "
        "checkpoints.",
        "The experiments are in the order of their studies. A control uses "
        "no learned signal for the concept, and a published baseline is a "
        "published method run on the same items. Every count of pairs, "
        "draws, questions or prompts is read from the stored configuration "
        "or summary of the runs and is the same on every checkpoint of the "
        "experiment. Checkpoints are counted per role.",
        ["Experiment", "Aspect", "Design"],
        rows,
        "lll",
        [],
        numbers,
        row_groups=tuple(range(0, len(rows), len(ASPECTS))),
        counts=counts,
    )
