"""The text and structure of the results paper.

Prose is written once here. A paragraph may hold references that the page
resolves: `[[fig:finders]]` and `[[table:reading]]` become numbered links
such as "Figure 3" and "Table 4", `[[sec:finders]]` becomes "Section 4.1",
`[[cite:zou2025representationengineeringtopdownapproach]]` a numbered
citation, `[[fn:text]]` a footnote, `[[num:fig:finders.random_max]]` a number
a figure or table computed and `[[count:fig:finders.pca_sentiment]]` its
per-role count. Only the one-sentence summary and the abstract state findings
in their own words, with no numbers; they summarize the computed findings and
tests pin each claim. Every number on the page comes from a figure or table
the builder computed from the committed results, except the design facts of
the method, the limitations and the reproducibility section, which tests
compare with the run files.

The page opens with a one-sentence summary, the overview figure, the
abstract and the key findings; the body follows, and the appendix holds the
detail behind each finding.
"""

import dataclasses

from tomography.site.cells import MINUS
from tomography.site.figs import knowledge_steering
from tomography.site.stats import FLUENT_MAX_RATIO
from tomography.site.stats import FLUENT_MIN_DISTINCT

PM = chr(0x00B1)
ELL = chr(0x2113)

TITLE = (
    "Reading and Steering Concepts with One Direction: Representation "
    "Engineering on GPT-2 and Qwen3"
)
REPO_URL = "https://github.com/javierdejesusda/tomography"
REPO_TEXT = "github.com/javierdejesusda/tomography"
FINDINGS_URL = REPO_URL + "/blob/main/docs/findings.md"
METHODOLOGY_URL = REPO_URL + "/blob/main/docs/methodology.md"
DATA_URL = REPO_URL + "/tree/main/results"

MAX_RATIO = f"{FLUENT_MAX_RATIO:g}"
MIN_DISTINCT = f"{FLUENT_MIN_DISTINCT:g}"
FLUENCY_RULE = (
    "neutral-text perplexity ratio at most "
    + MAX_RATIO
    + " and distinct-1 at least "
    + MIN_DISTINCT
)

TLDR = (
    "A logistic direction reads utility and morality above a random "
    "direction on every checkpoint and truth on every Qwen3 checkpoint; the "
    "finder of the paper is weaker, a truth reader does not beat the words "
    "alone, a known-against-unknown direction does not beat the model's "
    "confidence, and utility steering that beats every random direction "
    "meets the fluency rule on only Qwen3-1.7B-Base."
)

FRONT_ITEMS = ("fig:teaser",)

ABSTRACT = (
    "We test representation engineering (Zou et al., 2023) on four GPT-2 "
    "and three Qwen3 sizes, each as base and post-trained "
    "checkpoint, against random-direction controls and, where the wording "
    "leaks the label, text-classifier controls. By the 95 percent interval, a "
    "logistic direction reads utility and morality above its random direction "
    "on every checkpoint and truth on every Qwen3 checkpoint but no GPT-2 "
    "size. The paper's finder is above a random direction for "
    "utility on every Qwen3 checkpoint, where the pairs are matched, but "
    "on fewer for sentiment and truth, where they are unrelated; matching "
    "is confounded with the concept and the data. A TruthfulQA reader is "
    "not above a classifier on the words alone, and a known-against-unknown "
    "direction, which ranks questions above popularity and text "
    "classifiers on most Qwen3 checkpoints, does not add to the confidence "
    "of the model. Steering at relative "
    "coefficient 0.1 moves judged sentiment on most checkpoints, while "
    "utility steering at relative coefficient 0.3 beats every random "
    "direction on only some. "
    "Post-training lowers held-out sentiment reading, with an interval "
    "below zero, on [[num:fig:base_post.sent_below]] of 3 Qwen3 sizes."
)


@dataclasses.dataclass(frozen=True)
class Section:
    """A numbered section or subsection.

    Attributes:
        key: Anchor name, used by `[[sec:key]]`.
        title: Heading text.
        paragraphs: Paragraphs before the items.
        runin: Run-in headed paragraphs, as (head, text), after them.
        listed: A numbered list of (head, text) entries, after the items.
        items: Figure and table keys placed after the paragraphs.
        after: Paragraphs after the items and the list.
        subsections: Subsections, rendered last.
    """

    key: str
    title: str
    paragraphs: tuple[str, ...] = ()
    runin: tuple[tuple[str, str], ...] = ()
    items: tuple[str, ...] = ()
    listed: tuple[tuple[str, str], ...] = ()
    after: tuple[str, ...] = ()
    subsections: tuple["Section", ...] = ()


KEY_FINDINGS = Section(
    "key_findings",
    "Key findings",
    listed=(
        (
            "The finder of the paper ([[sec:finders]]).",
            "The leading principal component of paired differences is above "
            "a random direction, with an interval excluding zero, for "
            "sentiment on [[count:fig:finders.pca_sentiment]], but for "
            "utility on every Qwen3 checkpoint; matching is confounded with "
            "the concept and the data. Logistic regression is above a random "
            "direction for sentiment on "
            "[[count:fig:finders.logistic_sentiment]] ([[fig:finders]]).",
        ),
        (
            "Where a concept is read ([[sec:reading]]).",
            "A logistic direction reads utility and morality above its "
            "random direction on [[count:fig:teaser.utility]] and truth on "
            "[[count:fig:teaser.truth]]; a change of wording moves a "
            "sentiment reader by more than the noise of the test pairs on "
            "[[count:fig:wording.span_above_noise]] ([[fig:reading]], "
            "[[fig:teaser]]).",
        ),
        (
            "Known and unknown questions ([[sec:known_unknown]]).",
            "A known-against-unknown direction ranks questions above "
            "popularity and the text classifiers, with an interval "
            "excluding zero, on "
            "[[count:fig:knowledge_reading.above_best]], but adds to the "
            "confidence of the model, with an interval excluding zero, on "
            "[[count:fig:teaser.knowledge_confidence]]; it moves unknown "
            "questions more than random directions on "
            "[[runs:fig:knowledge_steering.distinguishable]], and a range "
            "reader is above its random control on "
            "[[count:fig:approximate.reader_above_random]] "
            "([[fig:knowledge_reading]], [[fig:knowledge_steering]], "
            "[[fig:approximate]]).",
        ),
        (
            "Truth ([[sec:truth]]).",
            "A TruthfulQA reader is above a classifier on the words alone, "
            "with an interval excluding zero, on "
            "[[count:fig:teaser.truthfulqa]] ([[fig:truth]]).",
        ),
        (
            "Steering ([[sec:steering]]).",
            "Steering at relative coefficient 0.1 moves the judged "
            "sentiment on [[count:fig:steering.moves01]] and keeps the "
            "fluency rule (" + FLUENCY_RULE + ") on "
            "[[count:fig:steering.fluent01]]; utility steering at "
            "relative coefficient 0.3 beats every random direction on "
            "[[count:fig:teaser.steering]] ([[fig:steering]]).",
        ),
        (
            "Post-training ([[sec:base_post]]).",
            "Held-out sentiment reading is lower after post-training, with "
            "an interval below zero, on [[num:fig:base_post.sent_below]] of "
            "3 Qwen3 sizes, and [[num:fig:base_post.paired_excluding_zero]] "
            "of [[num:fig:base_post.paired_total]] other seed-paired "
            "differences exclude zero ([[fig:base_post]]).",
        ),
    ),
)

INTRODUCTION = Section(
    "introduction",
    "Introduction",
    paragraphs=(
        "Zou et al. [[cite:zou2025representationengineeringtopdownapproach]] "
        "propose representation engineering: find a direction for a concept "
        "from the activations of contrasting inputs, read the concept by "
        "projecting new activations on it, and control the model by adding "
        "it to the residual stream. The recipe rests on the linear "
        "representation hypothesis, the view that concepts correspond to "
        "directions in activation space [[cite:park2024linear]]. The paper "
        "reports reading and control results on LLaMA-2 chat models for "
        "truthfulness, honesty, utility, morality, emotions and other "
        "concepts.",
        "We run the recipe on smaller open models where every step can be "
        "checked: GPT-2 [[cite:radford2019language]] at four sizes and "
        "Qwen3 [[cite:yang2025qwen3]] at three sizes, each as a base and a "
        "post-trained checkpoint. Our readers are logistic regression "
        "unless a figure names the finder of the paper, because that finder "
        "is above a random direction by its interval for only some concepts "
        "and roles ([[sec:finders]]). For each question we name its "
        "controls, which use no direction fitted for the concept: a random "
        "direction oriented like the reader, a classifier on the text (for "
        "the readings, TruthfulQA, knowledge and approximate answers, where "
        "the wording can give the label away) and the popularity of the "
        "subject. The baselines are published methods: logistic "
        "regression, mass-mean probes and, for knowledge, model confidence "
        "built from published signals such as P(True) "
        "[[cite:kadavath2022language]] ([[sec:controls_method]]). Three "
        "terms recur. The margin is the accuracy of a reader minus that of a "
        "random direction of the same run; the relative depth is a layer "
        "index divided by the number of layers; and the relative "
        "coefficient is the steering strength in units of the mean "
        "residual norm at the layer. [[fig:teaser]] summarises the "
        "answers.",
        "Our contributions are:",
    ),
    listed=(
        (
            "Readings against controls ([[sec:method]]).",
            "Every reading is reported against a random direction fitted "
            "with the same procedure and, where the wording can give the "
            "label away, against a text control, with counts per role and "
            "never pooled.",
        ),
        (
            "A test of the finder ([[sec:finders]]).",
            "The finder of the paper is compared with the logistic-"
            "regression baseline on matched pairs (utility) and on unrelated "
            "pairs (sentiment and truth).",
        ),
        (
            "Two families, base and post-trained ([[sec:base_post]]).",
            "Four GPT-2 sizes and three Qwen3 sizes, each Qwen3 size as a "
            "base and a post-trained checkpoint, so a difference between "
            "roles is measured, not assumed.",
        ),
        (
            "Known and unknown questions ([[sec:known_unknown]]).",
            "A known-against-unknown direction is compared with popularity, "
            "text classifiers and the confidence of the model, its steering "
            "with random directions of the same norm, and a range direction "
            "for approximate answers with a random control.",
        ),
        (
            "A page rebuilt from the result files ([[sec:reproducibility]]).",
            "Every number comes from the committed results and is "
            "recomputed when the page is built.",
        ),
    ),
)

RELATED = Section(
    "related",
    "Related work",
    paragraphs=(
        "Fitting a classifier on hidden states to read a property is "
        "probing [[cite:alain2016understanding,belinkov2022probing]]. A "
        "probe can succeed because it learns the task rather than reads it, "
        "which control tasks are built to expose [[cite:hewitt2019designing]]"
        "; our controls are a random direction and, where the wording can "
        "give the label away, a classifier on the text alone. "
        "Representation engineering "
        "[[cite:zou2025representationengineeringtopdownapproach]] reads a "
        "concept with the first principal component of differences between "
        "paired activations and compares that finder with others on utility "
        "in its Section 5.1; several finders recover a similar sentiment "
        "direction in GPT-2 and Pythia [[cite:tigges2024language]], and on a "
        "shared benchmark difference-in-means is the best concept detector, "
        "prompting beats every method for steering (followed by finetuning) "
        "and sparse autoencoders are not competitive on either "
        "[[cite:wu2025axbench]].",
        "Truth has been read from hidden states with supervised probes "
        "[[cite:azaria2023internal]], with mass-mean probes whose directions "
        "transfer across datasets of true and false statements "
        "[[cite:marks2024geometry]], and without labels by contrast-"
        "consistent search [[cite:burns2023discovering]]; such probes can "
        "fail on negated statements [[cite:levinstein2025still]], which our "
        "negated sets test.",
        "Adding a direction to the residual stream follows activation "
        "addition [[cite:turner2023steering]] and contrastive activation "
        "addition [[cite:rimsky2024steering]]; inference-time intervention "
        "shifts the outputs of selected attention heads along probe "
        "directions [[cite:li2023inference]], and one direction mediates "
        "refusal in chat models [[cite:arditi2024refusal]]. Steering vectors "
        "are unreliable for many inputs and concepts "
        "[[cite:tan2024analysing]].",
        "Language models can report whether they know an answer "
        "[[cite:kadavath2022language]], answerability can be decoded from "
        "their hidden states [[cite:slobodkin2023curious]], probes of the "
        "subject representation predict which PopQA facts a model knows "
        "better than popularity does [[cite:gottesman2024estimating]], and "
        "directions for known and unknown entities steer refusal in chat "
        "models [[cite:ferrando2025do]]. Popularity itself predicts factual "
        "recall on PopQA [[cite:mallen2023when]].",
        "Model behaviour is sensitive to formatting choices in the prompt "
        "[[cite:sclar2024quantifying]], which motivates our wording test. Du "
        "et al. [[cite:du2025how]] compare base and post-trained models on "
        "knowledge, truthfulness, refusal and confidence; our comparison is "
        "limited to the released Qwen3 pairs.",
    ),
)

METHOD = Section(
    "method",
    "Method",
    paragraphs=(
        "[[table:checkpoints]] lists the checkpoints, [[fig:method]] shows "
        "the pipeline and [[table:notation]] defines the symbols and terms; "
        "[[sec:appendix_method]] gives every design detail.",
    ),
    items=("table:checkpoints",),
    subsections=(
        Section(
            "pipeline",
            "Reading a concept",
            paragraphs=(
                "A reading experiment builds pairs of inputs that differ in "
                "the concept: two phrases of opposite sentiment, two "
                "versions of a scenario with different utility or morality, "
                "a true and a false statement. Each input is wrapped in the "
                "template of the paper for the concept, and we read h(x, "
                + ELL
                + "), the residual stream at the last token of the input x "
                "after layer " + ELL + ". A finder turns the training pairs "
                "into one unit direction v(" + ELL + ") per layer; unless a "
                "figure names another finder, it is logistic regression on "
                "the paired activations. The direction is oriented with the "
                "training labels, so that the positive member of a training "
                "pair tends to project higher. A held-out pair is read "
                "correctly when its positive member projects higher than "
                "its negative member, and ties count one half; the "
                "accuracy is the share of held-out pairs read correctly, so "
                "a direction with no information scores 0.5 and no "
                "threshold is involved.",
                "The layer chosen on validation is picked on pairs that the "
                "reported score does not use. The main "
                "sentiment run is the exception: it chooses its layer on the "
                "same 256 SST-2 validation pairs it is scored on, so its "
                "value is optimistic. We therefore report sentiment from the "
                "sample-size run at 64 training pairs, which scores a "
                "separate report split of 214 pairs, and keep the optimistic "
                "run in [[table:reading_full]] only.",
            ),
            items=("fig:method",),
        ),
        Section(
            "checkpoints",
            "Checkpoints and data",
            paragraphs=(
                "GPT-2 [[cite:radford2019language]] small, medium, large and "
                "XL, and Qwen3 [[cite:yang2025qwen3]] 0.6B, 1.7B and 4B, "
                "each as the base checkpoint and the post-trained one. A "
                "post-trained checkpoint is used with its chat template and "
                "thinking disabled. Sentiment is read on phrases of "
                "the Stanford Sentiment Treebank "
                "[[cite:socher2013recursive]], utility and morality on "
                "ETHICS [[cite:hendrycks2021aligning]], and truth on the "
                "true and false statements of the representation "
                "engineering repository "
                "[[cite:zou2025representationengineeringtopdownapproach]], "
                "tested out of distribution on the Geometry of Truth sets "
                "[[cite:marks2024geometry]]. TruthfulQA "
                "[[cite:lin2022truthfulqa]], PopQA [[cite:mallen2023when]] "
                "and Wikidata quantities [[cite:vrandecic2014wikidata]] give "
                "the questions of the TruthfulQA, knowledge and "
                "approximate-answer readings; generated text is "
                "scored by the positive-class probability of SiEBERT "
                "[[cite:hartmann2023more]]. Datasets, judges and revisions "
                "are in [[table:assets]].",
            ),
        ),
        Section(
            "controls_method",
            "Controls and baselines",
            paragraphs=(
                "Every reading run also scores random unit directions u("
                + ELL
                + ") at every layer, oriented on the training labels "
                "exactly like the reader. Because the orientation uses the "
                "labels, a random direction can score far above 0.5 when the "
                "concept dominates the activations (up to "
                "[[num:fig:finders.random_max]], on "
                "[[label:fig:finders.random_max]], in the finder "
                "comparison), "
                "so every reading is reported as its margin over the random "
                "direction of the same run, not over 0.5; the two share a "
                "layer except in the finder comparison, which reads each at "
                "the layer chosen on its own select split. Where the wording "
                "of an "
                "input can give the label away (the sentiment, utility, "
                "morality and truth readings, TruthfulQA, knowledge questions, "
                "approximate answers), we add classifiers on the "
                "words and on the characters of the input, fitted on the "
                "same training items without the model ([[table:reading]] "
                "gives them as words alone); for knowledge we "
                "add the popularity of the subject. The steering comparisons "
                "use three random directions of the same norm.",
                "Beside the finder of the paper we run logistic regression, "
                "a linear probe [[cite:alain2016understanding]], and the "
                "mean difference, the mass-mean probe "
                "[[cite:marks2024geometry]] ([[table:finders_all]]); for "
                "knowledge the baseline is the confidence of the model, which "
                "stacks five output signals, among them P(True) "
                "[[cite:kadavath2022language]].",
            ),
        ),
        Section(
            "steering_method",
            "Steering",
            paragraphs=(
                "Steering adds c r(" + ELL + ") v(" + ELL + ") to the "
                "residual stream at three consecutive layers (for the "
                "sentiment run only; the utility steering run steers at 40, 55 "
                "and 70 percent of the depth), at every position of the "
                "response, where v("
                + ELL
                + ") is the unit direction, r("
                + ELL
                + ") is the mean residual norm at layer "
                + ELL
                + " on "
                "calibration text and c is the relative coefficient. The "
                "judged score of a checkpoint is the mean SiEBERT "
                "positive-class probability of its completions. We call "
                "steered text fluent only when two conditions hold, the "
                "fluency rule: the " + FLUENCY_RULE + " (the share of "
                "distinct tokens in the completions "
                "[[cite:li2016diversity]]). Three random directions of the "
                "same norm are the reference for the win rate of the "
                "utility test, not 0.5.",
            ),
        ),
        Section(
            "intervals",
            "Intervals and counts",
            paragraphs=(
                "Every reading interval on this page is computed from the "
                "test size: a 95 percent Wilson interval "
                "[[cite:wilson1927probable]] for an accuracy, a conservative "
                "interval for a difference between two accuracies, and the "
                "approximation of Hanley and McNeil "
                "[[cite:hanley1982meaning]] for an AUROC. No interval "
                "includes the variation between independent runs, because "
                "each cell is one run. Counts of checkpoints are given per "
                "role (GPT-2, Qwen3 base, Qwen3 post-trained) and never "
                "pooled; they are descriptive, and no correction for the "
                "number of comparisons is applied.",
            ),
        ),
    ),
)

RESULTS = Section(
    "results",
    "Results",
    subsections=(
        Section(
            "finders",
            "Does the finder of the paper find the concept?",
            paragraphs=(
                "The finder of the paper is above a random direction by the "
                "95 percent interval for utility on every Qwen3 checkpoint, "
                "where the pairs are matched, and on fewer checkpoints for "
                "sentiment and truth, "
                "where they are unrelated, while logistic regression is "
                "above one for sentiment on every checkpoint.",
            ),
            items=("fig:finders",),
            after=(
                "The paper compares finders on utility, where every pair is "
                "two versions of one scenario. On sentiment and truth our "
                "pairs are unrelated inputs with opposite labels, the "
                "setting of most probing work. Matching is confounded with "
                "the concept and the data, so the comparison does not show "
                "that matching is what matters. The layer curves are in "
                "[[fig:a_finders_gpt2]] and [[fig:a_finders_qwen3]] and "
                "every finder in [[table:finders_all]] "
                "([[sec:appendix_reading]]).",
            ),
        ),
        Section(
            "reading",
            "Where and how well is a concept read?",
            paragraphs=(
                "A logistic direction reads utility and morality above its "
                "random direction on every checkpoint and truth on every "
                "Qwen3 checkpoint but no GPT-2 checkpoint, and a change of "
                "the wording of the template moves a sentiment reader by "
                "more than the noise of the test pairs on every Qwen3 "
                "checkpoint and on two of the four GPT-2 checkpoints.",
            ),
            items=("fig:reading", "table:reading"),
            after=(
                "The margin over a random direction exceeds 0.1 for "
                "sentiment from a relative depth of "
                "[[num:fig:reading.onset_sentiment_gpt2_min]] to "
                "[[num:fig:reading.onset_sentiment_gpt2_max]] on GPT-2, "
                "[[num:fig:reading.onset_sentiment_base_min]] to "
                "[[num:fig:reading.onset_sentiment_base_max]] on Qwen3 "
                "base and "
                "[[num:fig:reading.onset_sentiment_post_min]] to "
                "[[num:fig:reading.onset_sentiment_post_max]] on "
                "post-trained checkpoints, and later for "
                "utility, morality and truth; across the GPT-2 sizes "
                "accuracy grows with size for utility and morality, and the "
                "truth reader on GPT-2 is not separable from its random "
                "direction at [[num:fig:reading.truth_n]] test pairs. The "
                "words alone read sentiment "
                "([[num:table:reading.word_sentiment]] with word n-grams, "
                "[[num:table:reading.char_sentiment]] with character "
                "n-grams on the held-out split) and utility "
                "([[num:table:reading.word_utility]], "
                "[[num:table:reading.char_utility]]) with an interval above "
                "chance, and do not separate from chance for morality, "
                "truth and honesty ([[table:reading]]). The wording study "
                "refits the sentiment reader on changed templates "
                "([[fig:wording]], [[fig:a_wording]], [[table:wordings]]; "
                "[[sec:appendix_wording]]). Layer curves and every reading "
                "with its layer are in [[sec:appendix_reading]] and "
                "[[fig:a_reading_curves]].",
            ),
        ),
        Section(
            "known_unknown",
            "Known and unknown questions",
            paragraphs=(
                "A known-against-unknown direction ranks questions above "
                "popularity and text classifiers on most Qwen3 checkpoints "
                "but adds at most a small amount to the confidence of the "
                "model, and a range reader "
                "is above its random control on only some checkpoints.",
            ),
            runin=(
                (
                    "Reading.",
                    "[[fig:knowledge_reading]] compares the direction with "
                    "popularity, text and the confidence of the model on "
                    "PopQA within relation: it ranks questions above the "
                    "strongest of popularity and the two text classifiers, "
                    "with an interval excluding zero, on "
                    "[[count:fig:knowledge_reading.above_best]], and its "
                    "increment over the confidence of the model has an "
                    "interval covering zero on "
                    "[[count:fig:knowledge_reading.dir_cover]], at most "
                    "[[num:fig:knowledge_reading.dir_inc_hi_max_gpt2]] "
                    "(GPT-2), "
                    "[[num:fig:knowledge_reading.dir_inc_hi_max_base]] "
                    "(Qwen3 base) and "
                    "[[num:fig:knowledge_reading.dir_inc_hi_max_post]] "
                    "(post-trained) at its upper end, while a "
                    "probe fitted on the correct and wrong labels has an "
                    "interval above zero on "
                    "[[count:fig:knowledge_reading.probe_above]]; GPT-2 small "
                    "has no knowledge-direction run, so the denominators "
                    "differ. "
                    "On Qwen3-4B-Base a classifier on the words or "
                    "characters comes within a small margin of the "
                    "direction or beats it on "
                    "[[num:fig:a_knowledge_matrix.close]] of "
                    "[[num:fig:a_knowledge_matrix.types]] question types "
                    "([[fig:a_knowledge_matrix]]).",
                ),
                (
                    "Steering.",
                    "Adding the direction moves the outcome on unknown "
                    "questions (the next-token entropy for GPT-2 and base "
                    "checkpoints, the abstention rate for post-trained ones) "
                    "more than each of three random directions of the same "
                    "norm on [[runs:fig:knowledge_steering.distinguishable]], "
                    "and more than on known questions on "
                    "[[runs:fig:knowledge_steering.selective]]; known "
                    "questions move at least as much as unknown ones on "
                    "[[runs:fig:knowledge_steering.known_ge_unknown]] "
                    "([[fig:knowledge_steering]]; the measures are in "
                    "[[sec:appendix_knowledge]]).",
                ),
                (
                    "Approximate answers.",
                    "Asked for a quantity whose sourced answer is a range, "
                    "Qwen3 base checkpoints write a range for "
                    "[[num:fig:approximate.range_base_min]] to "
                    "[[num:fig:approximate.range_base_max]] percent of the "
                    "items and post-trained ones for "
                    "[[num:fig:approximate.range_post_min]] to "
                    "[[num:fig:approximate.range_post_max]] percent when the "
                    "last example of the prompt is a range, and, told to "
                    "give one exact number, for at most "
                    "[[num:fig:approximate.forced_base_max]] percent (base) "
                    "and [[num:fig:approximate.forced_post_max]] percent "
                    "(post-trained). A range reader fitted on heights and "
                    "wingspans is above its random control on held-out "
                    "durations on [[count:fig:approximate.reader_above_random]]"
                    ", and above a classifier on the question words, with "
                    "the interval of the difference above zero, on "
                    "[[count:table:approximate.above_words]] "
                    "([[fig:approximate]]).",
                ),
            ),
            items=(
                "fig:knowledge_reading",
                "fig:knowledge_steering",
                "fig:approximate",
            ),
            after=(
                "Every dose curve is in [[sec:appendix_knowledge]] and the "
                "values behind the range reader in "
                "[[sec:appendix_approximate]].",
            ),
        ),
        Section(
            "truth",
            "Does a truth reader beat the words alone?",
            paragraphs=(
                "A TruthfulQA reader is not above a classifier on the words "
                "alone.",
            ),
            items=("fig:truth",),
            after=(
                "TruthfulQA can be answered in part from the answer text "
                "alone, and negation flips the label without changing the "
                "topic: [[fig:truth]] pairs the reader with two text "
                "classifiers, and at least one negated set falls below "
                "chance with its interval on "
                "[[count:fig:truth.neg_below_half]] "
                "([[fig:a_truth_cities]]). The readers and the honesty "
                "transfer are in [[sec:appendix_truth]].",
            ),
        ),
        Section(
            "steering",
            "Does steering change the output without breaking it?",
            paragraphs=(
                "Steering at relative coefficient 0.1 moves the judged "
                "sentiment on most checkpoints and keeps the fluency rule on "
                "most, but at relative coefficient 0.3 the steered sentiment "
                "text meets the rule on none, and utility steering beats every "
                "random direction on only some checkpoints.",
            ),
            items=("fig:steering",),
            after=(
                "The comparison with random directions is in "
                "[[fig:teaser]] (last row), and [[table:steering]] adds the "
                "completion perplexity ([[sec:appendix_steering]]).",
            ),
        ),
        Section(
            "base_post",
            "What does post-training change?",
            paragraphs=(
                "Post-training lowers held-out sentiment reading, with an "
                "interval below zero, on [[num:fig:base_post.sent_below]] of "
                "3 Qwen3 sizes, and on the other readings most intervals "
                "include zero, which is not evidence of no change.",
            ),
            items=("fig:base_post",),
            after=(
                "[[fig:base_post]] puts every measure of the earlier "
                "sections side by side as post-trained minus base for the "
                "three Qwen3 sizes; the stored per-seed comparisons are in "
                "[[table:base_post]] ([[sec:appendix_post]]).",
            ),
        ),
    ),
)

DISCUSSION = Section(
    "limitations",
    "Discussion and limitations",
    paragraphs=(
        "A direction reads a concept "
        "better than chance far more often than it beats the right "
        "comparison: orientation on the labels lifts a random direction, the "
        "wording of the input often carries the label, and the confidence "
        "of the model already holds what a knowledge direction reads, so a "
        "reading claim needs its random direction and, where the text can "
        "carry the label, a text control ([[fig:teaser]]). Steering moves "
        "the judged sentiment at small coefficients and "
        "breaks fluency at larger ones (a perplexity on neutral text above "
        + MAX_RATIO
        + " times the unsteered model, or a distinct-1 below "
        + MIN_DISTINCT
        + "; "
        "[[sec:steering]]), and post-training lowers held-out sentiment "
        "reading, with an interval below zero, on "
        "[[num:fig:base_post.sent_below]] of 3 Qwen3 sizes.",
    ),
    runin=(
        (
            "Scale.",
            "The largest model is Qwen3-4B, we cover two model families, and "
            "we do not reproduce the LLaMA-2 results of the paper; effects "
            "that appear only at larger scale are outside this study. Most "
            "concepts use one prompt format, the template of the paper, "
            "and the wording study varies it for sentiment only.",
        ),
        (
            "Test size and intervals.",
            "Truth and honesty are scored on 59 test pairs and TruthfulQA "
            "on 173 questions, so differences below about "
            "[[num:fig:teaser.truth_half_gpt2_min]] to "
            "[[num:fig:teaser.truth_half_gpt2_max]] on GPT-2, "
            "[[num:fig:teaser.truth_half_base_min]] to "
            "[[num:fig:teaser.truth_half_base_max]] on Qwen3 base and "
            "[[num:fig:teaser.truth_half_post_min]] to "
            "[[num:fig:teaser.truth_half_post_max]] on Qwen3 post-trained "
            "checkpoints are not resolved; GPT-2 truth readings are "
            "inconclusive rather than negative. Layers are selected on "
            "small validation sets, so the chosen layer carries its own "
            "uncertainty. Our intervals cover the sampling of test items "
            "only: each cell is one run, so variation between runs, prompts "
            "or seeds of the fitting procedure is not included, and the "
            "difference intervals ignore the pairing of the arms and are "
            "conservative.",
        ),
        (
            "Base against post-trained.",
            "A post-trained Qwen3 checkpoint differs from its base in "
            "post-training data, chat format and distillation, so a "
            "difference is a contrast between those recipes. The stored "
            "per-seed comparison covers only the 0.6B and 1.7B pairs; "
            "[[fig:base_post]] adds the 4B pair from the run summaries.",
        ),
        (
            "Unmatched pairs.",
            "The finder of the paper is tested on matched pairs only for "
            "utility; for sentiment and truth our pairs are unrelated "
            "inputs, and we did not build matched pairs for them.",
        ),
        (
            "Judges.",
            "Sentiment is the positive-class probability of a classifier, "
            "the utility steering test is judged by the same sentiment "
            "classifier, which makes one judge family, and abstention is "
            "detected by phrase matching without human labels, which does "
            "not separate a generic answer to an unknowable detail from a "
            "specific invented one, because the runs do not label which a "
            "non-abstaining answer is.",
        ),
        (
            "Labels and approximate answers.",
            "Known and unknown questions are defined by the answers of each "
            "model, so question sets differ between models and an AUROC is "
            "not a measure of the same questions across models; the labels "
            "of knowledge questions are alias string matches with no hand "
            "check, and GPT-2 small has too few known questions for a "
            "contrast. For approximate answers the range side is a living "
            "thing and the exact side an artifact in every family, the "
            "wording differs by side, test sets hold 22 to 30 questions per "
            "side, and an item bootstrap interval collapses at an AUROC of "
            "1.000.",
        ),
        (
            "Controls and comparisons.",
            "The readings are compared with model-free word and character "
            "n-gram classifiers from a separate run ([[table:reading]]), "
            "which see only the text, not the activations at the layer of "
            "the reader; the honesty control is fitted on the true and "
            "false labels of the facts, not on the persona labels, so it "
            "is a control for the transfer pairs only, and its interval "
            "covers test-pair sampling only. The sentiment steering run has no "
            "random-direction arm (the utility steering run has three), and "
            "knowledge steering has three random directions per run, so a "
            "single run passes against them by chance with probability "
            "1/4. The page reports many comparisons across 10 checkpoints "
            "and counts how many pass; the counts are descriptive and not "
            "corrected for the number of comparisons.",
        ),
    ),
)

REPRODUCIBILITY = Section(
    "reproducibility",
    "Reproducibility",
    paragraphs=(
        "Every number on this page is computed when the page is built, "
        "from the result files in the results directory of the repository; "
        "each figure and table links the data file it draws, and the "
        "tests of the site recompute every number in a caption from the "
        "result files. Links to the findings document open it at the "
        "commit of these results. Each run directory "
        "holds its configuration, its run identity (code commit and "
        "configuration hash) and its outputs. Models and datasets are "
        "pinned to the revisions in [[table:assets]]. The experiments ran "
        "on a single consumer GPU, on rented single-GPU cloud machines and "
        "on a 16-thread CPU machine. One experiment trains a model beyond "
        "the readers: a LoRRA adapter fitted on all 10 checkpoints "
        "([[sec:appendix_other]]). "
        "Every experiment is one command, tomography run followed by the "
        "experiment and the checkpoint, with base seed 0 from which the "
        "seeds of the readers are derived by a hash, so a rerun reproduces "
        "its files; the section of the README on running experiments lists "
        "the command and the options of each ([[table:notation]] explains "
        "the run codes of the run directories).",
    ),
)

APPENDICES = (
    Section(
        "appendix_assets",
        "Assets and notation",
        paragraphs=(
            "[[table:assets]] lists every checkpoint, judge and dataset "
            "with its pinned revision and the license its source states, "
            "which is what makes the results reproducible, and "
            "[[table:notation]] defines the symbols and terms of the "
            "paper.",
        ),
        items=("table:assets", "table:notation"),
    ),
    Section(
        "appendix_method",
        "Method details",
        paragraphs=(
            "These are the design details that the method summarises; they "
            "support every result of the paper, because each one is "
            "reported against a control built the same way.",
            "[[table:design]] gives, for every experiment shown on this "
            "page, its question, stimuli, labels, splits, controls, "
            "published baselines, metric and checkpoints, and the run-ins "
            "headed Design state, study by study, what is asked, why the "
            "design answers it and what limits the answer. No hypothesis was "
            "fixed before its run, except the tolerance of the views run, "
            "which was set before those runs. So a run-in names what would "
            "count as evidence and not a hypothesis that was tested.",
        ),
        runin=(
            (
                "Readers and draws.",
                "A run fits 10 readers (15 for TruthfulQA and 20 for the "
                "held-out sentiment run), each on its own draw of 64 "
                "training pairs, and reports the mean of their accuracies. "
                "The honesty readers read the response tokens of the "
                "persona prompts instead of the last token of the input.",
            ),
            (
                "Layer choice.",
                "The layer is picked on a validation split for utility, "
                "morality and truth, on 53 validation statements for the "
                "honesty transfer, on a select split for the finder "
                "comparison and the sample-size run, and on validation "
                "questions for TruthfulQA. The main sentiment run chooses "
                "its layer on the 256 SST-2 validation pairs it is scored "
                "on; the sample-size run scores a separate report split of "
                "214 pairs.",
            ),
            (
                "Data and judges.",
                "Truth is read on the true and false statements that the "
                "representation engineering repository "
                "[[cite:zou2025representationengineeringtopdownapproach]] "
                "derives from Azaria and Mitchell "
                "[[cite:azaria2023internal]], and tested out of "
                "distribution on the Geometry of Truth sets "
                "[[cite:marks2024geometry]], whose common claims come from "
                "Casper et al. [[cite:casper2023explore]]; honesty is read "
                "on the same statements under the persona prompts of the "
                "paper. TruthfulQA [[cite:lin2022truthfulqa]] and ARC "
                "[[cite:clark2018think]] give multiple-choice questions. "
                "Knowledge is read on questions the model can or cannot "
                "answer, drawn from PopQA [[cite:mallen2023when]], TriviaQA "
                "[[cite:joshi2017triviaqa]], KUQ "
                "[[cite:amayuelas2024knowledge]], FalseQA "
                "[[cite:hu2023wont]], SyntHal [[cite:lu2026relational]], "
                "NQ-open [[cite:kwiatkowski2019natural,lee2019latent]] and "
                "Wikidata [[cite:vrandecic2014wikidata]]. Questions are "
                "labelled known or unknown from the behaviour of each "
                "model, so each model has its own question sets. Generated "
                "text is scored by SiEBERT, with an SST-2 DistilBERT "
                "classifier [[cite:sanh2019distilbert]] as a check; "
                "perplexity is measured on held-out WikiText-103 passages "
                "[[cite:merity2017pointer]].",
            ),
            (
                "Controls and baselines.",
                "The popularity of the subject, a control, is its monthly "
                "Wikipedia page views [[cite:mallen2023when]]. The "
                "confidence of the model, a baseline, stacks five output "
                "signals: the probability of the most likely next token "
                "after the prompt, the negative entropy of that distribution, "
                "the mean log-probability of the answer tokens, the share of "
                "sampled answers that equal the greedy answer, and P(True) "
                "[[cite:kadavath2022language]], the probability the model "
                "assigns to its own answer being correct. Only the pooled "
                "level of the stack is stored.",
            ),
            (
                "Steering.",
                "The sentiment steering run uses the logistic sentiment "
                "direction on all 10 checkpoints, with 3 steering vectors "
                "and the relative coefficients " + MINUS + "0.3 to 0.3, "
                "and compares steering with a positive or negative "
                "instruction. Steered text is fluent only when the "
                "perplexity of 200 held-out WikiText-103 passages under "
                "the steered model is at most "
                + MAX_RATIO
                + " times that under the "
                "unsteered model and the distinct-1 of the completions is "
                "at least "
                + MIN_DISTINCT
                + ". [[table:steering]] also gives the "
                "perplexity of the completions themselves under the "
                "unsteered model. The steering comparison of finders "
                "follows the manipulation test of the paper on ETHICS "
                "utility: for the same prompt and seed, the completion "
                "steered at +0.3 is compared with the one at "
                + MINUS
                + "0.3, the sentiment judge decides which is more "
                "positive, and the win rate is the share of decisive pairs "
                "won by the +0.3 completion. Knowledge steering adds the "
                "known-against-unknown direction while the model answers; "
                "base checkpoints cannot abstain, so their outcome is the "
                "shift of the entropy of the next-token distribution, and "
                "post-trained checkpoints are asked to say when they do "
                "not know, so their outcome is the abstention rate, "
                "detected by phrase matching. The run of published steering "
                "methods steers toward sentiment, happiness and anger at "
                "three layers, with at most "
                "[[num:table:steering_baselines.new_tokens]] new tokens per "
                "prompt; an LLM judge scores each criterion on "
                "a three-point scale, the overall score of a generation is "
                "the harmonic mean of its criteria, and the interval of the "
                "mean overall score of an arm is a percentile bootstrap over "
                "the held-out prompts with "
                "[[num:table:steering_baselines.resamples]] resamples.",
            ),
            (
                "Intervals.",
                "The intervals stored by the reading runs are percentile "
                "bootstrap intervals over the 10 readers; they do not "
                "resample the test pairs and are narrower than the "
                "sampling of the test pairs. The interval for an accuracy "
                "on n test pairs is Wilson, and for a difference between "
                "two accuracies it is d " + PM + " 1.96 times the square "
                "root of p1 (1 "
                + MINUS
                + " p1) / n1 + p2 (1 "
                + MINUS
                + " p2) / n2, which treats the two arms as independent "
                "and is conservative because they share the test items. "
                "AUROC intervals use the numbers of positive and negative "
                "test items; increments in error prediction keep their "
                "stored paired bootstrap over test questions; win rates "
                "keep their Wilson interval on decisive pairs; slopes over "
                "the four GPT-2 sizes are least-squares slopes with a t "
                "interval on 2 degrees of freedom.",
            ),
            (
                "Design: reading concepts.",
                "The question is whether one direction in the residual "
                "stream separates the two sides of a concept on pairs it was "
                "not fitted on. A direction is oriented with the training "
                "labels, so even a random one can score above chance; the "
                "evidence is therefore the margin over the random direction "
                "of the same run and, where the words can give the label "
                "away, the distance to classifiers that see only the text. "
                "The sentiment, utility and morality runs each fit "
                "[[num:table:design.utility_n_seeds]] readers on "
                "[[num:table:design.utility_n_pairs]] training pairs and "
                "score held-out pairs. Three further runs vary the number of "
                "training pairs, the token that is read and the wording of "
                "the template. The token and wording runs keep the items and "
                "the draws fixed, so a change belongs to the factor that was "
                "varied, and the training-set size run draws "
                "[[num:table:design.training_size_n_seeds]] training sets per "
                "size. The utility "
                "run covers [[count:table:design.utility]]. The limits are "
                "that the stored intervals resample the draws and not the "
                "test pairs, that the main sentiment run chooses its layer "
                "on the pairs it scores, and that each cell is one run.",
            ),
            (
                "Design: finders.",
                "The question is whether the finder of the paper finds a "
                "concept as well as other ways of turning the same pairs "
                "into a direction. One run varies the pairing, the "
                "preprocessing of the differences and the finder one at a "
                "time around the finder of the paper, and reads every "
                "series, the random direction included, at the layer chosen "
                "on its own select split, so no series borrows a layer tuned "
                "for another. A second run asks whether five finders that "
                "read utility also steer it, with "
                "[[num:table:design.finders_steering_n_steer_seeds]] "
                "steering vectors per finder and a random finder as the "
                "reference for the win rate. A third sets the finder of the "
                "paper beside four published finders fitted on the same "
                "[[num:table:design.published_finders_n_pairs]] pairs, on "
                "[[count:table:design.published_finders]]. Logistic "
                "regression and the mean difference are published baselines; "
                "the random direction, the shuffled labels and the text "
                "classifiers are controls. The limit is that matched pairs "
                "exist only for utility, so the effect of matching cannot be "
                "told apart from the concept and the data.",
            ),
            (
                "Design: known and unknown questions.",
                "The question is whether a model carries a direction that "
                "separates the questions it answers from those it cannot, "
                "and whether that direction says more than the output "
                "already does. The labels come from the behaviour of each "
                "model: a question is known when the greedy answer and at "
                "least "
                "[[num:table:design.labels_recipe_known_min_correct]] "
                "of [[num:table:design.labels_n_samples]] sampled "
                "answers are correct, unknown when none is, and left out "
                "otherwise, so every model has its own sets. Datasets differ "
                "in wording and relations differ in how often they are "
                "known, so a reader is set beside classifiers on the words "
                "and the characters of the question and beside the "
                "popularity of the subject, and the headline is the AUROC "
                "inside each relation. Error prediction compares the "
                "direction with the confidence of the model and with "
                "published detectors on the same test questions; steering "
                "compares it with "
                "[[num:table:design.knowledge_steering_n_random]] random "
                "directions of the same norm, on "
                "[[count:table:design.knowledge_steering]]. The views run "
                "reads the train and test questions of the reading run again "
                "at every layer that run read, refits the readers and "
                "compares their AUROC with the stored scores, within a "
                "tolerance of [[num:table:knowledge_projections.tolerance]] "
                "set before the run; its controls are a random direction, "
                "readers refitted on labels permuted inside each group, a "
                "view of the text alone and the popularity of the subject, "
                "and it covers [[count:table:design.knowledge_views]]. The "
                "limits are that no human checked the answer matching or the "
                "abstention phrases, and that the interval of a reader "
                "resamples its "
                "draws, not the test questions.",
            ),
            (
                "Design: approximate answers.",
                "The question is whether a model treats a quantity whose "
                "sourced answer is a range differently from one with a "
                "single exact value, in what it says and in its hidden "
                "states. The label is the kind of the sourced answer, taken "
                "from the bounds of the Wikidata statement and not from a "
                "model, with "
                "[[num:table:design.approximate_answers_n_per_side]] "
                "questions per side in each of three families. In every "
                "family the range side is a living thing and the exact side "
                "an artifact, and the wording differs by side, so "
                "[[num:table:design.approximate_reading_n_people]] people "
                "with one exact height serve as the control: a reader of "
                "range against exact should score them like the buildings, "
                "and a reader of living against artifact like the dog "
                "breeds. Readers are also tested on a family held out of "
                "training, beside classifiers on the text, the length of the "
                "question and the popularity of the subject. The links to "
                "the knowledge directions are compared with the band of "
                "[[num:table:design.approximate_links_n_random]] random "
                "directions, never with chance. The limits are that the "
                "families are few and that no human checked the parsing of "
                "the answers.",
            ),
            (
                "Design: truth.",
                "The question is whether a reader of truth reads the model "
                "or the wording. A direction fitted on true and false "
                "statements is scored on held-out statements and, with no "
                "refit, on sets of statements from another source, which "
                "tests whether it holds beyond its own data. On TruthfulQA, "
                "split by hash into [[num:table:design.truthfulqa_n_train]] "
                "train, [[num:table:design.truthfulqa_n_val]] validation and "
                "[[num:table:design.truthfulqa_n_test]] test questions, the "
                "readers are compared with the two zero-shot scorers of the "
                "paper, which are published baselines; the classifiers on "
                "the words and the characters of the text are controls, and "
                "a reader that is not above the words alone has not been "
                "shown to read the model. The honesty run fits its "
                "direction on persona prompts and tests it on "
                "[[num:table:design.honesty_n_eval_pairs_transfer_test]] "
                "pairs of plain statements, so it asks for transfer from a "
                "function to a concept. The limits are that the stimuli the "
                "paper generates with a larger chat model are not reproduced "
                "and that the test sets are small.",
            ),
            (
                "Design: steering.",
                "The question is whether adding a direction to the residual "
                "stream moves the text toward the concept without breaking "
                "it. The sentiment run steers completions of "
                "[[num:table:design.sentiment_steering_n_prompts]] neutral "
                "prompts over a grid of relative coefficients; a layer sweep "
                "on one list of prompts chooses where to steer and the "
                "results come from a second list, so the layer is not tuned "
                "on the reported text. A classifier scores every completion, "
                "with a second classifier as a check, and the fluency rule, "
                "on the perplexity of held-out text and the distinct-1 of "
                "the completions, guards against a shift bought with "
                "degenerate text. This run has no random direction: its "
                "comparisons are the unsteered model and an instruction in "
                "the prompt, so it can show whether the direction moves the "
                "judged sentiment, not whether only this direction would. The "
                "utility test and the knowledge steering run steer random "
                "directions of the same norm, and the run of published "
                "steering methods steers a random unit direction like its "
                "vector arms. That run sets the reader beside five published "
                "steering methods as baselines: prompt prepending, DiffMean, "
                "CAA, ActAdd and mean-centring. Its controls are the "
                "unsteered model and that random direction, and it covers "
                "[[count:table:design.steering_baselines]]. Each arm "
                "chooses its coefficient on "
                "[[num:table:design.steering_baselines_n_tune]] tuning "
                "prompts per concept, and an LLM judge rates the concept and "
                "the fluency of its generations, and on the post-trained "
                "checkpoint how well they follow the instruction, on "
                "[[num:table:design.steering_baselines_n_held_out]] held-out "
                "prompts, so the coefficient is not tuned on the reported "
                "text either. The limits are that no human validated the "
                "judges, neither the classifiers of the sentiment run nor the "
                "LLM judge, whose two slots are the same model, and that one "
                "run stands behind each cell.",
            ),
            (
                "Design: post-training.",
                "The question is what post-training changes in reading and "
                "in steering. Each Qwen3 size exists as a base checkpoint "
                "and as a post-trained one, so the comparison holds the "
                "architecture and the size fixed. The post-trained "
                "checkpoint is read through its chat template with thinking "
                "disabled, so the prompt format changes together with the "
                "weights; the wording run varies the format on its own. The "
                "paired comparison runs no model: it reads the finished runs "
                "of both checkpoints of a pair and, because one draw uses "
                "the same stimuli on both, takes the difference per draw "
                "with a bootstrap of "
                "[[num:table:design.post_training_n_boot]] resamples over "
                "the draws. That stored run covers "
                "[[count:table:design.post_training]]. The limits are that "
                "the pairs come from one model family and that the intervals "
                "do not resample the test pairs.",
            ),
        ),
        items=("table:design",),
    ),
    Section(
        "appendix_reading",
        "Reading details",
        paragraphs=(
            "These support the finding that a logistic direction reads "
            "utility and morality on every checkpoint and truth on Qwen3 "
            "only: the margin by depth and role "
            "([[fig:a_reading_curves]]), the layer curves of every "
            "reading ([[fig:a_layers]]), "
            "every finder ([[fig:a_finders_gpt2]], "
            "[[fig:a_finders_qwen3]], [[table:finders_all]]), every "
            "reading at its layer ([[table:reading_full]]) and the "
            "sample-size and decoy runs ([[table:sizes]], "
            "[[table:prompts]]).",
            "Comparisons of each row of [[fig:teaser]]: rows 1 to 5, the "
            "random direction of the same run at the same layer; row 6, the "
            "random direction of the finder comparison; row 7, a word "
            "classifier; row 8, the strongest of popularity and two text "
            "classifiers; row 9, the model's own confidence, five stacked "
            "output signals; row 10, steering against random directions of "
            "the same norm. Rows 1 to 8 and row 10 are controls and row 9 is "
            "a baseline, the confidence of the model, whose five signals "
            "include P(True) [[cite:kadavath2022language]]. A direction for "
            "questions the model can answer is above that confidence, with "
            "an interval excluding zero, on "
            "[[count:fig:teaser.knowledge_confidence]]. For steering on the "
            "utility manipulation task of the paper, a checkpoint passes "
            "when every logistic steering vector (3 per checkpoint) of the "
            "utility direction beats every random direction of the same "
            "norm (3 per checkpoint) on the sentiment judge at relative "
            "coefficient 0.3, which has chance 0.05: "
            "[[count:fig:teaser.steering]]. At that coefficient the steered "
            "text meets the fluency rule (perplexity at most "
            + MAX_RATIO
            + " times the "
            "unsteered text and distinct-1 at least " + MIN_DISTINCT + ") on "
            "[[count:fig:teaser.steering_fluent]] (GPT-2 small has no "
            "fluency measure). Intervals cover the sampling of the test "
            "items and treat the two arms as independent, which makes them "
            "conservative; rows 8 and 9 use the Hanley and McNeil "
            "approximation and the stored paired bootstrap.",
            "[[table:finder_baselines]] puts the finder of the paper beside "
            "four published finders as baselines, fitted on the same "
            "[[num:table:finder_baselines.seeds]] training sets of "
            "[[num:table:finder_baselines.train_pairs]] pairs, on "
            "sentiment and on three truth sets: the mass-mean probe and its "
            "whitened form [[cite:marks2024geometry]], logistic regression "
            "[[cite:alain2016understanding]] and contrast-consistent search "
            "[[cite:burns2023discovering]]. Each of the four is above the "
            "finder of the paper on at least one dataset, with a "
            "conservative interval excluding zero, on "
            "[[count:table:finder_baselines.every_baseline_above]]. The "
            "finder of the paper is above its control, the random "
            "direction, on at least one dataset, with the same interval, on "
            "[[count:table:finder_baselines.finder_above_random]].",
        ),
        items=(
            "fig:a_reading_curves",
            "fig:a_layers",
            "fig:a_finders_gpt2",
            "fig:a_finders_qwen3",
            "table:finders_all",
            "table:finder_baselines",
            "table:reading_full",
            "table:sizes",
            "table:prompts",
        ),
    ),
    Section(
        "appendix_wording",
        "Wording",
        paragraphs=(
            "These support the finding that a change of wording moves a "
            "sentiment reader by more than the noise of the test pairs on "
            "[[count:fig:wording.span_above_noise]]. The span between the "
            "best and the worst wording, the bare reader against its random "
            "direction, the cosines, the rank of the paper wording and the "
            "noise are given per role in [[fig:wording]]; the transfer "
            "between wordings is in [[fig:a_wording]] and the wordings "
            "themselves in [[table:wordings]].",
        ),
        items=("fig:wording", "fig:a_wording", "table:wordings"),
    ),
    Section(
        "appendix_truth",
        "Truth and honesty",
        paragraphs=(
            "These support the finding that a truth reader does not beat "
            "words alone on TruthfulQA: [[fig:a_truth_cities]] gives the "
            "reader on statements and their negations, [[table:truth]] "
            "the readers and the honesty transfer, and "
            "[[table:truth_sets]] the reader on all nine Geometry of "
            "Truth sets.",
        ),
        items=("fig:a_truth_cities", "table:truth", "table:truth_sets"),
    ),
    Section(
        "appendix_steering",
        "Steering details",
        paragraphs=(
            "[[table:steering]] gives the values behind the steering "
            "finding, including the perplexity of the completions.",
            "[[table:steering_baselines]] steers toward sentiment, "
            "happiness and anger along the reader and along five published "
            "steering methods as baselines, under the protocol of AxBench "
            "[[cite:wu2025axbench]] adapted to each role, on one checkpoint "
            "per role: prompt prepending, the difference of class means "
            "[[cite:marks2024geometry]], contrastive activation addition "
            "[[cite:rimsky2024steering]], activation addition "
            "[[cite:turner2023steering]] and mean-centring "
            "[[cite:jorgensen2023improving]], beside the unsteered model "
            "and a random direction as controls. An LLM judge rates each "
            "arm on [[num:table:steering_baselines.held_out]] held-out "
            "prompts per concept, at the coefficient the arm chose on "
            "[[num:table:steering_baselines.tune]] tuning prompts.",
        ),
        items=("table:steering", "table:steering_baselines"),
    ),
    Section(
        "appendix_knowledge",
        "Knowledge",
        paragraphs=(
            "These support the finding on known and unknown questions: the "
            "transfer between question types on Qwen3-4B-Base "
            "([[fig:a_knowledge_matrix]]), the values of the main figures "
            "([[table:knowledge]]) and every dose curve "
            "([[fig:a_dose_runs]], [[fig:a_dose_types]]) behind "
            "[[fig:knowledge_steering]].",
            knowledge_steering.APPENDIX_DETAIL,
            "The confidence of the model stacks five output signals, and "
            "only the pooled level of the stack is stored, so the single "
            "probability of the model's own answer token stands for it in "
            "[[fig:knowledge_reading]]; no random direction is drawn there "
            "because its within-relation level is not stored. Its intervals "
            "use the Hanley and McNeil approximation. By point estimate the "
            "direction is above the strongest of popularity and the two text "
            "classifiers on "
            "[[count:fig:knowledge_reading.above_best_point]]. A "
            "supervised probe, fitted on the correct and wrong labels of "
            "the training questions (the direction is not), adds "
            "[[num:fig:knowledge_reading.probe_inc_min_gpt2]] to "
            "[[num:fig:knowledge_reading.probe_inc_max_gpt2]] on GPT-2, "
            "[[num:fig:knowledge_reading.probe_inc_min_base]] to "
            "[[num:fig:knowledge_reading.probe_inc_max_base]] on Qwen3 "
            "base and "
            "[[num:fig:knowledge_reading.probe_inc_min_post]] to "
            "[[num:fig:knowledge_reading.probe_inc_max_post]] on "
            "post-trained checkpoints to the AUROC for predicting wrong "
            "answers, with an interval above zero on "
            "[[count:fig:knowledge_reading.probe_above]], while the "
            "direction changes it by "
            "[[num:fig:knowledge_reading.dir_inc_min_gpt2]] to "
            "[[num:fig:knowledge_reading.dir_inc_max_gpt2]], "
            "[[num:fig:knowledge_reading.dir_inc_min_base]] to "
            "[[num:fig:knowledge_reading.dir_inc_max_base]] and "
            "[[num:fig:knowledge_reading.dir_inc_min_post]] to "
            "[[num:fig:knowledge_reading.dir_inc_max_post]] on the same "
            "three groups, with the largest upper end of its interval at "
            "[[num:fig:knowledge_reading.dir_inc_hi_max_gpt2]], "
            "[[num:fig:knowledge_reading.dir_inc_hi_max_base]] and "
            "[[num:fig:knowledge_reading.dir_inc_hi_max_post]]. GPT-2 small "
            "has no "
            "known-against-unknown contrast, so it has no row in the "
            "figure; its probe adds "
            "[[num:fig:knowledge_reading.gpt2_small_probe_gain]].",
            "[[table:detection_baselines]] sets the direction beside "
            "published error detectors as baselines, on held-out PopQA "
            "questions: the exact-answer probe "
            "[[cite:orgad2025llms]], semantic entropy "
            "[[cite:farquhar2024detecting]], EigenScore "
            "[[cite:chen2024inside]], P(True) "
            "[[cite:kadavath2022language]], the probability of the model's "
            "own answer token and a probe at the end of the question. The "
            "exact-answer probe is fitted on the correctness of training "
            "questions and the direction never sees it; by point estimate "
            "that probe is above the direction on "
            "[[count:table:detection_baselines.answer_probe_above]], and, "
            "by point estimate, the direction is above P(True) and above "
            "both controls, "
            "popularity and length, on "
            "[[count:table:detection_baselines.above_ptrue_and_controls]].",
            "[[fig:a_knowledge_geometry]] shows the hidden states that the "
            "entity readers read, for one checkpoint per role, beside two "
            "controls: layer 1 and the words of the questions alone. The "
            "entity views are stored for "
            "[[count:fig:a_knowledge_geometry.with_views]]. The better of "
            "two components of the hidden states at the reader layer "
            "separates known from unknown entity questions more than the "
            "better of two components of the words on "
            "[[count:fig:a_knowledge_geometry.reader_above_words]]. It is a "
            "picture and not a test: its numbers carry no interval.",
            "[[fig:a_knowledge_projections]] sets the test questions of "
            "every drawn knowledge contrast of the same three checkpoints "
            "along the all-pairs reader, and the AUROC of each reader "
            "beside two controls: its five random directions and the same "
            "reader refitted on shuffled labels. Two entity contrasts, "
            "[[label:fig:a_knowledge_projections.undrawn]], are scored and "
            "not drawn; [[table:knowledge_projections]] holds them. On "
            "[[count:fig:a_knowledge_projections.above_every_draw]] the "
            "all-pairs reader is above every one of its random directions "
            "on every drawn contrast. The exception is "
            "[[label:fig:a_knowledge_projections.exceptions]], where the "
            "highest random direction is above the reader on "
            "[[label:fig:a_knowledge_projections.exception_contrasts]]. A "
            "random direction is oriented on the training labels like the "
            "reader, so on a small contrast a single draw can score high: "
            "the highest on the three checkpoints is "
            "[[num:fig:a_knowledge_projections.random_max]].",
            "[[table:knowledge_projections]] gives the AUROC of the reader "
            "of every contrast of the checkpoints with stored views beside "
            "its "
            "recomputation in float32 and beside its controls: the random "
            "direction and the shuffled-label reader of each draw, and the "
            "classifiers on the words, on the characters and on the "
            "popularity of the subject. The recomputed means stay within "
            "[[num:table:knowledge_projections.tolerance]] of the committed "
            "ones on [[count:table:knowledge_projections.within_tolerance]] "
            "with stored views, and a note under the table names every "
            "exception.",
        ),
        items=(
            "fig:a_knowledge_matrix",
            "table:knowledge",
            "table:detection_baselines",
            "fig:a_dose_runs",
            "fig:a_dose_types",
            "fig:a_knowledge_geometry",
            "fig:a_knowledge_projections",
            "table:knowledge_projections",
        ),
    ),
    Section(
        "appendix_approximate",
        "Approximate answers",
        paragraphs=(
            "These support the finding on approximate answers. Wikidata "
            "gives quantities whose sourced answer is a range (the height "
            "of a breed) or a single value (the height of a building); "
            "[[table:approximate]] gives the values behind "
            "[[fig:approximate]], which shows what the models answer and "
            "what a range reader reads, with the heights of people as a "
            "diagnostic. Qwen3 base writes a range on "
            "[[num:fig:approximate.range_base_min]] to "
            "[[num:fig:approximate.range_base_max]] percent of range items "
            "and post-trained on "
            "[[num:fig:approximate.range_post_min]] to "
            "[[num:fig:approximate.range_post_max]] percent; the highest "
            "share on exact items is "
            "[[num:fig:approximate.exact_base_max]] percent (base) and "
            "[[num:fig:approximate.exact_post_max]] percent "
            "(post-trained), but the range "
            "reader is above a classifier on the "
            "question words, with the interval of the difference above "
            "zero, on [[count:table:approximate.above_words]], sits at the "
            "lowest layer on [[count:table:approximate.layer_one]], and "
            "its random control falls below 0.5, a sign flip between the "
            "training and the test concepts, on "
            "[[count:fig:approximate.control_below_half]] and above 0.95, "
            "so the random direction already separates the test concepts, "
            "on [[count:fig:approximate.control_above_95]]. The control is "
            "the mean of random directions whose individual draws are not "
            "stored. The reader places the people at "
            "[[num:fig:approximate.position_gpt2_min]] to "
            "[[num:fig:approximate.position_gpt2_max]] on GPT-2, "
            "[[num:fig:approximate.position_base_min]] to "
            "[[num:fig:approximate.position_base_max]] on Qwen3 base and "
            "[[num:fig:approximate.position_post_min]] to "
            "[[num:fig:approximate.position_post_max]] on post-trained "
            "checkpoints, next to the breeds, so "
            "it does not separate a range-valued answer from a living "
            "subject; the position lies outside the random band on "
            "[[count:fig:approximate.outside]], but the random bands span "
            "[[num:fig:approximate.band_gpt2_min]] to "
            "[[num:fig:approximate.band_gpt2_max]], "
            "[[num:fig:approximate.band_base_min]] to "
            "[[num:fig:approximate.band_base_max]] and "
            "[[num:fig:approximate.band_post_min]] to "
            "[[num:fig:approximate.band_post_max]] for the same three "
            "groups, so the position cannot separate the reader from "
            "random. A random band is the range of the random control's "
            "position over its [[num:fig:approximate.n_band_sets]] "
            "training sets, and the random control of the figure and of "
            "[[table:approximate]] is chosen at its own validation layer, "
            "which differs from the layer of the reader on "
            "[[count:fig:approximate.control_other_layer]]. "
            "[[fig:a_links]] compares the range direction with the "
            "knowledge directions.",
        ),
        items=("fig:a_links", "table:approximate"),
    ),
    Section(
        "appendix_post",
        "Post-training details",
        paragraphs=(
            "[[table:base_post]] gives the stored base-against-post-trained "
            "comparisons behind the finding that post-training lowers "
            "held-out sentiment reading.",
        ),
        items=("table:base_post",),
    ),
    Section(
        "appendix_other",
        "Other experiments",
        paragraphs=(
            "Further experiments ran on the same checkpoints and are "
            "reported in the findings document of the repository. They "
            "have no figure or table on this page: the [[findings:"
            "lorra-adapter-lr1-and-the-truthfulqa-control-c4|LoRRA adapter "
            "and its TruthfulQA control]], [[findings:emotions-r4|emotion "
            "reading]] and [[findings:emotion-steering-c2|emotion "
            "steering]], the [[findings:lie-gallery-c5|lie gallery]] with "
            "honesty and lie steering, [[findings:template-steering-p7|"
            "template steering]], the [[findings:"
            "direction-agreement-and-logit-lens-on-gpt2-a1-a2|logit lens]] "
            "and [[findings:token-position-and-a-misleading-sentence-p2|"
            "token position]] of the reader. Each link opens the section "
            "of the findings document that reports the experiment; this "
            "page makes no claim about what they show. Read them as a "
            "record of what was run, not as results of this page.",
        ),
    ),
)

BODY = (
    INTRODUCTION,
    RELATED,
    METHOD,
    RESULTS,
    DISCUSSION,
    REPRODUCIBILITY,
)
