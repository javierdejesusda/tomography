"""The text and structure of the results paper.

Prose is written once here. A paragraph may hold references that the page
resolves: `[[fig:finders]]` and `[[table:reading]]` become numbered links
such as "Figure 4" and "Table A4", `[[sec:finders]]` becomes "Section 4.2",
`[[cite:zou2025representationengineeringtopdownapproach]]` a numbered
citation, `[[fn:text]]` a footnote, `[[num:fig:finders.random_max]]` a number
a figure or table computed, `[[tally:fig:finders.pca_sentiment]]` its
per-role count as three fractions and `[[link:findings|the findings
document]]` a link to one of `LINKS`. Every number on the page comes from a
figure or table the builder computed from the committed results, except the
design facts of the method and its appendix, which tests compare with the run
files. A count written in words ("every checkpoint", "none") is pinned to its
computed count by a test.

The page opens with the abstract and the overview figure; the body follows,
then an unnumbered paragraph on code and data, and the appendix holds only
what the main text cites.
"""

import dataclasses

from tomography.site.cells import MINUS
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
LINKS = {
    "repository": REPO_URL,
    "findings": FINDINGS_URL,
    "methodology": METHODOLOGY_URL,
    "results": DATA_URL,
}

MAX_RATIO = f"{FLUENT_MAX_RATIO:g}"
MIN_DISTINCT = f"{FLUENT_MIN_DISTINCT:g}"
FLUENCY_RULE = (
    "neutral-text perplexity ratio at most "
    + MAX_RATIO
    + " and distinct-1 at least "
    + MIN_DISTINCT
)

ABSTRACT = (
    "We test representation engineering "
    "[[cite:zou2025representationengineeringtopdownapproach]] on four GPT-2 "
    "sizes and on three Qwen3 sizes, base and post-trained, against random "
    "directions and classifiers on the words alone, by 95 percent "
    "intervals. A direction beats its random direction far more often than "
    "the stronger comparison for its question. A logistic direction reads "
    "sentiment, utility and morality above its random direction on every "
    "checkpoint, and truth on every Qwen3 but no GPT-2 checkpoint. That "
    "paper's finder beats a random "
    "direction for sentiment on one checkpoint, and four published finders "
    "each beat it on some dataset. No TruthfulQA reader beats the words "
    "alone, and a knowledge direction beats popularity and the words alone "
    "on most Qwen3 checkpoints but not the model's confidence. Sentiment "
    "steering at relative coefficient 0.1 moves judged sentiment on most "
    "checkpoints and at coefficient 0.3 is never fluent; utility steering "
    "that beats every random direction is fluent on one. Post-training "
    "lowers held-out sentiment reading, with an interval below zero, on "
    "[[num:fig:base_post.sent_below]] of 3 Qwen3 sizes."
)

ABSTRACT_ITEMS = ("fig:teaser",)


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


INTRODUCTION = Section(
    "introduction",
    "Introduction",
    paragraphs=(
        "Representation engineering "
        "[[cite:zou2025representationengineeringtopdownapproach]] finds a "
        "direction for a concept from the activations of contrasting "
        "inputs, reads the concept by projecting activations on it, and "
        "controls the model by adding it to the residual stream. It rests "
        "on the linear representation hypothesis [[cite:park2024linear]].",
        "We run the recipe of the paper on ten small open checkpoints, "
        "where every step can be checked: GPT-2 "
        "[[cite:radford2019language]] at four sizes and Qwen3 "
        "[[cite:yang2025qwen3]] at three, each Qwen3 size as base and "
        "post-trained checkpoint. The reader is a logistic direction. Its "
        "controls are a random direction oriented like the reader, a "
        "classifier on the words alone and the subject's popularity; its "
        "baselines are published methods ([[sec:controls_method]]).",
        "One direction reads a concept above its random direction far more "
        "often than it beats the stronger comparison for the question, the "
        "words alone or the model's confidence ([[fig:teaser]]).",
        "Three fractions count GPT-2, Qwen3 base and post-trained "
        "checkpoints, in that order.",
    ),
    listed=(
        (
            "Reading ([[sec:reading]]).",
            "By the 95 percent interval, a logistic direction reads "
            "sentiment, utility and morality above its random direction on "
            "every checkpoint, and truth on [[tally:fig:teaser.truth]] "
            "([[fig:reading]]).",
        ),
        (
            "The finder of the paper ([[sec:finders]]).",
            "By interval, the leading principal component of paired "
            "differences is above a random direction for utility on every "
            "Qwen3 checkpoint but for sentiment on "
            "[[tally:fig:finders.pca_sentiment]] ([[fig:finders]]).",
        ),
        (
            "Stronger comparisons ([[sec:truth]], [[sec:known_unknown]]).",
            "No TruthfulQA reader beats the words alone by interval. A "
            "knowledge direction, fitted on questions a model answers "
            "correctly against those it does not, is above popularity and "
            "the words alone, by interval, on "
            "[[tally:fig:knowledge_reading.above_best]] but adds no "
            "distinguishable increment to the model's confidence "
            "([[fig:knowledge_reading]]).",
        ),
        (
            "Steering ([[sec:steering]]).",
            "At relative coefficient 0.1 steering moves judged sentiment on "
            "[[tally:fig:steering.moves01]]; at relative coefficient 0.3 "
            "it meets the fluency rule on none ([[fig:steering]]). "
            "Utility steering that beats every random direction meets the "
            "rule on only Qwen3-1.7B-Base.",
        ),
    ),
)

RELATED = Section(
    "related",
    "Related work",
    paragraphs=(
        "Probing reads a property with a classifier on hidden states "
        "[[cite:alain2016understanding,belinkov2022probing]]; control tasks "
        "expose a probe that learns the task instead "
        "[[cite:hewitt2019designing]], the role of our random direction and "
        "the words alone. Beyond the principal component of representation "
        "engineering "
        "[[cite:zou2025representationengineeringtopdownapproach]], several "
        "finders recover a similar sentiment direction in GPT-2 "
        "[[cite:tigges2024language]], and on a shared benchmark "
        "difference-in-means detects concepts best and prompting steers "
        "best [[cite:wu2025axbench]]. Truth is read with supervised probes "
        "[[cite:azaria2023internal]], mass-mean probes "
        "[[cite:marks2024geometry]] and contrast-consistent search "
        "[[cite:burns2023discovering]]; these can fail on negated "
        "statements [[cite:levinstein2025still]], which our negated sets "
        "test. Steering follows activation addition "
        "[[cite:turner2023steering,rimsky2024steering]] and related "
        "interventions [[cite:li2023inference,arditi2024refusal]], and "
        "steering vectors are unreliable for many inputs and concepts "
        "[[cite:tan2024analysing]]. Models can report whether they know an "
        "answer [[cite:kadavath2022language]], and answerability can be "
        "decoded from hidden states [[cite:slobodkin2023curious]]; probes "
        "of the subject predict which PopQA [[cite:mallen2023when]] facts a "
        "model knows better than popularity does "
        "[[cite:gottesman2024estimating]], and directions for known and "
        "unknown entities steer refusal [[cite:ferrando2025do]]. Prompt "
        "formatting changes model behaviour [[cite:sclar2024quantifying]], "
        "which motivates our wording test. Base and post-trained models "
        "have been compared on knowledge, truthfulness, refusal and "
        "confidence [[cite:du2025how]]; we compare only the released Qwen3 "
        "pairs.",
    ),
)

METHOD = Section(
    "method",
    "Method",
    subsections=(
        Section(
            "pipeline",
            "Models, data and readers",
            paragraphs=(
                "We study [[num:table:checkpoints.checkpoints]] checkpoints "
                "([[table:checkpoints]]). Readings use input pairs that "
                "differ in one concept: SST-2 phrases "
                "[[cite:socher2013recursive]] for sentiment, ETHICS "
                "scenarios [[cite:hendrycks2021aligning]] for utility "
                "(which scenario is more pleasant) and morality "
                "(which is acceptable), true and false statements "
                "[[cite:zou2025representationengineeringtopdownapproach]] "
                "for truth, plus the out-of-distribution Geometry of Truth "
                "sets [[cite:marks2024geometry]], some negated, and "
                "true statements under an honest and an untruthful persona "
                "for honesty, tested on plain statements (honesty transfer) "
                "([[table:assets]]). Knowledge sets PopQA questions "
                "[[cite:mallen2023when]] a checkpoint "
                "answers correctly (known) against those it does not "
                "(unknown), within each relation such as occupation.",
                "We wrap each input in the paper's template and read "
                "h(x, " + ELL + "), its residual stream at the last token "
                "after layer " + ELL + " ([[fig:method]], "
                "[[table:notation]]). A finder turns training pairs "
                "into a unit direction v(" + ELL + ") per layer, oriented "
                "on the training labels. A held-out pair is read correctly "
                "when its positive member projects higher, so no "
                "threshold is needed; chance is 0.5. Our default finder "
                "is logistic regression, chosen after pilot runs (post "
                "hoc). A run fits 10 readers, each on its "
                "own draw of 64 training pairs, and reports their mean "
                "accuracy at the "
                "layer chosen on validation pairs. The main sentiment "
                "run chooses its layer on the 256 pairs it scores, so "
                "sentiment comes from a run with a separate split of 214 "
                "pairs ([[sec:appendix_method]]).",
            ),
            items=("table:checkpoints", "fig:method"),
        ),
        Section(
            "controls_method",
            "Controls and baselines",
            paragraphs=(
                "A random direction u(" + ELL + "), oriented on the "
                "training labels like the reader, can score far above 0.5: "
                "up to [[num:fig:finders.random_max]] on "
                "[[label:fig:finders.random_max]] in the finder comparison. "
                "We therefore report each reading as a margin: the reader's "
                "accuracy minus that of the random direction of the same "
                "run, not minus 0.5. Where the wording can give the label "
                "away, we add the words alone: classifiers on the words and "
                "characters of the input, fitted on the same training "
                "items, that see only the text. For knowledge we add the "
                "popularity of the question's subject. Controls use no "
                "direction fitted for "
                "the concept; baselines are published methods run on the "
                "same items. The baselines are the finder of the paper (the "
                "leading principal component of paired differences), "
                "mass-mean probes [[cite:marks2024geometry]], "
                "contrast-consistent search [[cite:burns2023discovering]], "
                "the model's confidence, which stacks five output signals, "
                "among them P(True) [[cite:kadavath2022language]], and "
                "published error detectors and steering methods.",
            ),
        ),
        Section(
            "steering_method",
            "Steering",
            paragraphs=(
                "Steering adds c r(" + ELL + ") v(" + ELL + ") to the "
                "residual stream at three layers and every response "
                "position; r(" + ELL + ") is the mean residual norm on "
                "calibration text and c the relative coefficient. Judged "
                "sentiment is the mean SiEBERT [[cite:hartmann2023more]] "
                "positive-class probability of the completions. Steered "
                "text is fluent when it meets the fluency rule: "
                + FLUENCY_RULE
                + " [[cite:li2016diversity]]; the perplexity is that of "
                "neutral text, not of the completions. A win rate is the "
                "share of decisive prompt pairs whose completion steered at "
                "+c is judged more positive than the one at " + MINUS + "c; "
                "its reference is random directions of the same norm, not "
                "0.5.",
            ),
        ),
        Section(
            "intervals",
            "Intervals and counts",
            paragraphs=(
                "Each cell is one run, so no interval includes the "
                "variation between runs. Reading intervals are computed "
                "from the test items: a 95 percent Wilson interval "
                "[[cite:wilson1927probable]] for an accuracy, a published "
                "approximation [[cite:hanley1982meaning]] for an AUROC and, "
                "for a difference between two accuracies, a conservative "
                "interval that treats the two arms as independent. The one "
                "exception is [[fig:base_post]], which keeps the stored "
                "seed-paired interval where a pair has one. We count "
                "checkpoints per role (GPT-2, Qwen3 base, Qwen3 "
                "post-trained) and never pool the roles. Counts are "
                "descriptive, with no correction for the number of "
                "comparisons.",
            ),
        ),
    ),
)

RESULTS = Section(
    "results",
    "Results",
    subsections=(
        Section(
            "reading",
            "What does one direction read?",
            paragraphs=(
                "A logistic direction reads sentiment, utility and morality "
                "above its random direction on every checkpoint and truth "
                "on every Qwen3 checkpoint, by the 95 percent interval "
                "([[fig:reading]]).",
                "No GPT-2 truth reader is separable from its random "
                "direction at [[num:fig:reading.truth_n]] test pairs. "
                "Across the four GPT-2 sizes, accuracy rises per tenfold "
                "increase in parameters by "
                "[[num:fig:reading.slope_utility]] for utility (interval "
                "[[num:fig:reading.slope_utility_lo]] to "
                "[[num:fig:reading.slope_utility_hi]]) and "
                "[[num:fig:reading.slope_morality]] for morality "
                "([[num:fig:reading.slope_morality_lo]] to "
                "[[num:fig:reading.slope_morality_hi]]); the sentiment and "
                "truth intervals include zero. The words alone read "
                "sentiment ([[num:table:reading.word_sentiment]] and "
                "[[num:table:reading.char_sentiment]]) and utility "
                "([[num:table:reading.word_utility]] and "
                "[[num:table:reading.char_utility]]) above chance by "
                "interval, and not morality, truth or honesty "
                "([[table:reading]]). Across "
                "[[num:fig:wording.wordings]] wordings of the sentiment "
                "template, reader accuracy spans more than test-pair noise "
                "on [[tally:fig:wording.span_above_noise]] checkpoints, a "
                "point comparison ([[fig:wording]]). In every role the margin "
                "exceeds 0.1 at the lowest relative depth for sentiment "
                "([[fig:a_reading_curves]]).",
            ),
            items=("fig:reading",),
        ),
        Section(
            "finders",
            "Does the finder of the paper find it?",
            paragraphs=(
                "The finder of the paper is above a random direction, by "
                "interval, for sentiment on a single Qwen3 "
                "base checkpoint and for truth on "
                "[[tally:fig:finders.pca_truth]] checkpoints "
                "([[fig:finders]]).",
            ),
            items=("fig:finders",),
            after=(
                "For utility it is above on every Qwen3 checkpoint, and on "
                "GPT-2 by point estimate only (margins of "
                "[[num:fig:finders.pca_utility_gpt2_min]] to "
                "[[num:fig:finders.pca_utility_gpt2_max]]). Logistic "
                "regression is above the random direction for sentiment on "
                "every checkpoint. Only for utility does a pair hold two "
                "versions of one scenario; sentiment and truth pair a "
                "positive with any negative input, so this matching is "
                "confounded with the concept and the data. Each of four "
                "published finders is above the finder of the paper on at "
                "least one dataset, by the same interval, on "
                "[[tally:table:finder_baselines.every_baseline_above]] "
                "checkpoints ([[table:finder_baselines]]).",
            ),
        ),
        Section(
            "truth",
            "Does a truth reader beat the words alone?",
            paragraphs=(
                "A TruthfulQA reader does not beat the words alone: over "
                "[[num:fig:truth.n_questions]] questions, the interval of "
                "its multiple-choice accuracy (MC1) minus that of the "
                "better text classifier ([[num:fig:truth.text_word]] and "
                "[[num:fig:truth.text_char]]; chance "
                "[[num:fig:truth.chance]]) includes zero on every "
                "checkpoint ([[fig:truth]]).",
            ),
            items=("fig:truth",),
            after=(
                "The honesty persona reader reaches "
                "[[num:fig:truth.persona_gpt2_min]] to "
                "[[num:fig:truth.persona_gpt2_max]] on GPT-2, "
                "[[num:fig:truth.persona_base_min]] to "
                "[[num:fig:truth.persona_base_max]] on Qwen3 base and "
                "[[num:fig:truth.persona_post_min]] to "
                "[[num:fig:truth.persona_post_max]] on post-trained "
                "checkpoints. The truth reader of [[sec:reading]] reads at "
                "least one negated set below 0.5 with its interval on "
                "[[tally:fig:truth.neg_below_half]] checkpoints "
                "([[fig:a_truth_cities]]).",
            ),
        ),
        Section(
            "known_unknown",
            "Does a knowledge direction add to the model's confidence?",
            paragraphs=(
                "A knowledge direction ranks PopQA questions above "
                "popularity and the words alone on most Qwen3 checkpoints, "
                "but adds no distinguishable increment to the model's "
                "confidence.",
            ),
            runin=(
                (
                    "Reading.",
                    "Within relation, it is above the strongest "
                    "of popularity and the two text classifiers, with an "
                    "interval excluding zero, on "
                    "[[tally:fig:knowledge_reading.above_best]] "
                    "checkpoints and by point estimate on "
                    "[[tally:fig:knowledge_reading.above_best_point]] "
                    "([[fig:knowledge_reading]]); GPT-2 small has no "
                    "direction. Predicting errors, its increment over the "
                    "model's confidence has an interval covering zero on "
                    "[[tally:fig:knowledge_reading.dir_cover]], while that "
                    "of a probe fitted on correct and wrong labels lies "
                    "above zero on "
                    "[[tally:fig:knowledge_reading.probe_above]]. By point "
                    "estimate on [[num:table:detection_baselines.questions]] "
                    "held-out questions, an answer probe "
                    "[[cite:orgad2025llms]] is above the direction on "
                    "[[tally:table:detection_baselines.answer_probe_above]], "
                    "and the direction above P(True), popularity and "
                    "question length on "
                    "[[tally:table:detection_baselines."
                    "above_ptrue_and_controls]] "
                    "([[table:detection_baselines]]). On Qwen3-4B-Base the "
                    "words alone come within "
                    "[[num:fig:a_knowledge_matrix.slack]] of the reader or "
                    "beat it on [[num:fig:a_knowledge_matrix.close]] of "
                    "[[num:fig:a_knowledge_matrix.types]] question types "
                    "([[fig:a_knowledge_matrix]]). Two views of the "
                    "activations are pictures without intervals, not tests "
                    "([[fig:a_knowledge_geometry]], "
                    "[[fig:a_knowledge_projections]]).",
                ),
                (
                    "Steering.",
                    "Adding the direction while the model answers shifts "
                    "the next-token entropy (GPT-2 and base) or the "
                    "abstention rate (post-trained, two prompts each, so "
                    "six runs). By slope over the coefficient, it moves "
                    "unknown questions more than each of three random "
                    "directions on "
                    "[[tally:fig:knowledge_steering.distinguishable]] runs "
                    "and more than known questions on "
                    "[[tally:fig:knowledge_steering.selective]] runs; by the "
                    "largest shift, known questions move at least as much on "
                    "[[tally:fig:knowledge_steering.known_ge_unknown]] runs "
                    "([[fig:knowledge_steering]]).",
                ),
                (
                    "Approximate answers.",
                    "A separate test asks whether a direction reads that an "
                    "answer is a range. When the prompt ends on a range "
                    "example, Qwen3 writes a range for "
                    "[[num:fig:approximate.range_base_min]] to "
                    "[[num:fig:approximate.range_base_max]] percent (base) "
                    "and [[num:fig:approximate.range_post_min]] to "
                    "[[num:fig:approximate.range_post_max]] percent "
                    "(post-trained) of the quantities whose sourced answer "
                    "is a range. A range reader is above its random control "
                    "on "
                    "[[tally:fig:approximate.reader_above_random]] "
                    "checkpoints and above the question words on "
                    "[[tally:table:approximate.above_words]], by interval "
                    "([[fig:approximate]], [[table:approximate]]).",
                ),
            ),
            items=("fig:knowledge_reading",),
        ),
        Section(
            "steering",
            "Does steering change the output without breaking it?",
            paragraphs=(
                "At relative coefficient 0.1, steering moves judged "
                "sentiment on [[tally:fig:steering.moves01]] checkpoints "
                "and keeps the fluency rule on "
                "[[tally:fig:steering.fluent01]]; at "
                "relative coefficient 0.3 none does ([[fig:steering]]).",
            ),
            items=("fig:steering",),
            after=(
                "Sentiment steering has no random-direction arm, so it does "
                "not show that only this direction moves judged sentiment. "
                "Utility steering has one: at relative coefficient 0.3, all "
                "three vectors beat all three random directions in win rate "
                "(probability 0.05 by chance) on "
                "[[tally:fig:teaser.steering]] checkpoints ([[fig:teaser]], "
                "last row), and only Qwen3-1.7B-Base also keeps the fluency "
                "rule, a result shown only in the run's files (GPT-2 small "
                "has no fluency measure). "
                "[[table:steering_baselines]] sets five published steering "
                "methods beside the reader.",
            ),
        ),
        Section(
            "base_post",
            "What does post-training change?",
            paragraphs=(
                "Post-training lowers held-out sentiment reading by "
                "[[num:fig:base_post.sent_fall_min]] to "
                "[[num:fig:base_post.sent_fall_max]], with an interval "
                "below zero on [[num:fig:base_post.sent_below]] of 3 Qwen3 "
                "sizes ([[fig:base_post]]).",
            ),
            items=("fig:base_post",),
            after=(
                "The other reading accuracies move by at most "
                "[[num:fig:base_post.max_abs_other]], with an interval "
                "excluding zero only for utility on 1.7B "
                "([[num:fig:base_post.utility_17b]]) and 4B "
                "([[num:fig:base_post.utility_4b]]), morality on 0.6B "
                "([[num:fig:base_post.morality_06b]]) and honesty transfer "
                "on 0.6B ([[num:fig:base_post.honesty_06b]]). Of these "
                "four, only utility on 4B has a test-size interval "
                "([[sec:intervals]]); the other three are seed-paired, which "
                "does not resample the test pairs, and their test-size "
                "intervals include zero. Knowledge "
                "AUROC and steering win rate fall with an interval below "
                "zero on 1.7B only. A post-trained checkpoint differs from "
                "its base in post-training data, chat format and "
                "distillation, so each difference contrasts two recipes.",
            ),
        ),
    ),
)

LIMITATIONS = Section(
    "limitations",
    "Limitations",
    paragraphs=(
        "A reading above chance is not a reading above the stronger "
        "comparison, so a claim about a direction needs its random "
        "direction and, where the wording can carry the label, the words "
        "alone as a control ([[fig:teaser]]).",
    ),
    runin=(
        (
            "Scale and coverage.",
            "The largest model is Qwen3-4B and we cover two model families; "
            "we do not reproduce the large-model results of the paper or "
            "the stimuli it generates with a larger chat model. Most "
            "concepts use one prompt format. We did not run the published "
            "finders and detectors on the 4B pair, the published steering "
            "methods beyond one checkpoint per role, or the activation "
            "views on every checkpoint.",
        ),
        (
            "Test size.",
            "Truth and honesty are scored on [[num:fig:reading.truth_n]] "
            "test pairs and TruthfulQA on [[num:fig:truth.n_questions]] "
            "questions, so differences below about "
            "[[num:fig:teaser.truth_half_base_min]] to "
            "[[num:fig:teaser.truth_half_gpt2_max]] are not resolved and "
            "the GPT-2 truth readings are inconclusive, not negative. Layers "
            "are chosen on small validation sets, so the chosen layer carries "
            "its own uncertainty.",
        ),
        (
            "Judges and labels.",
            "Sentiment classifiers judge both sentiment and utility "
            "steering, the two slots of the LLM judge are one model, and no "
            "human validated any judge. Abstention is detected by phrase "
            "matching, which cannot tell a generic answer from an invented "
            "one. Knowledge labels are alias matches on each model's own "
            "answers with no hand check, so the question sets behind an "
            "AUROC differ between models.",
        ),
        (
            "Choices after results.",
            "No hypothesis was fixed before its run, so no result tests a "
            "prediction made in advance. The PopQA pool was widened from "
            "four to twelve relations after earlier labelling runs left too "
            "few known questions ([[table:design]]).",
        ),
    ),
)

CODE = Section(
    "code",
    "Code and data",
    paragraphs=(
        "Every number in this paper is recomputed from the committed result "
        "files when the paper is built. The code and [[link:results|the "
        "result files]] are in [[link:repository|the repository]]. "
        "[[link:findings|The findings document]] records every result with "
        "its run identity, including the experiments that have no figure "
        "here, and [[link:methodology|the methodology document]] records "
        "the deviations from the paper and the statistics.",
    ),
)

APPENDICES = (
    Section(
        "appendix_assets",
        "Assets and notation",
        paragraphs=(
            "[[table:assets]] lists every checkpoint, judge and dataset "
            "with its source and the license its source states, and "
            "[[table:notation]] defines the symbols and terms.",
        ),
        items=("table:assets", "table:notation"),
    ),
    Section(
        "appendix_method",
        "Method details",
        paragraphs=(
            "[[table:design]] gives the design of each experiment the main "
            "text uses; a cell marked post hoc records a choice made after "
            "results were seen.",
        ),
        runin=(
            (
                "Readers and draws.",
                "The honesty readers read the response tokens of the "
                "persona prompts, not the last token. [[table:design]] "
                "names the runs that fit "
                "[[num:table:design.knowledge_reading_n_seeds]], "
                "[[num:table:design.truthfulqa_n_seeds]] or "
                "[[num:table:design.training_size_n_seeds]] readers, not "
                "10.",
            ),
            (
                "Layer choice.",
                "A reader shares its layer with its random direction, "
                "except in the finder comparison, where each series is "
                "read at the layer chosen on its own select split.",
            ),
            (
                "Data and judges.",
                "The true and false statements derive from "
                "[[cite:azaria2023internal]], and the common claims of the "
                "Geometry of Truth sets from [[cite:casper2023explore]]. "
                "TruthfulQA [[cite:lin2022truthfulqa]] readers are also "
                "fitted on ARC items [[cite:clark2018think]]. The knowledge "
                "questions come from PopQA [[cite:mallen2023when]], "
                "TriviaQA [[cite:joshi2017triviaqa]], KUQ "
                "[[cite:amayuelas2024knowledge]], FalseQA "
                "[[cite:hu2023wont]], SyntHal [[cite:lu2026relational]], "
                "NQ-open [[cite:kwiatkowski2019natural,lee2019latent]] and "
                "Wikidata [[cite:vrandecic2014wikidata]]. Generated text is "
                "scored by SiEBERT, with an SST-2 DistilBERT classifier "
                "[[cite:sanh2019distilbert]] as a check, and perplexity is "
                "measured on held-out WikiText-103 passages "
                "[[cite:merity2017pointer]].",
            ),
            (
                "Controls and baselines.",
                "The popularity of a subject is its monthly Wikipedia page "
                "views [[cite:mallen2023when]]. The output signals of the "
                "model's confidence are the probability of the most likely "
                "next token after the prompt, the negative entropy of that "
                "distribution, the mean log-probability of the answer "
                "tokens, the share of sampled answers that equal the greedy "
                "answer, and P(True) [[cite:kadavath2022language]], the "
                "probability the model assigns to its own answer being "
                "correct.",
            ),
            (
                "Steering.",
                "The sentiment steering run uses the logistic sentiment "
                "direction on all 10 checkpoints, with 3 steering vectors "
                "at three consecutive layers and relative coefficients "
                "from " + MINUS + "0.3 to 0.3, and compares steering with a "
                "positive or negative instruction. The perplexity ratio of "
                "the fluency rule divides the perplexity of 200 held-out "
                "WikiText-103 passages under the steered model by that "
                "under the unsteered model; distinct-1 is the share of "
                "distinct tokens in the completions. The utility test "
                "follows the manipulation test of the paper and steers at "
                "40, 55 and 70 percent of the depth. For each prompt and "
                "seed the sentiment classifier compares the completions "
                "steered at +0.3 and " + MINUS + "0.3, and the win rate is "
                "the share of decisive pairs won by the +0.3 completion. "
                "Knowledge steering adds the knowledge direction while the "
                "model answers. Post-trained checkpoints are asked to say "
                "when they do not know, and their outcome is the abstention "
                "rate, detected by phrase matching; the others cannot "
                "abstain, and their outcome is the shift of the next-token "
                "entropy.",
            ),
            (
                "Intervals.",
                "The reading runs store percentile bootstrap intervals over "
                "their readers, which do not resample the test pairs; we "
                "recompute every reading interval from the test size "
                "([[sec:intervals]]), except the seed-paired intervals of "
                "[[fig:base_post]]. For accuracies p1 and p2 on n1 and n2 "
                "test pairs, the interval of their difference d is d "
                + PM
                + " 1.96 times the square root of p1 (1 "
                + MINUS
                + " p1) / n1 + p2 (1 "
                + MINUS
                + " p2) / n2. Increments in error prediction keep their "
                "stored paired bootstrap over test questions, and win rates "
                "their Wilson interval on decisive pairs. Slopes over the "
                "four GPT-2 sizes are least-squares slopes with a t "
                "interval on 2 degrees of freedom.",
            ),
        ),
        items=("table:design",),
    ),
    Section(
        "appendix_reading",
        "Reading details",
        paragraphs=(
            "This appendix holds the detail behind the reading, finder and "
            "truth results: every reading beside its random direction and "
            "the words alone ([[table:reading]]), the margin at every layer "
            "([[fig:a_reading_curves]]), the finder of the paper beside "
            "four published finders ([[table:finder_baselines]]), the "
            "wordings of the sentiment template ([[fig:wording]]) and the "
            "truth reader on negated statements ([[fig:a_truth_cities]]).",
        ),
        items=(
            "table:reading",
            "fig:a_reading_curves",
            "table:finder_baselines",
            "fig:wording",
            "fig:a_truth_cities",
        ),
    ),
    Section(
        "appendix_steering",
        "Published steering methods",
        paragraphs=(
            "[[table:steering_baselines]] sets the reader beside five "
            "published steering methods as baselines and two controls, on "
            "one checkpoint per role. Over the three concepts of each "
            "checkpoint, prompt prepending has the highest overall score on "
            "[[tally:table:steering_baselines.prompt_top]], and the interval "
            "of the reader lies below its own on "
            "[[tally:table:steering_baselines.reader_below]]; the "
            "[[findings:published-steering-baselines-b3|findings "
            "document]] compares every arm.",
        ),
        items=("table:steering_baselines",),
    ),
    Section(
        "appendix_knowledge",
        "Knowledge",
        paragraphs=(
            "These items support [[sec:known_unknown]]: the steering test "
            "of every run ([[fig:knowledge_steering]]), the reader beside "
            "the words alone by question type ([[fig:a_knowledge_matrix]]), "
            "the published error detectors ([[table:detection_baselines]]), "
            "two views of the activations behind the readers "
            "([[fig:a_knowledge_geometry]], "
            "[[fig:a_knowledge_projections]]) and the approximate answers "
            "([[fig:approximate]], [[table:approximate]]).",
            "Added to the model's confidence, the direction changes the "
            "AUROC for wrong answers by "
            "[[num:fig:knowledge_reading.dir_inc_min_gpt2]] to "
            "[[num:fig:knowledge_reading.dir_inc_max_gpt2]] (GPT-2), "
            "[[num:fig:knowledge_reading.dir_inc_min_base]] to "
            "[[num:fig:knowledge_reading.dir_inc_max_base]] (Qwen3 base) "
            "and [[num:fig:knowledge_reading.dir_inc_min_post]] to "
            "[[num:fig:knowledge_reading.dir_inc_max_post]] "
            "(post-trained), and the probe fitted on correct and wrong "
            "labels by [[num:fig:knowledge_reading.probe_inc_min_gpt2]] to "
            "[[num:fig:knowledge_reading.probe_inc_max_gpt2]], "
            "[[num:fig:knowledge_reading.probe_inc_min_base]] to "
            "[[num:fig:knowledge_reading.probe_inc_max_base]] and "
            "[[num:fig:knowledge_reading.probe_inc_min_post]] to "
            "[[num:fig:knowledge_reading.probe_inc_max_post]].",
            "The range reader places people, whose height is exact, next "
            "to dog breeds, whose height is a range, so it does not "
            "separate a range from a living subject. The random bands of "
            "[[table:approximate]] are wide, so that position does not "
            "separate the reader from a random direction either. Its "
            "random control is above 0.95 on "
            "[[tally:fig:approximate.control_above_95]] checkpoints, "
            "where a random direction already separates the test concepts, "
            "and below 0.5 on "
            "[[tally:fig:approximate.control_below_half]]. The reader sits "
            "at the lowest layer on "
            "[[tally:table:approximate.layer_one]] checkpoints.",
        ),
        items=(
            "fig:knowledge_steering",
            "fig:a_knowledge_matrix",
            "table:detection_baselines",
            "fig:a_knowledge_geometry",
            "fig:a_knowledge_projections",
            "fig:approximate",
            "table:approximate",
        ),
    ),
)

BODY = (
    INTRODUCTION,
    RELATED,
    METHOD,
    RESULTS,
    LIMITATIONS,
)
