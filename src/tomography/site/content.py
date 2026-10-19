"""The text and structure of the results paper.

Prose is written once here. A paragraph may hold references that the page
resolves: `[[fig:finders]]` and `[[table:reading]]` become numbered links
such as "Figure 3" and "Table 4", `[[sec:finders]]` becomes "Section 4.1",
`[[cite:zou2025representationengineeringtopdownapproach]]` a numbered
citation, `[[fn:text]]` a footnote, `[[num:fig:finders.random_max]]` a number
a figure or table computed and `[[count:fig:finders.pca_sentiment]]` its
per-role count. No paragraph states a result in its own words: every number
on the page comes from a figure or table the builder computed from the
committed results, except the design facts of the method, the limitations
and the reproducibility section, which tests compare with the run files.
"""

import dataclasses

MINUS = chr(0x2212)
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

ABSTRACT_FINDER = (
    "The paper's own finder, the first principal component of paired "
    "differences, is above a random direction for sentiment on "
    "[[count:fig:teaser.paper_finder]] when the pairs are not matched."
)
ABSTRACT_STEERING = (
    "Steering along a sentiment direction at relative coefficient 0.1 "
    "beats random directions of the same norm on "
    "[[count:fig:teaser.steering]], and the steered model stays within 1.5 "
    "times the unsteered perplexity on neutral text on "
    "[[count:fig:steering.fluent01]]."
)
ABSTRACT = (
    "Representation engineering (Zou et al., 2023) reads a concept from a "
    "language model by projecting its activations on one direction and "
    "steers the model by adding that direction to its residual stream. We "
    "run the recipe on four GPT-2 sizes and on three Qwen3 sizes, each as a "
    "base and a post-trained checkpoint, and set every reading beside a "
    "random direction of the same run and, for TruthfulQA, knowledge and "
    "approximate answers, where the wording can give the label away, beside "
    "a classifier that sees only the text. A logistic direction reads "
    "utility above its random direction on [[count:fig:teaser.utility]], "
    "morality on [[count:fig:teaser.morality]] and truth on "
    "[[count:fig:teaser.truth]] at our test size. "
    + ABSTRACT_FINDER
    + " Against stronger baselines the readings rarely win: a TruthfulQA "
    "reader is above a classifier on the question and answer words on "
    "[[count:fig:teaser.truthfulqa]], and a direction for questions the "
    "model can answer adds to the confidence of the model on "
    "[[count:fig:teaser.knowledge_confidence]]. "
    + ABSTRACT_STEERING
    + " Post-training lowers held-out sentiment reading on "
    "[[num:fig:base_post.sent_below]] of the three Qwen3 sizes; three other "
    "seed-paired differences exclude zero (0.6B morality, 1.7B utility and "
    "0.6B honesty) and the rest lie within their intervals. Every number is "
    "computed from result files in the public repository."
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
        "post-trained checkpoint. We do not reproduce the LLaMA-2 numbers of "
        "the paper. Our readers are logistic regression unless a figure "
        "names the finder of the paper, because that finder is not reliably "
        "above a random direction on unmatched pairs ([[sec:finders]]). For "
        "each question we name the baseline that could produce the result "
        "without the direction: a random direction fitted with the same "
        "procedure, a classifier on the text, the popularity of the "
        "subject, or the confidence of the model. [[fig:teaser]] summarises "
        "the answers.",
    ),
    items=("fig:teaser",),
    listed=(
        (
            "Finders ([[sec:finders]]).",
            "On pairs of unrelated inputs, the finder of the paper is above "
            "a random direction for sentiment on "
            "[[count:fig:finders.pca_sentiment]], while logistic regression "
            "is on [[count:fig:finders.logistic_sentiment]]; on matched "
            "utility pairs the finder of the paper is above it on "
            "[[count:fig:finders.pca_utility_point]] by point estimate "
            "([[fig:finders]]).",
        ),
        (
            "Where a concept is read ([[sec:reading]]).",
            "The margin over a random direction exceeds 0.1 for sentiment "
            "from a relative depth of [[num:fig:reading.onset_sentiment_min]] "
            "to [[num:fig:reading.onset_sentiment_max]] and later for "
            "utility, morality and truth; across the GPT-2 sizes "
            "accuracy grows with size for utility and morality, and the "
            "truth reader on GPT-2 is not separable from its random "
            "direction at [[num:fig:reading.truth_n]] test pairs "
            "([[fig:reading]]).",
        ),
        (
            "Wording ([[sec:wording]]).",
            "Changing the wording of the template moves a sentiment reader "
            "by up to [[num:fig:wording.span_gpt2_max]] on GPT-2, "
            "[[num:fig:wording.span_base_max]] on Qwen3 base and "
            "[[num:fig:wording.span_post_max]] on post-trained checkpoints, "
            "and a direction fitted "
            "on a bare wording reads the wording of the paper at "
            "[[num:fig:wording.bare_to_paper_min]] to "
            "[[num:fig:wording.bare_to_paper_max]] with cosine "
            "[[num:fig:wording.bare_cos_min]] to "
            "[[num:fig:wording.bare_cos_max]] to the direction of the paper "
            "([[fig:wording]]).",
        ),
        (
            "Truth beyond the text ([[sec:truth]]).",
            "A TruthfulQA reader stays within the band of classifiers that "
            "see only the words on [[count:fig:truth.tqa_inside]], and on "
            "[[count:fig:truth.neg_below_half]] at least one negated set "
            "falls below chance with its interval ([[fig:truth]]).",
        ),
        (
            "Steering ([[sec:steering]]).",
            "At relative coefficient " + PM + "0.1 steering moves the "
            "judged sentiment and keeps the fluency rule (neutral-text "
            "perplexity ratio at most 1.5 and distinct-1 at least 0.15) on "
            "[[count:fig:steering.fluent01]]; at " + PM + "0.3 on "
            "[[count:fig:steering.fluent03]] ([[fig:steering]]).",
        ),
        (
            "Knowing what one knows ([[sec:knowledge_reading]]).",
            "A known-against-unknown direction ranks PopQA questions above "
            "popularity with an interval excluding zero on "
            "[[count:fig:knowledge_reading.above_pop]], and adds to the "
            "confidence of the model by at most "
            "[[num:table:knowledge.gain_upper_max]] at the upper end of its "
            "interval on any checkpoint "
            "([[fig:knowledge_reading]]).",
        ),
        (
            "Steering with that direction ([[sec:knowledge_steering]]).",
            "It moves unknown questions more than each of three random "
            "directions on [[runs:fig:knowledge_steering.distinguishable]], "
            "and, by the largest shift of the mean dose curve, known "
            "questions move at least as much as unknown ones on "
            "[[runs:fig:knowledge_steering.known_ge_unknown]] "
            "([[fig:knowledge_steering]]).",
        ),
        (
            "Approximate answers ([[sec:approximate]]).",
            "Qwen3 writes ranges where the sourced answer is a range, but "
            "the range reader is above the 95 percent interval of a "
            "classifier on the question words on "
            "[[count:table:approximate.above_words]], sits at the lowest "
            "layer on "
            "[[count:table:approximate.layer_one]], lies inside the "
            "interval of its random control on "
            "[[count:fig:approximate.reader_in_random]] and places the "
            "people outside the band of the control on "
            "[[count:fig:approximate.people_outside_band]] "
            "([[fig:approximate]]).",
        ),
        (
            "Post-training ([[sec:base_post]]).",
            "Held-out sentiment reading is lower after post-training on "
            "[[num:fig:base_post.sent_below]] of the three Qwen3 sizes; "
            "other readings stay within their intervals on most sizes "
            "([[fig:base_post]]).",
        ),
    ),
    after=(
        "Code, result files and this page are in the repository; "
        "[[sec:reproducibility]] says how to rebuild every number.",
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
        "; our control is a random direction fitted with the same "
        "orientation rule and, where the wording can give the label away, a "
        "classifier on the text alone. Representation engineering "
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
        "[[table:checkpoints]] lists the checkpoints and [[table:notation]] "
        "the symbols and terms used below.",
    ),
    items=("table:checkpoints", "table:notation"),
    subsections=(
        Section(
            "pipeline",
            "Reading a concept",
            paragraphs=(
                "A reading experiment builds pairs of inputs that differ in "
                "the concept: two phrases of opposite sentiment, two "
                "versions of a scenario with different utility or morality, "
                "a true and a false statement. Each input is wrapped in the "
                "template of the paper for the concept, which asks for the "
                "amount of the concept in the input, and we read h(x, "
                + ELL
                + "), the residual stream at the last token of the input x "
                "after layer " + ELL + ". A finder turns the training pairs "
                "into one unit direction v(" + ELL + ") per layer; unless a "
                "figure names another finder, it is logistic regression on "
                "the paired activations. The "
                "direction is oriented with the training labels, so that "
                "the positive member of a training pair tends to project "
                "higher. A held-out pair is read correctly when its "
                "positive member projects higher than its negative member, "
                "and ties count one half; the accuracy is the share of "
                "held-out pairs read correctly, so a direction with no "
                "information scores 0.5 and no threshold is involved. A run "
                "fits 10 readers (15 for TruthfulQA and 20 for the held-out "
                "sentiment run), each on its own draw of 64 training pairs, "
                "and reports the mean of their accuracies. The honesty "
                "readers read the response tokens of the persona prompts "
                "instead of the last token of the input. "
                "[[fig:method]] shows the pipeline.",
                "The layer chosen on validation, at which a reading is "
                "reported, is picked on "
                "pairs that the reported score does not use: a validation "
                "split for utility, morality and truth, 53 validation "
                "statements for the honesty transfer, a select split for "
                "the finder comparison and the sample-size run, and "
                "validation questions for TruthfulQA. The main sentiment run "
                "is the exception: it chooses its layer on the same 256 "
                "SST-2 validation pairs it is scored on, so its value is "
                "optimistic. We therefore report sentiment from the "
                "sample-size run at 64 training pairs, which chooses the "
                "layer on its select split and scores a separate report "
                "split of 214 pairs, and keep the optimistic run in "
                "[[table:reading_full]] only.",
            ),
            items=("fig:method",),
        ),
        Section(
            "checkpoints",
            "Checkpoints",
            paragraphs=(
                "GPT-2 [[cite:radford2019language]] small, medium, large and "
                "XL, and Qwen3 [[cite:yang2025qwen3]] 0.6B, 1.7B and 4B, "
                "each as the base checkpoint and the post-trained one "
                "([[table:checkpoints]]). A post-trained checkpoint is used "
                "with its chat template and thinking disabled; it differs "
                "from its base in post-training data, in prompt format and "
                "in distillation from a larger model, so a difference "
                "between the two is a difference between these recipes and "
                "not an effect of instruction tuning alone.",
            ),
        ),
        Section(
            "data",
            "Data and judges",
            paragraphs=(
                "Sentiment is read on phrases of the Stanford Sentiment "
                "Treebank [[cite:socher2013recursive]], and utility and "
                "morality on the ETHICS dataset "
                "[[cite:hendrycks2021aligning]]. Truth is read on the true "
                "and false statements that the representation engineering "
                "repository "
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
                "Wikidata [[cite:vrandecic2014wikidata]], and approximate "
                "answers on Wikidata quantities. Questions are labelled "
                "known or unknown from the behaviour of each model, so each "
                "model has its own question sets. Generated text is scored "
                "by the positive-class probability of SiEBERT "
                "[[cite:hartmann2023more]], with an SST-2 DistilBERT "
                "classifier [[cite:sanh2019distilbert]] as a check; "
                "perplexity is measured on held-out WikiText-103 passages "
                "[[cite:merity2017pointer]]. Revisions and licenses are in "
                "[[table:assets]].",
            ),
        ),
        Section(
            "baselines",
            "Baselines",
            paragraphs=(
                "Every reading run also scores random unit directions u("
                + ELL
                + ") at "
                "every layer, oriented on the training labels exactly like "
                "the reader. Because the orientation uses the labels, a "
                "random direction can score far above 0.5 when the concept "
                "dominates the activations (up to "
                "[[num:fig:finders.random_max]] in the finder comparison), "
                "so every reading is reported as its margin over the random "
                "direction of the same run at the same layer, not over 0.5. "
                "The margin plays the role of selectivity in the probing "
                "literature [[cite:hewitt2019designing]], but its control "
                "is a random direction, not a control task. Where the "
                "wording of an input can give the label away (TruthfulQA, "
                "knowledge questions, approximate answers), we add "
                "classifiers on the words and on the characters of the "
                "input, fitted on the same training items without the "
                "model. For knowledge we add the popularity of the subject, "
                "its monthly Wikipedia page views "
                "[[cite:mallen2023when]], and the answer probability of the "
                "model. The steering comparisons use three random "
                "directions of the same norm.",
            ),
        ),
        Section(
            "steering_method",
            "Steering",
            paragraphs=(
                "Steering adds c r(" + ELL + ") v(" + ELL + ") to the "
                "residual stream at three consecutive layers (for the "
                "sentiment run only; the finder comparison steers at 40, 55 "
                "and 70 percent "
                "of the depth), at every "
                "position of the response, where v(" + ELL + ") is the unit "
                "direction, r(" + ELL + ") is the mean residual norm at "
                "layer " + ELL + " on calibration text and c is the "
                "relative coefficient. The sentiment steering run uses the "
                "logistic sentiment direction on each of the 10 "
                "checkpoints, with 3 "
                "steering vectors and the relative coefficients "
                + MINUS
                + "0.3 to 0.3, and compares steering with a positive or "
                "negative instruction. The judged score of a checkpoint is "
                "the mean SiEBERT positive-class probability of its "
                "completions. We call steered text fluent only when two "
                "conditions hold: the perplexity of 200 held-out "
                "WikiText-103 passages under the steered model is at most "
                "1.5 times that under the unsteered model, and the "
                "distinct-1 of the completions [[cite:li2016diversity]], "
                "the share of distinct tokens, is at least 0.15. "
                "[[table:steering]] also gives the perplexity of the "
                "completions themselves under the unsteered model. The "
                "steering comparison of finders follows the manipulation "
                "test of the paper on ETHICS utility: for the same prompt "
                "and seed, the completion steered at +0.3 is compared with "
                "the one at " + MINUS + "0.3, the sentiment judge decides "
                "which is more positive, and the win rate is the share of "
                "decisive pairs won by the +0.3 completion. Three random "
                "directions of the same norm are the reference for the win "
                "rate, not 0.5. Knowledge steering adds the "
                "known-against-unknown direction while the model answers; "
                "base checkpoints cannot abstain, so their outcome is the "
                "shift of the entropy of the next-token distribution, and "
                "post-trained checkpoints are asked to say when they do not "
                "know, so their outcome is the abstention rate, detected by "
                "phrase matching.",
            ),
        ),
        Section(
            "intervals",
            "Intervals and counts",
            paragraphs=(
                "The intervals stored by the reading runs are percentile "
                "bootstrap intervals over the 10 readers; they do not "
                "resample the test pairs and are narrower than the sampling "
                "of the test pairs. Every reading interval on this page is "
                "therefore computed from the test size: a 95 percent Wilson "
                "interval [[cite:wilson1927probable]] for an accuracy on n "
                "test pairs, and for a difference between two accuracies "
                "the interval d " + PM + " 1.96 times the square root of "
                "p1 (1 " + MINUS + " p1) / n1 + p2 (1 " + MINUS + " p2) / "
                "n2, which treats the two arms as independent and is "
                "conservative because they share the test items. AUROC "
                "intervals use the approximation of Hanley and McNeil "
                "[[cite:hanley1982meaning]] from the numbers of positive "
                "and negative test items; increments in error prediction "
                "keep their stored paired bootstrap over test questions; "
                "win rates keep their Wilson interval on decisive pairs; "
                "slopes over the four GPT-2 sizes are least-squares slopes "
                "with a t interval on 2 degrees of freedom. No interval "
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
                "The paper compares finders on utility, where every pair is "
                "two versions of one scenario. On sentiment and truth our "
                "pairs are unrelated inputs with opposite labels, the "
                "setting of most probing work. [[fig:finders]] shows the "
                "margin of each finder over a random direction and "
                "[[table:finders]] the values; the layer curves are in "
                "[[fig:a_finders_gpt2]] and [[fig:a_finders_qwen3]] and "
                "every finder in [[table:finders_all]].",
            ),
            items=("fig:finders", "table:finders"),
        ),
        Section(
            "reading",
            "Where and how well is a concept read?",
            paragraphs=(
                "[[fig:reading]] maps the margin over every layer and shows "
                "the held-out accuracy at the chosen layer against model "
                "size; [[table:reading]] gives the margins, "
                "[[fig:a_layers]] the raw layer curves and "
                "[[table:reading_full]] every reading with its layer. "
                "Sentiment accuracy against the number of training pairs "
                "and under a misleading sentence is in [[table:sizes]] and "
                "[[table:prompts]].",
            ),
            items=("fig:reading", "table:reading"),
        ),
        Section(
            "wording",
            "Does the wording of the template matter?",
            paragraphs=(
                "We refit the sentiment reader on wordings that change the "
                "frame, the wording or the concept word of the template of "
                "the paper ([[table:wordings]]) and score each reader on "
                "every wording ([[fig:wording]], [[fig:a_wording]]).",
            ),
            items=("fig:wording",),
        ),
        Section(
            "truth",
            "Is truth read beyond the text?",
            paragraphs=(
                "TruthfulQA can be answered in part from the answer text "
                "alone, and negation flips the label without changing the "
                "topic. [[fig:truth]] shows both tests, [[table:truth]] the "
                "readers and the honesty transfer, and [[table:truth_sets]] "
                "all nine Geometry of Truth sets.",
            ),
            items=("fig:truth", "table:truth"),
        ),
        Section(
            "steering",
            "Does steering change the output without breaking it?",
            paragraphs=(
                "[[fig:steering]] shows the judged sentiment against the "
                "relative coefficient and the fluency rule per checkpoint "
                "and coefficient; [[table:steering]] adds the completion "
                "perplexity, and the comparison with random directions is "
                "in [[fig:teaser]] (last row) and [[table:finders_all]].",
            ),
            items=("fig:steering", "table:steering"),
        ),
        Section(
            "knowledge_reading",
            "Does a direction know what the model knows?",
            paragraphs=(
                "[[fig:knowledge_reading]] compares the "
                "known-against-unknown direction with popularity, text and "
                "the confidence of the model; [[table:knowledge]] gives "
                "the values and [[fig:a_knowledge_matrix]] the transfer "
                "between question types on Qwen3-4B-Base.",
            ),
            items=("fig:knowledge_reading", "table:knowledge"),
        ),
        Section(
            "knowledge_steering",
            "Does adding that direction change answers?",
            paragraphs=(
                "[[fig:knowledge_steering]] gives the slope of each run "
                "against random directions and known questions; every dose "
                "curve is in [[fig:a_dose_runs]] and the post-trained runs "
                "by question type in [[fig:a_dose_types]].",
            ),
            items=("fig:knowledge_steering",),
        ),
        Section(
            "approximate",
            "Are approximate answers read as approximate?",
            paragraphs=(
                "Wikidata gives quantities whose sourced answer is a range "
                "(the height of a breed) or a single value (the height of a "
                "building). [[fig:approximate]] shows what the models "
                "answer and what a range reader reads, with the heights of "
                "people as a diagnostic; [[table:approximate]] gives the "
                "values and [[fig:a_links]] the comparison with the "
                "knowledge directions.",
            ),
            items=("fig:approximate", "table:approximate"),
        ),
        Section(
            "base_post",
            "What does post-training change?",
            paragraphs=(
                "[[fig:base_post]] puts every measure of the earlier "
                "sections side by side as post-trained minus base for the "
                "three Qwen3 sizes; the stored per-seed comparisons are in "
                "[[table:base_post]].",
            ),
            items=("fig:base_post",),
        ),
    ),
)

DISCUSSION = Section(
    "discussion",
    "Discussion",
    paragraphs=(
        "Two patterns run through the results. One is that a direction "
        "reads a concept better than chance far more often than it beats "
        "the right baseline: orientation on the labels already lifts a "
        "random direction, the wording of the input often carries the "
        "label, and the confidence of the model already holds what a "
        "knowledge direction reads. A reading claim therefore needs its "
        "random direction and, where the text can carry the label, a text "
        "baseline; [[fig:teaser]] shows how many of the claims survive "
        "both. The other is that the finder of the paper depends on matched "
        "pairs: it works on utility, where each pair is one scenario in two "
        "versions, and is not reliably above a random direction on "
        "unrelated pairs, where logistic regression still is. Steering "
        "moves the judged sentiment at small coefficients and breaks "
        "fluency at larger ones (a perplexity on neutral text above 1.5 "
        "times the unsteered model, or a distinct-1 below 0.15; "
        "[[sec:steering]]), and post-training lowers held-out sentiment "
        "reading and changes three other readings beyond their intervals "
        "(0.6B morality, 1.7B utility and 0.6B honesty, in "
        "[[fig:base_post]]), with the rest within them.",
    ),
)

LIMITATIONS = Section(
    "limitations",
    "Limitations",
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
            "Test size.",
            "Truth and honesty are scored on 59 test pairs and TruthfulQA "
            "on 173 questions, so differences below about 0.15 on truth are "
            "not resolved; GPT-2 truth readings are inconclusive rather "
            "than negative. Layers are selected on small validation sets, "
            "so the chosen layer carries its own uncertainty.",
        ),
        (
            "Intervals.",
            "Our intervals cover the sampling of test items only. Each cell "
            "is one run, so variation between runs, prompts or seeds of the "
            "fitting procedure is not included, and the difference "
            "intervals ignore the pairing of the arms and are conservative.",
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
            "detected by phrase matching without human labels.",
        ),
        (
            "Multiple comparisons.",
            "The page reports many comparisons across 10 checkpoints and "
            "counts how many pass; the counts are descriptive and not "
            "corrected for the number of comparisons.",
        ),
        (
            "Behavioural labels.",
            "Known and unknown questions are defined by the answers of each "
            "model, so question sets differ between models and an AUROC is "
            "not a measure of the same questions across models; the labels of "
            "knowledge questions are alias string matches with no hand "
            "check; GPT-2 small has too few known questions for a "
            "contrast.",
        ),
        (
            "Approximate answers.",
            "In every family the range side is a living thing and the exact "
            "side an artifact, the wording differs by side, test sets hold "
            "22 to 30 questions per side, and an item bootstrap interval "
            "collapses at an AUROC of 1.000; the people diagnostic also "
            "differs from the breeds in popularity.",
        ),
        (
            "Missing controls.",
            "The sentiment, utility, morality and truth readings have no "
            "text-only baseline, the sentiment steering run has no "
            "random-direction arm (the finder comparison has three), and "
            "knowledge steering has three random directions per run, so a "
            "single run passes against them by chance with probability "
            "1/4.",
        ),
    ),
)

REPRODUCIBILITY = Section(
    "reproducibility",
    "Reproducibility",
    paragraphs=(
        "Every number on this page is computed when the page is built, "
        "from the result files in the results directory of the repository "
        "at the commit in the masthead; each figure and table links the "
        "data file it draws, and the tests of the site recompute every "
        "number in a caption from the result files. Each run directory "
        "holds its configuration, its run identity (code commit and "
        "configuration hash) and its outputs. Models and datasets are "
        "pinned to the revisions in [[table:assets]]. The experiments ran "
        "on a single consumer GPU, on rented single-GPU cloud machines and "
        "on a 16-thread CPU machine. One experiment trains a model beyond "
        "the readers: a LoRRA adapter fitted on each of the 10 checkpoints. "
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
        "Assets and licenses",
        paragraphs=(
            "[[table:assets]] lists the models, judges and datasets with "
            "their pinned revisions and the licenses their sources state.",
        ),
        items=("table:assets",),
    ),
    Section(
        "appendix_reading",
        "Reading details",
        paragraphs=(
            "Layer curves of every reading ([[fig:a_layers]]), every "
            "finder ([[fig:a_finders_gpt2]], [[fig:a_finders_qwen3]], "
            "[[table:finders_all]]), every reading at its layer "
            "([[table:reading_full]]), and the sample-size and decoy runs "
            "([[table:sizes]], [[table:prompts]]).",
        ),
        items=(
            "fig:a_layers",
            "fig:a_finders_gpt2",
            "fig:a_finders_qwen3",
            "table:finders_all",
            "table:reading_full",
            "table:sizes",
            "table:prompts",
        ),
    ),
    Section(
        "appendix_wording",
        "Wording and truth details",
        paragraphs=(
            "The transfer between wordings ([[fig:a_wording]]), the "
            "wordings themselves ([[table:wordings]]) and the truth reader "
            "on every Geometry of Truth set ([[table:truth_sets]]).",
        ),
        items=("fig:a_wording", "table:wordings", "table:truth_sets"),
    ),
    Section(
        "appendix_knowledge",
        "Knowledge, approximate answers and post-training details",
        paragraphs=(
            "The transfer between question types ([[fig:a_knowledge_matrix]]"
            "), every dose curve ([[fig:a_dose_runs]], "
            "[[fig:a_dose_types]]), the range direction against the "
            "knowledge directions ([[fig:a_links]]) and the stored "
            "base-against-post-trained comparisons ([[table:base_post]]).",
        ),
        items=(
            "fig:a_knowledge_matrix",
            "fig:a_dose_runs",
            "fig:a_dose_types",
            "fig:a_links",
            "table:base_post",
        ),
    ),
)

BODY = (
    INTRODUCTION,
    RELATED,
    METHOD,
    RESULTS,
    DISCUSSION,
    LIMITATIONS,
    REPRODUCIBILITY,
)
