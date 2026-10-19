"""The text and structure of the results paper.

Prose is written once here. A paragraph may hold references that the page
resolves to numbered links: `[[fig3]]` and `[[table2]]` become "Figure 3" and
"Table 2", `[[sec:reading]]` becomes "Section 3", `[[cite:zou2025]]` becomes
a numbered citation and `[[fn:text]]` a footnote. No paragraph states a
result: every number on the page comes from a figure or table caption that
the builder computed from the committed results.
"""

import dataclasses

TITLE = "Is a concept a direction inside a language model?"
REPO_URL = "https://github.com/javierdejesusda/tomography"
REPO_TEXT = "github.com/javierdejesusda/tomography"
FINDINGS_URL = REPO_URL + "/blob/main/docs/findings.md"
METHODOLOGY_URL = REPO_URL + "/blob/main/docs/methodology.md"
DATA_URL = REPO_URL + "/tree/main/results"

ABSTRACT = (
    "Representation engineering [[cite:zou2025]] treats a concept such as "
    "honesty, morality or an emotion as a direction in the activations of a "
    "language model: the concept can be read by projecting activations on "
    "the direction, and the model can be steered by adding it back while it "
    "writes. We re-implement the method on open checkpoints, GPT-2 from "
    "small to XL and Qwen3 from 0.6B to 4B, and ask where it holds. Every "
    "reading is shown next to a random direction and, where the wording of "
    "a question can give the answer away, next to a text-only baseline. "
    "Each finding is one sentence, one figure and one table, computed from "
    "the result files committed with the code."
)


@dataclasses.dataclass(frozen=True)
class Section:
    """A numbered section or subsection.

    Attributes:
        key: Anchor name, used by `[[sec:key]]`.
        title: Heading text.
        paragraphs: Paragraphs before the items.
        runin: Run-in headed paragraphs, as (head, text), after them.
        items: Figure and table keys placed after the paragraphs.
        after: Paragraphs after the items.
        subsections: Subsections, rendered last.
    """

    key: str
    title: str
    paragraphs: tuple[str, ...] = ()
    runin: tuple[tuple[str, str], ...] = ()
    items: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    subsections: tuple["Section", ...] = ()


INTRODUCTION = Section(
    "introduction",
    "Introduction",
    paragraphs=(
        "The central claim of representation engineering [[cite:zou2025]] is "
        "that a concept lives in a language model as a direction: a vector "
        "in the residual stream whose projection reads the concept and whose "
        "addition steers the text. We test it with the recipe of the paper "
        "on four GPT-2 sizes and on Qwen3 at three sizes, each as a base and "
        "a post-trained checkpoint.",
        "A reading score has no meaning alone, so each one is paired with "
        "the score of a random direction found by the same procedure and, "
        "where the words of a prompt can give the answer away, with a "
        "classifier on the words alone. Base and post-trained checkpoints "
        "are never pooled in a trend.[[fn:Every number in a claim is "
        "recomputed from the stored result files when the page is built, and "
        "the page is rebuilt when results change.]]",
    ),
)

METHOD = Section(
    "method",
    "Method",
    subsections=(
        Section(
            "pipeline",
            "Reading and steering",
            paragraphs=(
                "[[fig1]] shows the pipeline. Pairs of stimuli that differ in "
                "the concept go through the model, and the residual stream "
                "at one layer and the last token is recorded. A direction is "
                "fitted by one of three finders: the first principal "
                "component of the differences between paired activations, "
                "which is the finder of the paper, the difference of the "
                "class means, or a logistic regression. A direction reads a "
                "concept by scoring held-out text with its projection, and "
                "its accuracy is the share of held-out items scored "
                "correctly. It steers by adding the direction to the "
                "residual stream while the model writes, with a coefficient "
                "proportional to the mean residual norm of the layer.",
            ),
            items=("fig1",),
        ),
        Section(
            "concepts",
            "Concepts and data",
            paragraphs=(
                "Sentiment is read on phrases of the Stanford Sentiment "
                "Treebank [[cite:socher2013]], and utility and morality on "
                "the ETHICS dataset [[cite:hendrycks2021]]. Truth is read on "
                "true and false statements, honesty on prompts with persona "
                "wording, and knowledge on questions that the model can or "
                "cannot answer, some of them from PopQA [[cite:mallen2023]]. "
                "Fitting a direction on activations to read a concept is "
                "probing; an unsupervised variant for truth is "
                "[[cite:burns2023]]. Adding a direction to the residual "
                "stream follows activation addition "
                "[[cite:turner2023steering]] "
                "and inference-time intervention [[cite:li2023inference]].",
            ),
        ),
        Section(
            "checkpoints",
            "Checkpoints",
            paragraphs=(
                "[[table1]] lists the checkpoints. GPT-2 gives four sizes of "
                "one family, and Qwen3 gives three sizes, each as a base and "
                "a post-trained model, so that size and post-training can be "
                "compared apart. Layers are placed on one axis by their "
                "relative depth, the layer index over the number of layers, "
                "and sizes by the base 10 logarithm of the parameter count.",
            ),
            items=("table1",),
        ),
        Section(
            "baselines",
            "Baselines and intervals",
            runin=(
                (
                    "Random direction.",
                    "A random unit direction is scored by the same "
                    "procedure at every layer, and a reading counts only "
                    "where it is above this baseline. Selectivity is the "
                    "accuracy of the reader minus that of the random "
                    "direction of the same run at the same layer.",
                ),
                (
                    "Shuffled labels.",
                    "A reader scored against shuffled evaluation labels "
                    "estimates the level of chance.",
                ),
                (
                    "Text-only baselines.",
                    "Where the words of a question can give the answer "
                    "away, a classifier on the words or the characters "
                    "alone is a baseline.",
                ),
                (
                    "Intervals.",
                    "Intervals are 95 percent intervals, and a caption says "
                    "which unit was resampled. An interval describes one "
                    "run and does not include variation between runs.",
                ),
            ),
        ),
    ),
)

RESULTS = (
    Section(
        "reading",
        "Reading concepts across layers and sizes",
        paragraphs=(
            "[[fig2]] reads sentiment, utility, morality and truth with a "
            "logistic reader at every layer of every checkpoint, and "
            "[[table2]] gives the accuracy at the layer chosen on "
            "validation data.",
        ),
        items=("fig2", "table2"),
    ),
    Section(
        "finders",
        "Direction finders",
        paragraphs=(
            "The paper finds directions with the first principal component "
            "of paired differences. [[fig3]] compares it with the difference "
            "of means and a logistic regression on three tasks, and "
            "[[table3]] gives the numbers for every checkpoint with this "
            "experiment.",
        ),
        items=("fig3", "table3"),
    ),
    Section(
        "wording",
        "Prompt wording",
        paragraphs=(
            "A reader is fitted and scored on prompts that share one "
            "template. [[fig4]] changes only the wording of the prompt and "
            "asks how much a reader moves.",
        ),
        items=("fig4",),
    ),
    Section(
        "knowledge",
        "Known and unknown questions",
        paragraphs=(
            "A direction for questions the model can answer against "
            "questions it cannot is fitted and compared with popularity and "
            "with the confidence of the answer ([[fig5]]), and then added to "
            "the residual stream to see whether it changes abstention "
            "([[fig6]]). [[table4]] gives the numbers.",
        ),
        items=("fig5", "fig6", "table4"),
    ),
    Section(
        "steering",
        "Steering and fluency",
        paragraphs=(
            "[[fig7]] steers sentiment along the reading direction and "
            "checks whether the text stays fluent. [[table6]] adds the "
            "other steering experiments.",
        ),
        items=("fig7", "table6"),
    ),
    Section(
        "approximate",
        "Approximate answers",
        paragraphs=(
            "Some questions have a range for an answer and some an exact "
            "value. [[fig8]] asks how often a model answers with a range and "
            "whether a direction reads the difference, and [[table5]] gives "
            "the numbers and the link to the knowledge directions.",
        ),
        items=("fig8", "table5"),
    ),
    Section(
        "limits",
        "Where reading does not beat text or noise",
        paragraphs=(
            "Some readings are matched by a classifier on the text or by a "
            "random direction. [[fig9]] collects them, and [[table7]] gives "
            "the numbers for TruthfulQA [[cite:lin2022]] and for honesty.",
        ),
        items=("fig9", "table7"),
    ),
    Section(
        "posttraining",
        "Base against post-trained",
        paragraphs=(
            "[[table8]] compares each Qwen3 base checkpoint with its "
            "post-trained counterpart.",
        ),
        items=("table8",),
    ),
)

CODE = Section(
    "code",
    "Code and data",
    paragraphs=("Everything on this page is rebuilt from the repository.",),
)

APPENDIX = Section(
    "appendix",
    "Additional figures",
    paragraphs=(
        "These figures give the data behind the figures of the main text.",
    ),
)
