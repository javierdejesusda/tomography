"""Assembles the single-page results site, laid out like a paper."""

from collections.abc import Sequence
import html
from pathlib import Path
import re

from tomography.site.findings import ExperimentInfo
from tomography.site.findings import Finding
from tomography.site.findings import ModelInfo
from tomography.site.findings import Overview
from tomography.viz import paper

REPO = "https://github.com/javierdejesusda/tomography"
PAPER = "https://arxiv.org/abs/2310.01405"
BIB = Path(__file__).resolve().parents[3] / "docs" / "references.bib"
FONTS = (
    "https://fonts.googleapis.com/css2?family=STIX+Two+Text:"
    "ital,wght@0,400;0,600;1,400&display=swap"
)

TITLE = "Is a concept a direction inside a language model?"
SUBTITLE = (
    "Representation engineering reproduced on GPT-2 and Qwen3: what holds, "
    "what does not, and what only looks like reading."
)

PITCH = (
    "Representation engineering says a concept such as truthfulness, "
    "morality or an emotion is a direction inside a language model: the "
    "concept can be read off it, and the model can be pushed along it. This "
    "project re-implements that recipe from the paper (Zou et al., 2023) on "
    "open checkpoints and asks where it holds. Every reading score sits "
    "next to a random direction and, where the text can give the answer "
    "away, a text-only baseline."
)

VERDICTS = (
    (
        "Holds",
        (
            ("reading morality and truth in larger models", "scaling"),
            ("a small steering push that keeps text fluent", "fluency"),
        ),
    ),
    (
        "Does not hold",
        (
            ("the paper's PCA finder on GPT-2", "finders"),
            ("TruthfulQA beyond text-only baselines", "truthfulqa"),
            ("a reading that survives a rewording of the prompt", "templates"),
        ),
    ),
    (
        "Looks like reading but is not shown to be",
        (
            ("a known against unknown direction", "knowledge"),
            ("pushing along that direction, and its side effects", "steering"),
            ("range against exact answers", "approximate"),
        ),
    ),
)

STEPS = (
    (
        "Write pairs.",
        "Short prompts that differ only in the concept, such as a positive "
        "and a negative sentence.",
    ),
    (
        "Record activations.",
        "Run the model and keep the hidden state of one layer at one token.",
    ),
    (
        "Find a direction.",
        "A difference of means, the first principal component, or a logistic "
        "regression that separates the two groups.",
    ),
    (
        "Score new text.",
        "Project held-out text onto the direction. The accuracy is the "
        "reading score. A random direction picked the same way is the "
        "baseline.",
    ),
    (
        "Push, if you like.",
        "Add the direction to the hidden state while the model writes. "
        "Check the text is still fluent.",
    ),
)

REFERENCES = (
    (
        "Zou, A. et al. (2023). Representation Engineering: A Top-Down "
        "Approach to AI Transparency. arXiv:2310.01405.",
        PAPER,
    ),
    (
        "Radford, A. et al. (2019). Language Models are Unsupervised "
        "Multitask Learners. OpenAI.",
        "",
    ),
    (
        "Qwen Team (2025). Qwen3 Technical Report. arXiv:2505.09388.",
        "https://arxiv.org/abs/2505.09388",
    ),
    (
        "Lin, S., Hilton, J. and Evans, O. (2022). TruthfulQA: Measuring "
        "How Models Mimic Human Falsehoods. ACL 2022.",
        "https://arxiv.org/abs/2109.07958",
    ),
    (
        "Mallen, A. et al. (2023). When Not to Trust Language Models: "
        "Investigating Effectiveness of Parametric and Non-Parametric "
        "Memories (PopQA). ACL 2023.",
        "https://arxiv.org/abs/2212.10511",
    ),
)

FALLBACK_BIB = """\
@misc{zou2025representationengineeringtopdownapproach,
      title={Representation Engineering: A Top-Down Approach to AI \
Transparency},
      year={2025},
      eprint={2310.01405},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2310.01405},
}"""


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def bib_entry(key_prefix: str, source: Path = BIB) -> str:
    """Reads one BibTeX entry from the repository's reference list.

    Args:
        key_prefix: Start of the entry key.
        source: The `.bib` file.

    Returns:
        The entry text, or an empty string when the file or key is missing.
    """
    try:
        text = source.read_text(encoding="utf-8")
    except OSError:
        return ""
    for chunk in re.split(r"\n(?=@)", text):
        if re.match(rf"@\w+\{{{re.escape(key_prefix)}", chunk):
            return chunk.strip()
    return ""


def inline_svg(raw: bytes, label: str) -> str:
    """Prepares a matplotlib SVG for embedding in the page.

    Args:
        raw: The SVG document.
        label: Accessible name, taken from the caption.

    Returns:
        The `svg` element with an image role and a label, without the XML
        prolog or the metadata block.
    """
    text = raw.decode("utf-8")
    text = text[text.index("<svg") :]
    text = re.sub(r"<metadata>.*?</metadata>\s*", "", text, flags=re.DOTALL)
    return text.replace(
        "<svg ", f'<svg class="fig-svg" role="img" aria-label="{_e(label)}" ', 1
    )


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "?", "!")) else f"{text}."


def _figure(number: int, finding: Finding, index: int, svg: bytes) -> str:
    figure = finding.figures[index]
    takeaway = _sentence(figure.title)
    note = figure.note
    label = f"Figure {number}. {takeaway} {note}".strip()
    return (
        f'<figure id="fig{number}">{inline_svg(svg, label)}'
        f"<figcaption><strong>Figure {number}.</strong> "
        f"<strong>{_e(takeaway)}</strong> {_e(note)}</figcaption>"
        f'<p class="data">Data: <a href="data/fig{number}.csv">CSV</a> '
        f'&middot; <a href="figures/fig{number}.pdf">PDF</a></p></figure>'
    )


def _finding(
    section: int, finding: Finding, first_figure: int, svgs: Sequence[bytes]
) -> str:
    opening = " ".join([_sentence(finding.claim), *(finding.numbers[:1])])
    figures = "".join(
        _figure(first_figure + i, finding, i, svgs[first_figure + i - 1])
        for i in range(len(finding.figures))
    )
    more = " ".join(finding.numbers[1:])
    numbers = f'<p class="numbers">{_e(more)}</p>' if more else ""
    caveats = " ".join(finding.caveats)
    sources = ", ".join(f"<code>{_e(s)}</code>" for s in finding.sources)
    return (
        f'<section id="{_e(finding.key)}">'
        f"<h2>{section}&ensp;{_e(finding.title)}</h2>"
        f"<p>{_e(opening)}</p>{figures}{numbers}"
        f'<p class="caveats"><strong>Caveats.</strong> {_e(caveats)}</p>'
        f'<p class="sources">Experiments: {sources}. Every number comes '
        f'from files under <a href="{REPO}/tree/main/results">results/</a>.'
        "</p></section>"
    )


def _abstract(keys: set[str], info: Overview) -> str:
    lines = []
    for title, items in VERDICTS:
        links = [
            f'<a href="#{key}">{_e(text)}</a>'
            for text, key in items
            if key in keys
        ]
        if links:
            lines.append(f"<strong>{_e(title)}:</strong> {'; '.join(links)}.")
    summary = f'<p class="summary">{" ".join(lines)}</p>' if lines else ""
    return (
        '<section id="abstract"><h2 class="plain">Abstract</h2>'
        f"<p>{_e(PITCH)}</p>{summary}</section>"
    )


def _method() -> str:
    steps = "".join(
        f"<li><strong>{_e(a)}</strong> {_e(b)}</li>" for a, b in STEPS
    )
    return (
        '<section id="method"><h2>1&ensp;How a concept is read</h2>'
        f"<ol>{steps}</ol>"
        "<p>How to read the figures: a marker is a score, a thin line is "
        "the interval stored with it, a grey band is what a random "
        "direction scores, and an open marker is a post-trained "
        "checkpoint. Base and post-trained checkpoints are never pooled. "
        "Every figure comes with its numbers as a CSV file and as a "
        "vector PDF.</p></section>"
    )


def _model_table(models: tuple[ModelInfo, ...]) -> str:
    rows = "".join(
        f"<tr><td>{_e(paper.display_name(m.key))}</td><td>{_e(m.role)}</td>"
        f'<td class="n">{_e(m.params)}</td>'
        f'<td class="n">{"" if m.blocks is None else m.blocks}</td>'
        f'<td class="n">{m.experiments}</td></tr>'
        for m in models
    )
    head = (
        "<tr><th>Checkpoint</th><th>Kind</th>"
        '<th class="n">Parameters</th><th class="n">Blocks</th>'
        '<th class="n">Experiments run</th></tr>'
    )
    return (
        '<div class="scroll"><table>'
        f"<thead>{head}</thead><tbody>{rows}</tbody></table></div>"
    )


def _experiment_table(experiments: tuple[ExperimentInfo, ...]) -> str:
    rows = "".join(
        f"<tr><td><code>{_e(e.key)}</code></td><td>{_e(e.question)}</td>"
        f'<td class="n">{e.models}</td><td>{_e(e.status)}</td></tr>'
        for e in experiments
    )
    head = (
        "<tr><th>Id</th><th>Question</th>"
        '<th class="n">Models</th><th>Status</th></tr>'
    )
    return (
        '<div class="scroll"><table>'
        f"<thead>{head}</thead><tbody>{rows}</tbody></table></div>"
    )


def _overview(first: int, info: Overview) -> str:
    done = sum(e.status == "done" for e in info.experiments)
    pending = [e.key for e in info.experiments if e.status == "pending"]
    waiting = f" Not yet run: {', '.join(pending)}." if pending else ""
    return (
        f'<section id="models"><h2>{first}&ensp;{len(info.models)} '
        "checkpoints</h2><p>GPT-2 small to XL, and Qwen3 0.6B, 1.7B and "
        "4B each as a base and a post-trained model. The post-trained Qwen3 "
        "checkpoints are small distilled models with thinking disabled."
        f"</p>{_model_table(info.models)}</section>"
        f'<section id="experiments"><h2>{first + 1}&ensp;'
        f"{len(info.experiments)} experiments</h2>"
        f"<p>{done} are complete on every checkpoint they were planned "
        f"for.{_e(waiting)} New results appear here on the next build.</p>"
        f"{_experiment_table(info.experiments)}</section>"
    )


def _links(number: int) -> str:
    items = (
        (REPO, "Code", "Python package, tests and CI"),
        (
            f"{REPO}/blob/main/docs/findings.md",
            "Findings",
            "every result with its run identity",
        ),
        (
            f"{REPO}/blob/main/docs/methodology.md",
            "Methodology",
            "deviations from the paper and the statistics",
        ),
        (
            f"{REPO}/tree/main/results",
            "Data",
            "the committed result files behind each figure",
        ),
    )
    lines = "".join(
        f'<li><a href="{_e(url)}">{_e(name)}</a>: {_e(note)}.</li>'
        for url, name, note in items
    )
    return (
        f'<section id="links"><h2>{number}&ensp;Code and data</h2>'
        f"<ul>{lines}</ul>"
        "<pre><code>uv sync\nuv run tomography run r3 --model gpt2\n"
        "uv run python -m tomography.site build --out _site</code></pre>"
        "</section>"
    )


def _references(cite: str) -> str:
    items = "".join(
        f"<li>{_e(text)}"
        + (f' <a href="{_e(url)}">{_e(url)}</a>' if url else "")
        + "</li>"
        for text, url in REFERENCES
    )
    return (
        '<section id="references"><h2 class="plain">References</h2>'
        f"<ol>{items}</ol></section>"
        '<section id="cite"><h2 class="plain">Cite</h2>'
        "<p>This repository reimplements the method of the first paper; "
        f"please cite the original work.</p><pre><code>{_e(cite)}</code>"
        "</pre></section>"
    )


def render_page(
    findings: list[Finding], info: Overview, svgs: Sequence[bytes] = ()
) -> str:
    """Renders the whole site as one HTML document.

    Args:
        findings: Headline findings in page order.
        info: Model and experiment tables.
        svgs: One SVG document per figure, in page order.

    Returns:
        The HTML document.
    """
    keys = {f.key for f in findings}
    sections = []
    figure = 1
    for i, finding in enumerate(findings):
        sections.append(_finding(i + 2, finding, figure, svgs))
        figure += len(finding.figures)
    after = len(findings) + 2
    cite = bib_entry("zou2025") or FALLBACK_BIB
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_e(TITLE)}</title>\n"
        f'<meta name="description" content="{_e(SUBTITLE)}">\n'
        '<link rel="icon" href="data:,">\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" '
        "crossorigin>\n"
        f'<link rel="stylesheet" href="{_e(FONTS)}">\n'
        '<link rel="stylesheet" href="site.css">\n</head>\n<body>\n'
        '<main id="top"><header>'
        f"<h1>{_e(TITLE)}</h1>"
        f'<p class="subtitle">{_e(SUBTITLE)}</p>'
        '<p class="links">'
        f'<a href="{REPO}">Code</a>'
        f'<a href="{REPO}/blob/main/docs/findings.md">Findings</a>'
        f'<a href="{REPO}/blob/main/docs/methodology.md">Methodology</a>'
        f'<a href="{REPO}/tree/main/results">Data</a></p></header>'
        f"{_abstract(keys, info)}{_method()}{''.join(sections)}"
        f"{_overview(after, info)}{_links(after + 2)}{_references(cite)}"
        "</main>\n"
        "<footer><p>Every figure is drawn from committed files under "
        f'<code>results/</code> by <a href="{REPO}">the code in this '
        "repository</a>, released under the MIT license. No analytics. "
        "The typeface is loaded from Google Fonts.</p></footer>\n"
        "</body>\n</html>\n"
    )
