"""Assembles the single-page results site as an HTML string."""

import dataclasses
import html
import json
from typing import Any

from tomography.site import svg
from tomography.site.findings import ExperimentInfo
from tomography.site.findings import Figure
from tomography.site.findings import Finding
from tomography.site.findings import ModelInfo
from tomography.site.findings import Overview

REPO = "https://github.com/javierdejesusda/tomography"
PAPER = "https://arxiv.org/abs/2310.01405"

PITCH = (
    "Representation engineering says a concept such as truthfulness, "
    "morality or an emotion is a direction inside a language model: you can "
    "read the concept off it, or push the model along it. This project "
    "re-implements that recipe from the paper (Zou et al.) on open "
    "checkpoints and asks where it holds. Every reading score sits next to "
    "a random direction and, where the text can give the answer away, a "
    "text-only baseline."
)

VERDICTS = (
    (
        "holds",
        "Holds",
        (
            ("Reading morality and truth in larger models", "scaling"),
            ("A small steering push that keeps text fluent", "fluency"),
        ),
    ),
    (
        "fails",
        "Does not hold",
        (
            ("The paper's PCA finder on GPT-2", "finders"),
            ("TruthfulQA beyond text-only baselines", "truthfulqa"),
            ("A reading that survives a rewording of the prompt", "templates"),
        ),
    ),
    (
        "looks",
        "Looks like reading, is not",
        (
            ("A known against unknown direction", "knowledge"),
            ("Pushing along it, and its side effects", "steering"),
            ("Range against exact answers", "approximate"),
        ),
    ),
)

ICONS = {
    "holds": '<path d="M5 12.5l4.2 4.2L19 7" />',
    "fails": '<path d="M7 7l10 10M17 7L7 17" />',
    "looks": '<path d="M12 6.5v7M12 17.2v.3" />',
}

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

DIAGRAM = """\
<svg viewBox="0 0 320 190" role="img" aria-labelledby="dg-t dg-d">
<title id="dg-t">A direction through two groups of activations</title>
<desc id="dg-d">Two groups of points, one blue and one orange, in a plane.
A solid arrow runs between the groups; projecting the points onto it separates
them. A dashed grey arrow, a random direction, does not separate them.</desc>
<g class="a"><circle cx="60" cy="118" r="5"/><circle cx="82" cy="96" r="5"/>
<circle cx="74" cy="132" r="5"/><circle cx="102" cy="112" r="5"/>
<circle cx="92" cy="140" r="5"/><circle cx="116" cy="92" r="5"/></g>
<g class="b"><circle cx="190" cy="66" r="5"/><circle cx="214" cy="48" r="5"/>
<circle cx="204" cy="86" r="5"/><circle cx="232" cy="72" r="5"/>
<circle cx="176" cy="44" r="5"/><circle cx="246" cy="52" r="5"/></g>
<line class="rand" x1="64" y1="40" x2="150" y2="170"/>
<path class="dir" d="M40 150 L280 30"/>
<path class="dir" d="M280 30 l-12 1 M280 30 l-5 10"/>
<line class="proj" x1="60" y1="118" x2="76" y2="148"/>
<line class="proj" x1="190" y1="66" x2="150" y2="88"/>
<text x="46" y="170">group A</text>
<text x="226" y="106">group B</text>
<text x="196" y="20">learned direction</text>
<text x="92" y="30">random direction</text>
</svg>
"""


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _panel(panel: svg.DotPanel | svg.LinePanel) -> str:
    chart = (
        svg.render_line_panel(panel)
        if isinstance(panel, svg.LinePanel)
        else svg.render_dot_panel(panel)
    )
    return (
        '<div class="panel">'
        f"<h4>{_e(panel.title)}</h4>{chart}"
        '<details class="table-view"><summary>Data table</summary>'
        f'<div class="scroll">{svg.panel_table(panel)}</div>'
        f'<p><a href="data/{_e(panel.uid)}.json">Download these numbers '
        "(JSON)</a></p></details></div>"
    )


def _figure(figure: Figure) -> str:
    count = len(figure.panels)
    note = f'<p class="note">{_e(figure.note)}</p>' if figure.note else ""
    panels = "".join(_panel(p) for p in figure.panels)
    return (
        '<figure class="fig"><figcaption>'
        f"<h3>{_e(figure.title)}</h3>{note}</figcaption>{figure.legend}"
        f'<div class="panels n{count}">{panels}</div></figure>'
    )


def _finding(finding: Finding) -> str:
    numbers = "".join(f"<li>{_e(n)}</li>" for n in finding.numbers)
    caveats = "".join(f"<li>{_e(c)}</li>" for c in finding.caveats)
    sources = ", ".join(f"<code>{_e(s)}</code>" for s in finding.sources)
    return (
        f'<section class="block" id="{_e(finding.key)}">'
        f'<h2 class="title">{_e(finding.title)}</h2>'
        f'<p class="claim">{_e(finding.claim)}</p>'
        f"{''.join(_figure(f) for f in finding.figures)}"
        '<div class="read"><div class="numbers">'
        f"<h3>What the numbers say</h3><ul>{numbers}</ul></div>"
        "<div><h3>Read this with</h3>"
        f'<ul class="caveats">{caveats}</ul></div></div>'
        f'<p class="sources">Experiments: {sources}. Every number comes '
        f'from files under <a href="{REPO}/tree/main/results">results/</a>.'
        "</p></section>"
    )


def _verdicts(keys: set[str]) -> str:
    cards = []
    for tone, title, items in VERDICTS:
        lines = []
        for text, key in items:
            if key in keys:
                lines.append(f'<li><a href="#{key}">{_e(text)}</a></li>')
        if not lines:
            continue
        icon = (
            '<svg class="mark-icon tone" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-hidden="true">{ICONS[tone]}</svg>'
        )
        cards.append(
            f'<div class="verdict {tone}"><h2>{icon}{_e(title)}</h2>'
            f"<ul>{''.join(lines)}</ul></div>"
        )
    return f'<div class="verdicts">{"".join(cards)}</div>' if cards else ""


def _method() -> str:
    steps = "".join(
        f"<li><span><strong>{_e(a)}</strong> {_e(b)}</span></li>"
        for a, b in STEPS
    )
    return (
        '<section class="block" id="method"><h2 class="title">How a '
        "concept is read</h2>"
        '<div class="method"><ol class="steps">'
        f'{steps}</ol><div class="diagram">{DIAGRAM}</div></div>'
        '<p class="readme">How to read the charts: a point is a score, a '
        "thin line is the interval stored with it, a grey band is what a "
        "random direction scores, and an open marker is a post-trained "
        "checkpoint. Base and post-trained checkpoints are never pooled. "
        "Every chart has a data table and a JSON file.</p></section>"
    )


def _model_table(models: tuple[ModelInfo, ...]) -> str:
    rows = "".join(
        f"<tr><td>{_e(m.key)}</td><td>{_e(m.role)}</td><td>{_e(m.params)}</td>"
        f"<td>{'' if m.blocks is None else m.blocks}</td>"
        f"<td>{m.experiments}</td></tr>"
        for m in models
    )
    head = (
        "<tr><th>Checkpoint</th><th>Kind</th><th>Parameters</th>"
        "<th>Blocks</th><th>Experiments run</th></tr>"
    )
    return f"<table>{head}{rows}</table>"


def _experiment_table(experiments: tuple[ExperimentInfo, ...]) -> str:
    def row(e: ExperimentInfo) -> str:
        return (
            f"<tr><td><code>{_e(e.key)}</code></td><td>{_e(e.question)}</td>"
            f"<td>{e.models}</td>"
            f'<td><span class="chip {_e(e.status)}">{_e(e.status)}</span></td>'
            "</tr>"
        )

    head = "<tr><th>Id</th><th>Question</th><th>Models</th><th>Status</th></tr>"
    return f"<table>{head}{''.join(row(e) for e in experiments)}</table>"


def _overview(info: Overview) -> str:
    done = sum(e.status == "done" for e in info.experiments)
    pending = [e.key for e in info.experiments if e.status == "pending"]
    waiting = f" Not yet run: {', '.join(pending)}." if pending else ""
    return (
        '<section class="block" id="models"><h2 class="title">'
        f"{len(info.models)} checkpoints</h2>"
        '<p class="claim">GPT-2 small to XL, and Qwen3 0.6B, 1.7B and 4B '
        "each as a base and a post-trained model. The post-trained Qwen3 "
        "checkpoints are small distilled models with thinking disabled.</p>"
        f'<div class="tables">{_model_table(info.models)}</div></section>'
        '<section class="block" id="experiments"><h2 class="title">'
        f"{len(info.experiments)} experiments</h2>"
        f'<p class="claim">{done} are complete on every checkpoint they '
        f"were planned for.{_e(waiting)} New results appear here on the "
        "next build.</p>"
        '<details class="all"><summary>Show every experiment</summary>'
        f'<div class="tables">{_experiment_table(info.experiments)}</div>'
        "</details></section>"
    )


def _links() -> str:
    items = (
        (REPO, "Code", "Python package, tests and CI"),
        (
            f"{REPO}/blob/main/docs/findings.md",
            "docs/findings.md",
            "Every result with its run identity",
        ),
        (
            f"{REPO}/blob/main/docs/methodology.md",
            "docs/methodology.md",
            "Deviations from the paper and the statistics",
        ),
        (
            f"{REPO}/tree/main/results",
            "Data",
            "The committed result files behind each chart",
        ),
        (PAPER, "The paper", "Zou et al., Representation Engineering"),
    )
    cards = "".join(
        f'<a href="{_e(url)}"><strong>{_e(name)}</strong>'
        f"<small>{_e(note)}</small></a>"
        for url, name, note in items
    )
    return (
        '<section class="block" id="links"><h2 class="title">Run it '
        "yourself</h2>"
        f'<div class="links">{cards}</div>'
        "<pre><code>uv sync\nuv run tomography run r3 --model gpt2\n"
        "uv run python -m tomography.site build --out _site</code></pre>"
        "</section>"
    )


def render_page(findings: list[Finding], info: Overview) -> str:
    """Renders the whole site as one HTML document.

    Args:
        findings: Headline findings in page order.
        info: Model and experiment tables.

    Returns:
        The HTML document.
    """
    keys = {f.key for f in findings}
    nav = ['<a href="#method">Method</a>']
    if findings:
        nav.append(f'<a href="#{findings[0].key}">Findings</a>')
    nav += [
        '<a href="#models">Models</a>',
        '<a href="#experiments">Experiments</a>',
        '<a href="#links">Code and data</a>',
    ]
    body = "".join(_finding(f) for f in findings)
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Tomography results</title>\n"
        '<meta name="description" content="Representation engineering '
        "reproduced on GPT-2 and Qwen3: what holds, what does not, and "
        'what only looks like reading.">\n'
        '<link rel="icon" href="data:,">\n'
        '<link rel="stylesheet" href="site.css">\n'
        '<script src="site.js" defer></script>\n</head>\n<body>\n'
        '<header class="topbar"><div class="wrap">'
        '<a class="brand" href="#top">Tomography</a>'
        f'<nav aria-label="Sections">{"".join(nav)}</nav>'
        '<button id="theme" type="button">Dark</button></div></header>\n'
        '<main id="top"><div class="wrap">'
        '<section class="hero"><h1>Is a concept a direction inside a '
        "language model?</h1>"
        f'<p class="lede">{_e(PITCH)}</p>{_verdicts(keys)}</section>'
        f"{_method()}{body}{_overview(info)}{_links()}"
        "</div></main>\n"
        '<footer><div class="wrap"><p>Every chart is drawn from committed '
        f'files under <code>results/</code> by <a href="{REPO}">the '
        "code in this repository</a>, released under the MIT license. No "
        "analytics, no remote assets.</p></div></footer>\n"
        '<div id="tip" role="status" aria-live="polite"></div>\n'
        "</body>\n</html>\n"
    )


def panel_payload(panel: svg.DotPanel | svg.LinePanel) -> dict[str, Any]:
    """Describes a panel's numbers as JSON-ready data.

    Args:
        panel: Dot or line panel.

    Returns:
        A dict with the panel kind, title, series and rows or points.
    """
    payload: dict[str, Any] = json.loads(json.dumps(dataclasses.asdict(panel)))
    payload["kind"] = "line" if isinstance(panel, svg.LinePanel) else "dot"
    payload.setdefault("rows", [])
    payload.setdefault("series", [])
    if isinstance(panel, svg.DotPanel):
        payload["series"] = [
            {"key": k, "label": s.label} for k, s in panel.styles.items()
        ]
    return payload
