"""Assembles the results paper as one HTML page.

The page is laid out like a compiled NeurIPS paper: a title between rules, an
abstract, numbered sections with numbered cross references, figures with the
caption below, booktabs tables with the caption above, numbered citations and
a reference list. Figures and tables are numbered in order of appearance, and
every number in the text comes from a figure or table the builder computed.
"""

import dataclasses
import html
import re

from tomography.site import content
from tomography.site import references
from tomography.site.content import Section
from tomography.site.registry import FigureOutput
from tomography.site.registry import Spec
from tomography.site.registry import TableOutput

TOKEN = re.compile(r"\[\[(.+?)\]\]")


@dataclasses.dataclass(frozen=True)
class FigureEntry:
    """A built figure ready for the page.

    Attributes:
        spec: Its registry entry.
        output: The figure, caption parts and numbers.
        svg: The rendered SVG document.
    """

    spec: Spec
    output: FigureOutput
    svg: bytes


@dataclasses.dataclass(frozen=True)
class TableEntry:
    """A built table ready for the page.

    Attributes:
        spec: Its registry entry.
        output: The table and its caption parts.
    """

    spec: Spec
    output: TableOutput


@dataclasses.dataclass(frozen=True)
class Document:
    """Everything the page needs.

    Attributes:
        figures: Main figures by key, in registry order.
        tables: Tables by key, in registry order.
        appendix: Appendix figures by key, in registry order.
        date: The date line.
        bib: Parsed bibliography.
    """

    figures: dict[str, FigureEntry]
    tables: dict[str, TableEntry]
    appendix: dict[str, FigureEntry]
    date: str
    bib: dict[str, references.Entry]


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


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


class _Page:
    """State of one render: numbering, citations and footnotes."""

    def __init__(self, doc: Document) -> None:
        self.doc = doc
        self.sections: dict[str, str] = {}
        self.figure_number: dict[str, str] = {}
        self.table_number: dict[str, str] = {}
        self.cited: list[str] = []
        self.footnotes: list[str] = []
        self.pending: list[tuple[int, str]] = []
        self._number()

    def _present(self, section: Section) -> bool:
        keys = section.items
        if not keys:
            return True
        return any(self._has(k) for k in keys)

    def _has(self, key: str) -> bool:
        return key in self.doc.figures or key in self.doc.tables

    def _number(self) -> None:
        figure = table = 0

        def visit(section: Section, label: str) -> None:
            nonlocal figure, table
            self.sections[section.key] = label
            for key in section.items:
                if key in self.doc.figures:
                    figure += 1
                    self.figure_number[key] = str(figure)
                elif key in self.doc.tables:
                    table += 1
                    self.table_number[key] = str(table)
            for i, sub in enumerate(section.subsections, 1):
                visit(sub, f"{label}.{i}")

        for number, section in enumerate(self.main_sections(), 1):
            visit(section, str(number))
        for key, entry in self.doc.appendix.items():
            self.figure_number[key] = entry.spec.number
        self.sections[content.APPENDIX.key] = "A"

    def main_sections(self) -> list[Section]:
        """Lists the numbered sections of the body that have content."""
        sections = [content.INTRODUCTION, content.METHOD]
        sections += [s for s in content.RESULTS if self._present(s)]
        sections.append(content.CODE)
        return sections

    def cite(self, prefix: str) -> str:
        """Numbers a citation in order of first use and links it."""
        if prefix not in self.cited:
            self.cited.append(prefix)
        number = self.cited.index(prefix) + 1
        return f'<a class="ref" href="#ref{number}">[{number}]</a>'

    def footnote(self, text: str) -> str:
        """Adds a footnote and returns its marker."""
        self.footnotes.append(text)
        n = len(self.footnotes)
        self.pending.append((n, text))
        return (
            f'<sup class="fn"><a class="ref" id="fnref{n}" '
            f'href="#fn{n}">{n}</a></sup>'
        )

    def reference(self, name: str) -> str:
        """Resolves one `[[...]]` token."""
        if name.startswith("cite:"):
            return self.cite(name[5:])
        if name.startswith("fn:"):
            return self.footnote(name[3:])
        if name.startswith("sec:"):
            key = name[4:]
            label = self.sections[key]
            word = "Appendix" if key == content.APPENDIX.key else "Section"
            return f'<a class="ref" href="#{key}">{word}&nbsp;{label}</a>'
        if name in self.figure_number:
            return (
                f'<a class="ref" href="#{name}">Figure&nbsp;'
                f"{self.figure_number[name]}</a>"
            )
        if name in self.table_number:
            return (
                f'<a class="ref" href="#{name}">Table&nbsp;'
                f"{self.table_number[name]}</a>"
            )
        raise KeyError(name)

    def text(self, raw: str) -> str:
        """Escapes a paragraph and resolves its references."""
        out, last = [], 0
        for match in TOKEN.finditer(raw):
            out.append(_e(raw[last : match.start()]))
            out.append(self.reference(match.group(1)))
            last = match.end()
        out.append(_e(raw[last:]))
        return "".join(out)

    def paragraph(self, raw: str) -> str:
        """Renders one paragraph."""
        return f"<p>{self.text(raw)}</p>"

    def figure(self, key: str) -> str:
        """Renders a figure with its caption below."""
        entry = self.doc.figures.get(key) or self.doc.appendix[key]
        label = f"Figure {self.figure_number[key]}"
        caption = f"{entry.output.claim} {entry.output.detail}".strip()
        return (
            f'<figure id="{key}">'
            f"{inline_svg(entry.svg, f'{label}: {caption}')}"
            f"<figcaption>{label}: {_e(caption)} "
            f'<span class="data">(<a href="data/{key}.csv">data</a>, '
            f'<a href="figures/{key}.pdf">PDF</a>)</span></figcaption>'
            "</figure>"
        )

    def table(self, key: str) -> str:
        """Renders a booktabs table with its caption above."""
        out = self.doc.tables[key].output
        label = f"Table {self.table_number[key]}"
        caption = f"{out.claim} {out.detail}".strip()
        head = "".join(
            f'<th class="r">{_e(c)}</th>' if a == "r" else f"<th>{_e(c)}</th>"
            for c, a in zip(out.columns, out.align, strict=True)
        )
        body = []
        for i, row in enumerate(out.rows):
            cells = "".join(
                f'<td class="r">{_e(c)}</td>'
                if a == "r"
                else f"<td>{_e(c)}</td>"
                for c, a in zip(row, out.align, strict=True)
            )
            cls = ' class="group"' if i and row[0] else ""
            body.append(f"<tr{cls}>{cells}</tr>")
        notes = "".join(f"<p>{_e(n)}</p>" for n in out.notes)
        notes_html = f'<div class="notes">{notes}</div>' if notes else ""
        return (
            f'<div class="table" id="{key}"><div class="scroll"><table>'
            f"<caption>{label}: {_e(caption)} "
            f'<span class="data">(<a href="data/{key}.csv">data</a>)</span>'
            f"</caption><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>{notes_html}</div>"
        )

    def item(self, key: str) -> str:
        """Renders a figure or table by key, or nothing when absent."""
        if key in self.doc.figures:
            return self.figure(key)
        if key in self.doc.tables:
            return self.table(key)
        return ""

    def findings(self) -> str:
        """Lists the claim of every data figure with a cross reference."""
        items = []
        for key, entry in self.doc.figures.items():
            if not entry.output.numbers:
                continue
            items.append(
                f"<li>{_e(entry.output.claim)} ({self.reference(key)})</li>"
            )
        if not items:
            return ""
        return (
            "<p>Our findings, each stated once in its section:</p>"
            f'<ol class="findings">{"".join(items)}</ol>'
        )

    def heading(self, section: Section, level: int, number: str) -> str:
        """Renders a numbered heading."""
        tag = f"h{level}"
        return (
            f'<{tag} id="{section.key}">{number}&ensp;{_e(section.title)}'
            f"</{tag}>"
        )

    def section(self, section: Section, number: str, level: int = 2) -> str:
        """Renders a section with its subsections."""
        parts = [self.heading(section, level, number)]
        parts += [self.paragraph(p) for p in section.paragraphs]
        if section.key == content.INTRODUCTION.key:
            parts.append(self.findings())
        for head, text in section.runin:
            parts.append(
                f'<p><span class="runin">{_e(head)}</span> '
                f"{self.text(text)}</p>"
            )
        parts += [self.item(k) for k in section.items]
        parts += [self.paragraph(p) for p in section.after]
        for i, sub in enumerate(section.subsections, 1):
            parts.append(self.section(sub, f"{number}.{i}", level + 1))
        return "".join(parts)

    def notes(self) -> str:
        """Renders and clears the footnotes collected so far."""
        if not self.pending:
            return ""
        lines = "".join(
            f'<p id="fn{n}"><sup>{n}</sup> {_e(text)} '
            f'<a class="ref" href="#fnref{n}">back</a></p>'
            for n, text in self.pending
        )
        self.pending = []
        return f'<aside class="footnotes"><hr>{lines}</aside>'

    def code(self) -> str:
        """Renders the closing section on code and data."""
        links = (
            f'<a href="{content.REPO_URL}">the repository</a>, '
            f'<a href="{content.FINDINGS_URL}">the findings document</a> '
            "(every result with its run identity), "
            f'<a href="{content.METHODOLOGY_URL}">the methodology document'
            "</a> (deviations from the paper and the statistics) and "
            f'<a href="{content.DATA_URL}">the committed result files</a>'
        )
        return (
            "<p>The code, the tests and every result are in "
            f"{links}. Each figure and table of this page is rebuilt from "
            "those files by the command below, and its numbers are "
            "available as a data file.</p>"
            "<pre><code>uv sync\n"
            "uv run tomography run r3 --model gpt2\n"
            "uv run python -m tomography.site build --out _site"
            "</code></pre>"
        )

    def references(self) -> str:
        """Renders the numbered reference list in order of first citation."""
        items = []
        for n, prefix in enumerate(self.cited, 1):
            entry = references.find(self.doc.bib, prefix)
            link = (
                f' <a href="{_e(entry.url)}">{_e(entry.url)}</a>'
                if entry.url
                else ""
            )
            items.append(
                f'<li id="ref{n}">[{n}]&ensp;'
                f"{_e(references.format_line(entry))}{link}</li>"
            )
        return (
            '<section id="references"><h2>References</h2>'
            f'<ol class="refs">{"".join(items)}</ol></section>'
        )

    def cite_block(self) -> str:
        """Renders the BibTeX of this repository and of the original paper."""
        paper = references.find(self.doc.bib, "zou2025")
        year = self.doc.date[-4:] if self.doc.date[-4:].isdigit() else ""
        own = (
            "@misc{tomography,\n"
            f"  title = {{{content.TITLE}}},\n"
            f"  howpublished = {{\\url{{{content.REPO_URL}}}}},\n"
            f"  year = {{{year}}},\n"
            "}"
        )
        return (
            '<section id="cite"><h2>Cite this repository</h2>'
            "<p>This repository reimplements the method of the first "
            "reference; please cite the original work as well.</p>"
            f"<pre><code>{_e(own)}\n\n{_e(paper.raw)}</code></pre>"
            "</section>"
        )

    def appendix(self) -> str:
        """Renders the appendix with its figures."""
        if not self.doc.appendix:
            return ""
        section = content.APPENDIX
        parts = [
            f'<h2 id="{section.key}">Appendix A&ensp;{_e(section.title)}</h2>'
        ]
        parts += [self.paragraph(p) for p in section.paragraphs]
        parts += [self.figure(k) for k in self.doc.appendix]
        return "".join(parts)

    def body(self) -> str:
        """Renders the sections, the appendix and the back matter."""
        parts = []
        for i, section in enumerate(self.main_sections(), 1):
            if section.key == content.CODE.key:
                text = self.heading(section, 2, str(i)) + self.code()
            else:
                text = self.section(section, str(i))
            parts.append(text + self.notes())
        parts.append(self.appendix())
        parts.append(self.references())
        parts.append(self.cite_block())
        return "".join(parts)


def render_page(doc: Document) -> str:
    """Renders the whole site as one HTML document.

    Args:
        doc: The built figures, tables, date and bibliography.

    Returns:
        The HTML document.
    """
    page = _Page(doc)
    abstract = page.paragraph(content.ABSTRACT)
    body = page.body()
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_e(content.TITLE)}</title>\n"
        '<meta name="description" content="Representation engineering '
        "reproduced on GPT-2 and Qwen3, with a random-direction baseline "
        'for every reading.">\n'
        '<link rel="icon" href="data:,">\n'
        '<link rel="stylesheet" href="site.css">\n</head>\n<body>\n'
        '<main id="top"><header class="title">'
        f"<h1>{_e(content.TITLE)}</h1>"
        f'<p class="repo"><a href="{content.REPO_URL}">'
        f"{_e(content.REPO_TEXT)}</a></p>"
        f'<p class="date">{_e(doc.date)}</p></header>'
        f'<section class="abstract" id="abstract"><h2>Abstract</h2>'
        f"{abstract}</section>"
        f"{body}</main>\n"
        '<footer class="colophon"><p>Set in Latin Modern under the GUST '
        "Font License; the code is released under the MIT license; no "
        "analytics.</p></footer>\n"
        "</body>\n</html>\n"
    )
