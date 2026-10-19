"""Assembles the results paper as one HTML page.

The page is laid out like a compiled NeurIPS paper: a title between rules, an
abstract, a contents line, numbered sections with numbered cross references,
figures and tables with their captions, numbered citations and a reference
list before the appendix. Main figures and tables are numbered in order of
appearance and appendix ones A1, A2 and so on in registry order; every number
in the text comes from a figure or table the builder computed.
"""

import dataclasses
import html
import re

from tomography.site import content
from tomography.site import references
from tomography.site import stats
from tomography.site.content import Section
from tomography.site.registry import FigureOutput
from tomography.site.registry import TableOutput
from tomography.site.tables import _cells

TOKEN = re.compile(r"\[\[(.+?)\]\]")


@dataclasses.dataclass(frozen=True)
class FigureEntry:
    """A built figure ready for the page.

    Attributes:
        key: Its registry key.
        output: The figure, caption parts and numbers.
        svg: The rendered SVG document.
        pdf: The rendered PDF document.
    """

    key: str
    output: FigureOutput
    svg: bytes
    pdf: bytes


@dataclasses.dataclass(frozen=True)
class TableEntry:
    """A built table ready for the page.

    Attributes:
        key: Its registry key.
        output: The table and its caption parts.
    """

    key: str
    output: TableOutput


@dataclasses.dataclass(frozen=True)
class Document:
    """Everything the page needs.

    Attributes:
        figures: Main figures by key, in registry order.
        tables: Tables by key, in registry order.
        appendix: Appendix figures by key, in registry order.
        appendix_tables: Appendix tables by key, in registry order.
        date: The masthead line.
        bib: Parsed bibliography.
        commit: The short commit of the results, shown as code in the
            masthead when the line names it.
    """

    figures: dict[str, FigureEntry]
    tables: dict[str, TableEntry]
    appendix: dict[str, FigureEntry]
    date: str
    bib: dict[str, references.Entry]
    appendix_tables: dict[str, TableEntry] = dataclasses.field(
        default_factory=dict
    )
    commit: str = ""


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def inline_svg(raw: bytes, prefix: str, keep_style: bool) -> str:
    """Prepares a matplotlib SVG for embedding in the page.

    Args:
        raw: The SVG document.
        prefix: The element id of the figure, such as `fig-3`; every id of
            the SVG and every reference to one gets it as a prefix, so ids
            stay unique on the page.
        keep_style: Whether to keep the `style` block; the page keeps the
            first one only.

    Returns:
        The `svg` element named by the caption of the figure, without the XML
        prolog or the metadata block.
    """
    text = raw.decode("utf-8")
    text = text[text.index("<svg") :]
    text = re.sub(r"<metadata>.*?</metadata>\s*", "", text, flags=re.DOTALL)
    if not keep_style:
        text = re.sub(r"<style\b.*?</style>\s*", "", text, flags=re.DOTALL)
    head = f"{prefix}-"
    text = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{head}{m.group(1)}"', text)
    text = re.sub(r'(href="#)([^"]+)"', lambda m: f'{m[1]}{head}{m[2]}"', text)
    text = re.sub(
        r"url\(#([^)]+)\)", lambda m: f"url(#{head}{m.group(1)})", text
    )
    return text.replace(
        "<svg ",
        f'<svg class="fig-svg" role="img" aria-labelledby="{prefix}-cap" ',
        1,
    )


class Page:
    """State of one render: numbering, citations and footnotes."""

    def __init__(self, doc: Document) -> None:
        """Numbers the items of a document.

        Args:
            doc: The built figures, tables, date and bibliography.
        """
        self.doc = doc
        self.sections: dict[str, str] = {}
        self.figure_number: dict[str, str] = {}
        self.table_number: dict[str, str] = {}
        self.cited: list[str] = []
        self.footnotes: list[str] = []
        self.pending: list[tuple[int, str]] = []
        self._styled = False
        self._number()

    def _has(self, item: str) -> bool:
        kind, _, key = item.partition(":")
        if kind == "fig":
            return key in self.doc.figures
        return key in self.doc.tables

    def _number(self) -> None:
        figure = table = 0

        def visit(section: Section, label: str) -> None:
            nonlocal figure, table
            self.sections[section.key] = label
            for item in section.items:
                kind, _, key = item.partition(":")
                if kind == "fig" and key in self.doc.figures:
                    figure += 1
                    self.figure_number[key] = str(figure)
                elif kind == "table" and key in self.doc.tables:
                    table += 1
                    self.table_number[key] = str(table)
            for i, sub in enumerate(section.subsections, 1):
                visit(sub, f"{label}.{i}")

        for number, section in enumerate(content.BODY, 1):
            visit(section, str(number))
        for letter, section in zip("ABCD", content.APPENDICES, strict=True):
            self.sections[section.key] = letter
        for i, key in enumerate(self.doc.appendix, 1):
            self.figure_number[key] = f"A{i}"
        for i, key in enumerate(self.doc.appendix_tables, 1):
            self.table_number[key] = f"A{i}"

    def cite(self, keys: str) -> str:
        """Numbers the citations of a token in order of first use."""
        links = []
        for key in keys.split(","):
            if key not in self.cited:
                self.cited.append(key)
            number = self.cited.index(key) + 1
            links.append(f'<a class="ref" href="#ref{number}">[{number}]</a>')
        return " ".join(links)

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
            word = "Appendix" if key.startswith("appendix_") else "Section"
            return f'<a class="ref" href="#{key}">{word}&nbsp;{label}</a>'
        if name.startswith("fig:"):
            number = self.figure_number[name[4:]]
            return (
                f'<a class="ref" href="#fig-{number}">Figure&nbsp;{number}</a>'
            )
        if name.startswith("table:"):
            number = self.table_number[name[6:]]
            return (
                f'<a class="ref" href="#table-{number}">Table&nbsp;{number}</a>'
            )
        if name.startswith("num:"):
            return _e(self.quoted_number(name[4:]))
        if name.startswith("count:"):
            return _e(self.quoted_count(name[6:]))
        raise KeyError(name)

    def _output(self, address: str) -> tuple[FigureOutput | TableOutput, str]:
        """Finds the output of `fig:key.name` or `table:key.name`."""
        kind, _, rest = address.partition(":")
        key, _, name = rest.rpartition(".")
        if kind == "fig":
            entry = self.doc.figures.get(key) or self.doc.appendix[key]
            return entry.output, name
        if kind == "table":
            found = self.doc.tables.get(key) or self.doc.appendix_tables[key]
            return found.output, name
        raise KeyError(address)

    def quoted_number(self, address: str) -> str:
        """Writes a number a figure or table computed."""
        output, name = self._output(address)
        value = output.numbers[name]
        if float(value).is_integer() and abs(value) >= 1:
            return f"{int(value):,}".replace("-", chr(0x2212))
        return _cells.num(value)

    def quoted_count(self, address: str) -> str:
        """Writes a per-role count a figure or table computed."""
        output, name = self._output(address)
        return stats.per_role_text(output.counts[name])

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
        appendix = key in self.doc.appendix
        entry = self.doc.appendix[key] if appendix else self.doc.figures[key]
        number = self.figure_number[key]
        label = f"Figure {number}"
        name = f"fig-{number}"
        caption = f"{entry.output.claim} {entry.output.detail}".strip()
        if appendix:
            graphic = (
                f'<img src="figures/{name}.svg" loading="lazy" '
                f'alt="{label}" aria-describedby="{name}-cap">'
            )
        else:
            graphic = inline_svg(entry.svg, name, not self._styled)
            self._styled = True
        links = []
        if entry.output.csv.strip():
            links.append(f'<a href="data/{name}.csv">data</a>')
        links.append(f'<a href="figures/{name}.pdf">PDF</a>')
        return (
            f'<figure id="{name}"><div class="fig-scroll">{graphic}</div>'
            f'<figcaption id="{name}-cap">{label}: {self.text(caption)} '
            f'<span class="data">({", ".join(links)})</span>'
            "</figcaption></figure>"
        )

    def table(self, key: str) -> str:
        """Renders a booktabs table with its caption above."""
        entry = self.doc.tables.get(key) or self.doc.appendix_tables[key]
        out = entry.output
        number = self.table_number[key]
        name = f"table-{number}"
        caption = f"{out.claim} {out.detail}".strip()
        groups = "".join(
            f'<th colspan="{span}" class="span{" muted" if muted else ""}">'
            f"{_e(label)}</th>"
            for label, span, muted in out.groups
        )
        groups = f'<tr class="groups"><th></th>{groups}</tr>' if groups else ""
        head = "".join(
            f'<th class="r">{_e(c)}</th>' if a == "r" else f"<th>{_e(c)}</th>"
            for c, a in zip(out.columns, out.align, strict=True)
        )
        body = []
        for i, row in enumerate(out.rows):
            cells = "".join(
                f'<td class="r">{self.text(c)}</td>'
                if a == "r"
                else f"<td>{self.text(c)}</td>"
                for c, a in zip(row, out.align, strict=True)
            )
            if out.row_groups is None:
                grouped = i and row[0] and not out.groups
            else:
                grouped = i in out.row_groups and i > 0
            cls = ' class="group"' if grouped else ""
            body.append(f"<tr{cls}>{cells}</tr>")
        notes = "".join(f"<p>{self.text(n)}</p>" for n in out.notes)
        notes_html = f'<div class="notes">{notes}</div>' if notes else ""
        return (
            f'<div class="table" id="{name}">'
            f'<p class="caption" id="{name}-cap">Table {number}: '
            f"{self.text(caption)} "
            f'<span class="data">(<a href="data/{name}.csv">data</a>)</span>'
            '</p><div class="scroll">'
            f'<table aria-describedby="{name}-cap">'
            f"<thead>{groups}<tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>{notes_html}</div>"
        )

    def item(self, item: str) -> str:
        """Renders a `fig:key` or `table:key` item."""
        kind, _, key = item.partition(":")
        return self.figure(key) if kind == "fig" else self.table(key)

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
        for head, text in section.runin:
            parts.append(
                f'<p><span class="runin">{self.text(head)}</span> '
                f"{self.text(text)}</p>"
            )
        parts += [self.item(k) for k in section.items]
        if section.listed:
            entries = "".join(
                f'<li><span class="runin">{self.text(head)}</span> '
                f"{self.text(text)}</li>"
                for head, text in section.listed
            )
            parts.append(f'<ol class="contributions">{entries}</ol>')
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
        """Renders the links and the build commands of the repository."""
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
            f"{links}. The page is rebuilt from those files by the commands "
            "below.</p>"
            "<pre><code>uv sync\n"
            "uv run tomography run r3 --model gpt2\n"
            "uv run python -m tomography.site build --out _site"
            "</code></pre>"
        )

    def references(self) -> str:
        """Renders the numbered reference list in order of first citation."""
        items = []
        for n, key in enumerate(self.cited, 1):
            entry = references.find(self.doc.bib, key)
            text, url = references.link(entry)
            link = f' <a href="{_e(url)}">{_e(text)}</a>' if url else ""
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
        paper = references.find(self.doc.bib, references.PAPER_KEY)
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
            "<p>This repository reimplements the method of the paper cited "
            "in the introduction; please cite the original work as "
            "well.</p>"
            f'<pre class="bibtex"><code>{_e(own)}\n\n{_e(paper.raw)}</code>'
            "</pre></section>"
        )

    def appendix(self) -> str:
        """Renders the four appendices."""
        parts = ['<section id="appendix">']
        for letter, section in zip("ABCD", content.APPENDICES, strict=True):
            parts.append(self.section(section, f"Appendix {letter}"))
        parts.append("</section>")
        return "".join(parts)

    def contents(self) -> str:
        """Renders the contents line under the abstract."""
        links = [
            f'<a href="#{s.key}">{n}&ensp;{_e(s.title)}</a>'
            for n, s in enumerate(content.BODY, 1)
        ]
        links += ['<a href="#references">References</a>']
        links += ['<a href="#appendix">Appendix</a>']
        return f'<p class="contents">Contents: {", ".join(links)}</p>'

    def body(self) -> str:
        """Renders the sections, the references and the appendix."""
        parts = []
        for i, section in enumerate(content.BODY, 1):
            text = self.section(section, str(i))
            if section.key == content.REPRODUCIBILITY.key:
                text += self.code()
            parts.append(text + self.notes())
        appendix = self.appendix() + self.notes()
        parts.append(self.references())
        parts.append(self.cite_block())
        parts.append(appendix)
        return "".join(parts)


def numbers(doc: Document) -> tuple[dict[str, str], dict[str, str]]:
    """Assigns the printed numbers of the figures and tables.

    Main items are numbered in order of appearance in the text, appendix items
    A1, A2 and so on in registry order.

    Args:
        doc: The built figures and tables.

    Returns:
        Figure key to printed number, and table key to printed number.
    """
    page = Page(doc)
    return page.figure_number, page.table_number


def render_page(doc: Document) -> str:
    """Renders the whole site as one HTML document.

    Args:
        doc: The built figures, tables, date and bibliography.

    Returns:
        The HTML document.
    """
    page = Page(doc)
    abstract = page.paragraph(content.ABSTRACT)
    contents = page.contents()
    body = page.body()
    masthead = _e(doc.date)
    if doc.commit:
        masthead = masthead.replace(
            _e(doc.commit), f"<code>{_e(doc.commit)}</code>", 1
        )
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
        f'<p class="date">{masthead}</p></header>'
        f'<section class="abstract" id="abstract"><h2>Abstract</h2>'
        f"{abstract}</section>"
        f"{contents}{body}</main>\n"
        '<footer class="colophon"><p>Set in Latin Modern under the GUST '
        "Font License; the code is released under the MIT license; no "
        "analytics.</p></footer>\n"
        "</body>\n</html>\n"
    )
