"""A full synthetic document: every registry item, every token resolvable.

The page tests use it to render the whole paper before the real builders of
every figure exist; the numbers are placeholders, never results.
"""

from pathlib import Path
import re

from matplotlib.figure import Figure

from tomography.site import content
from tomography.site import page
from tomography.site import references
from tomography.site import registry
from tomography.site import stats

SVG = (
    b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
    b'<style type="text/css">*{stroke-linecap: butt}</style>'
    b'<defs><clipPath id="p1"><rect width="5" height="5"/></clipPath></defs>'
    b'<g clip-path="url(#p1)" id="axes_1"><use xlink:href="#p1" '
    b'xmlns:xlink="http://www.w3.org/1999/xlink"/></g></svg>'
)
BIB = Path(__file__).resolve().parents[1] / "docs" / "references.bib"
TOKEN = re.compile(r"\[\[(.+?)\]\]")


def _texts() -> list[str]:
    texts = [content.ABSTRACT, content.TLDR]
    sections = [*content.BODY, *content.APPENDICES, content.KEY_FINDINGS]
    while sections:
        section = sections.pop()
        texts += [*section.paragraphs, *section.after]
        for head, text in (*section.runin, *section.listed):
            texts += [head, text]
        sections += list(section.subsections)
    return texts


def placeholders() -> tuple[dict, dict]:
    """Collects the numbers and counts that the text asks for.

    Returns:
        Numbers and counts by address `kind:key`, then field name.
    """
    numbers: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, dict[str, stats.RoleCount]]] = {}
    for text in _texts():
        for token in TOKEN.findall(text):
            kind, _, rest = token.partition(":")
            if kind not in ("num", "count", "runs"):
                continue
            owner, _, name = rest.partition(":")
            key, _, field = name.rpartition(".")
            address = f"{owner}:{key}"
            if kind == "num":
                numbers.setdefault(address, {})[field] = 3.0
            else:
                counts.setdefault(address, {})[field] = {
                    "gpt2": stats.RoleCount(4, 4),
                    "base": stats.RoleCount(2, 3),
                    "post": stats.RoleCount(0, 3),
                }
    return numbers, counts


def document(commit: str = "abc1234") -> page.Document:
    """Builds the stub document.

    Args:
        commit: The commit shown in the masthead.

    Returns:
        A document with every main and appendix figure and table.
    """
    numbers, counts = placeholders()

    def figure(key: str) -> page.FigureEntry:
        address = f"fig:{key}"
        out = registry.FigureOutput(
            Figure(),
            f"Claim of {key}.",
            "Detail.",
            numbers.get(address, {}),
            "a,b\n1,2\n" if key != "method" else "",
            counts.get(address, {}),
        )
        return page.FigureEntry(key, out, SVG, b"%PDF-1.4")

    def table(key: str) -> page.TableEntry:
        address = f"table:{key}"
        out = registry.TableOutput(
            f"Claim of {key}.",
            "Detail.",
            ["A", "B"],
            [["x", "1"], ["y", "2"]],
            "lr",
            [],
            numbers.get(address, {}),
            counts=counts.get(address, {}),
            row_groups=(),
        )
        return page.TableEntry(key, out)

    return page.Document(
        {k: figure(k) for k in registry.MAIN_FIGURES},
        {k: table(k) for k in registry.MAIN_TABLES},
        {k: figure(k) for k in registry.APPENDIX_FIGURES},
        f"Results at commit {commit}, committed 1 January 2027",
        references.parse(BIB.read_text(encoding="utf-8")),
        {k: table(k) for k in registry.APPENDIX_TABLES},
        commit,
    )
