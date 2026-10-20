"""Reads the repository bibliography for the numbered reference list.

`docs/references.bib` holds the entries; the page cites them by exact key,
lists the cited ones in order of first citation and prints the BibTeX of the
paper that the repository reimplements.
"""

import dataclasses
import re
from typing import Final
import unicodedata

PAPER_KEY: Final = "zou2025representationengineeringtopdownapproach"
MAX_AUTHORS: Final = 10
ACCENTS: Final = {
    "`": chr(0x300),
    "'": chr(0x301),
    "^": chr(0x302),
    "~": chr(0x303),
    '"': chr(0x308),
    "=": chr(0x304),
    ".": chr(0x307),
    "u": chr(0x306),
    "v": chr(0x30C),
    "c": chr(0x327),
    "H": chr(0x30B),
}


@dataclasses.dataclass(frozen=True)
class Entry:
    """One BibTeX entry.

    Attributes:
        key: Citation key.
        kind: Entry type such as `inproceedings`.
        fields: Field name to plain text value.
        raw: The entry exactly as it is written in the file.
    """

    key: str
    kind: str
    fields: dict[str, str]
    raw: str

    @property
    def title(self) -> str:
        """The title without braces."""
        return plain(self.fields.get("title", ""))

    @property
    def url(self) -> str:
        """The link of the entry, empty when it has none."""
        return self.fields.get("url", "")

    @property
    def year(self) -> str:
        """The year as written."""
        return self.fields.get("year", "")


def _accent(match: re.Match[str]) -> str:
    letter = match.group(2) or match.group(3)
    return unicodedata.normalize("NFC", letter + ACCENTS[match.group(1)])


def plain(text: str) -> str:
    """Removes BibTeX markup from a field value.

    Args:
        text: A field value with braces and accent commands.

    Returns:
        Plain text with LaTeX accents turned into Unicode letters, page
        ranges with a hyphen and whitespace collapsed.
    """
    text = re.sub(
        r"\\([`'^\"~=.uvcH])\s*(?:\{\\?([A-Za-z])\}|\\?([A-Za-z]))",
        _accent,
        text,
    )
    text = text.replace("{", "").replace("}", "").replace("\\&", "&")
    text = text.replace("--", "-")
    return re.sub(r"\s+", " ", text).strip()


def _fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    pattern = re.compile(r"(\w+)\s*=\s*", re.ASCII)
    pos = 0
    while True:
        match = pattern.search(body, pos)
        if not match:
            return fields
        start = match.end()
        if start < len(body) and body[start] == "{":
            depth, end = 0, start
            while end < len(body):
                depth += body[end] == "{"
                depth -= body[end] == "}"
                end += 1
                if depth == 0:
                    break
            value = body[start + 1 : end - 1]
        else:
            end = body.find(",", start)
            end = len(body) if end < 0 else end
            value = body[start:end]
        fields[match.group(1).lower()] = value.strip()
        pos = end


def parse(text: str) -> dict[str, Entry]:
    """Parses a BibTeX file.

    Args:
        text: The file contents.

    Returns:
        Entries by citation key, in file order.
    """
    entries: dict[str, Entry] = {}
    for chunk in re.split(r"\n(?=@)", text):
        head = re.match(r"@(\w+)\{([^,\s]+),", chunk)
        if not head:
            continue
        key = head.group(2)
        entries[key] = Entry(
            key,
            head.group(1).lower(),
            _fields(chunk[head.end() :]),
            chunk.strip(),
        )
    return entries


def find(entries: dict[str, Entry], key: str) -> Entry:
    """Looks an entry up by its exact key.

    Args:
        entries: Parsed entries.
        key: The citation key.

    Returns:
        The entry.

    Raises:
        KeyError: If there is no entry with that key.
    """
    return entries[key]


def _initials(given: str) -> str:
    return " ".join(f"{part[0]}." for part in given.replace("-", " ").split())


def _author(name: str) -> str:
    name = plain(name)
    if "," in name:
        family, _, given = name.partition(",")
    else:
        *given_parts, family = name.split()
        given = " ".join(given_parts)
    initials = _initials(given.strip())
    return f"{initials} {family.strip()}".strip()


def format_authors(entry: Entry) -> str:
    """Writes the author list as initials and surnames.

    Args:
        entry: The entry.

    Returns:
        All authors when there are at most ten, otherwise the first ten
        followed by et al.
    """
    names = [
        n for n in re.split(r"\s+and\s+", entry.fields.get("author", "")) if n
    ]
    cut = len(names) > MAX_AUTHORS
    people = [_author(n) for n in names[:MAX_AUTHORS]]
    if not people:
        return ""
    if cut:
        return ", ".join(people) + " et al."
    if len(people) == 1:
        return people[0]
    return ", ".join(people[:-1]) + " and " + people[-1]


def link(entry: Entry) -> tuple[str, str]:
    """Names the link of an entry by its DOI or its arXiv id.

    Args:
        entry: The entry.

    Returns:
        Link text and address; the DOI is preferred, and both are empty for
        an entry with neither.
    """
    doi = entry.fields.get("doi", "")
    if doi:
        return f"doi:{doi}", f"https://doi.org/{doi}"
    eprint = entry.fields.get("eprint", "")
    if eprint:
        return f"arXiv:{eprint}", f"https://arxiv.org/abs/{eprint}"
    return "", ""


def format_line(entry: Entry) -> str:
    """Writes the reference list line of an entry, without its link.

    Args:
        entry: The entry.

    Returns:
        Authors, year, title and venue with volume, number and pages.
    """
    fields = entry.fields
    venue = plain(
        fields.get("booktitle")
        or fields.get("journal")
        or fields.get("institution")
        or fields.get("howpublished")
        or ""
    )
    volume = plain(fields.get("volume", ""))
    if volume and fields.get("number"):
        volume += f"({plain(fields['number'])})"
    pages = plain(fields.get("pages", ""))
    place = " ".join(p for p in (venue, volume) if p)
    if pages:
        place += f", {pages}" if volume else f", pp. {pages}"
    if not place and "eprint" in fields and fields.get("doi"):
        place = f"arXiv:{fields['eprint']}"
    parts = [f"{format_authors(entry)} ({entry.year}).", f"{entry.title}."]
    if place:
        parts.append(f"{place}.")
    return " ".join(p for p in parts if p.strip())
