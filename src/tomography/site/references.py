"""Reads the repository bibliography for the numbered reference list.

`docs/references.bib` holds the entries; the page cites a few of them by key
prefix, lists them in order of first citation and prints the BibTeX of the
paper that the repository reimplements.
"""

import dataclasses
import re
from typing import Final

CITED: Final = (
    "zou2025",
    "socher2013",
    "hendrycks2021",
    "lin2022",
    "burns2023",
    "turner2023steering",
    "li2023inference",
    "mallen2023",
)


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


def plain(text: str) -> str:
    """Removes BibTeX markup from a field value.

    Args:
        text: A field value with braces and accent commands.

    Returns:
        Plain text with accents dropped and whitespace collapsed.
    """
    text = re.sub(r"\\[`'^\"~cvu]\s*\{?([A-Za-z])\}?", r"\1", text)
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


def find(entries: dict[str, Entry], prefix: str) -> Entry:
    """Looks an entry up by the start of its key.

    Args:
        entries: Parsed entries.
        prefix: Start of the citation key.

    Returns:
        The first entry whose key starts with the prefix.

    Raises:
        KeyError: If no key starts with the prefix.
    """
    for key, entry in entries.items():
        if key.startswith(prefix):
            return entry
    raise KeyError(prefix)


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
        The first author, followed by et al. when there are several.
    """
    names = [
        n for n in re.split(r"\s+and\s+", entry.fields.get("author", "")) if n
    ]
    if not names:
        return ""
    first = _author(names[0])
    return f"{first} et al." if len(names) > 1 else first


def format_line(entry: Entry) -> str:
    """Writes the reference list line of an entry, without its link.

    Args:
        entry: The entry.

    Returns:
        Authors, year, title and venue.
    """
    venue = plain(
        entry.fields.get("booktitle")
        or entry.fields.get("journal")
        or entry.fields.get("howpublished")
        or ""
    )
    if not venue and "eprint" in entry.fields:
        venue = f"arXiv:{entry.fields['eprint']}"
    parts = [f"{format_authors(entry)} ({entry.year}).", f"{entry.title}."]
    if venue:
        parts.append(f"{venue}.")
    return " ".join(p for p in parts if p.strip())
