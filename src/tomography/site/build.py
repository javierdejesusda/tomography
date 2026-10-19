"""Builds the results site into a directory."""

import csv
import datetime
import io
from pathlib import Path
import re
import shutil
import subprocess

from tomography.site import page
from tomography.site import references
from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import paper

STATIC = Path(__file__).parent / "static"
BIB = Path(__file__).resolve().parents[3] / "docs" / "references.bib"
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def results_date(manifest: Manifest) -> str:
    """Writes the date line from the newest run in the results.

    Args:
        manifest: The run resolver.

    Returns:
        Text such as `Results of 19 October 2026`, or `Results` when no run
        stores a start time.
    """
    newest = ""
    for exp in manifest.experiments():
        for model in manifest.models(exp):
            for run in manifest.runs(exp, model):
                newest = max(newest, run.started_at)
    if not newest:
        return "Results"
    day = datetime.datetime.fromisoformat(newest).date()
    return f"Results of {day.day} {MONTHS[day.month - 1]} {day.year}"


def results_commit(results: Path) -> tuple[str, str]:
    """Finds the commit that last changed the results.

    Args:
        results: The committed `results/` directory.

    Returns:
        The short commit hash and its date as `19 October 2026`, or two
        empty strings when git or the history is not available.
    """
    try:
        done = subprocess.run(
            [
                "git",
                "-C",
                str(results.parent),
                "log",
                "-1",
                "--format=%h %cs",
                "--",
                results.name,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "", ""
    found = re.fullmatch(r"(\w+) (\d{4})-(\d{2})-(\d{2})", done.stdout.strip())
    if not found:
        return "", ""
    sha, year, month, day = found.groups()
    return sha, f"{int(day)} {MONTHS[int(month) - 1]} {year}"


def _plain(cell: str) -> str:
    return re.sub(
        r"\[\[cite:([^\]]+)\]\]",
        lambda m: m.group(1).replace(",", "; "),
        cell,
    )


def _csv_text(columns: list[str], rows: list[list[str]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows([[_plain(c) for c in row] for row in rows])
    return buffer.getvalue()


def build(
    results: Path,
    out: Path,
    figures: tuple[str, ...] = registry.MAIN_FIGURES,
    tables: tuple[str, ...] = registry.MAIN_TABLES,
    appendix: tuple[str, ...] = registry.APPENDIX_FIGURES,
    appendix_tables: tuple[str, ...] = registry.APPENDIX_TABLES,
) -> page.Document:
    """Writes the page, its assets and the files behind every figure.

    Files are named by the printed number: each figure is written as
    `figures/fig-<n>.svg`, `figures/fig-<n>.pdf` and `data/fig-<n>.csv`
    (only when the figure has rows), and each table as
    `data/table-<n>.csv`. A figure or table that cannot be built stops the
    build.

    Args:
        results: The committed `results/` directory to read.
        out: Directory to write; created if missing.
        figures: Main figure keys in registry order.
        tables: Main table keys in registry order.
        appendix: Appendix figure keys in registry order.
        appendix_tables: Appendix table keys in registry order.

    Returns:
        The document that was rendered.
    """
    manifest = Manifest(results)
    commit, built = results_commit(results)

    def figure(key: str) -> page.FigureEntry:
        output = registry.build_figure(key, manifest)
        return page.FigureEntry(
            key,
            output,
            paper.svg_bytes(output.figure),
            paper.pdf_bytes(output.figure),
        )

    def table(key: str) -> page.TableEntry:
        return page.TableEntry(key, registry.build_table(key, manifest))

    doc = page.Document(
        figures={k: figure(k) for k in figures},
        tables={k: table(k) for k in tables},
        appendix={k: figure(k) for k in appendix},
        appendix_tables={k: table(k) for k in appendix_tables},
        date=(
            f"Results at commit {commit}, built {built}"
            if commit
            else results_date(manifest)
        ),
        bib=references.parse(BIB.read_text(encoding="utf-8")),
        commit=commit,
    )
    figure_number, table_number = page.numbers(doc)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    for key, fig in {**doc.figures, **doc.appendix}.items():
        name = f"fig-{figure_number[key]}"
        (out / "figures" / f"{name}.svg").write_bytes(fig.svg)
        (out / "figures" / f"{name}.pdf").write_bytes(fig.pdf)
        if fig.output.csv.strip():
            (out / "data" / f"{name}.csv").write_text(
                fig.output.csv, encoding="utf-8", newline="\n"
            )
    for key, tab in {**doc.tables, **doc.appendix_tables}.items():
        (out / "data" / f"table-{table_number[key]}.csv").write_text(
            _csv_text(tab.output.columns, tab.output.rows),
            encoding="utf-8",
            newline="\n",
        )
    (out / "index.html").write_text(
        page.render_page(doc), encoding="utf-8", newline="\n"
    )
    shutil.copyfile(STATIC / "site.css", out / "site.css")
    shutil.copytree(STATIC / "fonts", out / "fonts", dirs_exist_ok=True)
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return doc
