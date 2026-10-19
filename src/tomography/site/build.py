"""Builds the results site into a directory."""

import csv
import datetime
import io
from pathlib import Path
import shutil

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


def _csv_text(columns: list[str], rows: list[list[str]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return buffer.getvalue()


def build(
    results: Path,
    out: Path,
    figures: tuple[registry.Spec, ...] = registry.FIGURES,
    tables: tuple[registry.Spec, ...] = registry.TABLES,
    appendix: tuple[registry.Spec, ...] = registry.APPENDIX,
) -> page.Document:
    """Writes the page, its assets and the files behind every figure.

    Each figure is written as `figures/<key>.svg`, `figures/<key>.pdf` and
    `data/<key>.csv`, and each table as `data/<key>.csv`. A figure or table
    that cannot be built stops the build.

    Args:
        results: The committed `results/` directory to read.
        out: Directory to write; created if missing.
        figures: Main figures in registry order.
        tables: Tables in registry order.
        appendix: Appendix figures in registry order.

    Returns:
        The document that was rendered.
    """
    manifest = Manifest(results)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)

    def figure(spec: registry.Spec) -> page.FigureEntry:
        output = registry.build_figure(spec, manifest)
        svg = paper.svg_bytes(output.figure)
        (out / "figures" / f"{spec.key}.svg").write_bytes(svg)
        (out / "figures" / f"{spec.key}.pdf").write_bytes(
            paper.pdf_bytes(output.figure)
        )
        (out / "data" / f"{spec.key}.csv").write_text(
            output.csv, encoding="utf-8", newline="\n"
        )
        return page.FigureEntry(spec, output, svg)

    def table(spec: registry.Spec) -> page.TableEntry:
        output = registry.build_table(spec, manifest)
        (out / "data" / f"{spec.key}.csv").write_text(
            _csv_text(output.columns, output.rows),
            encoding="utf-8",
            newline="\n",
        )
        return page.TableEntry(spec, output)

    doc = page.Document(
        figures={s.key: figure(s) for s in figures},
        tables={s.key: table(s) for s in tables},
        appendix={s.key: figure(s) for s in appendix},
        date=results_date(manifest),
        bib=references.parse(BIB.read_text(encoding="utf-8")),
    )
    (out / "index.html").write_text(
        page.render_page(doc), encoding="utf-8", newline="\n"
    )
    shutil.copyfile(STATIC / "site.css", out / "site.css")
    shutil.copytree(STATIC / "fonts", out / "fonts", dirs_exist_ok=True)
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return doc
