"""Builds the results site into a directory."""

from pathlib import Path
import shutil

from tomography.site import figures
from tomography.site import findings as findings_module
from tomography.site import page
from tomography.site.store import Store

STATIC = Path(__file__).parent / "static"


def build(results: Path, out: Path) -> list[findings_module.Finding]:
    """Writes the page, its stylesheet and the files behind every figure.

    Each figure is written as `figures/fig<N>.svg`, `figures/fig<N>.pdf` and
    `data/fig<N>.csv`, numbered in page order.

    Args:
        results: The committed `results/` directory to read.
        out: Directory to write; created if missing.

    Returns:
        The findings that were rendered.
    """
    store = Store(results)
    found = findings_module.build_findings(store)
    info = findings_module.overview(store)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    svgs = []
    number = 0
    for finding in found:
        for figure in finding.figures:
            number += 1
            svg = figures.svg(figure)
            svgs.append(svg)
            (out / "figures" / f"fig{number}.svg").write_bytes(svg)
            (out / "figures" / f"fig{number}.pdf").write_bytes(
                figures.pdf(figure)
            )
            (out / "data" / f"fig{number}.csv").write_text(
                figures.csv_text(figure), encoding="utf-8", newline="\n"
            )
    (out / "index.html").write_text(
        page.render_page(found, info, svgs), encoding="utf-8", newline="\n"
    )
    shutil.copyfile(STATIC / "site.css", out / "site.css")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return found
