"""Builds the results site into a directory."""

import json
from pathlib import Path
import shutil

from tomography.site import findings as findings_module
from tomography.site import page
from tomography.site.store import Store

STATIC = Path(__file__).parent / "static"


def build(results: Path, out: Path) -> list[findings_module.Finding]:
    """Writes the page, its assets and one data file per chart.

    Args:
        results: The committed `results/` directory to read.
        out: Directory to write; created if missing.

    Returns:
        The findings that were rendered.
    """
    store = Store(results)
    found = findings_module.build_findings(store)
    info = findings_module.overview(store)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(
        page.render_page(found, info), encoding="utf-8"
    )
    for name in ("site.css", "site.js"):
        shutil.copyfile(STATIC / name, out / name)
    (out / ".nojekyll").write_text("", encoding="utf-8")
    for finding in found:
        for figure in finding.figures:
            for panel in figure.panels:
                payload = page.panel_payload(panel)
                (out / "data" / f"{panel.uid}.json").write_text(
                    json.dumps(payload, indent=1) + "\n", encoding="utf-8"
                )
    return found
