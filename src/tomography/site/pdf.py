"""Prints the built site to a PDF with a headless browser."""

from collections.abc import Callable
from pathlib import Path
import shutil
import subprocess
from typing import Any

NAME = "tomography-results.pdf"
PRINT_NAME = "print.html"
BROWSERS = ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge")
VIRTUAL_TIME_BUDGET_MS = 30000
TIMEOUT_SECONDS = 300


def find_browser() -> str:
    """Finds a Chromium based browser on the path.

    Returns:
        The path of the first of `BROWSERS` that is installed.

    Raises:
        FileNotFoundError: If none of the candidates is installed.
    """
    for name in BROWSERS:
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError(
        "no browser found on the path; looked for " + ", ".join(BROWSERS)
    )


def print_page(
    site: Path,
    browser: str,
    run: Callable[..., Any] = subprocess.run,
) -> Path:
    """Prints the built page to `NAME` inside the site directory.

    The browser prints a copy of the page without lazy image loading, so every
    appendix figure is loaded before the page is printed. The copy sits next
    to `index.html` so that its relative URLs resolve, and it is always
    removed.

    Args:
        site: The directory that holds the built `index.html`.
        browser: The browser executable.
        run: The process runner, replaceable for tests.

    Returns:
        The path of the written PDF.

    Raises:
        subprocess.CalledProcessError: If the browser exits with an error.
        RuntimeError: If the browser wrote no PDF or a file that is not one.
    """
    page = site / "index.html"
    copy = site / PRINT_NAME
    target = site / NAME
    target.unlink(missing_ok=True)
    html = page.read_text(encoding="utf-8").replace(' loading="lazy"', "")
    copy.write_text(html, encoding="utf-8", newline="\n")
    try:
        run(
            [
                browser,
                "--headless",
                "--no-pdf-header-footer",
                f"--virtual-time-budget={VIRTUAL_TIME_BUDGET_MS}",
                f"--print-to-pdf={target.resolve()}",
                copy.resolve().as_uri(),
            ],
            check=True,
            timeout=TIMEOUT_SECONDS,
        )
    finally:
        copy.unlink(missing_ok=True)
    if not target.is_file():
        raise RuntimeError(f"the browser wrote no PDF to {target}")
    with target.open("rb") as handle:
        if handle.read(4) != b"%PDF":
            raise RuntimeError(f"{target} is not a PDF")
    return target
