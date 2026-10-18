import json
from pathlib import Path
import re

import pytest

from tomography.site import build as build_module
from tomography.site.__main__ import main

RESULTS = Path(__file__).resolve().parents[1] / "results"
HEADLINES = (
    "scaling",
    "knowledge",
    "steering",
    "approximate",
    "fluency",
    "templates",
    "finders",
    "truthfulqa",
)
STATIC = ("method", "models", "experiments", "links")


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("site")
    build_module.build(RESULTS, out)
    return out


def test_real_results_build_every_headline_section(built: Path) -> None:
    page = (built / "index.html").read_text(encoding="utf-8")
    for key in (*HEADLINES, *STATIC):
        assert f'id="{key}"' in page, key


def test_each_chart_has_a_data_file_and_a_table_view(built: Path) -> None:
    page = (built / "index.html").read_text(encoding="utf-8")
    charts = re.findall(r'<svg[^>]*class="chart"', page)
    data = sorted((built / "data").glob("*.json"))
    assert len(charts) == len(data) > 0
    assert page.count('<details class="table-view"') == len(charts)
    for path in data:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["rows"] or payload["series"]


def test_every_finding_shows_its_caveats_next_to_the_chart(built: Path) -> None:
    page = (built / "index.html").read_text(encoding="utf-8")
    for key in HEADLINES:
        section = page.split(f'id="{key}"', 1)[1].split("</section>", 1)[0]
        assert 'class="caveats"' in section, key
        assert "<svg" in section, key


def test_page_has_no_dashes_trackers_or_remote_assets(built: Path) -> None:
    for name in ("index.html", "site.css", "site.js"):
        text = (built / name).read_text(encoding="utf-8")
        assert "\u2014" not in text, name
        assert "\u2013" not in text, name
    page = (built / "index.html").read_text(encoding="utf-8")
    remote = re.findall(r'(?:src|href)="(https?://[^"]+)"', page)
    allowed = ("github.com/javierdejesusda/tomography", "arxiv.org/abs/")
    assert all(any(a in u for a in allowed) for u in remote)
    assert '<script src="http' not in page


def test_page_links_to_code_findings_and_data(built: Path) -> None:
    page = (built / "index.html").read_text(encoding="utf-8")
    for target in (
        "docs/findings.md",
        "docs/methodology.md",
        "tree/main/results",
    ):
        assert target in page


def test_page_is_responsive_and_themed(built: Path) -> None:
    page = (built / "index.html").read_text(encoding="utf-8")
    css = (built / "site.css").read_text(encoding="utf-8")
    assert 'name="viewport"' in page
    assert '<html lang="en"' in page
    assert "prefers-color-scheme: dark" in css
    assert 'data-theme="dark"' in css
    assert "prefers-reduced-motion" in css
    assert (built / ".nojekyll").is_file()


def test_empty_results_still_render_the_static_sections(tmp_path: Path) -> None:
    empty = tmp_path / "results"
    empty.mkdir()
    out = tmp_path / "out"
    build_module.build(empty, out)
    page = (out / "index.html").read_text(encoding="utf-8")
    for key in ("method", "links"):
        assert f'id="{key}"' in page
    for key in HEADLINES:
        assert f'id="{key}"' not in page


def test_command_line_build_writes_the_site(tmp_path: Path) -> None:
    code = main(["build", "--results", str(RESULTS), "--out", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "index.html").is_file()
