from pathlib import Path
import re

import pytest

from tomography.site import build as build_module
from tomography.site import page
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
STATIC = ("abstract", "method", "models", "experiments", "links", "references")


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("site")
    build_module.build(RESULTS, out)
    return out


def _page(built: Path) -> str:
    return (built / "index.html").read_text(encoding="utf-8")


def test_real_results_build_every_section(built: Path) -> None:
    text = _page(built)
    for key in (*HEADLINES, *STATIC):
        assert f'id="{key}"' in text, key


def test_sections_are_numbered_in_order(built: Path) -> None:
    text = _page(built)
    numbers = [int(n) for n in re.findall(r"<h2>(\d+)&ensp;", text)]
    assert numbers == list(range(1, len(numbers) + 1))
    assert len(numbers) >= len(HEADLINES) + 3


def test_each_figure_has_svg_pdf_csv_and_a_numbered_caption(
    built: Path,
) -> None:
    text = _page(built)
    figures = re.findall(r'<figure id="fig(\d+)">', text)
    assert len(figures) >= len(HEADLINES)
    assert figures == [str(i) for i in range(1, len(figures) + 1)]
    for number in figures:
        assert (built / "figures" / f"fig{number}.svg").is_file()
        assert (built / "figures" / f"fig{number}.pdf").is_file()
        assert (built / "data" / f"fig{number}.csv").is_file()
        assert f"<strong>Figure {number}.</strong>" in text
        assert f'href="data/fig{number}.csv"' in text
        assert f'href="figures/fig{number}.pdf"' in text


def test_every_inline_svg_is_a_labelled_image(built: Path) -> None:
    text = _page(built)
    svgs = re.findall(r"<svg [^>]*>", text)
    assert svgs
    for tag in svgs:
        assert 'role="img"' in tag
        assert re.search(r'aria-label="Figure \d+\.', tag), tag
    assert "<metadata>" not in text
    assert "<?xml" not in text


def test_every_finding_keeps_its_caveats_and_figure(built: Path) -> None:
    text = _page(built)
    for key in HEADLINES:
        section = text.split(f'id="{key}"', 1)[1].split("</section>", 1)[0]
        assert 'class="caveats"' in section, key
        assert "<svg" in section, key
        assert "<figcaption>" in section, key


def test_page_has_no_cards_toggles_or_scripts(built: Path) -> None:
    text = _page(built)
    css = (built / "site.css").read_text(encoding="utf-8")
    assert "<script" not in text
    assert "<details" not in text
    assert "<button" not in text
    assert not (built / "site.js").exists()
    for token in ("border-radius", "box-shadow"):
        assert token not in css


def test_page_has_no_dashes_or_unexpected_remote_assets(built: Path) -> None:
    for name in ("index.html", "site.css"):
        text = (built / name).read_text(encoding="utf-8")
        assert "\u2014" not in text, name
        assert "\u2013" not in text, name
    text = _page(built)
    remote = re.findall(r'(?:src|href)="(https?://[^"]+)"', text)
    allowed = (
        "github.com/javierdejesusda/tomography",
        "arxiv.org/abs/",
        "fonts.googleapis.com",
        "fonts.gstatic.com",
    )
    assert all(any(a in u for a in allowed) for u in remote)


def test_header_has_title_links_and_abstract_but_no_author(built: Path) -> None:
    text = _page(built)
    head = text.split('id="abstract"', 1)[0]
    assert "<h1>" in head
    for name in ("Code", "Findings", "Methodology", "Data"):
        assert f">{name}</a>" in head
    assert "Abstract" in text
    assert "author" not in head.lower()


def test_tables_are_plain_booktabs(built: Path) -> None:
    css = (built / "site.css").read_text(encoding="utf-8")
    text = _page(built)
    assert "<thead>" in text
    assert "nth-child" not in css
    assert "border-left" not in css
    assert "border-right" not in css


def test_page_links_to_code_findings_and_data(built: Path) -> None:
    text = _page(built)
    for target in (
        "docs/findings.md",
        "docs/methodology.md",
        "tree/main/results",
    ):
        assert target in text


def test_page_is_responsive_and_cites_the_paper(built: Path) -> None:
    text = _page(built)
    css = (built / "site.css").read_text(encoding="utf-8")
    assert 'name="viewport"' in text
    assert '<html lang="en"' in text
    assert "max-width: 680px" in css
    assert "STIX Two Text" in css
    assert "@misc{zou2025" in text
    assert (built / ".nojekyll").is_file()


def test_p10_section_uses_the_p10c_results(built: Path) -> None:
    section = (
        _page(built).split('id="approximate"', 1)[1].split("</section>", 1)[0]
    )
    assert "yet." not in section
    assert "123 cosines" in section
    assert "P9" in section


def test_build_is_byte_identical_twice(tmp_path: Path) -> None:
    build_module.build(RESULTS, tmp_path / "a")
    build_module.build(RESULTS, tmp_path / "b")
    names = sorted(
        p.relative_to(tmp_path / "a")
        for p in (tmp_path / "a").rglob("*")
        if p.is_file()
    )
    assert names
    for name in names:
        assert (tmp_path / "a" / name).read_bytes() == (
            tmp_path / "b" / name
        ).read_bytes(), name


def test_bib_entry_reads_a_key_and_tolerates_a_missing_file(
    tmp_path: Path,
) -> None:
    bib = tmp_path / "x.bib"
    bib.write_text("@misc{aa1,\n title={A},\n}\n\n@misc{bb2,\n title={B},\n}\n")
    assert page.bib_entry("bb", bib).startswith("@misc{bb2")
    assert "title={A}" not in page.bib_entry("bb", bib)
    assert page.bib_entry("zz", bib) == ""
    assert page.bib_entry("aa", tmp_path / "missing.bib") == ""


def test_empty_results_still_render_the_static_sections(tmp_path: Path) -> None:
    empty = tmp_path / "results"
    empty.mkdir()
    out = tmp_path / "out"
    build_module.build(empty, out)
    text = (out / "index.html").read_text(encoding="utf-8")
    for key in ("abstract", "method", "links", "references"):
        assert f'id="{key}"' in text
    for key in HEADLINES:
        assert f'id="{key}"' not in text


def test_command_line_build_writes_the_site(tmp_path: Path) -> None:
    code = main(["build", "--results", str(RESULTS), "--out", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "index.html").is_file()
