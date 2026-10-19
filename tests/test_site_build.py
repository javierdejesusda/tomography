from html.parser import HTMLParser
import json
from pathlib import Path
import re

import pytest

from tomography.site import __main__ as cli
from tomography.site import build as build_module
from tomography.site import content
from tomography.site import page
from tomography.site import references
from tomography.site import registry
from tomography.site.manifest import Manifest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STATIC = ROOT / "src/tomography/site/static"
DASHES = (chr(0x2014), chr(0x2013))
FIGURES = registry.FIGURES[:4]
TABLES = registry.TABLES[:3]
APPENDIX = registry.APPENDIX
APPENDIX_TABLES = registry.APPENDIX_TABLES


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("site")
    build_module.build(RESULTS, out, FIGURES, TABLES, APPENDIX, APPENDIX_TABLES)
    return out


@pytest.fixture(scope="module")
def html(built: Path) -> str:
    return (built / "index.html").read_text("utf-8")


class Ids(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.hrefs: list[str] = []
        self.tags: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.tags.append(tag)
        for name, value in attrs:
            if name == "id" and value:
                self.ids.add(value)
            if name == "href" and value:
                self.hrefs.append(value)


def test_the_page_is_english_and_has_no_scripts(html: str) -> None:
    assert html.startswith("<!doctype html>")
    assert '<html lang="en">' in html
    assert "<script" not in html
    assert 'name="viewport"' in html


def test_the_title_block_has_a_title_a_repo_link_and_a_date_but_no_author(
    html: str,
) -> None:
    head = html[html.index('<header class="title">') : html.index("</header>")]
    assert f"<h1>{content.TITLE}</h1>" in head
    assert content.REPO_TEXT in head
    assert re.search(r'class="date">Results of \d+ [A-Z][a-z]+ \d{4}<', head)
    assert head.count("<p") == 2


def test_the_abstract_comes_before_the_first_numbered_section(
    html: str,
) -> None:
    assert html.index('id="abstract"') < html.index('id="introduction"')
    assert "<h2>Abstract</h2>" in html


def test_sections_and_subsections_are_numbered_in_order(html: str) -> None:
    heads = re.findall(r"<h([23]) id=\"[^\"]+\">([^<]+)</h", html)
    texts = [t.replace("&ensp;", " ") for _, t in heads]
    assert texts[0] == "1 Introduction"
    assert texts[1] == "2 Method"
    assert texts[2:6] == [
        "2.1 Reading and steering",
        "2.2 Concepts and data",
        "2.3 Checkpoints",
        "2.4 Baselines and intervals",
    ]
    assert any(t.startswith("Appendix A ") for t in texts)


def test_every_internal_link_points_at_an_id_on_the_page(html: str) -> None:
    parser = Ids()
    parser.feed(html)
    missing = [
        h for h in parser.hrefs if h.startswith("#") and h[1:] not in parser.ids
    ]
    assert missing == []


def test_figures_and_tables_are_numbered_in_order_of_appearance(
    html: str,
) -> None:
    figures = re.findall(r"<figcaption>Figure (A?\d+): ", html)
    assert figures == ["1", "2", "3", "4", "A1", "A2", "A3", "A4", "A5", "A6"]
    tables = re.findall(r"<caption>Table (A?\d+): ", html)
    assert tables == ["1", "2", "3", "A1"]


def test_a_figure_caption_follows_its_graphic_and_a_table_caption_leads(
    html: str,
) -> None:
    for chunk in re.findall(r"<figure .*?</figure>", html, flags=re.S):
        assert chunk.index("<svg") < chunk.index("<figcaption>")
    for chunk in re.findall(r"<table>.*?</table>", html, flags=re.S):
        assert chunk.index("<caption>") < chunk.index("<thead>")


def test_the_caption_label_is_plain_and_followed_by_a_colon(html: str) -> None:
    assert "<strong>Figure" not in html
    assert "<b>Table" not in html
    caption = re.search(r"<figcaption>(.*?)</figcaption>", html, flags=re.S)
    assert caption is not None
    assert caption.group(1).startswith("Figure 1: ")


def test_every_inline_svg_is_a_labelled_image(html: str) -> None:
    svgs = re.findall(r"<svg [^>]*>", html)
    assert len(svgs) == len(FIGURES) + len(APPENDIX)
    for tag in svgs:
        assert 'role="img"' in tag
        assert 'aria-label="Figure ' in tag


def test_tables_are_booktabs_without_vertical_rules(html: str) -> None:
    css = (STATIC / "site.css").read_text("utf-8")
    assert "border-left" not in css
    assert "border-right" not in css
    assert "<thead>" in html
    assert " border=" not in html
    assert "0.12em solid" in css
    assert "0.06em solid" in css


def test_numeric_columns_are_right_aligned(html: str) -> None:
    assert '<td class="r">' in html
    assert "font-variant-numeric: tabular-nums" in (
        STATIC / "site.css"
    ).read_text("utf-8")


def test_citations_are_numbered_by_first_use_and_listed(html: str) -> None:
    cites = re.findall(r'<a class="ref" href="#ref(\d+)">\[(\d+)\]</a>', html)
    assert cites
    assert all(a == b for a, b in cites)
    first = [int(a) for a, _ in cites]
    assert first[0] == 1
    seen: list[int] = []
    for n in first:
        if n not in seen:
            seen.append(n)
    assert seen == sorted(seen)
    listed = re.findall(r'<li id="ref(\d+)">', html)
    assert [int(n) for n in listed] == list(range(1, len(listed) + 1))
    assert "Representation Engineering" in html


def test_the_reference_list_follows_the_order_of_first_citation(
    html: str,
) -> None:
    bib = references.parse((ROOT / "docs/references.bib").read_text("utf-8"))
    refs = re.search(r'<ol class="refs">(.*?)</ol>', html, flags=re.S)
    assert refs is not None
    items = re.findall(r"<li [^>]*>(.*?)</li>", refs.group(1), flags=re.S)
    expected = [
        references.plain(references.format_line(references.find(bib, key)))
        for key in ("zou2025", "socher2013", "hendrycks2021")
    ]
    for item, want in zip(items, expected, strict=False):
        assert want.split(".")[0] in item
    assert len(items) >= 3


def test_the_cite_block_gives_bibtex_of_the_repository_and_the_paper(
    html: str,
) -> None:
    block = html[html.index('id="cite"') :]
    assert "<pre><code>@misc{tomography," in block
    assert "@misc{zou2025" in block
    assert content.REPO_URL in block


def test_footnotes_have_a_marker_and_a_note_with_a_back_link(
    html: str,
) -> None:
    assert (
        '<sup class="fn"><a class="ref" id="fnref1" href="#fn1">1</a></sup>'
        in html
    )
    assert '<aside class="footnotes"><hr><p id="fn1">' in html
    assert 'href="#fnref1">back</a>' in html


def test_every_figure_and_table_has_a_data_file_and_figures_a_pdf(
    built: Path,
) -> None:
    for spec in (*FIGURES, *APPENDIX):
        assert (built / "figures" / f"{spec.key}.svg").stat().st_size > 0
        assert (built / "figures" / f"{spec.key}.pdf").read_bytes()[
            :4
        ] == b"%PDF"
        size = (built / "data" / f"{spec.key}.csv").stat().st_size
        assert (size > 0) == (spec.key != "fig1"), spec.key
    for spec in (*TABLES, *APPENDIX_TABLES):
        header = (built / "data" / f"{spec.key}.csv").read_text("utf-8")
        assert header.splitlines()[0].count(",") >= 2


def test_assets_are_written_next_to_the_page(built: Path) -> None:
    assert (built / "site.css").is_file()
    assert (built / ".nojekyll").is_file()
    assert (built / "fonts" / "lmroman10-regular.woff2").is_file()
    assert (built / "fonts" / "GUST-FONT-LICENSE.txt").is_file()


def test_the_page_loads_no_remote_assets(html: str) -> None:
    assert not re.search(r'src="https?:', html)
    stylesheets = re.findall(r'<link [^>]*rel="stylesheet"[^>]*>', html)
    assert stylesheets == ['<link rel="stylesheet" href="site.css">']
    assert "fonts.googleapis" not in html
    assert not re.search(r'(?:src|href)="[^"]*cdn', html)


def test_no_dash_characters_in_the_page_stylesheet_or_text(html: str) -> None:
    texts = [
        html,
        (STATIC / "site.css").read_text("utf-8"),
        content.ABSTRACT,
        json.dumps([content.INTRODUCTION.paragraphs, content.METHOD.title]),
    ]
    assert not any(d in t for t in texts for d in DASHES)


def test_every_section_the_page_shows_has_its_items_placed_once(
    html: str,
) -> None:
    for spec in (*FIGURES, *TABLES, *APPENDIX, *APPENDIX_TABLES):
        assert html.count(f'id="{spec.key}"') == 1, spec.key


def test_the_page_names_no_person_beyond_the_cited_authors(html: str) -> None:
    head = html[: html.index('id="abstract"')]
    assert "author" not in head.lower()


def test_results_date_is_the_newest_run_start(tmp_path: Path) -> None:
    for digest, started in (
        ("a", "2026-10-01T10:00:00+00:00"),
        ("b", "2026-10-17T23:00:00+00:00"),
    ):
        run = tmp_path / "r1" / "gpt2" / digest
        run.mkdir(parents=True)
        (run / "meta.json").write_text(json.dumps({"started_at": started}))
        (run / "config.json").write_text("{}")
    assert (
        build_module.results_date(Manifest(tmp_path))
        == "Results of 17 October 2026"
    )
    assert build_module.results_date(Manifest(tmp_path / "none")) == "Results"


def test_a_missing_figure_the_text_refers_to_stops_the_page() -> None:
    bib = references.parse((ROOT / "docs/references.bib").read_text("utf-8"))
    doc = page.Document({}, {}, {}, "Results", bib)
    with pytest.raises(KeyError, match="fig1"):
        page.render_page(doc)


def test_build_is_byte_identical_twice(tmp_path: Path) -> None:
    specs = (FIGURES, TABLES, APPENDIX, APPENDIX_TABLES)
    for name in ("one", "two"):
        build_module.build(RESULTS, tmp_path / name, *specs)
    names = sorted(
        p.relative_to(tmp_path / "one")
        for p in (tmp_path / "one").rglob("*")
        if p.is_file()
    )
    assert names
    for rel in names:
        assert (tmp_path / "one" / rel).read_bytes() == (
            tmp_path / "two" / rel
        ).read_bytes(), rel


def test_command_line_build_writes_the_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake(results: Path, out: Path) -> page.Document:
        calls.append((results, out))
        return page.Document({}, {}, {}, "Results", {})

    monkeypatch.setattr(build_module, "build", fake)
    code = cli.main(["build", "--results", "r", "--out", str(tmp_path)])
    assert code == 0
    assert calls == [(Path("r"), tmp_path)]


def test_phone_layout_tightens_tables_so_the_caption_is_not_clipped() -> None:
    css = (STATIC / "site.css").read_text("utf-8")
    phone = css[
        css.index("@media (max-width: 600px)") : css.index("@media print")
    ]
    assert "padding: 0.18em 0.35em" in phone
    assert "font-size: 0.72em" in phone
    assert "padding: 1.4rem 16px 2rem" in phone
    assert "width: calc(100vw - 32px)" in phone
    assert "position: sticky" in phone


def test_table_two_has_grouped_headers_with_a_muted_exploratory_group(
    html: str,
) -> None:
    table = re.search(r"<table><caption>Table 2:.*?</table>", html, re.S)
    assert table is not None
    chunk = table.group(0)
    assert chunk.count('<th colspan="2" class="span">') == 4
    assert (
        '<th colspan="2" class="span muted">R5 honesty (exploratory)' in chunk
    )
    assert chunk.count("<tbody><tr>") == 1
    assert len(re.findall(r"<tr", chunk.split("<tbody>")[1])) == 10
    assert 'class="group"' not in chunk
    assert chunk.count('<th class="r">LAT</th>') == 5
