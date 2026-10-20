import csv
import json
import logging
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

import pytest

from tomography.site import __main__ as cli
from tomography.site import build as build_module
from tomography.site import content
from tomography.site import page
from tomography.site import pdf
from tomography.site import references
from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import checks

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STATIC = ROOT / "src/tomography/site/static"
DASHES = (chr(0x2014), chr(0x2013))
STUB = runpy.run_path(str(Path(__file__).parent / "_site_stub.py"))
MISSING = registry.missing()


@pytest.fixture(scope="module")
def site(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, page.Document]:
    assert MISSING == []
    out = tmp_path_factory.mktemp("site")
    return out, build_module.build(RESULTS, out)


@pytest.fixture(scope="module")
def built(site: tuple[Path, page.Document]) -> Path:
    return site[0]


@pytest.fixture(scope="module")
def html(built: Path) -> str:
    return (built / "index.html").read_text("utf-8")


@pytest.fixture(scope="module")
def css() -> str:
    return (STATIC / "site.css").read_text("utf-8")


def test_the_results_commit_is_the_last_commit_of_the_results() -> None:
    sha, built = build_module.results_commit(RESULTS)
    expected = subprocess.run(
        ["git", "log", "-1", "--format=%h", "--", "results"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert sha == expected
    assert re.fullmatch(r"\d{1,2} [A-Z][a-z]+ \d{4}", built)


def test_without_history_the_results_commit_is_empty(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    assert build_module.results_commit(tmp_path / "results") == ("", "")


def test_csv_files_hold_plain_text_not_tokens(
    site: tuple[Path, page.Document],
) -> None:
    view = page.Page(site[1])
    text = build_module._csv_text(
        ["A", "B"],
        [
            [
                "x",
                "[[cite:radford2019language,yang2025qwen3]] [[table:notation]]",
            ]
        ],
        view,
    )
    assert text == "A,B\nx,radford2019language; yang2025qwen3 Table A2\n"


def test_the_year_is_the_newest_run_start_year(tmp_path: Path) -> None:
    for digest, started in (
        ("a", "2026-10-01T10:00:00+00:00"),
        ("b", "2026-10-17T23:00:00+00:00"),
    ):
        run = tmp_path / "r1" / "gpt2" / digest
        run.mkdir(parents=True)
        (run / "meta.json").write_text(json.dumps({"started_at": started}))
        (run / "config.json").write_text("{}")
    assert build_module.newest_run_year(Manifest(tmp_path)) == "2026"
    assert build_module.newest_run_year(Manifest(tmp_path / "none")) == ""


def test_a_missing_figure_the_text_refers_to_stops_the_page() -> None:
    bib = references.parse((ROOT / "docs/references.bib").read_text("utf-8"))
    doc = page.Document({}, {}, {}, "", bib)
    with pytest.raises(KeyError):
        page.render_page(doc)


def test_command_line_build_writes_the_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake(results: Path, out: Path) -> page.Document:
        calls.append((results, out))
        return page.Document({}, {}, {}, "", {})

    monkeypatch.setattr(build_module, "build", fake)
    code = cli.main(["build", "--results", "r", "--out", str(tmp_path)])
    assert code == 0
    assert calls == [(Path("r"), tmp_path)]


def test_tables_are_booktabs_without_vertical_rules(css: str) -> None:
    assert "border-left" not in css
    assert "border-right" not in css
    assert "border-top: 1.5px solid" in css
    assert "border-bottom: 1.5px solid" in css
    assert "thead tr:last-child th {\n  border-bottom: 1px solid" in css
    assert "tbody tr.group td {\n  border-top: 0.5px solid" in css


def test_numeric_cells_do_not_wrap_and_use_tabular_figures(css: str) -> None:
    assert "td.k {\n  white-space: nowrap;\n}" in css
    assert "td.r," not in css
    assert "font-variant-numeric: tabular-nums" in css
    assert "td.id {\n  overflow-wrap: anywhere;\n}" in css
    assert "td:not(.r)" not in css


def test_table_text_is_compact_but_not_below_the_readable_floor(
    css: str,
) -> None:
    rule = css.split(chr(10) + "table {")[1].split("}")[0]
    size = float(rule.split("font-size:")[1].split("em")[0])
    assert 0.78 <= size <= 0.8
    cells = css.split(chr(10) + "th,\ntd {")[1].split("}")[0]
    assert "padding: 0.16em 0.25em;" in cells


def test_captions_of_figures_and_tables_share_one_rule(css: str) -> None:
    assert "figcaption,\n.caption {\n  font-size: 0.9em;" in css
    assert "text-align: justify" in css
    assert "caption {" not in css.replace(".caption {", "").replace(
        "figcaption", ""
    )
    assert "position: sticky" not in css


def test_phones_scroll_the_figure_not_the_page(css: str) -> None:
    phone = css[
        css.index("@media screen and (max-width: 600px)") : css.index(
            "@media print"
        )
    ]
    assert ".fig-scroll {\n    overflow-x: auto;" in phone
    assert "min-width: 560px" in phone


def test_paper_never_gets_the_phone_layout(css: str) -> None:
    assert "@media (max-width: 600px)" not in css
    assert css.count("max-width: 600px") == 1


def test_printed_tables_have_no_scroll_box_and_wide_ones_are_set_smaller(
    css: str,
) -> None:
    printed = css[css.index("@media print") :].split("\n}\n", 1)[0]
    assert ".table .scroll {\n    overflow: visible;" in printed
    assert "margin-left: -" not in printed
    assert ".table.wide table {\n    font-size: 0.68em;" in printed
    assert (
        ".table.wide th,\n  .table.wide td {\n    padding: 0.18em 0.12em;"
        in printed
    )


def test_a_printed_figure_shrinks_to_one_page_instead_of_splitting(
    css: str,
) -> None:
    printed = css[css.index("@media print") :].split("\n}\n", 1)[0]
    assert (
        "  figure {\n    display: flex;\n    flex-direction: column;" in printed
    )
    assert "max-height: 840px;" in printed
    assert "figure .fig-scroll {\n    display: flex;" in printed
    assert "justify-content: center;" in printed
    assert "figure .fig-scroll img {\n    width: auto;" in printed
    assert "max-height: 100%;" in printed
    assert "figure,\n  pre,\n  tr {\n    break-inside: avoid;" in printed


def test_the_bibtex_block_wraps(css: str) -> None:
    assert (
        "pre.bibtex {\n  white-space: pre-wrap;\n  overflow-wrap: anywhere;"
        in css
    )


def test_the_pdf_button_is_a_quiet_outlined_link(css: str) -> None:
    block = css[css.index(".pdf-button {") :].split("}", 1)[0]
    assert "border: 1px solid var(--ink)" in block
    assert "border-radius: 3px" in block
    assert "padding: 0.3em 0.9em" in block
    assert "box-shadow" not in block
    assert "background:" not in block
    assert "transition: background-color" in block
    assert "text-decoration: none" in block
    assert ".pdf-button:hover {" in css
    assert ".pdf-button:focus-visible {" in css
    assert "outline: 2px solid var(--ink)" in css


def test_print_rules(css: str) -> None:
    printed = css[css.index("@media print") :]
    assert "size: letter" in printed
    assert "margin: 1in 1.5in" in printed
    assert "thead {\n    display: table-header-group;" in printed
    block = "  figure,\n  pre,\n  tr {\n    break-inside: avoid;\n  }"
    assert block in printed
    assert ".data,\n  .pdf {\n    display: none;" in printed


def test_text_colours_are_the_two_allowed_greys(css: str) -> None:
    assert "--ink: #1a1a1a" in css
    assert "--muted: #4d4d4d" in css
    colours = set(re.findall(r"(?<![\w-])color:\s*([^;]+);", css))
    assert colours <= {"var(--ink)", "var(--muted)", "var(--link)", "inherit"}


def test_no_dash_characters_in_the_stylesheet_or_the_sources(css: str) -> None:
    assert not any(d in css for d in DASHES)
    for path in (ROOT / "src/tomography/site").rglob("*.py"):
        text = path.read_text("utf-8")
        assert not any(d in text for d in DASHES), path
    for name in ("references.bib",):
        text = (ROOT / "docs" / name).read_text("utf-8")
        assert not any(d in text for d in DASHES), name


def test_the_stub_page_is_english_scriptless_and_loads_no_remote_assets() -> (
    None
):
    html = page.render_page(STUB["document"]())
    assert html.startswith("<!doctype html>")
    assert '<html lang="en">' in html
    assert "<script" not in html
    assert not re.search(r'src="https?:', html)
    assert re.findall(r'<link [^>]*rel="stylesheet"[^>]*>', html) == [
        '<link rel="stylesheet" href="site.css">'
    ]


def test_the_stub_page_has_no_aria_label_longer_than_twenty_characters() -> (
    None
):
    html = page.render_page(STUB["document"]())
    for label in re.findall(r'aria-label="([^"]*)"', html):
        assert len(label) <= 20, label


def test_the_title_block_has_a_title_a_repo_link_and_no_author() -> None:
    html = page.render_page(STUB["document"]())
    head = html[html.index('<header class="title">') : html.index("</header>")]
    assert f"<h1>{content.TITLE}</h1>" in head
    assert content.REPO_TEXT in head
    assert head.count("<p") == 2
    assert "author" not in head.lower()


def test_assets_and_files_by_printed_number(
    site: tuple[Path, page.Document],
) -> None:
    built, doc = site
    figure_number, table_number = page.numbers(doc)
    for key, number in figure_number.items():
        name = f"fig-{number}"
        assert (built / "figures" / f"{name}.svg").stat().st_size > 0
        pdf = (built / "figures" / f"{name}.pdf").read_bytes()
        assert pdf[:4] == b"%PDF"
        has_rows = bool({**doc.figures, **doc.appendix}[key].output.csv.strip())
        assert (built / "data" / f"{name}.csv").is_file() == has_rows, key
    for key, number in table_number.items():
        text = (built / "data" / f"table-{number}.csv").read_text("utf-8")
        assert text.splitlines()[0].count(",") >= 1, key
        assert "[[" not in text
    names = {p.name for p in (built / "data").iterdir()}
    assert not any(n.startswith("figure-") for n in names)
    assert (built / "site.css").is_file()
    assert (built / ".nojekyll").is_file()
    assert (built / "fonts" / "lmroman10-regular.woff2").is_file()


def test_the_method_figure_has_no_data_file_and_no_data_link(
    site: tuple[Path, page.Document], html: str
) -> None:
    built, doc = site
    number = page.numbers(doc)[0]["method"]
    assert not (built / "data" / f"fig-{number}.csv").exists()
    assert f"data/fig-{number}.csv" not in html


def test_the_built_page_resolves_every_token_and_link(html: str) -> None:
    assert "[[" not in html
    ids = re.findall(r'\bid="([^"]+)"', html)
    assert len(ids) == len(set(ids))
    hrefs = re.findall(r'href="#([^"]+)"', html)
    assert [h for h in hrefs if h not in set(ids)] == []


@pytest.mark.parametrize(
    "pattern",
    [
        r'<div class="table(?: wide)?" id="table-(\d+)"',
        r'<div class="table(?: wide)?" id="table-A(\d+)"',
        r'<figure id="fig-(\d+)"',
        r'<figure id="fig-A(\d+)"',
    ],
)
def test_the_built_page_numbers_figures_and_tables_in_page_order(
    html: str, pattern: str
) -> None:
    numbers = [int(n) for n in re.findall(pattern, html)]
    assert numbers
    assert numbers == list(range(1, len(numbers) + 1))


def test_the_built_masthead_links_the_pdf_and_names_no_commit(
    html: str,
) -> None:
    assert "Results at commit" not in html
    assert 'class="date"' not in html
    assert (
        f'<p class="pdf"><a class="pdf-button" href="{pdf.NAME}" '
        'target="_blank" rel="noopener">View PDF</a></p></header>'
    ) in html


def test_the_built_page_text_names_no_commit_or_masthead(html: str) -> None:
    sha, _ = build_module.results_commit(RESULTS)
    text = re.sub(
        r"<[^>]+>", " ", re.sub(r"<svg.*?</svg>", "", html, flags=re.S)
    )
    assert "masthead" not in text
    assert sha not in text


def test_the_built_cite_block_year_is_the_results_commit_year(
    site: tuple[Path, page.Document], html: str
) -> None:
    _, built = build_module.results_commit(RESULTS)
    assert site[1].year == built[-4:]
    assert f"year = {{{built[-4:]}}}," in html


def test_the_built_page_has_no_dashes_backticks_or_stray_underscores(
    html: str,
) -> None:
    assert not any(d in html for d in DASHES)
    body = re.sub(r"<pre.*?</pre>", "", html, flags=re.S)
    body = re.sub(r"<svg.*?</svg>", "", body, flags=re.S)
    body = re.sub(r"<table.*?</table>", "", body, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", body)
    text = re.sub(r"doi:\S+", "", text)
    assert "`" not in text
    assert "_" not in text


def test_the_built_page_lists_only_cited_references(html: str) -> None:
    listed = re.findall(r'<li id="ref(\d+)">', html)
    cites = {int(n) for n in re.findall(r'href="#ref(\d+)">\[', html)}
    assert cites == set(range(1, len(listed) + 1))
    assert html.index('id="references"') < html.index('id="appendix"')


def _bare_handles(text: str) -> list[str]:
    return [
        text[m.start() - 11 : m.end()]
        for m in re.finditer("javierdejesusda", text)
        if text[m.start() - 11 : m.end()] != "github.com/javierdejesusda"
    ]


def test_the_page_names_no_person_beyond_the_cited_authors(html: str) -> None:
    assert _bare_handles(html) == []


def test_a_bare_handle_in_the_text_would_be_caught() -> None:
    assert _bare_handles("by javierdejesusda in the abstract") != []
    assert _bare_handles("at github.com/javierdejesusda/tomography") == []


def _difference(rel: Path, first: bytes, second: bytes) -> str:
    """Describes where two builds of one file first differ."""
    offset = next(
        (
            i
            for i, (x, y) in enumerate(zip(first, second, strict=False))
            if x != y
        ),
        min(len(first), len(second)),
    )
    lo = max(0, offset - 100)
    return (
        f"{rel} differs at offset {offset}\n"
        f"one: {first[lo : offset + 100]!r}\n"
        f"two: {second[lo : offset + 100]!r}"
    )


def test_the_difference_report_names_the_file_offset_and_context() -> None:
    report = _difference(Path("index.html"), b"abc" * 100, b"abc" * 99 + b"abx")
    assert "index.html differs at offset 299" in report
    assert "one: " in report
    assert "two: " in report
    assert report.count("abx") == 1
    assert "differs at offset 3" in _difference(Path("a"), b"abc", b"abcd")


def test_build_is_byte_identical_in_two_fresh_processes(
    tmp_path: Path,
) -> None:
    outs = [tmp_path / "one", tmp_path / "two"]
    for seed, out in zip(("1", "2"), outs, strict=True):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tomography.site",
                "build",
                "--results",
                str(RESULTS),
                "--out",
                str(out),
            ],
            check=True,
            capture_output=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
    names = sorted(
        p.relative_to(outs[0]) for p in outs[0].rglob("*") if p.is_file()
    )
    assert names
    for rel in names:
        first, second = [(out / rel).read_bytes() for out in outs]
        assert first == second, _difference(rel, first, second)


def test_every_figure_passes_the_palette_font_and_text_guards(
    site: tuple[Path, page.Document],
) -> None:
    doc = site[1]
    for key, entry in doc.figures.items():
        assert checks.guard_problems(entry.output.figure) == [], key
    for key, entry in doc.appendix.items():
        problems = checks.guard_problems(entry.output.figure, appendix=True)
        assert problems == [], key


def test_the_built_page_has_no_pooled_count_of_ten(html: str) -> None:
    text = re.sub(r"<[^>]+>", " ", html)
    assert not re.search(r"\b\d+ of (the )?10\b(?! readers)", text)


def test_every_appendix_item_is_cited_in_the_text(
    html: str,
) -> None:
    body = html
    for kind, ids in (
        (
            "fig",
            [f"A{i}" for i in range(1, len(registry.APPENDIX_FIGURES) + 1)],
        ),
        (
            "table",
            [f"A{i}" for i in range(1, len(registry.APPENDIX_TABLES) + 1)],
        ),
    ):
        for number in ids:
            assert f'href="#{kind}-{number}"' in body, (kind, number)


def test_the_font_timestamp_warning_is_silenced_but_errors_are_not(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = logging.getLogger("fontTools.ttLib.tables._h_e_a_d")
    with caplog.at_level(logging.WARNING):
        log.warning("'modified' timestamp seems very low")
        log.error("a real error")
    assert [r.getMessage() for r in caplog.records] == ["a real error"]


def test_clip_ids_are_numbered_in_order_not_hashed() -> None:
    svg = (
        b'<defs><clipPath id="p3b30372de5"/><clipPath id="pe4a55143d0"/>'
        b'</defs><g clip-path="url(#pe4a55143d0)"/>'
        b'<use xlink:href="#m1234567890"/>'
    )
    out = build_module.stable_clip_ids(svg)
    assert out == (
        b'<defs><clipPath id="clip1"/><clipPath id="clip2"/>'
        b'</defs><g clip-path="url(#clip2)"/>'
        b'<use xlink:href="#m1234567890"/>'
    )


def test_each_contents_entry_stays_on_one_line(css: str) -> None:
    assert ".contents a {\n  white-space: nowrap;\n}" in css


def test_wide_tables_break_out_of_the_text_column_on_large_screens(
    css: str,
) -> None:
    block = css.split("@media (min-width: 1000px)")[1].split("\n}\n")[0]
    assert ".table.wide {" in block
    assert "margin-left: -6rem;" in block
    assert "margin-right: -6rem;" in block


def test_the_resolved_abstract_stays_short(
    site: tuple[Path, page.Document],
) -> None:
    view = page.Page(site[1])
    words = len(view.plain(content.ABSTRACT).split())
    assert 150 <= words <= 180, words


def test_grouped_csv_columns_carry_their_group(
    site: tuple[Path, page.Document],
) -> None:
    view = page.Page(site[1])
    text = build_module._csv_text(
        ["Model", "Random", "PCA", "Random"],
        [["x", "1", "2", "3"]],
        view,
        (("Sentiment", 2, False), ("Truth", 1, False)),
    )
    assert text.splitlines()[0] == (
        "Model,Sentiment: Random,Sentiment: PCA,Truth: Random"
    )


def test_every_grouped_table_csv_has_unique_headers(
    site: tuple[Path, page.Document],
) -> None:
    out, doc = site
    view = page.Page(doc)
    grouped = 0
    for key, tab in {**doc.tables, **doc.appendix_tables}.items():
        header = (
            (out / "data" / f"table-{view.table_number[key]}.csv")
            .read_text("utf-8")
            .splitlines()[0]
        )
        names = next(csv.reader([header]))
        assert len(names) == len(set(names)), (key, names)
        if tab.output.groups:
            grouped += 1
            for label, _, _ in tab.output.groups:
                assert any(n.startswith(f"{label}: ") for n in names), key
    assert grouped >= 3


def _words(fragment: str) -> int:
    fragment = re.sub(r"<style.*?</style>", " ", fragment, flags=re.S)
    fragment = re.sub(r"<svg.*?</svg>", " ", fragment, flags=re.S)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    return len(fragment.split())


def test_the_main_prose_is_about_four_thousand_words(html: str) -> None:
    main = html[: html.index('id="references"')]
    captions = re.findall(
        r"<(?:figcaption|caption)[^>]*>.*?</(?:figcaption|caption)>",
        main,
        flags=re.S,
    )
    tables = re.findall(r"<table.*?</table>", main, flags=re.S)
    prose = _words(main) - sum(map(_words, captions)) - sum(map(_words, tables))
    assert 3000 <= prose <= 4600, prose


def test_the_page_opens_with_the_summary_and_the_overview(html: str) -> None:
    assert html.index('<p class="tldr">') < html.index('<figure id="fig-1">')
    assert html.index('<figure id="fig-1">') < html.index('id="abstract"')
    assert html.index('id="abstract"') < html.index('id="key_findings"')


def test_every_count_token_of_the_text_has_at_most_two_homes() -> None:
    texts = [content.ABSTRACT, content.TLDR]
    sections = [*content.BODY, *content.APPENDICES, content.KEY_FINDINGS]
    while sections:
        section = sections.pop()
        texts += [*section.paragraphs, *section.after]
        for head, text in (*section.runin, *section.listed):
            texts += [head, text]
        sections += list(section.subsections)
    tokens = re.findall(r"\[\[(?:count|runs):([^\]]+)\]\]", " ".join(texts))
    for address in set(tokens):
        assert tokens.count(address) <= 2, address
