import dataclasses
from html.parser import HTMLParser
from pathlib import Path
import re
import runpy

from matplotlib.figure import Figure
import pytest

from tomography.site import page
from tomography.site import references
from tomography.site import registry
from tomography.site import stats

STUB = runpy.run_path(str(Path(__file__).parent / "_site_stub.py"))
MINUS = chr(0x2212)
SVG = STUB["SVG"]
BIB = Path(__file__).resolve().parents[1] / "docs" / "references.bib"


def _figure(
    numbers: dict[str, float] | None = None,
    counts: dict[str, dict[str, stats.RoleCount]] | None = None,
    csv: str = "a,b\n1,2\n",
    claim: str = "Claim.",
) -> page.FigureEntry:
    output = registry.FigureOutput(
        Figure(), claim, "Detail.", numbers or {}, csv, counts or {}
    )
    return page.FigureEntry("x", output, SVG, b"%PDF-1.4")


def _table(
    numbers: dict[str, float] | None = None,
    rows: list[list[str]] | None = None,
    claim: str = "Claim.",
) -> page.TableEntry:
    output = registry.TableOutput(
        claim,
        "Detail.",
        ["A", "B"],
        rows or [["x", "1"]],
        "lr",
        [],
        numbers or {},
    )
    return page.TableEntry("x", output)


def _doc(
    figures: dict[str, page.FigureEntry] | None = None,
    tables: dict[str, page.TableEntry] | None = None,
    appendix: dict[str, page.FigureEntry] | None = None,
    appendix_tables: dict[str, page.TableEntry] | None = None,
    bib: dict[str, references.Entry] | None = None,
) -> page.Document:
    return page.Document(
        figures or {},
        tables or {},
        appendix or {},
        "",
        bib or {},
        appendix_tables or {},
    )


def test_main_items_are_numbered_in_order_of_appearance() -> None:
    doc = _doc(
        {"method": _figure(), "finders": _figure()},
        {"checkpoints": _table(), "reading": _table()},
    )
    figure_number, table_number = page.numbers(doc)
    assert figure_number == {"method": "1", "finders": "2"}
    assert table_number == {"checkpoints": "1", "reading": "2"}


def test_appendix_items_are_numbered_a1_a2_in_registry_order() -> None:
    doc = _doc(
        appendix={"a_layers": _figure(), "a_wording": _figure()},
        appendix_tables={"assets": _table(), "reading_full": _table()},
    )
    figure_number, table_number = page.numbers(doc)
    assert figure_number == {"a_layers": "A1", "a_wording": "A2"}
    assert table_number == {"assets": "A1", "reading_full": "A2"}


def test_fig_and_table_tokens_become_numbered_links_to_stable_ids() -> None:
    doc = _doc({"method": _figure()}, {"checkpoints": _table()})
    text = page.Page(doc).text("See [[fig:method]] and [[table:checkpoints]].")
    assert 'href="#fig-1">Figure&nbsp;1</a>' in text
    assert 'href="#table-1">Table&nbsp;1</a>' in text


def test_a_section_token_links_the_section_number() -> None:
    view = page.Page(_doc())
    assert view.text("[[sec:finders]]") == (
        '<a class="ref" href="#finders">Section&nbsp;4.1</a>'
    )
    assert "Appendix&nbsp;C" in view.text("[[sec:appendix_reading]]")


def test_figure_markup_names_the_graphic_and_keeps_the_caption_outside() -> (
    None
):
    doc = _doc({"method": _figure(csv="")})
    figure = page.Page(doc).figure("method")
    assert '<figure id="fig-1">' in figure
    assert '<div class="fig-scroll"><svg ' in figure
    assert 'role="img" aria-labelledby="fig-1-cap"' in figure
    assert "aria-label=" not in figure
    assert '<figcaption id="fig-1-cap">Figure 1: Claim. Detail.' in figure
    assert figure.index("</div>") < figure.index("<figcaption")
    assert 'href="figures/fig-1.pdf"' in figure
    assert "data/fig-1.csv" not in figure
    with_rows = page.Page(_doc({"method": _figure()})).figure("method")
    assert 'href="data/fig-1.csv"' in with_rows


def test_inline_svg_ids_are_prefixed_and_the_style_is_kept_once() -> None:
    first = page.inline_svg(SVG, "fig-1", True)
    second = page.inline_svg(SVG, "fig-2", False)
    for text, prefix in ((first, "fig-1-"), (second, "fig-2-")):
        assert f'id="{prefix}p1"' in text
        assert f'id="{prefix}axes_1"' in text
        assert f"url(#{prefix}p1)" in text
        assert f'href="#{prefix}p1"' in text
        assert 'id="p1"' not in text
        assert 'aria-labelledby="' in text
    assert "<style" in first
    assert "<style" not in second


def test_an_id_repeated_inside_one_svg_is_made_unique() -> None:
    raw = (
        b'<svg xmlns="http://www.w3.org/2000/svg"><g id="tile"/>'
        b'<g id="tile"/><g id="tile"/></svg>'
    )
    text = page.inline_svg(raw, "fig-7", True)
    ids = re.findall(r'id="([^"]+)"', text)
    assert len(ids) == len(set(ids)) == 3
    assert ids[0] == "fig-7-tile"


def test_an_appendix_figure_is_a_lazy_image_not_inline_svg() -> None:
    doc = _doc(appendix={"a_layers": _figure()})
    figure = page.Page(doc).figure("a_layers")
    assert '<figure id="fig-A1">' in figure
    assert '<img src="figures/fig-A1.svg" loading="lazy"' in figure
    assert 'alt="Figure A1"' in figure
    assert "<svg" not in figure
    assert 'href="data/fig-A1.csv"' in figure


def test_table_markup_has_its_caption_outside_the_scroller() -> None:
    doc = _doc(tables={"checkpoints": _table()})
    table = page.Page(doc).table("checkpoints")
    assert "<caption" not in table
    assert table.startswith(
        '<div class="table" id="table-1">'
        '<p class="caption" id="table-1-cap">Table 1: Claim. Detail.'
    )
    assert table.index("</p>") < table.index('<div class="scroll">')
    assert '<table aria-describedby="table-1-cap">' in table
    assert 'href="data/table-1.csv"' in table


def test_captions_and_cells_resolve_tokens() -> None:
    bib = references.parse(BIB.read_text("utf-8"))
    doc = _doc(
        {"method": _figure(claim="See [[table:checkpoints]].")},
        {
            "checkpoints": _table(
                rows=[["a", "[[cite:radford2019language]]"]],
                claim="See [[fig:method]].",
            )
        },
        bib=bib,
    )
    view = page.Page(doc)
    assert 'href="#table-1">Table&nbsp;1</a>' in view.figure("method")
    table = view.table("checkpoints")
    assert 'href="#fig-1">Figure&nbsp;1</a>' in table
    assert '<td class="r"><a class="ref" href="#ref1">[1]</a></td>' in table
    assert "[[" not in table


def test_num_token_prints_integers_plainly_and_decimals_to_three_places() -> (
    None
):
    doc = _doc(
        {
            "method": _figure(
                {
                    "n": 1234.0,
                    "zero": 0.0,
                    "nzero": -0.0,
                    "neg": -0.5,
                    "p": 0.12345,
                }
            )
        }
    )
    view = page.Page(doc)
    assert view.text("[[num:fig:method.n]]") == "1,234"
    assert view.text("[[num:fig:method.zero]]") == "0"
    assert view.text("[[num:fig:method.nzero]]") == "0"
    assert view.text("[[num:fig:method.neg]]") == f"{MINUS}0.500"
    assert view.text("[[num:fig:method.p]]") == "0.123"


def test_num_token_reads_a_table_when_asked() -> None:
    doc = _doc(tables={"checkpoints": _table({"k": 3.0})})
    assert page.Page(doc).text("[[num:table:checkpoints.k]]") == "3"


def test_count_token_writes_the_per_role_phrase() -> None:
    counts = {
        "pca": {
            "gpt2": stats.RoleCount(4, 4),
            "base": stats.RoleCount(3, 3),
            "post": stats.RoleCount(2, 3),
        }
    }
    doc = _doc({"method": _figure(counts=counts)})
    assert page.Page(doc).text("[[count:fig:method.pca]]") == (
        "4 of 4 GPT-2, 3 of 3 Qwen3 base and 2 of 3 post-trained checkpoints"
    )


def test_runs_token_writes_the_per_role_phrase_over_runs() -> None:
    counts = {
        "dist": {
            "gpt2": stats.RoleCount(3, 3),
            "base": stats.RoleCount(3, 3),
            "post": stats.RoleCount(3, 6),
        }
    }
    doc = _doc({"method": _figure(counts=counts)})
    view = page.Page(doc)
    expected = (
        "3 of 3 GPT-2 runs, 3 of 3 Qwen3 base runs and 3 of 6 post-trained runs"
    )
    assert view.text("[[runs:fig:method.dist]]") == expected
    assert view.plain("[[runs:fig:method.dist]]") == expected


@pytest.mark.parametrize(
    "token",
    [
        "[[runs:fig:method.absent]]",
        "[[fig:absent]]",
        "[[table:absent]]",
        "[[sec:absent]]",
        "[[num:fig:method.absent]]",
        "[[count:fig:method.absent]]",
        "[[num:fig:absent.x]]",
        "[[bogus]]",
    ],
)
def test_a_token_that_cannot_resolve_stops_the_page(token: str) -> None:
    doc = _doc({"method": _figure()})
    with pytest.raises(KeyError):
        page.Page(doc).text(token)


def test_cite_and_footnote_tokens_still_resolve() -> None:
    bib = references.parse(
        "@inproceedings{zou2025,title={T},author={Zou, A.},year={2025},"
        "booktitle={B}}"
    )
    view = page.Page(_doc(bib=bib))
    assert '<a class="ref" href="#ref1">[1]</a>' in view.text(
        "[[cite:zou2025]]"
    )
    assert 'id="fnref1"' in view.text("x[[fn:a note]]")


def test_the_reference_list_holds_only_cited_entries_by_exact_key() -> None:
    bib = references.parse(BIB.read_text("utf-8"))
    view = page.Page(_doc(bib=bib))
    view.text("[[cite:radford2019language]][[cite:yang2025qwen3]]")
    listing = view.references()
    assert listing.count("<li ") == 2
    assert "Language Models are Unsupervised Multitask Learners" in listing
    assert "arXiv:2505.09388</a>" in listing
    assert "https://arxiv.org/abs/2505.09388" in listing
    assert ">https://" not in listing


def test_a_prefix_of_a_key_does_not_cite() -> None:
    bib = references.parse(BIB.read_text("utf-8"))
    view = page.Page(_doc(bib=bib))
    view.text("[[cite:zou2025]]")
    with pytest.raises(KeyError):
        view.references()


def _grouped(row_groups: tuple[int, ...] | None) -> str:
    output = registry.TableOutput(
        "Claim.",
        "Detail.",
        ["A", "B"],
        [["a", "1"], ["", "2"], ["b", "3"], ["", "4"]],
        "lr",
        [],
        {},
        row_groups=row_groups,
    )
    doc = _doc(tables={"checkpoints": page.TableEntry("checkpoints", output)})
    return page.Page(doc).table("checkpoints")


def test_row_groups_set_the_rows_that_get_a_group_rule() -> None:
    html = _grouped((0, 2))
    assert html.count('<tr class="group">') == 1
    assert '<tr class="group"><td class="k">b</td>' in html


def test_a_declared_empty_row_groups_draws_no_group_rule() -> None:
    assert 'class="group"' not in _grouped(())


def test_without_row_groups_a_labelled_row_starts_a_group() -> None:
    assert _grouped(None).count('<tr class="group">') == 1


class Ids(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.hrefs: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        for name, value in attrs:
            if name == "id" and value:
                self.ids.append(value)
            if name == "href" and value:
                self.hrefs.append(value)


@pytest.fixture(scope="module")
def full() -> tuple[page.Document, str]:
    doc = STUB["document"]()
    return doc, page.render_page(doc)


def test_the_stub_page_has_unique_ids_and_resolving_links(
    full: tuple[page.Document, str],
) -> None:
    parser = Ids()
    parser.feed(full[1])
    assert len(parser.ids) == len(set(parser.ids))
    missing = [
        h for h in parser.hrefs if h.startswith("#") and h[1:] not in parser.ids
    ]
    assert missing == []
    assert "[[" not in full[1]


def test_printed_numbers_match_ids_and_registry_order(
    full: tuple[page.Document, str],
) -> None:
    doc, html = full
    figure_number, table_number = page.numbers(doc)
    assert figure_number["teaser"] == "1"
    assert sorted(int(figure_number[k]) for k in registry.MAIN_FIGURES) == list(
        range(1, 11)
    )
    assert sorted(int(table_number[k]) for k in registry.MAIN_TABLES) == [1, 2]
    for key, number in figure_number.items():
        assert f'<figure id="fig-{number}">' in html, key
        assert f"Figure {number}: " in html, key
    for key, number in table_number.items():
        assert f'id="table-{number}"' in html, key
        assert f"Table {number}: " in html, key


def test_sections_run_in_order_with_references_before_the_appendix(
    full: tuple[page.Document, str],
) -> None:
    html = full[1]
    heads = [
        t.replace("&ensp;", " ")
        for t in re.findall(r"<h2[^>]*>([^<]+)</h2>", html)
    ]
    assert heads[:2] == ["Abstract", "Key findings"]
    assert heads[2:8] == [
        "1 Introduction",
        "2 Related work",
        "3 Method",
        "4 Results",
        "5 Discussion and limitations",
        "6 Reproducibility",
    ]
    assert heads[8] == "References"
    assert heads[9] == "Cite this repository"
    assert heads[10:] == [
        "Appendix A Assets and notation",
        "Appendix B Method details",
        "Appendix C Reading details",
        "Appendix D Wording",
        "Appendix E Truth and honesty",
        "Appendix F Steering details",
        "Appendix G Knowledge",
        "Appendix H Approximate answers",
        "Appendix I Post-training details",
        "Appendix J Other experiments",
    ]
    subs = [
        t.replace("&ensp;", " ")
        for t in re.findall(r"<h3[^>]*>([^<]+)</h3>", html)
    ]
    assert subs[0] == "3.1 Reading a concept"
    assert any(s.startswith("4.5 ") for s in subs)


def test_the_contents_line_links_every_section(
    full: tuple[page.Document, str],
) -> None:
    html = full[1]
    line = re.search(r'<p class="contents">(.*?)</p>', html, flags=re.S)
    assert line is not None
    text = re.sub(r"<[^>]+>", "", line.group(1)).replace("&ensp;", " ")
    assert text.startswith(
        "Contents: Key findings, 1 Introduction, 2 Related work, 3 Method"
    )
    assert text.endswith("6 Reproducibility, References, Appendix")
    ids = set(re.findall(r'id="([^"]+)"', html))
    for target in re.findall(r'href="#([^"]+)"', line.group(1)):
        assert target in ids, target


def test_only_cited_references_are_listed_in_order_of_first_citation(
    full: tuple[page.Document, str],
) -> None:
    html = full[1]
    listed = re.findall(r'<li id="ref(\d+)">', html)
    assert [int(n) for n in listed] == list(range(1, len(listed) + 1))
    cites = [int(n) for n in re.findall(r'href="#ref(\d+)">\[', html)]
    assert set(cites) == set(range(1, len(listed) + 1))
    first: list[int] = []
    for n in cites:
        if n not in first:
            first.append(n)
    assert first == sorted(first)
    assert "(2023). Representation Engineering" in html


def test_the_masthead_links_the_pdf_instead_of_naming_the_commit(
    full: tuple[page.Document, str],
) -> None:
    html = full[1]
    assert "Results at commit" not in html
    assert 'class="date"' not in html
    assert (
        '<p class="pdf"><a class="pdf-button" href="tomography-results.pdf" '
        'target="_blank" rel="noopener">View PDF</a></p></header>'
    ) in html
    assert "year = {2027}," in html


def _visible_text(html: str) -> str:
    body = re.sub(r"<svg.*?</svg>", "", html, flags=re.S)
    return re.sub(r"<[^>]+>", " ", body)


def test_no_commit_or_masthead_is_named_in_the_page_text(
    full: tuple[page.Document, str],
) -> None:
    text = _visible_text(full[1])
    assert "masthead" not in text
    assert "Results at commit" not in text
    assert "abc1234" not in text
    assert "blob/abc1234/" in full[1]


def test_the_reproducibility_and_appendix_sentences_name_no_commit(
    full: tuple[page.Document, str],
) -> None:
    text = " ".join(_visible_text(full[1]).split())
    assert (
        "from the result files in the results directory of the repository; "
        "each figure and table links the data file it draws"
    ) in text
    assert (
        "Links to the findings document open it at the commit of these results."
    ) in text
    assert "reported in the findings document of the repository. They" in text


def test_appendix_figures_are_lazy_images_and_main_figures_inline(
    full: tuple[page.Document, str],
) -> None:
    html = full[1]
    assert html.count("<svg ") == len(registry.MAIN_FIGURES)
    assert html.count('loading="lazy"') == len(registry.APPENDIX_FIGURES)
    assert html.count("<style type=") == 1


def test_the_handle_is_only_in_repository_urls(
    full: tuple[page.Document, str],
) -> None:
    html = full[1]
    for match in re.finditer("javierdejesusda", html):
        around = html[max(0, match.start() - 12) : match.end()]
        assert "github.com/javierdejesusda" in around
    assert "author" not in html[: html.index('id="abstract"')].lower()


def test_no_dash_characters_and_no_backticks_in_the_page(
    full: tuple[page.Document, str],
) -> None:
    html = full[1]
    assert chr(0x2014) not in html
    assert chr(0x2013) not in html
    body = re.sub(r"<pre.*?</pre>", "", html, flags=re.S)
    assert "`" not in re.sub(r"<[^>]+>", " ", body)


def test_the_cite_block_uses_the_exact_paper_key(
    full: tuple[page.Document, str],
) -> None:
    block = full[1][full[1].index('id="cite"') :]
    assert "@misc{tomography," in block
    assert f"{references.PAPER_KEY}," in block


def test_table_numbers_increase_in_page_order(
    full: tuple[page.Document, str],
) -> None:
    html = full[1]
    main = [
        int(n) for n in re.findall(r'<div class="table" id="table-(\d+)"', html)
    ]
    appendix = [
        int(n)
        for n in re.findall(r'<div class="table" id="table-A(\d+)"', html)
    ]
    assert main == sorted(main) == list(range(1, len(main) + 1))
    assert appendix == list(range(1, len(appendix) + 1))
    figures = [int(n) for n in re.findall(r'<figure id="fig-A(\d+)"', html)]
    assert figures == list(range(1, len(figures) + 1))


NBSP = "\u00a0"


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("0.065 ± 0.072", f"0.065 ±{NBSP}0.072"),
        (
            f"0.011 [{MINUS}0.006, 0.032]",
            f"0.011 [{MINUS}0.006,{NBSP}0.032]",
        ),
        (
            f"[{MINUS}0.031, {MINUS}0.004]",
            f"[{MINUS}0.031,{NBSP}{MINUS}0.004]",
        ),
        ("0.622 (0.490)", "0.622 (0.490)"),
        ("1 of 1", f"1{NBSP}of{NBSP}1"),
        ("no contrast", "no contrast"),
        ("0.186*", "0.186*"),
    ],
)
def test_a_numeric_cell_breaks_only_between_its_numbers(
    cell: str, expected: str
) -> None:
    assert page.glue(cell) == expected


def test_numeric_cells_wrap_through_their_break_points_only() -> None:
    output = registry.TableOutput(
        "Claim.", "Detail.", ["A", "B"], [["x", "1 of 3"]], "lr", [], {}
    )
    doc = _doc(tables={"checkpoints": page.TableEntry("x", output)})
    cell = f'<td class="r">1{NBSP}of{NBSP}3</td>'
    assert cell in page.Page(doc).table("checkpoints")


def test_plain_text_resolves_every_token_type_without_markup() -> None:
    doc = _doc(
        {"method": _figure({"n": 3.0})},
        {"checkpoints": _table()},
        bib=references.parse(BIB.read_text("utf-8")),
    )
    view = page.Page(doc)
    cell = (
        "[[table:checkpoints]] and [[fig:method]], "
        "[[num:fig:method.n]] [[cite:radford2019language]][[fn:note]]"
    )
    assert view.plain(cell) == "Table 1 and Figure 1, 3 radford2019language"


def _two_column(rows: list[list[str]], columns: list[str]) -> str:
    output = registry.TableOutput(
        "Claim.",
        "Detail.",
        columns,
        rows,
        "l" + "r" * (len(columns) - 1),
        [],
        {},
    )
    doc = _doc(tables={"checkpoints": page.TableEntry("x", output)})
    return page.Page(doc).table("checkpoints")


def test_repeated_text_cells_of_a_row_become_one_spanning_cell() -> None:
    row = [
        "m",
        "no contrast",
        "no contrast",
        "no contrast",
        "0.5",
        "none",
        "none",
    ]
    html = _two_column([row], ["Model", "a", "b", "c", "d", "e", "f"])
    assert '<td class="r" colspan="3">no contrast</td>' in html
    assert html.count("no contrast") == 1
    assert html.count("none") == 2


def test_short_labels_stay_whole_and_long_identifiers_may_break() -> None:
    long_id = "siebert/sentiment-roberta-large-english"
    output = registry.TableOutput(
        "Claim.",
        "Detail.",
        ["Finder", "Source"],
        [["logistic regression", long_id]],
        "ll",
        [],
        {},
    )
    doc = _doc(tables={"checkpoints": page.TableEntry("x", output)})
    html = page.Page(doc).table("checkpoints")
    assert '<td class="k">logistic regression</td>' in html
    assert f'<td class="id">{long_id}</td>' in html
    assert '<td class="id">Qwen/Qwen3-0.6B</td>' in _hub_cell("Qwen/Qwen3-0.6B")
    wide = _two_column([["x", "1"]], ["Model", *"abcdefghi"])
    assert 'class="table wide"' in wide
    narrow = _two_column([["x", "1"]], ["Model", "a"])
    assert 'class="table wide"' not in narrow


def test_a_figure_that_keeps_its_size_on_paper_gets_a_class() -> None:
    def render(full_size_print: bool) -> str:
        output = registry.FigureOutput(
            Figure(),
            "Claim.",
            "Detail.",
            {},
            "a,b\n1,2\n",
            full_size_print=full_size_print,
        )
        entry = page.FigureEntry("x", output, SVG, b"%PDF-1.4")
        return page.Page(_doc({"method": entry})).figure("method")

    assert '<figure id="fig-1" class="full-size-print">' in render(True)
    assert '<figure id="fig-1">' in render(False)
    assert registry.FigureOutput(
        Figure(), "c", "d", {}, ""
    ).full_size_print is (False)


def test_a_table_that_may_wrap_its_labels_on_paper_gets_a_class() -> None:
    def render(wrap_labels: bool) -> str:
        output = registry.TableOutput(
            "Claim.",
            "Detail.",
            ["Model", *"abcdefghi"],
            [["x", *"123456789"]],
            "l" + "r" * 9,
            [],
            {},
            wrap_labels=wrap_labels,
        )
        doc = _doc(tables={"checkpoints": page.TableEntry("x", output)})
        return page.Page(doc).table("checkpoints")

    assert 'class="table wide wrap-labels"' in render(True)
    assert "wrap-labels" not in render(False)
    assert 'class="table wide"' in render(False)


def test_a_table_that_needs_compact_print_gets_a_class() -> None:
    def render(compact_print: bool) -> str:
        output = registry.TableOutput(
            "Claim.",
            "Detail.",
            ["Model", *"abcdefghi"],
            [["x", *"123456789"]],
            "l" + "r" * 9,
            [],
            {},
            compact_print=compact_print,
        )
        doc = _doc(tables={"checkpoints": page.TableEntry("x", output)})
        return page.Page(doc).table("checkpoints")

    assert 'class="table wide compact-print"' in render(True)
    assert "compact-print" not in render(False)
    assert registry.TableOutput("c", "d", [], [], "", [], {}).compact_print is (
        False
    )


def _hub_cell(text: str) -> str:
    output = registry.TableOutput(
        "Claim.", "Detail.", ["Model", "Hub ID"], [["m", text]], "ll", [], {}
    )
    doc = _doc(tables={"checkpoints": page.TableEntry("x", output)})
    return page.Page(doc).table("checkpoints")


def test_a_text_table_of_six_columns_is_wide_too() -> None:
    output = registry.TableOutput(
        "Claim.",
        "Detail.",
        list("abcdef"),
        [list("uvwxyz")],
        "llllll",
        [],
        {},
    )
    doc = _doc(tables={"checkpoints": page.TableEntry("x", output)})
    assert 'class="table wide"' in page.Page(doc).table("checkpoints")


def test_a_findings_link_is_pinned_to_the_results_commit() -> None:
    doc = dataclasses.replace(_doc(), commit="abc1234")
    html = page.Page(doc).text("[[findings:lie-gallery-c5|the lie gallery]]")
    assert (
        '<a href="https://github.com/javierdejesusda/tomography/blob/abc1234/'
        'docs/findings.md#lie-gallery-c5">the lie gallery</a>'
    ) in html
    assert "blob/main" not in html
    doc = dataclasses.replace(doc, commit="")
    assert "blob/main/docs/findings.md#x" in page.Page(doc).text(
        "[[findings:x|y]]"
    )
    assert page.Page(doc).plain("[[findings:x|y]] z") == "y z"


def test_the_page_opens_with_summary_overview_abstract_and_findings(
    full: tuple[page.Document, str],
) -> None:
    html = full[1]
    order = [
        html.index('<p class="tldr">'),
        html.index('<figure id="fig-1">'),
        html.index('id="abstract"'),
        html.index('<h2 id="key_findings">Key findings</h2>'),
        html.index('<h2 id="introduction">'),
    ]
    assert order == sorted(order)
    assert "TL;DR." in html
    assert page.numbers(full[0])[0]["teaser"] == "1"
    findings = html.split('id="key_findings"')[1].split("<h2")[0]
    assert findings.count("<li>") == 6


def test_label_token_writes_the_name_a_figure_computed() -> None:
    output = registry.FigureOutput(
        Figure(), "C.", "D.", {}, "a,b", labels={"top": "Qwen3-4B"}
    )
    entry = page.FigureEntry("x", output, SVG, b"%PDF-1.4")
    view = page.Page(_doc({"method": entry}))
    assert view.text("[[label:fig:method.top]]") == "Qwen3-4B"
    assert view.plain("[[label:fig:method.top]]") == "Qwen3-4B"
