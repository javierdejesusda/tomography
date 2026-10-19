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
        "Results",
        bib or {},
        appendix_tables or {},
    )


def test_main_items_are_numbered_in_order_of_appearance() -> None:
    doc = _doc(
        {"method": _figure(), "finders": _figure()},
        {"checkpoints": _table(), "finders": _table()},
    )
    figure_number, table_number = page.numbers(doc)
    assert figure_number == {"method": "1", "finders": "2"}
    assert table_number == {"checkpoints": "1", "finders": "2"}


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
    assert "Appendix&nbsp;B" in view.text("[[sec:appendix_reading]]")


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
                {"n": 1234.0, "zero": 0.0, "neg": -0.5, "p": 0.12345}
            )
        }
    )
    view = page.Page(doc)
    assert view.text("[[num:fig:method.n]]") == "1,234"
    assert view.text("[[num:fig:method.zero]]") == "0.000"
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


@pytest.mark.parametrize(
    "token",
    [
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
    assert '<tr class="group"><td>b</td>' in html


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
    assert [figure_number[k] for k in registry.MAIN_FIGURES] == [
        str(i) for i in range(1, 12)
    ]
    assert [table_number[k] for k in registry.MAIN_TABLES] == [
        str(i) for i in range(1, 9)
    ]
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
    assert heads[:1] == ["Abstract"]
    assert heads[1:8] == [
        "1 Introduction",
        "2 Related work",
        "3 Method",
        "4 Results",
        "5 Discussion",
        "6 Limitations",
        "7 Reproducibility",
    ]
    assert heads[8] == "References"
    assert heads[9] == "Cite this repository"
    assert heads[10:] == [
        "Appendix A Assets and licenses",
        "Appendix B Reading details",
        "Appendix C Wording and truth details",
        "Appendix D Knowledge, approximate answers and post-training details",
    ]
    subs = [
        t.replace("&ensp;", " ")
        for t in re.findall(r"<h3[^>]*>([^<]+)</h3>", html)
    ]
    assert subs[0] == "3.1 Reading a concept"
    assert any(s.startswith("4.9 ") for s in subs)


def test_the_contents_line_links_every_section(
    full: tuple[page.Document, str],
) -> None:
    html = full[1]
    line = re.search(r'<p class="contents">(.*?)</p>', html, flags=re.S)
    assert line is not None
    text = re.sub(r"<[^>]+>", "", line.group(1)).replace("&ensp;", " ")
    assert text.startswith("Contents: 1 Introduction, 2 Related work, 3 Method")
    assert text.endswith("7 Reproducibility, References, Appendix")
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


def test_the_masthead_names_the_commit(
    full: tuple[page.Document, str],
) -> None:
    html = full[1]
    assert (
        "Results at commit <code>abc1234</code>, built 1 January 2027" in html
    )


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
