from matplotlib.figure import Figure
import pytest

from tomography.site import page
from tomography.site import references
from tomography.site import registry
from tomography.site import stats

MINUS = chr(0x2212)
SVG = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>'


def _figure(
    numbers: dict[str, float] | None = None,
    counts: dict[str, dict[str, stats.RoleCount]] | None = None,
) -> page.FigureEntry:
    fig = Figure()
    output = registry.FigureOutput(
        fig, "Claim.", "Detail.", numbers or {}, "a,b\n1,2\n", counts or {}
    )
    return page.FigureEntry("x", output, SVG, b"%PDF-1.4")


def _table(numbers: dict[str, float] | None = None) -> page.TableEntry:
    output = registry.TableOutput(
        "Claim.", "Detail.", ["A", "B"], [["x", "1"]], "lr", [], numbers or {}
    )
    return page.TableEntry("x", output)


def _doc(
    figures: dict[str, page.FigureEntry] | None = None,
    tables: dict[str, page.TableEntry] | None = None,
    appendix: dict[str, page.FigureEntry] | None = None,
    appendix_tables: dict[str, page.TableEntry] | None = None,
) -> page.Document:
    return page.Document(
        figures or {},
        tables or {},
        appendix or {},
        "Results",
        {},
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
    view = page.Page(doc)
    text = view.text("See [[fig:method]] and [[table:checkpoints]].")
    assert 'href="#figure-1">Figure&nbsp;1</a>' in text
    assert 'href="#table-1">Table&nbsp;1</a>' in text


def test_figure_and_table_markup_uses_the_printed_number() -> None:
    doc = _doc({"method": _figure()}, {"checkpoints": _table()})
    view = page.Page(doc)
    figure = view.figure("method")
    assert '<figure id="figure-1">' in figure
    assert 'href="data/figure-1.csv"' in figure
    assert 'href="figures/figure-1.pdf"' in figure
    table = view.table("checkpoints")
    assert 'id="table-1"' in table
    assert 'href="data/table-1.csv"' in table


def test_appendix_markup_uses_the_a_number() -> None:
    doc = _doc(
        appendix={"a_layers": _figure()},
        appendix_tables={"assets": _table()},
    )
    view = page.Page(doc)
    assert '<figure id="figure-A1">' in view.figure("a_layers")
    assert 'id="table-A1"' in view.table("assets")


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
    doc = page.Document({}, {}, {}, "Results", bib)
    view = page.Page(doc)
    assert '<a class="ref" href="#ref1">[1]</a>' in view.text(
        "[[cite:zou2025]]"
    )
    assert 'id="fnref1"' in view.text("x[[fn:a note]]")
