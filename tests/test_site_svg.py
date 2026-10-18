from xml.etree import ElementTree

import pytest

from tomography.site import svg

STYLES = {
    "reader": svg.Style(css="s1", shape="circle", label="Reader"),
    "text": svg.Style(css="s2", shape="diamond", label="Text only"),
}


def _panel(**kwargs: object) -> svg.DotPanel:
    rows = (
        svg.Row(
            label="gpt2",
            group="gpt2",
            band=(0.48, 0.55),
            marks=(
                svg.Mark("reader", 0.77, lo=0.75, hi=0.79),
                svg.Mark("text", 0.6),
            ),
        ),
        svg.Row(
            label="qwen3-0.6b",
            group="qwen3",
            marks=(svg.Mark("reader", 0.86, hollow=True),),
        ),
    )
    defaults: dict[str, object] = {
        "uid": "t1",
        "title": "Accuracy by model",
        "rows": rows,
        "styles": STYLES,
        "domain": (0.4, 1.0),
        "ticks": (0.4, 0.6, 0.8, 1.0),
        "refs": ((0.5, "chance"),),
        "band_label": "random direction",
    }
    defaults.update(kwargs)
    return svg.DotPanel(**defaults)  # type: ignore[arg-type]


def _parse(markup: str) -> ElementTree.Element:
    return ElementTree.fromstring(markup)


def _all(root: ElementTree.Element, cls: str) -> list[ElementTree.Element]:
    return [e for e in root.iter() if cls in (e.get("class") or "").split()]


def test_dot_panel_is_well_formed_svg_with_text_alternative() -> None:
    root = _parse(svg.render_dot_panel(_panel()))
    assert root.tag.endswith("svg")
    assert root.get("role") == "img"
    texts = [e.text for e in root.iter() if e.tag.endswith("title")]
    assert "Accuracy by model" in texts
    assert any(e.tag.endswith("desc") for e in root.iter())


def test_dot_panel_draws_one_mark_per_value_with_series_class() -> None:
    root = _parse(svg.render_dot_panel(_panel()))
    assert len(_all(root, "s1")) >= 2
    marks = _all(root, "mark")
    assert len(marks) == 3
    assert {m.get("data-series") for m in marks} == {"reader", "text"}
    assert any("0.770" in (m.get("data-tip") or "") for m in marks)


def test_dot_panel_labels_rows_axis_ticks_and_reference() -> None:
    markup = svg.render_dot_panel(_panel())
    for text in ("gpt2", "qwen3-0.6b", "0.4", "1.0", "chance"):
        assert f">{text}<" in markup


def test_dot_panel_band_and_interval_use_value_scale() -> None:
    root = _parse(svg.render_dot_panel(_panel()))
    (band,) = _all(root, "band")
    (whisker,) = _all(root, "whisker")
    assert float(band.get("width", "0")) > 0
    assert float(whisker.get("x2", "0")) > float(whisker.get("x1", "0"))


def test_hollow_mark_is_marked_for_styling() -> None:
    root = _parse(svg.render_dot_panel(_panel()))
    assert len(_all(root, "hollow")) == 1


def test_values_outside_domain_are_clamped_not_dropped() -> None:
    row = svg.Row("m", marks=(svg.Mark("reader", 2.0),))
    root = _parse(svg.render_dot_panel(_panel(rows=(row,))))
    (mark,) = _all(root, "mark")
    assert mark.get("data-series") == "reader"


def test_unknown_series_is_rejected() -> None:
    row = svg.Row("m", marks=(svg.Mark("nope", 0.5),))
    with pytest.raises(KeyError):
        svg.render_dot_panel(_panel(rows=(row,)))


def test_panel_table_lists_every_row_and_series_value() -> None:
    markup = svg.panel_table(_panel())
    root = _parse(markup)
    cells = ["".join(c.itertext()) for c in root.iter() if c.tag == "td"]
    assert "gpt2" in cells
    assert any("0.770" in c and "0.750" in c for c in cells)
    assert "n/a" in cells


def test_legend_lists_each_series_label() -> None:
    markup = svg.legend(STYLES)
    assert "Reader" in markup
    assert "Text only" in markup


def _line_panel(**kwargs: object) -> svg.LinePanel:
    series = (
        svg.LineSeries(
            "a", "Base", "s1", ((-0.1, 0.2), (0.0, 0.5), (0.1, 0.9))
        ),
        svg.LineSeries(
            "b", "Post", "s2", ((-0.1, 0.3), (0.1, 0.7)), dashed=True
        ),
    )
    defaults: dict[str, object] = {
        "uid": "l1",
        "title": "Score by coefficient",
        "series": series,
        "x_domain": (-0.1, 0.1),
        "x_ticks": (-0.1, 0.0, 0.1),
        "y_domain": (0.0, 1.0),
        "y_ticks": (0.0, 0.5, 1.0),
        "x_label": "coefficient",
        "refs": ((0.5, "midpoint"),),
    }
    defaults.update(kwargs)
    return svg.LinePanel(**defaults)  # type: ignore[arg-type]


def test_line_panel_has_one_polyline_per_series() -> None:
    root = _parse(svg.render_line_panel(_line_panel()))
    lines = _all(root, "line")
    assert [e.get("data-series") for e in lines] == ["a", "b"]
    assert len(_all(root, "dashed")) == 1
    assert len(_all(root, "mark")) == 5


def test_line_panel_log_axis_places_equal_ratios_equally() -> None:
    series = (
        svg.LineSeries("a", "A", "s1", ((0.0, 1.0), (1.0, 10.0), (2.0, 100.0))),
    )
    panel = _line_panel(
        series=series,
        y_domain=(1.0, 100.0),
        y_ticks=(1.0, 10.0, 100.0),
        y_log=True,
        x_domain=(0.0, 2.0),
        x_ticks=(0.0, 2.0),
    )
    root = _parse(svg.render_line_panel(panel))
    ys = [float(m.get("cy", "0")) for m in _all(root, "mark")]
    assert ys[0] - ys[1] == pytest.approx(ys[1] - ys[2])


def test_line_panel_table_has_a_column_per_series() -> None:
    root = _parse(svg.panel_table(_line_panel()))
    heads = ["".join(h.itertext()) for h in root.iter() if h.tag == "th"]
    assert "Base" in heads
    assert "Post" in heads
