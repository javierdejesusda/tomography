from pathlib import Path
import re

from tomography.site import registry
from tomography.site.manifest import Manifest

RESULTS = Path(__file__).resolve().parents[1] / "results"
DASHES = (chr(0x2014), chr(0x2013))
TERMS = [
    "pair",
    "h(x, \u2113)",
    "v(\u2113)",
    "u(\u2113)",
    "finder",
    "accuracy",
    "margin",
    "relative depth",
    "layer chosen on validation",
    "AUROC",
    "relative coefficient c",
    "win rate",
    "role",
    "run codes",
]


def _build() -> registry.TableOutput:
    return registry.build_table("notation", Manifest(RESULTS))


def test_the_table_has_two_columns_and_the_agreed_terms_in_order() -> None:
    out = _build()
    assert out.columns == ["Symbol or term", "Meaning"]
    assert out.align == "ll"
    assert [r[0] for r in out.rows] == TERMS
    assert out.row_groups == ()


def test_accuracy_is_a_pairwise_comparison_with_ties_at_one_half() -> None:
    rows = {r[0]: r[1] for r in _build().rows}
    assert "projects higher" in rows["accuracy"]
    assert "one half" in rows["accuracy"]
    assert "threshold" not in " ".join(rows.values())


def test_the_win_rate_is_the_positive_against_the_negative_completion() -> None:
    rows = {r[0]: r[1] for r in _build().rows}
    assert "+c" in rows["win rate"]
    assert "judged more positive" in rows["win rate"]


def test_the_run_codes_row_is_the_only_place_with_a_run_code() -> None:
    out = _build()
    code = re.compile(r"\b[RPMCA][0-9]+[a-d]?\b")
    for term, meaning in out.rows:
        found = code.search(term + " " + meaning)
        assert bool(found) == (term == "run codes"), term


def test_no_underscore_or_dash_in_the_table() -> None:
    out = _build()
    texts = [out.claim, out.detail, *(c for r in out.rows for c in r)]
    assert not any("_" in t for t in texts)
    assert not any(ch in t for t in texts for ch in DASHES)
