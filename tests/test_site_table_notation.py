from pathlib import Path
import re

from tomography.p9 import unknowns
from tomography.site import registry
from tomography.site import stats
from tomography.site.manifest import Manifest

RESULTS = Path(__file__).resolve().parents[1] / "results"
DASHES = (chr(0x2014), chr(0x2013))
TERMS = [
    "pair",
    "h(x, \u2113)",
    "v(\u2113)",
    "u(\u2113)",
    "the paper",
    "finder",
    "reader",
    "accuracy",
    "margin",
    "relative depth",
    "layer chosen on validation",
    "AUROC",
    "relative coefficient c",
    "win rate",
    "fluency rule",
    "contrast",
    "role",
]
GONE = ("all-pairs reader", "series", "run codes")


def _build() -> registry.TableOutput:
    return registry.build_table("notation", Manifest(RESULTS))


def test_the_table_has_two_columns_and_the_agreed_terms_in_order() -> None:
    out = _build()
    assert out.columns == ["Symbol or term", "Meaning"]
    assert out.align == "ll"
    assert [r[0] for r in out.rows] == TERMS
    assert not set(GONE) & {r[0] for r in out.rows}
    assert out.row_groups == ()


def test_the_caption_is_one_sentence() -> None:
    out = _build()
    assert out.claim == "Symbols and terms used in the text."
    assert out.detail == ""


def test_the_fluency_rule_states_its_limits_and_its_two_measures() -> None:
    rows = {r[0]: r[1] for r in _build().rows}
    rule = rows["fluency rule"]
    assert rule.startswith(
        "Steered text is fluent when its neutral-text perplexity ratio is "
        f"at most {stats.FLUENT_MAX_RATIO:g} and its distinct-1 at least "
        f"{stats.FLUENT_MIN_DISTINCT:g}."
    )
    assert (
        "divides the perplexity of held-out WikiText-103 passages under the "
        "steered model by that under the unsteered model"
    ) in rule
    assert "the share of distinct tokens in the completions" in rule


def test_accuracy_is_a_pairwise_comparison_with_ties_at_one_half() -> None:
    rows = {r[0]: r[1] for r in _build().rows}
    assert "projects higher" in rows["accuracy"]
    assert "one half" in rows["accuracy"]
    assert "threshold" not in " ".join(rows.values())


def test_the_win_rate_is_the_positive_against_the_negative_completion() -> None:
    rows = {r[0]: r[1] for r in _build().rows}
    assert "+c" in rows["win rate"]
    assert "judged more positive" in rows["win rate"]


def test_the_glossary_names_every_knowledge_contrast() -> None:
    rows = {r[0]: r[1] for r in _build().rows}
    names = rows["contrast"]
    for contrast in (
        *unknowns.ENTITY_TYPES,
        *unknowns.UNKNOWN_TYPES,
        "unknowable_detail",
        "false_premise",
        "kuq",
    ):
        assert contrast.replace("_", " ") in names, contrast
    for key in ("lu2026relational", "amayuelas2024knowledge"):
        assert f"[[cite:{key}]]" in names
    assert "one draw of training pairs" in rows["reader"]
    paper = "[[cite:zou2025representationengineeringtopdownapproach]]"
    assert paper in rows["the paper"]
    assert "finder" in rows["reader"]


def test_no_row_spells_out_a_run_code() -> None:
    out = _build()
    code = re.compile(r"\b[RPMCA][0-9]+[a-d]?\b")
    for term, meaning in out.rows:
        assert code.search(term + " " + meaning) is None, term


def test_no_underscore_or_dash_in_the_table() -> None:
    out = _build()
    texts = [out.claim, out.detail, *(c for r in out.rows for c in r)]
    assert not any("_" in t for t in texts)
    assert not any(ch in t for t in texts for ch in DASHES)


def test_the_margin_has_no_single_layer_claim() -> None:
    text = " ".join(
        cell
        for row in registry.build_table("notation", Manifest(RESULTS)).rows
        for cell in row
    )
    assert "on the same test pairs at the same layer." not in text
    assert "each at the layer chosen on its own select split" in text
