import csv
import io
from pathlib import Path
import re
import runpy

import pytest

from tomography.site import content
from tomography.site import registry
from tomography.site import stats
from tomography.site.manifest import Manifest

_HERE = Path(__file__).parent
_R = runpy.run_path(str(_HERE / "_results.py"))
ROOT = _R["ROOT"]
WORDS = (
    "above a classifier on the question words, with the interval of the "
    "difference above zero, on "
)
BELOW = "with an interval below zero"
FLUENCY = (
    "neutral-text perplexity ratio at most 1.5 and distinct-1 at least 0.15"
)


def _sections(section: content.Section) -> list[content.Section]:
    found = [section]
    for sub in section.subsections:
        found += _sections(sub)
    return found


ALL = [
    s for top in (*content.BODY, *content.APPENDICES) for s in _sections(top)
]


def _texts(section: content.Section) -> list[str]:
    parts = [*section.paragraphs, *section.after]
    for head, text in (*section.runin, *section.listed):
        parts += [head, text]
    return parts


def _everything() -> str:
    texts = [content.ABSTRACT]
    for section in ALL:
        texts += _texts(section)
    return " ".join(texts)


def _opener(key: str) -> str:
    subs = {s.key: s for s in content.RESULTS.subsections}
    return subs[key].paragraphs[0]


def _intro() -> str:
    return " ".join(
        h + " " + t for h, t in content.INTRODUCTION.listed
    ) + " ".join(content.INTRODUCTION.paragraphs)


def test_the_words_criterion_is_the_difference_interval_in_three_places() -> (
    None
):
    approximate = registry.build_table("approximate", Manifest(ROOT))
    assert WORDS in _intro()
    assert WORDS in _opener("approximate")
    assert WORDS in approximate.claim
    assert "above the 95 percent interval of a classifier" not in (
        _everything() + approximate.claim
    )
    for role in approximate.counts["above_words"].values():
        assert role.hits == 0


def test_the_knowledge_opener_names_the_interval_criterion_twice() -> None:
    opener = _opener("knowledge_reading")
    assert (
        "above their popularity, with an interval excluding zero, on "
        "[[count:fig:knowledge_reading.above_pop]]"
    ) in opener
    assert (
        "adds to the confidence of the model, with an interval excluding "
        "zero, on [[count:fig:teaser.knowledge_confidence]]"
    ) in opener


def test_the_abstract_names_the_criterion_beside_every_count() -> None:
    abstract = content.ABSTRACT
    assert (
        "above a random direction by the 95 percent interval for sentiment"
        in abstract
    )
    assert (
        "above it by the 95 percent interval on [[count:fig:teaser.truthfulqa]]"
    ) in abstract


def test_the_post_training_sentiment_count_names_its_interval() -> None:
    sentences = [
        t
        for t in re.split(r"(?<=[.;]) ", _everything())
        if "[[num:fig:base_post.sent_below]]" in t
    ]
    assert len(sentences) >= 3
    for sentence in sentences:
        assert BELOW in sentence, sentence


def test_ten_checkpoints_are_written_as_all_ten() -> None:
    text = _everything()
    assert "each of the 10" not in text
    assert "on all 10 checkpoints" in text
    assert "fitted on all 10 checkpoints" in text


def test_the_wording_paragraph_gives_the_random_baseline() -> None:
    text = " ".join(h + " " + t for h, t in content.INTRODUCTION.listed)
    assert (
        "[[num:fig:wording.bare_random_min]] to "
        "[[num:fig:wording.bare_random_max]]"
    ) in text


def test_the_test_size_limit_is_computed_per_role_not_written_down() -> None:
    limitations = " ".join(_texts(content.LIMITATIONS))
    assert "about 0.15 on truth" not in limitations
    for token in (
        "truth_half_gpt2_min",
        "truth_half_gpt2_max",
        "truth_half_qwen3_min",
        "truth_half_qwen3_max",
    ):
        assert f"[[num:fig:teaser.{token}]]" in limitations


def test_the_teaser_exports_the_truth_half_widths_per_role() -> None:
    output = registry.build_figure("teaser", Manifest(ROOT))
    rows = [
        r
        for r in csv.DictReader(io.StringIO(output.csv))
        if r["row"] == "truth"
    ]
    half = {
        "gpt2": [],
        "qwen3": [],
    }
    for r in rows:
        key = "gpt2" if r["role"] == "gpt2" else "qwen3"
        half[key].append((float(r["hi"]) - float(r["lo"])) / 2)
    for key, values in half.items():
        assert output.numbers[f"truth_half_{key}_min"] == pytest.approx(
            min(values)
        )
        assert output.numbers[f"truth_half_{key}_max"] == pytest.approx(
            max(values)
        )
    assert stats.ROLES == ("gpt2", "base", "post")


def test_the_fluency_rule_is_defined_where_the_abstract_first_uses_it() -> None:
    assert content.FLUENCY_RULE == FLUENCY
    assert f"keeps the fluency rule ({FLUENCY}) on" in content.ABSTRACT
    first = content.ABSTRACT.index("fluency rule")
    assert content.ABSTRACT[first:].startswith(f"fluency rule ({FLUENCY})")
    opener = _opener("steering")
    assert f"keeps the fluency rule ({FLUENCY}) on" in opener
    assert content.ABSTRACT.count(FLUENCY) == 1


def test_the_figure_8_caption_scopes_its_strongest_baseline() -> None:
    output = registry.build_figure("knowledge_reading", Manifest(ROOT))
    assert "popularity, the strongest baseline" not in output.detail
    assert (
        "popularity, the strongest of popularity and the two text classifiers"
    ) in output.detail
    pops = [
        float(r["value"])
        for r in csv.DictReader(io.StringIO(output.csv))
        if r["panel"] == "a" and r["series"] == "popularity"
    ]
    assert (
        f"{min(pops):.3f} to {max(pops):.3f} for popularity"
    ) in output.detail
