import csv
import io
from pathlib import Path
import re
import runpy

from tomography.site import content
from tomography.site import registry
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
    s
    for top in (*content.BODY, *content.APPENDICES, content.KEY_FINDINGS)
    for s in _sections(top)
]


def _texts(section: content.Section) -> list[str]:
    parts = [*section.paragraphs, *section.after]
    for head, text in (*section.runin, *section.listed):
        parts += [head, text]
    return parts


def _everything() -> str:
    texts = [content.ABSTRACT, content.TLDR]
    for section in ALL:
        texts += _texts(section)
    return " ".join(texts)


def _findings(head: str) -> str:
    for h, t in content.KEY_FINDINGS.listed:
        if h.startswith(head):
            return t
    raise AssertionError(head)


def test_the_words_criterion_is_the_difference_interval() -> None:
    approximate = registry.build_table("approximate", Manifest(ROOT))
    assert WORDS in _everything()
    assert WORDS in approximate.claim
    assert "above the 95 percent interval of a classifier" not in (
        _everything() + approximate.claim
    )
    for role in approximate.counts["above_words"].values():
        assert role.hits == 0


def test_the_baseline_findings_name_the_interval_criterion() -> None:
    text = _findings("Baselines")
    assert (
        "above a classifier on the words alone, with an interval excluding "
        "zero, on [[count:fig:teaser.truthfulqa]]"
    ) in text
    assert (
        "adds to the confidence of the model, with an interval excluding "
        "zero, on [[count:fig:teaser.knowledge_confidence]]"
    ) in text


def test_the_post_training_sentiment_count_names_its_interval() -> None:
    sentences = [
        t
        for t in re.split(r"(?<=[.;]) ", _everything())
        if "[[num:fig:base_post.sent_below]]" in t
    ]
    assert len(sentences) >= 4
    for sentence in sentences:
        assert BELOW in sentence, sentence


def test_ten_checkpoints_are_written_as_all_ten() -> None:
    text = _everything()
    assert "each of the 10" not in text
    assert "fitted on all 10 checkpoints" in text
    assert "direction on all 10 checkpoints" in text


def test_the_wording_numbers_give_the_random_baseline() -> None:
    (section,) = [s for s in content.APPENDICES if s.key == "appendix_wording"]
    text = " ".join(_texts(section))
    assert (
        "[[num:fig:wording.bare_random_min]] to "
        "[[num:fig:wording.bare_random_max]]"
    ) in text


def test_the_test_size_limit_is_computed_per_role_not_written_down() -> None:
    limits = " ".join(_texts(content.DISCUSSION))
    assert "about 0.15 on truth" not in limits
    assert "post-trained checkpoints are not resolved" in limits
    for role in ("gpt2", "base", "post"):
        for edge in ("min", "max"):
            assert f"[[num:fig:teaser.truth_half_{role}_{edge}]]" in limits
    assert "truth_half_qwen3" not in limits


def test_the_fluency_rule_is_defined_where_it_is_first_used() -> None:
    assert content.FLUENCY_RULE == FLUENCY
    assert f"keeps the fluency rule ({FLUENCY}) on" in _findings("Steering")
    method = " ".join(t for s in _sections(content.METHOD) for t in _texts(s))
    assert f"the fluency rule: the {FLUENCY}" in method
    assert "fluency rule" not in content.ABSTRACT


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
