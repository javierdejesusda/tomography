import importlib
from pathlib import Path
import re
import runpy

import pytest

from tomography.site import content
from tomography.site import registry
from tomography.site import runfiles
from tomography.site import stats
from tomography.site.figs import teaser as teaser_fig
from tomography.site.manifest import Manifest
from tomography.viz import paper

_HERE = Path(__file__).parent
_R = runpy.run_path(str(_HERE / "_results.py"))
ROOT = _R["ROOT"]
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
    for top in (*content.BODY, content.CODE, *content.APPENDICES)
    for s in _sections(top)
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


def _section(key: str) -> str:
    (section,) = [s for s in ALL if s.key == key]
    return " ".join(_texts(section))


def test_the_words_criterion_of_the_range_reader_is_by_interval() -> None:
    approximate = registry.build_table("approximate", Manifest(ROOT))
    assert (
        "above the question words on "
        "[[tally:table:approximate.above_words]], by interval"
    ) in _section("known_unknown")
    for role in approximate.counts["above_words"].values():
        assert role.hits == 0


def test_the_post_training_sentiment_count_names_its_interval() -> None:
    sentences = [
        t
        for t in re.split(r"(?<=[.;]) ", _everything())
        if "[[num:fig:base_post.sent_below]]" in t
    ]
    assert len(sentences) == 2
    for sentence in sentences:
        assert "interval" in sentence, sentence


def test_the_wording_test_counts_per_role_and_has_no_pooled_number() -> None:
    text = _everything()
    assert "[[tally:fig:wording.span_above_noise]]" in _section("reading")
    assert "[[fig:wording]]" in _section("appendix_reading")
    for pooled in ("bare_to_paper", "bare_random", "bare_cos"):
        assert f"wording.{pooled}_min" not in text
        assert f"wording.{pooled}_max" not in text


def test_the_fluency_rule_is_defined_where_it_is_first_used() -> None:
    assert content.FLUENCY_RULE == FLUENCY
    method = _section("steering_method")
    assert f"meets the fluency rule: {FLUENCY}" in method
    assert "fluency rule" not in content.ABSTRACT
    assert _everything().count(FLUENCY) == 1


def test_the_knowledge_caption_scopes_its_strongest_control() -> None:
    output = registry.build_figure("knowledge_reading", Manifest(ROOT))
    assert "popularity, the strongest baseline" not in output.detail
    assert "Direction minus the strongest control" in output.detail
    assert (
        "it is above the strongest of popularity and the two text classifiers"
    ) in _section("known_unknown")


def test_the_fluent_utility_steering_checkpoint_is_the_computed_one() -> None:
    manifest = Manifest(ROOT)
    steering = registry.build_figure("steering", manifest)
    models = manifest.models("m1")
    fluent = [m for m in models if teaser_fig._steering_fluent(manifest, m)]
    beats = [
        m
        for m in models
        if (cell := teaser_fig._steering(manifest, m)) is not None
        and cell.margin > 0
    ]
    assert len(beats) == 5
    assert [m for m in fluent if m in beats] == fluent
    assert [paper.display_name(m) for m in fluent] == ["Qwen3-1.7B-Base"]
    assert (
        "Utility steering that beats every random direction meets the "
        "rule on only Qwen3-1.7B-Base"
    ) in _section("introduction")
    broken = [
        m for m in beats if teaser_fig._steering_fluent(manifest, m) is False
    ]
    unmeasured = [m for m in beats if m not in fluent and m not in broken]
    assert [teaser_fig._steering_fluent(manifest, m) for m in unmeasured] == [
        None
    ]
    assert [paper.display_name(m) for m in unmeasured] == ["GPT-2 small"]
    for model in broken:
        path = manifest.run("m1", model)
        assert path is not None
        rows = runfiles.rows(path, "table_effects.json")
        (none,) = [r for r in rows if r["method"] == "none"]
        steered = [
            float(r["perplexity"])
            for r in rows
            if r["method"] == "logistic"
            and r["coeff"] == teaser_fig.STEER_STRENGTH
        ]
        ratio = sum(steered) / len(steered) / float(none["perplexity"])
        assert ratio > stats.FLUENT_MAX_RATIO, model
    names = stats.join_and([paper.display_name(m) for m in broken])
    assert (
        "only Qwen3-1.7B-Base also keeps the fluency rule, with perplexity "
        "ratio [[num:fig:teaser.steering_kept_ratio]] and distinct-1 "
        "[[num:fig:teaser.steering_kept_distinct]] (mean of its three "
        f"vectors, positive sign); on {names} the ratio is "
        "[[num:fig:teaser.steering_broken_ratio_min]] to "
        "[[num:fig:teaser.steering_broken_ratio_max]], above the limit "
        "(GPT-2 small has no fluency measure)"
    ) in _section("steering")
    assert "run's files" not in _section("steering")
    for role in ("gpt2", "base", "post"):
        assert steering.counts["fluent03"][role].hits == 0, role


def test_fluency_thresholds_in_the_text_come_from_the_stats_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stats, "FLUENT_MAX_RATIO", 2.5)
    monkeypatch.setattr(stats, "FLUENT_MIN_DISTINCT", 0.35)
    try:
        reloaded = importlib.reload(content)
        text = repr((reloaded.FLUENCY_RULE, reloaded.BODY, reloaded.APPENDICES))
        assert (
            "at most 2.5 and distinct-1 at least 0.35" in reloaded.FLUENCY_RULE
        )
        assert text.count("2.5") == 2
        assert text.count("0.35") == 2
        assert "1.5" not in text
        assert "0.15" not in text
    finally:
        monkeypatch.undo()
        importlib.reload(content)
