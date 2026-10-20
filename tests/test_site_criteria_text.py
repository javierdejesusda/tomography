import importlib
from pathlib import Path
import re
import runpy

import pytest

from tomography.site import content
from tomography.site import registry
from tomography.site import stats
from tomography.site.figs import teaser as teaser_fig
from tomography.site.manifest import Manifest
from tomography.viz import paper

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
    assert (
        "above a classifier on the words alone, with an interval excluding "
        "zero, on [[count:fig:teaser.truthfulqa]]"
    ) in _findings("Truth")
    assert (
        "adds to the confidence of the model, with an interval excluding "
        "zero, on [[count:fig:teaser.knowledge_confidence]]"
    ) in _findings("Known and unknown")


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


def test_the_wording_appendix_counts_per_role_and_points_to_the_ranges() -> (
    None
):
    (section,) = [s for s in content.APPENDICES if s.key == "appendix_wording"]
    text = " ".join(_texts(section))
    assert "[[count:fig:wording.span_above_noise]]" in text
    assert "[[fig:wording]]" in text
    for pooled in ("bare_to_paper", "bare_random", "bare_cos"):
        assert f"wording.{pooled}_min" not in text
        assert f"wording.{pooled}_max" not in text
    assert "all 10 checkpoints" not in text


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
        "direction minus the strongest of popularity and the two text "
        "classifiers"
    ) in output.detail


def _appendix(key: str) -> str:
    (section,) = [s for s in content.APPENDICES if s.key == key]
    return " ".join(_texts(section))


def test_the_overview_method_detail_is_an_appendix_c_paragraph() -> None:
    text = _appendix("appendix_reading")
    assert (
        "Baselines of each row of [[fig:teaser]]: rows 1 to 5, the "
        "random direction" in text
    )
    assert "row 7, a word classifier" in text
    assert "row 8, the strongest of popularity and two text classifiers" in text
    assert "five stacked output signals" in text
    assert "[[count:fig:teaser.knowledge_confidence]]" in text
    assert "[[count:fig:teaser.steering]]" in text
    assert "[[count:fig:teaser.steering_fluent]]" in text
    assert "which has chance 0.05" in text
    assert (
        f"the fluency rule ({FLUENCY.replace('neutral-text ', '')}" in text
        or ("the fluency rule" in text)
    )
    assert "Hanley and McNeil" in text
    assert "GPT-2 small has no fluency measure" in text
    assert "[[fig:teaser]]" in text
    assert "answer probability alone" not in text


def test_the_knowledge_detail_moved_to_appendix_g() -> None:
    text = _appendix("appendix_knowledge")
    assert "stacks five output signals" in text
    assert "[[count:fig:knowledge_reading.above_best_point]]" in text
    assert "[[count:fig:knowledge_reading.probe_above]]" in text
    assert "[[num:fig:knowledge_reading.gpt2_small_probe_gain]]" in text
    assert "no known-against-unknown contrast" in text
    assert "fitted on the correct and wrong labels" in text
    assert "Hanley and McNeil" in text


def test_the_summary_holds_per_role_on_the_data() -> None:
    manifest = Manifest(ROOT)
    teaser = registry.build_figure("teaser", manifest)
    finders = registry.build_figure("finders", manifest)
    steering = registry.build_figure("steering", manifest)
    roles = ("gpt2", "base", "post")
    tldr = content.TLDR
    assert tldr.count(". ") == 0
    assert (
        "reads utility and morality above a random direction on every "
        "checkpoint and truth on every Qwen3 checkpoint"
    ) in tldr
    for name in ("utility", "morality"):
        assert all(c.hits == c.total for c in teaser.counts[name].values()), (
            name
        )
    truth = teaser.counts["truth"]
    assert truth["gpt2"].hits == 0
    assert truth["base"].hits == truth["base"].total
    assert truth["post"].hits == truth["post"].total
    assert "the finder of the paper is weaker" in tldr
    pairs = (
        ("pca_sentiment", "logistic_sentiment"),
        ("pca_utility", None),
        ("pca_truth", None),
    )
    logistic = {"pca_utility": "utility", "pca_truth": "truth"}
    for pca, other in pairs:
        for role in roles:
            if other:
                ref = finders.counts[other][role].hits
            else:
                ref = teaser.counts[logistic[pca]][role].hits
            assert finders.counts[pca][role].hits <= ref, (pca, role)
    assert (
        "a truth reader does not beat the words alone, a known-against-"
        "unknown direction does not beat the model's confidence"
    ) in tldr
    assert len(tldr.split()) <= 60
    for name in ("truthfulqa", "knowledge_confidence"):
        assert all(c.hits == 0 for c in teaser.counts[name].values()), name
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
        "utility steering that beats every random direction meets the "
        "fluency rule on only Qwen3-1.7B-Base"
    ) in tldr
    for role in roles:
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
        assert text.count("2.5") >= 4
        assert text.count("0.35") >= 4
        assert "1.5" not in text
        assert "0.15" not in text
    finally:
        monkeypatch.undo()
        importlib.reload(content)
