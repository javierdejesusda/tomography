import dataclasses
import json
from pathlib import Path
import re
import runpy

import pytest

from tomography.site import content
from tomography.site import params
from tomography.site import references
from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.site.tables import notation

_HERE = Path(__file__).parent
_R = runpy.run_path(str(_HERE / "_results.py"))
ROOT, data, models, run = _R["ROOT"], _R["data"], _R["models"], _R["run"]
BIB = ROOT.parent / "docs" / "references.bib"
DASHES = (chr(0x2014), chr(0x2013))
TOKEN = re.compile(r"\[\[(.+?)\]\]")
RUN_CODE = re.compile(r"\b[RPMCA][0-9]+[a-d]?\b")
GENERIC = (
    r"of 3(?= Qwen3 sizes)",
    r"(?<![\d.])20\d\d(?!\d)",
    r"(?<![\d.])\d\.\dB\b",
    r"(?<![\d.])\dB\b",
    r"GPT-2",
    r"Qwen3",
    r"LLaMA-2",
    r"SST-2",
    r"WikiText-103",
    r"distinct-1",
    r"(?<![\d.])±0\.[13](?!\d)",
    r"coefficient 0\.[13](?!\d)",
    r"above 0\.95(?!\d)",
    r"exceeds 0\.1(?!\d)",
    r"(?:within|above) 1\.5 times",
    r"below 0\.15(?!\d)",
    r"at most 1\.5(?!\d)",
    r"at least 0\.15(?!\d)",
    r"below 0\.5(?!\d)",
    r"\ball 10\b",
    r"(?<![\d.])95 percent(?!\d)",
    r"(?<![\d.])40, 55 and 70 percent",
    r"(?<![\d.])20 for the held-out",
    r"none of the 10\b",
)


def _sections(
    section: content.Section,
) -> list[content.Section]:
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


def _all_texts() -> list[str]:
    texts = [content.ABSTRACT, content.TITLE]
    for section in ALL:
        texts += [section.title, *_texts(section)]
    return texts


def _bare(text: str) -> str:
    return TOKEN.sub("", text)


def _digits_left(text: str, allowed: tuple[str, ...] = GENERIC) -> list[str]:
    text = _bare(text)
    for pattern in allowed:
        text = re.sub(pattern, " ", text)
    return re.findall(r"\d+", text)


def _opening_texts() -> list[str]:
    texts = [content.ABSTRACT]
    for section in (content.INTRODUCTION, content.DISCUSSION):
        texts += _texts(section)
    for sub in content.RESULTS.subsections:
        texts += _texts(sub)
    return texts


def test_the_title_and_abstract_are_the_planned_ones() -> None:
    assert content.TITLE == (
        "Reading and Steering Concepts with One Direction: Representation "
        "Engineering on GPT-2 and Qwen3"
    )
    assert content.ABSTRACT.startswith(
        "We test representation engineering (Zou et al., 2023) on four GPT-2"
    )
    assert content.ABSTRACT.endswith(content.ABSTRACT_STEERING)


def test_the_body_sections_are_the_seven_of_the_paper_in_order() -> None:
    assert [s.title for s in content.BODY] == [
        "Introduction",
        "Related work",
        "Method",
        "Results",
        "Discussion",
        "Limitations",
        "Reproducibility",
    ]
    assert [s.key for s in content.RESULTS.subsections] == [
        "finders",
        "reading",
        "wording",
        "truth",
        "steering",
        "knowledge_reading",
        "knowledge_steering",
        "approximate",
        "base_post",
    ]
    assert [s.title for s in content.APPENDICES] == [
        "Assets and licenses",
        "Reading details",
        "Wording and truth details",
        "Knowledge, approximate answers and post-training details",
        "Other experiments",
    ]


def test_the_introduction_lists_nine_contributions_with_a_pointer() -> None:
    listed = content.INTRODUCTION.listed
    assert len(listed) == 9
    for head, text in listed:
        assert "[[sec:" in head, head
        assert "[[fig:" in text, head


def test_the_limitations_are_ten_prose_items_without_labels() -> None:
    heads = [h for h, _ in content.LIMITATIONS.runin]
    assert len(heads) == 10
    assert not any(re.fullmatch(r"L\d+", h) for h in heads)
    assert not content.LIMITATIONS.paragraphs


def test_section_keys_are_unique() -> None:
    keys = [s.key for s in ALL]
    assert len(keys) == len(set(keys))


def test_every_token_names_a_known_target() -> None:
    sections = {s.key for s in ALL}
    figures = {*registry.MAIN_FIGURES, *registry.APPENDIX_FIGURES}
    tables = {*registry.MAIN_TABLES, *registry.APPENDIX_TABLES}
    bib = references.parse(BIB.read_text("utf-8"))
    for text in _all_texts():
        for token in TOKEN.findall(text):
            kind, _, rest = token.partition(":")
            if kind == "sec":
                assert rest in sections, token
            elif kind == "fig":
                assert rest in figures, token
            elif kind == "table":
                assert rest in tables, token
            elif kind == "cite":
                for key in rest.split(","):
                    assert key in bib, (token, key)
            elif kind in ("num", "count", "runs"):
                owner, _, name = rest.partition(":")
                key, _, field = name.rpartition(".")
                assert owner in ("fig", "table"), token
                assert key in (figures if owner == "fig" else tables), token
                assert field, token
            elif kind == "findings":
                slug, bar, shown = rest.partition("|")
                assert re.fullmatch(r"[a-z0-9-]+", slug), token
                assert bar, token
                assert shown, token
            else:
                pytest.fail(f"unknown token {token}")


def test_every_item_is_placed_once_and_every_appendix_item_is_cited() -> None:
    placed = [i for s in ALL for i in s.items]
    wanted = [f"fig:{k}" for k in registry.MAIN_FIGURES]
    wanted += [f"table:{k}" for k in registry.MAIN_TABLES]
    wanted += [f"fig:{k}" for k in registry.APPENDIX_FIGURES]
    wanted += [f"table:{k}" for k in registry.APPENDIX_TABLES]
    assert sorted(placed) == sorted(wanted)
    body = " ".join(
        t for top in content.BODY for s in _sections(top) for t in _texts(s)
    )
    for item in wanted[
        len(registry.MAIN_FIGURES) + len(registry.MAIN_TABLES) :
    ]:
        kind, _, key = item.partition(":")
        assert f"[[{kind}:{key}]]" in body, item


def test_items_follow_the_registry_order_of_the_main_text() -> None:
    placed = [
        i
        for top in (content.INTRODUCTION, content.METHOD, content.RESULTS)
        for s in _sections(top)
        for i in s.items
    ]
    figures = [i[4:] for i in placed if i.startswith("fig:")]
    tables = [i[6:] for i in placed if i.startswith("table:")]
    assert figures == list(registry.MAIN_FIGURES)
    assert tables == list(registry.MAIN_TABLES)


def test_content_numbers_have_no_stray_digits() -> None:
    for text in _opening_texts():
        assert _digits_left(text) == [], text


def test_the_numbers_test_would_catch_a_pasted_value() -> None:
    assert _digits_left("it reads 0.847 on [[num:fig:a.b]]") != []
    assert _digits_left("GPT-2 at relative coefficient 0.1 in 2023") == []
    assert _digits_left("it holds on 10 of 10 checkpoints") != []
    assert _digits_left("a margin of 0.5 on the reader") != []
    assert _digits_left("a margin of 0.55 below 0.5") != []
    assert _digits_left("all 10 checkpoints, none of the 10, below 0.5") == []


def test_no_dashes_backticks_or_underscores_in_prose() -> None:
    for text in _all_texts():
        bare = _bare(text)
        assert not any(d in bare for d in DASHES), text
        assert "`" not in bare, text
        assert "_" not in bare, text


def test_run_codes_stay_out_of_the_prose() -> None:
    kept = [
        t
        for s in ALL
        if s.key != content.REPRODUCIBILITY.key
        for t in [s.title, *_texts(s)]
    ]
    for text in [content.ABSTRACT, content.TITLE, *kept]:
        assert RUN_CODE.search(_bare(text)) is None, text
    assert RUN_CODE.search("".join(_texts(content.REPRODUCIBILITY))) is None


def test_the_word_first_is_only_the_principal_component() -> None:
    for text in _all_texts():
        stripped = _bare(text).replace("first principal component", "")
        assert re.search(r"\bfirst\b", stripped, re.I) is None, text


def test_no_claim_pools_base_and_post_trained_checkpoints() -> None:
    for text in _all_texts():
        assert "all ten" not in text.lower()
        assert not re.search(r"\b\d+ of (the )?10 checkpoints", text), text


def test_the_repository_handle_appears_only_in_repository_links() -> None:
    handle = "javierdejesusda"
    assert handle in content.REPO_URL
    for text in _all_texts():
        assert handle not in text
    for url in (
        content.REPO_TEXT,
        content.FINDINGS_URL,
        content.METHODOLOGY_URL,
        content.DATA_URL,
    ):
        assert url.startswith(("github.com/", "https://github.com/"))
        assert url.count(handle) == 1


def test_the_notation_terms_are_used_in_the_text() -> None:
    out = notation.build(Manifest(ROOT))
    text = " ".join(_bare(t) for t in _all_texts()).lower()
    for term, _ in out.rows:
        term = term.lower().removesuffix(" c")
        if term == "pair":
            assert "pairs" in text
        else:
            assert term in text, term


def test_the_wording_of_the_baseline_rules_is_in_the_text() -> None:
    method = " ".join(
        _bare(t) for s in _sections(content.METHOD) for t in _texts(s)
    )
    assert "margin over the random direction of the same run" in method
    assert "never pooled" in method
    assert "distinct-1" in method
    assert "perplexity" in method
    abstract = _bare(content.ABSTRACT)
    assert "fluent" not in abstract


def test_content_design_facts() -> None:
    method = {
        "r1": json.loads(
            (run("r1", "gpt2") / "config.json").read_text("utf-8")
        ),
        "r6": json.loads(
            (run("r6", "gpt2") / "config.json").read_text("utf-8")
        ),
        "c1": json.loads(
            (run("c1", "gpt2") / "config.json").read_text("utf-8")
        ),
        "m1": json.loads(
            (run("m1", "gpt2") / "config.json").read_text("utf-8")
        ),
        "p9d": json.loads(
            (run("p9d", "qwen3-1.7b-base") / "config.json").read_text("utf-8")
        ),
    }
    assert method["r1"]["n_eval_pairs"] == 256
    assert method["r1"]["n_pairs"] == 64
    assert method["r1"]["n_seeds"] == 10
    assert method["r6"]["n_seeds"] == 15
    assert params.REPORT_PAIRS["sentiment"] == 214
    r5 = data(run("r5", "gpt2"), "summary.json")["n_eval_pairs"]["transfer"]
    assert (r5["val"], r5["test"]) == (53, 59)
    assert data(run("r6", "gpt2"), "summary.json")["n_test"] == 173
    assert method["c1"]["n_neutral"] == 200
    assert method["c1"]["n_steer_seeds"] == 3
    assert method["m1"]["n_steer_seeds"] == 3
    assert "random" in method["m1"]["methods"]
    assert method["p9d"]["n_random"] == 3
    sizes = {
        count
        for model in models("p10b")
        for family in data(run("p10b", model), "summary.json")[
            "counts"
        ].values()
        for side in family.values()
        for split, count in side.items()
        if split == "test"
    }
    assert (min(sizes), max(sizes)) == (22, 30)


def test_the_design_digits_of_method_limitations_and_reproducibility() -> None:
    facts = (
        r"scores 0\.5(?!\d)",
        r"base seed 0",
        r"(?:above|over) 0\.5(?!\d)",
        r"fits 10 readers",
        r"(?:draw of|at) 64 training pairs",
        r"(?:each of|across|on) the 10 checkpoints",
        r"across 10 checkpoints",
        r"over the 10 readers",
        r"below about 0\.15(?!\d)",
        r"coefficients \u22120\.3 to 0\.3",
        r"at most 1\.5 times",
        r"at least 0\.15(?!\d)",
        r"[+\u2212]0\.3(?!\d)",
        r"not 0\.5(?!\d)",
        r"256",
        r"214",
        r"53",
        r"59",
        r"173",
        r"200",
        r"15",
        r"22",
        r"30",
        r"95",
        r"1\.96",
        r"1\.000",
        r"0\.6B",
        r"1/4",
        r"\bp[12]\b",
        r"\bn[12]\b",
        r"\b16\b",
        r"\b[234]\b",
        r"\b1\b",
        r"P9b",
        r"R2",
        r"SiEBERT",
    )
    for top in (content.METHOD, content.LIMITATIONS, content.REPRODUCIBILITY):
        for section in _sections(top):
            for text in _texts(section):
                assert _digits_left(text, (*GENERIC, *facts)) == [], text


def test_sections_are_frozen_dataclasses() -> None:
    assert dataclasses.is_dataclass(content.Section)
    with pytest.raises(dataclasses.FrozenInstanceError):
        content.INTRODUCTION.title = "x"  # type: ignore[misc]


def test_the_discussion_names_the_fluency_rule_it_says_steering_breaks() -> (
    None
):
    text = " ".join(_texts(content.DISCUSSION))
    sentence = text[text.index("Steering moves") :].split(". ")[0]
    assert "breaks fluency" in sentence
    assert "perplexity" in sentence
    assert "distinct-1" in sentence
    assert "1.5" in sentence
    assert "0.15" in sentence


def test_the_approximate_contribution_counts_the_reader_and_the_control() -> (
    None
):
    text = " ".join(_all_texts())
    assert "is matched by" not in text
    assert "[[count:table:approximate.above_words]]" in text
    assert "[[count:fig:approximate.reader_above_random]]" in text
    assert "[[count:fig:approximate.control_below_half]]" in text
    assert "[[count:fig:approximate.control_above_95]]" in text
    assert "a sign flip between the training and the test concepts" in text
    assert "already separates the test concepts" in text
    assert "reader_in_random" not in text
    assert "people_outside_band" not in text
    assert "random_high" not in text


def test_the_negated_set_claim_counts_checkpoints_below_chance() -> None:
    text = " ".join(_all_texts())
    assert "at least one negated set falls below chance" in text
    assert "[[count:fig:truth.neg_below_half]]" in text


def test_counts_over_runs_use_the_runs_token() -> None:
    text = " ".join(_all_texts())
    assert "[[runs:fig:knowledge_steering.distinguishable]]" in text
    assert "[[count:fig:knowledge_steering.distinguishable]]" not in text


def test_section_one_counts_the_readings_and_the_confidence_gain() -> None:
    reading = _listed("Where a concept")
    for token in ("utility", "morality", "truth"):
        assert f"[[count:fig:teaser.{token}]]" in reading
    knowing = _listed("Knowing what")
    assert "[[count:fig:teaser.knowledge_confidence]]" in knowing
    for text in (content.ABSTRACT, reading, knowing):
        assert "knowledge_confidence_n" not in text
        assert "any of the" not in text


def test_the_steering_contribution_names_both_fluency_thresholds() -> None:
    text = _listed("Steering")
    start = text.index("keeps the fluency rule")
    sentence = text[start : start + 220]
    assert "perplexity ratio at most 1.5" in sentence
    assert "distinct-1 at least 0.15" in sentence


def test_known_questions_use_the_runs_token_and_name_the_criterion() -> None:
    text = " ".join(_all_texts())
    assert "[[runs:fig:knowledge_steering.known_ge_unknown]]" in text
    assert "by the largest shift of the mean dose curve" in text
    assert "known_ge_unknown_" not in text
    assert "knowledge_steering.n_" not in text


def _text() -> str:
    return " ".join(_all_texts())


def test_axbench_is_described_by_what_it_found() -> None:
    text = _text()
    assert "difference-in-means is the best concept detector" in text
    assert "prompting beats every method for steering" in text
    assert "sparse autoencoders are not competitive on either" in text
    assert "simple baselines match or beat" not in text


def test_reproducibility_admits_the_lorra_adapter_and_names_no_run_code() -> (
    None
):
    paragraphs = " ".join(_texts(content.REPRODUCIBILITY))
    assert "no experiment trains a model beyond" not in paragraphs
    assert "LoRRA adapter" in paragraphs
    assert "each of the 10 checkpoints" in paragraphs
    assert not RUN_CODE.search(paragraphs)
    assert "seed" in paragraphs
    assert "tomography run" in paragraphs


def test_the_text_baseline_is_scoped_in_the_abstract() -> None:
    assert "where the wording leaks the label" in content.ABSTRACT
    assert "for TruthfulQA, knowledge and approximate answers" in " ".join(
        content.INTRODUCTION.paragraphs
    )


def test_method_states_the_draws_positions_and_steering_depths() -> None:
    text = _text()
    assert "20 for the held-out sentiment run" in text
    assert "response tokens" in text
    assert "40, 55 and 70 percent of the depth" in text
    assert "for the sentiment run" in text


def test_the_approximate_claims_name_the_interval_criterion() -> None:
    text = _text()
    assert "above the 95 percent interval of a classifier" in text
    assert "[[count:table:approximate.layer_one]]" in text


def test_the_confidence_gain_is_bounded_not_called_nothing() -> None:
    text = _text()
    assert "adds nothing to the confidence" not in text
    assert "[[num:fig:knowledge_reading.dir_inc_hi_max]]" in text
    assert "gain_upper_max" not in text


def test_the_paired_differences_are_counted_by_token() -> None:
    for text in (_listed("Post-training"), content.DISCUSSION.paragraphs[0]):
        assert "[[num:fig:base_post.paired_excluding_zero]]" in text
        assert "[[num:fig:base_post.paired_total]]" in text
        assert "three other" not in text
    assert "[[num:fig:base_post.paired_total]]" in _opener("base_post")


def test_the_three_excluding_differences_are_named() -> None:
    text = _listed("Post-training")
    assert "0.6B morality, 1.7B utility and 0.6B honesty" in text
    assert "leaves most other readings" not in _text()


def test_limitations_cover_the_design_choices_and_stay_ten_items() -> None:
    items = " ".join(h + " " + t for h, t in content.LIMITATIONS.runin)
    assert len(content.LIMITATIONS.runin) == 10
    for phrase in (
        "one prompt format",
        "one judge family",
        "two model families",
        "small validation sets",
        "alias string matches",
        "no hand check",
    ):
        assert phrase in items, phrase


def test_the_discussion_cites_a_section_for_steering() -> None:
    text = " ".join(_texts(content.DISCUSSION))
    assert "behaves as the paper describes" not in text
    assert "[[sec:steering]]" in text


def test_the_abstract_is_prose_built_from_replaceable_finding_sentences() -> (
    None
):
    assert content.ABSTRACT_FINDER in content.ABSTRACT
    assert content.ABSTRACT_STEERING in content.ABSTRACT
    assert content.ABSTRACT_FINDER.count("[[count:") == 3
    assert content.ABSTRACT_STEERING.count("[[count:") == 4
    assert content.ABSTRACT.count("[[count:") == 8
    assert "[[count:fig:teaser.truthfulqa]]" in content.ABSTRACT
    abstract = content.ABSTRACT
    assert "utility and morality on all 10 checkpoints" in abstract
    assert "truth on every Qwen3 checkpoint and no GPT-2 size" in abstract
    assert "by the 95 percent interval" in abstract.lower()
    assert "rarely win" not in abstract
    assert "does not beat a words classifier" in abstract
    assert "of the three Qwen3" not in _text()
    assert "of 3 Qwen3 sizes" in _listed("Post-training")
    assert "sent_below" not in content.ABSTRACT
    assert "[[count:fig:teaser.knowledge]]" not in content.ABSTRACT


def test_the_introduction_defines_its_terms_before_the_list_uses_them() -> None:
    first = " ".join(content.INTRODUCTION.paragraphs)
    listed = " ".join(h + " " + t for h, t in content.INTRODUCTION.listed)
    for term in ("margin", "relative depth", "relative coefficient"):
        assert term in listed
        assert f"{term} is" in first


def _listed(head: str) -> str:
    for h, t in content.INTRODUCTION.listed:
        if h.startswith(head):
            return t
    raise AssertionError(head)


def test_the_steering_claims_are_two_sentences_with_their_thresholds() -> None:
    first, second = content.ABSTRACT_STEERING.split(". ")
    assert "at most 1.5" not in content.ABSTRACT_STEERING
    assert "[[count:fig:steering.moves01]]" in first
    assert "[[count:fig:steering.fluent01]]" in first
    assert "[[count:fig:teaser.steering]]" in second
    assert "[[count:fig:teaser.steering_fluent]]" in second
    assert "GPT-2 small has no fluency measure" in _listed("Steering")
    for text in (_listed("Steering"),):
        assert "at most 1.5" in text
        assert "distinct-1 at least 0.15" in text
        assert "second judge" in text
    section = _listed("Steering")
    for token in ("moves01", "fluent01"):
        assert f"[[count:fig:steering.{token}]]" in section
    for token in ("steering", "steering_fluent"):
        assert f"[[count:fig:teaser.{token}]]" in section
    assert "no fluency measure" in section
    assert "no checkpoint is fluent" not in _text()


def test_the_finder_claims_are_per_concept_and_not_causal() -> None:
    for text in (content.ABSTRACT_FINDER, _listed("Finders")):
        for token in ("pca_sentiment", "pca_truth", "pca_utility"):
            assert f"[[count:fig:finders.{token}]]" in text
        assert "confounded with the concept and" in text
    assert "by point estimate only" in _listed("Finders")
    assert "[[count:fig:finders.pca_utility_point]]" in _listed("Finders")
    everything = _text()
    assert "depends on matched" not in everything
    assert "not reliably above a random direction" not in everything


RESULT_KEYS = (
    "finders",
    "reading",
    "wording",
    "truth",
    "steering",
    "knowledge_reading",
    "knowledge_steering",
    "approximate",
    "base_post",
)


def test_each_results_subsection_opens_with_its_answer_before_the_figures() -> (
    None
):
    subs = {s.key: s for s in content.RESULTS.subsections}
    assert tuple(subs) == RESULT_KEYS
    for key, sub in subs.items():
        opener, pointer = sub.paragraphs[0], sub.paragraphs[1]
        assert re.search(r"\[\[(count|runs|num):", opener), key
        assert "[[fig:" not in opener, key
        assert "[[table:" not in opener, key
        assert "[[fig:" in pointer or "[[table:" in pointer, key
        assert len(re.findall(r"[.;] ", opener)) <= 2, key


def _opener(key: str) -> str:
    subs = {s.key: s for s in content.RESULTS.subsections}
    return subs[key].paragraphs[0]


def test_openers_state_the_criterion_and_the_counts_not_a_verdict() -> None:
    finders = _opener("finders")
    assert not finders.startswith("Not reliably")
    for token in ("pca_utility", "pca_truth", "pca_sentiment"):
        assert f"[[count:fig:finders.{token}]]" in finders
    assert "by the 95 percent interval" in finders.lower()
    assert "[[count:fig:finders.pca_utility_point]]" in finders
    assert "by the 95 percent interval" in _opener("reading").lower()
    truth = _opener("truth")
    assert "by point estimate" in truth.lower()
    assert "[[count:fig:teaser.truthfulqa]]" in truth
    assert "by the 95 percent interval" in truth
    post = _opener("base_post")
    assert not post.startswith("Little")
    assert "[[num:fig:base_post.sent_below]] of 3 Qwen3 sizes" in post
    assert "[[num:fig:base_post.paired_excluding_zero]]" in post


def test_the_wording_opener_leads_with_the_noise_count() -> None:
    opener = _opener("wording")
    assert opener.startswith("Wording moves a sentiment reader by more than")
    assert "[[count:fig:wording.span_above_noise]]" in opener
    assert "point comparison" in opener
    assert not opener.startswith("Yes")


def test_the_finder_opener_clause_is_true_as_printed() -> None:
    opener = _opener("finders")
    assert "on GPT-2 it is above" not in opener
    assert (
        "by point estimate it is above it for utility on "
        "[[count:fig:finders.pca_utility_point]]"
    ) in opener


def test_the_abstract_uses_one_term_and_explains_the_gpt2_gap() -> None:
    abstract = content.ABSTRACT
    assert "above a random direction" in abstract
    assert "beats every random direction" in abstract
    assert abstract.count("relative coefficient") == 2
    assert "GPT-2 small has no fluency measure" in abstract
    assert "above it for sentiment" not in abstract


OTHER_SLUGS = (
    "lorra-adapter-lr1-and-the-truthfulqa-control-c4",
    "emotions-r4",
    "emotion-steering-c2",
    "lie-gallery-c5",
    "template-steering-p7",
    "direction-agreement-and-logit-lens-on-gpt2-a1-a2",
    "token-position-and-a-misleading-sentence-p2",
)


def _other() -> str:
    (section,) = [s for s in content.APPENDICES if s.key == "appendix_other"]
    return " ".join(section.paragraphs)


def test_the_other_experiments_section_points_to_each_findings_section() -> (
    None
):
    text = _other()
    for slug in OTHER_SLUGS:
        assert f"[[findings:{slug}|" in text, slug
    shown = re.sub(r"\[\[findings:[^|]*\|(.*?)\]\]", r"", text)
    assert 80 <= len(_bare(shown).split()) <= 150


def test_the_other_experiments_section_has_no_numbers_or_run_codes() -> None:
    bare = _bare(_other())
    assert re.search(r"\d", bare) is None
    assert RUN_CODE.search(bare) is None
    for covered in ("seed agreement", "misleading sentence", "R1", "sentiment"):
        assert covered not in bare


def test_the_reproducibility_section_points_to_the_other_experiments() -> None:
    text = " ".join(_texts(content.REPRODUCIBILITY))
    assert "LoRRA" in text
    assert "[[sec:appendix_other]]" in text
