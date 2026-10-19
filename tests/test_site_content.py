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
from tomography.site.stats import RoleCount
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
RESULT_KEYS = (
    "finders",
    "reading",
    "known_unknown",
    "truth",
    "steering",
    "base_post",
)
APPENDIX_KEYS = (
    "appendix_assets",
    "appendix_method",
    "appendix_reading",
    "appendix_wording",
    "appendix_truth",
    "appendix_steering",
    "appendix_knowledge",
    "appendix_approximate",
    "appendix_post",
    "appendix_other",
)


def _sections(
    section: content.Section,
) -> list[content.Section]:
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


def _all_texts() -> list[str]:
    texts = [content.ABSTRACT, content.TITLE, content.TLDR]
    for section in ALL:
        texts += [section.title, *_texts(section)]
    return texts


def _text() -> str:
    return " ".join(_all_texts())


def _bare(text: str) -> str:
    return TOKEN.sub("", text)


def _digits_left(text: str, allowed: tuple[str, ...] = GENERIC) -> list[str]:
    text = _bare(text)
    for pattern in allowed:
        text = re.sub(pattern, " ", text)
    return re.findall(r"\d+", text)


def _opening_texts() -> list[str]:
    texts = [content.ABSTRACT, content.TLDR]
    for section in (
        content.KEY_FINDINGS,
        content.INTRODUCTION,
        *content.RESULTS.subsections,
    ):
        texts += _texts(section)
    return texts


def _findings(head: str) -> str:
    for h, t in content.KEY_FINDINGS.listed:
        if h.startswith(head):
            return t
    raise AssertionError(head)


def _opener(key: str) -> str:
    subs = {s.key: s for s in content.RESULTS.subsections}
    return subs[key].paragraphs[0]


def test_the_title_and_abstract_are_the_planned_ones() -> None:
    assert content.TITLE == (
        "Reading and Steering Concepts with One Direction: Representation "
        "Engineering on GPT-2 and Qwen3"
    )
    assert content.ABSTRACT.startswith(
        "We test representation engineering (Zou et al., 2023) on four GPT-2"
    )


def test_the_body_sections_are_the_six_of_the_paper_in_order() -> None:
    assert [s.title for s in content.BODY] == [
        "Introduction",
        "Related work",
        "Method",
        "Results",
        "Discussion and limitations",
        "Reproducibility",
    ]
    assert [s.key for s in content.RESULTS.subsections] == list(RESULT_KEYS)
    assert [s.key for s in content.APPENDICES] == list(APPENDIX_KEYS)
    assert [s.title for s in content.APPENDICES] == [
        "Assets and notation",
        "Method details",
        "Reading details",
        "Wording",
        "Truth and honesty",
        "Steering details",
        "Knowledge",
        "Approximate answers",
        "Post-training details",
        "Other experiments",
    ]


def test_the_page_opens_with_a_summary_the_overview_and_six_findings() -> None:
    assert content.FRONT_ITEMS == ("fig:teaser",)
    assert content.TLDR.count(". ") == 0
    assert len(content.TLDR.split()) <= 60
    heads = [h for h, _ in content.KEY_FINDINGS.listed]
    assert len(heads) == 6
    for head, text in content.KEY_FINDINGS.listed:
        assert "[[sec:" in head, head
        assert "[[fig:" in text, head
        assert len(text.split()) <= 70, head
    assert not content.KEY_FINDINGS.items
    assert "[[fig:teaser]]" in " ".join(_texts(content.INTRODUCTION))


def test_the_introduction_ends_with_a_list_of_contributions() -> None:
    listed = content.INTRODUCTION.listed
    assert len(listed) == 5
    for head, _ in listed:
        assert "[[sec:" in head, head
    assert content.INTRODUCTION.paragraphs[-1] == "Our contributions are:"


def test_the_discussion_merges_the_limitations_into_seven_items() -> None:
    heads = [h for h, _ in content.DISCUSSION.runin]
    assert len(heads) == 7
    assert not any(re.fullmatch(r"L\d+", h) for h in heads)
    assert len(content.DISCUSSION.paragraphs) == 1
    assert content.DISCUSSION.key == "limitations"
    assert "Missing controls." not in heads
    assert "Multiple comparisons." not in heads


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
    placed = [i for s in ALL for i in s.items] + list(content.FRONT_ITEMS)
    wanted = [f"fig:{k}" for k in registry.MAIN_FIGURES]
    wanted += [f"table:{k}" for k in registry.MAIN_TABLES]
    wanted += [f"fig:{k}" for k in registry.APPENDIX_FIGURES]
    wanted += [f"table:{k}" for k in registry.APPENDIX_TABLES]
    assert sorted(placed) == sorted(wanted)
    for section in content.APPENDICES:
        text = " ".join(_texts(section))
        for item in section.items:
            kind, _, key = item.partition(":")
            assert f"[[{kind}:{key}]]" in text, item


def test_every_appendix_section_is_pointed_to_from_the_body() -> None:
    body = " ".join(
        t for top in content.BODY for s in _sections(top) for t in _texts(s)
    )
    for section in content.APPENDICES:
        if section.key in ("appendix_assets", "appendix_steering"):
            continue
        assert f"[[sec:{section.key}]]" in body, section.key
    assert "[[table:assets]]" in body
    assert "[[table:steering]]" in body


def test_items_follow_the_registry_order_of_the_main_text() -> None:
    placed = list(content.FRONT_ITEMS) + [
        i
        for top in (content.INTRODUCTION, content.METHOD, content.RESULTS)
        for s in _sections(top)
        for i in s.items
    ]
    figures = [i[4:] for i in placed if i.startswith("fig:")]
    tables = [i[6:] for i in placed if i.startswith("table:")]
    assert figures == list(registry.MAIN_FIGURES)
    assert tables == list(registry.MAIN_TABLES)


def test_each_appendix_section_opens_with_what_it_holds_and_supports() -> None:
    for section in content.APPENDICES:
        assert section.paragraphs, section.key
        if section.key in ("appendix_assets", "appendix_other"):
            continue
        first = section.paragraphs[0].split(". ")[0]
        assert len(first.split()) <= 60, section.key


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
    for text in [content.ABSTRACT, content.TITLE, content.TLDR, *kept]:
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


def _method_text() -> str:
    return " ".join(
        _bare(t) for s in _sections(content.METHOD) for t in _texts(s)
    )


def test_the_wording_of_the_baseline_rules_is_in_the_text() -> None:
    method = _method_text()
    assert "margin over the random direction of the same run" in method
    assert "never pooled" in method
    assert "distinct-1" in method
    assert "perplexity" in method
    assert "fluent" not in _bare(content.ABSTRACT)
    assert "fluent" not in _bare(content.TLDR)


def test_the_margin_text_does_not_claim_one_layer_for_every_comparison() -> (
    None
):
    method = _method_text()
    assert "at the same layer, not over" not in method
    assert "except in the finder comparison" in method
    assert "layer chosen on its own select split" in method


def test_the_confidence_baseline_is_five_stacked_signals() -> None:
    text = _text()
    assert "answer probability" not in text
    assert "stacks five output signals" in text


def test_the_method_is_about_nine_hundred_words() -> None:
    words = len(
        " ".join(
            t for s in _sections(content.METHOD) for t in _texts(s)
        ).split()
    )
    assert 700 <= words <= 1000, words


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


def test_the_design_digits_of_method_discussion_and_reproducibility() -> None:
    facts = (
        r"scores 0\.5(?!\d)",
        r"base seed 0",
        r"(?:above|over) 0\.5(?!\d)",
        r"fits 10 readers",
        r"A run fits 10 readers",
        r"(?:draw of|at) 64 training pairs",
        r"(?:each of|across|on) the 10 checkpoints",
        r"across 10 checkpoints",
        r"over the 10 readers",
        r"below about 0\.15(?!\d)",
        rf"coefficients {content.MINUS}0\.3 to 0\.3",
        r"at most 1\.5 times",
        r"at least 0\.15(?!\d)",
        rf"[+{content.MINUS}]0\.3(?!\d)",
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
    appendix = [s for s in content.APPENDICES if s.key == "appendix_method"]
    for top in (
        content.METHOD,
        content.DISCUSSION,
        content.REPRODUCIBILITY,
        *appendix,
    ):
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


def test_the_approximate_counts_and_criteria_are_in_the_appendix() -> None:
    (section,) = [
        s for s in content.APPENDICES if s.key == "appendix_approximate"
    ]
    text = " ".join(_texts(section))
    assert "is matched by" not in _text()
    assert "[[count:table:approximate.above_words]]" in text
    assert "[[count:table:approximate.layer_one]]" in text
    assert "[[count:fig:approximate.control_below_half]]" in text
    assert "[[count:fig:approximate.control_above_95]]" in text
    assert "a sign flip between the training and the test concepts" in text
    assert "already separates the test concepts" in text
    for gone in ("reader_in_random", "people_outside_band", "random_high"):
        assert gone not in _text()


def _section(key: str) -> content.Section:
    (found,) = [s for s in ALL if s.key == key]
    return found


def test_known_and_unknown_questions_have_a_finding_and_a_subsection() -> None:
    heads = [h for h, _ in content.KEY_FINDINGS.listed]
    assert [h.split(" (")[0] for h in heads][2:5] == [
        "Known and unknown questions",
        "Truth",
        "Steering",
    ]
    keys = [s.key for s in content.RESULTS.subsections]
    assert keys.index("known_unknown") == keys.index("reading") + 1
    assert keys.index("truth") == keys.index("known_unknown") + 1
    sub = _section("known_unknown")
    assert sub.title == "Known and unknown questions"
    assert sub.items == (
        "fig:knowledge_reading",
        "fig:knowledge_steering",
        "fig:approximate",
    )
    assert [h for h, _ in sub.runin] == [
        "Reading.",
        "Steering.",
        "Approximate answers.",
    ]
    text = " ".join(_texts(sub))
    for token in (
        "[[count:fig:knowledge_reading.above_best]]",
        "[[num:fig:knowledge_reading.dir_inc_hi_max_gpt2]]",
        "[[runs:fig:knowledge_steering.distinguishable]]",
        "[[runs:fig:knowledge_steering.selective]]",
        "[[runs:fig:knowledge_steering.known_ge_unknown]]",
        "[[num:fig:a_knowledge_matrix.close]]",
        "[[num:fig:a_knowledge_matrix.types]]",
        "[[num:fig:approximate.forced_base_max]]",
        "[[num:fig:approximate.forced_post_max]]",
        "[[count:fig:approximate.reader_above_random]]",
        "[[count:table:approximate.above_words]]",
        "[[sec:appendix_knowledge]]",
        "[[sec:appendix_approximate]]",
        "[[fig:a_knowledge_matrix]]",
    ):
        assert token in text, token
    assert "told to give one exact number" in text


def test_the_knowledge_and_approximate_figures_left_the_appendix() -> None:
    assert "fig:knowledge_steering" not in _section("appendix_knowledge").items
    assert "fig:approximate" not in _section("appendix_approximate").items
    assert "fig:a_links" in _section("appendix_approximate").items


def test_the_approximate_appendix_holds_the_people_position_counts() -> None:
    text = " ".join(_texts(_section("appendix_approximate")))
    assert "[[count:fig:approximate.outside]]" in text
    for role in ("gpt2", "base", "post"):
        for name in ("position", "band"):
            for end in ("min", "max"):
                assert f"[[num:fig:approximate.{name}_{role}_{end}]]" in text
    assert "approximate.position_min" not in text
    assert "approximate.band_min" not in text


def test_the_abstention_limit_names_phrase_matching_and_generic_answers() -> (
    None
):
    text = " ".join(h + " " + t for h, t in content.DISCUSSION.runin)
    assert "phrase matching without human labels" in text
    assert "a generic answer to an unknowable detail" in text
    assert "a specific invented one" in text
    assert "label which a non-abstaining answer is" in text


def test_the_summary_and_abstract_name_the_knowledge_result() -> None:
    assert "known-against-unknown direction does not beat the model's" in (
        content.TLDR
    )
    assert "ranks questions above popularity and text classifiers" in (
        content.ABSTRACT
    )
    contribution = content.INTRODUCTION.listed[-2]
    assert "[[sec:known_unknown]]" in contribution[0]
    assert "confidence of the model" in contribution[1]


def test_the_negated_set_claim_counts_checkpoints_below_chance() -> None:
    text = _text()
    assert "at least one negated set falls below chance" in text
    assert "[[count:fig:truth.neg_below_half]]" in text


def test_counts_over_runs_use_the_runs_token() -> None:
    text = _text()
    assert "[[runs:fig:knowledge_steering.distinguishable]]" in text
    assert "[[count:fig:knowledge_steering.distinguishable]]" not in text
    assert "[[runs:fig:knowledge_steering.known_ge_unknown]]" in text
    assert "by the largest shift of the mean dose curve" in text.lower()
    assert "known_ge_unknown_" not in text
    assert "knowledge_steering.n_" not in text


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
    assert "fitted on all 10 checkpoints" in paragraphs
    assert not RUN_CODE.search(paragraphs)
    assert "seed" in paragraphs
    assert "tomography run" in paragraphs


def test_the_text_baseline_is_scoped_in_the_abstract_and_introduction() -> None:
    assert "where the wording leaks the label" in content.ABSTRACT
    assert "for the readings, TruthfulQA, knowledge and approximate" in (
        " ".join(content.INTRODUCTION.paragraphs)
    )


def test_the_details_the_main_text_dropped_are_in_appendix_b() -> None:
    (section,) = [s for s in content.APPENDICES if s.key == "appendix_method"]
    text = " ".join(_texts(section))
    assert "20 for the held-out sentiment run" in text
    assert "response tokens" in text
    assert "perplexity of 200 held-out WikiText-103 passages" in text
    assert "1.96" in text
    method = _method_text()
    assert "40, 55 and 70 percent of the depth" in method
    assert "for the sentiment run" in method
    assert "1.96" not in method


def test_the_confidence_gain_is_bounded_not_called_nothing() -> None:
    text = _text()
    assert "adds nothing to the confidence" not in text
    for role in ("gpt2", "base", "post"):
        assert f"[[num:fig:knowledge_reading.dir_inc_hi_max_{role}]]" in text
    assert "gain_upper_max" not in text


def test_the_paired_differences_are_counted_by_token_and_not_named() -> None:
    text = _findings("Post-training")
    assert "[[num:fig:base_post.paired_excluding_zero]]" in text
    assert "[[num:fig:base_post.paired_total]]" in text
    assert "three other" not in _text()
    assert "0.6B morality" not in _text()
    assert "leaves most other readings" not in _text()


def test_the_post_training_findings_state_the_interval_criterion() -> None:
    for text in (
        _findings("Post-training"),
        _opener("base_post"),
        content.ABSTRACT,
        " ".join(_texts(content.DISCUSSION)),
    ):
        assert "with an interval below zero" in text, text
        assert "[[num:fig:base_post.sent_below]] of 3 Qwen3 sizes" in text


def test_limitations_cover_the_design_choices() -> None:
    items = " ".join(h + " " + t for h, t in content.DISCUSSION.runin)
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


def test_the_abstract_is_prose_without_run_counts() -> None:
    abstract = content.ABSTRACT
    assert "[[count:" not in abstract
    assert "of 3 Qwen3 sizes" in abstract
    assert "by the 95 percent interval" in abstract.lower()
    assert "does not add to the confidence of the model" in abstract
    assert "not above a classifier on the words alone" in abstract
    assert "confounded with the concept and the data" in abstract
    assert abstract.count("relative coefficient") == 2
    assert "rarely" not in abstract
    assert "all 10" not in abstract
    assert "of the three Qwen3" not in _text()


def test_the_introduction_defines_its_terms_before_the_list_uses_them() -> None:
    first = " ".join(content.INTRODUCTION.paragraphs)
    for term in ("margin", "relative depth", "relative coefficient"):
        assert f"{term} is" in first


def test_each_key_finding_names_its_counts_and_its_figures() -> None:
    finder = _findings("The finder")
    assert "[[count:fig:finders.pca_sentiment]]" in finder
    assert "confounded with the concept and the data" in finder
    reading = _findings("Where a concept")
    assert "[[count:fig:teaser.truth]]" in reading
    steering = _findings("Steering")
    for token in ("moves01", "fluent01"):
        assert f"[[count:fig:steering.{token}]]" in steering
    assert "[[count:fig:teaser.steering]]" in steering
    assert content.FLUENCY_RULE in steering
    assert "no checkpoint is fluent" not in _text()


def test_each_results_subsection_opens_with_its_answer_before_the_figures() -> (
    None
):
    subs = {s.key: s for s in content.RESULTS.subsections}
    assert tuple(subs) == RESULT_KEYS
    for key, sub in subs.items():
        assert len(sub.paragraphs) == 1, key
        opener = sub.paragraphs[0]
        assert "[[fig:" not in opener, key
        assert "[[table:" not in opener, key
        assert len(re.findall(r"[.;] ", opener)) == 0, key
        assert opener.endswith("."), key
        assert sub.items, key
        assert sub.after, key
        limit = 420 if key == "known_unknown" else 260
        assert len(" ".join(_texts(sub)).split()) <= limit, key


def test_the_results_leads_carry_no_per_role_counts() -> None:
    for sub in content.RESULTS.subsections:
        assert "[[count:" not in sub.paragraphs[0], sub.key
        assert "[[runs:" not in sub.paragraphs[0], sub.key


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
    shown = re.sub(r"\[\[findings:[^|]*\|(.*?)\]\]", r"", text)
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


@pytest.fixture(scope="module")
def built() -> dict[str, registry.FigureOutput]:
    manifest = Manifest(ROOT)
    return {
        key: registry.build_figure(key, manifest)
        for key in ("teaser", "finders", "steering", "wording")
    }


def _all(count: dict[str, RoleCount]) -> bool:
    return all(c.hits == c.total for c in count.values())


def _none(count: dict[str, RoleCount]) -> bool:
    return all(c.hits == 0 for c in count.values())


def _hits(count: dict[str, RoleCount]) -> int:
    return sum(c.hits for c in count.values())


def _total(count: dict[str, RoleCount]) -> int:
    return sum(c.total for c in count.values())


def test_the_written_reading_claims_match_the_computed_counts(
    built: dict[str, registry.FigureOutput],
) -> None:
    teaser = built["teaser"].counts
    assert _all(teaser["utility"])
    assert _total(teaser["utility"]) == 10
    assert _all(teaser["morality"])
    assert _total(teaser["morality"]) == 10
    truth = teaser["truth"]
    assert truth["gpt2"].hits == 0
    assert truth["base"].hits == truth["base"].total
    assert truth["post"].hits == truth["post"].total
    assert "all 10" not in content.ABSTRACT
    assert "truth on every Qwen3 checkpoint but no GPT-2" in content.ABSTRACT
    assert "reads utility and morality above its random direction on every" in (
        _opener("reading")
    )
    assert "truth on every Qwen3 checkpoint but no GPT-2" in _opener("reading")


def test_the_written_finder_claims_match_the_computed_counts(
    built: dict[str, registry.FigureOutput],
) -> None:
    counts = built["finders"].counts
    utility = counts["pca_utility"]
    for role in ("base", "post"):
        assert utility[role].hits == utility[role].total
    qwen = ("base", "post")
    for name in ("pca_sentiment", "pca_truth"):
        assert sum(counts[name][r].hits for r in qwen) < sum(
            utility[r].hits for r in qwen
        )
    assert _all(counts["logistic_sentiment"])
    assert "on every Qwen3 checkpoint" in _findings("The finder")
    assert "all 10" not in _findings("The finder")
    assert "[[count:fig:finders.logistic_sentiment]]" in _findings("The finder")
    assert "above one for sentiment on every checkpoint" in _opener("finders")


def test_the_written_steering_and_wording_claims_match_the_counts(
    built: dict[str, registry.FigureOutput],
) -> None:
    steering = built["steering"].counts
    assert _hits(steering["moves01"]) * 2 > _total(steering["moves01"])
    assert _hits(steering["fluent01"]) * 2 > _total(steering["fluent01"])
    assert _none(steering["fluent03"])
    teaser = built["teaser"].counts["steering"]
    assert 0 < _hits(teaser) < _total(teaser)
    opener = _opener("steering")
    assert "on most checkpoints" in opener
    assert "meets the rule on none" in opener
    assert "on only some checkpoints" in opener
    noise = built["wording"].counts["span_above_noise"]
    assert _hits(noise) * 2 > _total(noise)
    assert "on every Qwen3 checkpoint and on two of the four GPT-2" in (
        _opener("reading")
    )


def test_the_written_baseline_claims_match_the_computed_counts(
    built: dict[str, registry.FigureOutput],
) -> None:
    teaser = built["teaser"].counts
    assert _none(teaser["truthfulqa"])
    assert _none(teaser["knowledge_confidence"])
    assert "not above a classifier on the words alone" in _opener("truth")
    assert "adds at most a small amount to the confidence of the model" in (
        _opener("known_unknown")
    )


def test_the_knowledge_steering_measures_are_in_appendix_g() -> None:
    from tomography.site.figs import knowledge_steering

    appendix = " ".join(_texts(_section("appendix_knowledge")))
    assert knowledge_steering.APPENDIX_DETAIL in appendix
    runin = dict(_section("known_unknown").runin)["Steering."]
    for gone in ("gap", "1/4", "absolute mean slopes", "95 percent"):
        assert gone not in runin
    assert "[[sec:appendix_knowledge]]" in runin
    assert "[[runs:fig:knowledge_steering.distinguishable]]" in runin
    assert "[[runs:fig:knowledge_steering.selective]]" in runin
    assert "[[runs:fig:knowledge_steering.known_ge_unknown]]" in runin


def test_the_knowledge_steering_counts_hold_per_role() -> None:
    steering = registry.build_figure("knowledge_steering", Manifest(ROOT))
    got = {
        name: {r: (c.hits, c.total) for r, c in counts.items()}
        for name, counts in steering.counts.items()
    }
    assert got["distinguishable"] == {
        "gpt2": (3, 3),
        "base": (3, 3),
        "post": (3, 6),
    }
    assert got["known_ge_unknown"] == {
        "gpt2": (2, 3),
        "base": (2, 3),
        "post": (0, 6),
    }
    text = _text()
    for gone in ("not selective", "false refusal", "4 of 10"):
        assert gone not in text


def test_the_reading_subsection_states_the_words_alone_baseline() -> None:
    text = " ".join(_texts(_section("reading")))
    for token in (
        "[[num:table:reading.word_sentiment]]",
        "[[num:table:reading.char_sentiment]]",
        "[[num:table:reading.word_utility]]",
        "[[num:table:reading.char_utility]]",
    ):
        assert token in text, token
    assert "The words alone read sentiment" in text
    assert "with an interval above chance" in text
    assert "do not separate from chance for morality, truth and honesty" in text
    table = registry.build_table("reading", Manifest(ROOT))
    for analyzer in ("word", "char"):
        for concept, above in (
            ("sentiment", 1.0),
            ("utility", 1.0),
            ("morality", 0.0),
            ("truth", 0.0),
            ("honesty", 0.0),
        ):
            assert table.numbers[f"{analyzer}_{concept}_above"] == above


POOLED_TOKENS = (
    "knowledge_reading.dir_inc_cover_all",
    "knowledge_reading.dir_inc_n",
    "knowledge_reading.dir_inc_hi_max]]",
    "knowledge_reading.dir_inc_min]]",
    "knowledge_reading.dir_inc_max]]",
    "knowledge_reading.probe_inc_min]]",
    "knowledge_reading.probe_inc_max]]",
    "knowledge_reading.above_pop",
    "reading.onset_sentiment_min",
    "reading.onset_sentiment_max",
    "approximate.position_min",
    "approximate.position_max",
    "approximate.band_min",
    "approximate.band_max",
)


def test_no_number_on_the_page_is_pooled_across_roles() -> None:
    text = _text()
    for token in POOLED_TOKENS:
        assert token not in text, token


def _roles_in_order(text: str, stem: str) -> bool:
    found = [text.find(f"{stem}_{role}_") for role in ("gpt2", "base", "post")]
    return all(i >= 0 for i in found) and found == sorted(found)


def test_the_range_numbers_are_quoted_per_role_in_order() -> None:
    reading = " ".join(_texts(_section("reading")))
    assert _roles_in_order(reading, "reading.onset_sentiment")
    appendix = " ".join(_texts(_section("appendix_approximate")))
    for stem in ("approximate.position", "approximate.band"):
        assert _roles_in_order(appendix, stem), stem
    knowledge = " ".join(_texts(_section("appendix_knowledge")))
    for stem in (
        "knowledge_reading.probe_inc_min",
        "knowledge_reading.dir_inc_min",
        "knowledge_reading.dir_inc_hi_max",
    ):
        found = [
            knowledge.find(f"{stem}_{r}") for r in ("gpt2", "base", "post")
        ]
        assert min(found) >= 0, stem
        assert found == sorted(found), stem
    run = " ".join(_texts(_section("known_unknown")))
    assert "[[num:fig:knowledge_reading.dir_inc_hi_max_gpt2]]" in run
    assert "[[count:fig:knowledge_reading.dir_cover]]" in run


def test_the_approximate_appendix_states_the_band_and_the_control_layer() -> (
    None
):
    text = " ".join(_texts(_section("appendix_approximate")))
    assert (
        "the range of the random control's position over its "
        "[[num:fig:approximate.n_band_sets]] training sets"
    ) in text
    assert "its own validation layer" in text
    assert "[[count:fig:approximate.control_other_layer]]" in text


def test_the_page_does_not_say_the_readings_lack_a_text_baseline() -> None:
    text = _text()
    assert "have no text-only baseline" not in text
    limits = " ".join(_texts(content.DISCUSSION))
    assert "model-free word and character n-gram classifiers" in limits
    assert "[[table:reading]]" in limits
    assert "persona labels" in limits
    assert "test-pair sampling only" in limits


def test_the_method_baselines_name_the_words_alone_rows() -> None:
    method = " ".join(t for s in _sections(content.METHOD) for t in _texts(s))
    assert "words alone" in method
    assert "[[table:reading]]" in method
    assert "sentiment, utility, morality" in method


def test_the_introduction_lists_the_reading_concepts_among_text_baselines() -> (
    None
):
    intro = " ".join(_texts(content.INTRODUCTION))
    assert "a classifier on the text (for the readings, TruthfulQA" in intro


def _result_texts() -> str:
    parts = [content.ABSTRACT, content.TLDR]
    parts += [t for _, t in content.KEY_FINDINGS.listed]
    for section in content.RESULTS.subsections:
        parts += _texts(section)
    parts += _texts(content.DISCUSSION)
    return " ".join(parts)


def test_no_result_says_all_ten_checkpoints() -> None:
    text = _result_texts()
    assert "all 10" not in text
    assert "none of the 10" not in text


def test_the_reading_counts_the_same_for_utility_and_morality() -> None:
    manifest = Manifest(ROOT)
    teaser = registry.build_figure("teaser", manifest).counts
    assert teaser["utility"] == teaser["morality"]
    finding = _findings("Where a concept")
    assert "[[count:fig:teaser.utility]]" in finding
    assert "[[count:fig:teaser.truth]]" in finding


def test_the_wording_noise_claim_is_counted_per_role() -> None:
    counts = registry.build_figure("wording", Manifest(ROOT)).counts
    got = {r: (c.hits, c.total) for r, c in counts["span_above_noise"].items()}
    assert got["gpt2"][0] < got["gpt2"][1]
    finding = _findings("Where a concept")
    assert "[[count:fig:wording.span_above_noise]]" in finding
    lead = _opener("reading")
    assert "noise of the test pairs on most" not in lead
    noise = counts["span_above_noise"]
    assert noise["base"].hits == noise["base"].total
    assert noise["gpt2"].hits == 2
    assert noise["gpt2"].total == 4
    assert (
        "on every Qwen3 checkpoint and on two of the four GPT-2 checkpoints"
    ) in lead


def test_the_finder_utility_claim_is_limited_to_qwen3() -> None:
    finders = registry.build_figure("finders", Manifest(ROOT)).counts
    assert finders["pca_utility"]["gpt2"].hits == 0
    assert finders["pca_utility"]["base"].hits == 3
    assert finders["pca_utility"]["post"].hits == 3
    assert "for utility on every Qwen3 checkpoint" in content.ABSTRACT
    assert "for utility on every Qwen3 checkpoint" in _opener("finders")


def test_the_knowledge_claim_is_limited_to_qwen3() -> None:
    counts = registry.build_figure("knowledge_reading", Manifest(ROOT)).counts
    above = counts["above_best"]
    assert above["gpt2"].hits == 0
    assert above["base"].hits > 0
    phrase = "on most Qwen3 checkpoints"
    assert phrase in content.ABSTRACT
    assert phrase in _opener("known_unknown")


def test_the_knowledge_denominators_are_explained() -> None:
    run = " ".join(_texts(_section("known_unknown")))
    assert "GPT-2 small has no knowledge-direction run" in run


def test_the_steering_lead_names_the_sentiment_run() -> None:
    lead = _opener("steering")
    assert "the steered sentiment text meets the rule on none" in lead
    steering = registry.build_figure("steering", Manifest(ROOT))
    assert all(c.hits == 0 for c in steering.counts["fluent03"].values())


def test_the_utility_steering_run_is_not_called_the_finder_comparison() -> None:
    method = " ".join(t for s in _sections(content.METHOD) for t in _texts(s))
    assert "the utility steering run steers at 40, 55" in method
    assert "finder comparison steers" not in method
    limits = " ".join(_texts(content.DISCUSSION))
    assert "(the utility steering run has three)" in limits
    assert "(the finder comparison has three)" not in limits
