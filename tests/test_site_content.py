import dataclasses
import json
from math import comb
from pathlib import Path
import re
import runpy

import pytest

from tomography.site import content
from tomography.site import params
from tomography.site import references
from tomography.site import registry
from tomography.site.figs import base_post
from tomography.site.figs import teaser
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
    r"(?<![\d.])\d\.\dB\b",
    r"(?<![\d.])\dB\b",
    r"GPT-2",
    r"Qwen3",
    r"SST-2",
    r"WikiText-103",
    r"distinct-1",
    r"MC1",
    r"coefficient 0\.[13](?!\d)",
    r"probability 0\.05(?= by chance)",
    r"above 0\.95(?!\d)",
    r"exceeds 0\.1(?!\d)",
    r"at most 1\.5(?!\d)",
    r"at least 0\.15(?!\d)",
    r"below 0\.5(?!\d)",
    r"\ball 10\b",
    r"(?<![\d.])95 percent(?!\d)",
    r"(?<![\d.])40, 55 and 70 percent",
)
BODY_TITLES = [
    "Introduction",
    "Related work",
    "Method",
    "Results",
    "Limitations",
]
METHOD_KEYS = ["pipeline", "controls_method", "steering_method", "intervals"]
RESULT_KEYS = [
    "reading",
    "finders",
    "truth",
    "known_unknown",
    "steering",
    "base_post",
]
APPENDIX_KEYS = [
    "appendix_assets",
    "appendix_method",
    "appendix_reading",
    "appendix_steering",
    "appendix_knowledge",
]
COUNTS = (
    "Three fractions count GPT-2, Qwen3 base and post-trained checkpoints, "
    "in that order."
)
REMOVED_NAMES = (
    "TLDR",
    "FRONT_ITEMS",
    "KEY_FINDINGS",
    "DISCUSSION",
    "REPRODUCIBILITY",
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
    for top in (*content.BODY, content.CODE, *content.APPENDICES)
    for s in _sections(top)
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
    texts = [content.ABSTRACT]
    for section in (content.INTRODUCTION, *content.RESULTS.subsections):
        texts += _texts(section)
    return texts


def _section(key: str) -> content.Section:
    (found,) = [s for s in ALL if s.key == key]
    return found


def _joined(key: str) -> str:
    return " ".join(_texts(_section(key)))


def _words(key: str) -> int:
    return len(_joined(key).split())


def _body_text() -> str:
    return " ".join(
        t for top in content.BODY for s in _sections(top) for t in _texts(s)
    )


def _contribution(head: str) -> str:
    for h, t in content.INTRODUCTION.listed:
        if h.startswith(head):
            return t
    raise AssertionError(head)


def test_the_title_and_abstract_are_the_planned_ones() -> None:
    assert content.TITLE == (
        "Reading and Steering Concepts with One Direction: Representation "
        "Engineering on GPT-2 and Qwen3"
    )
    assert content.ABSTRACT.startswith(
        "We test representation engineering "
        "[[cite:zou2025representationengineeringtopdownapproach]] on four "
        "GPT-2 sizes and on three Qwen3 sizes, base and post-trained"
    )


def test_the_body_is_five_sections_and_five_appendices_in_order() -> None:
    assert [s.title for s in content.BODY] == BODY_TITLES
    assert [s.key for s in content.METHOD.subsections] == METHOD_KEYS
    assert [s.key for s in content.RESULTS.subsections] == RESULT_KEYS
    assert [s.title for s in content.RESULTS.subsections] == [
        "What does one direction read?",
        "Does the finder of the paper find it?",
        "Does a truth reader beat the words alone?",
        "Does a knowledge direction add to the model's confidence?",
        "Does steering change the output without breaking it?",
        "What does post-training change?",
    ]
    assert [s.key for s in content.APPENDICES] == APPENDIX_KEYS
    assert [s.title for s in content.APPENDICES] == [
        "Assets and notation",
        "Method details",
        "Reading details",
        "Published steering methods",
        "Knowledge",
    ]


def test_the_summary_the_findings_list_and_the_long_closing_are_gone() -> None:
    for name in REMOVED_NAMES:
        assert not hasattr(content, name), name
    assert content.ABSTRACT_ITEMS == ("fig:teaser",)
    assert "[[fig:teaser]]" in " ".join(_texts(content.INTRODUCTION))


def test_the_closing_paragraph_is_unnumbered_and_links_the_documents() -> None:
    closing = content.CODE
    assert (closing.key, closing.title) == ("code", "Code and data")
    assert closing not in content.BODY
    (paragraph,) = closing.paragraphs
    assert len(paragraph.split()) <= 70
    for name in ("repository", "findings", "methodology", "results"):
        assert f"[[link:{name}|" in paragraph, name
    assert "the experiments that have no figure here" in paragraph
    assert content.LINKS == {
        "repository": content.REPO_URL,
        "findings": content.FINDINGS_URL,
        "methodology": content.METHODOLOGY_URL,
        "results": content.DATA_URL,
    }


def test_the_introduction_ends_with_four_contributions() -> None:
    listed = content.INTRODUCTION.listed
    assert len(listed) == 4
    for head, text in listed:
        assert head.count("[[sec:") >= 1, head
        assert len(text.split()) <= 45, head
        assert text.count("[[fig:") == 1, head
        assert text.count("[[tally:") == 1, head
    assert content.INTRODUCTION.paragraphs[-1] == COUNTS
    assert len(content.INTRODUCTION.paragraphs) == 4


def test_the_limitations_are_one_paragraph_and_four_run_ins() -> None:
    limits = content.LIMITATIONS
    assert limits.key == "limitations"
    assert [h for h, _ in limits.runin] == [
        "Scale and coverage.",
        "Test size.",
        "Judges and labels.",
        "Choices after results.",
    ]
    assert len(limits.paragraphs) == 1
    assert _words("limitations") <= 270


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
            elif kind in ("num", "tally", "label"):
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
            elif kind == "link":
                target, bar, shown = rest.partition("|")
                assert target in content.LINKS, token
                assert bar, token
                assert shown, token
            else:
                pytest.fail(f"unknown token {token}")


def test_every_item_is_placed_once_and_the_main_text_cites_the_appendix() -> (
    None
):
    placed = [i for s in ALL for i in s.items] + list(content.ABSTRACT_ITEMS)
    wanted = [f"fig:{k}" for k in registry.MAIN_FIGURES]
    wanted += [f"table:{k}" for k in registry.MAIN_TABLES]
    wanted += [f"fig:{k}" for k in registry.APPENDIX_FIGURES]
    wanted += [f"table:{k}" for k in registry.APPENDIX_TABLES]
    assert sorted(placed) == sorted(wanted)
    body = _body_text()
    for section in content.APPENDICES:
        text = " ".join(_texts(section))
        for item in section.items:
            assert f"[[{item}]]" in text, item
            assert f"[[{item}]]" in body, item


def test_the_appendix_items_sit_in_the_planned_order() -> None:
    assert [s.items for s in content.APPENDICES] == [
        ("table:assets", "table:notation"),
        ("table:design",),
        (
            "table:reading",
            "fig:a_reading_curves",
            "table:finder_baselines",
            "fig:wording",
            "fig:a_truth_cities",
        ),
        ("table:steering_baselines",),
        (
            "fig:knowledge_steering",
            "fig:a_knowledge_matrix",
            "table:detection_baselines",
            "fig:a_knowledge_geometry",
            "fig:a_knowledge_projections",
            "fig:approximate",
            "table:approximate",
        ),
    ]


def test_items_follow_the_registry_order_of_the_main_text() -> None:
    placed = list(content.ABSTRACT_ITEMS) + [
        i
        for top in (content.INTRODUCTION, content.METHOD, content.RESULTS)
        for s in _sections(top)
        for i in s.items
    ]
    figures = [i[4:] for i in placed if i.startswith("fig:")]
    tables = [i[6:] for i in placed if i.startswith("table:")]
    assert figures == list(registry.MAIN_FIGURES)
    assert tables == list(registry.MAIN_TABLES)
    order = [k for s in content.APPENDICES for k in s.items]
    assert [k[4:] for k in order if k.startswith("fig:")] == list(
        registry.APPENDIX_FIGURES
    )
    assert [k[6:] for k in order if k.startswith("table:")] == list(
        registry.APPENDIX_TABLES
    )


def test_each_appendix_section_opens_with_what_it_holds() -> None:
    for section in content.APPENDICES:
        assert section.paragraphs, section.key
        first = section.paragraphs[0].split(". ")[0]
        assert len(first.split()) <= 60, section.key
    assert _words("appendix_method") <= 520
    assert _words("appendix_knowledge") <= 200


def test_content_numbers_have_no_stray_digits() -> None:
    for text in _opening_texts():
        assert _digits_left(text) == [], text


def test_the_numbers_test_would_catch_a_pasted_value() -> None:
    assert _digits_left("it reads 0.847 on [[num:fig:a.b]]") != []
    assert _digits_left("GPT-2 at relative coefficient 0.1") == []
    assert _digits_left("it holds on 10 of 10 checkpoints") != []
    assert _digits_left("a margin of 0.5 on the reader") != []
    assert _digits_left("a margin of 0.55 below 0.5") != []
    assert _digits_left("all 10 checkpoints, below 0.5") == []


def test_no_dashes_backticks_or_underscores_in_prose() -> None:
    for text in _all_texts():
        bare = _bare(text)
        assert not any(d in bare for d in DASHES), text
        assert "`" not in bare, text
        assert "_" not in bare, text


def test_run_codes_stay_out_of_the_prose() -> None:
    for text in _all_texts():
        assert RUN_CODE.search(_bare(text)) is None, text


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
    for url in (content.REPO_TEXT, *content.LINKS.values()):
        assert url.startswith(("github.com/", "https://github.com/"))
        assert url.count(handle) == 1


def test_the_shared_terms_have_one_spelling() -> None:
    text = _text()
    for gone in (
        "known-against-unknown",
        "confidence of the model",
        "text control",
        "text-classifier control",
        "Discussion",
        "published reader",
        "answer probability",
        "Zou et al",
    ):
        assert gone not in text, gone
    assert "A knowledge direction, fitted on questions a model answers" in text
    assert "the leading principal component of paired differences" in (
        _joined("controls_method")
    )


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


def test_the_method_states_its_rules_once() -> None:
    method = _method_text()
    assert "minus that of the random direction of the same run" in method
    assert "not minus 0.5" in method
    assert "never pool the roles" in method
    assert "that see only the text" in method
    assert "stacks five output signals" in method
    assert "no correction for the number of comparisons" in method
    assert "Each cell is one run" in method
    assert "treats the two arms as independent" in method
    assert f"meets the fluency rule: {content.FLUENCY_RULE}" in method
    assert "not of the completions" in method
    assert "chosen after pilot runs (post hoc)" in method
    assert "chooses its layer on the 256 pairs it scores" in method
    assert "separate split of 214 pairs" in method
    assert "Controls use no direction fitted for the concept" in method
    assert _text().count("no direction fitted for the concept") == 1
    assert _text().count("each Qwen3 size as") == 1
    assert "baselines are published methods run on the same items" in method


def test_the_layer_exception_of_the_finder_comparison_is_in_appendix_b() -> (
    None
):
    appendix = _joined("appendix_method")
    assert "except in the finder comparison" in appendix
    assert "layer chosen on its own select split" in appendix
    assert "at the same layer, not over" not in _text()


def test_the_method_is_about_six_hundred_words() -> None:
    words = len(
        " ".join(
            t for s in _sections(content.METHOD) for t in _texts(s)
        ).split()
    )
    assert 420 <= words <= 610, words


def test_content_design_facts() -> None:
    method = {
        name: json.loads((run(name, model) / "config.json").read_text("utf-8"))
        for name, model in (
            ("r1", "gpt2"),
            ("c1", "gpt2"),
            ("m1", "gpt2"),
            ("p9d", "qwen3-1.7b-base"),
        )
    }
    assert method["r1"]["n_eval_pairs"] == 256
    assert method["r1"]["n_pairs"] == 64
    assert method["r1"]["n_seeds"] == 10
    assert params.REPORT_PAIRS["sentiment"] == 214
    assert method["c1"]["n_neutral"] == 200
    assert method["c1"]["n_steer_seeds"] == 3
    assert method["m1"]["n_steer_seeds"] == 3
    assert "random" in method["m1"]["methods"]
    assert method["p9d"]["n_random"] == 3
    vectors = method["m1"]["n_steer_seeds"]
    assert 1 / comb(2 * vectors, vectors) == 0.05
    steering = _joined("steering")
    assert "all three vectors beat all three random directions" in steering
    assert "(probability 0.05 by chance)" in steering
    assert "each of three random directions" in _joined("known_unknown")


def test_the_design_digits_of_the_method_and_its_appendix() -> None:
    facts = (
        r"chance is 0\.5(?!\d)",
        r"(?:above|minus) 0\.5(?!\d)",
        r"not 0\.5(?!\d)",
        r"A run fits 10 readers",
        r"readers, not 10\.",
        r"draw of 64 training pairs",
        r"the 256 pairs it scores",
        r"split of 214 pairs",
        r"with 3 steering vectors",
        rf"coefficients from {content.MINUS}0\.3 to 0\.3",
        rf"[+{content.MINUS}]0\.3(?!\d)",
        r"perplexity of 200 held-out",
        r"1\.96",
        r"\bp[12]\b",
        r"\bn[12]\b",
        r"\(1 ",
        r"on 2 degrees of freedom",
    )
    for top in (content.METHOD, _section("appendix_method")):
        for section in _sections(top):
            for text in _texts(section):
                assert _digits_left(text, (*GENERIC, *facts)) == [], text
    for key in ("limitations", "code", "appendix_knowledge"):
        for text in _texts(_section(key)):
            assert _digits_left(text) == [], text


def test_sections_are_frozen_dataclasses() -> None:
    assert dataclasses.is_dataclass(content.Section)
    with pytest.raises(dataclasses.FrozenInstanceError):
        content.INTRODUCTION.title = "x"  # type: ignore[misc]


def test_the_abstract_is_prose_with_one_number() -> None:
    tokens = TOKEN.findall(content.ABSTRACT)
    assert tokens == [
        "cite:zou2025representationengineeringtopdownapproach",
        "num:fig:base_post.sent_below",
    ]
    assert "all 10" not in content.ABSTRACT


def test_each_results_subsection_opens_with_its_answer() -> None:
    for sub in content.RESULTS.subsections:
        closes = sub.key == "reading"
        assert len(sub.paragraphs) == (2 if closes else 1), sub.key
        assert bool(sub.after) == (sub.key != "reading")
        opener = sub.paragraphs[0]
        assert opener.rstrip().endswith("."), sub.key
        assert ". " not in _bare(opener).replace("0. ", ""), sub.key
        assert len(sub.items) == 1, sub.key
        assert sub.items[0].startswith("fig:"), sub.key
        limit = 350 if sub.key == "known_unknown" else 160
        assert _words(sub.key) <= limit, (sub.key, _words(sub.key))
        if sub.key != "known_unknown":
            assert f"[[{sub.items[0]}]]" in opener, sub.key
    assert [h for h, _ in _section("known_unknown").runin] == [
        "Reading.",
        "Steering.",
    ]
    assert "[[fig:knowledge_reading]]" in _section("known_unknown").runin[0][1]


@pytest.fixture(scope="module")
def built() -> dict[str, registry.FigureOutput]:
    manifest = Manifest(ROOT)
    return {
        key: registry.build_figure(key, manifest)
        for key in (
            "teaser",
            "reading",
            "finders",
            "truth",
            "knowledge_reading",
            "steering",
            "base_post",
        )
    }


def _all(count: dict[str, RoleCount]) -> bool:
    return all(c.hits == c.total for c in count.values())


def _none(count: dict[str, RoleCount]) -> bool:
    return all(c.hits == 0 for c in count.values())


def _shape(count: dict[str, RoleCount]) -> list[tuple[int, int]]:
    return [(count[r].hits, count[r].total) for r in ("gpt2", "base", "post")]


def test_the_written_reading_claims_match_the_computed_counts(
    built: dict[str, registry.FigureOutput],
) -> None:
    counts = built["reading"].counts
    for concept in ("sentiment", "utility", "morality"):
        assert _all(counts[concept]), concept
        assert sum(c.total for c in counts[concept].values()) == 10
    for owner in (counts["truth"], built["teaser"].counts["truth"]):
        assert owner["gpt2"].hits == 0
        assert owner["base"].hits == owner["base"].total
        assert owner["post"].hits == owner["post"].total
    claim = (
        "a logistic direction reads sentiment, utility and morality above "
        "its random direction on every checkpoint"
    )
    assert (
        claim.replace("a logistic", "A logistic")
        + ", and truth on every Qwen3 but no GPT-2 checkpoint"
    ) in content.ABSTRACT
    assert claim in _contribution("Reading")
    opener = _section("reading").paragraphs[0]
    assert claim.replace("a logistic", "A logistic") in opener
    assert "and truth on every Qwen3 checkpoint" in opener
    after = _joined("reading")
    assert "No GPT-2 truth reader is separable from its random direction" in (
        after
    )
    numbers = built["reading"].numbers
    assert numbers["n_gpt2"] == 4
    assert "Across the four GPT-2 sizes" in after
    for concept in ("utility", "morality"):
        assert numbers[f"slope_{concept}_lo"] > 0, concept
    for concept in ("sentiment", "truth"):
        low, high = (numbers[f"slope_{concept}_{e}"] for e in ("lo", "hi"))
        assert low < 0 < high, concept
    assert "the sentiment and truth intervals include zero" in after
    for role in ("gpt2", "base", "post"):
        latest = numbers[f"onset_sentiment_{role}_max"]
        for concept in ("utility", "morality", "truth"):
            assert numbers[f"onset_{concept}_{role}_min"] > latest, (
                role,
                concept,
            )
    assert (
        "In every role the margin exceeds 0.1 at the lowest relative depth "
        "for sentiment"
    ) in (after)


def test_the_written_finder_claims_match_the_computed_counts(
    built: dict[str, registry.FigureOutput],
) -> None:
    counts = built["finders"].counts
    assert _shape(counts["pca_sentiment"]) == [(0, 4), (1, 3), (0, 3)]
    assert "for sentiment on one checkpoint" in content.ABSTRACT
    opener = _section("finders").paragraphs[0]
    assert "for sentiment on a single Qwen3 base checkpoint" in opener
    assert "on [[tally:fig:finders.pca_truth]] checkpoints" in opener
    utility = counts["pca_utility"]
    assert utility["gpt2"].hits == 0
    assert _all({r: utility[r] for r in ("base", "post")})
    assert _all({"gpt2": counts["pca_utility_point"]["gpt2"]})
    after = _joined("finders")
    assert (
        "For utility it is above on every Qwen3 checkpoint, and on GPT-2 by "
        "point estimate only"
    ) in after
    assert "for utility on every Qwen3 checkpoint" in _contribution(
        "The finder"
    )
    assert _contribution("The finder").startswith(
        "By interval, the leading principal component"
    )
    assert "Only for utility does a pair hold two versions of one" in after
    assert _all(counts["logistic_sentiment"])
    assert (
        "Logistic regression is above the random direction for sentiment on "
        "every checkpoint"
    ) in after
    assert after.count("matching is confounded") == 1
    assert _text().count("matching is confounded") == 1
    baselines = registry.build_table("finder_baselines", Manifest(ROOT))
    assert _all(baselines.counts["every_baseline_above"])
    assert (
        "on at least one dataset, by the same interval, on "
        "[[tally:table:finder_baselines.every_baseline_above]] checkpoints"
    ) in after
    assert "four published finders each beat it on some dataset" in (
        content.ABSTRACT
    )


def test_the_written_comparison_claims_match_the_computed_counts(
    built: dict[str, registry.FigureOutput],
) -> None:
    assert _none(built["truth"].counts["diff_excludes_zero"])
    assert _none(built["teaser"].counts["truthfulqa"])
    assert "No TruthfulQA reader beats the words alone" in content.ABSTRACT
    assert "No TruthfulQA reader beats the words alone" in _contribution(
        "Stronger"
    )
    opener = _section("truth").paragraphs[0]
    assert "A TruthfulQA reader does not beat the words alone" in opener
    assert "includes zero on every checkpoint" in opener
    above = built["knowledge_reading"].counts["above_best"]
    assert above["gpt2"].hits == 0
    qwen = [above[r] for r in ("base", "post")]
    assert sum(c.hits for c in qwen) * 2 > sum(c.total for c in qwen)
    assert not _all({"base": above["base"], "post": above["post"]})
    assert _none(built["teaser"].counts["knowledge_confidence"])
    assert (
        "beats popularity and the words alone on most Qwen3 checkpoints but "
        "not the model's confidence"
    ) in content.ABSTRACT
    known = _section("known_unknown").paragraphs[0]
    assert "on most Qwen3 checkpoints" in known
    increment = "adds no distinguishable increment to the model's confidence"
    for text in (known, _contribution("Stronger")):
        assert increment in text
    stronger = _contribution("Stronger")
    assert "No TruthfulQA reader beats the words alone by interval" in stronger
    assert (
        "the words alone, by interval, on "
        "[[tally:fig:knowledge_reading.above_best]]"
    ) in stronger
    assert (
        "On a second task, telling wrong from correct answers, its "
        "increment over the model's confidence"
        in (_section("known_unknown").runin[0][1])
    )
    assert "adds nothing" not in _text()


def test_the_written_steering_claims_match_the_computed_counts(
    built: dict[str, registry.FigureOutput],
) -> None:
    steering = built["steering"].counts
    moves = steering["moves01"]
    assert sum(c.hits for c in moves.values()) * 2 > sum(
        c.total for c in moves.values()
    )
    assert _none(steering["fluent03"])
    assert (
        "moves judged sentiment on most checkpoints and at coefficient 0.3 "
        "is never fluent"
    ) in content.ABSTRACT
    assert "meets the fluency rule on none" in _contribution("Steering")
    assert (
        "at relative coefficient 0.3 none does"
        in (_section("steering").paragraphs[0])
    )
    fluent = built["teaser"].counts["steering_fluent"]
    assert _shape(fluent) == [(0, 3), (1, 3), (0, 3)]
    beats = built["teaser"].counts["steering"]
    assert 0 < sum(c.hits for c in beats.values()) < 10
    assert "fluent on one" in content.ABSTRACT


def test_the_post_training_sizes_are_the_computed_ones(
    built: dict[str, registry.FigureOutput],
) -> None:
    numbers = built["base_post"].numbers
    sizes = {"06b": "0.6B", "17b": "1.7B", "4b": "4B"}
    readings = ("utility", "morality", "truth", "honesty", "truthfulqa")
    excluding = {
        (name, size)
        for name in readings
        for size in sizes
        if numbers[f"{name}_{size}_lo"] > 0 or numbers[f"{name}_{size}_hi"] < 0
    }
    assert excluding == {
        ("utility", "17b"),
        ("utility", "4b"),
        ("morality", "06b"),
        ("honesty", "06b"),
    }
    assert numbers["other_excluding"] == len(excluding)
    by_test_size = set()
    for name, size in excluding:
        _, lo, hi = base_post._difference(
            Manifest(ROOT), name, size.replace("06", "0.6").replace("17", "1.7")
        )
        if lo > 0 or hi < 0:
            by_test_size.add((name, size))
    assert by_test_size == {("utility", "4b")}
    text = _joined("base_post")
    assert ("only utility on 4B has a test-size interval") in text
    assert "the other three are seed-paired, which does not resample" in text
    assert "The one exception is [[fig:base_post]]" in _joined("intervals")
    assert "except the seed-paired intervals of [[fig:base_post]]" in (
        _joined("appendix_method")
    )
    assert "We study [[num:table:checkpoints.checkpoints]] checkpoints" in (
        _joined("pipeline")
    )
    assert (
        "only for utility on 1.7B ([[num:fig:base_post.utility_17b]]) and 4B "
        "([[num:fig:base_post.utility_4b]]), morality on 0.6B "
        "([[num:fig:base_post.morality_06b]]) and honesty transfer on 0.6B "
        "([[num:fig:base_post.honesty_06b]])"
    ) in text
    for name in ("knowledge", "steering"):
        below = [s for s in sizes if numbers[f"{name}_{s}_hi"] < 0]
        assert below == ["17b"], name
    assert "fall with an interval below zero on 1.7B only" in text
    falls = [-numbers[f"sentiment_{s}"] for s in sizes]
    assert numbers["sent_fall_min"] == pytest.approx(min(falls))
    assert numbers["sent_fall_max"] == pytest.approx(max(falls))
    assert (
        "with an interval below zero on [[num:fig:base_post.sent_below]]"
    ) in text
    assert "so each difference contrasts two recipes" in text


def test_the_test_size_limit_spans_the_computed_half_widths(
    built: dict[str, registry.FigureOutput],
) -> None:
    numbers = built["teaser"].numbers
    halves = [
        numbers[f"truth_half_{role}_{edge}"]
        for role in ("gpt2", "base", "post")
        for edge in ("min", "max")
    ]
    assert numbers["truth_half_base_min"] == min(halves)
    assert numbers["truth_half_gpt2_max"] == max(halves)
    limits = _joined("limitations")
    assert (
        "differences below about [[num:fig:teaser.truth_half_base_min]] to "
        "[[num:fig:teaser.truth_half_gpt2_max]] are not resolved"
    ) in limits
    assert "inconclusive, not negative" in limits


def test_the_checkpoints_named_in_the_text_are_the_computed_ones(
    built: dict[str, registry.FigureOutput],
) -> None:
    manifest = Manifest(ROOT)
    assert (
        registry.build_table("checkpoints", manifest).numbers["checkpoints"]
        == 10
    )
    assert "on ten small open checkpoints" in _joined("introduction")
    matrix = registry.build_figure("a_knowledge_matrix", manifest)
    assert matrix.claim.startswith("The knowledge reader of Qwen3-4B-Base ")
    assert "On Qwen3-4B-Base the words alone come within" in (
        _joined("known_unknown")
    )
    assert built["finders"].labels["random_max"] == "Qwen3-4B-Base"
    assert (
        "up to [[num:fig:finders.random_max]] on "
        "[[label:fig:finders.random_max]] in the finder comparison"
    ) in _joined("controls_method")


def test_counts_over_runs_say_runs() -> None:
    text = _joined("known_unknown")
    for name in ("distinguishable", "selective", "known_ge_unknown"):
        assert f"[[tally:fig:knowledge_steering.{name}]] runs" in text, name


def test_counts_are_three_fractions_defined_before_their_first_use() -> None:
    text = _text()
    assert text.lower().count("three fractions") == 1
    assert "[[tally:" not in content.ABSTRACT
    for paragraph in content.INTRODUCTION.paragraphs[:-1]:
        assert "[[tally:" not in paragraph
    assert "[[count:" not in text
    assert "[[runs:" not in text
    assert (
        "We count checkpoints per role (GPT-2, Qwen3 base, Qwen3 "
        "post-trained) and never pool the roles."
    ) in _joined("intervals")
    for address, unit in re.findall(r"\[\[tally:([^\]]+)\]\]( \w+)?", text):
        assert address.startswith(("fig:", "table:")), address
        assert unit in (" checkpoints", " runs", " but", ""), (address, unit)


def test_every_count_of_the_text_has_at_most_two_homes() -> None:
    tokens = re.findall(r"\[\[tally:([^\]]+)\]\]", _text())
    for address in set(tokens):
        assert tokens.count(address) <= 2, address


CAVEATS = (
    ("can score far above 0.5", "controls_method"),
    ("chosen after pilot runs (post hoc)", "pipeline"),
    ("No hypothesis was fixed before its run", "limitations"),
    ("widened from four to twelve relations", "limitations"),
    ("inconclusive, not negative", "limitations"),
    ("the chosen layer carries its own uncertainty", "limitations"),
    ("two versions of one scenario", "finders"),
    ("GPT-2 small has no direction", "known_unknown"),
    ("By point estimate on", "known_unknown"),
    ("known questions move at least as much", "known_unknown"),
    ("pictures without intervals, not tests", "known_unknown"),
    ("a point comparison", "reading"),
    ("no random-direction arm", "steering"),
    ("GPT-2 small has no fluency measure", "steering"),
    ("no human validated any judge", "limitations"),
    ("cannot tell a generic answer from an invented one", "limitations"),
    ("with no hand check", "limitations"),
    ("post-training data, chat format and distillation", "base_post"),
    ("The largest model is Qwen3-4B", "limitations"),
    ("Most concepts use one prompt format", "limitations"),
    ("on the 4B pair", "limitations"),
    ("does not separate a range from a living subject", "appendix_knowledge"),
)


@pytest.mark.parametrize(("phrase", "key"), CAVEATS)
def test_each_caveat_is_stated_once_in_its_place(phrase: str, key: str) -> None:
    assert phrase in _joined(key), (phrase, key)
    assert _text().count(phrase) == 1, phrase


def test_the_steering_appendix_states_its_outcome_and_links_the_record() -> (
    None
):
    lead = _joined("appendix_steering")
    assert (
        "Over the three concepts of each checkpoint, prompt prepending has "
        "the highest overall score on "
        "[[tally:table:steering_baselines.prompt_top]], and the interval of "
        "the reader lies below its own on "
        "[[tally:table:steering_baselines.reader_below]]"
    ) in lead
    assert "[[findings:published-steering-baselines-b3|findings document]]" in (
        lead
    )
    assert "five published steering methods as baselines and two controls" in (
        lead
    )
    findings = (ROOT.parent / "docs" / "findings.md").read_text("utf-8")
    assert "## Published steering baselines (B3)" in findings


def test_no_number_on_the_page_is_pooled_across_roles() -> None:
    text = _text()
    for token in (
        "knowledge_reading.dir_inc_hi_max]]",
        "knowledge_reading.dir_inc_min]]",
        "knowledge_reading.dir_inc_max]]",
        "knowledge_reading.probe_inc_min]]",
        "knowledge_reading.probe_inc_max]]",
        "reading.onset_sentiment_min",
        "approximate.position_min",
        "approximate.band_min",
    ):
        assert token not in text, token
    stems = ("dir_inc_min", "dir_inc_max", "probe_inc_min", "probe_inc_max")
    for stem in stems:
        found = re.findall(rf"knowledge_reading\.{stem}_(\w+)\]\]", text)
        assert found == ["gpt2", "base", "post"], stem
    assert re.findall(r"approximate\.range_(\w+)\]\]", text) == [
        "base_min",
        "base_max",
        "post_min",
        "post_max",
    ]


def test_the_knowledge_appendix_keeps_the_range_reader_limits() -> None:
    text = _joined("appendix_knowledge")
    assert "[[tally:table:approximate.layer_one]]" in text
    assert "[[tally:fig:approximate.control_below_half]]" in text
    assert "[[tally:fig:approximate.control_above_95]]" in text
    assert "already separates the test concepts" in text
    assert "[[tally:table:approximate.above_words]]" in _joined("known_unknown")
    assert "by interval" in _section("known_unknown").after[0]


def test_no_result_says_all_ten_checkpoints() -> None:
    text = " ".join(_joined(k) for k in RESULT_KEYS)
    assert "all 10" not in text
    assert "on 10 checkpoints" not in text


def test_the_abstract_states_the_thesis_that_the_scorecard_counts(
    built: dict[str, registry.FigureOutput],
) -> None:
    thesis = (
        "A direction beats its random direction far more often than the "
        "stronger comparison for its question."
    )
    assert thesis in content.ABSTRACT
    assert "on the words alone, by 95 percent intervals." in content.ABSTRACT
    assert "That paper's finder" in content.ABSTRACT
    counts = built["teaser"].counts

    def share(block: int) -> float:
        rows = [r.key for r in teaser.ROWS if r.block == block]
        hits = sum(c.hits for r in rows for c in counts[r].values())
        return hits / sum(c.total for r in rows for c in counts[r].values())

    assert share(0) > 2 * share(1)
    assert "or a published method" not in _joined("introduction")


def test_the_terms_of_the_results_are_defined_in_the_method() -> None:
    method = _method_text()
    for definition in (
        "utility (which scenario is more pleasant)",
        "some negated",
        "tested on plain statements (honesty transfer)",
        "answers correctly (known) against those it does not (unknown)",
        "within each relation such as capital",
        "each on its own draw of 64 training pairs",
        "A win rate is the share of decisive prompt pairs",
        "the popularity of the question's subject",
    ):
        assert definition in method, definition
    steering = _joined("steering")
    assert "(probability 0.05 by chance)" in steering
    assert "checkpoints ([[fig:teaser]], last row)" in steering
    assert "the next-token entropy" in _section("known_unknown").runin[1][1]
    assert "six runs" in _section("known_unknown").runin[1][1]
    approximate = _section("known_unknown").after[0]
    assert approximate.startswith(
        "A separate test, on approximate answers, asks whether a direction "
        "reads that an answer is a range."
    )


def test_the_example_relation_has_known_questions_on_every_checkpoint() -> None:
    (relation,) = re.findall(r"within each relation such as (\w+)\.", _text())
    for model in models("p9a"):
        path = run("p9a", model)
        guess = data(path, "table_guess.json")["rows"]
        assert [r[6] for r in guess if r[0] == f"popqa:{relation}"] == [True]
        rows = data(path, "table_popularity.json")["rows"]
        assert sum(r[4] for r in rows if r[0] == relation) > 0, model
