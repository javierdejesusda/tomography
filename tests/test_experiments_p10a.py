"""Tests for the P10a quantity behavior experiment."""

from collections import Counter
from collections.abc import Callable
from collections.abc import Sequence
import csv
import dataclasses
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tomography.cli import load_experiment
from tomography.control.generate import Completion
from tomography.datasets.quantities import FAMILIES
from tomography.datasets.quantities import PEOPLE_FAMILY
from tomography.datasets.quantities import QuantityQuestion
from tomography.datasets.sources import OfflineCacheMissError
from tomography.experiments import p10a_quantities
from tomography.experiments.base import RunDir
from tomography.experiments.p10a_quantities import FORCED_INSTRUCTION
from tomography.experiments.p10a_quantities import FORCED_SHOTS
from tomography.experiments.p10a_quantities import FREE_SHOTS
from tomography.experiments.p10a_quantities import UNIT_WORDS
from tomography.experiments.p10a_quantities import Answer
from tomography.experiments.p10a_quantities import Config
from tomography.experiments.p10a_quantities import Outcome
from tomography.experiments.p10a_quantities import Results
from tomography.experiments.p10a_quantities import bootstrap_gap_ci
from tomography.experiments.p10a_quantities import build_prompt
from tomography.experiments.p10a_quantities import build_table
from tomography.experiments.p10a_quantities import cluster_cis
from tomography.experiments.p10a_quantities import diagnostic_table
from tomography.experiments.p10a_quantities import execute
from tomography.experiments.p10a_quantities import forced_in_range_table
from tomography.experiments.p10a_quantities import hand_check_samples_table
from tomography.experiments.p10a_quantities import hand_check_table
from tomography.experiments.p10a_quantities import hedge_gap_table
from tomography.experiments.p10a_quantities import plot
from tomography.experiments.p10a_quantities import run
from tomography.experiments.p10a_quantities import score_answer
from tomography.experiments.p10a_quantities import shots_for
from tomography.experiments.p10a_quantities import width_table
from tomography.experiments.sweep import chat_format_for
from tomography.model import MODELS
from tomography.model import LoadedModel
from tomography.reading.metrics import wilson_ci

DEFAULT_HASH = "826d4cc1b3e6"


def test_config_defaults_and_round_trip() -> None:
    config = Config()

    assert config.experiment == "p10a"
    assert config.model == "gpt2"
    assert config.families is None
    assert config.n_per_side == 120
    assert config.min_sitelinks == 5
    assert config.conditions == ("free", "forced")
    assert config.formats == ("fewshot", "chat")
    assert config.n_samples == 5
    assert config.temperature == 1.0
    assert config.max_new_tokens == 32
    assert config.rel_tol == 0.05
    assert config.n_hand_check == 120
    assert config.batch_size == 8
    assert config.shot_orders == ("range_last", "exact_last")
    assert config.offline is False
    assert config.people is True
    assert config.seed == 0
    assert Config.from_dict(config.to_dict()) == config


def test_a_family_subset_round_trips() -> None:
    config = Config(families=("height", "duration"))

    assert Config.from_dict(config.to_dict()) == config


def test_the_default_config_hash_is_stable() -> None:
    assert Config().config_hash() == DEFAULT_HASH


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("families", ()),
        ("families", ("height", "height")),
        ("families", ("height", "no such family")),
        ("n_per_side", 0),
        ("n_samples", 0),
        ("temperature", 0.0),
        ("max_new_tokens", 0),
        ("rel_tol", -0.1),
        ("n_hand_check", -1),
        ("batch_size", 0),
        ("conditions", ()),
        ("conditions", ("free", "free")),
        ("conditions", ("free", "pressured")),
        ("formats", ()),
        ("formats", ("fewshot", "fewshot")),
        ("formats", ("zeroshot",)),
        ("shot_orders", ()),
        ("shot_orders", ("range_last", "range_last")),
        ("shot_orders", ("random",)),
    ],
)
def test_bad_settings_raise_before_any_work(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        Config(**{field: value})


def test_unit_words_cover_every_unit_the_loader_can_emit() -> None:
    units = {
        side.unit
        for family in FAMILIES
        for side in (family.range_side, family.exact_side)
    }

    assert units <= set(UNIT_WORDS)
    for unit in units:
        assert unit in UNIT_WORDS[unit]


@pytest.mark.parametrize(
    ("unit", "words"),
    [
        ("centimetres", {"cm", "centimeters", "centimeter", "centimetre"}),
        ("metres", {"m", "meters", "meter", "metre"}),
        ("days", {"day"}),
        ("minutes", {"min", "mins", "minute"}),
    ],
)
def test_unit_words_hold_spellings_and_abbreviations(
    unit: str, words: set[str]
) -> None:
    assert words <= set(UNIT_WORDS[unit])


def test_an_unknown_unit_fails_loudly() -> None:
    with pytest.raises(ValueError, match="furlongs"):
        p10a_quantities.unit_words("furlongs")


def test_the_free_and_forced_shots_share_their_questions() -> None:
    assert [q for q, _ in FREE_SHOTS] == [q for q, _ in FORCED_SHOTS]
    assert len(FREE_SHOTS) == 6


def test_free_shots_mix_exact_and_range_answers() -> None:
    answers = [a for _, a in FREE_SHOTS]

    assert sum(" to " in a for a in answers) == 3
    assert sum(" to " not in a for a in answers) == 3


def test_forced_shots_replace_each_range_by_one_number_inside_it() -> None:
    for (_, free), (_, forced) in zip(FREE_SHOTS, FORCED_SHOTS, strict=True):
        if " to " in free:
            low, high = (float(part) for part in free.split(" to "))
            assert low <= float(forced) <= high
        else:
            assert forced == free


def test_the_forced_range_replacements_are_72_8_and_7_4() -> None:
    replaced = [
        forced
        for (_, free), (_, forced) in zip(FREE_SHOTS, FORCED_SHOTS, strict=True)
        if " to " in free
    ]

    assert replaced == ["72", "8", "7.4"]


def test_the_free_fewshot_prompt_is_shots_then_the_question() -> None:
    prompt = build_prompt("How tall is X?", "free", "fewshot", None)

    expected = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in FREE_SHOTS)
    assert prompt == f"{expected}Q: How tall is X?\nA:"


def test_the_forced_fewshot_prompt_opens_with_the_instruction() -> None:
    prompt = build_prompt("How tall is X?", "forced", "fewshot", None)

    expected = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in FORCED_SHOTS)
    assert FORCED_INSTRUCTION == "Answer each question with one exact number."
    assert prompt == f"{FORCED_INSTRUCTION}\n\n{expected}Q: How tall is X?\nA:"


@pytest.mark.parametrize(
    ("condition", "instruction"),
    [
        ("free", "Answer the question briefly."),
        ("forced", "Answer with one exact number only."),
    ],
)
def test_the_chat_prompt_renders_the_instruction_and_question(
    tiny_qwen3: LoadedModel, condition: str, instruction: str
) -> None:
    post_trained = dataclasses.replace(
        tiny_qwen3, checkpoint=MODELS["qwen3-1.7b"]
    )
    chat = chat_format_for(post_trained)

    prompt = build_prompt("How tall is X?", condition, "chat", chat)

    assert (
        prompt
        == chat.render_turn(f"{instruction}\n\nQuestion: How tall is X?")[0]
    )
    assert instruction in prompt


def test_an_unknown_condition_or_format_raises() -> None:
    with pytest.raises(ValueError, match="condition"):
        build_prompt("Q?", "pressured", "fewshot", None)
    with pytest.raises(ValueError, match="format"):
        build_prompt("Q?", "free", "zeroshot", None)


def _item(**overrides: Any) -> QuantityQuestion:
    fields: dict[str, Any] = {
        "uid": "height-range-Q1",
        "family": "height",
        "kind": "range",
        "entity": "Beagle",
        "entity_id": "Q1",
        "question": "How tall is a Beagle, in centimetres?",
        "low": 33.0,
        "high": 41.0,
        "unit": "centimetres",
        "sitelinks": 40,
        "split": "train",
        "retrieved_on": "2026-10-17",
        "sha256": "ab" * 32,
    }
    return QuantityQuestion(**{**fields, **overrides})


def _score(text: str, fmt: str = "fewshot", **overrides: Any) -> Answer:
    return score_answer(_item(**overrides), text, fmt, rel_tol=0.05)


def test_a_fewshot_answer_is_cut_at_the_first_newline() -> None:
    text = " 35 cm\nQ: How many keys?\nA: 88"

    answer = _score(text)

    assert (answer.form, answer.low, answer.high) == ("point", 35.0, 35.0)
    assert answer.text == text


def test_a_fewshot_completion_that_starts_a_new_line_is_no_answer() -> None:
    answer = _score("\nQ: What year was 1969?\nA: 1969")

    assert answer.form == "none"
    assert answer.correct is None


def test_a_chat_answer_reads_its_first_non_empty_line() -> None:
    answer = _score("\n\n35 to 40 cm\nBeagles are small.", "chat")

    assert (answer.form, answer.low, answer.high) == ("range", 35.0, 40.0)


def test_the_entity_digits_are_not_the_answer() -> None:
    item = {
        "entity": "Boeing 737-800",
        "question": "What is the wingspan of the Boeing 737-800, in metres?",
        "unit": "metres",
        "low": 35.0,
        "high": 35.0,
        "kind": "exact",
    }

    answer = _score(" The Boeing 737-800 has a wingspan of 35.8.", **item)

    assert (answer.form, answer.low) == ("point", 35.8)
    assert answer.correct is True


def test_the_quantity_followed_by_the_unit_wins() -> None:
    answer = _score(" 3 dogs weigh 38 cm", unit="centimetres")

    assert answer.low == 38.0


@pytest.mark.parametrize(
    ("text", "form", "correct"),
    [
        (" 37", "point", True),
        (" 30", "point", False),
        (" 32", "point", True),
        (" 31", "point", False),
        (" 20 to 34", "range", True),
        (" 50 to 60", "range", False),
        (" I do not know", "abstain", None),
        (" tall", "none", None),
    ],
)
def test_the_answer_is_graded_against_the_sourced_range(
    text: str, form: str, correct: bool | None
) -> None:
    answer = _score(text)

    assert answer.form == form
    assert answer.correct is correct


def test_the_relative_tolerance_widens_the_range() -> None:
    item = _item()

    tight = score_answer(item, " 42", "fewshot", rel_tol=0.0)
    loose = score_answer(item, " 42", "fewshot", rel_tol=0.05)

    assert (tight.correct, loose.correct) == (False, True)


def test_an_approximate_marker_is_kept() -> None:
    assert _score(" about 37 cm").approximate is True
    assert _score(" 37 cm").approximate is False


@pytest.mark.parametrize(
    ("unit", "text", "flag"),
    [
        ("centimetres", " 37 cm", False),
        ("centimetres", " 37", False),
        ("centimetres", " 0.37 m", True),
        ("centimetres", " 14 inches", True),
        ("centimetres", " 37 centimeters", False),
        ("metres", " 35.8 m", False),
        ("metres", " 35.8m", False),
        ("metres", " 3580 cm", True),
        ("metres", " 117 feet", True),
        ("metres", " 0.0358 km", True),
        ("days", " 63 days", False),
        ("days", " 9 months", True),
        ("days", " 2 weeks", True),
        ("days", " 63", False),
        ("days", " 2 months or 63 days", False),
        ("minutes", " 142 min", False),
        ("minutes", " 142 mins", False),
        ("minutes", " 2 hours", True),
        ("minutes", " 2 hours 22 minutes", True),
        ("minutes", " 2 hours and 22 minutes", True),
        ("minutes", " two hours and 22 minutes", True),
        ("minutes", " an hour and 22 minutes", True),
        ("minutes", " half an hour and 22 minutes", True),
        ("minutes", " 142 minutes (2 hours 22 minutes)", False),
        ("minutes", " six months", False),
        ("minutes", " 142 hours, dozens of 5 minutes", True),
        ("minutes", " 22 minutes 2 hours", True),
        ("minutes", " 10 to 20 minutes", False),
        ("minutes", " ranked second, 142", False),
        ("minutes", " 142 minutes, a second longer", False),
        ("metres", " 12 m 30 cm", True),
        ("centimetres", " I'm not sure, maybe 60", False),
        ("centimetres", " it's 60 cm", False),
        ("minutes", " I do not know", False),
        ("minutes", " about a century", False),
    ],
)
def test_a_unit_mismatch_needs_a_number_and_a_foreign_unit(
    unit: str, text: str, flag: bool
) -> None:
    assert _score(text, unit=unit).unit_mismatch is flag


def test_an_entity_name_that_holds_a_unit_word_is_not_a_mismatch() -> None:
    film = {"entity": "Days of Heaven", "unit": "minutes"}

    assert _score(" Days of Heaven runs 94", **film).unit_mismatch is False
    assert _score(" Days of Heaven runs 94 hours", **film).unit_mismatch is True


def test_an_unknown_unit_fails_loudly_when_scoring() -> None:
    with pytest.raises(ValueError, match="furlongs"):
        _score(" 3", unit="furlongs")


def _answer(
    form: str = "point",
    value: float | None = 10.0,
    *,
    correct: bool | None = None,
    approximate: bool = False,
    mismatch: bool = False,
    truncated: bool = False,
) -> Answer:
    high = value if form == "point" else None if value is None else value + 5
    if form in ("abstain", "none"):
        value = high = None
    return Answer(
        text="",
        form=form,
        low=value,
        high=high,
        approximate=approximate,
        correct=correct,
        unit_mismatch=mismatch,
        finished=not truncated,
        truncated=truncated,
    )


def _outcome(greedy: Answer, *samples: Answer) -> Outcome:
    return Outcome(greedy=greedy, samples=samples or (greedy,))


def _pool(
    families: tuple[str, ...] = ("alpha", "beta"), per_side: int = 4
) -> list[QuantityQuestion]:
    items = []
    for family in families:
        for kind in ("range", "exact"):
            for i in range(per_side):
                low = 100.0 + 10 * i
                items.append(
                    _item(
                        uid=f"{family}-{kind}-Q{i}",
                        family=family,
                        kind=kind,
                        entity=f"{family}{kind}{i}",
                        low=low,
                        high=low + 20 * i + 10 if kind == "range" else low,
                        question=f"Q {family} {kind} {i}?",
                    )
                )
    return items


def _rows(columns: list[str], rows: list[list[Any]]) -> list[dict[str, Any]]:
    return [dict(zip(columns, row, strict=True)) for row in rows]


def _find(rows: list[dict[str, Any]], **keys: Any) -> dict[str, Any]:
    (row,) = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    return row


def test_the_table_counts_forms_per_cell_with_wilson_intervals() -> None:
    items = _pool(("alpha",))
    forms = ["point", "range", "range", "abstain", "none", "point", "point"]
    results: Results = {("free", "fewshot", "range_last"): []}
    for i, item in enumerate(items):
        if item.kind == "range":
            greedy = _answer(
                forms[i],
                correct=True if forms[i] in ("point", "range") else None,
            )
        else:
            greedy = _answer("point", correct=i % 2 == 0, mismatch=i == 4)
        results[("free", "fewshot", "range_last")].append(_outcome(greedy))

    columns, body = build_table(items, results, seed=0)
    rows = _rows(columns, body)

    assert columns[:10] == [
        "family",
        "kind",
        "condition",
        "format",
        "shot_order",
        "decoding",
        "interval",
        "item_wilson",
        "n",
        "n_gradable",
    ]
    row = _find(rows, family="alpha", kind="range", decoding="greedy")
    assert (row["n"], row["point"], row["range"]) == (4, 1, 2)
    assert (row["abstain"], row["none"]) == (1, 0)
    assert row["n_gradable"] == 3
    assert row["correct"] == 3
    assert row["point_rate"] == pytest.approx(0.25)
    assert (row["point_lo"], row["point_hi"]) == pytest.approx(wilson_ci(1, 4))
    assert (row["correct_lo"], row["correct_hi"]) == pytest.approx(
        wilson_ci(3, 3)
    )
    exact = _find(rows, family="alpha", kind="exact", decoding="greedy")
    assert exact["unit_mismatch"] == 1
    assert exact["correct"] == 2


def test_the_table_has_a_pooled_family_and_both_decodings() -> None:
    items = _pool()
    samples = (_answer("range"), _answer("point"), _answer("range"))
    results: Results = {
        ("forced", "chat", "none"): [
            _outcome(_answer("point", correct=True), *samples) for _ in items
        ]
    }

    columns, body = build_table(items, results, seed=0)
    rows = _rows(columns, body)

    families = {r["family"] for r in rows}
    assert families == {"alpha", "beta", "all"}
    pooled = _find(rows, family="all", kind="range", decoding="sampled")
    assert pooled["n"] == 8 * 3
    assert pooled["range"] == 16
    assert pooled["condition"] == "forced"
    assert pooled["format"] == "chat"
    greedy = _find(rows, family="all", kind="exact", decoding="greedy")
    assert greedy["n"] == 8
    assert len(rows) == 3 * 2 * 2


def test_the_table_counts_approximate_answers_of_any_form() -> None:
    items = _pool(("alpha",), per_side=2)
    results: Results = {
        ("free", "fewshot", "range_last"): [
            _outcome(_answer("range", approximate=True)),
            _outcome(_answer("none", approximate=True)),
            _outcome(_answer("point")),
            _outcome(_answer("point")),
        ]
    }

    columns, body = build_table(items, results, seed=0)
    row = _find(
        _rows(columns, body), family="alpha", kind="range", decoding="greedy"
    )

    assert row["approximate"] == 2
    assert row["approximate_rate"] == 1.0


def test_a_cell_without_gradable_answers_has_no_correct_rate() -> None:
    items = _pool(("alpha",), per_side=1)
    results: Results = {
        ("free", "fewshot", "range_last"): [
            _outcome(_answer("abstain")) for _ in items
        ]
    }

    columns, body = build_table(items, results, seed=0)
    row = _find(
        _rows(columns, body), family="alpha", kind="range", decoding="greedy"
    )

    assert row["n_gradable"] == 0
    assert row["correct"] == 0
    assert row["correct_rate"] is None
    assert row["correct_lo"] is None


def _hedge_results(
    items: list[QuantityQuestion], range_share: dict[str, list[float]]
) -> Results:
    """Greedy range answers where the item has share 1; 10 samples each."""
    outcomes = []
    seen = {"range": 0, "exact": 0}
    for item in items:
        share = range_share[item.kind][seen[item.kind] % 4]
        seen[item.kind] += 1
        samples = tuple(
            _answer("range" if k < round(10 * share) else "point")
            for k in range(10)
        )
        greedy = _answer("range" if share >= 0.5 else "point")
        outcomes.append(Outcome(greedy=greedy, samples=samples))
    return {("free", "fewshot", "range_last"): outcomes}


def test_the_hedge_gap_is_the_range_rate_on_range_minus_on_exact_items() -> (
    None
):
    items = _pool(("alpha",))
    results = _hedge_results(
        items,
        {"range": [1.0, 1.0, 0.0, 1.0], "exact": [0.0, 0.0, 0.0, 1.0]},
    )

    columns, body = hedge_gap_table(items, results, seed=0)
    rows = _rows(columns, body)

    greedy = _find(rows, family="alpha", decoding="greedy")
    assert greedy["range_rate_on_range_items"] == pytest.approx(0.75)
    assert greedy["range_rate_on_exact_items"] == pytest.approx(0.25)
    assert greedy["gap"] == pytest.approx(0.5)
    assert (greedy["n_range_items"], greedy["n_exact_items"]) == (4, 4)
    assert greedy["gap_lo"] < 0.5 < greedy["gap_hi"]
    sampled = _find(rows, family="alpha", decoding="sampled")
    assert sampled["gap"] == pytest.approx(0.5)


def test_the_hedge_gap_interval_is_degenerate_for_identical_items() -> None:
    items = _pool(("alpha",))
    results = _hedge_results(items, {"range": [1.0] * 4, "exact": [0.0] * 4})

    columns, body = hedge_gap_table(items, results, seed=0)
    row = _find(_rows(columns, body), family="alpha", decoding="greedy")

    assert (row["gap"], row["gap_lo"], row["gap_hi"]) == (1.0, 1.0, 1.0)


def test_the_hedge_gap_has_a_pooled_row_and_is_deterministic() -> None:
    items = _pool()
    results = _hedge_results(
        items, {"range": [1.0, 0.0, 1.0, 1.0], "exact": [0.0, 1.0, 0.0, 0.0]}
    )

    first = hedge_gap_table(items, results, seed=3)
    again = hedge_gap_table(items, results, seed=3)

    assert first == again
    rows = _rows(*first)
    pooled = _find(rows, family="all", decoding="greedy")
    assert pooled["n_range_items"] == pooled["n_exact_items"] == 8
    assert pooled["gap"] == pytest.approx(0.5)


def test_bootstrap_gap_ci_defaults_to_2000_resamples() -> None:
    default = inspect.signature(bootstrap_gap_ci).parameters["n_boot"].default

    assert default == 2000


def test_bootstrap_gap_ci_brackets_the_gap_and_depends_on_the_seed() -> None:
    range_rates = np.array([0.1, 0.35, 0.9, 0.42, 0.77, 0.2, 0.65])
    exact_rates = np.array([0.05, 0.15, 0.3, 0.12, 0.4, 0.22, 0.01])

    first = bootstrap_gap_ci(range_rates, exact_rates, seed=1)

    assert first == bootstrap_gap_ci(range_rates, exact_rates, seed=1)
    assert first != bootstrap_gap_ci(range_rates, exact_rates, seed=2)
    gap = range_rates.mean() - exact_rates.mean()
    assert first[0] < gap < first[1]


def test_the_hedge_gap_forwards_its_seed_to_every_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds: list[int] = []

    def spy(
        range_rates: np.ndarray, exact_rates: np.ndarray, *, seed: int
    ) -> tuple[float, float]:
        seeds.append(seed)
        return 0.0, 0.0

    monkeypatch.setattr(p10a_quantities, "bootstrap_gap_ci", spy)
    items = _pool()
    results = _hedge_results(items, {"range": [1.0] * 4, "exact": [0.0] * 4})

    hedge_gap_table(items, results, seed=7)

    assert seeds == [7] * 6


def test_forced_in_range_counts_greedy_points_inside_the_range() -> None:
    items = _pool(("alpha",))
    greedy = {
        "alpha-range-Q0": _answer("point", 105.0),
        "alpha-range-Q1": _answer("point", 125.0),
        "alpha-range-Q2": _answer("point", 400.0),
        "alpha-range-Q3": _answer("range", 120.0),
        "alpha-exact-Q0": _answer("point", 100.0),
        "alpha-exact-Q1": _answer("point", 109.0),
        "alpha-exact-Q2": _answer("point", 120.0),
        "alpha-exact-Q3": _answer("abstain"),
    }
    results: Results = {
        ("forced", "fewshot", "range_last"): [
            _outcome(greedy[i.uid]) for i in items
        ],
        ("free", "fewshot", "range_last"): [
            _outcome(_answer("point", 1.0)) for _ in items
        ],
    }

    columns, body = forced_in_range_table(items, results, rel_tol=0.05)
    rows = _rows(columns, body)

    assert {r["format"] for r in rows} == {"fewshot"}
    assert columns[:5] == ["family", "kind", "format", "shot_order", "n_items"]
    row = _find(rows, family="alpha", kind="range")
    assert (row["n_items"], row["n_point"]) == (4, 3)
    assert row["inside"] == 2
    assert row["inside_rate"] == pytest.approx(2 / 3)
    assert (row["inside_lo"], row["inside_hi"]) == pytest.approx(
        wilson_ci(2, 3)
    )
    assert row["inside_band"] == 2
    exact = _find(rows, family="alpha", kind="exact")
    assert (exact["n_point"], exact["inside"], exact["inside_band"]) == (
        3,
        2,
        3,
    )


def test_forced_in_range_widens_by_the_tolerance_on_both_ends() -> None:
    item = _item(low=100.0, high=200.0)
    values = [94.9, 95.1, 100.0, 200.0, 209.9, 210.1]
    results: Results = {
        ("forced", "chat", "none"): [
            _outcome(_answer("point", v)) for v in values
        ]
    }

    columns, body = forced_in_range_table(
        [dataclasses.replace(item, uid=f"u{k}") for k in range(6)],
        results,
        rel_tol=0.05,
    )
    row = _find(_rows(columns, body), family="all", kind="range")

    assert row["inside"] == 2
    assert row["inside_band"] == 4


def test_forced_in_range_is_empty_without_the_forced_condition() -> None:
    items = _pool(("alpha",))
    results: Results = {
        ("free", "fewshot", "range_last"): [
            _outcome(_answer("point")) for _ in items
        ]
    }

    columns, body = forced_in_range_table(items, results, rel_tol=0.05)

    assert body == []
    assert "inside_rate" in columns


def test_width_tertiles_split_range_items_by_relative_width() -> None:
    items = [
        _item(uid=f"u{k}", low=100.0, high=100.0 + 10 * (k + 1))
        for k in range(6)
    ]
    forms = ["range", "point", "range", "range", "point", "point"]
    results: Results = {
        ("free", "fewshot", "range_last"): [
            _outcome(_answer(f, correct=k % 2 == 0))
            for k, f in enumerate(forms)
        ]
    }

    columns, body = width_table(items, results, seed=0)
    rows = [
        r
        for r in _rows(columns, body)
        if r["decoding"] == "greedy" and r["family"] == "all"
    ]

    assert [r["tertile"] for r in rows] == [1, 2, 3]
    assert [r["n_items"] for r in rows] == [2, 2, 2]
    assert rows[0]["width_hi"] <= rows[1]["width_lo"]
    assert rows[1]["width_hi"] <= rows[2]["width_lo"]
    assert [r["range"] for r in rows] == [1, 2, 0]
    assert [r["correct"] for r in rows] == [1, 1, 1]
    assert rows[0]["range_rate"] == 0.5
    assert (rows[0]["range_lo"], rows[0]["range_hi"]) == pytest.approx(
        wilson_ci(1, 2)
    )


def test_width_covers_range_items_only_and_drops_empty_tertiles() -> None:
    items = [
        _item(uid="u0", low=100.0, high=120.0),
        _item(uid="u1", kind="exact", low=100.0, high=100.0),
    ]
    results: Results = {
        ("free", "chat", "none"): [_outcome(_answer("range")) for _ in items]
    }

    columns, body = width_table(items, results, seed=0)
    rows = _rows(columns, body)

    assert {r["family"] for r in rows} == {"height", "all"}
    assert {r["tertile"] for r in rows} == {1}
    assert {r["n_items"] for r in rows} == {1}
    assert {r["decoding"] for r in rows} == {"greedy", "sampled"}


Rule = Callable[[QuantityQuestion, bool, bool, int, str], str]


def _planted(
    item: QuantityQuestion, forced: bool, greedy: bool, k: int, order: str
) -> str:
    """Range items hedge when free; samples alternate with an abstention."""
    midpoint = (item.low + item.high) / 2
    if item.kind == "exact":
        return f" {item.low:g} cm"
    if forced:
        return f" {midpoint:g}"
    return (
        f" {item.low:g} to {item.high:g} cm" if k % 2 == 0 else " I do not know"
    )


def _order_of(prompt: str) -> str:
    """Tells the shot order of a prompt from the shot before the question."""
    if "<|im_start|>" in prompt:
        return "none"
    shots = prompt.rsplit("\n\nQ: ", 1)[0]
    last_is_range = shots.endswith(("A: 7.35 to 7.45", "A: 7.4"))
    return "range_last" if last_is_range else "exact_last"


class _Calls:
    """Replaces `generate` with planted completions and records the calls."""

    def __init__(
        self,
        items: list[QuantityQuestion],
        rule: Rule = _planted,
        finished: Callable[[bool], bool] = lambda greedy: True,
    ) -> None:
        self.items = items
        self.rule = rule
        self.finished = finished
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        loaded: LoadedModel,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        temperature: float | None = None,
        num_samples: int = 1,
        seed: int = 0,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> list[list[Completion]]:
        self.calls.append(
            {
                "prompts": list(prompts),
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "num_samples": num_samples,
                "seed": seed,
                "batch_size": batch_size,
            }
        )
        out = []
        for prompt in prompts:
            item = next(i for i in self.items if i.question in prompt)
            forced = "exact number" in prompt
            order = _order_of(prompt)
            out.append(
                [
                    Completion(
                        text=self.rule(
                            item, forced, temperature is None, k, order
                        ),
                        token_ids=(),
                        finished=self.finished(temperature is None),
                    )
                    for k in range(num_samples)
                ]
            )
        return out


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        n_samples=4,
        n_hand_check=8,
        batch_size=5,
        shot_orders=("range_last",),
    )
    return dataclasses.replace(base, **overrides)


def _executed(
    loaded: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    items: list[QuantityQuestion] | None = None,
    rule: Rule = _planted,
    finished: Callable[[bool], bool] = lambda greedy: True,
    **overrides: Any,
) -> tuple[RunDir, _Calls]:
    items = items if items is not None else _pool(per_side=6)
    calls = _Calls(items, rule, finished)
    monkeypatch.setattr(p10a_quantities, "generate", calls)
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, loaded, items)
    return run_dir, calls


def _post_trained(loaded: LoadedModel) -> LoadedModel:
    return dataclasses.replace(loaded, checkpoint=MODELS["qwen3-1.7b"])


def _csv(run_dir: RunDir, name: str) -> list[dict[str, str]]:
    path = run_dir.path / name
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_execute_writes_the_files_of_a_run(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(tiny_gpt2, tmp_path, monkeypatch)

    for name in (
        "table.csv",
        "hedge_gap.csv",
        "forced_in_range.csv",
        "width.csv",
        "hand_check.csv",
        "items.json",
        "summary.json",
    ):
        assert (run_dir.path / name).is_file(), name


def test_the_table_holds_the_planted_rates(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(tiny_gpt2, tmp_path, monkeypatch)
    rows = _csv(run_dir, "table.csv")

    def row(kind: str, condition: str, decoding: str) -> dict[str, str]:
        (found,) = [
            r
            for r in rows
            if (r["family"], r["kind"], r["condition"], r["format"])
            == ("all", kind, condition, "fewshot")
            and r["decoding"] == decoding
        ]
        return found

    free = row("range", "free", "greedy")
    assert (free["n"], free["range"], free["correct"]) == ("12", "12", "12")
    sampled = row("range", "free", "sampled")
    assert (sampled["n"], sampled["range"], sampled["abstain"]) == (
        "48",
        "24",
        "24",
    )
    assert row("exact", "free", "greedy")["point"] == "12"
    forced = row("range", "forced", "greedy")
    assert (forced["point"], forced["range"]) == ("12", "0")
    assert float(free["range_rate"]) == 1.0
    assert float(sampled["range_rate"]) == 0.5


def test_the_hedge_gap_and_forced_files_follow_the_planted_answers(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(tiny_gpt2, tmp_path, monkeypatch)

    gaps = {
        (r["condition"], r["decoding"]): float(r["gap"])
        for r in _csv(run_dir, "hedge_gap.csv")
        if r["family"] == "all" and r["format"] == "fewshot"
    }
    assert gaps == {
        ("free", "greedy"): 1.0,
        ("free", "sampled"): 0.5,
        ("forced", "greedy"): 0.0,
        ("forced", "sampled"): 0.0,
    }
    forced = [
        r
        for r in _csv(run_dir, "forced_in_range.csv")
        if r["family"] == "all" and r["kind"] == "range"
    ]
    assert [(r["n_point"], r["inside"]) for r in forced] == [("12", "12")]
    assert len(_csv(run_dir, "width.csv")) == 3 * 3 * 2 * 2


def test_generation_runs_greedy_then_sampled_per_cell(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, calls = _executed(
        tiny_gpt2, tmp_path, monkeypatch, temperature=0.7, max_new_tokens=6
    )

    greedy, sampled, *rest = calls.calls
    assert len(rest) == 2
    assert (greedy["temperature"], greedy["num_samples"]) == (None, 1)
    assert (sampled["temperature"], sampled["num_samples"]) == (0.7, 4)
    for call in calls.calls:
        assert call["max_new_tokens"] == 6
        assert call["batch_size"] == 5
        assert len(call["prompts"]) == 24
    assert len({c["seed"] for c in calls.calls[1::2]}) == 2


def test_the_seed_of_the_samples_follows_the_config_seed(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, first = _executed(tiny_gpt2, tmp_path / "a", monkeypatch, seed=1)
    _, again = _executed(tiny_gpt2, tmp_path / "b", monkeypatch, seed=1)
    _, other = _executed(tiny_gpt2, tmp_path / "c", monkeypatch, seed=2)

    def seeds(calls: _Calls) -> list[int]:
        return [c["seed"] for c in calls.calls[1::2]]

    assert seeds(first) == seeds(again)
    assert seeds(first) != seeds(other)


def test_a_base_model_skips_the_chat_format(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, calls = _executed(tiny_qwen3, tmp_path, monkeypatch)

    assert {r["format"] for r in _csv(run_dir, "table.csv")} == {"fewshot"}
    assert len(calls.calls) == 4
    assert _summary(run_dir)["formats"] == ["fewshot"]


def test_a_post_trained_model_runs_both_formats(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, calls = _executed(_post_trained(tiny_qwen3), tmp_path, monkeypatch)

    assert {r["format"] for r in _csv(run_dir, "table.csv")} == {
        "fewshot",
        "chat",
    }
    assert len(calls.calls) == 8
    chat_prompts = [p for c in calls.calls for p in c["prompts"] if "<|im" in p]
    assert chat_prompts
    assert all("Question: " in p for p in chat_prompts)
    assert _summary(run_dir)["formats"] == ["fewshot", "chat"]


def test_chat_only_on_a_base_model_raises(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="no format"):
        _executed(tiny_qwen3, tmp_path, monkeypatch, formats=("chat",))


def test_the_conditions_setting_picks_the_conditions(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, calls = _executed(
        tiny_gpt2, tmp_path, monkeypatch, conditions=("forced",)
    )

    assert {r["condition"] for r in _csv(run_dir, "table.csv")} == {"forced"}
    assert len(calls.calls) == 2
    assert _csv(run_dir, "hedge_gap.csv")


def test_an_unknown_unit_fails_before_any_generation(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = [*_pool(per_side=2), _item(uid="x", unit="furlongs")]
    calls = _Calls(items)
    monkeypatch.setattr(p10a_quantities, "generate", calls)
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match="furlongs"):
        execute(config, run_dir, tiny_gpt2, items)

    assert calls.calls == []


def test_no_questions_raises(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="no questions"):
        _executed(tiny_gpt2, tmp_path, monkeypatch, items=[])


def test_items_json_keeps_every_parsed_answer(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(tiny_gpt2, tmp_path, monkeypatch)
    records = run_dir.read_json("items")

    assert isinstance(records, list)
    assert [r["uid"] for r in records] == [i.uid for i in _pool(per_side=6)]
    record = next(r for r in records if r["uid"] == "alpha-range-Q1")
    assert record["entity_id"] == "Q1"
    assert record["sha256"] == "ab" * 32
    cell = record["answers"]["free/fewshot/range_last"]
    assert cell["greedy"]["form"] == "range"
    assert cell["greedy"]["correct"] is True
    assert cell["greedy"]["unit_mismatch"] is False
    assert [a["form"] for a in cell["samples"]] == [
        "range",
        "abstain",
        "range",
        "abstain",
    ]
    assert set(record["answers"]) == {
        "free/fewshot/range_last",
        "forced/fewshot/range_last",
    }


def test_the_hand_check_is_stratified_by_family_and_kind(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(tiny_gpt2, tmp_path, monkeypatch, n_hand_check=8)
    rows = _csv(run_dir, "hand_check.csv")

    strata: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["family"], row["kind"])
        strata[key] = strata.get(key, 0) + 1
    assert strata == {
        ("alpha", "range"): 2,
        ("alpha", "exact"): 2,
        ("beta", "range"): 2,
        ("beta", "exact"): 2,
    }
    assert _summary(run_dir)["hand_check_rows"] == 8


def test_the_hand_check_rows_are_traceable_and_blank_for_the_human(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(tiny_gpt2, tmp_path, monkeypatch, n_hand_check=100)
    rows = _csv(run_dir, "hand_check.csv")
    row = next(r for r in rows if r["uid"] == "alpha-range-Q3")

    assert len(rows) == 24
    assert row["wikidata_url"] == "https://www.wikidata.org/wiki/Q1"
    assert row["question"] == "Q alpha range 3?"
    assert (row["gold_low"], row["gold_high"]) == ("130.0", "200.0")
    assert row["unit"] == "centimetres"
    assert row["free_fewshot_range_last_completion"] == " 130 to 200 cm"
    assert row["free_fewshot_range_last_form"] == "range"
    assert row["free_fewshot_range_last_value"] == "130 to 200"
    assert row["free_fewshot_range_last_grade"] == "correct"
    assert row["forced_fewshot_range_last_value"] == "165"
    for column in ("human_form", "human_correct", "human_unit_mismatch"):
        assert row[column] == ""


def test_the_hand_check_draw_follows_the_seed(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def uids(seed: int, out: str) -> list[str]:
        run_dir, _ = _executed(
            tiny_gpt2, tmp_path / out, monkeypatch, n_hand_check=8, seed=seed
        )
        return [r["uid"] for r in _csv(run_dir, "hand_check.csv")]

    first = uids(1, "a")

    assert first == uids(1, "b")
    assert first != uids(2, "c")


def test_a_hand_check_completion_keeps_its_newlines_visible(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def rule(
        item: QuantityQuestion, forced: bool, greedy: bool, k: int, order: str
    ) -> str:
        return " 12\nQ: next?"

    run_dir, _ = _executed(
        tiny_gpt2, tmp_path, monkeypatch, rule=rule, n_hand_check=100
    )

    row = _csv(run_dir, "hand_check.csv")[0]
    assert row["free_fewshot_range_last_completion"] == " 12\\nQ: next?"
    assert row["free_fewshot_range_last_form"] == "point"


def test_the_summary_records_counts_snapshots_and_the_config(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(tiny_gpt2, tmp_path, monkeypatch)
    summary = _summary(run_dir)

    assert summary["model"] == "tiny"
    assert summary["n_items"] == 24
    assert summary["families"] == ["alpha", "beta"]
    assert summary["counts"] == {
        "alpha": {"range": 6, "exact": 6},
        "beta": {"range": 6, "exact": 6},
    }
    assert summary["snapshots"]["alpha"]["range"] == {
        "retrieved_on": ["2026-10-17"],
        "sha256": ["ab" * 32],
    }
    assert summary["conditions"] == ["free", "forced"]
    assert summary["config"] == _config().to_dict()


def _summary(run_dir: RunDir) -> dict[str, Any]:
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    return summary


def test_run_loads_the_questions_then_the_model(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: dict[str, Any] = {}
    order: list[str] = []
    items = _pool(per_side=2)

    def fake_load(**kwargs: Any) -> list[QuantityQuestion]:
        order.append("questions")
        asked.update(kwargs)
        return items

    def fake_model(key: str) -> LoadedModel:
        order.append(f"model {key}")
        return tiny_gpt2

    monkeypatch.setattr(p10a_quantities, "load_quantities", fake_load)
    monkeypatch.setattr(p10a_quantities, "load_model", fake_model)
    monkeypatch.setattr(p10a_quantities, "generate", _Calls(items))
    config = _config(
        families=("wingspan", "height"),
        n_per_side=7,
        min_sitelinks=9,
        offline=True,
        seed=4,
        people=False,
    )
    run_dir = RunDir.create(config, out_root=tmp_path)

    run(config, run_dir)

    assert order == ["questions", "model tiny"]
    assert [f.name for f in asked["families"]] == ["height", "wingspan"]
    assert {k: v for k, v in asked.items() if k != "families"} == {
        "n_per_side": 7,
        "min_sitelinks": 9,
        "seed": 4,
        "offline": True,
    }
    assert (run_dir.path / "summary.json").is_file()


def test_run_loads_every_family_by_default(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: dict[str, Any] = {}
    items = _pool(per_side=2)

    def fake_load(**kwargs: Any) -> list[QuantityQuestion]:
        asked.update(kwargs)
        return items

    monkeypatch.setattr(p10a_quantities, "load_quantities", fake_load)
    monkeypatch.setattr(p10a_quantities, "load_model", lambda key: tiny_gpt2)
    monkeypatch.setattr(p10a_quantities, "generate", _Calls(items))
    config = _config(people=False)

    run(config, RunDir.create(config, out_root=tmp_path))

    assert [f.name for f in asked["families"]] == [f.name for f in FAMILIES]


def test_an_offline_cache_miss_stops_before_the_model_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_model(key: str) -> LoadedModel:
        raise AssertionError("the model must not load on a cache miss")

    monkeypatch.setenv("TOMOGRAPHY_CACHE", str(tmp_path / "empty"))
    monkeypatch.setattr(p10a_quantities, "load_model", no_model)
    config = _config(offline=True)

    with pytest.raises(OfflineCacheMissError):
        run(config, RunDir.create(config, out_root=tmp_path / "runs"))


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("p10a")

    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_plot_draws_the_form_shares(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(tiny_gpt2, tmp_path, monkeypatch)

    paths = plot(run_dir, tmp_path / "figs")

    assert sorted(p.name for p in paths) == [
        "form_shares.pdf",
        "form_shares.png",
        "form_shares.svg",
    ]
    assert all(p.stat().st_size > 0 for p in paths)


def test_plot_falls_back_to_the_chat_format_without_fewshot(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(
        _post_trained(tiny_qwen3), tmp_path, monkeypatch, formats=("chat",)
    )

    assert len(plot(run_dir, tmp_path / "figs")) == 3


def test_a_real_tiny_run_writes_every_file(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    short = (("Keys?", "88"),)
    monkeypatch.setattr(p10a_quantities, "FREE_SHOTS", short)
    monkeypatch.setattr(p10a_quantities, "FORCED_SHOTS", short)
    config = _config(max_new_tokens=3, n_samples=2)
    run_dir = RunDir.create(config, out_root=tmp_path)

    execute(config, run_dir, tiny_arch, _pool(per_side=3))

    rows = _csv(run_dir, "table.csv")
    assert {r["decoding"] for r in rows} == {"greedy", "sampled"}
    assert len(_csv(run_dir, "hand_check.csv")) == 8
    assert len(plot(run_dir, tmp_path / "figs")) == 3


@pytest.mark.slow
def test_p10a_on_gpt2_reads_a_small_cached_pool(tmp_path: Path) -> None:
    config = Config(n_per_side=6, n_samples=2, n_hand_check=12, offline=True)
    run_dir = RunDir.create(config, out_root=tmp_path)
    try:
        run(config, run_dir)
    except OfflineCacheMissError as error:
        pytest.skip(str(error))

    assert (run_dir.path / "table.csv").is_file()


def test_the_last_range_last_shot_is_the_blood_ph_range() -> None:
    question, answer = FREE_SHOTS[-1]

    assert question == "What is the normal pH of human blood?"
    assert answer == "7.35 to 7.45"
    assert FORCED_SHOTS[-1] == (question, "7.4")
    assert not any("cat" in q for q, _ in FREE_SHOTS)


@pytest.mark.parametrize("condition", ["free", "forced"])
def test_exact_last_reverses_the_shots_and_keeps_the_alternation(
    condition: str,
) -> None:
    base = shots_for(condition, "range_last")
    reverse = shots_for(condition, "exact_last")

    assert reverse == tuple(reversed(base))
    assert base[-1][1] != reverse[-1][1]
    if condition == "free":
        forms = ["range" if " to " in a else "point" for _, a in reverse]
        assert forms == ["range", "point"] * 3


def test_range_last_is_the_alternating_list_of_the_constants() -> None:
    assert shots_for("free", "range_last") == FREE_SHOTS
    assert shots_for("forced", "range_last") == FORCED_SHOTS


def test_an_unknown_shot_order_or_condition_raises() -> None:
    with pytest.raises(ValueError, match="shot order"):
        shots_for("free", "random")
    with pytest.raises(ValueError, match="condition"):
        shots_for("pressured", "range_last")


@pytest.mark.parametrize("condition", ["free", "forced"])
def test_the_fewshot_prompt_follows_the_shot_order(condition: str) -> None:
    prompt = build_prompt(
        "How tall is X?", condition, "fewshot", None, order="exact_last"
    )

    last_question, last_answer = shots_for(condition, "exact_last")[-1]
    tail = f"Q: {last_question}\nA: {last_answer}\n\nQ: How tall is X?\nA:"
    assert prompt.endswith(tail)
    assert prompt != build_prompt("How tall is X?", condition, "fewshot", None)


def test_the_chat_prompt_ignores_the_shot_order(
    tiny_qwen3: LoadedModel,
) -> None:
    chat = chat_format_for(
        dataclasses.replace(tiny_qwen3, checkpoint=MODELS["qwen3-1.7b"])
    )

    assert build_prompt(
        "Q?", "free", "chat", chat, order="exact_last"
    ) == build_prompt("Q?", "free", "chat", chat, order="range_last")


def test_the_fewshot_prompt_rejects_an_unknown_order() -> None:
    with pytest.raises(ValueError, match="shot order"):
        build_prompt("Q?", "free", "fewshot", None, order="random")


ORDERS = ("range_last", "exact_last")


def _order_rule(
    item: QuantityQuestion, forced: bool, greedy: bool, k: int, order: str
) -> str:
    """Range items hedge only after a range last shot; exact items never."""
    if item.kind == "exact" or order == "exact_last":
        return f" {item.low:g} cm"
    return f" {item.low:g} to {item.high:g} cm"


def _both_orders(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **overrides: Any,
) -> tuple[RunDir, _Calls]:
    return _executed(
        tiny_gpt2,
        tmp_path,
        monkeypatch,
        rule=_order_rule,
        shot_orders=ORDERS,
        **overrides,
    )


def test_every_table_carries_the_shot_order(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _both_orders(tiny_gpt2, tmp_path, monkeypatch)

    for name in ("table", "hedge_gap", "forced_in_range", "width"):
        rows = _csv(run_dir, f"{name}.csv")
        assert {r["shot_order"] for r in rows} >= set(ORDERS), name


def test_the_table_splits_the_cells_by_shot_order(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, calls = _both_orders(tiny_gpt2, tmp_path, monkeypatch)
    rows = [
        r
        for r in _csv(run_dir, "table.csv")
        if (r["family"], r["kind"], r["condition"], r["decoding"])
        == ("all", "range", "free", "greedy")
    ]

    by_order = {r["shot_order"]: r for r in rows}
    assert set(by_order) == set(ORDERS)
    assert by_order["range_last"]["range"] == "12"
    assert by_order["exact_last"]["range"] == "0"
    assert by_order["exact_last"]["point"] == "12"
    assert len(calls.calls) == 8
    assert len({c["seed"] for c in calls.calls[1::2]}) == 4


def test_the_hedge_gap_has_a_both_row_that_averages_the_orders(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _both_orders(tiny_gpt2, tmp_path, monkeypatch)
    rows = [
        r
        for r in _csv(run_dir, "hedge_gap.csv")
        if (r["family"], r["condition"], r["decoding"])
        == ("alpha", "free", "greedy")
    ]

    gaps = {r["shot_order"]: r for r in rows}
    assert set(gaps) == {"range_last", "exact_last", "both"}
    assert float(gaps["range_last"]["gap"]) == 1.0
    assert float(gaps["exact_last"]["gap"]) == 0.0
    both = gaps["both"]
    assert float(both["gap"]) == pytest.approx(0.5)
    assert float(both["range_rate_on_range_items"]) == pytest.approx(0.5)
    assert float(both["range_rate_on_exact_items"]) == 0.0
    assert float(both["gap_lo"]) == float(both["gap_hi"]) == 0.5


def test_a_both_row_covers_each_family_condition_and_decoding(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _both_orders(tiny_gpt2, tmp_path, monkeypatch)
    both = [
        r for r in _csv(run_dir, "hedge_gap.csv") if r["shot_order"] == "both"
    ]

    assert len(both) == 3 * 2 * 2
    assert {r["format"] for r in both} == {"fewshot"}


def test_one_order_gives_no_both_row(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(tiny_gpt2, tmp_path, monkeypatch)

    orders = {r["shot_order"] for r in _csv(run_dir, "hedge_gap.csv")}
    assert orders == {"range_last"}


def test_the_both_row_resamples_items_with_the_config_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds: list[int] = []

    def spy(
        range_rates: np.ndarray, exact_rates: np.ndarray, *, seed: int
    ) -> tuple[float, float]:
        seeds.append(seed)
        return 0.0, 0.0

    monkeypatch.setattr(p10a_quantities, "bootstrap_gap_ci", spy)
    items = _pool(("alpha",))
    cells = {
        ("free", "fewshot", order): [_outcome(_answer("range")) for _ in items]
        for order in ORDERS
    }

    columns, body = hedge_gap_table(items, cells, seed=5)

    assert seeds == [5] * 12
    assert {r["shot_order"] for r in _rows(columns, body)} == {
        "range_last",
        "exact_last",
        "both",
    }


def test_the_chat_cells_have_no_shot_order(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, calls = _executed(
        _post_trained(tiny_qwen3),
        tmp_path,
        monkeypatch,
        shot_orders=ORDERS,
    )

    rows = _csv(run_dir, "table.csv")
    chat = {r["shot_order"] for r in rows if r["format"] == "chat"}
    few = {r["shot_order"] for r in rows if r["format"] == "fewshot"}
    assert (chat, few) == ({"none"}, set(ORDERS))
    assert len(calls.calls) == 12
    both = [
        r for r in _csv(run_dir, "hedge_gap.csv") if r["shot_order"] == "both"
    ]
    assert {r["format"] for r in both} == {"fewshot"}


def test_the_hand_check_has_a_column_set_per_order(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _both_orders(
        tiny_gpt2, tmp_path, monkeypatch, n_hand_check=100
    )
    rows = _csv(run_dir, "hand_check.csv")
    row = next(r for r in rows if r["uid"] == "alpha-range-Q3")

    for stem in ("free", "forced"):
        for order in ORDERS:
            for field in ("completion", "form", "value", "grade", "unit_flag"):
                assert f"{stem}_fewshot_{order}_{field}" in row
    assert row["free_fewshot_range_last_form"] == "range"
    assert row["free_fewshot_exact_last_form"] == "point"
    assert row["free_fewshot_exact_last_value"] == "130"


def test_the_hand_check_names_chat_columns_without_an_order(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(
        _post_trained(tiny_qwen3), tmp_path, monkeypatch, n_hand_check=100
    )

    row = _csv(run_dir, "hand_check.csv")[0]
    assert "free_chat_completion" in row
    assert "free_chat_none_completion" not in row


def test_items_json_keys_the_answers_by_shot_order(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(
        _post_trained(tiny_qwen3), tmp_path, monkeypatch, shot_orders=ORDERS
    )

    records = run_dir.read_json("items")
    assert isinstance(records, list)
    assert set(records[0]["answers"]) == {
        f"{condition}/{fmt}/{order}"
        for condition in ("free", "forced")
        for fmt, order in [
            ("fewshot", "range_last"),
            ("fewshot", "exact_last"),
            ("chat", "none"),
        ]
    }


def test_the_summary_lists_the_shot_orders_that_ran(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _both_orders(tiny_qwen3, tmp_path / "a", monkeypatch)
    chat_only, _ = _executed(
        _post_trained(tiny_qwen3),
        tmp_path / "b",
        monkeypatch,
        formats=("chat",),
        shot_orders=ORDERS,
    )

    assert _summary(run_dir)["shot_orders"] == list(ORDERS)
    assert _summary(chat_only)["shot_orders"] == []


def test_plot_draws_one_panel_per_shot_order(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _both_orders(tiny_gpt2, tmp_path, monkeypatch)

    assert p10a_quantities.panel_orders(_csv(run_dir, "table.csv")) == (
        "fewshot",
        ["range_last", "exact_last"],
    )
    assert len(plot(run_dir, tmp_path / "figs")) == 3


def test_the_plot_panels_fall_back_to_chat_without_fewshot(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(
        _post_trained(tiny_qwen3), tmp_path, monkeypatch, formats=("chat",)
    )

    assert p10a_quantities.panel_orders(_csv(run_dir, "table.csv")) == (
        "chat",
        ["none"],
    )


def _clustered(
    per_item_ranges: list[int], n_samples: int = 5
) -> tuple[list[QuantityQuestion], Results]:
    """Four range items whose samples are all range or all not."""
    items = _pool(("alpha",), per_side=len(per_item_ranges))
    outcomes = []
    for item in items:
        if item.kind == "exact":
            outcomes.append(_outcome(_answer("point"), _answer("point")))
            continue
        shares = per_item_ranges[int(item.uid[-1])]
        samples = tuple(
            _answer("range" if k < shares else "point", correct=True)
            for k in range(n_samples)
        )
        outcomes.append(Outcome(greedy=_answer("range"), samples=samples))
    return items, {("free", "fewshot", "range_last"): outcomes}


def test_cluster_cis_is_degenerate_for_identical_items() -> None:
    hits = np.full((6, 1), 3.0)
    trials = np.full((6, 1), 5.0)

    cis = cluster_cis(hits, trials, seed=1)

    assert cis.shape == (1, 2)
    assert cis[0].tolist() == [0.6, 0.6]


def test_cluster_cis_spans_the_items_not_the_samples() -> None:
    hits = np.array([[10.0], [0.0]])
    trials = np.array([[10.0], [10.0]])

    low, high = cluster_cis(hits, trials, seed=1)[0]

    assert (low, high) == (0.0, 1.0)


def test_cluster_cis_is_seeded_and_resamples_2000_times_by_default() -> None:
    rng = np.random.default_rng(0)
    hits = rng.integers(0, 6, size=(9, 2)).astype(float)
    trials = np.full((9, 2), 5.0)

    first = cluster_cis(hits, trials, seed=1)

    assert np.array_equal(first, cluster_cis(hits, trials, seed=1))
    assert not np.array_equal(first, cluster_cis(hits, trials, seed=2))
    default = inspect.signature(cluster_cis).parameters["n_boot"].default
    assert default == 2000


def test_cluster_cis_skips_resamples_without_a_denominator() -> None:
    hits = np.array([[1.0, 0.0], [0.0, 0.0]])
    trials = np.array([[1.0, 0.0], [1.0, 0.0]])

    cis = cluster_cis(hits, trials, seed=1)

    assert not np.isnan(cis[0]).any()
    assert np.isnan(cis[1]).all()


def _uniform(form: str, correct: bool | None) -> tuple[list[Any], Results]:
    items = _pool(("alpha",), per_side=3)
    samples = tuple(_answer(form, correct=correct) for _ in range(4))
    outcomes = [Outcome(greedy=_answer(form, correct=correct), samples=samples)]
    results: Results = {
        ("free", "fewshot", "range_last"): outcomes * len(items)
    }
    return items, results


def test_a_sampled_rate_of_zero_gets_an_item_wilson_interval() -> None:
    items, results = _uniform("point", True)

    columns, body = build_table(items, results, seed=0)
    row = _find(
        _rows(columns, body), family="alpha", kind="range", decoding="sampled"
    )

    assert row["range_rate"] == 0.0
    assert (row["range_lo"], row["range_hi"]) == wilson_ci(0, 3)
    assert row["range_hi"] > 0.0
    assert row["interval"] == "item_wilson"


def test_a_sampled_rate_of_one_gets_an_item_wilson_interval() -> None:
    items, results = _uniform("range", True)

    columns, body = build_table(items, results, seed=0)
    row = _find(
        _rows(columns, body), family="alpha", kind="range", decoding="sampled"
    )

    assert row["range_rate"] == 1.0
    assert (row["range_lo"], row["range_hi"]) == wilson_ci(3, 3)
    assert row["range_lo"] < 1.0
    assert (row["correct_lo"], row["correct_hi"]) == wilson_ci(3, 3)


def test_a_mixed_row_names_the_metrics_with_an_item_wilson_interval() -> None:
    items, results = _clustered([5, 5, 0, 0])

    columns, body = build_table(items, results, seed=3)
    row = _find(
        _rows(columns, body), family="alpha", kind="range", decoding="sampled"
    )

    assert row["interval"] == "item_bootstrap+item_wilson"
    assert row["item_wilson"] == (
        "approximate;abstain;none;unit_mismatch;truncated;correct"
    )
    assert (row["range_lo"], row["range_hi"]) == (0.0, 1.0)
    greedy = _find(
        _rows(columns, body), family="alpha", kind="range", decoding="greedy"
    )
    assert greedy["item_wilson"] == ""


def test_the_item_wilson_count_is_the_items_with_a_denominator() -> None:
    items = _pool(("alpha",), per_side=3)
    graded = _answer("range", correct=True)
    ungraded = _answer("abstain")
    outcomes = [Outcome(greedy=graded, samples=(graded,) * 2)] * 2
    outcomes += [Outcome(greedy=ungraded, samples=(ungraded,) * 2)]
    results: Results = {
        ("free", "fewshot", "range_last"): outcomes + list(outcomes)
    }

    columns, body = build_table(items, results, seed=0)
    row = _find(
        _rows(columns, body), family="alpha", kind="range", decoding="sampled"
    )

    assert (row["correct_lo"], row["correct_hi"]) == wilson_ci(2, 2)


def test_the_sampled_rows_use_the_item_bootstrap() -> None:
    items, results = _clustered([5, 5, 0, 0])

    columns, body = build_table(items, results, seed=3)
    rows = _rows(columns, body)

    sampled = _find(rows, family="alpha", kind="range", decoding="sampled")
    assert sampled["interval"] == "item_bootstrap+item_wilson"
    assert sampled["range_rate"] == 0.5
    assert (sampled["range_lo"], sampled["range_hi"]) == (0.0, 1.0)
    assert (sampled["range_lo"], sampled["range_hi"]) != wilson_ci(10, 20)
    greedy = _find(rows, family="alpha", kind="range", decoding="greedy")
    assert greedy["interval"] == "wilson"
    assert (greedy["range_lo"], greedy["range_hi"]) == wilson_ci(4, 4)


def test_the_sampled_correct_rate_is_over_the_gradable_samples() -> None:
    items, results = _clustered([5, 5, 0, 0])

    columns, body = build_table(items, results, seed=3)
    sampled = _find(
        _rows(columns, body), family="alpha", kind="range", decoding="sampled"
    )

    assert (sampled["n"], sampled["n_gradable"]) == (20, 20)
    assert sampled["correct"] == 20
    assert sampled["correct_rate"] == 1.0
    assert (sampled["correct_lo"], sampled["correct_hi"]) == wilson_ci(4, 4)


def test_a_sampled_cell_without_gradable_samples_has_no_interval() -> None:
    items = _pool(("alpha",), per_side=2)
    results: Results = {
        ("free", "fewshot", "range_last"): [
            _outcome(_answer("abstain"), _answer("abstain")) for _ in items
        ]
    }

    columns, body = build_table(items, results, seed=0)
    row = _find(
        _rows(columns, body), family="alpha", kind="range", decoding="sampled"
    )

    assert row["correct_rate"] is None
    assert row["correct_lo"] is None
    assert row["correct_hi"] is None


def test_the_table_forwards_the_seed_to_the_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds: list[int] = []
    real = p10a_quantities.cluster_cis

    def spy(hits: Any, trials: Any, *, seed: int) -> Any:
        seeds.append(seed)
        return real(hits, trials, seed=seed)

    monkeypatch.setattr(p10a_quantities, "cluster_cis", spy)
    items, results = _clustered([5, 0, 5, 0])

    build_table(items, results, seed=11)

    assert seeds == [11] * 4


def test_width_sampled_rows_with_degenerate_rates_use_item_wilson() -> None:
    items = [
        _item(uid=f"u{k}", low=100.0, high=100.0 + 10 * (k + 1))
        for k in range(3)
    ]
    outcomes = [
        Outcome(
            greedy=_answer("range", correct=True),
            samples=(_answer("point", correct=False),),
        )
        for _ in range(3)
    ]
    results: Results = {("free", "fewshot", "range_last"): outcomes}

    columns, body = width_table(items, results, seed=1)
    rows = _rows(columns, body)

    assert {r["interval"] for r in rows if r["decoding"] == "greedy"} == {
        "wilson"
    }
    assert {r["interval"] for r in rows if r["decoding"] == "sampled"} == {
        "item_wilson"
    }


def test_width_cuts_tertiles_within_each_family_and_pooled() -> None:
    items = [
        _item(
            uid=f"{family}-{k}",
            family=family,
            low=100.0,
            high=100.0 + 10 * (k + 1) * scale,
        )
        for family, scale in (("alpha", 1.0), ("beta", 10.0))
        for k in range(6)
    ]
    outcomes = [_outcome(_answer("range")) for _ in items]
    results: Results = {("free", "fewshot", "range_last"): outcomes}

    columns, body = width_table(items, results, seed=0)
    rows = [r for r in _rows(columns, body) if r["decoding"] == "greedy"]

    for family in ("alpha", "beta", "all"):
        tertiles = [r for r in rows if r["family"] == family]
        assert [r["tertile"] for r in tertiles] == [1, 2, 3]
        assert [r["n_items"] for r in tertiles] == [
            2 if family != "all" else 4
        ] * 3
    alpha = [r for r in rows if r["family"] == "alpha"]
    pooled = [r for r in rows if r["family"] == "all"]
    assert alpha[2]["width_hi"] < pooled[2]["width_hi"]
    assert pooled[0]["width_hi"] <= pooled[1]["width_lo"]
    beta = [r for r in rows if r["family"] == "beta"]
    assert beta[0]["width_lo"] > alpha[0]["width_lo"]


def test_width_forwards_the_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    seeds: list[int] = []
    real = p10a_quantities.cluster_cis

    def spy(hits: Any, trials: Any, *, seed: int) -> Any:
        seeds.append(seed)
        return real(hits, trials, seed=seed)

    monkeypatch.setattr(p10a_quantities, "cluster_cis", spy)
    items = [_item(uid=f"u{k}", low=100.0, high=110.0 + k) for k in range(3)]
    results: Results = {
        ("free", "fewshot", "range_last"): [
            _outcome(_answer("range"), _answer("range")) for _ in items
        ]
    }

    width_table(items, results, seed=13)

    assert seeds
    assert set(seeds) == {13}


def test_the_run_files_use_the_config_seed_for_the_intervals(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(tiny_gpt2, tmp_path, monkeypatch, seed=2)
    rows = _csv(run_dir, "table.csv")

    assert {r["interval"] for r in rows if r["decoding"] == "greedy"} == {
        "wilson"
    }
    assert {r["interval"] for r in rows if r["decoding"] == "sampled"} <= {
        "item_bootstrap",
        "item_wilson",
        "item_bootstrap+item_wilson",
    }
    widths = _csv(run_dir, "width.csv")
    assert {r["family"] for r in widths} == {"alpha", "beta", "all"}
    assert {r["interval"] for r in widths} >= {"wilson"}


def test_forced_in_range_reads_the_greedy_answer_not_the_samples() -> None:
    items = _pool(("alpha",), per_side=2)
    outside = _answer("point", value=900.0)
    outcomes = [
        Outcome(
            greedy=_answer("point", value=(item.low + item.high) / 2),
            samples=(outside, outside),
        )
        for item in items
    ]
    results: Results = {("forced", "fewshot", "range_last"): outcomes}

    columns, body = forced_in_range_table(items, results, rel_tol=0.05)
    row = _find(_rows(columns, body), family="alpha", kind="range")

    assert (row["n_point"], row["inside"], row["inside_band"]) == (2, 2, 2)


def test_cluster_cis_uses_the_2_5_and_97_5_percentiles() -> None:
    rng = np.random.default_rng(4)
    hits = rng.integers(0, 6, size=(9, 1)).astype(float)
    trials = np.full((9, 1), 5.0)

    drawn = np.random.default_rng(7).integers(0, 9, size=(2000, 9))
    rates = hits[drawn].sum(axis=1) / trials[drawn].sum(axis=1)
    expected = np.percentile(rates, [2.5, 97.5], axis=0)

    cis = cluster_cis(hits, trials, seed=7)

    assert cis[0].tolist() == pytest.approx(expected[:, 0].tolist())
    assert cis[0, 0] < np.percentile(rates, 5) < np.percentile(rates, 95)
    assert np.percentile(rates, 95) < cis[0, 1]


def test_bootstrap_gap_ci_uses_the_2_5_and_97_5_percentiles() -> None:
    range_rates = np.array([1.0, 1.0, 0.0, 1.0, 0.5, 0.0, 1.0])
    exact_rates = np.array([0.0, 0.5, 0.0, 1.0, 0.0, 0.0])
    rng = np.random.default_rng(5)
    draws = [
        rates[rng.integers(0, len(rates), size=(2000, len(rates)))].mean(axis=1)
        for rates in (range_rates, exact_rates)
    ]
    gaps = draws[0] - draws[1]

    low, high = bootstrap_gap_ci(range_rates, exact_rates, seed=5)

    assert (low, high) == pytest.approx(tuple(np.percentile(gaps, [2.5, 97.5])))
    assert low < np.percentile(gaps, 5) < np.percentile(gaps, 95) < high


def test_the_relative_width_is_over_the_midpoint() -> None:
    items = [
        _item(uid=f"u{k}", low=100.0, high=116.0 + 20.0 * k) for k in range(3)
    ]
    outcomes = [_outcome(_answer("range", correct=True))] * 3
    results: Results = {("free", "fewshot", "range_last"): outcomes}

    columns, body = width_table(items, results, seed=0)
    first = _find(
        _rows(columns, body), family="all", decoding="greedy", tertile=1
    )

    assert first["width_lo"] == first["width_hi"] == pytest.approx(16 / 108)


def test_the_hand_check_rows_keep_the_order_of_the_items() -> None:
    items = _pool(("alpha", "beta"), per_side=3)
    outcomes = [_outcome(_answer("point")) for _ in items]
    results: Results = {("free", "fewshot", "range_last"): outcomes}

    _, rows = hand_check_table(items, results, n=12, seed=0)

    assert [row[0] for row in rows] == [item.uid for item in items]


@pytest.mark.parametrize(
    ("fmt", "text", "finished", "truncated"),
    [
        ("chat", " 35 cm", False, True),
        ("chat", " 35 cm", True, False),
        ("chat", " 35 cm\nmore", False, True),
        ("fewshot", " 35 cm", False, True),
        ("fewshot", " 35 cm\nQ: next", False, False),
        ("fewshot", " 35 cm", True, False),
    ],
)
def test_an_answer_is_truncated_when_the_budget_cut_it(
    fmt: str, text: str, finished: bool, truncated: bool
) -> None:
    answer = score_answer(_item(), text, fmt, rel_tol=0.05, finished=finished)

    assert answer.finished is finished
    assert answer.truncated is truncated


def test_the_table_rates_the_truncated_answers_with_an_interval() -> None:
    items = _pool(("alpha",), per_side=2)
    cut = _answer("point", truncated=True)
    done = _answer("point")
    outcomes = [
        _outcome(cut if i == 0 else done, done, cut, done, cut)
        for i, _ in enumerate(items)
    ]
    results: Results = {("free", "fewshot", "range_last"): outcomes}

    columns, body = build_table(items, results, seed=0)
    rows = _rows(columns, body)

    greedy = _find(rows, family="alpha", kind="range", decoding="greedy")
    assert (greedy["truncated"], greedy["truncated_rate"]) == (1, 0.5)
    assert (greedy["truncated_lo"], greedy["truncated_hi"]) == wilson_ci(1, 2)
    sampled = _find(rows, family="alpha", kind="range", decoding="sampled")
    assert (sampled["truncated"], sampled["truncated_rate"]) == (4, 0.5)
    assert sampled["truncated_lo"] <= 0.5 <= sampled["truncated_hi"]


def test_items_json_keeps_whether_each_completion_finished(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(
        tiny_gpt2,
        tmp_path,
        monkeypatch,
        finished=lambda greedy: not greedy,
    )

    records = run_dir.read_json("items")
    cell = records[0]["answers"]["free/fewshot/range_last"]

    assert cell["greedy"]["finished"] is False
    assert cell["greedy"]["truncated"] is True
    assert all(a["finished"] is True for a in cell["samples"])
    assert not any(a["truncated"] for a in cell["samples"])
    rows = _csv(run_dir, "table.csv")
    greedy = [r for r in rows if r["decoding"] == "greedy"]
    assert {r["truncated_rate"] for r in greedy} == {"1.0"}
    sampled = [r for r in rows if r["decoding"] == "sampled"]
    assert {r["truncated_rate"] for r in sampled} == {"0.0"}


def _sampled(
    n_flagged: int, per_item: int = 4
) -> tuple[list[QuantityQuestion], Results]:
    """Items whose samples are plain points except n_flagged abstentions."""
    items = _pool(("alpha", "beta"), per_side=5)
    outcomes = []
    flagged = 0
    for i, _ in enumerate(items):
        samples = []
        for k in range(per_item):
            form = "point"
            if flagged < n_flagged and (i + k) % 2 == 0:
                form = "abstain"
                flagged += 1
            samples.append(dataclasses.replace(_answer(form), text=f"c{i}.{k}"))
        outcomes.append(
            Outcome(greedy=_answer("point"), samples=tuple(samples))
        )
    return items, {("free", "fewshot", "range_last"): outcomes}


def _is_flagged(row: list[Any], columns: list[str]) -> bool:
    record = dict(zip(columns, row, strict=True))
    return record["form"] in ("none", "abstain") or record["unit_flag"] != ""


def test_the_sample_hand_check_has_one_completion_per_row() -> None:
    items, results = _sampled(n_flagged=6)

    columns, rows = hand_check_samples_table(items, results, n=12, seed=0)

    assert columns == [
        "uid",
        "wikidata_url",
        "family",
        "kind",
        "question",
        "gold_low",
        "gold_high",
        "unit",
        "condition",
        "format",
        "shot_order",
        "sample",
        "completion",
        "form",
        "value",
        "grade",
        "unit_flag",
        "human_form",
        "human_correct",
        "human_unit_mismatch",
    ]
    assert len(rows) == 12
    record = dict(zip(columns, rows[0], strict=True))
    item = next(i for i in items if i.uid == record["uid"])
    index = items.index(item)
    sample = results[("free", "fewshot", "range_last")][index].samples[
        record["sample"]
    ]
    assert record["completion"] == sample.text
    assert record["wikidata_url"].endswith(item.entity_id)
    assert (record["condition"], record["format"]) == ("free", "fewshot")
    assert record["shot_order"] == "range_last"
    assert (record["human_form"], record["human_correct"]) == ("", "")


def test_the_sample_hand_check_reserves_a_quarter_for_flagged_answers() -> None:
    items, results = _sampled(n_flagged=30)

    columns, rows = hand_check_samples_table(items, results, n=12, seed=0)

    assert sum(_is_flagged(r, columns) for r in rows) == 3


def test_the_sample_hand_check_takes_what_flagged_answers_exist() -> None:
    items, results = _sampled(n_flagged=1)

    columns, rows = hand_check_samples_table(items, results, n=12, seed=0)

    assert len(rows) == 12
    assert sum(_is_flagged(r, columns) for r in rows) == 1


def test_the_sample_hand_check_fills_up_from_flagged_when_plain_run_out() -> (
    None
):
    items = _pool(("alpha",), per_side=1)
    plain = dataclasses.replace(_answer("point"), text="p")
    flagged = dataclasses.replace(_answer("none"), text="f")
    outcomes = [
        Outcome(greedy=plain, samples=(plain, flagged, flagged, flagged))
        for _ in items
    ]
    results: Results = {("free", "fewshot", "range_last"): outcomes}

    columns, rows = hand_check_samples_table(items, results, n=6, seed=0)

    assert len(rows) == 6
    assert sum(_is_flagged(r, columns) for r in rows) == 4


def test_the_sample_hand_check_is_stratified_by_family_and_kind() -> None:
    items, results = _sampled(n_flagged=0)

    columns, rows = hand_check_samples_table(items, results, n=12, seed=0)
    records = [dict(zip(columns, r, strict=True)) for r in rows]

    strata = [(r["family"], r["kind"]) for r in records]
    assert {s: strata.count(s) for s in set(strata)} == {
        ("alpha", "range"): 3,
        ("alpha", "exact"): 3,
        ("beta", "range"): 3,
        ("beta", "exact"): 3,
    }


def test_the_sample_hand_check_follows_the_seed_and_keeps_item_order() -> None:
    items, results = _sampled(n_flagged=6)

    first = hand_check_samples_table(items, results, n=12, seed=3)[1]
    again = hand_check_samples_table(items, results, n=12, seed=3)[1]
    other = hand_check_samples_table(items, results, n=12, seed=4)[1]

    assert first == again
    assert first != other
    positions = [
        (
            [i.uid for i in items].index(row[0]),
            row[11],
        )
        for row in first
    ]
    assert positions == sorted(positions)


def test_the_run_writes_the_sample_hand_check_with_a_bom(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(tiny_gpt2, tmp_path, monkeypatch, n_hand_check=8)

    for name in ("hand_check.csv", "hand_check_samples.csv"):
        raw = (run_dir.path / name).read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")
        assert not raw.startswith(b"\xef\xbb\xbf\xef\xbb\xbf")
    rows = _csv(run_dir, "hand_check_samples.csv")
    assert len(rows) == 8
    assert {r["sample"] for r in rows} <= {"0", "1", "2", "3"}
    assert _summary(run_dir)["hand_check_sample_rows"] == 8


def _people_items(n: int = 4) -> list[QuantityQuestion]:
    return [
        _item(
            uid=f"{PEOPLE_FAMILY}-exact-P{i}",
            family=PEOPLE_FAMILY,
            kind="exact",
            entity=f"Person {i}",
            entity_id=f"P{i}",
            question=f"How tall is Person {i}, in centimetres?",
            low=170.0 + i,
            high=170.0 + i,
            split="test",
        )
        for i in range(n)
    ]


def _group_of(item: QuantityQuestion) -> str:
    if item.family == PEOPLE_FAMILY:
        return "people"
    return "breeds" if item.kind == "range" else "buildings"


def _group_results(
    items: list[QuantityQuestion],
    shares: dict[str, list[float]],
    cell: tuple[str, str, str] = ("free", "fewshot", "range_last"),
) -> Results:
    """Ten samples per item; the greedy answer hedges at a share of 0.5."""
    outcomes = []
    seen = {name: 0 for name in shares}
    for item in items:
        group = _group_of(item)
        share = shares[group][seen[group] % len(shares[group])]
        seen[group] += 1
        outcomes.append(
            Outcome(
                greedy=_answer("range" if share >= 0.5 else "point"),
                samples=tuple(
                    _answer("range" if k < round(10 * share) else "point")
                    for k in range(10)
                ),
            )
        )
    return {cell: outcomes}


def _without_people(
    items: list[QuantityQuestion], results: Results
) -> tuple[list[QuantityQuestion], Results]:
    keep = [i.family != PEOPLE_FAMILY for i in items]
    return (
        [i for i, k in zip(items, keep, strict=True) if k],
        {
            cell: [o for o, k in zip(outcomes, keep, strict=True) if k]
            for cell, outcomes in results.items()
        },
    )


GROUP_SHARES = {
    "people": [0.0, 0.0, 0.0, 1.0],
    "breeds": [1.0, 1.0, 0.0, 1.0],
    "buildings": [0.0, 0.0, 0.0, 0.0],
}


def _with_people() -> tuple[list[QuantityQuestion], Results]:
    items = [*_pool(("height", "wingspan")), *_people_items()]
    return items, _group_results(items, GROUP_SHARES)


def _height_and_people() -> list[QuantityQuestion]:
    return [
        i for i in _with_people()[0] if i.family in ("height", PEOPLE_FAMILY)
    ]


def test_people_can_be_turned_off_and_round_trip() -> None:
    config = Config(people=False)

    assert Config.from_dict(config.to_dict()) == config


def test_the_table_has_person_rows_and_family_only_pooled_rows() -> None:
    items, results = _with_people()
    kept_items, kept_results = _without_people(items, results)

    rows = _rows(*build_table(items, results, seed=2))

    people = [r for r in rows if r["family"] == PEOPLE_FAMILY]
    assert {r["kind"] for r in people} == {"exact"}
    assert {r["n"] for r in people if r["decoding"] == "greedy"} == {4}
    pooled = [r for r in rows if r["family"] == "all"]
    assert pooled == [
        r
        for r in _rows(*build_table(kept_items, kept_results, seed=2))
        if r["family"] == "all"
    ]


def test_forced_in_range_has_person_rows_and_family_only_pooled_rows() -> None:
    items, _ = _with_people()
    cell = ("forced", "fewshot", "range_last")
    results: Results = {
        cell: [_outcome(_answer("point", i.low + 1)) for i in items]
    }
    kept_items, kept_results = _without_people(items, results)

    rows = _rows(*forced_in_range_table(items, results, rel_tol=0.05))

    people = [r for r in rows if r["family"] == PEOPLE_FAMILY]
    assert [(r["kind"], r["n_items"]) for r in people] == [("exact", 4)]
    kept = forced_in_range_table(kept_items, kept_results, rel_tol=0.05)
    assert [r for r in rows if r["family"] == "all"] == [
        r for r in _rows(*kept) if r["family"] == "all"
    ]


def test_hedge_gap_and_width_ignore_the_people() -> None:
    items, results = _with_people()
    kept_items, kept_results = _without_people(items, results)

    for table in (hedge_gap_table, width_table):
        with_people = table(items, results, seed=3)

        assert with_people == table(kept_items, kept_results, seed=3)
        assert PEOPLE_FAMILY not in {row[0] for row in with_people[1]}


def test_the_diagnostic_rates_people_breeds_and_buildings() -> None:
    items = _height_and_people()

    columns, body = diagnostic_table(
        items, _group_results(items, GROUP_SHARES), seed=0
    )
    rows = _rows(columns, body)

    assert len(rows) == 2
    greedy = _find(rows, decoding="greedy")
    assert greedy["people_rate"] == pytest.approx(0.25)
    assert greedy["breeds_rate"] == pytest.approx(0.75)
    assert greedy["buildings_rate"] == pytest.approx(0.0)
    assert greedy["people_minus_breeds"] == pytest.approx(-0.5)
    assert greedy["people_minus_buildings"] == pytest.approx(0.25)
    assert (greedy["people_lo"], greedy["people_hi"]) == wilson_ci(1, 4)
    assert (
        greedy["people_minus_breeds_lo"]
        < -0.5
        < greedy["people_minus_breeds_hi"]
    )
    assert (
        greedy["people_minus_buildings_lo"]
        <= 0.25
        <= greedy["people_minus_buildings_hi"]
    )
    assert (greedy["n_people"], greedy["n_breeds"]) == (4, 4)
    assert greedy["n_buildings"] == 4
    assert greedy["people_interval"] == "wilson"
    sampled = _find(rows, decoding="sampled")
    assert sampled["people_rate"] == pytest.approx(0.25)
    assert sampled["people_minus_breeds"] == pytest.approx(-0.5)
    assert sampled["people_interval"] == "item_bootstrap"
    assert sampled["buildings_interval"] == "item_wilson"
    assert sampled["people_lo"] <= 0.25 <= sampled["people_hi"]


def test_the_diagnostic_uses_the_gap_bootstrap_with_people_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[float], list[float], int]] = []

    def spy(
        first: np.ndarray, second: np.ndarray, *, seed: int
    ) -> tuple[float, float]:
        calls.append((list(first), list(second), seed))
        return 0.0, 0.0

    monkeypatch.setattr(p10a_quantities, "bootstrap_gap_ci", spy)
    items = _height_and_people()

    diagnostic_table(items, _group_results(items, GROUP_SHARES), seed=7)

    assert calls[0] == ([0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], 7)
    assert calls[1] == ([0.0, 0.0, 0.0, 1.0], [0.0] * 4, 7)
    assert {c[2] for c in calls} == {7}


def test_the_diagnostic_has_a_both_row_that_averages_the_shot_orders() -> None:
    items = _height_and_people()
    last_range = _group_results(
        items,
        {"people": [0.0], "breeds": [1.0], "buildings": [0.0]},
        ("free", "fewshot", "range_last"),
    )
    last_exact = _group_results(
        items,
        {"people": [1.0], "breeds": [0.0], "buildings": [0.0]},
        ("free", "fewshot", "exact_last"),
    )

    columns, body = diagnostic_table(
        items, {**last_range, **last_exact}, seed=0
    )
    rows = _rows(columns, body)

    assert {r["shot_order"] for r in rows} == {
        "range_last",
        "exact_last",
        "both",
    }
    both = _find(rows, shot_order="both", decoding="greedy")
    assert both["people_rate"] == pytest.approx(0.5)
    assert both["breeds_rate"] == pytest.approx(0.5)
    assert both["people_minus_breeds"] == pytest.approx(0.0)
    assert both["people_minus_buildings"] == pytest.approx(0.5)
    first = _find(rows, shot_order="range_last", decoding="greedy")
    assert first["people_minus_breeds"] == pytest.approx(-1.0)


def test_the_diagnostic_is_empty_without_people_or_the_height_family() -> None:
    items, results = _with_people()
    kept_items, kept_results = _without_people(items, results)
    other = [i for i in items if i.family in ("wingspan", PEOPLE_FAMILY)]

    assert diagnostic_table(kept_items, kept_results, seed=0)[1] == []
    assert (
        diagnostic_table(other, _group_results(other, GROUP_SHARES), seed=0)[1]
        == []
    )


def _people_pool() -> list[QuantityQuestion]:
    return [*_pool(("height",), per_side=6), *_people_items(6)]


def test_execute_writes_the_diagnostic_with_people(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, calls = _executed(
        tiny_gpt2, tmp_path, monkeypatch, items=_people_pool()
    )

    rows = _csv(run_dir, "diagnostic.csv")
    greedy = _find(rows, condition="free", decoding="greedy")
    assert float(greedy["people_rate"]) == 0.0
    assert float(greedy["breeds_rate"]) == 1.0
    assert float(greedy["people_minus_breeds"]) == -1.0
    assert len(calls.calls[0]["prompts"]) == 18
    summary = _summary(run_dir)
    assert len(summary["diagnostic"]) == len(rows)
    assert summary["diagnostic"][0]["people_minus_breeds"] is not None
    assert PEOPLE_FAMILY in summary["families"]
    table = _csv(run_dir, "table.csv")
    assert PEOPLE_FAMILY in {r["family"] for r in table}


def test_the_diagnostic_is_not_written_when_people_are_off(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(
        tiny_gpt2, tmp_path, monkeypatch, items=_people_pool(), people=False
    )

    assert not (run_dir.path / "diagnostic.csv").exists()
    assert "diagnostic" not in _summary(run_dir)


def test_the_diagnostic_is_not_written_without_the_height_family(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = [*_pool(("wingspan",), per_side=6), *_people_items(6)]
    run_dir, _ = _executed(tiny_gpt2, tmp_path, monkeypatch, items=items)

    assert not (run_dir.path / "diagnostic.csv").exists()
    assert "diagnostic" not in _summary(run_dir)


def test_the_hand_check_has_a_person_stratum_and_keeps_its_size(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(
        tiny_gpt2, tmp_path, monkeypatch, items=_people_pool()
    )

    for name in ("hand_check.csv", "hand_check_samples.csv"):
        rows = _csv(run_dir, name)
        assert len(rows) == 8, name
        strata = Counter((r["family"], r["kind"]) for r in rows)
        assert set(strata) == {
            ("height", "range"),
            ("height", "exact"),
            (PEOPLE_FAMILY, "exact"),
        }, name
        if name == "hand_check.csv":
            assert max(strata.values()) - min(strata.values()) <= 1


def test_the_plot_survives_person_rows(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _ = _executed(
        tiny_gpt2, tmp_path, monkeypatch, items=_people_pool()
    )

    paths = plot(run_dir, tmp_path / "figures")

    assert paths
    assert all(p.is_file() for p in paths)


def _run_with(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **overrides: Any,
) -> tuple[list[dict[str, Any]], list[QuantityQuestion]]:
    people_calls: list[dict[str, Any]] = []
    executed: list[QuantityQuestion] = []
    family_items = _pool(("height",), per_side=2)

    def fake_people(**kwargs: Any) -> list[QuantityQuestion]:
        people_calls.append(kwargs)
        return _people_items(2)

    def fake_execute(
        config: Config,
        run_dir: RunDir,
        loaded: LoadedModel,
        items: Sequence[QuantityQuestion],
    ) -> None:
        executed.extend(items)

    monkeypatch.setattr(
        p10a_quantities, "load_quantities", lambda **kw: family_items
    )
    monkeypatch.setattr(p10a_quantities, "load_people_heights", fake_people)
    monkeypatch.setattr(p10a_quantities, "load_model", lambda key: tiny_gpt2)
    monkeypatch.setattr(p10a_quantities, "execute", fake_execute)
    config = _config(**overrides)

    run(config, RunDir.create(config, out_root=tmp_path))

    return people_calls, executed


def test_run_loads_the_people_with_the_offline_setting(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, executed = _run_with(tiny_gpt2, tmp_path, monkeypatch, offline=True)

    assert calls == [{"offline": True}]
    assert [i.family for i in executed] == [
        *["height"] * 4,
        PEOPLE_FAMILY,
        PEOPLE_FAMILY,
    ]


def test_run_skips_the_people_when_they_are_off(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, executed = _run_with(tiny_gpt2, tmp_path, monkeypatch, people=False)

    assert calls == []
    assert PEOPLE_FAMILY not in {i.family for i in executed}
