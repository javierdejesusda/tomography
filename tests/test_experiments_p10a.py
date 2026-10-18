"""Tests for the P10a quantity behavior experiment."""

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
from tomography.experiments.p10a_quantities import execute
from tomography.experiments.p10a_quantities import forced_in_range_table
from tomography.experiments.p10a_quantities import hedge_gap_table
from tomography.experiments.p10a_quantities import plot
from tomography.experiments.p10a_quantities import run
from tomography.experiments.p10a_quantities import score_answer
from tomography.experiments.p10a_quantities import width_table
from tomography.experiments.sweep import chat_format_for
from tomography.model import MODELS
from tomography.model import LoadedModel
from tomography.reading.metrics import wilson_ci

DEFAULT_HASH = "989620d61461"


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
    assert config.offline is False
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
        assert forced.isdigit()
        if " to " in free:
            low, high = (int(part) for part in free.split(" to "))
            assert low <= int(forced) <= high
        else:
            assert forced == free


def test_the_forced_range_replacements_are_72_8_and_15() -> None:
    replaced = [
        forced
        for (_, free), (_, forced) in zip(FREE_SHOTS, FORCED_SHOTS, strict=True)
        if " to " in free
    ]

    assert replaced == ["72", "8", "15"]


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
        ("minutes", " 2 hours 22 minutes", False),
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
    results: Results = {("free", "fewshot"): []}
    for i, item in enumerate(items):
        if item.kind == "range":
            greedy = _answer(
                forms[i],
                correct=True if forms[i] in ("point", "range") else None,
            )
        else:
            greedy = _answer("point", correct=i % 2 == 0, mismatch=i == 4)
        results[("free", "fewshot")].append(_outcome(greedy))

    columns, body = build_table(items, results)
    rows = _rows(columns, body)

    assert columns[:7] == [
        "family",
        "kind",
        "condition",
        "format",
        "decoding",
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
        ("forced", "chat"): [
            _outcome(_answer("point", correct=True), *samples) for _ in items
        ]
    }

    columns, body = build_table(items, results)
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
        ("free", "fewshot"): [
            _outcome(_answer("range", approximate=True)),
            _outcome(_answer("none", approximate=True)),
            _outcome(_answer("point")),
            _outcome(_answer("point")),
        ]
    }

    columns, body = build_table(items, results)
    row = _find(
        _rows(columns, body), family="alpha", kind="range", decoding="greedy"
    )

    assert row["approximate"] == 2
    assert row["approximate_rate"] == 1.0


def test_a_cell_without_gradable_answers_has_no_correct_rate() -> None:
    items = _pool(("alpha",), per_side=1)
    results: Results = {
        ("free", "fewshot"): [_outcome(_answer("abstain")) for _ in items]
    }

    columns, body = build_table(items, results)
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
    return {("free", "fewshot"): outcomes}


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
        ("forced", "fewshot"): [_outcome(greedy[i.uid]) for i in items],
        ("free", "fewshot"): [_outcome(_answer("point", 1.0)) for _ in items],
    }

    columns, body = forced_in_range_table(items, results, rel_tol=0.05)
    rows = _rows(columns, body)

    assert {r["format"] for r in rows} == {"fewshot"}
    assert [c for c in columns[:4]] == ["family", "kind", "format", "n_items"]
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
        ("forced", "chat"): [_outcome(_answer("point", v)) for v in values]
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
        ("free", "fewshot"): [_outcome(_answer("point")) for _ in items]
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
        ("free", "fewshot"): [
            _outcome(_answer(f, correct=k % 2 == 0))
            for k, f in enumerate(forms)
        ]
    }

    columns, body = width_table(items, results)
    rows = _rows(columns, [r for r in body if r[2] == "greedy"])

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
        ("free", "chat"): [_outcome(_answer("range")) for _ in items]
    }

    columns, body = width_table(items, results)
    rows = _rows(columns, body)

    assert {r["tertile"] for r in rows} == {1}
    assert {r["n_items"] for r in rows} == {1}
    assert {r["decoding"] for r in rows} == {"greedy", "sampled"}


def _planted(item: QuantityQuestion, forced: bool, greedy: bool, k: int) -> str:
    """Range items hedge when free; samples alternate with an abstention."""
    midpoint = (item.low + item.high) / 2
    if item.kind == "exact":
        return f" {item.low:g} cm"
    if forced:
        return f" {midpoint:g}"
    return (
        f" {item.low:g} to {item.high:g} cm" if k % 2 == 0 else " I do not know"
    )


class _Calls:
    """Replaces `generate` with planted completions and records the calls."""

    def __init__(
        self,
        items: list[QuantityQuestion],
        rule: Callable[[QuantityQuestion, bool, bool, int], str] = _planted,
    ) -> None:
        self.items = items
        self.rule = rule
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
            out.append(
                [
                    Completion(
                        text=self.rule(item, forced, temperature is None, k),
                        token_ids=(),
                        finished=True,
                    )
                    for k in range(num_samples)
                ]
            )
        return out


def _config(**overrides: Any) -> Config:
    base = Config(model="tiny", n_samples=4, n_hand_check=8, batch_size=5)
    return dataclasses.replace(base, **overrides)


def _executed(
    loaded: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    items: list[QuantityQuestion] | None = None,
    rule: Callable[[QuantityQuestion, bool, bool, int], str] = _planted,
    **overrides: Any,
) -> tuple[RunDir, _Calls]:
    items = items if items is not None else _pool(per_side=6)
    calls = _Calls(items, rule)
    monkeypatch.setattr(p10a_quantities, "generate", calls)
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, loaded, items)
    return run_dir, calls


def _post_trained(loaded: LoadedModel) -> LoadedModel:
    return dataclasses.replace(loaded, checkpoint=MODELS["qwen3-1.7b"])


def _csv(run_dir: RunDir, name: str) -> list[dict[str, str]]:
    with (run_dir.path / name).open(encoding="utf-8", newline="") as handle:
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
    assert len(_csv(run_dir, "width.csv")) == 2 * 3 * 2


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
    cell = record["answers"]["free/fewshot"]
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
        "free/fewshot",
        "forced/fewshot",
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
    assert row["free_fewshot_completion"] == " 130 to 200 cm"
    assert row["free_fewshot_form"] == "range"
    assert row["free_fewshot_value"] == "130 to 200"
    assert row["free_fewshot_grade"] == "correct"
    assert row["forced_fewshot_value"] == "165"
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
    def rule(item: QuantityQuestion, forced: bool, greedy: bool, k: int) -> str:
        return " 12\nQ: next?"

    run_dir, _ = _executed(
        tiny_gpt2, tmp_path, monkeypatch, rule=rule, n_hand_check=100
    )

    row = _csv(run_dir, "hand_check.csv")[0]
    assert row["free_fewshot_completion"] == " 12\\nQ: next?"
    assert row["free_fewshot_form"] == "point"


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
    config = _config()

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
