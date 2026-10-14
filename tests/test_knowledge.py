import csv
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from tomography.control.generate import Completion
from tomography.datasets.records import QAExample
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import ItemLabel
from tomography.knowledge import Label
from tomography.knowledge import LabelRecipe
from tomography.knowledge import extract_answer
from tomography.knowledge import few_shot_prompt
from tomography.knowledge import is_correct
from tomography.knowledge import label_from_counts
from tomography.knowledge import label_questions
from tomography.knowledge import normalize_answer
from tomography.knowledge import write_hand_check
from tomography.model import LoadedModel
from tomography.model import load_model

SHOTS = (("What is 2 plus 2?", "four"), ("Capital of France?", "Paris"))


def test_few_shot_prompt_is_exact_for_two_shots() -> None:
    prompt = few_shot_prompt("Who wrote Hamlet?", SHOTS)

    assert prompt == (
        "Q: What is 2 plus 2?\nA: four\n\n"
        "Q: Capital of France?\nA: Paris\n\n"
        "Q: Who wrote Hamlet?\nA:"
    )


def test_default_shots_are_fifteen_distinct_single_line_pairs() -> None:
    assert len(DEFAULT_SHOTS) == 15
    for question, answer in DEFAULT_SHOTS:
        assert question.strip()
        assert answer.strip()
        assert "\n" not in question
        assert "\n" not in answer
    assert len({question for question, _ in DEFAULT_SHOTS}) == 15


@pytest.mark.parametrize(
    "question",
    [
        "What is the capital of Canada?",
        "What is the chemical symbol for gold?",
        "What is the square root of 81?",
        "What is the largest ocean on Earth?",
        "What is the hardest natural substance?",
    ],
    ids=[
        "capital_of_canada",
        "symbol_for_gold",
        "square_root_of_81",
        "largest_ocean",
        "hardest_substance",
    ],
)
def test_default_shots_exclude_benchmark_items(question: str) -> None:
    assert question not in {q for q, _ in DEFAULT_SHOTS}


@pytest.mark.parametrize(
    ("completion", "stop", "expected"),
    [
        (" Paris\nQ: next", "\n", "Paris"),
        (" Paris  ", "\n", "Paris"),
        ("\n Paris", "\n", ""),
        ("", "\n", ""),
        (" one. two. three", ".", "one"),
        (" Paris\nQ: next", "", "Paris\nQ: next"),
    ],
    ids=[
        "cut_at_newline",
        "strip_spaces",
        "leading_stop",
        "empty",
        "custom_stop",
        "empty_stop_keeps_all",
    ],
)
def test_extract_answer(completion: str, stop: str, expected: str) -> None:
    assert extract_answer(completion, stop=stop) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Café Münster", "cafe munster"),
        ("The Lord of the Rings", "lord of rings"),
        ("An apple, a pear!", "apple pear"),
        ("N.Y.C.", "n y c"),
        ("  Hello---World  ", "hello world"),
        ("Zoë's 3rd", "zoe s 3rd"),
        ("the", ""),
        ("", ""),
        ("?!", ""),
        ("Москва", "москва"),
        ("Straße", "straße"),
        ("Søren", "søren"),
        ("Zoë Ж", "zoe ж"),
    ],
    ids=[
        "accents",
        "articles",
        "article_and_punctuation",
        "dotted_letters",
        "punctuation_runs",
        "apostrophe_and_digits",
        "only_article",
        "empty",
        "only_punctuation",
        "cyrillic_kept",
        "sharp_s_kept",
        "slashed_o_kept",
        "accent_stripped_next_to_cyrillic",
    ],
)
def test_normalize_answer(text: str, expected: str) -> None:
    assert normalize_answer(text) == expected


@pytest.mark.parametrize(
    ("answer", "aliases", "expected"),
    [
        ("Paris", ("paris",), True),
        ("It is the city of Paris, France", ("Paris",), True),
        ("party", ("art",), False),
        ("the art of war", ("Art",), True),
        ("New York City", ("new york",), True),
        ("York New", ("new york",), False),
        ("Café", ("cafe",), True),
        ("anything", ("", "!!"), False),
        ("anything", (), False),
        ("", ("Paris",), False),
        ("Lyon", ("", "Lyon"), True),
        ("Москва", ("Москва",), True),
        ("Zoe Smith", ("Zoë Ж",), False),
        ("Constanța's port", ("Буцуреști",), False),  # noqa: RUF001
        ("Straße", ("Straße",), True),
        ("Søren Kierkegaard", ("Søren",), True),
        ("", ("!!", "Москва"), False),
    ],
    ids=[
        "exact",
        "inside_sentence",
        "substring_is_not_a_word",
        "whole_word",
        "multiword_sequence",
        "word_order_matters",
        "accent_insensitive",
        "empty_aliases_ignored",
        "no_aliases",
        "empty_answer",
        "empty_alias_does_not_block_others",
        "cyrillic_alias_matches",
        "cyrillic_alias_is_not_degraded",
        "mixed_script_alias_is_not_degraded",
        "sharp_s_round_trip",
        "slashed_o_round_trip",
        "empty_answer_ignores_empty_aliases",
    ],
)
def test_is_correct(
    answer: str, aliases: tuple[str, ...], expected: bool
) -> None:
    assert is_correct(answer, aliases) is expected


@pytest.mark.parametrize(
    ("greedy", "n_correct", "expected"),
    [
        *[(True, n, "uncertain") for n in range(3)],
        *[(True, n, "known") for n in range(3, 6)],
        (False, 0, "unknown"),
        *[(False, n, "uncertain") for n in range(1, 6)],
    ],
    ids=[
        *[f"greedy_right_{n}_of_5" for n in range(6)],
        *[f"greedy_wrong_{n}_of_5" for n in range(6)],
    ],
)
def test_label_from_counts_over_every_case(
    greedy: bool, n_correct: int, expected: str
) -> None:
    assert label_from_counts(greedy, n_correct, n_samples=5) == expected


def test_label_from_counts_threshold_is_a_keyword() -> None:
    assert label_from_counts(True, 3, n_samples=5, known_min_correct=3) == (
        "known"
    )
    assert label_from_counts(True, 4, n_samples=5, known_min_correct=5) == (
        "uncertain"
    )


@pytest.mark.parametrize(
    ("n_correct", "expected"),
    [(3, "uncertain"), (4, "known")],
    ids=["three_of_five_is_uncertain", "four_of_five_is_known"],
)
def test_label_from_counts_robust_threshold_of_four(
    n_correct: int, expected: str
) -> None:
    assert (
        label_from_counts(True, n_correct, n_samples=5, known_min_correct=4)
        == expected
    )


@pytest.mark.parametrize(
    ("n_correct", "n_samples", "known_min_correct"),
    [(-1, 5, 4), (6, 5, 4), (0, 5, 0), (0, 5, 6)],
    ids=["negative_count", "count_above_samples", "min_zero", "min_above"],
)
def test_label_from_counts_rejects_impossible_arguments(
    n_correct: int, n_samples: int, known_min_correct: int
) -> None:
    with pytest.raises(ValueError, match="must be in"):
        label_from_counts(
            True,
            n_correct,
            n_samples=n_samples,
            known_min_correct=known_min_correct,
        )


def test_label_recipe_defaults_match_the_frozen_rule() -> None:
    recipe = LabelRecipe()

    assert (recipe.n_samples, recipe.temperature) == (5, 1.0)
    assert (recipe.known_min_correct, recipe.max_new_tokens) == (3, 16)
    assert recipe.stop == "\n"


def test_records_are_frozen() -> None:
    item = ItemLabel("q1", "known", "behavior", "Paris", True, ("Paris",), 1)

    with pytest.raises(dataclasses.FrozenInstanceError):
        item.label = "unknown"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        LabelRecipe().n_samples = 3  # type: ignore[misc]


def _items(
    labels: tuple[Label, ...],
) -> tuple[list[QAExample], list[ItemLabel]]:
    examples = [
        QAExample(f"q{i}", f"Question {i}?", (f"a{i}", f"b{i}"), "real", "test")
        for i in range(len(labels))
    ]
    item_labels = [
        ItemLabel(f"q{i}", label, "behavior", f"g{i}", False, ("x", "y"), 0)
        for i, label in enumerate(labels)
    ]
    return examples, item_labels


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_write_hand_check_columns_and_cells(tmp_path: Path) -> None:
    examples, labels = _items(("known", "unknown"))
    path = tmp_path / "check.csv"

    written = write_hand_check(path, examples, labels, n=10)

    rows = _read(path)
    assert written == 2
    assert list(rows[0]) == [
        "uid",
        "kind",
        "question",
        "gold",
        "greedy",
        "samples",
        "label",
        "human_label",
    ]
    assert rows[0] == {
        "uid": "q0",
        "kind": "real",
        "question": "Question 0?",
        "gold": "a0 | b0",
        "greedy": "g0",
        "samples": "x | y",
        "label": "known",
        "human_label": "",
    }


def test_write_hand_check_round_robin_over_labels(tmp_path: Path) -> None:
    labels: tuple[Label, ...] = (
        ("known",) * 6 + ("unknown",) * 6 + ("uncertain",) * 2
    )
    examples, item_labels = _items(labels)
    path = tmp_path / "check.csv"

    written = write_hand_check(path, examples, item_labels, n=7)

    counts = {
        label: sum(row["label"] == label for row in _read(path))
        for label in ("known", "unknown", "uncertain")
    }
    assert written == 7
    assert counts == {"known": 3, "unknown": 2, "uncertain": 2}


def test_write_hand_check_is_deterministic_per_seed(tmp_path: Path) -> None:
    examples, labels = _items(("known", "unknown") * 10)

    write_hand_check(tmp_path / "a.csv", examples, labels, n=6, seed=1)
    write_hand_check(tmp_path / "b.csv", examples, labels, n=6, seed=1)
    write_hand_check(tmp_path / "c.csv", examples, labels, n=6, seed=2)

    a = (tmp_path / "a.csv").read_text(encoding="utf-8")
    assert a == (tmp_path / "b.csv").read_text(encoding="utf-8")
    assert a != (tmp_path / "c.csv").read_text(encoding="utf-8")


def test_write_hand_check_rows_keep_the_order_of_examples(
    tmp_path: Path,
) -> None:
    examples, labels = _items(("known", "unknown", "uncertain") * 5)
    path = tmp_path / "check.csv"

    write_hand_check(path, examples, labels, n=9)

    uids = [int(row["uid"][1:]) for row in _read(path)]
    assert uids == sorted(uids)


def test_write_hand_check_n_above_item_count_returns_all(
    tmp_path: Path,
) -> None:
    examples, labels = _items(("known", "known", "unknown"))
    path = tmp_path / "check.csv"

    assert write_hand_check(path, examples, labels, n=200) == 3
    assert [row["uid"] for row in _read(path)] == ["q0", "q1", "q2"]


def test_write_hand_check_missing_label_gives_its_quota_away(
    tmp_path: Path,
) -> None:
    examples, labels = _items(("known",) * 5 + ("unknown",) * 2)
    path = tmp_path / "check.csv"

    written = write_hand_check(path, examples, labels, n=6)

    unknown = [row for row in _read(path) if row["label"] == "unknown"]
    assert written == 6
    assert len(unknown) == 2


def test_write_hand_check_rejects_unpaired_labels(tmp_path: Path) -> None:
    examples, labels = _items(("known", "unknown"))

    with pytest.raises(ValueError, match="pair up"):
        write_hand_check(tmp_path / "a.csv", examples, labels[:1])
    with pytest.raises(ValueError, match="pair up"):
        write_hand_check(tmp_path / "b.csv", examples, labels[::-1])


def _fake_generate(
    greedy: dict[str, str], sampled: dict[str, list[str]], calls: list[Any]
) -> Any:
    def question_of(prompt: str) -> str:
        return prompt.rsplit("Q: ", 1)[1].split("\n")[0]

    def fake(
        loaded: Any,
        prompts: list[str],
        *,
        temperature: float | None = None,
        num_samples: int = 1,
        **kwargs: Any,
    ) -> list[list[Completion]]:
        calls.append(
            {
                "prompts": list(prompts),
                "temperature": temperature,
                "num_samples": num_samples,
                **kwargs,
            }
        )
        if temperature is None:
            texts = [[greedy[question_of(p)]] for p in prompts]
        else:
            texts = [sampled[question_of(p)] for p in prompts]
        return [[Completion(t, (), True) for t in row] for row in texts]

    return fake


def _qa(uid: str, answers: tuple[str, ...]) -> QAExample:
    return QAExample(uid, f"Question {uid}?", answers, "real", "test")


def test_label_questions_known_unknown_uncertain_and_order(
    monkeypatch: pytest.MonkeyPatch, tiny_gpt2: LoadedModel
) -> None:
    calls: list[Any] = []
    greedy = {
        "Question k?": " Paris\nQ: more",
        "Question n?": " Rome",
        "Question u?": " Paris",
    }
    sampled = {
        "Question k?": [" Paris"] * 4 + [" Rome"],
        "Question n?": [" Rome", " Oslo", " Bern", " Rome", " Rome"],
        "Question u?": [" Paris", " Rome", " Oslo", " Rome", " Paris"],
    }
    monkeypatch.setattr(
        "tomography.knowledge.generate", _fake_generate(greedy, sampled, calls)
    )
    examples = [
        _qa("u", ("Paris",)),
        _qa("none", ()),
        _qa("k", ("Paris",)),
        _qa("n", ("Paris",)),
    ]

    labels = label_questions(tiny_gpt2, examples, shots=SHOTS, seed=7)

    assert [(x.uid, x.label, x.basis) for x in labels] == [
        ("u", "uncertain", "behavior"),
        ("none", "unknown", "construction"),
        ("k", "known", "behavior"),
        ("n", "unknown", "behavior"),
    ]
    assert labels[2].greedy == "Paris"
    assert labels[2].greedy_correct is True
    assert labels[2].samples == ("Paris",) * 4 + ("Rome",)
    assert labels[2].n_correct == 4
    assert labels[1] == ItemLabel(
        "none", "unknown", "construction", "", False, (), 0
    )
    assert labels[0].n_correct == 2


@pytest.mark.parametrize(
    ("n_right", "expected"),
    [(3, "known"), (2, "uncertain")],
    ids=["three_of_five_is_known", "two_of_five_is_uncertain"],
)
def test_label_questions_default_recipe_needs_three_of_five(
    monkeypatch: pytest.MonkeyPatch,
    tiny_gpt2: LoadedModel,
    n_right: int,
    expected: str,
) -> None:
    calls: list[Any] = []
    sampled = {"Question a?": [" Paris"] * n_right + [" Rome"] * (5 - n_right)}
    monkeypatch.setattr(
        "tomography.knowledge.generate",
        _fake_generate({"Question a?": " Paris"}, sampled, calls),
    )

    (item,) = label_questions(tiny_gpt2, [_qa("a", ("Paris",))], shots=SHOTS)

    assert item.greedy_correct is True
    assert item.n_correct == n_right
    assert item.label == expected


def test_label_questions_makes_two_calls_greedy_first(
    monkeypatch: pytest.MonkeyPatch, tiny_gpt2: LoadedModel
) -> None:
    calls: list[Any] = []
    table = {"Question a?": " x", "Question b?": " x"}
    samples = {k: [" x"] * 5 for k in table}
    monkeypatch.setattr(
        "tomography.knowledge.generate", _fake_generate(table, samples, calls)
    )
    examples = [_qa("a", ("y",)), _qa("none", ()), _qa("b", ("y",))]

    label_questions(tiny_gpt2, examples, shots=SHOTS, seed=3, batch_size=2)

    prompts = [few_shot_prompt("Question a?", SHOTS)]
    prompts.append(few_shot_prompt("Question b?", SHOTS))
    assert len(calls) == 2
    assert calls[0] == {
        "prompts": prompts,
        "temperature": None,
        "num_samples": 1,
        "max_new_tokens": 16,
        "batch_size": 2,
    }
    assert calls[1] == {
        "prompts": prompts,
        "temperature": 1.0,
        "num_samples": 5,
        "seed": 3,
        "max_new_tokens": 16,
        "batch_size": 2,
    }


def test_label_questions_without_gold_answers_makes_no_call(
    monkeypatch: pytest.MonkeyPatch, tiny_gpt2: LoadedModel
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(
        "tomography.knowledge.generate", _fake_generate({}, {}, calls)
    )

    labels = label_questions(tiny_gpt2, [_qa("a", ()), _qa("b", ())])
    empty = label_questions(tiny_gpt2, [])

    assert calls == []
    assert [x.basis for x in labels] == ["construction", "construction"]
    assert empty == []


@pytest.mark.parametrize(
    "examples",
    [[_qa("a", ("y",))], [_qa("a", ()), _qa("b", ())], []],
    ids=["with_gold_answers", "without_gold_answers", "no_examples"],
)
def test_label_questions_rejects_a_bad_recipe_before_generating(
    monkeypatch: pytest.MonkeyPatch,
    tiny_gpt2: LoadedModel,
    examples: list[QAExample],
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(
        "tomography.knowledge.generate", _fake_generate({}, {}, calls)
    )
    recipe = LabelRecipe(n_samples=3, known_min_correct=4)

    with pytest.raises(ValueError, match="known_min_correct"):
        label_questions(tiny_gpt2, examples, recipe=recipe, shots=SHOTS)

    assert calls == []


def test_label_questions_uses_the_recipe(
    monkeypatch: pytest.MonkeyPatch, tiny_gpt2: LoadedModel
) -> None:
    calls: list[Any] = []
    table = {"Question a?": " y."}
    monkeypatch.setattr(
        "tomography.knowledge.generate",
        _fake_generate(table, {"Question a?": [" y. z", " y. z"]}, calls),
    )
    recipe = LabelRecipe(
        n_samples=2,
        temperature=0.7,
        known_min_correct=2,
        max_new_tokens=4,
        stop=".",
    )

    (item,) = label_questions(
        tiny_gpt2, [_qa("a", ("y",))], recipe=recipe, shots=SHOTS
    )

    assert item.label == "known"
    assert item.greedy == "y"
    assert item.samples == ("y", "y")
    assert calls[0]["max_new_tokens"] == 4
    assert calls[1]["temperature"] == 0.7
    assert calls[1]["num_samples"] == 2
    assert calls[1]["max_new_tokens"] == 4


def test_label_questions_keeps_items_with_a_repeated_uid_apart(
    monkeypatch: pytest.MonkeyPatch, tiny_gpt2: LoadedModel
) -> None:
    calls: list[Any] = []
    greedy = {"Question a?": " x", "Question b?": " x"}
    sampled = {"Question a?": [" x"] * 5, "Question b?": [" y"] * 5}
    monkeypatch.setattr(
        "tomography.knowledge.generate", _fake_generate(greedy, sampled, calls)
    )
    first = QAExample("same", "Question a?", ("x",), "real", "test")
    second = QAExample("same", "Question b?", ("x",), "real", "test")

    labels = label_questions(tiny_gpt2, [first, second], shots=SHOTS)

    assert [x.label for x in labels] == ["known", "uncertain"]


def test_label_questions_plumbing_on_a_tiny_model(
    tiny_arch: LoadedModel,
) -> None:
    tiny_arch.model.generation_config.eos_token_id = None
    examples = [_qa("a", ("the",)), _qa("none", ()), _qa("b", ("fox", "dog"))]
    recipe = LabelRecipe(n_samples=3, known_min_correct=2, max_new_tokens=4)

    def run(seed: int) -> list[ItemLabel]:
        return label_questions(
            tiny_arch,
            examples,
            recipe=recipe,
            shots=SHOTS[:1],
            seed=seed,
            batch_size=1,
        )

    first = run(0)

    assert [x.uid for x in first] == ["a", "none", "b"]
    for item in (first[0], first[2]):
        assert item.basis == "behavior"
        assert len(item.samples) == 3
        assert 0 <= item.n_correct <= 3
    assert first[1].basis == "construction"
    assert first == run(0)


def test_label_questions_default_shots_exceed_a_tiny_model(
    tiny_gpt2: LoadedModel,
) -> None:
    with pytest.raises(ValueError, match="positions of this model"):
        label_questions(tiny_gpt2, [_qa("a", ("x",))])


@pytest.mark.slow
def test_gpt2_labels_popqa_style_items() -> None:
    gpt2 = load_model("gpt2")
    facts = [
        ("What is the capital of France?", ("Paris",)),
        ("What is the capital of Italy?", ("Rome",)),
        ("What is the capital of Spain?", ("Madrid",)),
        ("What is the capital of Germany?", ("Berlin",)),
        ("What is the capital of Japan?", ("Tokyo",)),
        ("What is the capital of Egypt?", ("Cairo",)),
        ("What is the capital of Russia?", ("Moscow",)),
        ("What is the capital of Greece?", ("Athens",)),
        ("What is the capital of Norway?", ("Oslo",)),
        ("What is the capital of Peru?", ("Lima",)),
        ("Who wrote Hamlet?", ("Shakespeare", "William Shakespeare")),
        ("Who painted the Mona Lisa?", ("Leonardo da Vinci",)),
        ("What is the capital of Malta?", ("Valletta",)),
        ("What is the capital of Laos?", ("Vientiane",)),
        ("What is the capital of Bhutan?", ("Thimphu",)),
        ("What is the capital of Nauru?", ("Yaren",)),
        ("What is the capital of Tuvalu?", ("Funafuti",)),
        ("What is the capital of Palau?", ("Ngerulmud",)),
        ("What is the capital of Brunei?", ("Bandar Seri Begawan",)),
        ("What is the capital of Eswatini?", ("Mbabane",)),
    ]
    examples = [
        QAExample(f"p{i}", question, answers, "real", "test")
        for i, (question, answers) in enumerate(facts)
    ]

    labels = label_questions(gpt2, examples, seed=0)

    counts = {
        name: sum(x.label == name for x in labels)
        for name in ("known", "unknown", "uncertain")
    }
    print(counts)
    assert len(labels) == 20
    assert sum(counts.values()) == 20
