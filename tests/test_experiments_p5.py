"""Tests for the P5 persona wording experiment."""

import dataclasses
from typing import Any

import numpy as np
import pytest

from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.experiments.base import derive_seed
from tomography.experiments.p5_persona_wording import CONDITIONS
from tomography.experiments.p5_persona_wording import CONTROLS
from tomography.experiments.p5_persona_wording import GOT_SUBSETS
from tomography.experiments.p5_persona_wording import MAX_CONDITIONS
from tomography.experiments.p5_persona_wording import PERSONA_PAIRS
from tomography.experiments.p5_persona_wording import PLAIN
from tomography.experiments.p5_persona_wording import PLAIN_POSITION
from tomography.experiments.p5_persona_wording import Config
from tomography.experiments.p5_persona_wording import condition_position
from tomography.experiments.p5_persona_wording import condition_stimuli
from tomography.experiments.p5_persona_wording import persona_template
from tomography.experiments.p5_persona_wording import plain_stimuli
from tomography.experiments.sweep import shuffle_within_groups
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat
from tomography.templates import ConceptTemplate
from tomography.templates import FunctionTemplate

WORDS = {
    "paper": ("an honest", "an untruthful"),
    "synonym_0": ("a truthful", "a deceitful"),
    "synonym_1": ("a sincere", "a lying"),
    "unrelated_0": ("a tall", "a short"),
    "unrelated_1": ("an old", "a young"),
}


PLAIN_CHAT = ChatFormat("plain")
PERSONA_CONDITIONS = (*WORDS, "random_label")


def _statements(
    count: int = 12, splits: tuple[Split, ...] = ("train", "val", "test")
) -> list[Example]:
    examples = []
    for split in splits:
        for r in range(count):
            body = "plain and long " * (r % 4 + 1)
            examples.append(
                Example(
                    uid=f"fact-{split}-{r}",
                    text=f"The fact number {r} is {body}",
                    label=1,
                    split=split,
                )
            )
            examples.append(
                Example(
                    uid=f"fact-{split}-{r}-false",
                    text=f"The claim number {r} is odd and {body}",
                    label=0,
                    split=split,
                )
            )
    return examples


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        n_seeds=3,
        n_pairs=8,
        n_eval_pairs=4,
        got_subsets=("cities", "larger_than"),
        n_boot=50,
    )
    return dataclasses.replace(base, **overrides)


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "p5"
    assert config.model == "gpt2"
    assert config.method == "logistic"
    assert config.position == "response"
    assert config.n_eval_pairs == 256
    assert config.conditions == CONDITIONS
    assert config.got_subsets == GOT_SUBSETS
    assert config.n_boot == 2000
    assert Config.from_dict(config.to_dict()) == config


def test_the_conditions_are_the_personas_then_the_controls() -> None:
    assert dict(PERSONA_PAIRS) == WORDS
    assert CONTROLS == ("instruction_only", "fact_only", "random_label")
    assert (*WORDS, *CONTROLS) == CONDITIONS
    assert len(CONDITIONS) == MAX_CONDITIONS == 8
    assert ConceptTemplate("honesty_plain", "{stimulus}", tagged=False) == PLAIN
    assert PLAIN_POSITION == "mean"
    assert GOT_SUBSETS == (
        "cities",
        "neg_cities",
        "sp_en_trans",
        "larger_than",
        "common_claim_true_false",
    )


def test_the_paper_persona_words_are_those_of_the_paper_template() -> None:
    paper = PAPER_TEMPLATES["honesty"]
    assert isinstance(paper, FunctionTemplate)
    assert PERSONA_PAIRS["paper"] == (
        paper.pos_personas[0],
        paper.neg_personas[0],
    )


@pytest.mark.parametrize("condition", sorted(WORDS))
def test_a_persona_template_changes_only_the_persona_words(
    condition: str,
) -> None:
    paper = PAPER_TEMPLATES["honesty"]
    template = persona_template(condition)
    pos, neg = WORDS[condition]
    assert template.name == f"honesty_{condition}"
    assert template.pos_personas == (pos,)
    assert template.neg_personas == (neg,)
    assert template.user == paper.user
    assert template.tagged == paper.tagged


@pytest.mark.parametrize("condition", [*CONTROLS, "shouting"])
def test_a_persona_template_exists_for_persona_conditions_only(
    condition: str,
) -> None:
    with pytest.raises(ValueError, match=condition):
        persona_template(condition)


def test_plain_stimuli_pair_true_against_false_statements() -> None:
    statements = _statements(count=6, splits=("train",))
    stimuli = plain_stimuli(statements, n_pairs=4, seed=1)
    texts = {e.text for e in statements}
    assert set(stimuli.texts) <= texts
    assert stimuli.groups.shape == (4, 2)
    assert stimuli.labels.shape == (8,)
    for group in stimuli.groups:
        assert sorted(stimuli.labels[group]) == [0, 1]


def test_plain_stimuli_cap_the_pairs_and_pass_none_through() -> None:
    statements = [
        e
        for e in _statements(count=6, splits=("train",))
        if e.label == 1 or e.uid.endswith(("-0-false", "-1-false", "-2-false"))
    ]
    assert plain_stimuli(statements, n_pairs=10, seed=1).groups.shape[0] == 3
    assert plain_stimuli(statements, n_pairs=2, seed=1).groups.shape[0] == 2
    assert plain_stimuli(statements, n_pairs=None, seed=1).groups.shape[0] == 3


@pytest.mark.parametrize("label", [0, 1])
def test_plain_stimuli_need_both_truth_values(label: int) -> None:
    statements = [e for e in _statements() if e.label == label]
    with pytest.raises(ValueError, match="true and false"):
        plain_stimuli(statements, n_pairs=4, seed=1)


def _persona(condition: str, **kwargs: Any) -> Any:
    settings: dict[str, Any] = {"n_pairs": 8, "seed": 3, "train": True}
    settings.update(kwargs)
    return condition_stimuli(condition, _statements(), PLAIN_CHAT, **settings)


def test_persona_conditions_draw_the_same_true_statements() -> None:
    reference = _persona("paper")
    true_uids = {e.uid for e in _statements() if e.label == 1}
    assert set(reference.uids) <= true_uids
    assert len(set(reference.uids)) == 8
    for condition in PERSONA_CONDITIONS:
        assert _persona(condition).uids == reference.uids
    assert _persona("paper", seed=4).uids != reference.uids


@pytest.mark.parametrize("condition", sorted(WORDS))
def test_a_persona_condition_renders_its_own_words(condition: str) -> None:
    stimuli = _persona(condition)
    pos, neg = WORDS[condition]
    for text, label in zip(stimuli.texts, stimuli.labels, strict=True):
        assert (pos if label == 1 else neg) in text
        assert (neg if label == 1 else pos) not in text
    assert stimuli.groups.shape == (8, 2)


def test_random_label_shuffles_the_training_labels_within_pairs() -> None:
    real = _persona("paper", n_pairs=30)
    shuffled = _persona("random_label", n_pairs=30)
    assert shuffled.texts == real.texts
    expected = shuffle_within_groups(
        real.labels, real.groups, seed=derive_seed(3, "random_label")
    )
    np.testing.assert_array_equal(shuffled.labels, expected)
    assert (shuffled.labels != real.labels).any()
    for group in shuffled.groups:
        assert sorted(shuffled.labels[group]) == [0, 1]
    evaluation = _persona("random_label", n_pairs=30, train=False)
    np.testing.assert_array_equal(evaluation.labels, real.labels)


def test_instruction_only_is_the_persona_instruction_alone() -> None:
    stimuli = _persona("instruction_only")
    reference = _persona("paper")
    assert stimuli.uids == reference.uids
    assert len(set(stimuli.texts)) == 2
    assert all(
        span is not None and span[0] == span[1]
        for span in stimuli.response_spans
    )
    assert not any(
        e.text in text for e in _statements() for text in stimuli.texts
    )
    for text, label in zip(stimuli.texts, stimuli.labels, strict=True):
        assert ("honest" if label == 1 else "untruthful") in text
    np.testing.assert_array_equal(stimuli.labels, reference.labels)


def test_fact_only_trains_on_plain_true_against_false_statements() -> None:
    stimuli = _persona("fact_only")
    expected = plain_stimuli(_statements(), n_pairs=8, seed=3)
    assert stimuli.texts == expected.texts
    assert stimuli.uids == expected.uids
    np.testing.assert_array_equal(stimuli.labels, expected.labels)
    assert {e.label for e in _statements() if e.uid in stimuli.uids} == {0, 1}
    assert not any(
        word in text for text in stimuli.texts for word in ("honest", "USER")
    )


def test_an_unknown_condition_raises() -> None:
    with pytest.raises(ValueError, match="shouting"):
        _persona("shouting")


@pytest.mark.parametrize("position", ["last", "response", "mean", "concept"])
def test_condition_positions(position: Any) -> None:
    assert condition_position("instruction_only", position) == "last"
    assert condition_position("fact_only", position) == "mean"
    for condition in (*WORDS, "random_label"):
        assert condition_position(condition, position) == position
