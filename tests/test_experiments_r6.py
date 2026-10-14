"""Tests for the R6 TruthfulQA experiment."""

import numpy as np
import pytest
import torch

from tomography.datasets.records import ChoiceExample
from tomography.experiments import r6_truthfulqa
from tomography.experiments.r6_truthfulqa import HEURISTIC_LEVELS
from tomography.experiments.r6_truthfulqa import LAYER_RULES
from tomography.experiments.r6_truthfulqa import best_layer_accuracy
from tomography.experiments.r6_truthfulqa import heuristic_scores
from tomography.experiments.r6_truthfulqa import standard_scores
from tomography.experiments.r6_truthfulqa import validation_layer_accuracy
from tomography.hooks import edit_residuals
from tomography.judges import heuristic_score
from tomography.judges import sequence_logprob
from tomography.model import LoadedModel

SHORT_TEMPLATE = "T {scenario} A:"


def _items() -> list[ChoiceExample]:
    return [
        ChoiceExample(
            uid="a", question="q", choices=("x", "yy"), correct=0, split="val"
        ),
        ChoiceExample(
            uid="b",
            question="r",
            choices=("z", "w", "vv"),
            correct=2,
            split="val",
        ),
    ]


def _bos(loaded: LoadedModel) -> int | None:
    return loaded.bos_id if loaded.default_bos == "prepend" else None


def test_standard_scores_sum_the_answer_logprob_after_the_user_prompt(
    tiny_arch: LoadedModel,
) -> None:
    items = _items()
    got = standard_scores(tiny_arch, items, batch_size=4)
    assert [g.shape for g in got] == [(2,), (3,)]
    assert all(g.dtype == np.float32 for g in got)
    for item, scores in zip(items, got, strict=True):
        expected = sequence_logprob(
            tiny_arch.model,
            tiny_arch.tokenizer,
            [f"USER: {item.question} ASSISTANT:"] * len(item.choices),
            [" " + choice for choice in item.choices],
            batch_size=4,
            bos_id=_bos(tiny_arch),
        )
        np.testing.assert_allclose(scores, expected, rtol=1e-4, atol=1e-4)


def test_standard_scores_are_deterministic(tiny_arch: LoadedModel) -> None:
    first = standard_scores(tiny_arch, _items())
    second = standard_scores(tiny_arch, _items())
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_standard_scores_run_under_the_callers_hooks(
    tiny_arch: LoadedModel,
) -> None:
    plain = standard_scores(tiny_arch, _items())

    def shift(hidden: torch.Tensor, info: object) -> torch.Tensor:
        return hidden + 3.0

    with edit_residuals(tiny_arch, {1: shift}):
        edited = standard_scores(tiny_arch, _items())
    assert not np.array_equal(np.concatenate(plain), np.concatenate(edited))


def test_the_heuristic_levels_run_from_low_to_high() -> None:
    assert HEURISTIC_LEVELS == (
        "Very low",
        "Low",
        "Somewhat low",
        "Neutral",
        "Somewhat high",
        "High",
        "Very high",
    )


def test_the_heuristic_template_has_one_scenario_slot() -> None:
    template = r6_truthfulqa.HEURISTIC_TEMPLATE
    assert template.count("{scenario}") == 1
    assert template.startswith(
        "USER: What is the amount of truthfulness of the following scenario?"
    )
    assert "'Very low', 'Low'" in template
    assert template.endswith("Assistant: Truthfulness:")


def test_heuristic_scores_map_the_seven_level_logprobs(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r6_truthfulqa, "HEURISTIC_TEMPLATE", SHORT_TEMPLATE)
    items = _items()
    got = heuristic_scores(tiny_arch, items, batch_size=8)
    assert [g.shape for g in got] == [(2,), (3,)]
    for item, scores in zip(items, got, strict=True):
        assert scores.dtype == np.float64
        assert ((scores >= -1.0) & (scores <= 1.0)).all()
        for choice, score in zip(item.choices, scores, strict=True):
            scenario = f"Question: {item.question}\nAnswer: {choice}"
            prompt = SHORT_TEMPLATE.replace("{scenario}", scenario)
            logprobs = sequence_logprob(
                tiny_arch.model,
                tiny_arch.tokenizer,
                [prompt] * 7,
                [" " + level for level in HEURISTIC_LEVELS],
                batch_size=8,
                bos_id=_bos(tiny_arch),
            )
            assert score == pytest.approx(
                float(heuristic_score(logprobs)), abs=1e-4
            )


def test_heuristic_scores_accept_a_template_argument_and_braces(
    tiny_arch: LoadedModel,
) -> None:
    item = ChoiceExample(
        uid="c", question="{q}", choices=("x", "y"), correct=1, split="val"
    )
    got = heuristic_scores(tiny_arch, [item], template=SHORT_TEMPLATE)
    assert got[0].shape == (2,)


def test_heuristic_scores_are_deterministic(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r6_truthfulqa, "HEURISTIC_TEMPLATE", SHORT_TEMPLATE)
    first = heuristic_scores(tiny_arch, _items())
    second = heuristic_scores(tiny_arch, _items())
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


VAL = np.array(
    [
        [0.1, 0.5, 0.5, 0.2],
        [0.9, 0.3, 0.9, 0.1],
        [0.2, 0.3, 0.4, 0.8],
    ]
)
TEST = np.array(
    [
        [0.6, 0.7, 0.9, 0.1],
        [0.5, 0.8, 0.2, 0.3],
        [0.4, 0.3, 0.6, 0.9],
    ]
)
LAYERS = (4, 6, 8, 10)


def test_the_layer_rules_are_named_in_table_order() -> None:
    assert LAYER_RULES == ("validation-layer", "best-layer")


def test_validation_layer_reports_the_test_accuracy_at_each_trials_layer() -> (
    None
):
    accuracy, chosen = validation_layer_accuracy(VAL, TEST, LAYERS)
    assert chosen.tolist() == [6, 4, 10]
    np.testing.assert_allclose(accuracy, [0.7, 0.5, 0.9])


def test_best_layer_uses_one_layer_for_every_trial() -> None:
    accuracy, layer = best_layer_accuracy(TEST, LAYERS)
    assert layer == 6
    np.testing.assert_allclose(accuracy, [0.7, 0.8, 0.3])


def test_best_layer_breaks_ties_to_the_first_layer() -> None:
    tied = np.array([[0.5, 0.75, 0.75], [0.5, 0.25, 0.25]])
    accuracy, layer = best_layer_accuracy(tied, (3, 5, 7))
    assert layer == 3
    np.testing.assert_allclose(accuracy, [0.5, 0.5])


@pytest.mark.parametrize(
    ("val", "test"),
    [(VAL[:2], TEST), (VAL[:, :3], TEST), (VAL[0], TEST[0])],
    ids=["trials", "layers", "ndim"],
)
def test_validation_layer_rejects_mismatched_cubes(
    val: np.ndarray, test: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="shape"):
        validation_layer_accuracy(val, test, LAYERS)


def test_best_layer_rejects_a_layer_count_mismatch() -> None:
    with pytest.raises(ValueError, match="layers"):
        best_layer_accuracy(TEST, (1, 2, 3))
