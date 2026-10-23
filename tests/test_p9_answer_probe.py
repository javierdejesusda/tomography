"""Tests of the exact-answer-token probe (`tomography.p9.answer_probe`)."""

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.activations import Activations
from tomography.activations import collect_activations
from tomography.model import LoadedModel
from tomography.p9 import answer_probe
from tomography.p9.answer_probe import answer_activations
from tomography.p9.answer_probe import answer_texts
from tomography.p9.answer_probe import fit_answer_probe
from tomography.reading.metrics import auroc
from tomography.reading.reader import RepReader

PROMPTS = [
    "Q: who wrote it?\nA:",
    "Q: where?\nA:",
    "Q: which city is the capital of the country in the north?\nA:",
    "Q: when?\nA:",
]
ANSWERS = ["Paris", "a vehicle", "an animal in the north", "1990"]
SHOTS = "Q: what is a cat?\nA: an animal\n\nQ: what is a car?\nA: a vehicle\n\n"


def _planted(
    n: int = 240, d: int = 8, *, signal_layer: int = 1, seed: int = 0
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Three layers of noise; only `signal_layer` carries the label."""
    rng = np.random.default_rng(seed)
    correct = (rng.random(n) < 0.5).astype(np.int64)
    h = rng.standard_normal((3, n, d))
    direction = rng.standard_normal(d)
    direction /= np.linalg.norm(direction)
    h[signal_layer] += 3.0 * np.outer(2 * correct - 1, direction)
    return h.astype(np.float32), correct


def _masks(n: int) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    index = np.arange(n)
    return index % 3 == 0, index % 3 == 1


def test_answer_texts_append_a_space_and_span_the_answer() -> None:
    texts, spans = answer_texts(["Q: x\nA:", "Q: y\nA:"], ["Paris", "a b"])

    assert texts == ["Q: x\nA: Paris", "Q: y\nA: a b"]
    assert [t[s:e] for t, (s, e) in zip(texts, spans, strict=True)] == [
        "Paris",
        "a b",
    ]


def test_answer_texts_reject_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="one answer per prompt"):
        answer_texts(["Q: x\nA:"], ["a", "b"])


def test_the_read_token_is_the_last_token_of_the_spaced_answer(
    tiny_arch: LoadedModel,
) -> None:
    texts, spans = answer_texts(PROMPTS, ANSWERS)

    acts = collect_activations(
        tiny_arch, texts, layers=[1], position="concept", concept_spans=spans
    )

    expected = [
        tiny_arch.tokenizer(" " + a, add_special_tokens=False)["input_ids"][-1]
        for a in ANSWERS
    ]
    assert acts.row_token_id.tolist() == expected


def test_activations_equal_the_last_token_of_prompt_and_answer(
    tiny_arch: LoadedModel,
) -> None:
    h = answer_activations(tiny_arch, PROMPTS, ANSWERS, layers=[0, 2])

    last = collect_activations(
        tiny_arch,
        [f"{p} {a}" for p, a in zip(PROMPTS, ANSWERS, strict=True)],
        layers=[0, 2],
    )
    assert h.shape == last.data.shape
    np.testing.assert_allclose(h, last.data, rtol=1e-4, atol=1e-3)


def test_an_empty_answer_gives_a_nan_row(tiny_arch: LoadedModel) -> None:
    answers = ["Paris", "", "an animal", ""]

    h = answer_activations(tiny_arch, PROMPTS, answers, layers=[1])

    assert np.isnan(h[:, [1, 3]]).all()
    full = answer_activations(
        tiny_arch, [PROMPTS[0], PROMPTS[2]], ["Paris", "an animal"], layers=[1]
    )
    np.testing.assert_allclose(h[:, [0, 2]], full, rtol=1e-4, atol=1e-3)


def test_all_empty_answers_give_only_nan_rows(tiny_arch: LoadedModel) -> None:
    h = answer_activations(tiny_arch, PROMPTS[:2], ["", ""], layers=[1, 2])

    assert h.shape == (2, 2, tiny_arch.d_model)
    assert np.isnan(h).all()


def test_activations_do_not_depend_on_the_batch_size(
    tiny_arch: LoadedModel,
) -> None:
    single = answer_activations(
        tiny_arch, PROMPTS, ANSWERS, layers=[1], batch_size=1
    )
    batched = answer_activations(
        tiny_arch, PROMPTS, ANSWERS, layers=[1], batch_size=4
    )

    np.testing.assert_allclose(single, batched, rtol=1e-4, atol=1e-3)


def test_the_shared_prefix_matches_the_default_path(
    tiny_arch: LoadedModel,
) -> None:
    prompts = [SHOTS + p for p in PROMPTS]

    default = answer_activations(tiny_arch, prompts, ANSWERS, layers=[1, 3])
    shared = answer_activations(
        tiny_arch, prompts, ANSWERS, layers=[1, 3], share_prefix=True
    )

    np.testing.assert_allclose(shared, default, rtol=1e-4, atol=1e-3)


def test_the_probe_picks_the_planted_layer_on_validation() -> None:
    h, correct = _planted(signal_layer=1)
    train, val = _masks(len(correct))

    probe = fit_answer_probe(h, (4, 7, 9), correct, train=train, val=val)

    assert probe.layer == 7
    assert probe.val_auroc.shape == (3,)
    assert probe.val_auroc[1] > 0.95
    assert probe.val_auroc[1] == probe.val_auroc.max()
    test = ~(train | val)
    assert auroc(probe.scores[test], correct[test]) > 0.95


def test_the_probe_is_the_p9c_logistic_probe_at_the_chosen_layer() -> None:
    h, correct = _planted(signal_layer=2, seed=1)
    train, val = _masks(len(correct))

    probe = fit_answer_probe(h, (1, 2, 3), correct, train=train, val=val)

    p9c = RepReader.fit(
        "logistic", h[2:3, train], (probe.layer,), labels=correct[train]
    )
    assert probe.layer == 3
    np.testing.assert_array_equal(probe.scores, p9c.score(h[2:3])[0])


def test_the_validation_curve_is_the_auroc_of_the_train_fit() -> None:
    h, correct = _planted(seed=2)
    train, val = _masks(len(correct))

    probe = fit_answer_probe(h, (0, 1, 2), correct, train=train, val=val)

    fitted = RepReader.fit(
        "logistic", h[:, train], (0, 1, 2), labels=correct[train]
    )
    expected = auroc(fitted.score(h)[:, val].astype(np.float64), correct[val])
    np.testing.assert_array_equal(probe.val_auroc, expected)


def test_scores_are_oriented_so_that_higher_means_correct() -> None:
    h, correct = _planted(seed=3)
    train, val = _masks(len(correct))

    probe = fit_answer_probe(h, (0, 1, 2), correct, train=train, val=val)

    assert probe.scores[correct == 1].mean() > probe.scores[correct == 0].mean()


def test_the_probe_is_deterministic() -> None:
    h, correct = _planted(seed=4)
    train, val = _masks(len(correct))

    first = fit_answer_probe(h, (0, 1, 2), correct, train=train, val=val)
    second = fit_answer_probe(h, (0, 1, 2), correct, train=train, val=val)

    np.testing.assert_array_equal(first.scores, second.scores)
    np.testing.assert_array_equal(first.val_auroc, second.val_auroc)


def test_nan_rows_are_left_out_of_the_fit_and_scored_nan() -> None:
    h, correct = _planted(seed=5)
    train, val = _masks(len(correct))
    missing = np.arange(len(correct)) % 10 == 0
    with_nan = h.copy()
    with_nan[:, missing] = np.nan

    probe = fit_answer_probe(with_nan, (0, 1, 2), correct, train=train, val=val)

    reference = fit_answer_probe(
        h[:, ~missing],
        (0, 1, 2),
        correct[~missing],
        train=train[~missing],
        val=val[~missing],
    )
    assert np.isnan(probe.scores[missing]).all()
    np.testing.assert_array_equal(probe.scores[~missing], reference.scores)
    np.testing.assert_array_equal(probe.val_auroc, reference.val_auroc)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"layers": (0, 1)}, "layers has 2 entries"),
        ({"correct": np.zeros(5, dtype=np.int64)}, "one label per item"),
        ({"train": np.zeros(5, dtype=bool)}, "one entry per item"),
        ({"val": np.ones(240, dtype=bool)}, "train and val overlap"),
    ],
)
def test_the_probe_rejects_inconsistent_arguments(
    change: dict[str, object], message: str
) -> None:
    h, correct = _planted()
    train, val = _masks(len(correct))
    arguments: dict[str, object] = {
        "layers": (0, 1, 2),
        "correct": correct,
        "train": train,
        "val": val,
    }
    arguments.update(change)

    with pytest.raises(ValueError, match=message):
        fit_answer_probe(
            h,
            arguments["layers"],  # type: ignore[arg-type]
            arguments["correct"],  # type: ignore[arg-type]
            train=arguments["train"],  # type: ignore[arg-type]
            val=arguments["val"],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("mask", ["train", "val"])
def test_the_probe_rejects_masks_that_are_not_boolean(mask: str) -> None:
    h, correct = _planted()
    train, val = _masks(len(correct))
    masks = {"train": train, "val": val}
    masks[mask] = masks[mask].astype(np.int64)

    with pytest.raises(ValueError, match="boolean"):
        fit_answer_probe(h, (0, 1, 2), correct, **masks)


def test_the_scores_come_from_the_probe_of_the_reported_curve() -> None:
    h, correct = _planted(seed=6)
    train, val = _masks(len(correct))

    probe = fit_answer_probe(h, (0, 1, 2), correct, train=train, val=val)

    k = (0, 1, 2).index(probe.layer)
    assert auroc(probe.scores[val], correct[val]) == probe.val_auroc[k]


def test_non_finite_activations_of_an_answer_raise(
    tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = answer_probe.collect_activations

    def overflowing(*args: object, **kwargs: object) -> Activations:
        acts = real(*args, **kwargs)  # type: ignore[arg-type]
        acts.data[0, 0, 0] = np.nan
        return acts

    monkeypatch.setattr(answer_probe, "collect_activations", overflowing)

    with pytest.raises(ValueError, match="non-finite activations"):
        answer_activations(tiny_gpt2, PROMPTS, ANSWERS, layers=[1])


def test_the_probe_needs_both_outcomes_in_validation() -> None:
    h, correct = _planted()
    train, val = _masks(len(correct))
    correct = correct.copy()
    correct[val] = 1

    with pytest.raises(ValueError, match="both classes"):
        fit_answer_probe(h, (0, 1, 2), correct, train=train, val=val)
