"""Tests for the P4 pairing, preprocessing and component study."""

import dataclasses
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.activations import Activations
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import p4_pairing
from tomography.experiments.p4_pairing import Pairing
from tomography.experiments.p4_pairing import component_accuracy
from tomography.experiments.p4_pairing import cv_component
from tomography.experiments.p4_pairing import diff_components
from tomography.experiments.p4_pairing import pairing_mode
from tomography.experiments.p4_pairing import pairing_set
from tomography.experiments.sweep import Prepared
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import Variant
from tomography.experiments.variants import template_variants
from tomography.reading.directions import Preprocess
from tomography.reading.directions import fit_direction
from tomography.templates import ChatFormat

PLAIN = ChatFormat("plain")


def _sst2(per_class: int = 12) -> list[Example]:
    return [
        Example(
            f"sst2-{label}-{i}", f"phrase {label} number {i}", label, "train"
        )
        for label in (0, 1)
        for i in range(per_class)
    ]


def _utility(pairs: int = 8) -> list[Example]:
    rows = []
    for i in range(pairs):
        pair = f"ethics-{i}"
        for suffix, label in (("a", 1), ("b", 0)):
            rows.append(
                Example(
                    f"{pair}:{suffix}",
                    f"scenario {i} {suffix}",
                    label,
                    "train",
                    {"pair": pair},
                )
            )
    return rows


def _paper(task: str) -> Variant:
    (variant,), _ = template_variants(
        READING_TASKS[task], PLAIN, kinds=("paper",)
    )
    return variant


@pytest.mark.parametrize(
    ("task", "pairing", "mode"),
    [
        ("utility", "matched", "by_group"),
        ("utility", "label_contrastive", "label_contrastive"),
        ("utility", "random", "random"),
        ("sentiment", "label_contrastive", "label_contrastive"),
        ("sentiment", "random", "random"),
    ],
)
def test_pairing_mode_names_the_stimulus_pairing(
    task: str, pairing: Pairing, mode: str
) -> None:
    assert pairing_mode(READING_TASKS[task], pairing) == mode


def test_matched_needs_a_task_with_groups() -> None:
    with pytest.raises(
        ValueError,
        match="matched pairing needs a task with groups, 'sentiment'",
    ):
        pairing_mode(READING_TASKS["sentiment"], "matched")


def test_an_unknown_pairing_raises() -> None:
    with pytest.raises(ValueError, match="unknown pairing 'strata'"):
        pairing_mode(READING_TASKS["sentiment"], "strata")  # type: ignore[arg-type]


def test_matched_pairs_hold_both_members_of_one_group() -> None:
    stimuli = pairing_set(
        READING_TASKS["utility"],
        _paper("utility"),
        _utility(),
        "matched",
        n_pairs=5,
        seed=1,
    )
    first, second = stimuli.pairs.T
    assert stimuli.pairs.shape == (5, 2)
    assert (stimuli.labels[first] != stimuli.labels[second]).all()
    for a, b in stimuli.pairs:
        assert stimuli.uids[a].split(":")[0] == stimuli.uids[b].split(":")[0]


def test_label_contrastive_pairs_join_a_positive_and_a_negative_stimulus() -> (
    None
):
    stimuli = pairing_set(
        READING_TASKS["sentiment"],
        _paper("sentiment"),
        _sst2(),
        "label_contrastive",
        n_pairs=8,
        seed=1,
    )
    first, second = stimuli.pairs.T
    assert stimuli.pairs.shape == (8, 2)
    assert (stimuli.labels[first] != stimuli.labels[second]).all()
    assert len({stimuli.uids[a].split("-")[2] for a in first}) > 1


def test_a_label_contrastive_pairing_of_a_group_task_ignores_its_groups() -> (
    None
):
    stimuli = pairing_set(
        READING_TASKS["utility"],
        _paper("utility"),
        _utility(),
        "label_contrastive",
        n_pairs=8,
        seed=1,
    )
    same_group = [
        stimuli.uids[a].split(":")[0] == stimuli.uids[b].split(":")[0]
        for a, b in stimuli.pairs
    ]
    assert not all(same_group)
    assert (
        stimuli.labels[stimuli.pairs[:, 0]]
        != stimuli.labels[stimuli.pairs[:, 1]]
    ).all()


def test_random_pairs_can_join_two_stimuli_of_the_same_label() -> None:
    stimuli = pairing_set(
        READING_TASKS["sentiment"],
        _paper("sentiment"),
        _sst2(),
        "random",
        n_pairs=12,
        seed=1,
    )
    first, second = stimuli.pairs.T
    assert stimuli.pairs.shape == (12, 2)
    assert (stimuli.labels[first] == stimuli.labels[second]).any()
    assert (stimuli.labels[first] != stimuli.labels[second]).any()
    assert stimuli.groups.shape[0] > 0
    assert (stimuli.labels[stimuli.groups].sum(axis=1) == 1).all()


@pytest.mark.parametrize("pairing", ["matched", "label_contrastive", "random"])
def test_a_pairing_set_is_a_function_of_the_seed(pairing: Pairing) -> None:
    def draw(seed: int) -> tuple[str, ...]:
        return pairing_set(
            READING_TASKS["utility"],
            _paper("utility"),
            _utility(),
            pairing,
            n_pairs=6,
            seed=seed,
        ).uids

    assert draw(3) == draw(3)
    assert draw(3) != draw(4)


def _rows(
    n_layers: int = 2, n_pairs: int = 12, width: int = 10, seed: int = 0
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Random activations of shape (K, 2m, d) and the pairs (2i, 2i + 1)."""
    h = np.random.default_rng(seed).normal(size=(n_layers, 2 * n_pairs, width))
    pairs = np.arange(2 * n_pairs, dtype=np.int64).reshape(n_pairs, 2)
    return h.astype(np.float32), pairs


def _from_differences(
    differences: list[list[float]],
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """One layer whose pair i has the given difference."""
    diffs = np.array(differences, dtype=np.float32)
    h = np.zeros((1, 2 * len(diffs), diffs.shape[1]), dtype=np.float32)
    h[0, 0::2] = diffs
    return h, np.arange(2 * len(diffs), dtype=np.int64).reshape(-1, 2)


def test_components_have_the_documented_shapes_and_are_orthonormal() -> None:
    h, pairs = _rows()
    components, explained = diff_components(
        h, pairs, preprocess="center", n_components=4
    )
    assert components.shape == (2, 4, 10)
    assert explained.shape == (2, 4)
    for layer in components:
        np.testing.assert_allclose(layer @ layer.T, np.eye(4), atol=1e-9)


@pytest.mark.parametrize("preprocess", ["center", "unit", "none"])
def test_the_first_component_is_the_pca_diff_direction(
    preprocess: Preprocess,
) -> None:
    h, pairs = _rows(n_layers=3, seed=5)
    components, _ = diff_components(
        h, pairs, preprocess=preprocess, n_components=3
    )
    for k in range(3):
        direction = fit_direction(
            "pca_diff", h[k], pairs=pairs, preprocess=preprocess
        ).direction
        assert components[k, 0] @ direction > 1 - 1e-5


@pytest.mark.parametrize("preprocess", ["center", "unit", "none"])
def test_every_component_has_its_largest_coordinate_positive(
    preprocess: Preprocess,
) -> None:
    h, pairs = _rows(seed=2)
    components, _ = diff_components(
        h, pairs, preprocess=preprocess, n_components=6
    )
    top = np.abs(components).argmax(axis=2)
    values = np.take_along_axis(components, top[..., None], axis=2)
    assert (values > 0).all()


def test_the_explained_variance_ratios_do_not_increase() -> None:
    h, pairs = _rows(seed=3)
    _, explained = diff_components(
        h, pairs, preprocess="center", n_components=8
    )
    assert (explained >= 0).all()
    assert (np.diff(explained, axis=1) <= 1e-12).all()
    assert (explained.sum(axis=1) <= 1 + 1e-9).all()


def test_all_components_explain_everything_without_centering() -> None:
    h, pairs = _rows(n_pairs=6, width=10, seed=4)
    _, explained = diff_components(h, pairs, preprocess="none", n_components=6)
    np.testing.assert_allclose(explained.sum(axis=1), 1.0)


def test_two_orthogonal_differences_match_the_hand_values() -> None:
    h, pairs = _from_differences([[2.0, 0.0], [0.0, 1.0]])
    components, explained = diff_components(
        h, pairs, preprocess="none", n_components=2
    )
    np.testing.assert_allclose(components[0], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(explained[0], [0.8, 0.2])


def test_centering_removes_the_mean_difference() -> None:
    h, pairs = _from_differences(
        [[10.0, 1.0], [10.0, -1.0], [10.0, 1.0], [10.0, -1.0]]
    )
    centered, _ = diff_components(h, pairs, preprocess="center", n_components=1)
    raw, _ = diff_components(h, pairs, preprocess="none", n_components=1)
    np.testing.assert_allclose(centered[0, 0], [0.0, 1.0], atol=1e-6)
    assert raw[0, 0, 0] > 0.99


def test_unit_scaling_equalizes_the_differences_before_centering() -> None:
    h, pairs = _from_differences([[100.0, 0.0], [0.0, 1.0]])
    scaled, _ = diff_components(h, pairs, preprocess="unit", n_components=1)
    plain, _ = diff_components(h, pairs, preprocess="center", n_components=1)
    assert abs(scaled[0, 0] @ np.array([1.0, -1.0])) > np.sqrt(2.0) * (1 - 1e-9)
    assert plain[0, 0, 0] > 0.9999


def test_each_layer_is_decomposed_on_its_own() -> None:
    h, pairs = _rows(n_layers=3, seed=6)
    together, explained = diff_components(
        h, pairs, preprocess="center", n_components=3
    )
    for k in range(3):
        alone, share = diff_components(
            h[k : k + 1], pairs, preprocess="center", n_components=3
        )
        np.testing.assert_allclose(together[k], alone[0])
        np.testing.assert_allclose(explained[k], share[0])


@pytest.mark.parametrize("n_components", [0, -1, 5])
def test_the_number_of_components_stays_within_the_pair_count(
    n_components: int,
) -> None:
    h, pairs = _rows(n_pairs=4, width=10)
    with pytest.raises(ValueError, match="n_components"):
        diff_components(
            h, pairs, preprocess="center", n_components=n_components
        )


def test_the_number_of_components_stays_within_the_width() -> None:
    h, pairs = _rows(n_pairs=12, width=3)
    diff_components(h, pairs, preprocess="center", n_components=3)
    with pytest.raises(ValueError, match="n_components"):
        diff_components(h, pairs, preprocess="center", n_components=4)


def test_differences_without_variance_raise() -> None:
    h = np.ones((1, 8, 4), dtype=np.float32)
    pairs = np.arange(8, dtype=np.int64).reshape(4, 2)
    with pytest.raises(ValueError, match="no variance"):
        diff_components(h, pairs, preprocess="center", n_components=2)


@pytest.mark.parametrize(
    ("h", "pairs"),
    [
        (np.zeros((6, 4)), np.array([[0, 1], [2, 3]])),
        (np.zeros((1, 6, 4)), np.array([0, 1, 2, 3])),
        (np.zeros((1, 6, 4)), np.array([[0, 1, 2], [3, 4, 5]])),
        (np.zeros((1, 6, 4)), np.array([[0, 1], [2, 6]])),
    ],
    ids=["two_dimensional_h", "flat_pairs", "triples", "out_of_range"],
)
def test_malformed_inputs_raise(
    h: NDArray[np.float64], pairs: NDArray[np.int64]
) -> None:
    with pytest.raises(ValueError, match="diff_components"):
        diff_components(h, pairs, preprocess="center", n_components=1)


def test_an_unknown_preprocessing_raises() -> None:
    h, pairs = _rows()
    with pytest.raises(ValueError, match="preprocess"):
        diff_components(
            h,
            pairs,
            preprocess="whiten",  # type: ignore[arg-type]
            n_components=2,
        )


def test_the_explained_variance_is_a_share_of_all_the_components() -> None:
    h, pairs = _from_differences([[2.0, 0.0], [0.0, 1.0], [0.0, 0.5]])
    _, explained = diff_components(h, pairs, preprocess="none", n_components=1)
    np.testing.assert_allclose(explained[0], [4.0 / 5.25])


WIDTH = 12


def _prepared(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
    pairs: NDArray[np.int64],
    groups: NDArray[np.int64] | None = None,
) -> Prepared:
    """Wraps activations of shape (K, n, d), one row per stimulus."""
    n = len(labels)
    index = np.arange(n, dtype=np.int64)
    stimuli = StimulusSet(
        texts=tuple(f"text {i}" for i in range(n)),
        response_spans=(None,) * n,
        concept_spans=(None,) * n,
        labels=labels,
        pairs=pairs,
        groups=pairs.copy() if groups is None else groups,
        uids=tuple(f"u{i}" for i in range(n)),
    )
    act = Activations(
        data=data.astype(np.float32),
        layers=tuple(range(1, data.shape[0] + 1)),
        position="last",
        row_stimulus=index,
        row_rank=np.zeros(n, dtype=np.int64),
        row_token=np.zeros(n, dtype=np.int64),
        row_token_id=np.zeros(n, dtype=np.int64),
        n_stimuli=n,
    )
    return stimuli, act


def _axes(seed: int = 7) -> NDArray[np.float64]:
    """Orthonormal axes of the planted signal and the nuisance."""
    return np.linalg.qr(
        np.random.default_rng(seed).normal(size=(WIDTH, WIDTH))
    )[0]


def _planted_pairs(
    n_pairs: int,
    *,
    seed: int,
    signal: float = 3.0,
    nuisance: float = 0.0,
    flip_labels: bool = False,
    n_layers: int = 2,
) -> Prepared:
    """Matched pairs whose label-1 member sits `signal` along the first axis.

    Each pair adds a shared draw to both members and, when `nuisance` is
    set, an independent draw along the second axis to the member picked as
    first, so the second axis carries variance that follows no label.
    """
    rng = np.random.default_rng(seed)
    axes = _axes()
    sign = -1.0 if flip_labels else 1.0
    labels = np.zeros(2 * n_pairs, dtype=np.int64)
    data = np.zeros((n_layers, 2 * n_pairs, WIDTH))
    for i in range(n_pairs):
        base = rng.normal(0.0, 0.1, size=(n_layers, WIDTH))
        swap = int(rng.integers(2))
        positive, negative = 2 * i + swap, 2 * i + 1 - swap
        labels[positive] = 1
        data[:, positive] = base + sign * signal * axes[:, 0]
        data[:, negative] = base - sign * signal * axes[:, 0]
        data[:, positive] += nuisance * rng.normal() * axes[:, 1]
    pairs = np.arange(2 * n_pairs, dtype=np.int64).reshape(n_pairs, 2)
    return _prepared(data, labels, pairs)


def _components_of(
    prepared: Prepared, n_components: int = 3
) -> NDArray[np.float64]:
    stimuli, act = prepared
    components, _ = diff_components(
        act.data,
        act.row_groups(stimuli.pairs),
        preprocess="center",
        n_components=n_components,
    )
    return components


def test_the_planted_direction_is_component_one_with_accuracy_one() -> None:
    train = _planted_pairs(16, seed=1)
    report = _planted_pairs(10, seed=2)
    components = _components_of(train)
    assert abs(components[0, 0] @ _axes()[:, 0]) > 0.999
    accuracy = component_accuracy(train, report, components)
    assert accuracy.shape == (2, 3)
    np.testing.assert_array_equal(accuracy[:, 0], [1.0, 1.0])


def test_a_component_is_oriented_on_the_training_groups() -> None:
    train = _planted_pairs(16, seed=1)
    report = _planted_pairs(10, seed=2)
    components = _components_of(train)
    plain = component_accuracy(train, report, components)
    flipped = component_accuracy(train, report, -components)
    np.testing.assert_array_equal(flipped[:, 0], plain[:, 0])
    assert (plain[:, 0] == 1.0).all()


def test_an_oriented_component_never_loses_on_its_own_training_groups() -> None:
    train = _planted_pairs(16, seed=1, nuisance=2.0)
    components = _components_of(train, n_components=6)
    for sign in (1.0, -1.0):
        accuracy = component_accuracy(train, train, sign * components)
        assert (accuracy >= 0.5).all()


def test_the_orientation_does_not_look_at_the_evaluated_labels() -> None:
    train = _planted_pairs(16, seed=1)
    reversed_report = _planted_pairs(10, seed=2, flip_labels=True)
    accuracy = component_accuracy(train, reversed_report, _components_of(train))
    np.testing.assert_array_equal(accuracy[:, 0], [0.0, 0.0])


def test_a_component_that_ties_every_group_scores_one_half() -> None:
    train = _planted_pairs(16, seed=1)
    report = _planted_pairs(10, seed=2)
    accuracy = component_accuracy(train, report, np.zeros((2, 1, WIDTH)))
    np.testing.assert_array_equal(accuracy, [[0.5], [0.5]])


@pytest.mark.parametrize(
    "components",
    [np.ones((2, WIDTH)), np.ones((3, 2, WIDTH)), np.ones((2, 2, WIDTH + 1))],
    ids=["two_dimensional", "wrong_layer_count", "wrong_width"],
)
def test_components_must_fit_the_activations(
    components: NDArray[np.float64],
) -> None:
    train = _planted_pairs(8, seed=1)
    with pytest.raises(ValueError, match="component_accuracy"):
        component_accuracy(train, train, components)


def test_a_component_at_half_on_the_training_groups_keeps_its_sign() -> None:
    labels = np.array([1, 0, 1, 0], dtype=np.int64)
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)
    train = _prepared(
        np.array([[[1.0, 0], [0, 0], [-1.0, 0], [0, 0]]]), labels, pairs
    )
    report = _prepared(
        np.array([[[1.0, 0], [0, 0], [2.0, 0], [0, 0]]]), labels, pairs
    )
    accuracy = component_accuracy(train, report, np.array([[[1.0, 0.0]]]))
    np.testing.assert_array_equal(accuracy, [[1.0]])


def _randomly_paired(prepared: Prepared, *, seed: int) -> Prepared:
    """The same stimuli, paired at random whatever their labels."""
    stimuli, act = prepared
    order = np.random.default_rng(seed).permutation(len(stimuli.labels))
    pairs = order.astype(np.int64).reshape(-1, 2)
    return dataclasses.replace(stimuli, pairs=pairs), act


def _cv(prepared: Prepared, **overrides: Any) -> int:
    options: dict[str, Any] = {
        "layer_index": 0,
        "preprocess": "center",
        "n_components": 3,
        "n_folds": 4,
        "seed": 0,
    }
    return cv_component(prepared, **{**options, **overrides})


def test_cross_validation_picks_the_planted_second_component() -> None:
    train = _planted_pairs(24, seed=3, signal=1.0, nuisance=5.0)
    assert abs(_components_of(train)[0, 0] @ _axes()[:, 1]) > 0.99
    assert _cv(train) == 1
    assert _cv(train, layer_index=1) == 1


def test_cross_validation_scores_only_pairs_with_distinct_labels() -> None:
    matched = _planted_pairs(30, seed=4, signal=2.0, nuisance=5.0)
    train = _randomly_paired(matched, seed=1)
    stimuli, _ = train
    first, second = stimuli.pairs.T
    assert (stimuli.labels[first] == stimuli.labels[second]).any()
    assert _cv(train) == 1


def test_cross_validation_is_deterministic_under_the_seed() -> None:
    train = _planted_pairs(24, seed=3, signal=1.0, nuisance=5.0)
    assert _cv(train, seed=5) == _cv(train, seed=5)


def test_the_first_component_wins_a_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _planted_pairs(12, seed=3)

    def same_component(
        h: NDArray[np.float32], pairs: NDArray[np.int64], **kwargs: Any
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        axis = np.tile(_axes()[:, 0], (1, kwargs["n_components"], 1))
        return axis, np.zeros((1, kwargs["n_components"]))

    monkeypatch.setattr(p4_pairing, "diff_components", same_component)
    assert _cv(train) == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_folds": 1}, "n_folds must be between 2 and the 8 pairs"),
        ({"n_folds": 9}, "n_folds must be between 2 and the 8 pairs"),
        ({"n_components": 7, "n_folds": 2}, "folds too small"),
        ({"n_components": 8, "n_folds": 8}, "folds too small"),
        ({"layer_index": 2}, "layer_index"),
        ({"layer_index": -1}, "layer_index"),
    ],
    ids=[
        "one_fold",
        "more_folds_than_pairs",
        "half_the_pairs_for_seven",
        "leave_one_out_for_eight",
        "layer_above_range",
        "negative_layer",
    ],
)
def test_cross_validation_rejects_folds_and_layers_that_do_not_fit(
    overrides: dict[str, Any], message: str
) -> None:
    train = _planted_pairs(8, seed=3)
    with pytest.raises(ValueError, match=message):
        _cv(train, **overrides)


def test_cross_validation_needs_pairs_with_distinct_labels() -> None:
    data = np.random.default_rng(0).normal(size=(1, 16, WIDTH))
    labels = np.array([1, 1, 0, 0] * 4, dtype=np.int64)
    pairs = np.arange(16, dtype=np.int64).reshape(-1, 2)
    with pytest.raises(ValueError, match="distinct labels"):
        _cv(_prepared(data, labels, pairs), n_components=2)


@pytest.mark.parametrize("flip_labels", [False, True])
def test_cross_validation_orients_each_component_on_the_fit_folds(
    flip_labels: bool,
) -> None:
    train = _planted_pairs(
        24, seed=3, signal=1.0, nuisance=5.0, flip_labels=flip_labels
    )
    assert _cv(train) == 1


def test_the_folds_hold_out_every_pair_once_and_follow_the_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _planted_pairs(10, seed=3, signal=1.0, nuisance=5.0)
    fitted: list[NDArray[np.int64]] = []
    real = p4_pairing.diff_components

    def record(
        h: NDArray[np.float32], pairs: NDArray[np.int64], **kwargs: Any
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        fitted.append(pairs)
        return real(h, pairs, **kwargs)

    monkeypatch.setattr(p4_pairing, "diff_components", record)
    _cv(train, n_folds=3, seed=11)
    everything = train[0].pairs
    order = np.random.default_rng(11).permutation(10)
    held_out = np.array_split(order, 3)
    assert sorted(i for fold in held_out for i in fold) == list(range(10))
    assert len(fitted) == 3
    for fold, pairs in zip(held_out, fitted, strict=True):
        kept = np.setdiff1d(np.arange(10), fold)
        np.testing.assert_array_equal(pairs, everything[kept])


def test_components_are_oriented_and_scored_on_distinct_label_pairs_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _randomly_paired(
        _planted_pairs(30, seed=4, signal=2.0, nuisance=5.0), seed=1
    )
    stimuli, _ = train
    labels = stimuli.labels
    distinct = labels[stimuli.pairs[:, 0]] != labels[stimuli.pairs[:, 1]]
    credited: list[NDArray[np.int64]] = []
    real = p4_pairing._pair_credit

    def record(*args: Any) -> NDArray[np.float64]:
        credited.append(args[2])
        return real(*args)

    monkeypatch.setattr(p4_pairing, "_pair_credit", record)
    _cv(train, n_folds=4, seed=2)
    order = np.random.default_rng(2).permutation(len(stimuli.pairs))
    assert len(credited) == 8
    for fold, held_out in enumerate(np.array_split(order, 4)):
        fit = np.setdiff1d(order, held_out)
        np.testing.assert_array_equal(
            credited[2 * fold], stimuli.pairs[fit[distinct[fit]]]
        )
        np.testing.assert_array_equal(
            credited[2 * fold + 1],
            stimuli.pairs[held_out[distinct[held_out]]],
        )
