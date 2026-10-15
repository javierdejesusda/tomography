"""Tests for the P3 stimulus-set size study."""

from collections.abc import Sequence
import dataclasses
import gc
import math
from pathlib import Path
import re
import time
from typing import Any
import weakref

from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.activations import Activations
from tomography.cli import load_experiment
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import p3_size
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.p3_size import COLUMNS
from tomography.experiments.p3_size import Config
from tomography.experiments.p3_size import directional_agreement
from tomography.experiments.p3_size import execute
from tomography.experiments.p3_size import oriented_differences
from tomography.experiments.p3_size import plot
from tomography.experiments.p3_size import plot_size_curves
from tomography.experiments.p3_size import run
from tomography.experiments.p3_size import saturation_ci
from tomography.experiments.p3_size import saturation_size
from tomography.experiments.p3_size import spearman
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import Prepared
from tomography.experiments.sweep import SweepResult
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import sweep_from_activations
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import eval_pairs
from tomography.experiments.variants import eval_seed
from tomography.experiments.variants import load_sources
from tomography.experiments.variants import prepare
from tomography.experiments.variants import task_set
from tomography.experiments.variants import task_splits
from tomography.experiments.variants import template_variants
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.metrics import auroc
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import bootstrap_ci


def _prepared(
    data: NDArray[np.float64],
    labels: list[int],
    pairs: list[tuple[int, int]],
) -> Prepared:
    """Wraps activations of shape (K, n, d), one row per stimulus."""
    n = len(labels)
    index = np.arange(n, dtype=np.int64)
    pair_array = np.array(pairs, dtype=np.int64)
    stimuli = StimulusSet(
        texts=tuple(f"text {i}" for i in range(n)),
        response_spans=(None,) * n,
        concept_spans=(None,) * n,
        labels=np.array(labels, dtype=np.int64),
        pairs=pair_array,
        groups=pair_array.copy(),
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


def test_spearman_of_a_swapped_pair_is_point_eight() -> None:
    got = spearman(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 3.0, 2.0, 4.0])
    )
    assert got == pytest.approx(0.8)
    assert isinstance(got, float)


def test_spearman_averages_the_ranks_of_ties() -> None:
    got = spearman(
        np.array([1.0, 2.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0])
    )
    assert got == pytest.approx(3 / math.sqrt(10))


def test_spearman_follows_ranks_not_values() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, np.array([1.0, 10.0, 100.0, 1000.0])) == pytest.approx(
        1.0
    )
    assert spearman(x, x[::-1]) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        ([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]),
    ],
    ids=["constant_x", "constant_y"],
)
def test_spearman_is_nan_when_one_side_is_constant(
    x: list[float], y: list[float]
) -> None:
    assert math.isnan(spearman(np.array(x), np.array(y)))


@pytest.mark.parametrize(
    ("x", "y"),
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0]),
        ([1.0], [1.0]),
        ([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]),
    ],
    ids=["different_lengths", "one_value", "two_dimensional"],
)
def test_spearman_rejects_bad_shapes(x: list[object], y: list[object]) -> None:
    with pytest.raises(ValueError, match="spearman"):
        spearman(np.array(x), np.array(y))


def test_oriented_differences_subtract_label_0_from_label_1() -> None:
    data = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3) ** 2
    got = oriented_differences(_prepared(data, [1, 0, 0, 1], [(0, 1), (2, 3)]))
    assert got.shape == (2, 2, 3)
    np.testing.assert_allclose(got[:, 0], data[:, 0] - data[:, 1])
    np.testing.assert_allclose(got[:, 1], data[:, 3] - data[:, 2])


def test_a_flipped_pair_gives_the_same_difference() -> None:
    data = np.random.default_rng(0).normal(size=(2, 4, 3))
    labels = [1, 0, 1, 0]
    straight = oriented_differences(_prepared(data, labels, [(0, 1), (2, 3)]))
    flipped = oriented_differences(_prepared(data, labels, [(1, 0), (2, 3)]))
    np.testing.assert_array_equal(straight, flipped)


def test_a_pair_with_equal_labels_raises() -> None:
    data = np.zeros((1, 4, 2))
    with pytest.raises(ValueError, match="pair 1 has equal labels"):
        oriented_differences(_prepared(data, [1, 0, 1, 1], [(0, 1), (2, 3)]))


def test_the_agreement_of_two_differences_matches_the_hand_value() -> None:
    diffs = np.array([[[1.0, 0.0], [1.0, 1.0]]])
    expected = (1 / math.sqrt(1.25) + 1.5 / math.sqrt(2.5)) / 2
    np.testing.assert_allclose(directional_agreement(diffs), [expected])


def test_identical_differences_agree_completely() -> None:
    diffs = np.tile(np.array([2.0, -1.0, 0.5]), (1, 5, 1))
    np.testing.assert_allclose(directional_agreement(diffs), [1.0])


def test_each_layer_has_its_own_agreement() -> None:
    first = np.array([[1.0, 0.0], [1.0, 1.0]])
    second = np.array([[1.0, 0.0], [1.0, 0.0]])
    got = directional_agreement(np.stack([first, second]))
    assert got.shape == (2,)
    np.testing.assert_allclose(got[1], 1.0)
    assert got[0] < 1.0


@pytest.mark.parametrize(
    "diffs",
    [
        [[[0.0, 0.0], [1.0, 1.0]]],
        [[[1.0, 0.0], [-1.0, 0.0]]],
    ],
    ids=["zero_difference", "zero_mean"],
)
def test_agreement_raises_on_a_zero_vector(diffs: list[object]) -> None:
    with pytest.raises(ValueError, match="zero"):
        directional_agreement(np.array(diffs))


@pytest.mark.parametrize(
    "diffs",
    [np.ones((1, 1, 3)), np.ones((4, 3))],
    ids=["one_pair", "two_dimensional"],
)
def test_agreement_needs_two_pairs_in_three_dimensions(
    diffs: NDArray[np.float64],
) -> None:
    with pytest.raises(ValueError, match="directional_agreement"):
        directional_agreement(diffs)


SIZES = (2, 4, 8, 16, 32)


def _planted(
    means: list[float],
    *,
    resamples: int = 10,
    noise: float = 0.0,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Accuracy of shape (Z, R) around a planted curve."""
    jitter = np.random.default_rng(seed).normal(
        0.0, noise, size=(len(means), resamples)
    )
    return np.asarray(means)[:, None] + jitter


def test_saturation_is_the_smallest_size_within_delta_of_the_largest() -> None:
    accuracy = _planted([0.55, 0.70, 0.895, 0.90, 0.90])
    assert saturation_size(SIZES, accuracy, delta=0.01) == 8
    assert saturation_size(SIZES, accuracy, delta=0.0) == 16
    assert saturation_size(SIZES, accuracy, delta=0.5) == 2


def test_saturation_accepts_a_size_exactly_delta_below() -> None:
    accuracy = _planted([0.5, 0.75, 1.0])
    assert saturation_size((2, 4, 8), accuracy, delta=0.25) == 4


def test_saturation_takes_the_first_size_even_if_the_curve_dips_later() -> None:
    accuracy = _planted([0.5, 0.9, 0.7, 0.9])
    assert saturation_size((2, 4, 8, 16), accuracy, delta=0.01) == 4


def test_saturation_averages_the_resamples_of_a_size() -> None:
    accuracy = np.array([[0.25, 0.75], [1.0, 0.5], [1.0, 1.0]])
    assert saturation_size((2, 4, 8), accuracy, delta=0.125) == 8


def test_saturation_returns_a_plain_int() -> None:
    got = saturation_size(SIZES, _planted([0.5, 0.6, 0.7, 0.8, 0.9]), delta=0.0)
    assert type(got) is int


@pytest.mark.parametrize(
    ("sizes", "shape", "delta"),
    [
        (SIZES, (4, 3), 0.01),
        (SIZES, (5,), 0.01),
        (SIZES, (5, 3), -0.01),
    ],
    ids=["wrong_size_count", "one_dimensional", "negative_delta"],
)
def test_saturation_rejects_bad_inputs(
    sizes: tuple[int, ...], shape: tuple[int, ...], delta: float
) -> None:
    with pytest.raises(ValueError, match="saturation"):
        saturation_size(sizes, np.ones(shape), delta=delta)


def test_the_saturation_interval_contains_the_point_of_a_planted_curve() -> (
    None
):
    accuracy = _planted([0.55, 0.70, 0.90, 0.91, 0.91], noise=0.01)
    point, lower, upper = saturation_ci(
        SIZES, accuracy, delta=0.03, n_boot=500, seed=0
    )
    assert point == 8
    assert lower <= point <= upper
    assert lower >= SIZES[0]
    assert upper <= SIZES[-1]


def test_the_saturation_interval_is_deterministic_under_the_seed() -> None:
    accuracy = _planted([0.55, 0.70, 0.90, 0.91, 0.91], noise=0.03)
    first = saturation_ci(SIZES, accuracy, delta=0.01, n_boot=300, seed=3)
    second = saturation_ci(SIZES, accuracy, delta=0.01, n_boot=300, seed=3)
    assert first == second
    assert all(type(value) is int for value in first)


def test_a_noisy_curve_has_a_wider_interval_than_a_clean_one() -> None:
    means = [0.55, 0.70, 0.90, 0.91, 0.91]
    clean = saturation_ci(
        SIZES, _planted(means), delta=0.01, n_boot=300, seed=0
    )
    noisy = saturation_ci(
        SIZES, _planted(means, noise=0.05), delta=0.01, n_boot=300, seed=0
    )
    assert clean[1] == clean[0] == clean[2]
    assert noisy[2] - noisy[1] > 0


def test_one_resample_per_size_gives_a_degenerate_interval() -> None:
    accuracy = _planted([0.55, 0.70, 0.90, 0.91, 0.91], resamples=1)
    point, lower, upper = saturation_ci(
        SIZES, accuracy, delta=0.02, n_boot=50, seed=0
    )
    assert lower == point == upper == 8


def test_the_interval_of_a_constant_curve_sits_at_the_first_size() -> None:
    got = saturation_ci(
        SIZES, np.ones((len(SIZES), 4)), delta=0.01, n_boot=10, seed=0
    )
    assert got == (2, 2, 2)


def test_the_sizes_are_bootstrapped_with_separate_draws() -> None:
    row = np.array([0.5, 0.7, 0.6, 0.8, 0.55, 0.65])
    accuracy = np.stack([row, row])
    got = saturation_ci((2, 4), accuracy, delta=0.0, n_boot=200, seed=0)
    assert got == (2, 2, 4)


def test_the_point_is_the_full_sample_value_and_not_the_bootstrap_median() -> (
    None
):
    accuracy = np.array(
        [
            [0.65, 0.66, 0.78, 0.47, 0.81],
            [0.74, 0.79, 0.50, 0.80, 0.50],
            [0.44, 0.83, 0.83, 0.84, 0.64],
        ]
    )
    got = saturation_ci((2, 4, 8), accuracy, delta=0.02, n_boot=100, seed=0)
    assert saturation_size((2, 4, 8), accuracy, delta=0.02) == 8
    assert got == (8, 2, 8)


def test_the_interval_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="saturation"):
        saturation_ci(SIZES, np.ones((4, 3)), delta=0.01, n_boot=10, seed=0)
    with pytest.raises(ValueError, match="n_boot"):
        saturation_ci(SIZES, np.ones((5, 3)), delta=0.01, n_boot=0, seed=0)


def _sst2() -> list[Example]:
    return [
        Example(
            f"sst2-{split}-{label}-{i}",
            f"{split} phrase {label} number {i}",
            label,
            split,
        )
        for split, count in (("train", 12), ("val", 20))
        for label in (0, 1)
        for i in range(count)
    ]


def _facts() -> list[Example]:
    return [
        Example(
            f"facts-{split}-{label}-{i}",
            f"fact {split} {label} {i}",
            label,
            split,
        )
        for split in ("train", "val", "test")
        for label in (0, 1)
        for i in range(10)
    ]


def _utility() -> list[Example]:
    rows = []
    for split in ("train", "val", "test"):
        for i in range(8):
            pair = f"ethics-{split}-{i}"
            for suffix, label in (("a", 1), ("b", 0)):
                rows.append(
                    Example(
                        f"{pair}:{suffix}",
                        f"scenario {split} {i} {suffix}",
                        label,
                        split,
                        {"pair": pair},
                    )
                )
    return rows


def _truth_sets() -> list[Example]:
    return [
        Example(
            f"got-{subset}-{split}-{label}-{i}",
            f"{subset} claim {split} {label} {i}",
            label,
            split,
            {"subset": subset},
        )
        for subset in ("cities", "common_claim_true_false", "neg_cities")
        for split in ("train", "val", "test")
        for label in (0, 1)
        for i in range(6)
    ]


def _sources() -> dict[str, list[Example]]:
    return {
        "sst2": _sst2(),
        "facts": _facts(),
        "ethics_utility": _utility(),
        "geometry_of_truth": _truth_sets(),
    }


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        n_seeds=3,
        n_eval_pairs=6,
        sizes=(2, 4, 8),
        reference_pairs=64,
        n_boot=50,
    )
    return dataclasses.replace(base, **overrides)


TASKS = ("sentiment", "truth", "truth_cities", "truth_common", "utility")
RUN_SIZES = {
    "sentiment": [2, 4, 8],
    "truth": [2, 4, 8],
    "truth_cities": [2, 4],
    "truth_common": [2, 4],
    "utility": [2, 4, 8],
}


@pytest.fixture(scope="module")
def runs() -> dict[str, tuple[Config, RunDir]]:
    return {}


@pytest.fixture
def finished(
    tiny_arch: LoadedModel,
    runs: dict[str, tuple[Config, RunDir]],
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Config, RunDir]:
    """One P3 run per architecture, shared by the tests that only read it."""
    key = str(tiny_arch.model.config.model_type)
    if key not in runs:
        config = _config()
        run_dir = RunDir.create(config, out_root=tmp_path_factory.mktemp(key))
        execute(config, run_dir, tiny_arch, _sources())
        runs[key] = (config, run_dir)
    return runs[key]


def _json(run_dir: RunDir, name: str) -> Any:
    return run_dir.read_json(name)


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "p3"
    assert config.model == "gpt2"
    assert config.method == "logistic"
    assert config.n_seeds == 20
    assert config.n_eval_pairs == 256
    assert config.tasks == TASKS
    assert config.sizes == (2, 4, 8, 16, 32, 64, 128, 256, 512)
    assert config.reference_pairs == 2048
    assert config.failure_accuracy == 0.6
    assert config.saturation_delta == 0.01
    assert config.n_boot == 2000
    assert Config.from_dict(config.to_dict()) == config


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("p3")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


CAPACITY = {
    "sentiment": 12,
    "truth": 10,
    "truth_cities": 6,
    "truth_common": 6,
    "utility": 8,
}
SOURCES_OF_TRUTH = ["truth", "truth_cities", "truth_common"]


def _target(task: str) -> str:
    return READING_TASKS[task].target or task


def _reference_sweep(
    loaded: LoadedModel, config: Config, name: str
) -> SweepResult:
    """Refits the reference reader of a task through the public pieces."""
    task = READING_TASKS[name]
    (paper,), _ = template_variants(
        task, chat_format_for(loaded), kinds=("paper",)
    )
    splits = task_splits(task, _sources(), seed=config.seed)
    train = prepare(
        loaded,
        task_set(
            task,
            paper,
            splits.train,
            n_pairs=min(CAPACITY[name], config.reference_pairs),
            seed=derive_seed(config.seed, "reference", name),
        ),
        config.position,
        config,
    )
    position = (
        config.position
        if config.eval_position is None
        else config.eval_position
    )
    evals = {
        split: prepare(
            loaded,
            task_set(
                task,
                paper,
                part,
                n_pairs=eval_pairs(task, part, config.n_eval_pairs),
                seed=eval_seed(task, split, config.seed),
            ),
            position,
            config,
        )
        for split, part in (
            ("select", splits.select),
            ("report", splits.report),
        )
    }
    return sweep_from_activations(
        [train], evals, method=config.method, preprocess=config.preprocess
    )


def test_execute_writes_the_files_of_every_task_and_the_sources(
    finished: tuple[Config, RunDir],
) -> None:
    _, run_dir = finished
    expected = {
        "config.json",
        "meta.json",
        "summary.json",
        "table.json",
        "curves_sources_truth.json",
        "curves_sources_truth.npz",
        "sources_truth.json",
        "sources_truth.npz",
    }
    for task in TASKS:
        for stem in (
            f"sizes_{task}",
            f"curves_{task}",
            f"curves_{task}_cosine",
        ):
            expected |= {f"{stem}.json", f"{stem}.npz"}
    assert {p.name for p in run_dir.path.iterdir()} == expected
    assert len(expected) == 38


def test_the_size_arrays_have_the_documented_shapes(
    finished: tuple[Config, RunDir],
) -> None:
    config, run_dir = finished
    for task in TASKS:
        sizes = RUN_SIZES[task]
        arrays = run_dir.read_arrays(f"sizes_{task}")
        n = config.n_seeds
        assert arrays["sizes"].tolist() == sizes
        assert arrays["layers"].tolist() == [1, 2, 3]
        assert arrays["accuracy"].shape == (len(sizes), 2, 3, n, 3)
        assert arrays["cosine"].shape == (len(sizes), n, 3)
        assert arrays["agreement"].shape == (len(sizes), n, 3)
        assert arrays["reference"].shape == (2, 3, 3)
        for key in ("accuracy", "cosine", "agreement", "reference"):
            assert np.isfinite(arrays[key]).all()
        assert arrays["accuracy"].min() >= 0.0
        assert arrays["accuracy"].max() <= 1.0
        assert np.abs(arrays["cosine"]).max() <= 1.0 + 1e-6
        assert np.abs(arrays["agreement"]).max() <= 1.0 + 1e-6
        meta = _json(run_dir, f"sizes_{task}")
        assert meta["sizes"] == sizes
        assert meta["splits"] == ["select", "report"]
        assert meta["series"] == list(SERIES)
        assert meta["target"] == _target(task)
        assert meta["reference_pairs"] == CAPACITY[task]


def test_the_reading_layer_is_the_references_best_select_layer(
    finished: tuple[Config, RunDir],
) -> None:
    _, run_dir = finished
    summary = _json(run_dir, "summary")
    for task in TASKS:
        arrays = run_dir.read_arrays(f"sizes_{task}")
        layers = arrays["layers"].tolist()
        reference = arrays["reference"]
        layer = best_layer(reference[0, 0], layers)
        at = layers.index(layer)
        assert summary["layers"][task] == layer
        assert _json(run_dir, f"sizes_{task}")["layer"] == layer
        assert summary["reference"][task] == {
            "pairs": CAPACITY[task],
            "select": pytest.approx(reference[0, 0, at]),
            "report": pytest.approx(reference[1, 0, at]),
        }


def test_the_reference_is_one_reader_on_the_shared_select_and_report_items(
    tiny_arch: LoadedModel, finished: tuple[Config, RunDir]
) -> None:
    config, run_dir = finished
    for task in ("sentiment", "truth_cities", "utility"):
        sweep = _reference_sweep(tiny_arch, config, task)
        reference = run_dir.read_arrays(f"sizes_{task}")["reference"]
        np.testing.assert_allclose(
            reference[0, 0], sweep.per_seed("select")[0, 0]
        )
        np.testing.assert_allclose(
            reference[1, 0], sweep.per_seed("report")[0, 0]
        )


def _resample_cosine(
    loaded: LoadedModel, config: Config, name: str, size: int, r: int
) -> NDArray[np.float64]:
    """Recomputes the cosine of one resample to the reference direction."""
    task = READING_TASKS[name]
    (paper,), _ = template_variants(
        task, chat_format_for(loaded), kinds=("paper",)
    )
    splits = task_splits(task, _sources(), seed=config.seed)
    train = prepare(
        loaded,
        task_set(
            task,
            paper,
            splits.train,
            n_pairs=size,
            seed=derive_seed(config.seed, "size", name, size, r),
        ),
        config.position,
        config,
    )
    resample = sweep_from_activations(
        [train],
        {"train": train},
        method=config.method,
        preprocess=config.preprocess,
    ).readers[0]
    reference = _reference_sweep(loaded, config, name).readers[0]
    ours = (resample.directions * resample.signs[:, None]).astype(np.float64)
    theirs = (reference.directions * reference.signs[:, None]).astype(
        np.float64
    )
    return np.clip(np.einsum("kd,kd->k", ours, theirs), -1.0, 1.0)


@pytest.mark.parametrize(("z", "size", "r"), [(0, 2, 0), (1, 4, 1)])
def test_the_cosine_compares_oriented_directions_at_the_same_layer(
    tiny_arch: LoadedModel,
    finished: tuple[Config, RunDir],
    z: int,
    size: int,
    r: int,
) -> None:
    config, run_dir = finished
    expected = _resample_cosine(tiny_arch, config, "utility", size, r)
    cosine = run_dir.read_arrays("sizes_utility")["cosine"]
    np.testing.assert_allclose(cosine[z, r], expected)
    assert (expected < 0.99).any()


def test_the_reference_takes_at_most_reference_pairs_pairs(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(
        tasks=("sentiment", "utility"),
        sizes=(2, 4),
        n_seeds=2,
        reference_pairs=5,
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _sources())
    summary = _json(run_dir, "summary")
    for task in ("sentiment", "utility"):
        assert summary["reference"][task]["pairs"] == 5
        assert _json(run_dir, f"sizes_{task}")["reference_pairs"] == 5
        sweep = _reference_sweep(tiny_arch, config, task)
        reference = run_dir.read_arrays(f"sizes_{task}")["reference"]
        np.testing.assert_allclose(
            reference[0, 0], sweep.per_seed("select")[0, 0]
        )
        np.testing.assert_allclose(
            reference[1, 0], sweep.per_seed("report")[0, 0]
        )
        np.testing.assert_allclose(
            run_dir.read_arrays(f"sizes_{task}")["cosine"][0, 0],
            _resample_cosine(tiny_arch, config, task, 2, 0),
        )


def test_the_evaluation_items_are_read_at_the_eval_position(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(
        tasks=("sentiment",), sizes=(2, 4), n_seeds=2, eval_position="mean"
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _sources())
    sweep = _reference_sweep(tiny_arch, config, "sentiment")
    last = _reference_sweep(
        tiny_arch, dataclasses.replace(config, eval_position=None), "sentiment"
    )
    assert not np.allclose(
        sweep.per_seed("report")[0, 0], last.per_seed("report")[0, 0]
    )
    reference = run_dir.read_arrays("sizes_sentiment")["reference"]
    np.testing.assert_allclose(reference[0, 0], sweep.per_seed("select")[0, 0])
    np.testing.assert_allclose(reference[1, 0], sweep.per_seed("report")[0, 0])


def test_a_size_releases_its_activations_before_the_next_size_is_read(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = p3_size.prepare
    prepared: list[weakref.ref[Activations]] = []
    alive_at_call: list[int] = []

    def tracked(*args: Any) -> Prepared:
        gc.collect()
        alive_at_call.append(sum(ref() is not None for ref in prepared[3:]))
        result = real(*args)
        prepared.append(weakref.ref(result[1]))
        return result

    monkeypatch.setattr(p3_size, "prepare", tracked)
    config = _config(tasks=("utility",), sizes=(2, 4, 8), n_seeds=2)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _sources())
    first_of_each_size = [3 + 2 * z for z in range(3)]
    assert len(prepared) == 3 + 3 * 2
    assert [alive_at_call[i] for i in first_of_each_size] == [0, 0, 0]


def test_a_resample_is_trained_on_the_seeded_pairs_of_its_size(
    tiny_arch: LoadedModel, finished: tuple[Config, RunDir]
) -> None:
    config, run_dir = finished
    task = READING_TASKS["utility"]
    (paper,), _ = template_variants(
        task, chat_format_for(tiny_arch), kinds=("paper",)
    )
    splits = task_splits(task, _sources(), seed=config.seed)
    train = prepare(
        tiny_arch,
        task_set(
            task,
            paper,
            splits.train,
            n_pairs=4,
            seed=derive_seed(config.seed, "size", "utility", 4, 1),
        ),
        config.position,
        config,
    )
    report = prepare(
        tiny_arch,
        task_set(
            task,
            paper,
            splits.report,
            n_pairs=eval_pairs(task, splits.report, config.n_eval_pairs),
            seed=eval_seed(task, "report", config.seed),
        ),
        config.position,
        config,
    )
    sweep = sweep_from_activations(
        [train],
        {"report": report},
        method=config.method,
        preprocess=config.preprocess,
    )
    arrays = run_dir.read_arrays("sizes_utility")
    np.testing.assert_allclose(
        arrays["accuracy"][1, 1, 0, 1], sweep.per_seed("report")[0, 0]
    )
    np.testing.assert_allclose(
        arrays["agreement"][1, 1],
        directional_agreement(oriented_differences(train)),
    )


def test_the_table_has_a_row_per_task_and_size(
    finished: tuple[Config, RunDir],
) -> None:
    _, run_dir = finished
    rows = _json(run_dir, "table")
    assert [(r["task"], r["size"]) for r in rows] == [
        (task, size) for task in TASKS for size in RUN_SIZES[task]
    ]
    assert all(list(row) == list(COLUMNS) for row in rows)
    assert len(rows) == 13


def test_the_table_rows_summarize_the_arrays_at_the_task_layer(
    finished: tuple[Config, RunDir],
) -> None:
    config, run_dir = finished
    rows = _json(run_dir, "table")
    summary = _json(run_dir, "summary")
    for task in TASKS:
        arrays = run_dir.read_arrays(f"sizes_{task}")
        layer = summary["layers"][task]
        at = arrays["layers"].tolist().index(layer)
        curves = run_dir.read_arrays(f"curves_{task}")
        for z, size in enumerate(RUN_SIZES[task]):
            (row,) = (
                r for r in rows if r["task"] == task and r["size"] == size
            )
            report = arrays["accuracy"][z, 1, 0, :, at]
            agreement = arrays["agreement"][z, :, at]
            cosine = arrays["cosine"][z, :, at]
            assert row["target"] == _target(task)
            assert row["layer"] == layer
            assert row["accuracy"] == pytest.approx(report.mean())
            assert row["lo"] == pytest.approx(curves["lo"][0, z])
            assert row["hi"] == pytest.approx(curves["hi"][0, z])
            assert row["lo"] <= row["accuracy"] <= row["hi"]
            assert row["sd"] == pytest.approx(report.std(ddof=1))
            assert row["q05"] == pytest.approx(np.quantile(report, 0.05))
            assert row["q95"] == pytest.approx(np.quantile(report, 0.95))
            assert row["failure"] == pytest.approx(
                (report < config.failure_accuracy).mean()
            )
            assert row["shuffled"] == pytest.approx(
                arrays["accuracy"][z, 1, 2, :, at].mean()
            )
            assert row["cosine"] == pytest.approx(cosine.mean())
            assert row["cosine_sd"] == pytest.approx(cosine.std(ddof=1))
            assert row["agreement"] == pytest.approx(agreement.mean())
            rho = spearman(agreement, report)
            if math.isnan(rho):
                assert row["rho"] is None
            else:
                assert row["rho"] == pytest.approx(rho)


def test_the_curves_show_the_report_accuracy_and_the_pair_statistics(
    finished: tuple[Config, RunDir],
) -> None:
    config, run_dir = finished
    summary = _json(run_dir, "summary")
    for task in TASKS:
        arrays = run_dir.read_arrays(f"sizes_{task}")
        at = arrays["layers"].tolist().index(summary["layers"][task])
        sizes = np.array(RUN_SIZES[task], dtype=np.float64)
        curves = run_dir.read_arrays(f"curves_{task}")
        np.testing.assert_array_equal(curves["x"], sizes)
        expected = np.moveaxis(arrays["accuracy"][:, 1, :, :, at], 0, -1)
        np.testing.assert_array_equal(curves["per_seed"], expected)
        lo, hi = bootstrap_ci(
            np.moveaxis(expected, 1, 0),
            n_boot=config.n_boot,
            seed=derive_seed(config.seed, "curves", f"curves_{task}"),
        )
        np.testing.assert_allclose(curves["lo"], lo)
        np.testing.assert_allclose(curves["hi"], hi)
        sidecar = _json(run_dir, f"curves_{task}")
        assert sidecar["series"] == list(SERIES)
        assert sidecar["xlabel"] == "Training pairs"
        assert sidecar["chance"] == 0.5
        statistics = run_dir.read_arrays(f"curves_{task}_cosine")
        np.testing.assert_array_equal(statistics["x"], sizes)
        np.testing.assert_array_equal(
            statistics["per_seed"][0], arrays["cosine"][:, :, at].T
        )
        np.testing.assert_array_equal(
            statistics["per_seed"][1], arrays["agreement"][:, :, at].T
        )
        assert _json(run_dir, f"curves_{task}_cosine")["series"] == [
            "cosine to reference",
            "pair agreement",
        ]


def test_the_sources_curve_scores_each_source_at_its_own_layer(
    finished: tuple[Config, RunDir],
) -> None:
    _, run_dir = finished
    summary = _json(run_dir, "summary")
    curves = run_dir.read_arrays("curves_sources_truth")
    sidecar = _json(run_dir, "curves_sources_truth")
    assert sidecar["series"] == SOURCES_OF_TRUTH
    assert sidecar["xlabel"] == "Training pairs"
    assert sidecar["chance"] == 0.5
    assert curves["x"].tolist() == [2.0, 4.0]
    for i, name in enumerate(SOURCES_OF_TRUTH):
        arrays = run_dir.read_arrays(f"sizes_{name}")
        at = arrays["layers"].tolist().index(summary["layers"][name])
        np.testing.assert_array_equal(
            curves["per_seed"][i], arrays["accuracy"][:2, 1, 0, :, at].T
        )


def test_the_sources_matrix_holds_the_cosines_of_the_reference_directions(
    tiny_arch: LoadedModel, finished: tuple[Config, RunDir]
) -> None:
    config, run_dir = finished
    summary = _json(run_dir, "summary")
    at = summary["layers"]["truth"] - 1
    unit = []
    for name in SOURCES_OF_TRUTH:
        reader = _reference_sweep(tiny_arch, config, name).readers[0]
        vector = (reader.directions * reader.signs[:, None])[at]
        unit.append(vector / np.linalg.norm(vector))
    expected = np.stack(unit) @ np.stack(unit).T
    values = run_dir.read_arrays("sources_truth")["values"]
    sidecar = _json(run_dir, "sources_truth")
    assert sidecar["rows"] == sidecar["cols"] == SOURCES_OF_TRUTH
    assert values.shape == (3, 3)
    np.testing.assert_allclose(np.diag(values), 1.0, atol=1e-6)
    np.testing.assert_allclose(values, values.T, atol=1e-6)
    np.testing.assert_allclose(values, expected, atol=1e-5)


def test_a_target_without_a_second_source_is_listed_as_skipped(
    finished: tuple[Config, RunDir],
) -> None:
    _, run_dir = finished
    skipped = _json(run_dir, "summary")["skipped_sources"]
    assert sorted(skipped) == ["sentiment", "utility"]
    for target, reason in skipped.items():
        assert target in reason
    assert not (run_dir.path / "sources_sentiment.npz").exists()
    assert not (run_dir.path / "curves_sources_utility.npz").exists()


def test_the_summary_lists_the_skipped_sizes_and_the_saturation(
    finished: tuple[Config, RunDir],
) -> None:
    config, run_dir = finished
    summary = _json(run_dir, "summary")
    assert set(summary) == {
        "tasks",
        "targets",
        "layers",
        "reference",
        "skipped_sizes",
        "skipped_sources",
        "saturation",
        "failure_auroc",
    }
    assert summary["tasks"] == list(TASKS)
    assert summary["targets"] == {t: _target(t) for t in TASKS}
    assert summary["skipped_sizes"] == {
        "sentiment": [],
        "truth": [],
        "truth_cities": [8],
        "truth_common": [8],
        "utility": [],
    }
    for task in TASKS:
        arrays = run_dir.read_arrays(f"sizes_{task}")
        at = arrays["layers"].tolist().index(summary["layers"][task])
        point, lower, upper = saturation_ci(
            RUN_SIZES[task],
            arrays["accuracy"][:, 1, 0, :, at],
            delta=config.saturation_delta,
            n_boot=config.n_boot,
            seed=derive_seed(config.seed, "saturation", task),
        )
        assert summary["saturation"][task] == {
            "size": point,
            "lo": lower,
            "hi": upper,
            "delta": config.saturation_delta,
        }


def test_the_failure_auroc_ranks_successes_by_their_pair_agreement(
    finished: tuple[Config, RunDir],
) -> None:
    config, run_dir = finished
    summary = _json(run_dir, "summary")
    got = summary["failure_auroc"]
    assert list(got) == list(TASKS)
    for task in TASKS:
        arrays = run_dir.read_arrays(f"sizes_{task}")
        at = arrays["layers"].tolist().index(summary["layers"][task])
        success = (
            arrays["accuracy"][:, 1, 0, :, at] >= config.failure_accuracy
        ).reshape(-1)
        agreement = arrays["agreement"][:, :, at].reshape(-1)
        if success.all() or not success.any():
            assert got[task] is None
        else:
            expected = auroc(agreement, success.astype(np.int64))
            assert got[task] == pytest.approx(float(expected))
    assert any(value is not None for value in got.values())


@pytest.mark.parametrize(
    "failure_accuracy", [0.0, 1.5], ids=["all_succeed", "all_fail"]
)
def test_the_failure_auroc_is_null_when_only_one_outcome_occurs(
    tiny_arch: LoadedModel, tmp_path: Path, failure_accuracy: float
) -> None:
    config = _config(
        tasks=("utility",),
        sizes=(2, 4),
        n_seeds=2,
        failure_accuracy=failure_accuracy,
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _sources())
    assert _json(run_dir, "summary")["failure_auroc"] == {"utility": None}


def test_a_resample_at_the_failure_accuracy_has_not_failed(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    base = _config(tasks=("utility",), sizes=(2, 4), n_seeds=4)
    first = RunDir.create(base, out_root=tmp_path / "first")
    execute(base, first, tiny_arch, _sources())
    at = _json(first, "summary")["layers"]["utility"] - 1
    accuracy = first.read_arrays("sizes_utility")["accuracy"]
    report = accuracy[:, 1, 0, :, at]
    threshold = float(np.sort(report.reshape(-1))[report.size // 2])
    config = dataclasses.replace(base, failure_accuracy=threshold)
    second = RunDir.create(config, out_root=tmp_path / "second")
    execute(config, second, tiny_arch, _sources())
    assert (report == threshold).any()
    for z, row in enumerate(_json(second, "table")):
        assert row["failure"] == pytest.approx((report[z] < threshold).mean())
    success = (report >= threshold).reshape(-1)
    agreement = second.read_arrays("sizes_utility")["agreement"][:, :, at]
    expected = (
        None
        if success.all() or not success.any()
        else float(auroc(agreement.reshape(-1), success.astype(np.int64)))
    )
    got = _json(second, "summary")["failure_auroc"]["utility"]
    assert got == (expected if expected is None else pytest.approx(expected))


def test_execute_is_deterministic(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(tasks=("utility",), sizes=(2, 4), n_seeds=2)
    first = RunDir.create(config, out_root=tmp_path / "a")
    second = RunDir.create(config, out_root=tmp_path / "b")
    execute(config, first, tiny_arch, _sources())
    execute(config, second, tiny_arch, _sources())
    for stem in ("sizes_utility", "curves_utility", "curves_utility_cosine"):
        a, b = first.read_arrays(stem), second.read_arrays(stem)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    assert _json(first, "table") == _json(second, "table")
    assert _json(first, "summary") == _json(second, "summary")


@pytest.fixture
def no_forward_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the model ran before the settings were checked")

    monkeypatch.setattr(p3_size, "prepare", refuse)


@pytest.mark.usefixtures("no_forward_pass")
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"tasks": ("emotion_happiness",)},
            "p3 reads binary tasks only, not 'emotion_happiness'",
        ),
        ({"position": "response"}, "'response' position"),
        ({"eval_position": "response"}, "'response' position"),
        ({"n_seeds": 1}, "n_seeds must be at least 2"),
        ({"sizes": (4, 2, 8)}, "sizes must be increasing"),
        ({"sizes": (2, 2, 8)}, "sizes must be increasing"),
        ({"sizes": (4,)}, "sizes must be increasing"),
        ({"sizes": (1, 2)}, "sizes must be increasing"),
        ({"tasks": ("nonsense",)}, "unknown task 'nonsense'"),
        (
            {"sizes": (2, 8, 16)},
            "task 'truth_cities' has 6 training pairs, fewer than two of "
            "the sizes (2, 8, 16)",
        ),
    ],
    ids=[
        "one_vs_rest",
        "response",
        "eval_response",
        "one_seed",
        "unsorted_sizes",
        "repeated_size",
        "one_size",
        "size_below_two",
        "unknown_task",
        "too_few_sizes_fit",
    ],
)
def test_execute_rejects_invalid_settings_before_any_forward_pass(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    overrides: dict[str, Any],
    message: str,
) -> None:
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=tmp_path)
    with pytest.raises(ValueError, match=re.escape(message)):
        execute(config, run_dir, tiny_arch, _sources())


def test_plot_draws_every_curve_and_the_sources_matrix(
    finished: tuple[Config, RunDir], tmp_path: Path
) -> None:
    _, run_dir = finished
    paths = plot(run_dir, tmp_path)
    stems = [
        *(f"curves_{task}" for task in TASKS),
        *(f"curves_{task}_cosine" for task in TASKS),
        "curves_sources_truth",
        "sources_truth",
    ]
    assert sorted(p.name for p in paths) == sorted(
        f"{stem}.{ext}" for stem in stems for ext in ("pdf", "png", "svg")
    )
    assert len(paths) == 36
    assert all(p.stat().st_size > 0 for p in paths)


def test_size_curves_are_drawn_on_a_log2_axis(
    finished: tuple[Config, RunDir],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, run_dir = finished
    figures = []
    save = p3_size.save_figure

    def capture(
        fig: Figure, out_dir: Path, name: str, **kwargs: Any
    ) -> list[Path]:
        figures.append(fig)
        return save(fig, out_dir, name, **kwargs)

    monkeypatch.setattr(p3_size, "save_figure", capture)
    paths = plot_size_curves(run_dir, tmp_path, stem="curves_sentiment")
    assert [p.name for p in paths] == [
        "curves_sentiment.pdf",
        "curves_sentiment.png",
        "curves_sentiment.svg",
    ]
    (fig,) = figures
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    np.testing.assert_allclose(
        ax.xaxis.get_transform().transform(np.array([2.0, 4.0, 8.0])),
        [1.0, 2.0, 3.0],
    )
    assert ax.get_xlabel() == "Training pairs"


def test_run_loads_the_model_and_the_sources_of_tasks_and_targets(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: dict[str, Any] = {}

    def fake_load_model(key: str) -> LoadedModel:
        asked["model"] = key
        return tiny_arch

    def fake_load_sources(sources: Sequence[str]) -> dict[str, list[Example]]:
        asked["sources"] = list(sources)
        return _sources()

    monkeypatch.setattr(p3_size, "load_model", fake_load_model)
    monkeypatch.setattr(p3_size, "load_sources", fake_load_sources)
    config = _config(
        tasks=("truth_cities", "truth_common"), sizes=(2, 4), n_seeds=2
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    run(config, run_dir)
    assert asked["model"] == "tiny"
    assert sorted(set(asked["sources"])) == ["facts", "geometry_of_truth"]
    assert _json(run_dir, "summary")["skipped_sources"] == {
        "truth": "'truth' is not in tasks"
    }
    assert not any(
        p.name.startswith("sources_") for p in run_dir.path.iterdir()
    )
    assert run_dir.meta["finished_at"] is None


@pytest.mark.slow
def test_gpt2_reads_sentiment_over_three_sizes(tmp_path: Path) -> None:
    config = Config(
        tasks=("sentiment",),
        sizes=(2, 8, 32),
        n_seeds=4,
        n_eval_pairs=64,
        reference_pairs=64,
        n_boot=100,
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    loaded = load_model("gpt2")
    examples = load_sources(["sst2"])
    start = time.perf_counter()
    execute(config, run_dir, loaded, examples)
    seconds = time.perf_counter() - start
    assert _json(run_dir, "summary")["reference"]["sentiment"]["report"] > 0.5
    assert [row["size"] for row in _json(run_dir, "table")] == [2, 8, 32]
    print(f"p3: {seconds / len(config.sizes):.1f} s per size on gpt2")
