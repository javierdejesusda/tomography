import numpy as np
import pytest

from tomography.datasets.records import Example
from tomography.experiments import c1_sentiment_steering
from tomography.experiments import c2_emotion_steering
from tomography.experiments import c4_honesty_control
from tomography.experiments import c5_gallery
from tomography.experiments import m1_methods
from tomography.experiments import p7_extraction_steering
from tomography.experiments import p9d_steering
from tomography.experiments.c4_honesty_control import calibration_texts
from tomography.experiments.c4_honesty_control import neutral_texts
from tomography.experiments.controls import max_abs
from tomography.experiments.controls import random_readers
from tomography.experiments.controls import recording
from tomography.experiments.effects import SideEffects
from tomography.experiments.effects import effects_row
from tomography.reading.reader import RepReader

# config_hash of each steering experiment on gpt2 before the refactor.
CONFIG_HASHES = {
    c1_sentiment_steering: "add59c39f574",
    c2_emotion_steering: "f009922c8c55",
    c4_honesty_control: "8dfb58fa4cf6",
    c5_gallery: "55bba5195878",
    m1_methods: "cc334aabf16f",
    p7_extraction_steering: "e1f366fd82be",
}
P9D_HASH = "92bd18d4c7a3"

EFFECTS = SideEffects(
    perplexity=12.5,
    fluency=-3.25,
    distinct_1=0.5,
    distinct_2=0.75,
    mean_length=7.0,
    empty_rate=0.125,
)


def _legacy_draws(
    seed: int, n: int, shape: tuple[int, int]
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape) for _ in range(n)]


def test_random_readers_draw_one_stream_in_order() -> None:
    readers = random_readers((3, 5, 8), 6, 4, seed=11)

    expected = _legacy_draws(11, 4, (3, 6))
    assert len(readers) == 4
    for reader, raw in zip(readers, expected, strict=True):
        unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        assert reader.method == "random"
        assert reader.layers == (3, 5, 8)
        np.testing.assert_allclose(reader.directions, unit, atol=1e-6)
        np.testing.assert_array_equal(reader.means, 0.0)


def test_random_readers_match_the_reader_shaped_call_of_c4() -> None:
    reference = RepReader.from_arrays(
        "lat",
        (2, 4),
        np.arange(1.0, 9.0).reshape(2, 4),
        np.zeros((2, 4)),
    )

    new = random_readers(
        reference.layers, reference.directions.shape[1], 3, seed=5
    )

    zeros = np.zeros_like(reference.directions)
    rng = np.random.default_rng(5)
    old = [
        RepReader.from_arrays(
            "random",
            reference.layers,
            rng.standard_normal(reference.directions.shape),
            zeros,
        )
        for _ in range(3)
    ]
    for a, b in zip(new, old, strict=True):
        np.testing.assert_array_equal(a.directions, b.directions)
        np.testing.assert_array_equal(a.means, b.means)
        assert a.layers == b.layers


def test_random_readers_differ_across_seeds_and_arms() -> None:
    first = random_readers((1,), 4, 2, seed=0)
    second = random_readers((1,), 4, 2, seed=1)

    assert not np.array_equal(first[0].directions, first[1].directions)
    assert not np.array_equal(first[0].directions, second[0].directions)


def test_max_abs_takes_the_largest_finite_magnitude() -> None:
    values = np.array([[1.0, -4.5], [np.nan, 2.0], [np.inf, -np.inf]])

    assert max_abs(values) == 4.5


def test_max_abs_is_nan_without_a_finite_value() -> None:
    assert np.isnan(max_abs(np.array([np.nan, np.inf])))
    assert np.isnan(max_abs(np.empty((0, 3))))


def test_effects_row_puts_the_lead_values_before_the_six_effects() -> None:
    row = effects_row(EFFECTS, "random", 2, None, 0.5)

    assert row == ["random", 2, None, 0.5, 12.5, -3.25, 0.5, 0.75, 7.0, 0.125]


def test_effects_row_keeps_a_missing_perplexity() -> None:
    row = effects_row(SideEffects(None, -1.0, 0.1, 0.2, 3.0, 0.0), "none")

    assert row == ["none", None, -1.0, 0.1, 0.2, 3.0, 0.0]


@pytest.mark.parametrize("module", list(CONFIG_HASHES))
def test_the_config_hash_is_the_one_before_the_refactor(module) -> None:
    config = module.Config(model="gpt2")

    assert config.config_hash() == CONFIG_HASHES[module]


def test_the_p9d_config_hash_is_the_one_before_the_refactor() -> None:
    config = p9d_steering.Config(model="gpt2", labels_run="x")

    assert config.config_hash() == P9D_HASH


def _alpaca() -> list[Example]:
    return [
        Example(
            uid=f"alpaca-test-{i}",
            text=f"The test answer {i} is a short reply.",
            label=-1,
            split="test",
            meta={"instruction": f"Name item {i}."},
        )
        for i in range(30)
    ]


@pytest.mark.parametrize(
    ("n_calibration", "n_neutral"), [(5, 4), (8, 30), (3, 1)]
)
def test_neutral_texts_are_the_tail_of_the_calibration_draw_of_c5(
    n_calibration: int, n_neutral: int
) -> None:
    rows = _alpaca()
    drawn = calibration_texts(rows, n_calibration + n_neutral, seed=9)

    assert (
        neutral_texts(rows, n_calibration, n_neutral, seed=9)
        == (drawn[n_calibration:])
    )


def test_recording_keeps_the_arm_built_for_each_coefficient() -> None:
    built: list[float] = []
    kept: dict[float, object] = {}

    def make(coeff: float) -> object:
        built.append(coeff)
        return object()

    record = recording(make, kept)
    first = record(0.5)
    second = record(-0.5)

    assert built == [0.5, -0.5]
    assert kept[0.5] is first
    assert kept[-0.5] is second


def test_recording_replaces_the_arm_of_a_repeated_coefficient() -> None:
    kept: dict[float, object] = {}
    record = recording(lambda coeff: object(), kept)

    record(1.0)
    latest = record(1.0)

    assert kept == {1.0: latest}
