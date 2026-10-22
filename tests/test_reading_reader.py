from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.reading.directions import Method
from tomography.reading.metrics import group_accuracy
from tomography.reading.reader import RepReader
from tomography.reading.reader import orient


def small_reader() -> RepReader:
    return RepReader.from_arrays(
        "hand",
        [4, 9],
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 1.0], [0.0, 0.0]]),
        np.array([1.0, -1.0]),
    )


def small_h() -> NDArray[np.float32]:
    return np.array(
        [[[2, 3], [0, 0], [1, 5]], [[1, 2], [3, 4], [0, 1]]],
        dtype=np.float32,
    )


def test_score_applies_the_mean_the_direction_and_the_sign() -> None:
    scores = small_reader().score(small_h())

    assert scores.dtype == np.float32
    np.testing.assert_allclose(scores, [[1, -1, 0], [-2, -4, -1]])


def test_raw_score_leaves_out_the_sign() -> None:
    raw = small_reader().raw_score(small_h())

    assert raw.dtype == np.float32
    np.testing.assert_allclose(raw, [[1, -1, 0], [2, 4, 1]])


def test_from_arrays_stores_float32_unit_directions() -> None:
    reader = RepReader.from_arrays(
        "prompt_diff",
        (7,),
        np.array([[3.0, 4.0]]),
        np.array([[0.5, 0.5]]),
    )

    assert reader.method == "prompt_diff"
    assert reader.layers == (7,)
    assert reader.directions.dtype == np.float32
    np.testing.assert_allclose(reader.directions, [[0.6, 0.8]], atol=1e-6)
    assert reader.means.dtype == np.float32
    assert reader.signs.dtype == np.float32
    np.testing.assert_array_equal(reader.signs, [1.0])


def test_from_arrays_copies_its_inputs() -> None:
    directions = np.array([[1.0, 0.0]], dtype=np.float32)

    reader = RepReader.from_arrays(
        "copy", [0], directions, np.zeros((1, 2), dtype=np.float32)
    )
    directions[0, 0] = 5.0

    assert reader.directions[0, 0] == 1.0


@pytest.mark.parametrize(
    ("layers", "directions", "means", "signs", "message"),
    [
        ([0], np.ones(2), np.ones((1, 2)), None, "directions"),
        ([0], np.ones((1, 2)), np.ones((1, 3)), None, "means"),
        ([0, 1], np.ones((1, 2)), np.ones((1, 2)), None, "layers"),
        ([0], np.ones((1, 2)), np.ones((1, 2)), np.array([0.5]), "signs"),
        ([0], np.ones((1, 2)), np.ones((1, 2)), np.ones(2), "signs"),
        ([0], np.zeros((1, 2)), np.ones((1, 2)), None, "zero"),
    ],
    ids=[
        "directions-rank",
        "means-shape",
        "layers-length",
        "sign-value",
        "signs-shape",
        "zero-direction",
    ],
)
def test_from_arrays_rejects_inconsistent_arrays(
    layers: list[int],
    directions: NDArray[np.float64],
    means: NDArray[np.float64],
    signs: NDArray[np.float64] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RepReader.from_arrays("bad", layers, directions, means, signs)


@pytest.mark.parametrize(
    "shape",
    [(3, 2), (1, 3, 2), (2, 3, 5)],
    ids=["two-dimensional", "wrong-layer-count", "wrong-width"],
)
def test_score_rejects_activations_of_the_wrong_shape(
    shape: tuple[int, ...],
) -> None:
    reader = small_reader()

    with pytest.raises(ValueError, match="shape"):
        reader.score(np.zeros(shape, dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        reader.raw_score(np.zeros(shape, dtype=np.float32))


def x_rows(*x_values: float) -> NDArray[np.float32]:
    """Builds one layer of 2-d rows that vary along the x axis only."""
    h = np.zeros((1, len(x_values), 2), dtype=np.float32)
    h[0, :, 0] = x_values
    return h


def fitted_sign(
    x_values: list[float],
    positives: list[int],
    *,
    groups: list[list[int]] | None = None,
    pairs: list[list[int]] | None = None,
) -> float:
    """Fits pca_raw on x-only rows, so the score is the centered x value."""
    labels = np.zeros(len(x_values), dtype=np.int64)
    labels[positives] = 1
    reader = RepReader.fit(
        "pca_raw",
        x_rows(*x_values),
        [0],
        labels=labels,
        pairs=None if pairs is None else np.array(pairs, dtype=np.int64),
        groups=None if groups is None else np.array(groups, dtype=np.int64),
    )
    np.testing.assert_allclose(np.abs(reader.directions), [[1.0, 0.0]])
    return float(reader.signs[0])


X = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


def test_sign_is_positive_when_the_positive_member_is_usually_highest() -> None:
    groups = [[0, 1], [2, 3], [4, 5]]

    assert fitted_sign(X, [1, 3, 4], groups=groups) == 1.0


def test_sign_is_negative_when_the_positive_member_is_usually_lowest() -> None:
    groups = [[0, 1], [2, 3], [4, 5]]

    assert fitted_sign(X, [0, 2, 5], groups=groups) == -1.0


def test_a_tie_between_highest_and_lowest_counts_as_positive() -> None:
    groups = [[0, 1], [2, 3]]

    assert fitted_sign(X, [1, 2], groups=groups) == 1.0


def test_sign_rule_handles_groups_of_more_than_two() -> None:
    groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    assert fitted_sign(X, [0, 3, 7], groups=groups) == -1.0
    assert fitted_sign(X, [2, 5, 7], groups=groups) == 1.0
    assert fitted_sign(X, [2, 3, 7], groups=groups) == 1.0


def test_sign_rule_ignores_padding_members() -> None:
    groups = [[0, 1, -1], [2, 3, -1]]

    assert fitted_sign(X, [0, 2], groups=groups) == -1.0


def test_members_tied_for_the_maximum_all_count_as_highest() -> None:
    x_values = [3.0, 3.0, 1.0, 5.0, 2.0, 0.0]
    groups = [[0, 1, 2], [3, 4, 5]]

    assert fitted_sign(x_values, [1, 5], groups=groups) == 1.0


def test_pairs_with_different_labels_are_the_default_groups() -> None:
    pairs = [[0, 1], [2, 3], [4, 5]]

    assert fitted_sign(X, [0, 2, 5], pairs=pairs) == -1.0
    assert fitted_sign(X, [1, 3, 4], pairs=pairs) == 1.0


def test_pairs_with_equal_labels_are_not_used_for_the_sign() -> None:
    pairs = [[0, 1], [2, 3], [4, 5], [6, 7]]

    assert fitted_sign(X, [0, 2, 4, 5, 6, 7], pairs=pairs) == -1.0


def test_explicit_groups_take_precedence_over_pairs() -> None:
    pairs = [[0, 1], [2, 3]]
    groups = [[4, 5], [6, 7]]

    assert fitted_sign(X, [0, 2, 4, 6], pairs=pairs, groups=groups) == -1.0
    assert fitted_sign(X, [0, 2, 5, 7], pairs=pairs, groups=groups) == 1.0


def test_without_groups_the_sign_orders_the_class_mean_scores() -> None:
    assert fitted_sign(X, [6, 7, 8, 9]) == 1.0
    assert fitted_sign(X, [0, 1, 2, 3]) == -1.0
    assert fitted_sign([0.0, 1.0, 2.0, 3.0], [0, 3]) == 1.0


def test_pairs_that_all_share_a_label_fall_back_to_class_means() -> None:
    assert fitted_sign(X, [0, 1, 2, 3], pairs=[[0, 1], [2, 3]]) == -1.0


def test_each_layer_gets_its_own_sign() -> None:
    h = x_rows(*X)
    h = np.concatenate([h, -h])
    labels = np.zeros(10, dtype=np.int64)
    labels[[6, 7, 8, 9]] = 1

    reader = RepReader.fit("pca_raw", h, [3, 8], labels=labels)

    assert reader.layers == (3, 8)
    np.testing.assert_array_equal(reader.signs, [1.0, -1.0])
    scores = reader.score(h)
    assert (scores[:, labels == 1].mean(axis=1) > 0).all()
    assert (scores[:, labels == 0].mean(axis=1) < 0).all()


def test_fit_stores_the_arrays_of_each_layer() -> None:
    rng = np.random.default_rng(0)
    h = rng.standard_normal((3, 20, 6)).astype(np.float32)
    labels = (np.arange(20) % 2).astype(np.int64)

    reader = RepReader.fit("mean_diff", h, [1, 2, 3], labels=labels)

    assert reader.method == "mean_diff"
    assert reader.directions.shape == (3, 6)
    assert reader.means.shape == (3, 6)
    assert reader.signs.shape == (3,)
    np.testing.assert_allclose(
        reader.means, h.astype(np.float64).mean(axis=1), atol=1e-5
    )
    np.testing.assert_allclose(np.linalg.norm(reader.directions, axis=1), 1.0)


Orientation = dict[str, NDArray[np.int64]]


def orientation_case(
    source: str,
) -> tuple[NDArray[np.float32], NDArray[np.int64], Orientation]:
    """Four layers of random rows with labels and one orientation source.

    The groups have one label-1 member each; half of the pairs share a
    label, so only the other half orient.
    """
    rng = np.random.default_rng(31)
    h = rng.standard_normal((4, 24, 6)).astype(np.float32)
    if source == "groups":
        labels = (np.arange(24) % 3 == 1).astype(np.int64)
        return h, labels, {"groups": np.arange(24).reshape(8, 3)}
    labels = rng.permutation(np.tile([0, 1], 12)).astype(np.int64)
    if source == "pairs":
        return h, labels, {"pairs": np.arange(24).reshape(12, 2)}
    return h, labels, {}


def reference_signs(
    projections: NDArray[np.float64],
    labels: NDArray[np.int64],
    orientation: Orientation,
) -> list[float]:
    """The sign rule of `RepReader.fit`, written out group by group."""
    if "pairs" in orientation:
        pairs = orientation["pairs"]
        groups = pairs[labels[pairs[:, 0]] != labels[pairs[:, 1]]]
    else:
        groups = orientation.get("groups")
    signs = []
    for row in projections:
        if groups is None:
            gap = row[labels == 1].mean() - row[labels == 0].mean()
            signs.append(-1.0 if gap < 0 else 1.0)
            continue
        top = bottom = 0
        for group in groups:
            own = row[group[labels[group] == 1][0]]
            top += own == row[group].max()
            bottom += own == row[group].min()
        signs.append(-1.0 if top < bottom else 1.0)
    return signs


@pytest.mark.parametrize("source", ["groups", "pairs", "class-means"])
def test_fit_signs_follow_the_reference_rule_on_every_layer(
    source: str,
) -> None:
    h, labels, orientation = orientation_case(source)

    reader = RepReader.fit(
        "random", h, [0, 1, 2, 3], labels=labels, seed=3, **orientation
    )

    centered = h.astype(np.float64) - reader.means[:, None].astype(np.float64)
    projections = np.einsum(
        "knd,kd->kn", centered, reader.directions.astype(np.float64)
    )
    expected = reference_signs(projections, labels, orientation)
    assert set(expected) == {-1.0, 1.0}
    np.testing.assert_array_equal(reader.signs, expected)


@pytest.mark.parametrize("source", ["groups", "pairs", "class-means"])
def test_orient_sets_the_signs_that_fit_sets(source: str) -> None:
    h, labels, orientation = orientation_case(source)
    fitted = RepReader.fit(
        "random", h, [0, 1, 2, 3], labels=labels, seed=3, **orientation
    )
    unsigned = replace(
        fitted, method="hand", signs=np.ones(4, dtype=np.float32)
    )

    oriented = orient(unsigned, h, labels=labels, **orientation)

    assert set(fitted.signs.tolist()) == {-1.0, 1.0}
    assert oriented.signs.dtype == np.float32
    np.testing.assert_array_equal(oriented.signs, fitted.signs)
    assert oriented.method == "hand"
    assert oriented.layers == fitted.layers
    np.testing.assert_array_equal(oriented.directions, fitted.directions)
    np.testing.assert_array_equal(oriented.means, fitted.means)
    np.testing.assert_array_equal(unsigned.signs, np.ones(4))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"labels": np.zeros(23, dtype=np.int64)}, "labels"),
        ({"h": np.zeros((4, 24, 5), dtype=np.float32)}, "shape"),
        ({"groups": np.array([[0, 2, 3]])}, "exactly one"),
        ({"pairs": np.array([[0, 99]])}, "pairs"),
        ({"labels": np.zeros(24, dtype=np.int64)}, "both classes"),
    ],
    ids=["label-count", "width", "group", "pairs", "one-class"],
)
def test_orient_rejects_inconsistent_inputs(
    change: dict[str, NDArray[np.generic]], message: str
) -> None:
    h, labels, _ = orientation_case("groups")
    reader = RepReader.fit("random", h, [0, 1, 2, 3], labels=labels)
    arguments = {"h": h, "labels": labels, **change}

    with pytest.raises(ValueError, match=message):
        orient(reader, **arguments)  # type: ignore[arg-type]


def test_random_directions_differ_across_layers_but_follow_the_seed() -> None:
    rng = np.random.default_rng(1)
    h = np.tile(rng.standard_normal((1, 12, 8)), (3, 1, 1)).astype(np.float32)
    labels = (np.arange(12) % 2).astype(np.int64)

    first = RepReader.fit("random", h, [0, 1, 2], labels=labels, seed=4)
    again = RepReader.fit("random", h, [0, 1, 2], labels=labels, seed=4)
    other = RepReader.fit("random", h, [0, 1, 2], labels=labels, seed=5)

    np.testing.assert_array_equal(first.directions, again.directions)
    assert not np.allclose(first.directions[0], first.directions[1])
    assert not np.allclose(first.directions, other.directions)
    assert not np.allclose(first.directions[1], other.directions[0])


def test_fit_passes_pairs_and_preprocess_to_the_finder() -> None:
    rng = np.random.default_rng(2)
    h = rng.standard_normal((1, 16, 5)).astype(np.float32)
    labels = np.tile([1, 0], 8).astype(np.int64)
    pairs = np.arange(16, dtype=np.int64).reshape(8, 2)

    centered = RepReader.fit("pca_diff", h, [0], labels=labels, pairs=pairs)
    raw = RepReader.fit(
        "pca_diff", h, [0], labels=labels, pairs=pairs, preprocess="none"
    )

    assert not np.allclose(centered.directions, raw.directions)


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        ([[0, 2]], "exactly one"),
        ([[1, 3]], "exactly one"),
        ([[0, 99]], "groups"),
        ([0, 1], "groups"),
        ([[1, -2]], "groups"),
    ],
    ids=[
        "no-positive",
        "two-positives",
        "out-of-range",
        "one-dimensional",
        "below-padding",
    ],
)
def test_fit_rejects_groups_without_exactly_one_positive_member(
    groups: list[list[int]], message: str
) -> None:
    labels = np.zeros(10, dtype=np.int64)
    labels[[1, 3]] = 1

    with pytest.raises(ValueError, match=message):
        RepReader.fit(
            "pca_raw",
            x_rows(*X),
            [0],
            labels=labels,
            groups=np.array(groups, dtype=np.int64),
        )


@pytest.mark.parametrize(
    ("orientation", "positives"),
    [
        ({"groups": [[0, 2]]}, [1, 3]),
        ({"pairs": [[0, 99]]}, [1, 3]),
        ({}, []),
    ],
    ids=["no-positive", "pairs-out-of-range", "one-class"],
)
def test_fit_validates_orientation_before_fitting_directions(
    monkeypatch: pytest.MonkeyPatch,
    orientation: dict[str, list[list[int]]],
    positives: list[int],
) -> None:
    def forbidden(*args: object, **options: object) -> None:
        raise AssertionError("fit_direction must not run")

    monkeypatch.setattr("tomography.reading.reader.fit_direction", forbidden)
    y = np.zeros(10, dtype=np.int64)
    y[positives] = 1

    with pytest.raises(ValueError, match=r"exactly one|pairs|classes"):
        RepReader.fit(
            "pca_raw",
            x_rows(*X),
            [0],
            labels=y,
            **{k: np.array(v, dtype=np.int64) for k, v in orientation.items()},
        )


@pytest.mark.parametrize(
    "pairs",
    [[[0, -1]], [[0, 99]], [[0, 1, 2], [3, 4, 5]]],
    ids=["negative-index", "out-of-range", "three-columns"],
)
def test_fit_rejects_malformed_pairs(pairs: list[list[int]]) -> None:
    labels = np.zeros(10, dtype=np.int64)
    labels[[1, 3]] = 1

    with pytest.raises(ValueError, match="pairs"):
        RepReader.fit(
            "pca_raw",
            x_rows(*X),
            [0],
            labels=labels,
            pairs=np.array(pairs, dtype=np.int64),
        )


@pytest.mark.parametrize(
    ("h", "layers", "labels", "message"),
    [
        (np.zeros((10, 2)), [0], np.zeros(10), "shape"),
        (np.zeros((2, 10, 2)), [0], np.zeros(10), "layers"),
        (np.zeros((1, 10, 2)), [0], np.zeros(9), "labels"),
    ],
    ids=["two-dimensional", "layer-count", "label-length"],
)
def test_fit_rejects_inconsistent_shapes(
    h: NDArray[np.float64],
    layers: list[int],
    labels: NDArray[np.float64],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RepReader.fit(
            "pca_raw",
            h.astype(np.float32),
            layers,
            labels=labels.astype(np.int64),
        )


def test_fit_without_groups_needs_both_classes() -> None:
    with pytest.raises(ValueError, match="both classes"):
        RepReader.fit(
            "random",
            x_rows(*X),
            [0],
            labels=np.ones(10, dtype=np.int64),
        )


def fitted_reader() -> tuple[RepReader, NDArray[np.float32]]:
    rng = np.random.default_rng(3)
    h = rng.standard_normal((3, 30, 7)).astype(np.float32)
    h[:, ::2] += 2.0
    labels = (np.arange(30) % 2 == 0).astype(np.int64)
    return RepReader.fit("logistic", h, [2, 5, 9], labels=labels), h


def test_save_and_load_round_trip_exactly(tmp_path: Path) -> None:
    reader, h = fitted_reader()
    path = tmp_path / "reader.npz"

    reader.save(path)
    loaded = RepReader.load(path)

    assert loaded.method == "logistic"
    assert loaded.layers == (2, 5, 9)
    assert all(isinstance(layer, int) for layer in loaded.layers)
    for name in ("directions", "means", "signs"):
        original = getattr(reader, name)
        restored = getattr(loaded, name)
        assert restored.dtype == original.dtype
        np.testing.assert_array_equal(restored, original)
    np.testing.assert_array_equal(loaded.score(h), reader.score(h))


def test_save_writes_one_file_at_exactly_the_given_path(
    tmp_path: Path,
) -> None:
    reader, _ = fitted_reader()
    bare = tmp_path / "reader_without_suffix"

    reader.save(str(bare))

    assert [entry.name for entry in tmp_path.iterdir()] == [bare.name]
    assert RepReader.load(bare).layers == reader.layers


def test_load_rejects_a_file_with_missing_arrays(tmp_path: Path) -> None:
    path = tmp_path / "partial.npz"
    np.savez(path, method=np.array("x"), layers=np.array([0]))

    with pytest.raises(ValueError, match="directions"):
        RepReader.load(path)


def test_load_rejects_inconsistent_arrays(tmp_path: Path) -> None:
    reader, _ = fitted_reader()
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        method=np.array(reader.method),
        layers=np.array([2, 5]),
        directions=reader.directions,
        means=reader.means,
        signs=reader.signs,
    )

    with pytest.raises(ValueError, match="layers"):
        RepReader.load(path)


class PairedData(NamedTuple):
    h: NDArray[np.float32]
    pairs: NDArray[np.int64]
    labels: NDArray[np.int64]


def planted_direction(d: int, seed: int) -> NDArray[np.float64]:
    direction = np.random.default_rng(seed).standard_normal(d)
    return direction / np.linalg.norm(direction)


def planted_pairs(
    direction: NDArray[np.float64],
    n_pairs: int,
    *,
    seed: int,
    signal: float,
) -> PairedData:
    """Builds pairs that differ by 2 * signal along `direction`.

    Rows 2i and 2i + 1 form pair i. The content shared by the two members
    and the in-pair order are random, as in the reference datasets.
    """
    rng = np.random.default_rng(seed)
    d = direction.shape[0]
    base = 2.0 * rng.standard_normal((n_pairs, d))
    pos = base + signal * direction + rng.standard_normal((n_pairs, d))
    neg = base - signal * direction + rng.standard_normal((n_pairs, d))
    pos_first = rng.random(n_pairs) < 0.5
    h = np.empty((2 * n_pairs, d))
    h[0::2] = np.where(pos_first[:, None], pos, neg)
    h[1::2] = np.where(pos_first[:, None], neg, pos)
    h += 10.0 * rng.standard_normal(d)
    labels = np.empty(2 * n_pairs, dtype=np.int64)
    labels[0::2] = pos_first
    labels[1::2] = ~pos_first
    pairs = np.arange(2 * n_pairs, dtype=np.int64).reshape(n_pairs, 2)
    return PairedData(h.astype(np.float32), pairs, labels)


@pytest.mark.parametrize("method", ["pca_diff", "mean_diff", "logistic"])
@pytest.mark.parametrize("polarity", [1.0, -1.0], ids=["as-is", "reversed"])
def test_planted_direction_is_recovered_and_read_with_the_right_sign(
    method: Method, polarity: float
) -> None:
    direction = polarity * planted_direction(48, seed=1)
    train = planted_pairs(direction, 256, seed=2, signal=3.0)
    test = planted_pairs(direction, 250, seed=3, signal=3.0)

    reader = RepReader.fit(
        method,
        train.h[None],
        [0],
        labels=train.labels,
        pairs=train.pairs,
    )

    cosine = abs(float(reader.directions[0] @ direction))
    accuracy = group_accuracy(
        reader.score(test.h[None]), test.pairs, test.labels
    )
    assert cosine > 0.95
    assert accuracy[0] > 0.95


@pytest.mark.parametrize("seed", range(5))
def test_random_direction_scores_near_chance_on_held_out_groups(
    seed: int,
) -> None:
    direction = planted_direction(512, seed=10 + seed)
    train = planted_pairs(direction, 64, seed=20 + seed, signal=2.0)
    test = planted_pairs(direction, 250, seed=30 + seed, signal=2.0)

    reader = RepReader.fit(
        "random",
        train.h[None],
        [0],
        labels=train.labels,
        pairs=train.pairs,
        seed=seed,
    )

    accuracy = group_accuracy(
        reader.score(test.h[None]), test.pairs, test.labels
    )
    assert test.pairs.shape[0] >= 200
    assert accuracy[0] == pytest.approx(0.5, abs=0.15)


def test_fit_passes_refit_at_zero_to_the_finder() -> None:
    rng = np.random.default_rng(18)
    labels = np.tile(np.array([1, 0], dtype=np.int64), 100)
    sign = 2.0 * labels - 1.0
    h = rng.standard_normal((200, 8))
    h -= np.outer(sign, sign @ h / (sign @ sign))
    h[:, 0] += 1e-4 * sign
    stacked = h[None].astype(np.float32)

    with pytest.raises(ValueError, match="zero norm"):
        RepReader.fit("logistic", stacked, [0], labels=labels)
    reader = RepReader.fit(
        "logistic", stacked, [0], labels=labels, refit_at_zero=True
    )

    assert abs(reader.directions[0, 0]) > 0.95
