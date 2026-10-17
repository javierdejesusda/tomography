"""Tests for the shared reading sweep."""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.activations import Activations
from tomography.activations import collect_activations
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.experiments import sweep
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import Prepared
from tomography.experiments.sweep import ReadingConfig
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import reading_sweep
from tomography.experiments.sweep import shuffle_within_groups
from tomography.experiments.sweep import stimulus_activations
from tomography.experiments.sweep import sweep_from_activations
from tomography.experiments.sweep import write_curves
from tomography.model import CheckpointSpec
from tomography.model import LoadedModel
from tomography.reading.metrics import bootstrap_ci
from tomography.reading.metrics import group_accuracy
from tomography.reading.reader import RepReader
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat

DIM = 64
LAYERS = (1, 2)


def _planted(
    n_pairs: int,
    *,
    seed: int,
    layers: Sequence[int] = LAYERS,
    offset: float = 3.0,
) -> Prepared:
    """Pairs whose label-1 member sits `offset` along a fixed direction.

    The in-pair order is random, so the mean difference cancels and only the
    direction of the differences carries the signal.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_pairs
    labels = np.zeros(n, dtype=np.int64)
    first_is_positive = rng.integers(0, 2, size=n_pairs).astype(bool)
    labels[2 * np.arange(n_pairs)] = first_is_positive
    labels[2 * np.arange(n_pairs) + 1] = ~first_is_positive
    direction = np.random.default_rng(1234).normal(size=(len(layers), DIM))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    signed = (2 * labels - 1)[None, :, None] * offset
    data = rng.normal(size=(len(layers), n, DIM)) + signed * direction[:, None]
    pairs = _pair_groups(n_pairs)
    stimuli = StimulusSet(
        texts=tuple(f"text {i}" for i in range(n)),
        response_spans=(None,) * n,
        concept_spans=(None,) * n,
        labels=labels,
        pairs=pairs,
        groups=pairs.copy(),
        uids=tuple(f"u{i // 2}" for i in range(n)),
    )
    act = Activations(
        data=data.astype(np.float32),
        layers=tuple(layers),
        position="last",
        row_stimulus=np.arange(n, dtype=np.int64),
        row_rank=np.zeros(n, dtype=np.int64),
        row_token=np.zeros(n, dtype=np.int64),
        row_token_id=np.zeros(n, dtype=np.int64),
        n_stimuli=n,
    )
    return stimuli, act


def _planted_rows(n_pairs: int, *, rows: int, seed: int) -> Prepared:
    """Like `_planted`, but every stimulus holds `rows` token rows.

    The rows of a stimulus are noisy copies of its `_planted` row, so a
    reader fitted on `_planted` sets stays perfect on them.
    """
    stimuli, single = _planted(n_pairs, seed=seed)
    n = single.n_stimuli
    noise = np.random.default_rng(seed + 1).normal(
        scale=0.1, size=(len(single.layers), n * rows, DIM)
    )
    act = dataclasses.replace(
        single,
        data=(np.repeat(single.data, rows, axis=1) + noise).astype(np.float32),
        position="response",
        row_stimulus=np.repeat(np.arange(n, dtype=np.int64), rows),
        row_rank=np.tile(np.arange(rows, dtype=np.int64), n),
        row_token=np.tile(np.arange(rows, dtype=np.int64), n),
        row_token_id=np.zeros(n * rows, dtype=np.int64),
    )
    return stimuli, act


def _inputs(
    n_train: int = 3,
) -> tuple[list[Prepared], Mapping[str, Prepared]]:
    train = [_planted(40, seed=10 + i) for i in range(n_train)]
    evals = {
        "val": _planted(200, seed=95),
        "other": _planted(50, seed=98),
    }
    return train, evals


def _pair_groups(n_pairs: int) -> NDArray[np.int64]:
    return np.arange(2 * n_pairs, dtype=np.int64).reshape(n_pairs, 2)


def _pair_labels(n_pairs: int, seed: int) -> NDArray[np.int64]:
    first = np.random.default_rng(seed).integers(0, 2, size=n_pairs)
    labels = np.zeros(2 * n_pairs, dtype=np.int64)
    labels[2 * np.arange(n_pairs) + (1 - first)] = 1
    return labels


def test_shuffle_keeps_one_positive_per_group() -> None:
    groups = _pair_groups(30)
    labels = _pair_labels(30, seed=1)
    shuffled = shuffle_within_groups(labels, groups, seed=3)
    assert shuffled.dtype == np.int64
    assert shuffled.shape == labels.shape
    assert (shuffled[groups].sum(axis=1) == 1).all()


def test_shuffle_handles_ragged_groups_and_keeps_other_rows() -> None:
    groups = np.array([[0, 1, 2], [4, 5, -1]], dtype=np.int64)
    labels = np.array([1, 0, 0, 1, 0, 1, 1], dtype=np.int64)
    shuffled = shuffle_within_groups(labels, groups, seed=0)
    assert shuffled[[0, 1, 2]].sum() == 1
    assert shuffled[[4, 5]].sum() == 1
    assert shuffled[3] == 1
    assert shuffled[6] == 1


def test_shuffle_ignores_padding_anywhere_in_a_group() -> None:
    groups = np.array([[-1, 0, 1], [3, -1, 2]], dtype=np.int64)
    labels = np.array([1, 0, 0, 1], dtype=np.int64)
    for seed in range(20):
        shuffled = shuffle_within_groups(labels, groups, seed=seed)
        assert shuffled[[0, 1]].sum() == 1
        assert shuffled[[2, 3]].sum() == 1


def test_shuffle_is_deterministic_per_seed_and_varies_across_seeds() -> None:
    groups = _pair_groups(40)
    labels = _pair_labels(40, seed=2)
    first = shuffle_within_groups(labels, groups, seed=5)
    again = shuffle_within_groups(labels, groups, seed=5)
    other = shuffle_within_groups(labels, groups, seed=6)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_shuffle_does_not_modify_its_input() -> None:
    groups = _pair_groups(10)
    labels = _pair_labels(10, seed=0)
    before = labels.copy()
    shuffle_within_groups(labels, groups, seed=1)
    np.testing.assert_array_equal(labels, before)


def test_shuffle_positive_position_is_uniform() -> None:
    groups = np.tile(np.arange(4, dtype=np.int64), (400, 1)) + 4 * np.arange(
        400, dtype=np.int64
    ).reshape(-1, 1)
    labels = np.zeros(1600, dtype=np.int64)
    labels[groups[:, 0]] = 1
    shuffled = shuffle_within_groups(labels, groups, seed=0)
    counts = np.bincount(shuffled[groups].argmax(axis=1), minlength=4)
    assert (np.abs(counts / 400 - 0.25) < 0.08).all()


def test_shuffle_rejects_groups_sharing_a_row() -> None:
    groups = np.array([[0, 1], [1, 2]], dtype=np.int64)
    labels = np.array([1, 0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="share"):
        shuffle_within_groups(labels, groups, seed=0)


def test_shuffle_rejects_non_binary_labels() -> None:
    groups = np.array([[0, 1]], dtype=np.int64)
    labels = np.array([1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="binary"):
        shuffle_within_groups(labels, groups, seed=0)


def test_shuffle_rejects_group_without_one_positive() -> None:
    groups = np.array([[0, 1]], dtype=np.int64)
    labels = np.array([1, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="exactly one"):
        shuffle_within_groups(labels, groups, seed=0)


def test_sweep_shapes_and_eval_order() -> None:
    train, evals = _inputs()
    result = sweep_from_activations(train, evals, seed=0)
    assert result.layers == LAYERS
    assert result.eval_names == ("val", "other")
    assert result.accuracy.shape == (2, len(SERIES), 3, len(LAYERS))
    assert len(result.readers) == 3
    assert all(reader.layers == LAYERS for reader in result.readers)
    per_seed = result.per_seed("other")
    assert per_seed.shape == (len(SERIES), 3, len(LAYERS))
    np.testing.assert_array_equal(per_seed, result.accuracy[1])


def test_sweep_per_seed_rejects_an_unknown_name() -> None:
    train, evals = _inputs(1)
    result = sweep_from_activations(train, evals, seed=0)
    with pytest.raises(ValueError, match="nope"):
        result.per_seed("nope")


def test_sweep_reader_recovers_a_planted_direction() -> None:
    train, evals = _inputs()
    result = sweep_from_activations(train, evals, seed=0)
    reader = result.per_seed("val")[SERIES.index("reader")]
    np.testing.assert_array_equal(reader, np.ones_like(reader))


def test_sweep_passes_refit_at_zero_to_the_reader_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []
    real = RepReader.fit.__func__  # type: ignore[attr-defined]

    def spy(cls: Any, *args: Any, **kwargs: Any) -> RepReader:
        seen.append(kwargs.get("refit_at_zero"))
        return real(cls, *args, **kwargs)  # type: ignore[no-any-return]

    monkeypatch.setattr(RepReader, "fit", classmethod(spy))
    train, evals = _inputs(n_train=1)

    sweep_from_activations(train, evals, method="logistic")
    sweep_from_activations(train, evals, method="logistic", refit_at_zero=True)

    assert seen == [False, False, True, True]


def test_sweep_shuffled_series_sits_at_chance() -> None:
    train, evals = _inputs()
    result = sweep_from_activations(train, evals, seed=0)
    shuffled = result.per_seed("val")[SERIES.index("shuffled")]
    assert np.abs(shuffled - 0.5).max() < 0.08


def test_sweep_shuffled_series_moves_whole_stimuli_not_token_rows() -> None:
    train = [_planted(40, seed=10 + i) for i in range(4)]
    evals = {"val": _planted_rows(25, rows=4, seed=95)}
    result = sweep_from_activations(train, evals, seed=0)
    per_seed = result.per_seed("val")
    reader = per_seed[SERIES.index("reader")]
    shuffled = per_seed[SERIES.index("shuffled")]
    np.testing.assert_array_equal(reader, np.ones_like(reader))
    np.testing.assert_allclose(
        shuffled * 25, np.round(shuffled * 25), atol=1e-9
    )


def test_sweep_baselines_follow_the_documented_seed_scheme() -> None:
    train, evals = _inputs(4)
    seed = 11
    result = sweep_from_activations(train, evals, seed=seed)
    for i, (stimuli, act) in enumerate(train):
        random = RepReader.fit(
            "random",
            act.data,
            act.layers,
            labels=act.row_values(stimuli.labels),
            pairs=act.row_groups(stimuli.pairs),
            groups=act.row_groups(stimuli.groups),
            seed=derive_seed(seed, "random", i),
        )
        for e, name in enumerate(result.eval_names):
            e_stimuli, e_act = evals[name]
            groups = e_act.row_groups(e_stimuli.groups)
            labels = e_act.row_values(e_stimuli.labels)
            shuffled = e_act.row_values(
                shuffle_within_groups(
                    e_stimuli.labels,
                    e_stimuli.groups,
                    seed=derive_seed(seed, "shuffled", i, name),
                )
            )
            np.testing.assert_array_equal(
                result.accuracy[e, SERIES.index("random"), i],
                group_accuracy(random.score(e_act.data), groups, labels),
            )
            np.testing.assert_array_equal(
                result.accuracy[e, SERIES.index("shuffled"), i],
                group_accuracy(
                    result.readers[i].score(e_act.data), groups, shuffled
                ),
            )


def test_sweep_random_series_is_below_the_reader() -> None:
    train, evals = _inputs()
    result = sweep_from_activations(train, evals, seed=0)
    per_seed = result.per_seed("val")
    assert (per_seed[SERIES.index("random")] < per_seed[0]).all()


def test_sweep_is_deterministic_per_seed() -> None:
    train, evals = _inputs()
    first = sweep_from_activations(train, evals, seed=4)
    again = sweep_from_activations(train, evals, seed=4)
    other = sweep_from_activations(train, evals, seed=5)
    np.testing.assert_array_equal(first.accuracy, again.accuracy)
    assert not np.array_equal(first.accuracy, other.accuracy)


def test_sweep_rejects_empty_train_and_evals() -> None:
    train, evals = _inputs(1)
    with pytest.raises(ValueError, match="train"):
        sweep_from_activations([], evals, seed=0)
    with pytest.raises(ValueError, match="evals"):
        sweep_from_activations(train, {}, seed=0)


def test_sweep_rejects_mismatching_layers() -> None:
    train, evals = _inputs(1)
    odd = {"val": _planted(20, seed=3, layers=(1, 3))}
    with pytest.raises(ValueError, match="layers"):
        sweep_from_activations(train, odd, seed=0)
    with pytest.raises(ValueError, match="layers"):
        sweep_from_activations(
            [train[0], _planted(20, seed=4, layers=(1, 3))], evals, seed=0
        )


@dataclass(frozen=True, kw_only=True)
class _Config(ReadingConfig):
    experiment: str = "t1"
    model: str = "tiny"


def test_reading_config_defaults() -> None:
    config = _Config()
    assert config.n_seeds == 10
    assert config.n_pairs == 64
    assert config.n_eval_pairs is None
    assert config.method == "pca_diff"
    assert config.preprocess == "center"
    assert config.position == "last"
    assert config.eval_position is None
    assert config.layers is None
    assert config.bos is None
    assert config.batch_size == 8


def test_reading_config_round_trips_through_a_dict() -> None:
    config = _Config(
        method="mean_diff",
        layers=(6, 7, 8),
        position="concept",
        bos="none",
        eval_position="last",
        n_eval_pairs=32,
    )
    assert _Config.from_dict(config.to_dict()) == config


def test_reading_config_rejects_a_wrong_type() -> None:
    data = _Config().to_dict()
    data["n_seeds"] = "ten"
    with pytest.raises(ValueError, match="n_seeds"):
        _Config.from_dict(data)


def _examples(split: str, per_class: int = 40) -> list[Example]:
    positive = ("good", "fine", "great", "nice")
    negative = ("bad", "poor", "dull", "grim")
    examples = []
    for i in range(per_class):
        for label, words in ((1, positive), (0, negative)):
            examples.append(
                Example(
                    uid=f"{split}-{label}-{i}",
                    text=f"a {words[i % 4]} movie {words[(i + 1) % 4]} {i}",
                    label=label,
                    split="train" if split == "train" else "val",
                )
            )
    return examples


def _build(examples: Sequence[Example], n_pairs: int, seed: int) -> StimulusSet:
    return build_stimulus_set(
        examples,
        PAPER_TEMPLATES["concept"],
        ChatFormat("plain"),
        pairing="label_contrastive",
        concept="sentiment",
        n_pairs=n_pairs,
        seed=seed,
    )


def _sets(n_train: int = 2) -> tuple[list[StimulusSet], dict[str, StimulusSet]]:
    train = [_build(_examples("train"), 8, seed=i) for i in range(n_train)]
    return train, {"val": _build(_examples("val"), 12, seed=7)}


def test_chat_format_for_is_plain_without_a_template_checkpoint(
    tiny_arch: LoadedModel,
) -> None:
    assert chat_format_for(tiny_arch) == ChatFormat("plain")


def test_chat_format_for_uses_the_tokenizer_template(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = ChatFormat("none")
    monkeypatch.setattr(
        ChatFormat, "from_tokenizer", classmethod(lambda cls, tok: sentinel)
    )
    checkpoint = CheckpointSpec(
        key="tiny",
        repo_id="x/tiny",
        revision="0" * 40,
        arch="gpt2",
        post_trained=True,
        license="mit",
        dtype="float32",
        chat_format="template",
    )
    templated = dataclasses.replace(tiny_arch, checkpoint=checkpoint)
    assert chat_format_for(templated) is sentinel
    plain = dataclasses.replace(
        tiny_arch,
        checkpoint=dataclasses.replace(checkpoint, chat_format="plain"),
    )
    assert chat_format_for(plain) == ChatFormat("plain")


def test_stimulus_activations_matches_collect_activations(
    tiny_arch: LoadedModel,
) -> None:
    stimuli = _sets(1)[0][0]
    got = stimulus_activations(
        tiny_arch, stimuli, layers=(1, 2), position="concept", batch_size=4
    )
    want = collect_activations(
        tiny_arch,
        stimuli.texts,
        layers=(1, 2),
        position="concept",
        response_spans=stimuli.response_spans,
        concept_spans=stimuli.concept_spans,
        batch_size=4,
    )
    assert got.layers == want.layers
    np.testing.assert_array_equal(got.data, want.data)
    np.testing.assert_array_equal(got.row_stimulus, want.row_stimulus)


def test_reading_sweep_defaults_to_layers_one_to_depth(
    tiny_arch: LoadedModel,
) -> None:
    train, evals = _sets(2)
    result = reading_sweep(tiny_arch, train, evals, seed=0)
    assert result.layers == tuple(range(1, tiny_arch.n_layers + 1))
    assert result.accuracy.shape == (1, 3, 2, tiny_arch.n_layers)
    assert result.eval_names == ("val",)


def test_reading_sweep_honors_explicit_layers(tiny_arch: LoadedModel) -> None:
    train, evals = _sets(2)
    result = reading_sweep(tiny_arch, train, evals, layers=(1, 3), seed=0)
    assert result.layers == (1, 3)
    assert result.accuracy.shape == (1, 3, 2, 2)


def test_reading_sweep_equals_the_hand_prepared_sweep(
    tiny_arch: LoadedModel,
) -> None:
    train, evals = _sets(2)
    layers = (1, 2, 3)
    got = reading_sweep(
        tiny_arch,
        train,
        evals,
        position="last",
        eval_position="concept",
        layers=layers,
        seed=3,
    )
    prepared = [
        (s, stimulus_activations(tiny_arch, s, layers=layers, position="last"))
        for s in train
    ]
    prepared_evals = {
        name: (
            s,
            stimulus_activations(
                tiny_arch, s, layers=layers, position="concept"
            ),
        )
        for name, s in evals.items()
    }
    want = sweep_from_activations(prepared, prepared_evals, seed=3)
    np.testing.assert_array_equal(got.accuracy, want.accuracy)
    for a, b in zip(got.readers, want.readers, strict=True):
        np.testing.assert_array_equal(a.directions, b.directions)


def test_reading_sweep_is_deterministic_per_seed(
    tiny_arch: LoadedModel,
) -> None:
    train, evals = _sets(2)
    first = reading_sweep(tiny_arch, train, evals, seed=1)
    again = reading_sweep(tiny_arch, train, evals, seed=1)
    other = reading_sweep(tiny_arch, train, evals, seed=2)
    np.testing.assert_array_equal(first.accuracy, again.accuracy)
    assert not np.array_equal(first.accuracy, other.accuracy)


@pytest.mark.parametrize(
    ("empty", "match"),
    [("train", "train_sets"), ("eval", "eval_sets")],
    ids=["no_train_sets", "no_eval_sets"],
)
def test_reading_sweep_rejects_an_empty_list_before_any_forward_pass(
    tiny_arch: LoadedModel,
    monkeypatch: pytest.MonkeyPatch,
    empty: str,
    match: str,
) -> None:
    forwards: list[StimulusSet] = []
    collect = sweep.stimulus_activations

    def spy(loaded: LoadedModel, stimuli: StimulusSet, **kwargs: Any) -> Any:
        forwards.append(stimuli)
        return collect(loaded, stimuli, **kwargs)

    monkeypatch.setattr(sweep, "stimulus_activations", spy)
    train, evals = _sets(2)
    with pytest.raises(ValueError, match=match):
        reading_sweep(
            tiny_arch,
            [] if empty == "train" else train,
            {} if empty == "eval" else evals,
        )
    assert forwards == []


def _run_dir(tmp_path: Path) -> RunDir:
    return RunDir.create(_Config(), out_root=tmp_path)


def _per_seed() -> NDArray[np.float64]:
    rng = np.random.default_rng(0)
    return np.clip(rng.normal(0.7, 0.05, size=(3, 6, 4)), 0.0, 1.0)


def test_write_curves_writes_arrays_and_sidecar(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    per_seed = _per_seed()
    write_curves(
        run_dir,
        per_seed,
        (1, 2, 3, 4),
        series=SERIES,
        chance=0.5,
        n_layers=4,
        n_boot=200,
    )
    arrays = run_dir.read_arrays("curves")
    assert set(arrays) == {"x", "mean", "lo", "hi", "per_seed"}
    np.testing.assert_array_equal(arrays["x"], np.array([1.0, 2.0, 3.0, 4.0]))
    assert arrays["x"].dtype == np.float64
    np.testing.assert_allclose(arrays["mean"], per_seed.mean(axis=1))
    np.testing.assert_array_equal(arrays["per_seed"], per_seed)
    assert (arrays["lo"] <= arrays["mean"]).all()
    assert (arrays["mean"] <= arrays["hi"]).all()
    assert (arrays["lo"] < arrays["hi"]).all()
    assert run_dir.read_json("curves") == {
        "series": list(SERIES),
        "xlabel": "Layer",
        "ylabel": "Accuracy",
        "chance": 0.5,
        "n_layers": 4,
    }


def test_write_curves_uses_the_stem_and_labels(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    write_curves(
        run_dir,
        _per_seed(),
        (1, 2, 3, 4),
        series=SERIES,
        stem="other",
        xlabel="Depth",
        ylabel="Score",
        n_boot=50,
    )
    sidecar = run_dir.read_json("other")
    assert isinstance(sidecar, dict)
    assert sidecar["xlabel"] == "Depth"
    assert sidecar["ylabel"] == "Score"
    assert sidecar["chance"] is None


def test_write_curves_band_is_the_bootstrap_of_the_seeds(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    per_seed = _per_seed()
    write_curves(
        run_dir,
        per_seed,
        (1, 2, 3, 4),
        series=SERIES,
        alpha=0.2,
        n_boot=300,
        seed=9,
    )
    lo, hi = bootstrap_ci(
        per_seed.transpose(1, 0, 2), alpha=0.2, n_boot=300, seed=9
    )
    arrays = run_dir.read_arrays("curves")
    np.testing.assert_array_equal(arrays["lo"], lo)
    np.testing.assert_array_equal(arrays["hi"], hi)


@pytest.mark.parametrize(
    ("per_seed", "layers", "series"),
    [
        (np.zeros((6, 4)), (1, 2, 3, 4), SERIES),
        (np.zeros((3, 6, 4)), (1, 2, 3, 4), SERIES[:2]),
        (np.zeros((3, 6, 4)), (1, 2, 3), SERIES),
    ],
    ids=["not-3d", "series-count", "layer-count"],
)
def test_write_curves_validates_shapes(
    tmp_path: Path,
    per_seed: NDArray[np.float64],
    layers: tuple[int, ...],
    series: Sequence[str],
) -> None:
    with pytest.raises(ValueError, match=r"per_seed|series|layers"):
        write_curves(_run_dir(tmp_path), per_seed, layers, series=series)


def test_plot_curves_file_writes_three_formats(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    write_curves(
        run_dir,
        _per_seed(),
        (1, 2, 3, 4),
        series=SERIES,
        chance=0.5,
        n_layers=4,
        n_boot=50,
    )
    out = tmp_path / "figs"
    paths = plot_curves_file(run_dir, out)
    assert sorted(p.name for p in paths) == [
        "curves.pdf",
        "curves.png",
        "curves.svg",
    ]
    assert all(p.parent == out and p.stat().st_size > 0 for p in paths)


def test_plot_curves_file_renames_the_figure(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    write_curves(
        run_dir, _per_seed(), (1, 2, 3, 4), series=SERIES, stem="c2", n_boot=50
    )
    paths = plot_curves_file(run_dir, tmp_path / "figs", stem="c2", name="fig")
    assert sorted(p.name for p in paths) == ["fig.pdf", "fig.png", "fig.svg"]
