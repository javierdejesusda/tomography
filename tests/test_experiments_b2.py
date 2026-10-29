"""Tests for the B2 finder baselines experiment."""

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.activations import Activations
from tomography.activations import collect_activations
from tomography.cli import load_experiment
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import b2_finder_baselines as b2
from tomography.experiments import r1_sentiment
from tomography.experiments.b2_finder_baselines import Config
from tomography.experiments.b2_finder_baselines import completion_texts
from tomography.experiments.b2_finder_baselines import dataset_items
from tomography.experiments.b2_finder_baselines import plot
from tomography.experiments.b2_finder_baselines import run
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import sweep_from_activations
from tomography.experiments.tb1_text_baselines import TextSet
from tomography.experiments.tb1_text_baselines import interval
from tomography.experiments.tb1_text_baselines import seed_accuracy
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import task_splits
from tomography.model import LoadedModel
from tomography.reading.ccs import CCSProbe
from tomography.reading.metrics import bootstrap_ci
from tomography.reading.metrics import group_accuracy
from tomography.templates import ChatFormat

TINY_CCS: dict[str, Any] = {"ccs_restarts": 1, "ccs_epochs": 20}
PLAIN = ChatFormat("plain")
SPLITS = ("train", "val", "test")
LAYERS = (1, 2)


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        datasets=("sentiment", "cities"),
        n_seeds=2,
        n_pairs=4,
        n_boot=50,
        **TINY_CCS,
    )
    return dataclasses.replace(base, **overrides)


def _sentiment(per_label: int = 12) -> list[Example]:
    words = {
        1: ("good", "great", "fine", "warm"),
        0: ("bad", "dull", "poor", "cold"),
    }
    examples = [
        Example(uid="sst2-short", text="good film", label=1, split="train"),
        *(
            Example(
                uid=f"sst2-val-short-{label}-{i}",
                text=f"{words[label][i]} film",
                label=label,
                split="val",
            )
            for label in (1, 0)
            for i in range(4)
        ),
    ]
    for split in ("train", "val"):
        for label in (1, 0):
            for i in range(per_label):
                first, second = words[label][i % 4], words[label][(i + 1) % 4]
                examples.append(
                    Example(
                        uid=f"sst2-{split}-{label}-{i}",
                        text=f"a {first} and {second} film {i}",
                        label=label,
                        split=split,
                    )
                )
    return examples


def _truth_sets(
    families: tuple[tuple[str, str], ...] = (("cities", "neg_cities"),),
    n: int = 24,
) -> list[Example]:
    """Statement sets and their negations, aligned row by row.

    Statement i has label i % 2 and split `SPLITS[i % 3]`; its negation has
    the other label and the next split, so a pair spans two loader splits.
    """
    examples = []
    for first, second in families:
        for subset, negated in ((first, 0), (second, 1)):
            for i in range(n):
                examples.append(
                    Example(
                        uid=f"got-{subset}-{i}",
                        text=f"town {i} is {'not ' * negated}in land {i % 5}.",
                        label=(i + negated) % 2,
                        split=SPLITS[(i + negated) % 3],
                        meta={"subset": subset},
                    )
                )
    return examples


def test_config_defaults_round_trip_and_pinned_hash() -> None:
    config = Config()
    assert config.experiment == "b2"
    assert config.model == "gpt2"
    assert config.datasets == (
        "sentiment",
        "cities",
        "larger_than",
        "sp_en_trans",
    )
    assert (config.n_seeds, config.n_pairs, config.n_eval_pairs) == (
        10,
        64,
        None,
    )
    assert (config.ccs_restarts, config.ccs_epochs) == (10, 1000)
    assert config.ccs_lr == 0.01
    assert Config.from_dict(config.to_dict()) == config
    assert config.config_hash() == "317ba03fe78b"


@pytest.mark.parametrize(
    ("datasets", "message"),
    [
        ((), "at least one dataset"),
        (("sentiment", "facts"), "unknown dataset 'facts'"),
        (("cities", "cities"), "repeated dataset 'cities'"),
    ],
)
def test_the_dataset_list_is_checked(
    datasets: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(datasets=datasets)


def test_truth_pairs_join_each_statement_to_its_negation_in_its_split() -> None:
    rows = _truth_sets()
    statements = {e.uid: e for e in rows if e.meta["subset"] == "cities"}
    items = dataset_items(_config(), "cities", PLAIN, [], rows)

    parts = {"train": items.train, "val": (items.val,), "test": (items.test,)}
    for split, sets in parts.items():
        for stimuli in sets:
            assert stimuli.groups.shape[1] == 2
            for first, second in stimuli.groups:
                uid = stimuli.uids[first]
                assert uid in statements
                assert statements[uid].split == split
                assert stimuli.uids[second] == uid.replace(
                    "cities", "neg_cities"
                )
                assert stimuli.labels[first] + stimuli.labels[second] == 1
            assert all("truthfulness" in text for text in stimuli.texts)
    assert [s.groups.shape[0] for s in items.train] == [4, 4]
    held_out = sorted(items.test.uids[g[0]] for g in items.test.groups)
    assert held_out == sorted(
        uid for uid, e in statements.items() if e.split == "test"
    )
    assert items.val.groups.shape[0] == 8
    assert items.text_of["got-neg_cities-3"] == "town 3 is not in land 3."


@pytest.mark.parametrize("broken", ["shorter", "same_label"])
def test_a_negation_set_must_negate_its_statements_row_by_row(
    broken: str,
) -> None:
    rows = _truth_sets()
    if broken == "shorter":
        rows = [e for e in rows if e.uid != "got-neg_cities-23"]
    else:
        rows = [
            dataclasses.replace(e, label=1 - e.label)
            if e.uid == "got-neg_cities-5"
            else e
            for e in rows
        ]
    with pytest.raises(ValueError, match="neg_cities does not negate cities"):
        dataset_items(_config(), "cities", PLAIN, [], rows)


def test_a_truth_set_without_examples_is_an_error() -> None:
    with pytest.raises(ValueError, match="no examples for larger_than"):
        dataset_items(_config(), "larger_than", PLAIN, [], _truth_sets())


def test_sentiment_trains_on_the_r1_sets_and_halves_the_validation_split() -> (
    None
):
    examples = _sentiment()
    config = _config()
    items = dataset_items(config, "sentiment", PLAIN, examples, [])

    train = r1_sentiment.select_examples(
        examples, split="train", min_words=r1_sentiment.Config().min_words
    )
    assert len(items.train) == config.n_seeds
    for i, stimuli in enumerate(items.train):
        expected = r1_sentiment._stimuli(
            train, PLAIN, config.n_pairs, derive_seed(config.seed, "train", i)
        )
        assert stimuli.texts == expected.texts
        assert stimuli.uids == expected.uids
        np.testing.assert_array_equal(stimuli.labels, expected.labels)
        np.testing.assert_array_equal(stimuli.pairs, expected.pairs)
    halves = task_splits(
        READING_TASKS["sentiment"], {"sst2": examples}, seed=config.seed
    )
    assert set(items.val.uids) <= {e.uid for e in halves.select}
    assert set(items.test.uids) <= {e.uid for e in halves.report}
    held_out = [
        items.text_of[uid] for uid in (*items.val.uids, *items.test.uids)
    ]
    assert min(len(text.split()) for text in held_out) >= 3
    report = r1_sentiment.select_examples(
        halves.report, split="val", min_words=3
    )
    assert items.test.groups.shape[0] == min(
        sum(e.label == label for e in report) for label in (0, 1)
    )
    assert items.text_of["sst2-train-1-0"] == "a good and great film 0"


def test_sentiment_contrast_pairs_render_each_text_with_both_completions() -> (
    None
):
    items = dataset_items(_config(), "sentiment", PLAIN, _sentiment(), [])

    texts = completion_texts(items.test, items.text_of, PLAIN)

    assert len(texts) == 2 * len(items.test.uids)
    for j, uid in enumerate(items.test.uids):
        assert texts[2 * j] == (
            "USER: Consider the following example: ''' "
            f"{items.text_of[uid]} '''\nBetween negative and positive, the "
            "sentiment of this example is ASSISTANT: positive"
        )
        assert texts[2 * j + 1] == texts[2 * j].removesuffix("positive") + (
            "negative"
        )


def _part(
    h: NDArray[np.float64],
    labels: NDArray[np.int64],
    texts: tuple[str, ...],
    uids: tuple[str, ...],
) -> b2.Part:
    """A truth part whose groups are the rows (2i, 2i + 1), statement first."""
    n = labels.shape[0]
    groups = np.arange(n, dtype=np.int64).reshape(-1, 2)
    stimuli = StimulusSet(
        texts=texts,
        response_spans=(None,) * n,
        concept_spans=(None,) * n,
        labels=labels,
        pairs=groups,
        groups=groups,
        uids=uids,
    )
    zeros = np.zeros(n, dtype=np.int64)
    act = Activations(
        data=h.astype(np.float32),
        layers=LAYERS,
        position="last",
        row_stimulus=np.arange(n, dtype=np.int64),
        row_rank=zeros,
        row_token=zeros,
        row_token_id=zeros,
        n_stimuli=n,
    )
    return b2.Part(stimuli, act, b2.pair_contrast(stimuli, act))


def _offset_part(
    n_pairs: int, seed: int, *, d: int = 8, offset: float = 3.0
) -> b2.Part:
    """Statement and negation sets offset along the truth direction.

    Row 2i is a statement and row 2i + 1 its negation; the statement is true
    for even i. Truth adds 1 along feature 0, and the statements sit `offset`
    above and the negations `offset` below on that feature, so the raw rows
    of a pair always rank the statement first, while each set centred on its
    own mean gives the truth.
    """
    rng = np.random.default_rng(seed)
    truth = np.where(np.arange(n_pairs) % 2 == 0, 1.0, -1.0)
    base = rng.standard_normal((len(LAYERS), n_pairs, d))
    shift = np.zeros_like(base)
    shift[..., 0] = truth + offset
    h = np.empty((len(LAYERS), 2 * n_pairs, d))
    h[:, 0::2] = base + shift + 0.1 * rng.standard_normal(base.shape)
    h[:, 1::2] = base - shift + 0.1 * rng.standard_normal(base.shape)
    labels = np.empty(2 * n_pairs, dtype=np.int64)
    labels[0::2] = truth > 0
    labels[1::2] = truth < 0
    rows = range(2 * n_pairs)
    uids = tuple(f"got-{'neg_' * (r % 2)}cities-{seed}-{r // 2}" for r in rows)
    texts = tuple(
        f"claim {r // 2} {'does not hold' if r % 2 else 'holds'}" for r in rows
    )
    return _part(h, labels, texts, uids)


def _offset_dataset(n_seeds: int = 1) -> b2.Dataset:
    train = tuple(_offset_part(40, seed=10 + i) for i in range(n_seeds))
    val, test = _offset_part(40, seed=2), _offset_part(40, seed=3)
    text_of = {
        uid: text
        for part in (*train, val, test)
        for uid, text in zip(part.stimuli.uids, part.stimuli.texts, strict=True)
    }
    return b2.Dataset("cities", train, val, test, text_of)


def test_ccs_accuracy_comes_from_pair_probabilities_on_offset_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes: list[CCSProbe] = []
    fit_ccs = b2.fit_ccs

    def spy(*args: Any, **kwargs: Any) -> CCSProbe:
        probes.append(fit_ccs(*args, **kwargs))
        return probes[-1]

    monkeypatch.setattr(b2, "fit_ccs", spy)
    data = _offset_dataset()
    config = _config(
        datasets=("cities",), n_seeds=1, ccs_restarts=2, ccs_epochs=500
    )

    scores = b2.score_dataset(config, data)

    (probe,) = probes
    test = data.test
    probability = probe.pair_probabilities(test.act.data, test.contrast.pairs)
    first_true = test.stimuli.labels[test.contrast.pairs[:, 0]] == 1
    by_pairs = ((probability > 0.5) == first_true).mean(axis=1)
    np.testing.assert_allclose(scores.test["ccs"][0], by_pairs)
    assert by_pairs.min() >= 0.95
    by_reader = group_accuracy(
        probe.reader.score(test.act.data),
        test.stimuli.groups,
        test.stimuli.labels,
    )
    assert by_reader.max() <= 0.6


def test_the_finders_and_two_controls_come_from_the_shared_sweep() -> None:
    data = _offset_dataset(n_seeds=2)
    config = _config(datasets=("cities",), seed=3)

    scores = b2.score_dataset(config, data)

    def prepared(part: b2.Part) -> tuple[StimulusSet, Activations]:
        return part.stimuli, part.act

    evals = {"val": prepared(data.val), "test": prepared(data.test)}
    for finder in b2.FINDERS:
        sweep = sweep_from_activations(
            [prepared(part) for part in data.train],
            evals,
            method=finder,
            seed=config.seed,
        )
        for split, accuracy in (("val", scores.val), ("test", scores.test)):
            series = sweep.per_seed(split)
            np.testing.assert_array_equal(accuracy[finder], series[0])
            if finder == "pca_diff":
                np.testing.assert_array_equal(accuracy["random"], series[1])
                np.testing.assert_array_equal(accuracy["shuffled"], series[2])
    assert scores.layers == LAYERS
    assert set(scores.val) == set(scores.test) == set(b2.LAYER_SERIES)


def test_the_text_controls_are_the_tb1_classifiers_on_the_test_pairs() -> None:
    data = _offset_dataset(n_seeds=2)

    scores = b2.score_dataset(_config(datasets=("cities",)), data)

    test = TextSet(
        texts=tuple(data.text_of[uid] for uid in data.test.stimuli.uids),
        uids=data.test.stimuli.uids,
        labels=data.test.stimuli.labels,
        groups=data.test.stimuli.groups,
    )
    assert b2.TEXT_CONTROLS == ("text_word", "text_char")
    assert set(scores.text) == set(b2.TEXT_CONTROLS)
    for analyzer in ("word", "char"):
        expected = [
            seed_accuracy(
                [data.text_of[uid] for uid in part.stimuli.uids],
                part.stimuli.labels,
                test,
                analyzer=analyzer,
            )
            for part in data.train
        ]
        np.testing.assert_allclose(scores.text[f"text_{analyzer}"], expected)


def _planted_scores() -> b2.Scores:
    """Random accuracies with a validation peak at layer 2 for LAT."""
    rng = np.random.default_rng(0)
    val = {s: 0.5 * rng.random((3, 3)) for s in b2.LAYER_SERIES}
    val["pca_diff"][:, 1] = 0.9
    test = {s: rng.random((3, 3)) for s in b2.LAYER_SERIES}
    text = {c: rng.random(3) for c in b2.TEXT_CONTROLS}
    return b2.Scores((1, 2, 3), val, test, text, rng.random((3, 3)))


def test_rows_give_the_validation_layer_wilson_and_bootstrap() -> None:
    scores = _planted_scores()
    config = _config(n_boot=200, seed=4)

    rows = b2.table_rows("cities", scores, 40, config)

    assert [row[1] for row in rows] == [*b2.LAYER_SERIES, *b2.TEXT_CONTROLS]
    table = {row[1]: dict(zip(b2.COLUMNS, row, strict=True)) for row in rows}
    assert table["pca_diff"]["val_layer"] == 2
    for name, row in table.items():
        if name in b2.TEXT_CONTROLS:
            assert row["val_layer"] is None
            per_seed = scores.text[name]
        else:
            at = int(np.argmax(scores.val[name].mean(axis=0)))
            assert row["val_layer"] == scores.layers[at]
            per_seed = scores.test[name][:, at]
        accuracy = float(per_seed.mean())
        assert row["accuracy"] == pytest.approx(accuracy)
        assert [row["wilson_lo"], row["wilson_hi"]] == pytest.approx(
            interval(accuracy, 40)
        )
        lo, hi = bootstrap_ci(per_seed, n_boot=200, seed=4)
        assert [row["boot_lo"], row["boot_hi"]] == pytest.approx([lo, hi])
        assert (row["dataset"], row["n_test_pairs"], row["n_seeds"]) == (
            "cities",
            40,
            3,
        )


def test_lat_is_the_reader_published_finders_baselines_the_rest_controls() -> (
    None
):
    expected = {
        "pca_diff": "reader",
        "mean_diff": "baseline",
        "whitened_mean_diff": "baseline",
        "logistic": "baseline",
        "ccs": "baseline",
        "random": "control",
        "shuffled": "control",
        "text_word": "control",
        "text_char": "control",
    }
    assert dict(b2.KIND_OF) == expected

    rows = b2.table_rows("cities", _planted_scores(), 40, _config())

    assert {row[1]: row[2] for row in rows} == expected


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("b2")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_ccs_reads_the_completions_for_sentiment_and_the_stimuli_for_truth(
    tiny_gpt2: LoadedModel,
) -> None:
    config = _config()
    chat = chat_format_for(tiny_gpt2)
    layers = list(range(1, tiny_gpt2.n_layers + 1))

    def acts(texts: list[str]) -> NDArray[np.float32]:
        return collect_activations(
            tiny_gpt2, texts, layers=layers, batch_size=config.batch_size
        ).data

    for name, sentiment, truth in (
        ("sentiment", _sentiment(), []),
        ("cities", [], _truth_sets()),
    ):
        items = dataset_items(config, name, chat, sentiment, truth)

        data = b2.prepare_dataset(config, name, items, tiny_gpt2, chat)

        assert len(data.train) == config.n_seeds
        for part, stimuli in zip(
            (*data.train, data.val, data.test),
            (*items.train, items.val, items.test),
            strict=True,
        ):
            assert part.stimuli is stimuli
            assert part.act.layers == tuple(layers)
            np.testing.assert_allclose(
                part.act.data, acts(list(stimuli.texts)), rtol=1e-5, atol=1e-5
            )
            if name == "sentiment":
                texts = completion_texts(stimuli, items.text_of, chat)
                np.testing.assert_allclose(
                    part.contrast.h, acts(texts), rtol=1e-5, atol=1e-5
                )
                n = stimuli.labels.shape[0]
                assert part.contrast.pairs.tolist() == [
                    [2 * j, 2 * j + 1] for j in range(n)
                ]
                np.testing.assert_array_equal(
                    part.contrast.labels[0::2], stimuli.labels
                )
            else:
                assert part.contrast.h is part.act.data
                np.testing.assert_array_equal(
                    part.contrast.pairs, stimuli.groups
                )
        assert data.text_of == items.text_of


def _rows(run_dir: RunDir) -> list[dict[str, Any]]:
    table = run_dir.read_json("table")
    assert isinstance(table, dict)
    assert table["columns"] == list(b2.COLUMNS)
    return [dict(zip(b2.COLUMNS, row, strict=True)) for row in table["rows"]]


def test_execute_writes_the_table_curves_and_summary(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes: list[CCSProbe] = []
    fit_ccs = b2.fit_ccs

    def spy(*args: Any, **kwargs: Any) -> CCSProbe:
        probes.append(fit_ccs(*args, **kwargs))
        return probes[-1]

    monkeypatch.setattr(b2, "fit_ccs", spy)
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)

    b2.execute(config, run_dir, tiny_arch, _sentiment(), _truth_sets())

    rows = _rows(run_dir)
    names = (*b2.LAYER_SERIES, *b2.TEXT_CONTROLS)
    assert [(r["dataset"], r["method"]) for r in rows] == [
        (d, s) for d in config.datasets for s in names
    ]
    layers = list(range(1, tiny_arch.n_layers + 1))
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["kinds"] == dict(b2.KIND_OF)
    assert list(summary["datasets"]) == list(config.datasets)
    for d, name in enumerate(config.datasets):
        curves = {}
        for split in ("val", "test"):
            stem = f"curves_{name}_{split}"
            curves[split] = run_dir.read_arrays(stem)
            assert curves[split]["x"].tolist() == layers
            assert curves[split]["per_seed"].shape == (
                len(b2.LAYER_SERIES),
                config.n_seeds,
                len(layers),
            )
            sidecar = run_dir.read_json(stem)
            assert isinstance(sidecar, dict)
            assert sidecar["series"] == list(b2.LAYER_SERIES)
            assert sidecar["chance"] == 0.5
            assert sidecar["n_layers"] == tiny_arch.n_layers
        own = [r for r in rows if r["dataset"] == name]
        for s, row in enumerate(own[: len(b2.LAYER_SERIES)]):
            at = int(np.argmax(curves["val"]["mean"][s]))
            assert row["val_layer"] == layers[at]
            assert row["accuracy"] == pytest.approx(
                curves["test"]["mean"][s, at]
            )
        items = dataset_items(
            config,
            name,
            chat_format_for(tiny_arch),
            _sentiment(),
            _truth_sets(),
        )
        n_test = int(items.test.groups.shape[0])
        assert {row["n_test_pairs"] for row in own} == {n_test}
        ccs_at = layers.index(own[b2.LAYER_SERIES.index("ccs")]["val_layer"])
        seeds = probes[d * config.n_seeds : (d + 1) * config.n_seeds]
        assert summary["datasets"][name] == {
            "n_train_pairs": config.n_pairs,
            "n_val_pairs": int(items.val.groups.shape[0]),
            "n_test_pairs": n_test,
            "ccs_loss": pytest.approx(
                np.mean([p.losses[ccs_at] for p in seeds])
            ),
        }
    assert len(probes) == len(config.datasets) * config.n_seeds


def test_execute_forwards_the_reading_settings(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def spy(loaded: LoadedModel, texts: Any, **kwargs: Any) -> Activations:
        calls.append(kwargs)
        return collect_activations(loaded, texts, **kwargs)

    monkeypatch.setattr(b2, "collect_activations", spy)
    config = _config(layers=(1,), n_eval_pairs=3, bos="never", batch_size=5)
    run_dir = RunDir.create(config, out_root=tmp_path)

    b2.execute(config, run_dir, tiny_gpt2, _sentiment(), _truth_sets())

    assert calls
    for kwargs in calls:
        assert kwargs == {
            "layers": (1,),
            "position": "last",
            "bos": "never",
            "batch_size": 5,
        }
    assert {row["n_test_pairs"] for row in _rows(run_dir)} == {3}
    for name in config.datasets:
        assert run_dir.read_arrays(f"curves_{name}_test")["x"].tolist() == [1]


def test_execute_is_deterministic(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    config = _config()
    first = RunDir.create(config, out_root=tmp_path / "a")
    second = RunDir.create(config, out_root=tmp_path / "b")

    for run_dir in (first, second):
        b2.execute(config, run_dir, tiny_gpt2, _sentiment(), _truth_sets())

    assert _rows(first) == _rows(second)
    for name in config.datasets:
        a = first.read_arrays(f"curves_{name}_test")
        b = second.read_arrays(f"curves_{name}_test")
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_plot_writes_three_formats_per_curves_file(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    b2.execute(config, run_dir, tiny_gpt2, _sentiment(), _truth_sets())

    paths = plot(run_dir, tmp_path / "figs")

    stems = [
        f"curves_{name}_{split}"
        for name in sorted(config.datasets)
        for split in ("test", "val")
    ]
    assert sorted(p.name for p in paths) == [
        f"{stem}.{ext}" for stem in stems for ext in ("pdf", "png", "svg")
    ]
    assert all(p.stat().st_size > 0 for p in paths)


@pytest.mark.parametrize(
    ("datasets", "subsets", "sst2"),
    [
        (("sentiment", "cities"), [("cities", "neg_cities")], True),
        (
            ("larger_than", "cities"),
            [("larger_than", "smaller_than", "cities", "neg_cities")],
            False,
        ),
        (("sentiment",), [], True),
    ],
)
def test_run_loads_the_model_and_only_the_data_it_reads(
    datasets: tuple[str, ...],
    subsets: list[tuple[str, ...]],
    sst2: bool,
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []
    seen: list[tuple[str, ...]] = []
    sst2_loads: list[bool] = []

    def fake_load_model(key: str) -> LoadedModel:
        asked.append(key)
        return tiny_gpt2

    def fake_load_sst2() -> list[Example]:
        sst2_loads.append(True)
        return _sentiment()

    def fake_load_truth(*, subsets: tuple[str, ...]) -> list[Example]:
        seen.append(tuple(subsets))
        return _truth_sets(
            (("cities", "neg_cities"), ("larger_than", "smaller_than"))
        )

    monkeypatch.setattr(b2, "load_model", fake_load_model)
    monkeypatch.setattr(b2, "load_sst2", fake_load_sst2)
    monkeypatch.setattr(b2, "load_geometry_of_truth", fake_load_truth)
    config = _config(datasets=datasets, n_seeds=1)
    run_dir = RunDir.create(config, out_root=tmp_path)

    run(config, run_dir)

    assert asked == ["tiny"]
    assert seen == subsets
    assert sst2_loads == ([True] if sst2 else [])
    assert {row["dataset"] for row in _rows(run_dir)} == set(datasets)
