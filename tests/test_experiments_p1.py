"""Tests for P1, the template ablation study."""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from pathlib import Path
import time
from typing import Any
from typing import cast

import numpy as np
import pytest

from tomography.activations import PositionStrategy
from tomography.cli import load_experiment
from tomography.datasets.records import Example
from tomography.datasets.sources import OfflineCacheMissError
from tomography.datasets.sst2 import load_sst2
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import p1_templates
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.p1_templates import COLUMNS
from tomography.experiments.p1_templates import Config
from tomography.experiments.p1_templates import execute
from tomography.experiments.p1_templates import plot
from tomography.experiments.p1_templates import run
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import Prepared
from tomography.experiments.sweep import ReadingConfig
from tomography.experiments.variants import VARIANT_KINDS
from tomography.experiments.variants import ReadingTask
from tomography.experiments.variants import Variant
from tomography.experiments.variants import VariantKind
from tomography.experiments.variants import VariantSweep
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.metrics import best_layer
from tomography.templates import EMOTIONS


def test_columns_are_the_documented_table_header() -> None:
    assert COLUMNS == (
        "task",
        "variant",
        "kind",
        "family",
        "layer",
        "select",
        "report",
        "lo",
        "hi",
        "random",
        "shuffled",
        "from_paper",
        "to_paper",
        "cosine_paper",
        "stability",
        "own_layer",
        "own_report",
    )


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "p1"
    assert config.model == "gpt2"
    assert config.method == "logistic"
    assert config.n_eval_pairs == 256
    assert config.tasks == ("sentiment", "truth", "emotion")
    assert config.kinds == VARIANT_KINDS
    assert config.n_seeds == 10
    assert Config.from_dict(config.to_dict()) == config


def test_from_dict_reads_the_kinds_as_a_tuple_of_literals() -> None:
    config = Config.from_dict({"kinds": ["paper", "bare"], "n_seeds": 2})
    assert config.kinds == ("paper", "bare")
    assert config.n_seeds == 2


def test_from_dict_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kinds"):
        Config.from_dict({"kinds": ["paper", "shuffle"]})


def _sentiment(per_class: int = 12) -> list[Example]:
    words = {1: ("good", "fine", "great"), 0: ("bad", "poor", "dull")}
    return [
        Example(
            f"sst2-{split}-{label}-{i}",
            f"a {words[label][i % 3]} one {i}",
            label,
            split,
        )
        for split in ("train", "val")
        for label in (0, 1)
        for i in range(per_class)
    ]


def _facts(per_class: int = 8) -> list[Example]:
    return [
        Example(
            f"facts-{split}-{label}-{i}",
            f"fact {i} is {'yes' if label else 'no'}",
            label,
            split,
        )
        for split in ("train", "val", "test")
        for label in (0, 1)
        for i in range(per_class)
    ]


def _utility(pairs: int = 8) -> list[Example]:
    words = {
        1: ("sun", "gift", "joy", "fun"),
        0: ("rain", "loss", "pain", "fear"),
    }
    rows = []
    for split in ("train", "val", "test"):
        for i in range(pairs):
            pair = f"ethics-{split}-{i}"
            for suffix, label in (("a", 1), ("b", 0)):
                rows.append(
                    Example(
                        f"{pair}:{suffix}",
                        f"{words[label][i % 4]} {i}",
                        label,
                        split,
                        {"pair": pair},
                    )
                )
    return rows


def _emotions(per_class: int = 6) -> list[Example]:
    words = ("sun", "rain", "dog", "gift")
    return [
        Example(
            f"emotion-{name}-{split}-{i}",
            f"{words[i % 4]} {i}",
            label,
            split,
            {"emotion": name},
        )
        for split in ("train", "val", "test")
        for label, name in enumerate(EMOTIONS)
        for i in range(per_class)
    ]


def _truth_sets(per_class: int = 6) -> list[Example]:
    return [
        Example(
            f"got-{subset}-{split}-{label}-{i}",
            f"{subset[:4]} {i} {label}",
            label,
            split,
            {"subset": subset},
        )
        for subset in ("cities", "common_claim_true_false")
        for split in ("train", "val", "test")
        for label in (0, 1)
        for i in range(per_class)
    ]


def _sources() -> dict[str, list[Example]]:
    return {
        "sst2": _sentiment(),
        "facts": _facts(),
        "ethics_utility": _utility(),
        "emotions": _emotions(),
        "geometry_of_truth": _truth_sets(),
    }


def _config(**overrides: Any) -> Config:
    base = Config(model="tiny", n_seeds=2, n_pairs=4, n_eval_pairs=3)
    return dataclasses.replace(base, **overrides)


def _run_dir(config: Config, tmp_path: Path) -> RunDir:
    return RunDir.create(config, out_root=tmp_path)


def test_one_seed_raises_before_any_model_call(tmp_path: Path) -> None:
    config = _config(n_seeds=1)
    with pytest.raises(ValueError, match="n_seeds"):
        execute(config, _run_dir(config, tmp_path), cast(Any, None), _sources())


def test_a_missing_source_raises_before_any_model_call(tmp_path: Path) -> None:
    config = _config(tasks=("sentiment",))
    examples = {k: v for k, v in _sources().items() if k != "sst2"}
    with pytest.raises(ValueError, match="no examples for source 'sst2'"):
        execute(config, _run_dir(config, tmp_path), cast(Any, None), examples)


def test_a_missing_target_source_raises_before_any_model_call(
    tmp_path: Path,
) -> None:
    config = _config(tasks=("truth_cities",))
    examples = {"geometry_of_truth": _truth_sets()}
    with pytest.raises(ValueError, match="no examples for source 'facts'"):
        execute(config, _run_dir(config, tmp_path), cast(Any, None), examples)


def test_an_unknown_task_raises_before_any_model_call(tmp_path: Path) -> None:
    config = _config(tasks=("sentiment", "x"))
    with pytest.raises(ValueError, match="unknown task 'x'"):
        execute(config, _run_dir(config, tmp_path), cast(Any, None), _sources())


READER = SERIES.index("reader")
RANDOM = SERIES.index("random")
SHUFFLED = SERIES.index("shuffled")
SMALL_KINDS: tuple[VariantKind, ...] = (
    "paper",
    "no_template",
    "paraphrase",
    "concept_swap",
)
SMALL_VARIANTS = [
    "paper",
    "no_template",
    "paraphrase_0",
    "paraphrase_1",
    "paraphrase_2",
    "concept_swap_0",
    "concept_swap_1",
    "concept_swap_2",
    "concept_swap_3",
]


def _execute(
    config: Config,
    loaded: LoadedModel,
    tmp_path: Path,
    sources: Mapping[str, Sequence[Example]] | None = None,
) -> RunDir:
    run_dir = _run_dir(config, tmp_path)
    execute(config, run_dir, loaded, _sources() if sources is None else sources)
    return run_dir


def _json(run_dir: RunDir, name: str) -> Any:
    return run_dir.read_json(name)


def _small(**overrides: Any) -> Config:
    return _config(tasks=("sentiment",), kinds=SMALL_KINDS, **overrides)


def _stems(run_dir: RunDir) -> set[str]:
    return {p.name for p in run_dir.path.iterdir()}


def test_execute_writes_the_files_of_a_task_and_the_two_summaries(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(_small(), tiny_arch, tmp_path)
    stems = [
        "sweep_sentiment",
        "curves",
        "curves_val",
        "curves_sentiment",
        "curves_sentiment_val",
        "curves_sentiment_frame",
        "curves_sentiment_wording",
        "curves_sentiment_concept",
        "transfer_sentiment",
        "cosine_sentiment",
    ]
    expected = {f"{stem}.{ext}" for stem in stems for ext in ("npz", "json")}
    expected |= {"table.json", "summary.json", "config.json", "meta.json"}
    assert _stems(run_dir) == expected


def test_the_sweep_file_keeps_the_whole_accuracy_cube(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(_small(), tiny_arch, tmp_path)
    arrays = run_dir.read_arrays("sweep_sentiment")
    layers = list(range(1, tiny_arch.n_layers + 1))
    n_variants, n_layers = len(SMALL_VARIANTS), len(layers)
    assert set(arrays) == {"accuracy", "cosine", "layers"}
    assert arrays["accuracy"].shape == (
        2,
        n_variants,
        n_variants,
        len(SERIES),
        2,
        n_layers,
    )
    assert arrays["cosine"].shape == (n_layers, n_variants, n_variants)
    assert arrays["layers"].tolist() == layers
    assert arrays["layers"].dtype == np.int64
    sidecar = _json(run_dir, "sweep_sentiment")
    assert sidecar["splits"] == ["select", "report"]
    assert sidecar["train_variants"] == SMALL_VARIANTS
    assert sidecar["eval_variants"] == SMALL_VARIANTS
    assert sidecar["series"] == list(SERIES)
    assert sidecar["layer"] in layers


def test_the_layer_is_the_paper_readers_best_select_layer(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(_small(), tiny_gpt2, tmp_path)
    arrays = run_dir.read_arrays("sweep_sentiment")
    select = arrays["accuracy"][0, 0, 0, READER].mean(axis=0)
    layer = best_layer(select, arrays["layers"].tolist())
    assert _json(run_dir, "sweep_sentiment")["layer"] == layer
    assert _json(run_dir, "summary")["layers"] == {"sentiment": layer}
    assert {row["layer"] for row in _json(run_dir, "table")} == {layer}


def test_the_matrices_are_the_report_split_at_that_layer(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(_small(), tiny_gpt2, tmp_path)
    arrays = run_dir.read_arrays("sweep_sentiment")
    layer = _json(run_dir, "sweep_sentiment")["layer"]
    k = arrays["layers"].tolist().index(layer)
    transfer = run_dir.read_arrays("transfer_sentiment")["values"]
    np.testing.assert_allclose(
        transfer, arrays["accuracy"][1, :, :, READER, :, k].mean(axis=-1)
    )
    cosine = run_dir.read_arrays("cosine_sentiment")["values"]
    np.testing.assert_allclose(cosine, arrays["cosine"][k])
    for stem in ("transfer_sentiment", "cosine_sentiment"):
        sidecar = _json(run_dir, stem)
        assert sidecar["rows"] == SMALL_VARIANTS
        assert sidecar["cols"] == SMALL_VARIANTS
        assert f"layer {layer}" in sidecar["title"]


def test_the_curves_hold_one_reader_series_per_variant(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(_small(), tiny_gpt2, tmp_path)
    accuracy = run_dir.read_arrays("sweep_sentiment")["accuracy"]
    pairs = (
        ("curves_sentiment", 1),
        ("curves_sentiment_val", 0),
        ("curves", 1),
        ("curves_val", 0),
    )
    for stem, split in pairs:
        curves = run_dir.read_arrays(stem)
        sidecar = _json(run_dir, stem)
        assert sidecar["series"] == SMALL_VARIANTS
        assert sidecar["chance"] == 0.5
        assert sidecar["n_layers"] == tiny_gpt2.n_layers
        assert curves["per_seed"].shape == (9, 2, tiny_gpt2.n_layers)
        for v in range(len(SMALL_VARIANTS)):
            np.testing.assert_array_equal(
                curves["per_seed"][v], accuracy[split, v, v, READER]
            )


def test_each_family_curve_is_the_paper_and_the_family_members(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(_small(), tiny_gpt2, tmp_path)
    accuracy = run_dir.read_arrays("sweep_sentiment")["accuracy"]
    members = {
        "frame": ["paper", "no_template"],
        "wording": ["paper", "paraphrase_0", "paraphrase_1", "paraphrase_2"],
        "concept": [
            "paper",
            "concept_swap_0",
            "concept_swap_1",
            "concept_swap_2",
            "concept_swap_3",
        ],
    }
    for family, names in members.items():
        stem = f"curves_sentiment_{family}"
        assert _json(run_dir, stem)["series"] == names
        per_seed = run_dir.read_arrays(stem)["per_seed"]
        for row, name in zip(per_seed, names, strict=True):
            v = SMALL_VARIANTS.index(name)
            np.testing.assert_array_equal(row, accuracy[1, v, v, READER])


def test_a_family_with_only_the_paper_variant_is_not_written(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    config = _config(
        tasks=("utility",), kinds=("paper", "no_template", "concept_swap")
    )
    run_dir = _execute(config, tiny_gpt2, tmp_path)
    assert (run_dir.path / "curves_utility_frame.npz").is_file()
    assert not (run_dir.path / "curves_utility_wording.npz").exists()
    assert not (run_dir.path / "curves_utility_concept.npz").exists()
    assert _json(run_dir, "summary")["skipped"]["utility"] == {
        "concept_swap": "template 'utility' has no concept slot"
    }


def test_the_table_follows_the_columns_and_the_arrays(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(_small(), tiny_gpt2, tmp_path)
    arrays = run_dir.read_arrays("sweep_sentiment")
    acc, cos = arrays["accuracy"], arrays["cosine"]
    layers = arrays["layers"].tolist()
    layer = _json(run_dir, "sweep_sentiment")["layer"]
    k = layers.index(layer)
    rows = _json(run_dir, "table")
    assert [row["variant"] for row in rows] == SMALL_VARIANTS
    for v, row in enumerate(rows):
        assert tuple(row) == COLUMNS
        assert row["task"] == "sentiment"
        assert row["layer"] == layer
        assert row["select"] == pytest.approx(acc[0, v, v, READER, :, k].mean())
        assert row["report"] == pytest.approx(acc[1, v, v, READER, :, k].mean())
        assert row["lo"] <= row["report"] <= row["hi"]
        assert row["random"] == pytest.approx(acc[1, v, v, RANDOM, :, k].mean())
        assert row["shuffled"] == pytest.approx(
            acc[1, v, v, SHUFFLED, :, k].mean()
        )
        assert row["from_paper"] == pytest.approx(
            acc[1, 0, v, READER, :, k].mean()
        )
        assert row["to_paper"] == pytest.approx(
            acc[1, v, 0, READER, :, k].mean()
        )
        assert row["cosine_paper"] == pytest.approx(cos[k, v, 0])
        assert row["stability"] == pytest.approx(cos[k, v, v])
        own_layer = best_layer(acc[0, v, v, READER].mean(axis=0), layers)
        assert row["own_layer"] == own_layer
        assert row["own_report"] == pytest.approx(
            acc[1, v, v, READER, :, layers.index(own_layer)].mean()
        )
    paper = rows[0]
    assert paper["from_paper"] == paper["to_paper"] == paper["report"]
    assert paper["cosine_paper"] == paper["stability"]
    assert [row["kind"] for row in rows][:2] == ["paper", "no_template"]
    assert {row["family"] for row in rows} == {"frame", "wording", "concept"}


def test_a_function_variant_is_a_train_only_row_on_the_paper_items(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(
        tasks=("emotion_happiness",), kinds=("paper", "framing_swap")
    )
    run_dir = _execute(config, tiny_arch, tmp_path)
    sidecar = _json(run_dir, "sweep_emotion_happiness")
    assert sidecar["train_variants"] == ["paper", "framing_swap"]
    assert sidecar["eval_variants"] == ["paper"]
    acc = run_dir.read_arrays("sweep_emotion_happiness")["accuracy"]
    layers = run_dir.read_arrays("sweep_emotion_happiness")["layers"].tolist()
    k = layers.index(sidecar["layer"])
    paper, framing = _json(run_dir, "table")
    assert framing["variant"] == "framing_swap"
    assert framing["from_paper"] is None
    assert framing["to_paper"] == pytest.approx(
        acc[1, 1, 0, READER, :, k].mean()
    )
    assert framing["report"] == pytest.approx(framing["to_paper"])
    assert framing["select"] == pytest.approx(acc[0, 1, 0, READER, :, k].mean())
    assert paper["from_paper"] == pytest.approx(paper["report"])
    transfer = run_dir.read_arrays("transfer_emotion_happiness")["values"]
    assert transfer.shape == (2, 1)
    assert _json(run_dir, "transfer_emotion_happiness")["cols"] == ["paper"]


def test_the_summary_lists_the_tasks_layers_skips_and_pair_counts(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    config = _config(
        tasks=("utility", "sentiment"),
        kinds=("paper", "concept_swap"),
        n_eval_pairs=1000,
    )
    run_dir = _execute(config, tiny_gpt2, tmp_path)
    summary = _json(run_dir, "summary")
    assert set(summary) == {
        "tasks",
        "layers",
        "skipped",
        "n_eval_pairs",
        "method",
        "curves_task",
    }
    assert summary["tasks"] == ["utility", "sentiment"]
    assert summary["curves_task"] == "utility"
    assert summary["method"] == "logistic"
    assert set(summary["layers"]) == {"utility", "sentiment"}
    assert summary["skipped"]["sentiment"] == {}
    assert list(summary["skipped"]["utility"]) == ["concept_swap"]
    assert summary["n_eval_pairs"]["utility"] == {"select": 8, "report": 8}
    sentiment = summary["n_eval_pairs"]["sentiment"]
    assert 1 <= sentiment["select"] <= 12
    assert 1 <= sentiment["report"] <= 12


def test_execute_is_deterministic(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    first = _execute(_small(), tiny_gpt2, tmp_path / "a")
    second = _execute(_small(), tiny_gpt2, tmp_path / "b")
    for stem in ("sweep_sentiment", "transfer_sentiment", "curves_sentiment"):
        a, b = first.read_arrays(stem), second.read_arrays(stem)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    assert _json(first, "table") == _json(second, "table")


def test_the_plain_curves_are_the_pair_of_the_first_task(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    config = _config(
        tasks=("utility", "sentiment"), kinds=("paper", "concept_swap")
    )
    run_dir = _execute(config, tiny_gpt2, tmp_path)
    assert _json(run_dir, "summary")["curves_task"] == "utility"
    for plain, stem in (
        ("curves", "curves_utility"),
        ("curves_val", "curves_utility_val"),
    ):
        assert _json(run_dir, plain)["series"] == ["paper"]
        a, b = run_dir.read_arrays(plain), run_dir.read_arrays(stem)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        assert _json(run_dir, plain)["n_layers"] == tiny_gpt2.n_layers


def test_four_tasks_write_seventy_files_and_sixty_one_rows(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(
        tasks=("sentiment", "truth", "utility", "emotion_happiness")
    )
    run_dir = _execute(config, tiny_arch, tmp_path)
    assert len(_stems(run_dir)) == 70
    rows = _json(run_dir, "table")
    assert len(rows) == 61
    counts = {name: 0 for name in _json(run_dir, "summary")["tasks"]}
    for row in rows:
        counts[row["task"]] += 1
    assert counts == {
        "sentiment": 17,
        "truth": 17,
        "utility": 9,
        "emotion_happiness": 18,
    }
    assert all(tuple(row) == COLUMNS for row in rows)
    (framing,) = (r for r in rows if r["variant"] == "framing_swap")
    assert framing["from_paper"] is None
    assert all(
        row["from_paper"] is not None
        for row in rows
        if row["variant"] != "framing_swap"
    )
    skipped = _json(run_dir, "summary")["skipped"]
    assert set(skipped["sentiment"]) == {"alternate", "framing_swap"}
    assert skipped["emotion_happiness"] == {}
    assert len(plot(run_dir, tmp_path / "figures")) == 57


def test_the_evaluable_variants_must_share_their_items(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = p1_templates.task_set

    def drifting(
        task: ReadingTask,
        variant: Variant,
        examples: Sequence[Example],
        *,
        n_pairs: int | None,
        seed: int,
    ) -> StimulusSet:
        stimuli = real(task, variant, examples, n_pairs=n_pairs, seed=seed)
        if variant.name == "no_template" and n_pairs == 3:
            return dataclasses.replace(
                stimuli, uids=tuple(f"{u}!" for u in stimuli.uids)
            )
        return stimuli

    monkeypatch.setattr(p1_templates, "task_set", drifting)
    with pytest.raises(ValueError, match="do not share their select items"):
        _execute(_small(), tiny_gpt2, tmp_path)


def test_a_source_and_its_target_are_scored_on_the_same_items(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = p1_templates.task_set
    calls: list[tuple[str, int | None, int, tuple[str, ...]]] = []

    def spy(
        task: ReadingTask,
        variant: Variant,
        examples: Sequence[Example],
        *,
        n_pairs: int | None,
        seed: int,
    ) -> StimulusSet:
        uids = tuple(e.uid for e in examples)
        calls.append((task.name, n_pairs, seed, uids))
        return real(task, variant, examples, n_pairs=n_pairs, seed=seed)

    monkeypatch.setattr(p1_templates, "task_set", spy)
    config = _config(tasks=("truth", "truth_cities"), kinds=("paper",))
    _execute(config, tiny_gpt2, tmp_path)
    evals = {
        name: [
            (n, seed, uids)
            for t, n, seed, uids in calls
            if t == name and n == 3
        ]
        for name in ("truth", "truth_cities")
    }
    assert len(evals["truth"]) == 2
    assert evals["truth"] == evals["truth_cities"]
    assert all(
        u.startswith("facts-")
        for _, _, uids in evals["truth_cities"]
        for u in uids
    )
    trained = {
        name: {
            u for t, n, _, uids in calls if t == name and n == 4 for u in uids
        }
        for name in ("truth", "truth_cities")
    }
    assert all(u.startswith("facts-train") for u in trained["truth"])
    assert all(
        u.startswith("got-cities-train") for u in trained["truth_cities"]
    )


def test_every_pair_is_read_when_n_eval_pairs_is_none(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    config = _config(tasks=("truth",), kinds=("paper",), n_eval_pairs=None)
    run_dir = _execute(config, tiny_gpt2, tmp_path)
    sizes = _json(run_dir, "summary")["n_eval_pairs"]
    assert sizes == {"truth": {"select": 8, "report": 8}}


def test_positions_are_trained_at_the_config_and_evaluated_at_eval_position(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = p1_templates.prepare
    calls: list[tuple[int, str]] = []

    def spy(
        loaded: LoadedModel,
        stimuli: StimulusSet,
        position: PositionStrategy,
        config: ReadingConfig,
    ) -> Prepared:
        calls.append((len(stimuli.texts), position))
        return real(loaded, stimuli, position, config)

    monkeypatch.setattr(p1_templates, "prepare", spy)
    config = _config(
        tasks=("emotion_happiness",),
        kinds=("paper", "framing_swap"),
        position="mean",
        eval_position="last",
    )
    _execute(config, tiny_gpt2, tmp_path)
    assert sorted(calls) == sorted(
        [(6, "last")] * 2 + [(8, "mean")] * 2 + [(8, "response")] * 2
    )


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("p1")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_run_loads_the_model_and_the_distinct_sources_once(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []
    loaded_sources: list[list[str]] = []

    def fake_load_model(key: str) -> LoadedModel:
        asked.append(key)
        return tiny_gpt2

    def fake_load_sources(sources: Sequence[str]) -> dict[str, list[Example]]:
        loaded_sources.append(list(sources))
        return _sources()

    monkeypatch.setattr(p1_templates, "load_model", fake_load_model)
    monkeypatch.setattr(p1_templates, "load_sources", fake_load_sources)
    config = _config(
        tasks=("truth_cities", "emotion_happiness"), kinds=("paper",)
    )
    run_dir = _run_dir(config, tmp_path)
    run(config, run_dir)
    assert asked == ["tiny"]
    assert loaded_sources == [["geometry_of_truth", "facts", "emotions"]]
    assert _json(run_dir, "summary")["tasks"] == [
        "truth_cities",
        "emotion_happiness",
    ]


def test_plot_draws_the_family_curves_and_both_matrices(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    config = _config(tasks=("sentiment", "truth", "utility"), kinds=SMALL_KINDS)
    run_dir = _execute(config, tiny_gpt2, tmp_path)
    paths = plot(run_dir, tmp_path / "figures")
    names = {path.name for path in paths}
    assert len(paths) == len(names)
    assert {p.suffix for p in paths} == {".pdf", ".png", ".svg"}
    for task in ("sentiment", "truth", "utility"):
        for stem in (f"transfer_{task}", f"cosine_{task}"):
            assert {f"{stem}.pdf", f"{stem}.png", f"{stem}.svg"} <= names
    assert len(paths) == 3 * 13
    assert "curves_sentiment_concept.pdf" in names
    assert "curves_utility_frame.pdf" in names
    assert "curves_utility_wording.pdf" not in names
    assert "curves_utility_concept.pdf" not in names
    assert all(path.is_file() for path in paths)


@pytest.mark.slow
def test_p1_on_gpt2_reads_sentiment_through_three_templates(
    tmp_path: Path,
) -> None:
    try:
        sst2 = load_sst2(offline=True)
    except (OfflineCacheMissError, OSError) as error:
        pytest.skip(f"a cache is missing: {error}")
    config = Config(
        tasks=("sentiment",),
        kinds=("paper", "no_trailing_space", "bare"),
        n_seeds=2,
        n_pairs=16,
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    started = time.perf_counter()
    execute(config, run_dir, load_model("gpt2"), {"sst2": sst2})
    elapsed = time.perf_counter() - started
    rows = cast(list[dict[str, Any]], run_dir.read_json("table"))
    for row in rows:
        print({key: row[key] for key in ("variant", "layer", "report", "lo")})
    print(f"{elapsed / len(rows):.1f} seconds per variant")
    paper = next(row for row in rows if row["variant"] == "paper")
    assert paper["report"] > 0.5


def test_the_layer_is_chosen_on_the_select_split(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = p1_templates.variant_sweep

    def skewed(*args: Any, **kwargs: Any) -> VariantSweep:
        sweep = real(*args, **kwargs)
        accuracy = sweep.accuracy.copy()
        accuracy[:, :, :, READER] = 0.5
        accuracy[0, :, :, READER, :, 0] = 0.9
        accuracy[1, :, :, READER, :, -1] = 0.9
        return dataclasses.replace(sweep, accuracy=accuracy)

    monkeypatch.setattr(p1_templates, "variant_sweep", skewed)
    run_dir = _execute(_small(), tiny_gpt2, tmp_path)
    assert _json(run_dir, "summary")["layers"] == {"sentiment": 1}
    assert _json(run_dir, "sweep_sentiment")["layer"] == 1
    rows = _json(run_dir, "table")
    assert {row["layer"] for row in rows} == {1}
    assert {row["own_layer"] for row in rows} == {1}


def test_every_variant_trains_on_the_same_seeded_draws(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = p1_templates.task_set
    seeds: dict[str, list[int]] = {}

    def spy(
        task: ReadingTask,
        variant: Variant,
        examples: Sequence[Example],
        *,
        n_pairs: int | None,
        seed: int,
    ) -> StimulusSet:
        if n_pairs == 4:
            seeds.setdefault(variant.name, []).append(seed)
        return real(task, variant, examples, n_pairs=n_pairs, seed=seed)

    monkeypatch.setattr(p1_templates, "task_set", spy)
    _execute(_small(), tiny_gpt2, tmp_path)
    expected = [derive_seed(0, "train", i) for i in range(2)]
    assert expected[0] != expected[1]
    assert seeds == {name: expected for name in SMALL_VARIANTS}


def test_the_written_cube_is_the_sweep_under_the_run_seed(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = p1_templates.variant_sweep
    calls: list[tuple[Any, Any]] = []

    def spy(train: Any, evals: Any, **kwargs: Any) -> VariantSweep:
        calls.append((train, evals))
        return real(train, evals, **kwargs)

    monkeypatch.setattr(p1_templates, "variant_sweep", spy)
    config = _small(seed=3)
    run_dir = _execute(config, tiny_gpt2, tmp_path)
    ((train, evals),) = calls
    direct = real(
        train,
        evals,
        method=config.method,
        preprocess=config.preprocess,
        seed=3,
    )
    cube = run_dir.read_arrays("sweep_sentiment")["accuracy"]
    np.testing.assert_array_equal(cube, direct.accuracy)


def test_no_task_is_refused_before_any_model_call(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one task"):
        execute(
            _config(tasks=()),
            _run_dir(_config(tasks=()), tmp_path),
            cast(Any, None),
            _sources(),
        )


@pytest.mark.parametrize("field", ["position", "eval_position"])
@pytest.mark.parametrize("position", ["concept", "response"])
def test_a_position_that_not_every_variant_has_is_refused(
    tmp_path: Path, field: str, position: PositionStrategy
) -> None:
    config = _config(**{field: position})
    with pytest.raises(ValueError, match=f"{field} '{position}'"):
        execute(config, _run_dir(config, tmp_path), cast(Any, None), _sources())


@pytest.mark.parametrize("eval_position", ["mean", None])
def test_the_mean_position_is_read(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    eval_position: PositionStrategy | None,
) -> None:
    real = p1_templates.prepare
    read: list[PositionStrategy] = []

    def spy(
        loaded: LoadedModel,
        stimuli: StimulusSet,
        position: PositionStrategy,
        config: ReadingConfig,
    ) -> Prepared:
        read.append(position)
        return real(loaded, stimuli, position, config)

    monkeypatch.setattr(p1_templates, "prepare", spy)
    config = _small(position="mean", eval_position=eval_position)
    run_dir = _execute(config, tiny_gpt2, tmp_path)
    assert len(_json(run_dir, "table")) == len(SMALL_VARIANTS)
    assert len(read) > len(SMALL_VARIANTS) * config.n_seeds
    assert set(read) == {"mean"}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_seeds": 1}, "n_seeds"),
        ({"tasks": ()}, "at least one task"),
        ({"tasks": ("nope",)}, "unknown task 'nope'"),
        ({"position": "concept"}, "position 'concept'"),
        ({"eval_position": "response"}, "eval_position 'response'"),
        ({"kinds": ("bare",)}, "kinds must include 'paper'"),
        ({"kinds": ("paper", "nope")}, r"unknown variant kinds \['nope'\]"),
    ],
)
def test_run_checks_the_config_before_loading_anything(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    message: str,
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("nothing may load before the config is checked")

    monkeypatch.setattr(p1_templates, "load_model", refuse)
    monkeypatch.setattr(p1_templates, "load_sources", refuse)
    config = _config(**overrides)
    with pytest.raises(ValueError, match=message):
        run(config, _run_dir(config, tmp_path))
