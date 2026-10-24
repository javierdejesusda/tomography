"""Tests for M2, the factorial reader comparison."""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from pathlib import Path
from typing import Any
from typing import cast
import warnings
import weakref

import numpy as np
from numpy.typing import NDArray
import pytest
from sklearn.exceptions import ConvergenceWarning
import torch

from tomography.activations import Activations
from tomography.activations import collect_activations
from tomography.cli import _parse_override
from tomography.cli import load_experiment
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import m2_readers
from tomography.experiments import p2_positions
from tomography.experiments.base import RunDir
from tomography.experiments.m2_readers import COLUMNS
from tomography.experiments.m2_readers import COMPARISON_COLUMNS
from tomography.experiments.m2_readers import CONTROLS
from tomography.experiments.m2_readers import LAYER_RULES
from tomography.experiments.m2_readers import POLARITY_TASKS
from tomography.experiments.m2_readers import POSITION_OF
from tomography.experiments.m2_readers import READOUTS
from tomography.experiments.m2_readers import SCORERS
from tomography.experiments.m2_readers import Cell
from tomography.experiments.m2_readers import Config
from tomography.experiments.m2_readers import applies
from tomography.experiments.m2_readers import cells
from tomography.experiments.m2_readers import execute
from tomography.experiments.m2_readers import plot
from tomography.experiments.m2_readers import polarity
from tomography.experiments.m2_readers import read_positions
from tomography.experiments.m2_readers import read_top_k
from tomography.experiments.m2_readers import run
from tomography.experiments.m2_readers import token_score_max
from tomography.experiments.m2_readers import tune_per_layer
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import prepare
from tomography.experiments.variants import task_set
from tomography.experiments.variants import template_variants
from tomography.model import LoadedModel
from tomography.reading.metrics import auroc
from tomography.reading.metrics import group_accuracy
from tomography.reading.stats import delong_test
from tomography.reading.stats import holm_correction
from tomography.reading.stats import paired_auroc_diff
from tomography.reading.stats import threshold_metrics
from tomography.reading.stats import top_k_layers

DEFAULT_HASH = "bb7e525e2aad"


def test_the_factorial_names_its_scorers_readouts_and_rules() -> None:
    assert SCORERS == (
        "pca_diff",
        "mean_diff",
        "logistic",
        "lda",
        "ttpd",
        "mlp",
    )
    assert READOUTS == (
        "last",
        "concept",
        "concept_mean",
        "mean",
        "max",
        "token_score_max",
        "attention",
    )
    assert LAYER_RULES == ("best", "top_k", "joint")
    assert CONTROLS == ("text", "random", "shuffled")
    assert dict(POSITION_OF) == {
        "last": "last",
        "concept": "concept",
        "concept_mean": "concept_mean",
        "mean": "mean",
        "max": "max",
        "token_score_max": "response",
        "attention": "response",
    }


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "m2"
    assert config.n_eval_pairs == 256
    assert config.tasks == ("sentiment", "truth", "truth_polarity", "utility")
    assert config.scorers == SCORERS
    assert config.readouts == READOUTS
    assert config.token_layers is None
    assert config.layer_rules == LAYER_RULES
    assert config.top_k == 5
    assert config.decoys == ("random", "against")
    assert config.shifts == ("paraphrase",)
    assert config.controls == CONTROLS
    assert config.logistic_c == (0.01, 0.1, 1.0, 10.0)
    assert config.mlp_alpha == (0.1, 0.01, 0.001, 0.0001)
    assert config.baselines == ("logistic/last", "mean_diff/last")
    assert Config.from_dict(config.to_dict()) == config


def test_the_default_config_hash_is_stable() -> None:
    assert Config().config_hash() == DEFAULT_HASH


def test_m2_is_discovered_by_the_cli() -> None:
    experiment = load_experiment("m2")
    assert experiment.config_cls is Config


@pytest.mark.parametrize("position", ["max", "concept_mean"])
def test_set_accepts_the_new_position_strategies(position: str) -> None:
    key, value = _parse_override(Config, f"position={position}")
    config = Config.from_dict({key: value})
    assert config.position == position


def test_set_reads_readouts_scorers_and_token_layers_as_tuples() -> None:
    data = dict(
        [
            _parse_override(Config, 'readouts=["max", "token_score_max"]'),
            _parse_override(Config, 'scorers=["ttpd"]'),
            _parse_override(Config, "token_layers=[7, 14]"),
        ]
    )
    config = Config.from_dict(data)
    assert config.readouts == ("max", "token_score_max")
    assert config.scorers == ("ttpd",)
    assert config.token_layers == (7, 14)


def test_attention_is_read_by_the_logistic_scorer_only() -> None:
    assert applies("logistic", "attention", "best")
    assert applies("logistic", "attention", "joint")
    for scorer in ("pca_diff", "mean_diff", "lda", "ttpd", "mlp", "random"):
        assert not applies(scorer, "attention", "best")


def test_the_joint_rule_fits_the_mlp_and_the_attention_reader() -> None:
    assert applies("mlp", "last", "joint")
    assert applies("mlp", "mean", "joint")
    assert not applies("mlp", "token_score_max", "joint")
    for scorer in ("pca_diff", "mean_diff", "logistic", "lda", "ttpd"):
        assert not applies(scorer, "last", "joint")
        assert applies(scorer, "last", "top_k")


def test_cells_list_the_factorial_then_the_controls() -> None:
    config = Config(scorers=("logistic", "mlp"), readouts=("last", "attention"))

    listed = cells(config, config.readouts)

    assert listed == [
        Cell("logistic", "last", "best"),
        Cell("logistic", "last", "top_k"),
        Cell("logistic", "attention", "best"),
        Cell("logistic", "attention", "top_k"),
        Cell("logistic", "attention", "joint"),
        Cell("mlp", "last", "best"),
        Cell("mlp", "last", "top_k"),
        Cell("mlp", "last", "joint"),
        Cell("random", "last", "best"),
        Cell("random", "last", "top_k"),
        Cell("text", "text", "none"),
    ]
    assert listed[0].name == "logistic/last/best"


def test_cells_leave_out_the_controls_that_are_not_asked_for() -> None:
    config = Config(scorers=("lda",), readouts=("mean",), controls=())

    assert cells(config, config.readouts) == [
        Cell("lda", "mean", "best"),
        Cell("lda", "mean", "top_k"),
    ]


def test_cells_list_only_the_scorers_a_task_can_read() -> None:
    config = Config(scorers=("lda", "ttpd"), readouts=("mean",), controls=())

    assert cells(config, config.readouts, scorers=("lda",)) == [
        Cell("lda", "mean", "best"),
        Cell("lda", "mean", "top_k"),
    ]


def test_polarity_is_read_from_the_subset_name() -> None:
    subsets = POLARITY_TASKS["truth_polarity"]

    signs = [
        polarity(Example(f"got-{s}-0", "x", 1, "train", {"subset": s}))
        for s in subsets
    ]

    assert subsets == ("cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans")
    assert signs == [1, -1, 1, -1]


def _token_rows(counts: Sequence[int]) -> Activations:
    n_rows = sum(counts)
    stimulus = np.repeat(np.arange(len(counts)), counts).astype(np.int64)
    return Activations(
        data=np.zeros((1, n_rows, 2), dtype=np.float32),
        layers=(1,),
        position="response",
        row_stimulus=stimulus,
        row_rank=np.concatenate([np.arange(c) for c in counts]).astype(
            np.int64
        ),
        row_token=np.arange(n_rows, dtype=np.int64),
        row_token_id=np.zeros(n_rows, dtype=np.int64),
        n_stimuli=len(counts),
    )


def test_token_score_max_takes_the_top_token_score_of_each_stimulus() -> None:
    acts = _token_rows([2, 3, 1])
    scores = np.array(
        [[0.5, -1.0, 2.0, 7.0, -3.0, -4.0], [1.0, 1.0, 0.0, 0.0, 9.0, 2.0]]
    )

    pooled = token_score_max(scores, acts)

    np.testing.assert_array_equal(pooled, [[0.5, 7.0, -4.0], [1.0, 9.0, 2.0]])


def test_token_score_max_rejects_a_stimulus_without_rows() -> None:
    acts = dataclasses.replace(_token_rows([2, 1]), n_stimuli=3)

    with pytest.raises(ValueError, match="stimulus 2 has no rows"):
        token_score_max(np.zeros((1, 3)), acts)


def test_token_score_max_rejects_scores_of_another_shape() -> None:
    with pytest.raises(ValueError, match="expected scores of shape"):
        token_score_max(np.zeros((1, 4)), _token_rows([2, 1]))


def test_tuning_keeps_the_first_best_grid_value_of_each_layer() -> None:
    metrics = np.array([[0.5, 0.9, 0.7], [0.8, 0.9, 0.6], [0.8, 0.2, 0.7]])

    np.testing.assert_array_equal(tune_per_layer(metrics), [1, 0, 0])


def test_read_top_k_skips_layers_whose_select_scores_are_constant() -> None:
    metric = np.array([0.9, 0.5, 0.8, 0.7])
    val = np.array(
        [[1.0, 1.0, 1.0], [0.0, 1.0, 2.0], [3.0, 1.0, 2.0], [1.0, 5.0, 0.0]]
    )
    test = np.arange(8.0).reshape(4, 2)

    choice = read_top_k(metric, [1, 2, 3, 4], val, test, k=2)

    expected = top_k_layers(metric[1:], [2, 3, 4], val[1:], test[1:], k=2)
    assert choice.layers == (3, 4)
    np.testing.assert_array_equal(choice.test_scores, expected.test_scores)


def test_read_top_k_reads_fewer_layers_when_few_vary() -> None:
    metric = np.array([0.9, 0.5])
    val = np.array([[1.0, 1.0], [0.0, 2.0]])

    choice = read_top_k(metric, [1, 2], val, np.ones((2, 3)), k=2)

    assert choice.layers == (2,)


def test_read_top_k_falls_back_to_the_best_layer_when_none_varies() -> None:
    val = np.ones((2, 3))
    test = np.array([[1.0, 2.0], [3.0, 4.0]])

    choice = read_top_k(np.array([0.5, 0.6]), [1, 2], val, test, k=2)

    assert choice.layers == (2,)
    np.testing.assert_array_equal(choice.test_scores, [3.0, 4.0])


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
                        f"a {words[label][i % 4]} day {i}",
                        label,
                        split,
                        {"pair": pair},
                    )
                )
    return rows


def _statements(per_class: int = 6) -> list[Example]:
    """Geometry of Truth rows of the polarity task, and of another subset."""
    rows = []
    for subset in (*POLARITY_TASKS["truth_polarity"], "larger_than"):
        negated = "not " if subset.startswith("neg_") else ""
        noun = subset.split("_")[-1]
        for split in ("train", "val", "test"):
            for label, value in ((0, "wet"), (1, "dry")):
                for i in range(per_class):
                    rows.append(
                        Example(
                            f"got-{subset}-{split}-{label}-{i}",
                            f"the {noun} {i} is {negated}{value}",
                            label,
                            split,
                            {"subset": subset},
                        )
                    )
    return rows


def _sources() -> dict[str, list[Example]]:
    return {
        "sst2": _sentiment(),
        "ethics_utility": _utility(),
        "geometry_of_truth": _statements(),
    }


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        tasks=("sentiment", "utility"),
        n_seeds=2,
        n_pairs=4,
        n_eval_pairs=3,
        top_k=2,
        scorers=SCORERS,
        logistic_c=(0.1, 10.0),
        mlp_alpha=(1.0, 0.01),
        mlp_hidden=(4,),
        n_boot=50,
    )
    return dataclasses.replace(base, **overrides)


def _refused(
    config: Config,
    tmp_path: Path,
    examples: Mapping[str, Sequence[Example]] | None = None,
) -> None:
    execute(
        config,
        RunDir.create(config, out_root=tmp_path),
        cast(Any, None),
        _sources() if examples is None else examples,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tasks": ()}, "at least one task"),
        ({"tasks": ("emotion_happiness",)}, "binary tasks only"),
        ({"tasks": ("nope",)}, "unknown task 'nope'"),
        ({"scorers": ()}, "at least one entry in scorers"),
        ({"readouts": ()}, "at least one entry in readouts"),
        ({"layer_rules": ()}, "at least one entry in layer_rules"),
        ({"logistic_c": ()}, "at least one entry in logistic_c"),
        ({"mlp_alpha": (0.1, 0.0)}, "mlp_alpha must hold positive values"),
        ({"top_k": 0}, "top_k must be at least 1"),
        ({"token_layers": ()}, "token_layers must name at least one layer"),
        ({"batch_size": 0}, "batch_size must be at least 1, got 0"),
        (
            {"logistic_c": (1.0, 0.1)},
            "logistic_c must run from the strongest penalty",
        ),
        (
            {"mlp_alpha": (0.01, 1.0)},
            "mlp_alpha must run from the strongest penalty",
        ),
        (
            {"layer_rules": ("best", "joint")},
            "the joint rule needs top_k",
        ),
        (
            {"baselines": ("lda/last",), "scorers": ("logistic",)},
            "baseline 'lda/last' is not a cell of the run",
        ),
        ({"baselines": ("mean_diff/attention",)}, "not a valid cell"),
        (
            {"tasks": ("sentiment",), "scorers": ("ttpd",), "baselines": ()},
            "task 'sentiment' has no scorer left to read",
        ),
    ],
)
def test_a_bad_config_is_refused_before_any_model_call(
    overrides: dict[str, Any], message: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=message):
        _refused(_config(**overrides), tmp_path)


def test_a_missing_source_is_refused(tmp_path: Path) -> None:
    examples = {"sst2": _sentiment()}
    with pytest.raises(ValueError, match="no examples for source"):
        _refused(_config(), tmp_path, examples)


def test_run_refuses_a_bad_config_before_loading_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def no_load(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the model was loaded")

    monkeypatch.setattr(m2_readers, "load_model", no_load)
    config = _config(top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        run(config, RunDir.create(config, out_root=tmp_path))


def test_run_loads_the_model_in_float32(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Any, ...]] = []

    class LoadedError(Exception):
        pass

    def record(key: str, **kwargs: Any) -> None:
        calls.append((key, kwargs))
        raise LoadedError

    monkeypatch.setattr(m2_readers, "load_sources", lambda sources: {})
    monkeypatch.setattr(m2_readers, "load_model", record)
    config = _config()
    with pytest.raises(LoadedError):
        run(config, RunDir.create(config, out_root=tmp_path))
    assert calls == [("tiny", {"dtype": torch.float32})]


def test_run_loads_the_sources_of_every_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    class LoadedError(Exception):
        pass

    def record(sources: list[str]) -> None:
        calls.append(list(sources))
        raise LoadedError

    monkeypatch.setattr(m2_readers, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(m2_readers, "load_sources", record)
    config = _config(tasks=("sentiment", "truth_polarity"))
    with pytest.raises(LoadedError):
        run(config, RunDir.create(config, out_root=tmp_path))
    assert calls == [["sst2", "geometry_of_truth"]]


def _quiet_execute(
    config: Config, run_dir: RunDir, loaded: LoadedModel
) -> None:
    # A few MLP fits on the tiny model's handful of rows stop on an L-BFGS
    # line search; the warning is expected there and not under test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        execute(config, run_dir, loaded, _sources())


@pytest.fixture(scope="module")
def runs() -> dict[str, RunDir]:
    return {}


@pytest.fixture
def finished(
    runs: dict[str, RunDir],
    tiny_gpt2: LoadedModel,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> RunDir:
    if "run" not in runs:
        monkeypatch.setattr(p2_positions, "DECOY_LINE", "Hint {{question}}.\n")
        config = _config()
        run_dir = RunDir.create(config, out_root=tmp_path_factory.mktemp("m2"))
        _quiet_execute(config, run_dir, tiny_gpt2)
        runs["run"] = run_dir
    return runs["run"]


def _table(run_dir: RunDir, name: str = "table") -> list[dict[str, Any]]:
    payload = cast(dict[str, Any], run_dir.read_json(name))
    return [
        dict(zip(payload["columns"], row, strict=True))
        for row in payload["rows"]
    ]


SENTIMENT_CONDITIONS = [
    "original",
    "random",
    "against",
    "paraphrase_0",
    "paraphrase_1",
    "paraphrase_2",
]


def test_a_run_writes_the_table_the_comparisons_and_the_scores(
    finished: RunDir,
) -> None:
    names = {p.name for p in finished.path.iterdir()}
    for stem in (
        "table",
        "comparisons",
        "summary",
        "scores_sentiment",
        "scores_utility",
        "auroc_sentiment",
        "auroc_utility",
    ):
        assert f"{stem}.json" in names
    assert {"scores_sentiment.npz", "auroc_utility.npz"} <= names
    table = cast(dict[str, Any], finished.read_json("table"))
    assert table["columns"] == list(COLUMNS)
    comparisons = cast(dict[str, Any], finished.read_json("comparisons"))
    assert comparisons["columns"] == list(COMPARISON_COLUMNS)


def test_conditions_are_the_original_the_decoys_and_the_shifts(
    finished: RunDir,
) -> None:
    summary = cast(dict[str, Any], finished.read_json("summary"))
    assert summary["conditions"] == {
        "sentiment": SENTIMENT_CONDITIONS,
        "utility": ["original", "random", "against"],
    }
    assert "shift:paraphrase" in summary["skipped"]["utility"]
    assert summary["n_eval_pairs"]["sentiment"] == {"select": 3, "report": 3}


def test_every_cell_has_a_row_per_condition_and_series(
    finished: RunDir,
) -> None:
    rows = _table(finished)
    sentiment = [r for r in rows if r["task"] == "sentiment"]
    scorers = [s for s in SCORERS if s != "ttpd"]
    listed = cells(_config(), READOUTS, scorers=scorers)
    expected = {
        (c.scorer, c.readout, c.layer_rule, series, condition)
        for c in listed
        for series in (
            ("reader",)
            if c.scorer in ("random", "text")
            else ("reader", "shuffled")
        )
        for condition in SENTIMENT_CONDITIONS
    }
    got = [
        (
            r["scorer"],
            r["readout"],
            r["layer_rule"],
            r["series"],
            r["condition"],
        )
        for r in sentiment
    ]
    assert len(got) == len(set(got))
    assert set(got) == expected


def test_the_metrics_come_from_the_stored_per_seed_scores(
    finished: RunDir,
) -> None:
    arrays = finished.read_arrays("scores_sentiment")
    sidecar = cast(dict[str, Any], finished.read_json("scores_sentiment"))
    rows = {
        (
            f"{r['scorer']}/{r['readout']}/{r['layer_rule']}/{r['series']}",
            r["condition"],
        ): r
        for r in _table(finished)
        if r["task"] == "sentiment"
    }
    labels = arrays["report_labels"]
    groups = arrays["report_groups"]
    for e, entry in enumerate(sidecar["entries"]):
        for m, condition in enumerate(sidecar["conditions"]):
            row = rows[(entry, condition)]
            report = arrays["report"][e, :, m]
            pairs = group_accuracy(report, groups, labels)
            assert row["pair_accuracy"] == pytest.approx(pairs.mean())
            assert row["auroc"] == pytest.approx(auroc(report, labels).mean())
            f1 = [
                threshold_metrics(
                    arrays["select"][e, i],
                    arrays["select_labels"],
                    report[i],
                    labels,
                ).f1
                for i in range(report.shape[0])
            ]
            assert row["f1"] == pytest.approx(np.mean(f1))
            assert row["pair_lo"] <= row["pair_accuracy"] <= row["pair_hi"]
            assert 0.0 <= row["tpr_at_fpr"] <= 1.0
            assert row["n_seeds"] == 2


def test_every_cell_scores_the_same_report_items(finished: RunDir) -> None:
    arrays = finished.read_arrays("scores_sentiment")
    sidecar = cast(dict[str, Any], finished.read_json("scores_sentiment"))
    n_entries = len(sidecar["entries"])
    assert arrays["report"].shape == (n_entries, 2, 6, 6)
    assert arrays["select"].shape == (n_entries, 2, 6)
    assert np.isfinite(arrays["report"]).all()


def test_layer_rules_read_one_layer_the_top_k_or_a_joint_fit(
    finished: RunDir,
) -> None:
    sidecar = cast(dict[str, Any], finished.read_json("scores_sentiment"))
    for entry, per_seed in sidecar["layers"].items():
        rule = entry.split("/")[2]
        for layers in per_seed:
            if rule == "best":
                assert len(layers) == 1
            elif rule in ("top_k", "joint"):
                assert len(layers) == 2
                assert len(set(layers)) == 2
            else:
                assert layers == []


def test_the_text_control_ignores_the_prompt_around_the_raw_text(
    finished: RunDir,
) -> None:
    arrays = finished.read_arrays("scores_sentiment")
    sidecar = cast(dict[str, Any], finished.read_json("scores_sentiment"))
    e = sidecar["entries"].index("text/text/none/reader")
    report = arrays["report"][e]
    for m in range(1, report.shape[1]):
        np.testing.assert_array_equal(report[:, m], report[:, 0])
    assert np.ptp(report[0, 0]) > 0.0


def _ensemble(select: NDArray[Any], report: NDArray[Any]) -> NDArray[Any]:
    mean = select.mean(axis=-1, keepdims=True)
    std = select.std(axis=-1, keepdims=True)
    return ((report - mean[:, None]) / std[:, None]).mean(axis=0)


def test_comparisons_test_each_cell_against_a_baseline_on_the_ensemble(
    finished: RunDir,
) -> None:
    arrays = finished.read_arrays("scores_sentiment")
    sidecar = cast(dict[str, Any], finished.read_json("scores_sentiment"))
    labels = arrays["report_labels"]
    entries = sidecar["entries"]

    def ensemble(name: str) -> NDArray[Any]:
        e = entries.index(name)
        return _ensemble(arrays["select"][e], arrays["report"][e])

    rows = [
        r
        for r in _table(finished, "comparisons")
        if r["task"] == "sentiment"
        and r["condition"] == "against"
        and r["layer_rule"] == "best"
        and r["baseline"] == "logistic/last"
    ]
    assert {(r["scorer"], r["readout"]) for r in rows} == {
        (c.scorer, c.readout)
        for c in cells(_config(), READOUTS)
        if c.layer_rule == "best" and c.scorer in SCORERS
    } - {("logistic", "last")} - {("ttpd", r) for r in READOUTS}
    base = ensemble("logistic/last/best/reader")[2]
    groups = arrays["report_groups"]
    clusters = np.empty(labels.shape[0], dtype=np.int64)
    for g, members in enumerate(groups):
        clusters[members] = g
    for row in rows:
        cell = ensemble(f"{row['scorer']}/{row['readout']}/best/reader")[2]
        diff = paired_auroc_diff(
            cell, base, labels, clusters, n_boot=50, seed=_config().seed
        )
        assert row["auroc_diff"] == pytest.approx(diff.difference)
        assert row["auroc_lo"] == pytest.approx(diff.lower)
        assert row["delong_p"] == pytest.approx(
            delong_test(cell, base, labels).p_value
        )
    for column in ("delong", "mcnemar"):
        holm = holm_correction(np.array([r[f"{column}_p"] for r in rows]))
        np.testing.assert_allclose(
            [r[f"{column}_holm"] for r in rows], holm.adjusted
        )


def test_joint_cells_are_compared_with_the_top_k_baseline(
    finished: RunDir,
) -> None:
    rows = [
        r for r in _table(finished, "comparisons") if r["layer_rule"] == "joint"
    ]
    assert rows
    assert {r["baseline_rule"] for r in rows} == {"top_k"}
    assert {(r["scorer"], r["readout"]) for r in rows} <= {
        ("mlp", readout) for readout in READOUTS
    } | {("logistic", "attention")}


def test_a_rerun_writes_the_same_table(
    finished: RunDir,
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p2_positions, "DECOY_LINE", "Hint {{question}}.\n")
    config = _config(tasks=("utility",))
    run_dir = RunDir.create(config, out_root=tmp_path)
    _quiet_execute(config, run_dir, tiny_gpt2)
    first = [r for r in _table(finished) if r["task"] == "utility"]
    assert _table(run_dir) == first


def test_plot_draws_the_auroc_matrix_of_every_task(
    finished: RunDir, tmp_path: Path
) -> None:
    paths = plot(finished, tmp_path)
    names = {p.name for p in paths}
    assert {"auroc_sentiment.png", "auroc_utility.pdf"} <= names


def test_the_layer_rules_of_a_cell_share_its_per_layer_fits(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    config = _config(
        tasks=("utility",),
        scorers=("mlp",),
        readouts=("last",),
        layer_rules=("best", "top_k"),
        controls=(),
        decoys=(),
        baselines=(),
        top_k=1,
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    _quiet_execute(config, run_dir, tiny_gpt2)
    arrays = run_dir.read_arrays("scores_utility")
    sidecar = cast(dict[str, Any], run_dir.read_json("scores_utility"))
    best = sidecar["entries"].index("mlp/last/best/reader")
    top = sidecar["entries"].index("mlp/last/top_k/reader")
    assert (
        sidecar["layers"]["mlp/last/best/reader"]
        == (sidecar["layers"]["mlp/last/top_k/reader"])
    )
    select = arrays["select"][best]
    mean = select.mean(axis=-1)[:, None, None]
    std = select.std(axis=-1)[:, None, None]
    np.testing.assert_allclose(
        arrays["report"][top], (arrays["report"][best] - mean) / std
    )


@pytest.fixture(scope="module")
def polar_runs() -> dict[str, tuple[RunDir, dict[str, set[str]]]]:
    return {}


@pytest.fixture
def polar(
    polar_runs: dict[str, tuple[RunDir, dict[str, set[str]]]],
    tiny_gpt2: LoadedModel,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[RunDir, dict[str, set[str]]]:
    """A run of TTPD on the polarity task and on sentiment, at one layer."""
    if "run" not in polar_runs:
        subsets: dict[str, set[str]] = {}

        def recording(task: Any, variant: Any, rows: Any, **kw: Any) -> Any:
            subsets.setdefault(task.name, set()).update(
                e.meta.get("subset", "") for e in rows
            )
            return task_set(task, variant, rows, **kw)

        monkeypatch.setattr(m2_readers, "task_set", recording)
        config = _config(
            tasks=("sentiment", "truth_polarity"),
            scorers=("mean_diff", "ttpd"),
            readouts=("last", "token_score_max"),
            layer_rules=("best",),
            layers=(1,),
            controls=("shuffled",),
            decoys=(),
            shifts=(),
            baselines=("mean_diff/last",),
            n_pairs=8,
        )
        run_dir = RunDir.create(config, out_root=tmp_path_factory.mktemp("m2"))
        _quiet_execute(config, run_dir, tiny_gpt2)
        polar_runs["run"] = (run_dir, subsets)
    return polar_runs["run"]


def test_the_polarity_task_reads_only_its_four_subsets(
    polar: tuple[RunDir, dict[str, set[str]]],
) -> None:
    _, subsets = polar

    assert subsets["truth_polarity"] == set(POLARITY_TASKS["truth_polarity"])


def test_ttpd_reads_the_polarity_task_apart_from_mean_diff(
    polar: tuple[RunDir, dict[str, set[str]]],
) -> None:
    run_dir, _ = polar
    arrays = run_dir.read_arrays("scores_truth_polarity")
    sidecar = cast(dict[str, Any], run_dir.read_json("scores_truth_polarity"))
    entries = sidecar["entries"]
    for readout in ("last", "token_score_max"):
        for series in ("reader", "shuffled"):
            ttpd = arrays["select"][
                entries.index(f"ttpd/{readout}/best/{series}")
            ]
            dom = arrays["select"][
                entries.index(f"mean_diff/{readout}/best/{series}")
            ]
            for seed in range(ttpd.shape[0]):
                corr = np.corrcoef(ttpd[seed], dom[seed])[0, 1]
                # Without polarity both read the class-mean difference, so
                # their scores would be one affine map apart (|corr| = 1).
                assert abs(corr) < 1.0 - 1e-5, (readout, series, seed)


def test_ttpd_is_skipped_with_its_reason_on_a_task_without_polarity(
    polar: tuple[RunDir, dict[str, set[str]]],
) -> None:
    run_dir, _ = polar
    summary = cast(dict[str, Any], run_dir.read_json("summary"))
    rows = _table(run_dir)

    assert summary["skipped"]["sentiment"]["ttpd"] == (
        "task 'sentiment' has no polarity labels: ttpd needs affirmative "
        "and negated statements"
    )
    assert "ttpd" not in summary["skipped"]["truth_polarity"]
    scorers = {(r["task"], r["scorer"]) for r in rows}
    assert ("sentiment", "ttpd") not in scorers
    assert ("sentiment", "mean_diff") in scorers
    assert ("truth_polarity", "ttpd") in scorers


POSITIONS = ("last", "concept", "concept_mean", "mean", "max", "response")


def _truth_stimuli(loaded: LoadedModel) -> StimulusSet:
    """Stimuli whose concept, truthfulness, spans more than one token."""
    task = READING_TASKS["truth_cities"]
    chat = chat_format_for(loaded)
    paper = template_variants(task, chat, kinds=("paper",))[0][0]
    rows = [e for e in _statements() if e.meta["subset"] != "larger_than"]
    return task_set(task, paper, rows, n_pairs=4, seed=0)


def _count_texts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records every text `read_positions` sends through the model."""
    texts: list[str] = []

    def counting(loaded: LoadedModel, batch: Any, **kw: Any) -> Activations:
        texts.extend(batch)
        return collect_activations(loaded, batch, **kw)

    monkeypatch.setattr(m2_readers, "collect_activations", counting)
    return texts


def test_read_positions_derives_every_readout_from_three_passes(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(batch_size=3)
    stimuli = _truth_stimuli(tiny_arch)
    texts = _count_texts(monkeypatch)
    read = read_positions(tiny_arch, stimuli, POSITIONS, config)

    assert len(texts) == 3 * len(stimuli.texts)
    assert list(read) == list(POSITIONS)
    for position in POSITIONS:
        expected = prepare(tiny_arch, stimuli, position, config)[1]
        got = read[position]
        assert got.position == position
        assert got.layers == expected.layers
        assert got.n_stimuli == expected.n_stimuli
        assert got.data.dtype == np.float32
        for field in ("row_stimulus", "row_rank", "row_token", "row_token_id"):
            np.testing.assert_array_equal(
                getattr(got, field), getattr(expected, field)
            )
        if position in ("mean", "concept_mean"):
            # The two sums round apart by float32 steps of the row scale.
            scale = float(np.abs(expected.data).max())
            np.testing.assert_allclose(
                got.data, expected.data, rtol=1e-5, atol=1e-6 * scale
            )
        else:
            np.testing.assert_array_equal(got.data, expected.data)


def test_read_positions_runs_only_the_passes_it_needs(
    tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    stimuli = _truth_stimuli(tiny_gpt2)
    texts = _count_texts(monkeypatch)
    read = read_positions(tiny_gpt2, stimuli, ("max", "mean"), _config())

    assert list(read) == ["max", "mean"]
    assert len(texts) == len(stimuli.texts)


def test_read_positions_names_the_stimulus_without_a_span(
    tiny_gpt2: LoadedModel,
) -> None:
    stimuli = _truth_stimuli(tiny_gpt2)
    spans = list(stimuli.concept_spans)
    spans[5] = None
    broken = dataclasses.replace(stimuli, concept_spans=tuple(spans))

    with pytest.raises(
        ValueError, match=r"^stimulus 5 has no span for 'concept_mean'$"
    ):
        read_positions(
            tiny_gpt2, broken, ("concept_mean",), _config(batch_size=2)
        )


def test_read_positions_names_the_batch_of_a_span_without_tokens(
    tiny_gpt2: LoadedModel,
) -> None:
    stimuli = _truth_stimuli(tiny_gpt2)
    spans = list(stimuli.response_spans)
    end = len(stimuli.texts[5])
    spans[5] = (end + 1, end + 2)
    broken = dataclasses.replace(stimuli, response_spans=tuple(spans))

    with pytest.raises(
        ValueError,
        match=r"^in the batch that starts at stimulus 4: stimulus 1: the "
        r"span .* overlaps no token$",
    ):
        read_positions(tiny_gpt2, broken, ("mean",), _config(batch_size=2))


def test_token_layers_keep_the_token_rows_of_those_layers_only(
    tiny_gpt2: LoadedModel,
) -> None:
    stimuli = _truth_stimuli(tiny_gpt2)
    positions = ("mean", "response")

    full = read_positions(tiny_gpt2, stimuli, positions, _config())
    kept = read_positions(
        tiny_gpt2, stimuli, positions, _config(token_layers=(3, 2))
    )

    assert full["response"].layers == (1, 2, 3)
    assert kept["response"].layers == (2, 3)
    np.testing.assert_array_equal(
        kept["response"].data, full["response"].data[1:]
    )
    np.testing.assert_array_equal(
        kept["response"].row_token, full["response"].row_token
    )
    assert kept["mean"].layers == (1, 2, 3)
    np.testing.assert_array_equal(kept["mean"].data, full["mean"].data)


def test_token_layers_must_be_layers_the_run_reads(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    config = _config(token_layers=(2, 4))
    with pytest.raises(
        ValueError,
        match=r"token_layers \(2, 4\) must be among the layers read "
        r"\(1, 2, 3\)",
    ):
        execute(
            config,
            RunDir.create(config, out_root=tmp_path),
            tiny_gpt2,
            _sources(),
        )


def test_the_token_readouts_read_the_token_layers_only(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    config = _config(
        tasks=("utility",),
        scorers=("logistic",),
        readouts=("last", "token_score_max", "attention"),
        token_layers=(2, 3),
        controls=(),
        decoys=(),
        baselines=(),
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    _quiet_execute(config, run_dir, tiny_gpt2)
    sidecar = cast(dict[str, Any], run_dir.read_json("scores_utility"))

    for entry, per_seed in sidecar["layers"].items():
        readout, rule = entry.split("/")[1:3]
        for layers in per_seed:
            if readout == "last":
                assert set(layers) <= {1, 2, 3}
            elif rule == "best":
                assert layers in ([2], [3])
            else:
                assert sorted(layers) == [2, 3]


def test_a_task_releases_its_activations_before_the_next_is_read(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held: list[weakref.ref[Any]] = []
    prepare_task = m2_readers._prepare_task

    def tracking(*args: Any, **kwargs: Any) -> Any:
        assert all(ref() is None for ref in held)
        splits, skipped = prepare_task(*args, **kwargs)
        held.append(weakref.ref(splits))
        return splits, skipped

    monkeypatch.setattr(m2_readers, "_prepare_task", tracking)
    config = _config(
        scorers=("mean_diff",),
        readouts=("last",),
        layer_rules=("best",),
        controls=(),
        decoys=(),
        shifts=(),
        baselines=(),
    )
    _quiet_execute(config, RunDir.create(config, out_root=tmp_path), tiny_gpt2)

    assert len(held) == 2
