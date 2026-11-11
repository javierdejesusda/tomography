"""Tests for M3's flat reader, the flat cell matrix under M3's protocol."""

from collections.abc import Iterator
import contextlib
import dataclasses
import json
from pathlib import Path
from typing import Any
from typing import cast
import warnings

import numpy as np
from numpy.typing import NDArray
import pytest
from sklearn.exceptions import ConvergenceWarning

from tomography.datasets.records import Example
from tomography.experiments import m2_readers
from tomography.experiments import m3_matrix
from tomography.experiments import p2_positions
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.m3_matrix import SEGMENTS
from tomography.experiments.m3_matrix import prompt_cells
from tomography.experiments.m3_matrix import select_cells
from tomography.experiments.sweep import shuffle_within_groups
from tomography.model import LoadedModel
from tomography.reading.flat import fit_flat
from tomography.reading.probes import RestartedFitWarning
from tomography.reading.probes import StalledFitWarning
from tomography.reading.stats import holm_correction
from tomography.reading.stats import paired_auroc_diff

RESULTS = Path(__file__).resolve().parents[1] / "results" / "m3"
LAYERS = (1, 2, 3)
CONDITIONS = ["original", "random", "against"]
FLAT_POOLINGS = ("cellattn", "cellmax", "flat")
CELL_SETS = ("prompt", "stimulus")
FLAT_READOUTS = tuple(f"flat_{cell_set}" for cell_set in CELL_SETS)
STIMULUS, FINAL, DECOY = (
    SEGMENTS.index(s) for s in ("stimulus", "final", "decoy")
)


@pytest.fixture(autouse=True)
def _short_decoy_line(monkeypatch: pytest.MonkeyPatch) -> None:
    # The paper's decoy line is too long for the tiny models' positions.
    monkeypatch.setattr(
        p2_positions, "DECOY_LINE", "Hint: {noun} is {{question}}.\n"
    )


def _sources() -> dict[str, list[Example]]:
    """Utility pairs whose report texts are longer than every training text.

    The report stimuli run past the widths the training stimuli set, so
    the flat reader truncates them.
    """
    words = {
        1: ("sun", "gift", "joy", "fun"),
        0: ("rain", "loss", "pain", "fear"),
    }
    tails = {"train": "", "val": " too", "test": " and so on and on"}
    return {
        "ethics_utility": [
            Example(
                f"ethics-{split}-{i}:{suffix}",
                f"a {words[label][i % 4]} day {i}{tail}",
                label,
                split,
                {"pair": f"ethics-{split}-{i}"},
            )
            for split, tail in tails.items()
            for i in range(8)
            for suffix, label in (("a", 1), ("b", 0))
        ]
    }


def _config(**overrides: Any) -> m3_matrix.Config:
    base = m3_matrix.Config(
        model="tiny",
        tasks=("utility",),
        n_seeds=2,
        n_pairs=4,
        n_eval_pairs=3,
        top_k=2,
        layers=LAYERS,
        cell_layers=(1, 3),
        upper_layers=(3,),
        cell_sets=CELL_SETS,
        logistic_c=(0.1, 10.0),
        n_boot=10,
    )
    return dataclasses.replace(base, **overrides)


@contextlib.contextmanager
def _quiet_fits() -> Iterator[None]:
    """Ignores the warnings of the cell fits that M3 counts."""
    with warnings.catch_warnings():
        for category in m3_matrix._COUNTED:
            warnings.simplefilter("ignore", category)
        yield


@dataclasses.dataclass
class _Run:
    """A finished run with what its flat fits and M2 saw.

    Attributes:
        config: The config of the run.
        run_dir: The finished run.
        items: The items M2 rendered for the task.
        flat_fits: The keyword arguments of every `fit_flat` call.
    """

    config: m3_matrix.Config
    run_dir: RunDir
    items: Any
    flat_fits: list[dict[str, Any]]


def _execute(
    config: m3_matrix.Config, out_root: Path, loaded: LoadedModel
) -> _Run:
    """Runs `config`, recording the `fit_flat` calls and M2's items."""
    calls: list[dict[str, Any]] = []
    captured: list[Any] = []
    fit, prepare = m3_matrix.fit_flat, m2_readers._prepare_task

    def recording(acts: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return fit(acts, **kwargs)

    def capturing(*args: Any, **kwargs: Any) -> Any:
        splits, skipped = prepare(*args, **kwargs)
        captured.append(splits)
        return splits, skipped

    run_dir = RunDir.create(config, out_root=out_root)
    with pytest.MonkeyPatch.context() as patch, _quiet_fits():
        patch.setattr(m3_matrix, "fit_flat", recording)
        patch.setattr(m2_readers, "_prepare_task", capturing)
        m3_matrix.execute(config, run_dir, loaded, _sources())
    return _Run(config, run_dir, captured[0], calls)


@pytest.fixture(scope="module")
def runs() -> dict[str, _Run]:
    return {}


@pytest.fixture
def flat(
    runs: dict[str, _Run],
    tiny_gpt2: LoadedModel,
    tmp_path_factory: pytest.TempPathFactory,
) -> _Run:
    """A run of every pooling, flat included, on two cell sets."""
    if "flat" not in runs:
        runs["flat"] = _execute(
            _config(poolings=FLAT_POOLINGS),
            tmp_path_factory.mktemp("m3"),
            tiny_gpt2,
        )
    return runs["flat"]


@pytest.fixture
def plain(
    runs: dict[str, _Run],
    tiny_gpt2: LoadedModel,
    tmp_path_factory: pytest.TempPathFactory,
) -> _Run:
    """The same run with the default poolings, so without the flat reader."""
    if "plain" not in runs:
        runs["plain"] = _execute(
            _config(), tmp_path_factory.mktemp("m3"), tiny_gpt2
        )
    return runs["plain"]


def _table(run_dir: RunDir, name: str) -> list[dict[str, Any]]:
    payload = cast(dict[str, Any], run_dir.read_json(name))
    return [
        dict(zip(payload["columns"], row, strict=True))
        for row in payload["rows"]
    ]


def _entries(run_dir: RunDir) -> list[str]:
    sidecar = cast(dict[str, Any], run_dir.read_json("scores_utility"))
    return cast(list[str], sidecar["entries"])


def test_flat_is_a_known_pooling_but_not_a_default() -> None:
    config = m3_matrix.Config.from_dict({"poolings": ["flat", "cellmax"]})

    assert m3_matrix.POOLINGS == ("cellattn", "cellmax", "flat")
    assert m3_matrix.Config().poolings == ("cellattn", "cellmax")
    assert config.poolings == ("flat", "cellmax")


def test_every_committed_m3_run_keeps_its_config_hash() -> None:
    paths = sorted(RESULTS.glob("*/*/config.json"))

    assert len(paths) >= 20
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        config = m3_matrix.Config.from_dict(data)
        assert config.config_hash() == path.parent.name, path


def test_a_run_without_flat_writes_what_it_wrote_before(plain: _Run) -> None:
    run_dir = plain.run_dir
    names = {p.name for p in run_dir.path.iterdir()}
    summary = cast(dict[str, Any], run_dir.read_json("summary"))
    sidecar = cast(dict[str, Any], run_dir.read_json("cellheads_utility"))

    assert "truncation.json" not in names
    assert set(summary) == {
        "tasks",
        "conditions",
        "skipped",
        "n_eval_pairs",
        "fits",
        "families",
        "diagnostics",
    }
    assert set(sidecar["axes"]) == {
        "values",
        "queries",
        "biases",
        "offsets",
        "signs",
        "c",
    }
    assert not any("flat" in name for name in _entries(run_dir))
    assert plain.flat_fits == []


def _without_seconds(fits: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {k: v for k, v in stats.items() if k != "seconds"}
        for name, stats in fits.items()
    }


def test_the_flat_reader_leaves_every_other_output_unchanged(
    flat: _Run, plain: _Run
) -> None:
    def other(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in rows if not r["readout"].startswith("flat_")]

    assert other(_table(flat.run_dir, "table")) == _table(
        plain.run_dir, "table"
    )
    assert _table(flat.run_dir, "diagnostics") == _table(
        plain.run_dir, "diagnostics"
    )
    entries, plain_entries = _entries(flat.run_dir), _entries(plain.run_dir)
    assert [e for e in entries if "/flat_" not in e] == plain_entries
    scores = flat.run_dir.read_arrays("scores_utility")
    plain_scores = plain.run_dir.read_arrays("scores_utility")
    for name in ("select", "report", "entry_select_labels"):
        kept = [entries.index(e) for e in plain_entries]
        np.testing.assert_array_equal(scores[name][kept], plain_scores[name])
    heads = flat.run_dir.read_arrays("cellheads_utility")
    plain_heads = plain.run_dir.read_arrays("cellheads_utility")
    for key, array in plain_heads.items():
        np.testing.assert_array_equal(heads[key], array)
    summary = cast(dict[str, Any], flat.run_dir.read_json("summary"))
    plain_summary = cast(dict[str, Any], plain.run_dir.read_json("summary"))
    fits = _without_seconds(summary["fits"]["utility"])
    assert {k: v for k, v in fits.items() if "/flat_" not in k} == (
        _without_seconds(plain_summary["fits"]["utility"])
    )


def test_each_flat_reader_has_a_row_per_condition_and_series(
    flat: _Run,
) -> None:
    got = [
        (
            r["scorer"],
            r["readout"],
            r["layer_rule"],
            r["series"],
            r["condition"],
        )
        for r in _table(flat.run_dir, "table")
        if r["readout"].startswith("flat_")
    ]

    assert len(got) == len(set(got))
    assert set(got) == {
        ("logistic", readout, "joint", series, condition)
        for readout in FLAT_READOUTS
        for series in ("reader", "shuffled")
        for condition in CONDITIONS
    }
    sidecar = cast(dict[str, Any], flat.run_dir.read_json("scores_utility"))
    layers = sidecar["layers"]
    for readout in FLAT_READOUTS:
        assert layers[f"logistic/{readout}/joint/reader"] == [[1, 3]] * 2


def _ensemble(select: NDArray[Any], report: NDArray[Any]) -> NDArray[Any]:
    mean = select.mean(axis=-1, keepdims=True)
    std = select.std(axis=-1, keepdims=True)
    return cast(
        NDArray[Any], ((report - mean[:, None]) / std[:, None]).mean(axis=0)
    )


def test_flat_readers_are_compared_with_each_baseline_at_top_k_and_best(
    flat: _Run,
) -> None:
    config = flat.config
    rows = [
        r
        for r in _table(flat.run_dir, "comparisons")
        if r["layer_rule"] == "joint"
    ]
    got = sorted(
        (r["condition"], r["baseline"], r["baseline_rule"], r["readout"])
        for r in rows
        if r["readout"].startswith("flat_")
    )
    assert got == sorted(
        (condition, baseline, rule, readout)
        for condition in CONDITIONS
        for baseline in config.baselines
        for rule in ("top_k", "best")
        for readout in FLAT_READOUTS
    )

    arrays = flat.run_dir.read_arrays("scores_utility")
    entries = _entries(flat.run_dir)
    labels = arrays["report_labels"]
    clusters = np.empty(labels.shape[0], dtype=np.int64)
    for g, members in enumerate(arrays["report_groups"]):
        clusters[members] = g
    against = CONDITIONS.index("against")

    def ensemble(name: str) -> NDArray[Any]:
        e = entries.index(name)
        return _ensemble(arrays["select"][e], arrays["report"][e])[against]

    family = [
        r
        for r in rows
        if r["condition"] == "against"
        and r["baseline"] == "logistic/stimulus_token"
        and r["baseline_rule"] == "best"
    ]
    # Every joint cell of both sets shares the family, the flat ones too.
    assert len(family) == len(FLAT_POOLINGS) * len(CELL_SETS)
    base = ensemble("logistic/stimulus_token/best/reader")
    for row in family:
        if row["readout"].startswith("flat_"):
            cell = ensemble(f"logistic/{row['readout']}/joint/reader")
            diff = paired_auroc_diff(
                cell,
                base,
                labels,
                clusters,
                n_boot=config.n_boot,
                seed=config.seed,
            )
            assert row["auroc_diff"] == pytest.approx(diff.difference)
    holm = holm_correction(np.array([r["delong_p"] for r in family]))
    np.testing.assert_allclose(
        [r["delong_holm"] for r in family], holm.adjusted
    )


def test_the_shuffled_flat_readers_fit_and_tune_on_shuffled_labels(
    flat: _Run,
) -> None:
    config = flat.config
    train, select = flat.items.train_items, flat.items.select_items
    shuffles = [
        (
            shuffle_within_groups(
                train[i].labels,
                train[i].groups,
                seed=derive_seed(config.seed, "shuffled", "utility", i),
            ),
            shuffle_within_groups(
                select.labels,
                select.groups,
                seed=derive_seed(config.seed, "shuffled-select", "utility", i),
            ),
        )
        for i in range(config.n_seeds)
    ]
    # Per draw and cell set: the reader series, then the shuffled one, each
    # at every grid value.
    expected = [
        labels
        for i, (shuffled, _) in enumerate(shuffles)
        for _ in CELL_SETS
        for labels in (train[i].labels, shuffled)
        for _ in config.logistic_c
    ]
    assert len(flat.flat_fits) == len(expected)
    for fit, labels in zip(flat.flat_fits, expected, strict=True):
        assert np.array_equal(fit["labels"], labels)
        assert fit["tol"] == config.cell_tol
        assert fit["n_segments"] == len(SEGMENTS)
    assert [fit["c"] for fit in flat.flat_fits] == list(config.logistic_c) * (
        2 * len(CELL_SETS) * config.n_seeds
    )
    arrays = flat.run_dir.read_arrays("scores_utility")
    entries = _entries(flat.run_dir)
    for readout in FLAT_READOUTS:
        e = entries.index(f"logistic/{readout}/joint/shuffled")
        for i, (_, select_labels) in enumerate(shuffles):
            np.testing.assert_array_equal(
                arrays["entry_select_labels"][e, i], select_labels
            )


def _runs(
    loaded: LoadedModel,
    stimuli: Any,
    config: m3_matrix.Config,
    cell_set: str,
    *,
    decoy: bool,
) -> NDArray[np.int64]:
    """The tokens of each text in each segment of a cell set."""
    cells, segments = prompt_cells(
        loaded,
        stimuli,
        config.cell_layers,
        bos=config.bos,
        batch_size=config.batch_size,
        decoy=decoy,
    )
    kept = m3_matrix._in_set(segments, cast(Any, cell_set))
    key = cells.row_stimulus[kept] * len(SEGMENTS) + segments[kept]
    counts = np.bincount(key, minlength=cells.n_stimuli * len(SEGMENTS))
    return counts.reshape(cells.n_stimuli, len(SEGMENTS))


def test_the_kept_flat_heads_hold_the_grid_value_bias_and_widths(
    flat: _Run, tiny_gpt2: LoadedModel
) -> None:
    config = flat.config
    heads = flat.run_dir.read_arrays("cellheads_utility")
    sidecar = cast(dict[str, Any], flat.run_dir.read_json("cellheads_utility"))

    for readout in FLAT_READOUTS:
        key = f"logistic_{readout}_reader"
        assert {k for k in heads if k.startswith(f"{key}_")} == {
            f"{key}_c",
            f"{key}_bias",
            f"{key}_widths",
        }
        assert heads[f"{key}_c"].shape == (config.n_seeds,)
        assert set(heads[f"{key}_c"].tolist()) <= set(config.logistic_c)
        assert heads[f"{key}_bias"].shape == (config.n_seeds,)
        cell_set = readout.removeprefix("flat_")
        for i, train in enumerate(flat.items.train_items):
            runs = _runs(tiny_gpt2, train, config, cell_set, decoy=False)
            np.testing.assert_array_equal(
                heads[f"{key}_widths"][i], runs.max(axis=0)
            )
        assert sidecar["entries"][key] == {
            "entry": f"logistic/{readout}/joint/reader",
            "layers": [1, 3],
        }
    # Training uses the paper template, so the decoy line has no slots.
    assert not heads["logistic_flat_prompt_reader_widths"][:, DECOY].any()
    assert sidecar["axes"]["bias"] == ["seed"]
    assert sidecar["axes"]["widths"] == ["seed", "segment"]
    assert sidecar["segments"] == list(SEGMENTS)


def test_truncation_counts_the_cells_each_flat_reader_drops(
    flat: _Run, tiny_gpt2: LoadedModel
) -> None:
    config = flat.config
    payload = cast(dict[str, Any], flat.run_dir.read_json("truncation"))
    summary = cast(dict[str, Any], flat.run_dir.read_json("summary"))
    heads = flat.run_dir.read_arrays("cellheads_utility")
    rows = _table(flat.run_dir, "truncation")
    n_layers = len(config.cell_layers)

    assert payload["columns"] == list(m3_matrix.TRUNCATION_COLUMNS)
    assert "no slots" in summary["truncation"]
    assert sorted(
        (r["readout"], r["series"], r["condition"]) for r in rows
    ) == sorted(
        (readout, series, condition)
        for readout in FLAT_READOUTS
        for series in ("reader", "shuffled")
        for condition in CONDITIONS
    )
    truncated = 0.0
    for row in rows:
        assert row["task"] == "utility"
        assert row["scorer"] == "logistic"
        cell_set = row["readout"].removeprefix("flat_")
        stimuli = flat.items.report_items[row["condition"]]
        runs = _runs(
            tiny_gpt2,
            stimuli,
            config,
            cell_set,
            decoy=row["condition"] in config.decoys,
        )
        widths = heads[f"logistic_{row['readout']}_reader_widths"]
        over = np.maximum(runs[None] - widths[:, None], 0)
        assert row["cells"] == pytest.approx(runs.sum(axis=1).mean() * n_layers)
        for s, segment in enumerate(SEGMENTS):
            assert row[segment] == pytest.approx(
                over[..., s].mean() * n_layers
            ), (row, segment)
        lost = (over * (widths[:, None] > 0)).sum(axis=-1) > 0
        assert row["truncated"] == pytest.approx(lost.mean())
        if row["condition"] != "original" and cell_set == "prompt":
            assert row["decoy"] == pytest.approx(
                runs[:, DECOY].mean() * n_layers
            )
            assert row["decoy"] > 0
        truncated += row["stimulus"]
    # The report stimuli run past the training ones.
    assert truncated > 0


def test_streamed_flat_scores_equal_whole_set_scores(
    flat: _Run, tiny_gpt2: LoadedModel
) -> None:
    train = flat.items.train_items[0]
    cells, segments = prompt_cells(
        tiny_gpt2, train, (1, 3), bos=None, batch_size=4, decoy=False
    )
    kept = m3_matrix._in_set(segments, "prompt")
    with _quiet_fits():
        reader = fit_flat(
            select_cells(cells, segments, "prompt", (1, 3)),
            segments=segments[kept],
            labels=train.labels,
            n_segments=len(SEGMENTS),
        )
    report = flat.items.report_items["against"]

    for batch_size in (1, 3):
        config = _config(batch_size=batch_size)
        streamed = m3_matrix.stream_scores(
            tiny_gpt2, report, [("prompt", reader)], config, decoy=True
        )
        # The whole set is read with the same forward passes.
        whole, whole_segments = prompt_cells(
            tiny_gpt2,
            report,
            (1, 3),
            bos=None,
            batch_size=batch_size,
            decoy=True,
        )
        acts = select_cells(whole, whole_segments, "prompt", (1, 3))
        codes = whole_segments[m3_matrix._in_set(whole_segments, "prompt")]
        np.testing.assert_array_equal(
            streamed.scores[0], reader.score(acts, codes)[0]
        )
        np.testing.assert_array_equal(
            streamed.dropped[0], reader.dropped(acts, codes)
        )
        assert streamed.dropped[0][:, DECOY].sum() > 0


def test_the_summary_counts_flat_fits_seconds_and_warnings(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = m3_matrix.fit_flat

    def planting(acts: Any, **kwargs: Any) -> Any:
        with _quiet_fits():
            reader = fit(acts, **kwargs)
        for category in (ConvergenceWarning, StalledFitWarning):
            warnings.warn(category.__name__, category, stacklevel=2)
        return reader

    monkeypatch.setattr(m3_matrix, "fit_flat", planting)
    config = _config(cell_sets=("prompt",), poolings=("flat",), decoys=())
    run_dir = RunDir.create(config, out_root=tmp_path)
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("default")
        m3_matrix.execute(config, run_dir, tiny_gpt2, _sources())

    summary = cast(dict[str, Any], run_dir.read_json("summary"))
    fits = summary["fits"]["utility"]
    n = config.n_seeds * len(config.logistic_c)
    assert set(fits) == {
        f"logistic/flat_prompt/joint/{series}"
        for series in ("reader", "shuffled")
    }
    for stats in fits.values():
        assert stats["fits"] == n
        assert stats["seconds"] > 0.0
        assert stats["convergence_warnings"] == n
        assert stats["stalled_fits"] == n
        assert stats["restarted_fits"] == 0
    for category in (ConvergenceWarning, StalledFitWarning):
        assert [w for w in shown if str(w.message) == category.__name__]
    assert not [w for w in shown if w.category is RestartedFitWarning]
