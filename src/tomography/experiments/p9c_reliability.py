"""P9c: do the reading direction and the output confidence predict errors?

The target is the model's own greedy few-shot answer, so the study asks about
errors and not about a label of the question. Every predictor is a score where
higher means more likely correct: the confidence of the output distribution,
the reading direction of a contrast (`lat`), a supervised probe of
correctness, and four controls that use no hidden state and no confidence:
the popularity of the subject, the accuracy of the relation, the length of
the question and the likelihood of the subject's name. The study consumes a
finished P9a run, whose gate `known_contrasts` only decides whether `lat` may
be the known versus unknown direction.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
from numpy.typing import NDArray

from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.intervals import write_interval_curves
from tomography.experiments.p9a_labels import load_labeled
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import write_curves
from tomography.knowledge import DEFAULT_SHOTS
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.p9.pool import LabeledItem
from tomography.p9.reading import ItemActivations
from tomography.p9.reading import check_pool
from tomography.p9.reading import collect_item_activations
from tomography.p9.reading import draw_seed
from tomography.p9.reading import fit_type_reader
from tomography.p9.reading import p9_template
from tomography.p9.risk import aurc
from tomography.p9.risk import auroc_ci
from tomography.p9.risk import auroc_diff_ci
from tomography.p9.risk import group_rate
from tomography.p9.risk import orient_on_train
from tomography.p9.risk import risk_band
from tomography.p9.risk import stack_scores
from tomography.p9.signals import item_signals
from tomography.p9.unknowns import TypePool
from tomography.p9.unknowns import type_pools
from tomography.reading.directions import Method
from tomography.reading.metrics import auroc
from tomography.reading.metrics import best_layer
from tomography.reading.reader import RepReader

PREDICTORS: tuple[str, ...] = (
    "first_token_prob",
    "neg_entropy",
    "answer_logprob",
    "mean_token_logprob",
    "consistency",
    "p_true",
    "lat",
    "probe",
    "popularity",
    "template",
    "length",
    "name_logprob",
)
CONFIDENCE: tuple[str, ...] = (
    "first_token_prob",
    "neg_entropy",
    "mean_token_logprob",
    "consistency",
    "p_true",
)
RISK_PANELS: dict[str, tuple[str, ...]] = {
    "outputs": (
        "first_token_prob",
        "neg_entropy",
        "answer_logprob",
        "mean_token_logprob",
        "consistency",
        "p_true",
        "lat",
        "probe",
    ),
    "controls": (
        "lat",
        "probe",
        "popularity",
        "template",
        "length",
        "name_logprob",
    ),
}
COVERAGE_GRID: tuple[int, ...] = tuple(range(5, 101, 5))
ORIENTED: tuple[str, ...] = ("popularity", "length", "name_logprob")
SPLITS: tuple[str, ...] = ("train", "val", "test")
GATE_REASON = (
    "the p9a gate admits no known versus unknown contrast for this model"
)
TABLE_COLUMNS = [
    "dataset",
    "predictor",
    "n",
    "n_error",
    "n_filled",
    "auroc",
    "lo",
    "hi",
    "aurc",
]
INCREMENTAL_COLUMNS = [
    "dataset",
    "features",
    "auroc",
    "diff_vs_confidence",
    "lo",
    "hi",
]


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """P9c settings: predicting the errors of a model.

    Attributes:
        labels_run: Path of a finished P9a run of the same model.
        template: How a question is rendered for the `lat` reader and the
            probe, one of `tomography.p9.reading.TEMPLATE_NAMES`.
        n_shots: Demonstrations of the "fewshot" prompt, and of the prompt
            the signals are scored at.
        datasets: The question sources whose errors are predicted.
        lat_type: The contrast whose direction is the `lat` predictor, any
            name `tomography.p9.unknowns.type_pools` builds.
        lat_method: The direction finder of that contrast.
        layers: Layers to read; None reads 1..L.
        layer: Layer of `lat` and the probe; None chooses it on the
            validation items of the first usable dataset.
        n_seeds: Draws of the pairs of the per-layer `lat` curve.
        n_pairs: Pairs per draw of that curve.
        min_pairs: Fewest train items on each side of the `lat` contrast, and
            fewest matched train pairs (pairs that never cross two strata);
            each split half of its train items must also hold 2 matched
            pairs, see `tomography.p9.reading.check_pool`.
        n_boot: Bootstrap resamples of every interval.
        min_probe: Fewest train items of a dataset that fit its probe.
        bos: BOS policy of the activations; None uses the model's default.
            The signals keep the default of the labeling run.
        batch_size: Prompts per forward pass.
    """

    experiment: str = "p9c"
    model: str = "gpt2"
    labels_run: str
    template: str = "fewshot"
    n_shots: int = 15
    datasets: tuple[str, ...] = ("popqa", "triviaqa", "nq_open")
    lat_type: str = "obscure_real"
    lat_method: Method = "logistic"
    layers: tuple[int, ...] | None = None
    layer: int | None = None
    n_seeds: int = 5
    n_pairs: int = 64
    min_pairs: int = 16
    n_boot: int = 1000
    min_probe: int = 50
    bos: BosMode | None = None
    batch_size: int = 8


@dataclass(frozen=True)
class _Dataset:
    """The answered questions of one source with their outcomes and splits."""

    name: str
    items: list[LabeledItem]
    correct: NDArray[np.int64]
    masks: dict[str, NDArray[np.bool_]]


def resolve_lat(
    items: Sequence[LabeledItem],
    lat_type: str,
    *,
    known_contrasts: bool,
    min_pairs: int,
) -> tuple[TypePool | None, str | None]:
    """Finds the contrast of the `lat` predictor, or says why there is none.

    Only the train pairs are used, so the validation and test sides are not
    checked.

    Args:
        items: The labeled questions.
        lat_type: The contrast the user asked for.
        known_contrasts: The P9a gate.
        min_pairs: Fewest train items on each side, and fewest matched train
            pairs (pairs that never cross two strata); each split half of the
            train items must also hold 2 matched pairs.

    Returns:
        The contrast and None, or None and the reason: the `check_pool`
        reason (fewer than `min_pairs` train items on one side, fewer than
        `min_pairs` matched train pairs, or fewer than 2 matched train pairs
        in one split half), the gate for a contrast that needs a known side,
        or the missing name.
    """
    available = type_pools(items, knowledge=known_contrasts)
    if lat_type in available:
        reason = check_pool(
            available[lat_type], min_pairs=min_pairs, min_eval=0
        )
        return (available[lat_type], None) if reason is None else (None, reason)
    if not known_contrasts and lat_type in type_pools(items):
        return None, GATE_REASON
    return None, f"no contrast named {lat_type!r} on these labels"


def _split_datasets(
    config: Config, items: Sequence[LabeledItem]
) -> tuple[list[_Dataset], dict[str, str]]:
    repeated = sorted(
        {n for n in config.datasets if config.datasets.count(n) > 1}
    )
    if repeated:
        raise ValueError(f"duplicate datasets in the config: {repeated}")
    usable: list[_Dataset] = []
    skipped: dict[str, str] = {}
    for name in config.datasets:
        rows = [
            i for i in items if i.group == name and i.label.basis == "behavior"
        ]
        if not rows:
            skipped[name] = "no behavior items"
            continue
        correct = np.array(
            [i.label.greedy_correct for i in rows], dtype=np.int64
        )
        split = np.array([i.example.split for i in rows])
        masks = {s: split == s for s in SPLITS}
        reason = next(
            (
                f"the {s} split does not hold both outcomes"
                for s in SPLITS
                if len(np.unique(correct[masks[s]])) < 2
            ),
            None,
        )
        if reason is None:
            usable.append(_Dataset(name, rows, correct, masks))
        else:
            skipped[name] = reason
    if not usable:
        reasons = "; ".join(f"{n}: {r}" for n, r in skipped.items())
        raise ValueError(
            f"no dataset holds both outcomes in every split: {reasons}"
        )
    return usable, skipped


def _choose_layer(
    dataset: _Dataset,
    h: NDArray[np.float32],
    layers: Sequence[int],
    reader: RepReader | None,
    probe: RepReader,
) -> tuple[int, str]:
    val = dataset.masks["val"]
    if reader is not None:
        scores, source = reader.score(h), "validation"
    else:
        scores, source = probe.score(h), "probe_validation"
    curve = auroc(scores[:, val].astype(np.float64), dataset.correct[val])
    return best_layer(curve, layers), source


def _dataset_signals(
    config: Config,
    loaded: LoadedModel,
    dataset: _Dataset,
    h: NDArray[np.float32],
    k: int,
    layer: int,
    reader: RepReader | None,
) -> tuple[dict[str, NDArray[np.float64]], str | None]:
    train = dataset.masks["train"]
    signals = item_signals(
        loaded,
        dataset.items,
        shots=DEFAULT_SHOTS[: config.n_shots],
        batch_size=config.batch_size,
    )
    if reader is not None:
        signals["lat"] = reader.score(h)[k].astype(np.float64)
    probe_skipped = None
    if int(train.sum()) >= config.min_probe:
        probe = RepReader.fit(
            "logistic",
            h[k : k + 1, train],
            (layer,),
            labels=dataset.correct[train],
        )
        signals["probe"] = probe.score(h[k : k + 1])[0].astype(np.float64)
    else:
        probe_skipped = f"fewer than {config.min_probe} train items"
    relation = [
        r if isinstance(r := i.example.meta.get("relation"), str) else ""
        for i in dataset.items
    ]
    if any(relation):
        signals["template"] = group_rate(
            [r for r, m in zip(relation, train, strict=True) if m],
            dataset.correct[train],
            relation,
        )
    return signals, probe_skipped


def _fill_and_orient(
    signals: dict[str, NDArray[np.float64]], dataset: _Dataset
) -> tuple[dict[str, NDArray[np.float64]], dict[str, int]]:
    train = dataset.masks["train"]
    clean: dict[str, NDArray[np.float64]] = {}
    n_filled: dict[str, int] = {}
    for name in PREDICTORS:
        if name not in signals:
            continue
        column = signals[name]
        finite = np.isfinite(column)
        if not (finite & train).any():
            continue
        n_filled[name] = int((~finite).sum())
        column = np.where(finite, column, column[finite & train].min())
        if name in ORIENTED:
            column = orient_on_train(
                column[train], dataset.correct[train], column
            )
        clean[name] = column
    return clean, n_filled


def _incremental_rows(
    config: Config,
    dataset: _Dataset,
    signals: dict[str, NDArray[np.float64]],
) -> list[list[object]]:
    """Rows of what `lat` and the probe add to the output confidence.

    The readers were fitted on the train items, so their train scores are
    in-sample and a stacker that learned its weights there would trust them
    too much. The stacker is fitted on the validation rows, where every
    column is out of sample, and the rows are scored on the test items.
    """
    fit, test = dataset.masks["val"], dataset.masks["test"]
    correct = dataset.correct
    columns = [c for c in CONFIDENCE if c in signals]
    base_train = np.stack([signals[c][fit] for c in columns], axis=1)
    base_test = np.stack([signals[c][test] for c in columns], axis=1)
    base = stack_scores(base_train, correct[fit], base_test, seed=config.seed)
    incremental: list[list[object]] = [
        [
            dataset.name,
            "confidence",
            float(auroc(base, correct[test])),
            0.0,
            0.0,
            0.0,
        ]
    ]
    for extra in ("lat", "probe"):
        if extra not in signals:
            continue
        plus = stack_scores(
            np.column_stack([base_train, signals[extra][fit]]),
            correct[fit],
            np.column_stack([base_test, signals[extra][test]]),
            seed=config.seed,
        )
        diff, lo, hi = auroc_diff_ci(
            plus,
            base,
            correct[test],
            n_boot=config.n_boot,
            seed=config.seed,
        )
        incremental.append(
            [
                dataset.name,
                f"confidence+{extra}",
                float(auroc(plus, correct[test])),
                diff,
                lo,
                hi,
            ]
        )
    return incremental


def _write_dataset(
    config: Config,
    run_dir: RunDir,
    dataset: _Dataset,
    signals: dict[str, NDArray[np.float64]],
    n_filled: dict[str, int],
) -> list[list[object]]:
    test = dataset.masks["test"]
    correct = dataset.correct
    names = list(signals)
    table: list[list[object]] = []
    bands: dict[str, tuple[Any, Any, Any]] = {}
    for name in names:
        scores = signals[name][test]
        point, lo, hi = auroc_ci(
            scores, correct[test], n_boot=config.n_boot, seed=config.seed
        )
        table.append(
            [
                dataset.name,
                name,
                int(test.sum()),
                int((correct[test] == 0).sum()),
                n_filled[name],
                point,
                lo,
                hi,
                aurc(scores, correct[test], seed=config.seed),
            ]
        )
        bands[name] = risk_band(
            scores,
            correct[test],
            COVERAGE_GRID,
            n_boot=config.n_boot,
            seed=config.seed,
        )
    for panel, members in RISK_PANELS.items():
        series = [n for n in members if n in bands]
        write_interval_curves(
            run_dir,
            [float(g) for g in COVERAGE_GRID],
            np.stack([bands[n][0] for n in series]),
            np.stack([bands[n][1] for n in series]),
            np.stack([bands[n][2] for n in series]),
            series=series,
            stem=f"riskcov_{dataset.name}_{panel}",
            xlabel="Coverage (%)",
            ylabel="Risk (error rate among answered)",
        )
    run_dir.write_arrays(
        f"signals_{dataset.name}",
        **{n: signals[n][test] for n in names},
        correct=correct[test],
    )
    return table


def _write_layer_curves(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    dataset: _Dataset,
    acts: ItemActivations,
    h: NDArray[np.float32],
    lat_pool: TypePool | None,
    probe: RepReader,
) -> None:
    test = dataset.masks["test"]
    probe_curve = auroc(
        probe.score(h)[:, test].astype(np.float64), dataset.correct[test]
    )
    series = ["probe"]
    curves = [np.tile(probe_curve, (config.n_seeds, 1))]
    if lat_pool is not None:
        readers = [
            fit_type_reader(
                acts,
                lat_pool,
                method=config.lat_method,
                n_pairs=config.n_pairs,
                seed=draw_seed(config.seed, s),
            )
            for s in range(config.n_seeds)
        ]
        series.insert(0, "lat")
        curves.insert(
            0,
            np.stack(
                [
                    auroc(
                        r.score(h)[:, test].astype(np.float64),
                        dataset.correct[test],
                    )
                    for r in readers
                ]
            ),
        )
    write_curves(
        run_dir,
        np.stack(curves),
        acts.layers,
        series=series,
        ylabel="AUROC (own errors)",
        chance=0.5,
        n_layers=loaded.n_layers,
        seed=config.seed,
    )


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    items: Sequence[LabeledItem],
    *,
    known_contrasts: bool,
) -> None:
    """Predicts the errors of a model on its labeled questions.

    A dataset whose train, validation or test split does not hold both
    outcomes is skipped. The signals are scored at the prompt the labels were
    made with, and a signal without a value is filled with the minimum finite
    train value (a missing answer is the least confident case). Writes
    `table` (the test AUROC, interval and AURC of every predictor),
    `table_incremental` (what `lat` and the probe add to the output
    confidence, as a paired AUROC difference; the stacker is fitted on the
    validation items because the readers were fitted on the train ones),
    `riskcov_<dataset>_outputs` and `riskcov_<dataset>_controls` (the
    risk-coverage curves of the panels), `signals_<dataset>` (the filled and
    oriented test signals and the outcomes), `curves` (the test AUROC per
    layer of `lat` and the probe on the first usable dataset; the probe is
    fitted there even when it is below `min_probe`) and `summary`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        items: The labeled questions of the P9a run.
        known_contrasts: The P9a gate: whether `lat` may be a contrast that
            needs a known side.

    Raises:
        ValueError: If the template is unknown, `config.datasets` repeats a
            name, no dataset holds both outcomes in every split, or
            `config.layer` is not among the layers.
    """
    template = p9_template(config.template, DEFAULT_SHOTS[: config.n_shots])
    lat_pool, lat_skipped = resolve_lat(
        items,
        config.lat_type,
        known_contrasts=known_contrasts,
        min_pairs=config.min_pairs,
    )
    datasets, skipped = _split_datasets(config, items)
    layers = (
        tuple(range(1, loaded.n_layers + 1))
        if config.layers is None
        else config.layers
    )
    if config.layer is not None and config.layer not in layers:
        raise ValueError(
            f"layer {config.layer} is not among the layers read {layers}"
        )
    pooled = (
        [] if lat_pool is None else [*lat_pool.positive, *lat_pool.negative]
    )
    union = {
        i.example.uid: i
        for i in (*(i for d in datasets for i in d.items), *pooled)
    }
    acts = collect_item_activations(
        loaded,
        list(union.values()),
        template,
        chat_format_for(loaded),
        layers=layers,
        bos=config.bos,
        batch_size=config.batch_size,
    )
    reader = (
        None
        if lat_pool is None
        else fit_type_reader(
            acts,
            lat_pool,
            method=config.lat_method,
            n_pairs=None,
            seed=config.seed,
        )
    )
    hidden = {d.name: acts.data[:, acts.rows(d.items)] for d in datasets}
    first = datasets[0]
    train = first.masks["train"]
    probe_all = RepReader.fit(
        "logistic",
        hidden[first.name][:, train],
        acts.layers,
        labels=first.correct[train],
    )
    if config.layer is None:
        layer, source = _choose_layer(
            first, hidden[first.name], acts.layers, reader, probe_all
        )
    else:
        layer, source = config.layer, "config"
    k = acts.layers.index(layer)
    table: list[list[object]] = []
    incremental: list[list[object]] = []
    probe_skipped: dict[str, str] = {}
    for dataset in datasets:
        h = hidden[dataset.name]
        raw, reason = _dataset_signals(
            config, loaded, dataset, h, k, layer, reader
        )
        if reason is not None:
            probe_skipped[dataset.name] = reason
        signals, n_filled = _fill_and_orient(raw, dataset)
        table += _write_dataset(config, run_dir, dataset, signals, n_filled)
        incremental += _incremental_rows(config, dataset, signals)
    _write_layer_curves(
        config,
        run_dir,
        loaded,
        first,
        acts,
        hidden[first.name],
        lat_pool,
        probe_all,
    )
    run_dir.write_json("table", {"columns": TABLE_COLUMNS, "rows": table})
    run_dir.write_json(
        "table_incremental",
        {"columns": INCREMENTAL_COLUMNS, "rows": incremental},
    )
    run_dir.write_json(
        "summary",
        {
            "layer": layer,
            "layer_source": source,
            "template": config.template,
            "datasets": [d.name for d in datasets],
            "skipped": skipped,
            "probe_skipped": probe_skipped,
            "lat_type": config.lat_type,
            "lat_method": config.lat_method,
            "lat_skipped": lat_skipped,
            "known_contrasts": known_contrasts,
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the labeled pool and the model, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.

    Raises:
        ValueError: If `config.labels_run` is not a finished run, it labeled
            another model than `config.model`, or it was made with another
            number of shots than `config.n_shots`.
    """
    items, summary = load_labeled(config.labels_run, model=config.model)
    labeled_shots = summary["recipe"]["n_shots"]
    if labeled_shots != config.n_shots:
        raise ValueError(
            f"the labels run was made with n_shots {labeled_shots} but "
            f"config.n_shots is {config.n_shots}; the signals must be scored "
            "at the prompt the labels were made with"
        )
    execute(
        config,
        run_dir,
        load_model(config.model),
        items,
        known_contrasts=bool(summary["known_contrasts"]),
    )


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the layer curves and every risk-coverage file of a P9c run.

    Args:
        run_dir: A finished P9c run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of `curves` and of the two risk-coverage
        files of each dataset in `summary.datasets`.
    """
    summary = cast(dict[str, Any], run_dir.read_json("summary"))
    paths = plot_curves_file(run_dir, out_dir)
    for dataset in summary["datasets"]:
        for panel in RISK_PANELS:
            paths += plot_curves_file(
                run_dir, out_dir, stem=f"riskcov_{dataset}_{panel}"
            )
    return paths
