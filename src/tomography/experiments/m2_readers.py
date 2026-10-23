"""M2: readers compared on the same features, readouts and layer rules.

A factorial of scorer by readout by layer rule. Every scorer (LAT,
difference of means, tuned logistic regression, LDA, TTPD and an MLP with
matched tuning) reads every readout (the last token, the concept token, the
mean of the concept tokens, the mean of the stimulus tokens, their
feature-wise maximum, and the maximum over the stimulus tokens of a reader's
per-token score), and the logistic scorer also reads the stimulus tokens
through learned attention. Each cell is reported at two layer rules, the
best validation layer and the standardized mean of the top k validation
layers, and the MLP and attention cells also at a joint fit over those k
layers. So a gain of pooling is never credited to nonlinearity. The readers
are trained on the paper template and scored on the report items of the
original prompts, of prompts with P2's misleading sentence and of held-out
template wordings, next to a text-only classifier, a random direction and a
refit on shuffled labels.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import cast
from typing import get_args

import numpy as np
from numpy.typing import NDArray
import torch

from tomography.activations import Activations
from tomography.activations import PositionStrategy
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.matrices import plot_matrix_file
from tomography.experiments.matrices import write_matrix
from tomography.experiments.p2_positions import Decoy
from tomography.experiments.p2_positions import _mark_word
from tomography.experiments.p2_positions import misleading_template
from tomography.experiments.p2_positions import with_decoys
from tomography.experiments.sweep import Prepared
from tomography.experiments.sweep import ReadingConfig
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import shuffle_within_groups
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import ReadingTask
from tomography.experiments.variants import Variant
from tomography.experiments.variants import VariantKind
from tomography.experiments.variants import eval_pairs
from tomography.experiments.variants import eval_seed
from tomography.experiments.variants import expand_tasks
from tomography.experiments.variants import load_sources
from tomography.experiments.variants import prepare
from tomography.experiments.variants import target_of
from tomography.experiments.variants import task_examples
from tomography.experiments.variants import task_set
from tomography.experiments.variants import task_sources
from tomography.experiments.variants import task_splits
from tomography.experiments.variants import template_variants
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.p9.risk import text_scores
from tomography.reading.directions import Method
from tomography.reading.directions import fit_direction
from tomography.reading.metrics import auroc
from tomography.reading.metrics import bootstrap_ci
from tomography.reading.metrics import group_accuracy
from tomography.reading.probes import fit_attention
from tomography.reading.probes import fit_lda
from tomography.reading.probes import fit_mlp
from tomography.reading.probes import fit_ttpd
from tomography.reading.reader import RepReader
from tomography.reading.reader import _layer_seed
from tomography.reading.reader import orient
from tomography.reading.stats import LayerChoice
from tomography.reading.stats import best_validation_layer
from tomography.reading.stats import delong_test
from tomography.reading.stats import holm_correction
from tomography.reading.stats import mcnemar_test
from tomography.reading.stats import paired_accuracy_diff
from tomography.reading.stats import paired_auroc_diff
from tomography.reading.stats import threshold_metrics
from tomography.reading.stats import top_k_layers
from tomography.reading.stats import tpr_at_fpr

Scorer = Literal["pca_diff", "mean_diff", "logistic", "lda", "ttpd", "mlp"]
Readout = Literal[
    "last",
    "concept",
    "concept_mean",
    "mean",
    "max",
    "token_score_max",
    "attention",
]
LayerRule = Literal["best", "top_k", "joint"]
Control = Literal["text", "random", "shuffled"]
SCORERS: tuple[Scorer, ...] = get_args(Scorer)
READOUTS: tuple[Readout, ...] = get_args(Readout)
LAYER_RULES: tuple[LayerRule, ...] = get_args(LayerRule)
CONTROLS: tuple[Control, ...] = get_args(Control)
POSITION_OF: Mapping[Readout, PositionStrategy] = {
    "last": "last",
    "concept": "concept",
    "concept_mean": "concept_mean",
    "mean": "mean",
    "max": "max",
    "token_score_max": "response",
    "attention": "response",
}
CONCEPT_READOUTS: tuple[Readout, ...] = ("concept", "concept_mean")
COLUMNS: tuple[str, ...] = (
    "task",
    "condition",
    "scorer",
    "readout",
    "layer_rule",
    "series",
    "layers",
    "pair_accuracy",
    "pair_lo",
    "pair_hi",
    "auroc",
    "auroc_lo",
    "auroc_hi",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "tpr_at_fpr",
    "n_seeds",
)
COMPARISON_COLUMNS: tuple[str, ...] = (
    "task",
    "condition",
    "layer_rule",
    "baseline",
    "baseline_rule",
    "scorer",
    "readout",
    "auroc_diff",
    "auroc_lo",
    "auroc_hi",
    "pair_diff",
    "pair_lo",
    "pair_hi",
    "delong_p",
    "delong_holm",
    "mcnemar_p",
    "mcnemar_holm",
    "a_only",
    "b_only",
)


@dataclass(frozen=True, kw_only=True)
class Config(ReadingConfig):
    """M2 settings: the factorial reader comparison.

    The shared reading defaults apply except for the experiment id and the
    number of evaluation pairs. `method` and `position` are not read: the
    scorers and readouts replace them.

    Attributes:
        tasks: Keys of `READING_TASKS`; the task must be binary.
        scorers: The scorers to compare, from `SCORERS`. `ttpd` is not a
            default: without polarity labels, which no reading task has, it
            scores as `mean_diff` does (both are the class-mean difference
            of the centered rows).
        readouts: The readouts every scorer reads, from `READOUTS`; the
            concept readouts are skipped for a task without a concept slot
            or word, and `attention` is read by the logistic scorer only.
        layer_rules: From `LAYER_RULES`; `joint` applies to the MLP on the
            one-row readouts and to the attention readout, and needs
            `top_k`, the rule of its baseline.
        top_k: Layers averaged by `top_k` and fitted jointly by `joint`, at
            most the number of layers read.
        decoys: The kinds of P2's misleading sentence to evaluate.
        shifts: Template variant kinds evaluated as held-out wordings; a
            kind that does not apply to a task is skipped.
        controls: The controls to add, from `CONTROLS`.
        logistic_c: The grid of the logistic scorer's inverse L2 strength,
            for the one-row, per-token and attention readouts alike, in
            increasing order, so a tie keeps the strongest penalty.
        mlp_alpha: The grid of the MLP's L2 strength, in decreasing order
            for the same reason, as many values as `logistic_c` so both
            scorers get the same tuning budget.
        mlp_hidden: Hidden units of the MLP.
        baselines: The cells every other cell is compared with, as
            `<scorer>/<readout>`.
        n_boot: Resamples of the paired bootstraps.
        max_fpr: The validation false positive rate of `tpr_at_fpr`.
    """

    experiment: str = "m2"
    model: str = "gpt2"
    n_eval_pairs: int | None = 256
    tasks: tuple[str, ...] = ("sentiment", "truth", "utility")
    scorers: tuple[Scorer, ...] = (
        "pca_diff",
        "mean_diff",
        "logistic",
        "lda",
        "mlp",
    )
    readouts: tuple[Readout, ...] = READOUTS
    layer_rules: tuple[LayerRule, ...] = LAYER_RULES
    top_k: int = 5
    decoys: tuple[Decoy, ...] = ("random", "against")
    shifts: tuple[VariantKind, ...] = ("paraphrase",)
    controls: tuple[Control, ...] = CONTROLS
    logistic_c: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    mlp_alpha: tuple[float, ...] = (0.1, 0.01, 0.001, 0.0001)
    mlp_hidden: tuple[int, ...] = (64,)
    baselines: tuple[str, ...] = ("logistic/last", "mean_diff/last")
    n_boot: int = 2000
    max_fpr: float = 0.01


class Cell(NamedTuple):
    """One cell of the factorial, or a control.

    Attributes:
        scorer: A scorer, or the control `random` or `text`.
        readout: A readout, or `text` for the text-only classifier.
        layer_rule: A layer rule, or `none` for the text-only classifier.
    """

    scorer: str
    readout: str
    layer_rule: str

    @property
    def name(self) -> str:
        """`<scorer>/<readout>/<layer_rule>`."""
        return "/".join(self)


def applies(scorer: str, readout: str, layer_rule: str) -> bool:
    """Whether a scorer reads a readout under a layer rule.

    The attention readout is a logistic value head with learned pooling,
    so only the logistic scorer reads it. A joint fit over several layers
    exists for the MLP and the attention reader; the per-token maximum has
    none.

    Args:
        scorer: A scorer, or `random`.
        readout: A readout.
        layer_rule: A layer rule.

    Returns:
        True when the cell can be fitted.
    """
    if readout == "attention":
        return scorer == "logistic"
    if layer_rule == "joint":
        return scorer == "mlp" and readout != "token_score_max"
    return True


def cells(config: Config, readouts: Sequence[str]) -> list[Cell]:
    """Lists the fitted cells and the controls of a run, in table order.

    Args:
        config: The experiment settings.
        readouts: The readouts the task can read.

    Returns:
        The scorer cells, then the random direction at every non-joint rule
        of every readout but attention, then the text-only classifier, the
        controls only when `config.controls` asks for them.
    """
    rows = [
        Cell(scorer, readout, rule)
        for scorer in config.scorers
        for readout in readouts
        for rule in config.layer_rules
        if applies(scorer, readout, rule)
    ]
    if "random" in config.controls:
        rows += [
            Cell("random", readout, rule)
            for readout in readouts
            for rule in config.layer_rules
            if readout != "attention" and rule != "joint"
        ]
    if "text" in config.controls:
        rows.append(Cell("text", "text", "none"))
    return rows


def token_score_max(
    scores: NDArray[np.floating[Any]], acts: Activations
) -> NDArray[np.float64]:
    """Takes the maximum of per-token scores over each stimulus's tokens.

    Args:
        scores: Scores of the token rows of `acts`, shape (K, n_rows).
        acts: The token rows, such as those of the `response` position.

    Returns:
        Scores of shape (K, n_stimuli).

    Raises:
        ValueError: If the scores do not have one column per row, or a
            stimulus has no rows.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != acts.row_stimulus.shape[0]:
        raise ValueError(
            f"expected scores of shape (K, {acts.row_stimulus.shape[0]}), "
            f"got {values.shape}"
        )
    counts = np.bincount(acts.row_stimulus, minlength=acts.n_stimuli)
    if not counts.all():
        raise ValueError(f"stimulus {int(np.argmin(counts))} has no rows")
    pooled = np.full((values.shape[0], acts.n_stimuli), -np.inf)
    for k in range(values.shape[0]):
        np.maximum.at(pooled[k], acts.row_stimulus, values[k])
    return pooled


def tune_per_layer(
    metrics: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Picks, per layer, the grid value with the best validation metric.

    Args:
        metrics: Validation metric of each grid value and layer, shape
            (G, K).

    Returns:
        The index of the first best grid value of each layer, shape (K,);
        list the grid from the strongest penalty so ties keep it.
    """
    return np.asarray(np.argmax(metrics, axis=0), dtype=np.int64)


def read_top_k(
    val_metric: NDArray[np.float64],
    layers: Sequence[int],
    val_scores: NDArray[np.float64],
    test_scores: NDArray[np.float64],
    *,
    k: int,
) -> LayerChoice:
    """Applies `top_k_layers` to the layers whose select scores vary.

    A layer whose scorer gives every select item the same score, such as an
    MLP its penalty flattened, cannot be standardized, so it is left out
    and fewer than k layers are read when fewer vary. When none varies, the
    best layer is read as `best_validation_layer` reads it.

    Args:
        val_metric: Validation metric of each layer, shape (K,).
        layers: The layer index of each entry of `val_metric`.
        val_scores: Validation scores per layer, shape (K, n_val).
        test_scores: Test scores per layer, shape (K, n_test).
        k: The most layers averaged, at least 1.

    Returns:
        The layers read, best first, with their validation and test scores.
    """
    varies = np.ptp(val_scores, axis=1) > 0.0
    if not varies.any():
        return best_validation_layer(
            val_metric, layers, val_scores, test_scores
        )
    kept = np.flatnonzero(varies)
    return top_k_layers(
        val_metric[kept],
        [layers[int(i)] for i in kept],
        val_scores[kept],
        test_scores[kept],
        k=min(k, kept.size),
    )


ScoreFn = Callable[[Activations], NDArray[np.float64]]


@dataclass(frozen=True)
class _Splits:
    """The prepared items of one task.

    Attributes:
        train: Per position, one prepared training set per seed.
        select: Per position, the select items of the original prompts.
        report: Per position, the report items of each condition.
        conditions: The condition names, `original` first.
        texts: The raw text of every example, by uid.
    """

    train: dict[PositionStrategy, list[Prepared]]
    select: dict[PositionStrategy, Prepared]
    report: dict[PositionStrategy, dict[str, Prepared]]
    conditions: list[str]
    texts: dict[str, str]


@dataclass(frozen=True)
class _Choice:
    """The scores one seed of a cell gives after its layer rule.

    Attributes:
        layers: The layers read, best first.
        select: Scores of the select items, shape (n_select,).
        report: Scores of the report items per condition, (M, n_report).
    """

    layers: tuple[int, ...]
    select: NDArray[np.float64]
    report: NDArray[np.float64]


def _grid(config: Config, scorer: str) -> tuple[float | None, ...]:
    if scorer == "logistic":
        return config.logistic_c
    if scorer == "mlp":
        return config.mlp_alpha
    return (None,)


def _subset(act: Activations, at: Sequence[int]) -> Activations:
    index = list(at)
    return dataclasses.replace(
        act,
        data=act.data[index],
        layers=tuple(act.layers[i] for i in index),
    )


def _row_scorer(
    config: Config,
    scorer: str,
    h: NDArray[np.float32],
    layers: tuple[int, ...],
    labels: NDArray[np.int64],
    pairs: NDArray[np.int64],
    groups: NDArray[np.int64],
    hyper: float | None,
    seed: int,
    joint: bool,
) -> Callable[[NDArray[np.float32]], NDArray[np.float64]]:
    """Fits a scorer on one row per stimulus (or per token) at each layer."""
    if scorer == "mlp":
        mlp = fit_mlp(
            h,
            layers,
            labels=labels,
            hidden_sizes=config.mlp_hidden,
            alpha=cast(float, hyper),
            seed=seed,
            joint=joint,
        )
        return lambda data: mlp.score(data).astype(np.float64)
    reader: RepReader
    if scorer == "logistic":
        fits = [
            fit_direction(
                "logistic",
                h[k],
                labels=labels,
                preprocess=config.preprocess,
                seed=_layer_seed(seed, k),
                c=cast(float, hyper),
            )
            for k in range(h.shape[0])
        ]
        reader = orient(
            RepReader.from_arrays(
                "logistic",
                layers,
                np.stack([fit.direction for fit in fits]),
                np.stack([fit.mean for fit in fits]),
            ),
            h,
            labels=labels,
            pairs=pairs,
            groups=groups,
        )
    elif scorer == "lda":
        reader = fit_lda(h, layers, labels=labels, pairs=pairs, groups=groups)
    elif scorer == "ttpd":
        reader = fit_ttpd(h, layers, labels=labels, pairs=pairs, groups=groups)
    else:
        reader = RepReader.fit(
            cast(Method, scorer),
            h,
            layers,
            labels=labels,
            pairs=pairs,
            groups=groups,
            preprocess=config.preprocess,
            seed=seed,
        )
    return lambda data: reader.score(data).astype(np.float64)


def _fit(
    config: Config,
    scorer: str,
    readout: str,
    train: Prepared,
    labels: NDArray[np.int64],
    hyper: float | None,
    seed: int,
    joint: bool,
) -> ScoreFn:
    """Fits one scorer on a readout and returns its stimulus scorer."""
    stimuli, act = train
    if readout == "attention":
        attention = fit_attention(
            act, labels=labels, c=cast(float, hyper), joint=joint
        )
        return lambda acts: attention.score(acts).astype(np.float64)
    if readout == "token_score_max":
        score = _row_scorer(
            config,
            scorer,
            act.data,
            act.layers,
            act.row_values(labels),
            act.row_groups(stimuli.pairs, ragged="truncate"),
            act.row_groups(stimuli.groups, ragged="truncate"),
            hyper,
            seed,
            joint,
        )
        return lambda acts: token_score_max(score(acts.data), acts)
    rows = _row_scorer(
        config,
        scorer,
        act.data,
        act.layers,
        labels,
        stimuli.pairs,
        stimuli.groups,
        hyper,
        seed,
        joint,
    )
    return lambda acts: rows(acts.data)


def _evaluate(
    fn: ScoreFn,
    select: Prepared,
    report: Sequence[Prepared],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Scores the select and report items and the select pair accuracy."""
    stimuli, act = select
    scores = fn(act)
    metric = group_accuracy(scores, stimuli.groups, stimuli.labels)
    return scores, np.stack([fn(a) for _, a in report]), metric


def _seed_choices(
    config: Config,
    scorer: str,
    readout: str,
    rules: Sequence[str],
    train: Prepared,
    select: Prepared,
    report: Sequence[Prepared],
    labels: NDArray[np.int64],
    seed: int,
) -> list[_Choice]:
    """Fits one seed per layer, tunes it, then applies each layer rule.

    The rules share the tuned per-layer fits, so two rules of a cell differ
    only in how they read the layers.
    """
    grid = _grid(config, scorer)
    tried = [
        _evaluate(
            _fit(config, scorer, readout, train, labels, g, seed, False),
            select,
            report,
        )
        for g in grid
    ]
    best = tune_per_layer(np.stack([metric for _, _, metric in tried]))
    at = np.arange(best.shape[0])
    val = np.stack([s for s, _, _ in tried])[best, at]
    test = np.stack([r for _, r, _ in tried])[best, :, at].transpose(1, 0, 2)
    metric = np.stack([m for _, _, m in tried])[best, at]
    layers = train[1].layers
    k = min(config.top_k, len(layers))
    choices = []
    for rule in rules:
        if rule == "joint":
            choices.append(
                _joint_choice(
                    config,
                    scorer,
                    readout,
                    train,
                    select,
                    report,
                    labels,
                    seed,
                    metric,
                    k,
                )
            )
            continue
        read = [
            best_validation_layer(metric, layers, val, test[m])
            if rule == "best"
            else read_top_k(metric, layers, val, test[m], k=k)
            for m in range(test.shape[0])
        ]
        choices.append(
            _Choice(
                tuple(int(layer) for layer in read[0].layers),
                read[0].val_scores,
                np.stack([choice.test_scores for choice in read]),
            )
        )
    return choices


def _joint_choice(
    config: Config,
    scorer: str,
    readout: str,
    train: Prepared,
    select: Prepared,
    report: Sequence[Prepared],
    labels: NDArray[np.int64],
    seed: int,
    metric: NDArray[np.float64],
    k: int,
) -> _Choice:
    """Fits one reader over the top k validation layers, tuned on select."""
    at = [int(i) for i in np.argsort(-metric, kind="stable")[:k]]
    stimuli, act = train
    sub_train = (stimuli, _subset(act, at))
    sub_select = (select[0], _subset(select[1], at))
    sub_report = [(s, _subset(a, at)) for s, a in report]
    tried = [
        _evaluate(
            _fit(
                config,
                scorer,
                readout,
                sub_train,
                labels,
                g,
                seed,
                True,
            ),
            sub_select,
            sub_report,
        )
        for g in _grid(config, scorer)
    ]
    chosen = int(np.argmax([float(m[0]) for _, _, m in tried]))
    val, test, _ = tried[chosen]
    return _Choice(sub_train[1].layers, val[0], test[:, 0])


def _clusters(groups: NDArray[np.int64], n: int) -> NDArray[np.int64]:
    """Gives each stimulus the index of its group; loose ones their own."""
    ids = np.arange(groups.shape[0], groups.shape[0] + n, dtype=np.int64)
    for g, members in enumerate(groups):
        ids[members[members >= 0]] = g
    return ids


class _Entry(NamedTuple):
    cell: Cell
    series: str
    choices: list[_Choice]


def _text_choices(splits: _Splits) -> list[_Choice]:
    """Scores the raw texts with the text-only classifier, one fit per seed.

    The raw text of an item is the same in every condition, so the report
    scores repeat across conditions.
    """
    position = next(iter(splits.train))
    select = splits.select[position][0]
    report = splits.report[position]["original"][0]
    items = [*select.uids, *report.uids]
    rows = []
    for stimuli, _ in splits.train[position]:
        scores = text_scores(
            [splits.texts[uid] for uid in stimuli.uids],
            stimuli.labels,
            [splits.texts[uid] for uid in items],
            analyzer="word",
        )
        rows.append(
            _Choice(
                (),
                scores[: len(select.uids)],
                np.tile(
                    scores[len(select.uids) :], (len(splits.conditions), 1)
                ),
            )
        )
    return rows


def _cell_entries(
    config: Config,
    task: ReadingTask,
    splits: _Splits,
    scorer: str,
    readout: str,
    rules: Sequence[str],
) -> list[_Entry]:
    """Fits every seed of a scorer on a readout, and its shuffled refit.

    Returns one entry per series and layer rule; the fits of a seed are
    shared by the rules and seeded by the scorer and readout alone.
    """
    position = POSITION_OF[cast(Readout, readout)]
    select = splits.select[position]
    report = [splits.report[position][c] for c in splits.conditions]
    series = ["reader"]
    if "shuffled" in config.controls and scorer != "random":
        series.append("shuffled")
    entries = []
    for name in series:
        per_seed = []
        for i, train in enumerate(splits.train[position]):
            stimuli = train[0]
            labels = (
                stimuli.labels
                if name == "reader"
                else shuffle_within_groups(
                    stimuli.labels,
                    stimuli.groups,
                    seed=derive_seed(config.seed, "shuffled", task.name, i),
                )
            )
            seed = derive_seed(
                config.seed, "fit", task.name, scorer, readout, i
            )
            per_seed.append(
                _seed_choices(
                    config,
                    scorer,
                    readout,
                    rules,
                    train,
                    select,
                    report,
                    labels,
                    seed,
                )
            )
        entries += [
            _Entry(Cell(scorer, readout, rule), name, [c[r] for c in per_seed])
            for r, rule in enumerate(rules)
        ]
    return entries


def _metric_rows(
    config: Config,
    task: ReadingTask,
    splits: _Splits,
    entry: _Entry,
) -> list[list[Any]]:
    """Builds the table rows of one entry, one per condition."""
    position = next(iter(splits.report))
    select = splits.select[position][0]
    rows = []
    for m, condition in enumerate(splits.conditions):
        items = splits.report[position][condition][0]
        pair = np.array(
            [
                float(group_accuracy(c.report[m], items.groups, items.labels))
                for c in entry.choices
            ]
        )
        area = np.array(
            [float(auroc(c.report[m], items.labels)) for c in entry.choices]
        )
        thresholded = np.array(
            [
                threshold_metrics(
                    c.select, select.labels, c.report[m], items.labels
                )[1:]
                for c in entry.choices
            ]
        )
        detection = np.array(
            [
                tpr_at_fpr(
                    c.select,
                    select.labels,
                    c.report[m],
                    items.labels,
                    max_fpr=config.max_fpr,
                ).true_positive_rate
                for c in entry.choices
            ]
        )
        pair_lo, pair_hi = bootstrap_ci(pair, seed=config.seed)
        auroc_lo, auroc_hi = bootstrap_ci(area, seed=config.seed)
        rows.append(
            [
                task.name,
                condition,
                *entry.cell,
                entry.series,
                [list(c.layers) for c in entry.choices],
                float(pair.mean()),
                float(pair_lo),
                float(pair_hi),
                float(area.mean()),
                float(auroc_lo),
                float(auroc_hi),
                *(float(v) for v in thresholded.mean(axis=0)),
                float(detection.mean()),
                len(entry.choices),
            ]
        )
    return rows


def _ensemble(choices: Sequence[_Choice]) -> NDArray[np.float64]:
    """Averages the select-standardized report scores over the seeds.

    Each seed's scores are centered and scaled with the mean and standard
    deviation of its select scores, so seeds weigh alike.
    """
    standardized = []
    for choice in choices:
        std = choice.select.std()
        scale = std if std > 0.0 else 1.0
        standardized.append((choice.report - choice.select.mean()) / scale)
    return np.mean(standardized, axis=0)


def _comparison_rows(
    config: Config,
    task: ReadingTask,
    splits: _Splits,
    entries: Sequence[_Entry],
) -> list[list[Any]]:
    """Compares every reader cell with each baseline on the seed ensemble.

    A family is one task, condition, layer rule and baseline: Holm's
    correction runs over its DeLong p values and, separately, its McNemar p
    values.
    """
    position = next(iter(splits.report))
    readers = {
        e.cell: _ensemble(e.choices)
        for e in entries
        if e.series == "reader" and e.cell.scorer in SCORERS
    }
    rows: list[list[Any]] = []
    for baseline in config.baselines:
        scorer, readout = baseline.split("/")
        for rule in config.layer_rules:
            base_rule = rule if applies(scorer, readout, rule) else "top_k"
            base = Cell(scorer, readout, base_rule)
            if base not in readers:
                continue
            family = [c for c in readers if c.layer_rule == rule and c != base]
            for m, condition in enumerate(splits.conditions):
                items = splits.report[position][condition][0]
                clusters = _clusters(items.groups, items.labels.shape[0])
                block = []
                for cell in family:
                    a = readers[cell][m]
                    b = readers[base][m]
                    area = paired_auroc_diff(
                        a,
                        b,
                        items.labels,
                        clusters,
                        n_boot=config.n_boot,
                        seed=config.seed,
                    )
                    wins_a = _group_wins(a, items)
                    wins_b = _group_wins(b, items)
                    pair = paired_accuracy_diff(
                        wins_a,
                        wins_b,
                        np.arange(wins_a.shape[0]),
                        n_boot=config.n_boot,
                        seed=config.seed,
                    )
                    delong = delong_test(a, b, items.labels)
                    mcnemar = mcnemar_test(wins_a == 1.0, wins_b == 1.0)
                    block.append(
                        [
                            task.name,
                            condition,
                            rule,
                            baseline,
                            base_rule,
                            cell.scorer,
                            cell.readout,
                            area.difference,
                            area.lower,
                            area.upper,
                            pair.difference,
                            pair.lower,
                            pair.upper,
                            delong.p_value,
                            None,
                            mcnemar.p_value,
                            None,
                            mcnemar.a_only,
                            mcnemar.b_only,
                        ]
                    )
                if not block:
                    continue
                for test in ("delong", "mcnemar"):
                    raw = COMPARISON_COLUMNS.index(f"{test}_p")
                    holm = holm_correction(
                        np.array([row[raw] for row in block])
                    )
                    for row, adjusted in zip(block, holm.adjusted, strict=True):
                        row[raw + 1] = float(adjusted)
                rows += block
    return rows


def _group_wins(
    scores: NDArray[np.float64], items: StimulusSet
) -> NDArray[np.float64]:
    """The credit of each group: 1 when its label-1 member tops it alone."""
    return np.array(
        [
            float(group_accuracy(scores, items.groups[g : g + 1], items.labels))
            for g in range(items.groups.shape[0])
        ]
    )


def _readable(
    task: ReadingTask, readouts: Sequence[Readout]
) -> tuple[list[Readout], dict[str, str]]:
    """Drops the concept readouts of a task that has no concept token."""
    has_concept = task.concept is not None or task.word is not None
    kept = [r for r in readouts if r not in CONCEPT_READOUTS or has_concept]
    reason = f"task {task.name!r} has no concept slot or word"
    return kept, {r: reason for r in readouts if r not in kept}


def _prepare_task(
    config: Config,
    loaded: LoadedModel,
    task: ReadingTask,
    readouts: Sequence[Readout],
    examples: Mapping[str, Sequence[Example]],
) -> tuple[_Splits, dict[str, str]]:
    """Renders and reads the train, select and report items of a task."""
    chat = chat_format_for(loaded)
    variants, skipped = template_variants(
        task, chat, kinds=("paper", *config.shifts)
    )
    paper, shifted = variants[0], [v for v in variants[1:] if v.evaluable]
    biased = dataclasses.replace(
        paper,
        name="misleading",
        template=misleading_template(task.template, task.noun),
    )
    positions = list(dict.fromkeys(POSITION_OF[r] for r in readouts))
    splits = task_splits(task, examples, seed=config.seed)

    def read(
        variant: Variant, rows: Sequence[Example], **kw: Any
    ) -> StimulusSet:
        return _mark_word(task_set(task, variant, rows, **kw), task)

    train_sets = [
        read(
            paper,
            splits.train,
            n_pairs=config.n_pairs,
            seed=derive_seed(config.seed, "train", i),
        )
        for i in range(config.n_seeds)
    ]
    cases: list[tuple[str, Variant, list[Example]]] = [
        ("original", paper, list(splits.report))
    ]
    cases += [
        (decoy, biased, with_decoys(splits.report, decoy, seed=config.seed))
        for decoy in config.decoys
    ]
    cases += [(v.name, v, list(splits.report)) for v in shifted]
    select_set = read(
        paper,
        splits.select,
        n_pairs=eval_pairs(task, splits.select, config.n_eval_pairs),
        seed=eval_seed(task, "select", config.seed),
    )
    n_report = eval_pairs(task, splits.report, config.n_eval_pairs)
    report_sets = {
        name: read(
            variant,
            rows,
            n_pairs=n_report,
            seed=eval_seed(task, "report", config.seed),
        )
        for name, variant, rows in cases
    }
    texts = {
        e.uid: e.text
        for e in (
            *task_examples(task, examples),
            *task_examples(target_of(task), examples),
        )
    }
    return (
        _Splits(
            train={
                p: [prepare(loaded, s, p, config) for s in train_sets]
                for p in positions
            },
            select={
                p: prepare(loaded, select_set, p, config) for p in positions
            },
            report={
                p: {
                    name: prepare(loaded, s, p, config)
                    for name, s in report_sets.items()
                }
                for p in positions
            },
            conditions=list(report_sets),
            texts=texts,
        ),
        {f"shift:{kind}": reason for kind, reason in skipped.items()},
    )


def _write_task(
    run_dir: RunDir,
    task: ReadingTask,
    splits: _Splits,
    entries: Sequence[_Entry],
    rows: Sequence[Sequence[Any]],
) -> None:
    """Writes the per-seed scores of every entry and the AUROC matrix."""
    position = next(iter(splits.report))
    select = splits.select[position][0]
    report = splits.report[position]["original"][0]
    names = [f"{e.cell.name}/{e.series}" for e in entries]
    run_dir.write_arrays(
        f"scores_{task.name}",
        select=np.stack([[c.select for c in e.choices] for e in entries]),
        report=np.stack([[c.report for c in e.choices] for e in entries]),
        select_labels=select.labels,
        select_groups=select.groups,
        report_labels=report.labels,
        report_groups=report.groups,
    )
    run_dir.write_json(
        f"scores_{task.name}",
        {
            "entries": names,
            "conditions": splits.conditions,
            "axes": {
                "select": ["entry", "seed", "item"],
                "report": ["entry", "seed", "condition", "item"],
            },
            "layers": {
                name: [list(c.layers) for c in e.choices]
                for name, e in zip(names, entries, strict=True)
            },
        },
    )
    readers = [e for e in entries if e.series == "reader"]
    at = COLUMNS.index("auroc")
    value = {(r[2], r[3], r[4], r[5], r[1]): r[at] for r in rows}
    write_matrix(
        run_dir,
        f"auroc_{task.name}",
        np.array(
            [
                [value[(*e.cell, "reader", c)] for c in splits.conditions]
                for e in readers
            ],
            dtype=np.float64,
        ),
        rows=[e.cell.name for e in readers],
        cols=splits.conditions,
        title=f"{task.name}: report AUROC of each cell (mean over seeds)",
    )


def _check(config: Config) -> tuple[str, ...]:
    """Checks what the config alone settles and expands its tasks."""
    tasks = expand_tasks(config.tasks)
    if not tasks:
        raise ValueError("m2 needs at least one task")
    for field, values in (
        ("scorers", config.scorers),
        ("readouts", config.readouts),
        ("layer_rules", config.layer_rules),
        ("logistic_c", config.logistic_c),
        ("mlp_alpha", config.mlp_alpha),
    ):
        if not values:
            raise ValueError(f"m2 needs at least one entry in {field}")
    for field, grid in (
        ("logistic_c", config.logistic_c),
        ("mlp_alpha", config.mlp_alpha),
    ):
        if not all(value > 0.0 for value in grid):
            raise ValueError(f"{field} must hold positive values, got {grid}")
    for field, grid, ordered in (
        ("logistic_c", config.logistic_c, np.diff(config.logistic_c) > 0),
        ("mlp_alpha", config.mlp_alpha, np.diff(config.mlp_alpha) < 0),
    ):
        if not ordered.all():
            raise ValueError(
                f"{field} must run from the strongest penalty to the "
                f"weakest, without repeats, got {grid}"
            )
    if "joint" in config.layer_rules and "top_k" not in config.layer_rules:
        raise ValueError(
            "the joint rule needs top_k: its cells are compared with the "
            "top_k baseline"
        )
    if config.top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {config.top_k}")
    for baseline in config.baselines:
        scorer, _, readout = baseline.partition("/")
        if scorer not in config.scorers or readout not in config.readouts:
            raise ValueError(
                f"baseline {baseline!r} is not a cell of the run: give "
                "<scorer>/<readout> from the configured scorers and readouts"
            )
        if not applies(scorer, readout, "best"):
            raise ValueError(f"baseline {baseline!r} is not a valid cell")
    for name in tasks:
        task = READING_TASKS[name]
        if task.pairing == "one_vs_rest":
            raise ValueError(f"m2 reads binary tasks only, not {name!r}")
        if not _readable(task, config.readouts)[0]:
            raise ValueError(
                f"task {name!r} has no readout left to read: its concept "
                "readouts need a concept slot or word"
            )
    return tasks


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    examples: Mapping[str, Sequence[Example]],
) -> None:
    """Runs the factorial on every task.

    Per task and seed, every cell is fitted on the training items of the
    paper template, its penalty tuned per layer on the select items of the
    original prompts, and its layer rule applied there; the threshold of
    the thresholded metrics also comes from those select items. Every cell
    then scores the same report items under every condition. Writes, per
    task, `scores_<task>` (the per-seed scores of every cell and series)
    and `auroc_<task>` (the AUROC matrix of the cells by condition); and
    `table.json` (metrics per cell, series and condition, mean over seeds
    with seed intervals), `comparisons.json` (paired comparisons with each
    baseline on the seed ensemble) and `summary.json`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model, in float32 for a real run.
        examples: The examples of each source, by source name.

    Raises:
        ValueError: If the config is refused (see `run`) or a source is
            missing (both checked before any model call).
    """
    tasks = _check(config)
    for name in tasks:
        task = READING_TASKS[name]
        task_examples(task, examples)
        task_examples(target_of(task), examples)
    rows: list[list[Any]] = []
    comparisons: list[list[Any]] = []
    summary: dict[str, Any] = {
        "tasks": list(tasks),
        "conditions": {},
        "skipped": {},
        "n_eval_pairs": {},
        "families": (
            "one per task, condition, layer rule and baseline; Holm over the "
            "DeLong and, separately, the McNemar p values of its cells"
        ),
    }
    for name in tasks:
        task = READING_TASKS[name]
        readouts, skipped = _readable(task, config.readouts)
        splits, shift_skips = _prepare_task(
            config, loaded, task, readouts, examples
        )
        rules: dict[tuple[str, str], list[str]] = {}
        for cell in cells(config, readouts):
            rules.setdefault((cell.scorer, cell.readout), []).append(
                cell.layer_rule
            )
        entries: list[_Entry] = []
        for (scorer, readout), cell_rules in rules.items():
            if scorer == "text":
                text = Cell(scorer, readout, "none")
                entries.append(_Entry(text, "reader", _text_choices(splits)))
            else:
                entries += _cell_entries(
                    config, task, splits, scorer, readout, cell_rules
                )
        task_rows = [
            row
            for e in entries
            for row in _metric_rows(config, task, splits, e)
        ]
        rows += task_rows
        comparisons += _comparison_rows(config, task, splits, entries)
        _write_task(run_dir, task, splits, entries, task_rows)
        position = next(iter(splits.report))
        summary["conditions"][name] = splits.conditions
        summary["skipped"][name] = {**skipped, **shift_skips}
        summary["n_eval_pairs"][name] = {
            "select": int(splits.select[position][0].groups.shape[0]),
            "report": int(
                splits.report[position]["original"][0].groups.shape[0]
            ),
        }
    run_dir.write_json("table", {"columns": list(COLUMNS), "rows": rows})
    run_dir.write_json(
        "comparisons",
        {"columns": list(COMPARISON_COLUMNS), "rows": comparisons},
    )
    run_dir.write_json("summary", summary)


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the model in float32 and the datasets, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.

    Raises:
        ValueError: If there is no task, scorer, readout or layer rule, a
            grid is empty or not positive, `top_k` is below 1, a baseline is
            not a cell of the run, or a task is unknown, not binary or has
            no readout left to read (all checked before the model is
            loaded).
    """
    _check(config)
    sources = task_sources(config.tasks)
    execute(
        config,
        run_dir,
        load_model(config.model, dtype=torch.float32),
        load_sources(sources),
    )


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the AUROC matrix of every task.

    Args:
        run_dir: A finished M2 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths.
    """
    summary = cast(dict[str, Any], run_dir.read_json("summary"))
    paths: list[Path] = []
    for task in summary["tasks"]:
        paths += plot_matrix_file(
            run_dir, out_dir, stem=f"auroc_{task}", vmin=0.0, vmax=1.0
        )
    return paths
