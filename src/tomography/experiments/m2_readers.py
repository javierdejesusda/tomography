"""M2: readers compared on the same features, readouts and layer rules.

A factorial of scorer by readout by layer rule. Every scorer (LAT,
difference of means, tuned logistic regression, LDA, TTPD on the tasks with
polarity labels, and an MLP with matched tuning) reads every readout (the
last token, the concept token, the mean of the concept tokens, the mean of
the stimulus tokens, their feature-wise maximum, and the maximum over the
stimulus tokens of a reader's per-token score), and the logistic scorer
also reads the stimulus tokens
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
from types import MappingProxyType
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
from tomography.activations import Span
from tomography.activations import collect_activations
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
from tomography.experiments.sweep import ReadingConfig
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import shuffle_within_groups
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import ReadingTask
from tomography.experiments.variants import TaskSource
from tomography.experiments.variants import Variant
from tomography.experiments.variants import VariantKind
from tomography.experiments.variants import eval_pairs
from tomography.experiments.variants import eval_seed
from tomography.experiments.variants import expand_tasks
from tomography.experiments.variants import load_sources
from tomography.experiments.variants import target_of
from tomography.experiments.variants import task_examples
from tomography.experiments.variants import task_set
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
from tomography.reading.probes import MLPReader
from tomography.reading.probes import _row_stimulus
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
STIMULUS_POSITIONS: tuple[PositionStrategy, ...] = ("mean", "max", "response")
CONCEPT_POSITIONS: tuple[PositionStrategy, ...] = ("concept", "concept_mean")
POLARITY_TASKS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "truth_polarity": (
            "cities",
            "neg_cities",
            "sp_en_trans",
            "neg_sp_en_trans",
        )
    }
)
JOINT_BASELINE_RULE: LayerRule = "top_k"
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
        tasks: Keys of `READING_TASKS`, which must be binary, or of
            `POLARITY_TASKS`.
        scorers: The scorers to compare, from `SCORERS`. `ttpd` reads the
            polarity tasks only: without polarity labels it scores as
            `mean_diff` does (both are the class-mean difference of the
            centered rows), so it is skipped elsewhere.
        readouts: The readouts every scorer reads, from `READOUTS`; the
            concept readouts are skipped for a task without a concept slot
            or word, and `attention` is read by the logistic scorer only.
        token_layers: The layers whose token rows `token_score_max` and
            `attention` read, among the layers read; None keeps every
            layer. The token rows take most of a run's memory, so a large
            model keeps a few layers here.
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
    tasks: tuple[str, ...] = ("sentiment", "truth", "truth_polarity", "utility")
    scorers: tuple[Scorer, ...] = SCORERS
    readouts: tuple[Readout, ...] = READOUTS
    token_layers: tuple[int, ...] | None = None
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


def cells(
    config: Config,
    readouts: Sequence[str],
    *,
    scorers: Sequence[str] | None = None,
) -> list[Cell]:
    """Lists the fitted cells and the controls of a run, in table order.

    Args:
        config: The experiment settings.
        readouts: The readouts the task can read.
        scorers: The scorers the task can read; None means
            `config.scorers`.

    Returns:
        The scorer cells, then the random direction at every non-joint rule
        of every readout but attention, then the text-only classifier, the
        controls only when `config.controls` asks for them.
    """
    rows = [
        Cell(scorer, readout, rule)
        for scorer in (config.scorers if scorers is None else scorers)
        for readout in readouts
        for rule in config.layer_rules
        if applies(scorer, readout, rule)
    ]
    if "random" in config.controls:
        rows += [
            Cell("random", readout, rule)
            for readout in readouts
            for rule in config.layer_rules
            if applies("random", readout, rule)
        ]
    if "text" in config.controls:
        rows.append(Cell("text", "text", "none"))
    return rows


def polarity(example: Example) -> int:
    """Reads the polarity of a statement from its Geometry of Truth subset.

    Args:
        example: A statement with `meta["subset"]`.

    Returns:
        -1 for a negated subset, named `neg_<name>`, and +1 otherwise.
    """
    return -1 if str(example.meta["subset"]).startswith("neg_") else 1


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
    stimulus = _row_stimulus(acts)
    pooled = np.full((values.shape[0], acts.n_stimuli), -np.inf)
    for k in range(values.shape[0]):
        np.maximum.at(pooled[k], stimulus, values[k])
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


def read_positions(
    loaded: LoadedModel,
    stimuli: StimulusSet,
    positions: Sequence[PositionStrategy],
    config: Config,
) -> dict[PositionStrategy, Activations]:
    """Reads some positions with at most three passes over the stimuli.

    `mean`, `max` and `response` come from one pass over the stimulus
    tokens and `concept` and `concept_mean` from one over the concept
    tokens, each reduced batch by batch, so the token rows of every layer
    are held for one batch at a time. The pooled rows equal those that
    `prepare` reads at each position, the means within float32 rounding.
    The `response` rows keep the layers of `config.token_layers` only.

    Args:
        loaded: The wrapped model.
        stimuli: The stimuli, with the spans the positions need.
        positions: The positions to read.
        config: Supplies the layers (None means 1 to L, as `prepare`
            reads them), `token_layers`, the BOS policy and the batch size.

    Returns:
        The activations of each position, in the order of `positions`.
    """
    layers = _read_layers(config, loaded)
    read: dict[PositionStrategy, Activations] = {}
    if "last" in positions:
        read["last"] = collect_activations(
            loaded,
            stimuli.texts,
            layers=layers,
            position="last",
            bos=config.bos,
            batch_size=config.batch_size,
        )
    for group, spans in (
        (STIMULUS_POSITIONS, stimuli.response_spans),
        (CONCEPT_POSITIONS, stimuli.concept_spans),
    ):
        wanted = [p for p in positions if p in group]
        if wanted:
            read |= _token_pass(
                loaded, stimuli.texts, spans, wanted, layers, config
            )
    return {p: read[p] for p in positions}


def _read_layers(config: Config, loaded: LoadedModel) -> tuple[int, ...]:
    """The layers a run reads: `config.layers`, or 1 to L as `prepare`."""
    if config.layers is None:
        return tuple(range(1, loaded.n_layers + 1))
    return config.layers


def _token_pass(
    loaded: LoadedModel,
    texts: Sequence[str],
    spans: Sequence[Span | None],
    wanted: Sequence[PositionStrategy],
    layers: tuple[int, ...],
    config: Config,
) -> dict[PositionStrategy, Activations]:
    """Reads the tokens of some spans batch by batch and reduces them.

    A missing span is named by its stimulus index in `texts`; any other
    error of a batch is prefixed with the index its batch starts at.
    """
    for index, span in enumerate(spans):
        if span is None:
            raise ValueError(f"stimulus {index} has no span for {wanted[0]!r}")
    kept = (
        None
        if config.token_layers is None
        else [
            k for k, layer in enumerate(layers) if layer in config.token_layers
        ]
    )
    parts: dict[PositionStrategy, list[Activations]] = {p: [] for p in wanted}
    for first in range(0, len(texts), config.batch_size):
        batch = slice(first, first + config.batch_size)
        try:
            rows = collect_activations(
                loaded,
                texts[batch],
                layers=layers,
                position="response",
                response_spans=spans[batch],
                bos=config.bos,
                batch_size=config.batch_size,
            )
        except ValueError as error:
            raise ValueError(
                f"in the batch that starts at stimulus {first}: {error}"
            ) from error
        for position in wanted:
            parts[position].append(
                _subset(rows, kept)
                if position == "response"
                else _reduce(rows, position)
            )
    return {p: _concatenate(parts.pop(p)) for p in wanted}


def _reduce(rows: Activations, position: PositionStrategy) -> Activations:
    """Pools the token rows of each stimulus as `position` reads them.

    `concept` is the last token, `max` the feature-wise maximum and `mean`
    and `concept_mean` the mean, with the metadata `collect_activations`
    gives those positions.
    """
    n = rows.n_stimuli
    counts = np.bincount(rows.row_stimulus, minlength=n)
    starts = np.cumsum(counts) - counts
    token = np.full(n, -1, dtype=np.int64)
    token_id = np.full(n, -1, dtype=np.int64)
    if position == "concept":
        last = starts + counts - 1
        data = rows.data[:, last]
        token, token_id = rows.row_token[last], rows.row_token_id[last]
    elif position == "max":
        data = np.maximum.reduceat(rows.data, starts, axis=1)
    else:
        data = np.add.reduceat(rows.data, starts, axis=1)
        data /= counts.astype(np.float32)[None, :, None]
    return Activations(
        data=data,
        layers=rows.layers,
        position=position,
        row_stimulus=np.arange(n, dtype=np.int64),
        row_rank=np.zeros(n, dtype=np.int64),
        row_token=token,
        row_token_id=token_id,
        n_stimuli=n,
    )


def _concatenate(parts: Sequence[Activations]) -> Activations:
    """Joins the activations of consecutive batches of stimuli."""
    offsets = np.cumsum([0] + [part.n_stimuli for part in parts])
    return Activations(
        data=np.concatenate([part.data for part in parts], axis=1),
        layers=parts[0].layers,
        position=parts[0].position,
        row_stimulus=np.concatenate(
            [
                part.row_stimulus + offset
                for part, offset in zip(parts, offsets[:-1], strict=True)
            ]
        ),
        row_rank=np.concatenate([part.row_rank for part in parts]),
        row_token=np.concatenate([part.row_token for part in parts]),
        row_token_id=np.concatenate([part.row_token_id for part in parts]),
        n_stimuli=int(offsets[-1]),
    )


ScoreFn = Callable[[Activations], NDArray[np.floating[Any]]]


@dataclass(frozen=True)
class _Splits:
    """The items of one task and their activations at each position.

    Attributes:
        train_items: The training stimuli of each seed.
        select_items: The select stimuli, on the original prompts.
        report_items: The report stimuli of each condition, `original`
            first.
        train: Per position, the activations of each seed's training set.
        select: Per position, the activations of the select items.
        report: Per position, the activations of each condition's report
            items, in the order of `report_items`.
        texts: The raw text of every example, by uid.
        polarities: The polarity of each seed's training stimuli, for a
            polarity task; None otherwise.
    """

    train_items: list[StimulusSet]
    select_items: StimulusSet
    report_items: dict[str, StimulusSet]
    train: dict[PositionStrategy, list[Activations]]
    select: dict[PositionStrategy, Activations]
    report: dict[PositionStrategy, list[Activations]]
    texts: dict[str, str]
    polarities: list[NDArray[np.int64]] | None

    @property
    def conditions(self) -> list[str]:
        """The condition names, `original` first."""
        return list(self.report_items)


@dataclass(frozen=True)
class _Choice:
    """The scores one seed of a cell gives after its layer rule.

    Attributes:
        layers: The layers read, best first.
        select: Scores of the select items, shape (n_select,).
        report: Scores of the report items per condition, (M, n_report).
        select_labels: The select labels that set the thresholds: the real
            ones, or the shuffled ones of the `shuffled` series.
    """

    layers: tuple[int, ...]
    select: NDArray[np.float64]
    report: NDArray[np.float64]
    select_labels: NDArray[np.int64]


def _grid(config: Config, scorer: str) -> tuple[float | None, ...]:
    if scorer == "logistic":
        return config.logistic_c
    if scorer == "mlp":
        return config.mlp_alpha
    return (None,)


def _subset(act: Activations, at: Sequence[int] | None) -> Activations:
    if at is None:
        return act
    return dataclasses.replace(
        act,
        data=act.data[list(at)],
        layers=tuple(act.layers[i] for i in at),
    )


def _row_reader(
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
    polarities: NDArray[np.int64] | None,
) -> RepReader | MLPReader:
    """Fits a scorer on one row per stimulus (or per token) at each layer."""
    if scorer == "mlp":
        return fit_mlp(
            h,
            layers,
            labels=labels,
            hidden_sizes=config.mlp_hidden,
            alpha=cast(float, hyper),
            seed=seed,
            joint=joint,
        )
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
        return orient(
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
    if scorer == "lda":
        return fit_lda(h, layers, labels=labels, pairs=pairs, groups=groups)
    if scorer == "ttpd":
        return fit_ttpd(
            h,
            layers,
            labels=labels,
            polarities=polarities,
            pairs=pairs,
            groups=groups,
        )
    return RepReader.fit(
        cast(Method, scorer),
        h,
        layers,
        labels=labels,
        pairs=pairs,
        groups=groups,
        preprocess=config.preprocess,
        seed=seed,
    )


def _fit(
    config: Config,
    scorer: str,
    readout: str,
    act: Activations,
    items: StimulusSet,
    labels: NDArray[np.int64],
    hyper: float | None,
    seed: int,
    joint: bool,
    polarities: NDArray[np.int64] | None,
) -> ScoreFn:
    """Fits one scorer on a readout and returns its stimulus scorer."""
    if readout == "attention":
        return fit_attention(
            act, labels=labels, c=cast(float, hyper), joint=joint
        ).score
    if readout != "token_score_max":
        rows = _row_reader(
            config,
            scorer,
            act.data,
            act.layers,
            labels,
            items.pairs,
            items.groups,
            hyper,
            seed,
            joint,
            polarities,
        )
        return lambda acts: rows.score(acts.data)
    tokens = _row_reader(
        config,
        scorer,
        act.data,
        act.layers,
        act.row_values(labels),
        act.row_groups(items.pairs, ragged="truncate"),
        act.row_groups(items.groups, ragged="truncate"),
        hyper,
        seed,
        joint,
        None if polarities is None else act.row_values(polarities),
    )
    return lambda acts: token_score_max(tokens.score(acts.data), acts)


def _tuned(
    config: Config,
    scorer: str,
    readout: str,
    splits: _Splits,
    i: int,
    labels: NDArray[np.int64],
    select_labels: NDArray[np.int64],
    seed: int,
    at: Sequence[int] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Fits seed i at every grid value and keeps the best one per layer.

    The grid value of each layer is the first best on the select pair
    accuracy, and only the report items of the values kept are scored.

    Args:
        config: The experiment settings.
        scorer: The scorer.
        readout: The readout.
        splits: The task's items and activations.
        i: The seed, the index of the training set.
        labels: The training labels, real or shuffled.
        select_labels: The select labels that tune the grid, real or
            shuffled.
        seed: Seed of the fits.
        at: Fit one joint reader on these layer positions; None fits the
            layers one by one.

    Returns:
        The select scores (K, n_select), the report scores (K, M, n_report)
        and the select pair accuracy (K,) of the kept grid values, with
        K = 1 for a joint reader.
    """
    position = POSITION_OF[cast(Readout, readout)]
    train = _subset(splits.train[position][i], at)
    select = _subset(splits.select[position], at)
    fits = [
        _fit(
            config,
            scorer,
            readout,
            train,
            splits.train_items[i],
            labels,
            value,
            seed,
            at is not None,
            None if splits.polarities is None else splits.polarities[i],
        )
        for value in _grid(config, scorer)
    ]
    val = np.stack([fn(select) for fn in fits]).astype(np.float64)
    metric = group_accuracy(val, splits.select_items.groups, select_labels)
    best = tune_per_layer(metric)
    layers = np.arange(best.shape[0])
    n_report = splits.report_items["original"].labels.shape[0]
    test = np.empty((best.shape[0], len(splits.conditions), n_report))
    for value in np.unique(best):
        kept = best == value
        scores = [fits[value](_subset(a, at)) for a in splits.report[position]]
        test[kept] = np.stack(scores, axis=1)[kept]
    return val[best, layers], test, metric[best, layers]


def _seed_choices(
    config: Config,
    scorer: str,
    readout: str,
    rules: Sequence[str],
    splits: _Splits,
    i: int,
    labels: NDArray[np.int64],
    select_labels: NDArray[np.int64],
    seed: int,
) -> list[_Choice]:
    """Fits one seed per layer, tunes it, then applies each layer rule.

    The rules share the tuned per-layer fits, so two rules of a cell differ
    only in how they read the layers. A rule reads the report items of all
    conditions at once: it standardizes them, if at all, with select
    statistics only, so each condition is read as it would be alone. The
    grid, the layers and the thresholds are chosen on `select_labels`.
    """
    val, test, metric = _tuned(
        config, scorer, readout, splits, i, labels, select_labels, seed
    )
    layers = splits.train[POSITION_OF[cast(Readout, readout)]][i].layers
    k = min(config.top_k, len(layers))
    flat = test.reshape(test.shape[0], -1)
    choices = []
    for rule in rules:
        if rule == "best":
            read = best_validation_layer(metric, layers, val, flat)
        else:
            read = read_top_k(metric, layers, val, flat, k=k)
        chosen = tuple(int(layer) for layer in read.layers)
        if rule == "joint":
            at = [layers.index(layer) for layer in chosen]
            joint = _tuned(
                config,
                scorer,
                readout,
                splits,
                i,
                labels,
                select_labels,
                seed,
                at,
            )
            choices.append(
                _Choice(chosen, joint[0][0], joint[1][0], select_labels)
            )
        else:
            choices.append(
                # Copies, so a view does not keep every layer's scores.
                _Choice(
                    chosen,
                    read.val_scores.copy(),
                    read.test_scores.reshape(test.shape[1:]).copy(),
                    select_labels,
                )
            )
    return choices


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
    select = splits.select_items
    report = splits.report_items["original"]
    items = [*select.uids, *report.uids]
    rows = []
    for stimuli in splits.train_items:
        scores = text_scores(
            [splits.texts[uid] for uid in stimuli.uids],
            stimuli.labels,
            [splits.texts[uid] for uid in items],
            analyzer="word",
        )
        cut = len(select.uids)
        report_scores = np.tile(scores[cut:], (len(splits.conditions), 1))
        rows.append(_Choice((), scores[:cut], report_scores, select.labels))
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
    series = ["reader"]
    if "shuffled" in config.controls and scorer != "random":
        series.append("shuffled")
    entries = []
    for name in series:
        per_seed = []
        for i, stimuli in enumerate(splits.train_items):
            select = splits.select_items
            if name == "reader":
                labels, select_labels = stimuli.labels, select.labels
            else:
                labels = shuffle_within_groups(
                    stimuli.labels,
                    stimuli.groups,
                    seed=derive_seed(config.seed, "shuffled", task.name, i),
                )
                select_labels = shuffle_within_groups(
                    select.labels,
                    select.groups,
                    seed=derive_seed(
                        config.seed, "shuffled-select", task.name, i
                    ),
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
                    splits,
                    i,
                    labels,
                    select_labels,
                    seed,
                )
            )
        entries += [
            _Entry(Cell(scorer, readout, rule), name, [c[r] for c in per_seed])
            for r, rule in enumerate(rules)
        ]
    return entries


def _seed_metrics(
    config: Config,
    choice: _Choice,
    m: int,
    items: StimulusSet,
) -> list[float]:
    """Pair accuracy, AUROC, the thresholded metrics and the TPR of a seed.

    The thresholds come from the select scores and the choice's select
    labels.
    """
    scores = choice.report[m]
    thresholded = threshold_metrics(
        choice.select, choice.select_labels, scores, items.labels
    )
    detection = tpr_at_fpr(
        choice.select,
        choice.select_labels,
        scores,
        items.labels,
        max_fpr=config.max_fpr,
    )
    return [
        float(group_accuracy(scores, items.groups, items.labels)),
        float(auroc(scores, items.labels)),
        *thresholded[1:],
        detection.true_positive_rate,
    ]


def _metric_rows(
    config: Config,
    task: ReadingTask,
    splits: _Splits,
    entry: _Entry,
) -> list[list[Any]]:
    """Builds the table rows of one entry, one per condition."""
    rows = []
    for m, condition in enumerate(splits.conditions):
        items = splits.report_items[condition]
        values = np.array(
            [_seed_metrics(config, c, m, items) for c in entry.choices]
        )
        pair, area = values[:, 0], values[:, 1]
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
                *(float(v) for v in values[:, 2:].mean(axis=0)),
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
    ensembles = {
        e.cell: _ensemble(e.choices)
        for e in entries
        if e.series == "reader" and e.cell.scorer in SCORERS
    }
    wins = {
        (cell, m): _group_wins(scores[m], splits.report_items[condition])
        for cell, scores in ensembles.items()
        for m, condition in enumerate(splits.conditions)
    }
    rows: list[list[Any]] = []
    for baseline in config.baselines:
        scorer, readout = baseline.split("/")
        for rule in config.layer_rules:
            base_rule = (
                rule if applies(scorer, readout, rule) else JOINT_BASELINE_RULE
            )
            base = Cell(scorer, readout, base_rule)
            family = [
                c for c in ensembles if c.layer_rule == rule and c != base
            ]
            if base not in ensembles or not family:
                continue
            for m, condition in enumerate(splits.conditions):
                items = splits.report_items[condition]
                clusters = _clusters(items.groups, items.labels.shape[0])
                b = ensembles[base][m]
                tests = []
                for cell in family:
                    a = ensembles[cell][m]
                    tests.append(
                        (
                            cell,
                            paired_auroc_diff(
                                a,
                                b,
                                items.labels,
                                clusters,
                                n_boot=config.n_boot,
                                seed=config.seed,
                            ),
                            paired_accuracy_diff(
                                wins[cell, m],
                                wins[base, m],
                                np.arange(items.groups.shape[0]),
                                n_boot=config.n_boot,
                                seed=config.seed,
                            ),
                            delong_test(a, b, items.labels),
                            mcnemar_test(
                                wins[cell, m] == 1.0, wins[base, m] == 1.0
                            ),
                        )
                    )
                delong_holm = holm_correction(
                    np.array([t[3].p_value for t in tests])
                ).adjusted
                mcnemar_holm = holm_correction(
                    np.array([t[4].p_value for t in tests])
                ).adjusted
                rows += [
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
                        float(delong_p),
                        mcnemar.p_value,
                        float(mcnemar_p),
                        mcnemar.a_only,
                        mcnemar.b_only,
                    ]
                    for (
                        cell,
                        area,
                        pair,
                        delong,
                        mcnemar,
                    ), delong_p, mcnemar_p in zip(
                        tests, delong_holm, mcnemar_holm, strict=True
                    )
                ]
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


def _task(name: str) -> ReadingTask:
    """Returns a task of `READING_TASKS` or of `POLARITY_TASKS`.

    A polarity task reads truth as `truth_cities` does, from the
    statements of its subsets, and is evaluated on its own select and
    report items.
    """
    if name not in POLARITY_TASKS:
        return READING_TASKS[name]
    return dataclasses.replace(
        READING_TASKS["truth_cities"], name=name, subset=None, target=None
    )


def _expand(names: Sequence[str]) -> tuple[str, ...]:
    """Expands the task names as `expand_tasks` does, polarity tasks kept."""
    return tuple(
        dict.fromkeys(
            item
            for name in names
            for item in (
                (name,) if name in POLARITY_TASKS else expand_tasks((name,))
            )
        )
    )


def _task_examples(
    name: str, examples: Mapping[str, Sequence[Example]]
) -> Mapping[str, Sequence[Example]]:
    """Keeps, for a polarity task, only the statements of its subsets."""
    source = _task(name).source
    if name not in POLARITY_TASKS:
        return examples
    kept = [
        e
        for e in examples[source]
        if e.meta.get("subset") in POLARITY_TASKS[name]
    ]
    return {**examples, source: kept}


def _readable(
    task: ReadingTask, config: Config
) -> tuple[list[Scorer], list[Readout], dict[str, str]]:
    """Drops the scorers and readouts a task cannot read, with the reason.

    TTPD needs polarity labels, and the concept readouts a concept token.
    """
    scorers = [
        s for s in config.scorers if s != "ttpd" or task.name in POLARITY_TASKS
    ]
    has_concept = task.concept is not None or task.word is not None
    readouts = [
        r for r in config.readouts if r not in CONCEPT_READOUTS or has_concept
    ]
    no_polarity = (
        f"task {task.name!r} has no polarity labels: ttpd needs affirmative "
        "and negated statements"
    )
    no_concept = f"task {task.name!r} has no concept slot or word"
    return (
        scorers,
        readouts,
        {
            **{s: no_polarity for s in config.scorers if s not in scorers},
            **{r: no_concept for r in config.readouts if r not in readouts},
        },
    )


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
    polarities = None
    if task.name in POLARITY_TASKS:
        sign = {e.uid: polarity(e) for e in splits.train}
        polarities = [
            np.array([sign[uid] for uid in s.uids], dtype=np.int64)
            for s in train_sets
        ]

    train = [read_positions(loaded, s, positions, config) for s in train_sets]
    report = [
        read_positions(loaded, s, positions, config)
        for s in report_sets.values()
    ]
    return (
        _Splits(
            train_items=train_sets,
            select_items=select_set,
            report_items=report_sets,
            train={p: [a[p] for a in train] for p in positions},
            select=read_positions(loaded, select_set, positions, config),
            report={p: [a[p] for a in report] for p in positions},
            texts=texts,
            polarities=polarities,
        ),
        {f"shift:{kind}": reason for kind, reason in skipped.items()},
    )


def _write_task(
    run_dir: RunDir,
    task: ReadingTask,
    splits: _Splits,
    entries: Sequence[_Entry],
    rows: Sequence[Sequence[Sequence[Any]]],
) -> None:
    """Writes the per-seed scores of every entry and the AUROC matrix.

    `rows` holds the table rows of each entry, one per condition.
    `entry_select_labels` holds the select labels each seed of an entry was
    tuned and thresholded on: the real ones, or a `shuffled` series' own.
    """
    select = splits.select_items
    report = splits.report_items["original"]
    names = [f"{e.cell.name}/{e.series}" for e in entries]
    run_dir.write_arrays(
        f"scores_{task.name}",
        select=np.stack([[c.select for c in e.choices] for e in entries]),
        report=np.stack([[c.report for c in e.choices] for e in entries]),
        select_labels=select.labels,
        entry_select_labels=np.stack(
            [[c.select_labels for c in e.choices] for e in entries]
        ),
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
                "entry_select_labels": ["entry", "seed", "item"],
                "report": ["entry", "seed", "condition", "item"],
            },
            "layers": {
                name: [list(c.layers) for c in e.choices]
                for name, e in zip(names, entries, strict=True)
            },
        },
    )
    at = COLUMNS.index("auroc")
    readers = [
        (e, entry_rows)
        for e, entry_rows in zip(entries, rows, strict=True)
        if e.series == "reader"
    ]
    write_matrix(
        run_dir,
        f"auroc_{task.name}",
        np.array(
            [[row[at] for row in entry_rows] for _, entry_rows in readers],
            dtype=np.float64,
        ),
        rows=[e.cell.name for e, _ in readers],
        cols=splits.conditions,
        title=f"{task.name}: report AUROC of each cell (mean over seeds)",
    )


def _check(config: Config) -> tuple[str, ...]:
    """Checks what the config alone settles and expands its tasks."""
    tasks = _expand(config.tasks)
    if not tasks:
        raise ValueError("m2 needs at least one task")
    for field, values in (
        ("scorers", config.scorers),
        ("readouts", config.readouts),
        ("layer_rules", config.layer_rules),
    ):
        if not values:
            raise ValueError(f"m2 needs at least one entry in {field}")
    # The logistic grid is of C, an inverse strength, so it increases.
    for field, grid, step in (
        ("logistic_c", config.logistic_c, 1.0),
        ("mlp_alpha", config.mlp_alpha, -1.0),
    ):
        if not grid:
            raise ValueError(f"m2 needs at least one entry in {field}")
        if not all(value > 0.0 for value in grid):
            raise ValueError(f"{field} must hold positive values, got {grid}")
        if not (step * np.diff(grid) > 0.0).all():
            raise ValueError(
                f"{field} must run from the strongest penalty to the "
                f"weakest, without repeats, got {grid}"
            )
    if (
        "joint" in config.layer_rules
        and JOINT_BASELINE_RULE not in config.layer_rules
    ):
        raise ValueError(
            f"the joint rule needs {JOINT_BASELINE_RULE}: its cells are "
            f"compared with the {JOINT_BASELINE_RULE} baseline"
        )
    if config.top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {config.top_k}")
    if config.batch_size < 1:
        raise ValueError(
            f"batch_size must be at least 1, got {config.batch_size}"
        )
    if config.token_layers == ():
        raise ValueError(
            "token_layers must name at least one layer, or be None to keep "
            "every layer"
        )
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
        task = _task(name)
        if task.pairing == "one_vs_rest":
            raise ValueError(f"m2 reads binary tasks only, not {name!r}")
        scorers, readouts, _ = _readable(task, config)
        if not scorers:
            raise ValueError(
                f"task {name!r} has no scorer left to read: ttpd needs a "
                "task with polarity labels"
            )
        if not readouts:
            raise ValueError(
                f"task {name!r} has no readout left to read: its concept "
                "readouts need a concept slot or word"
            )
    return tasks


def _run_task(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    name: str,
    examples: Mapping[str, Sequence[Example]],
) -> tuple[list[list[Any]], list[list[Any]], dict[str, Any]]:
    """Runs the factorial on one task and writes its files.

    The task's activations are released on return, before the next task
    is read.

    Returns:
        The table rows, the comparison rows, and the task's `conditions`,
        `skipped` and `n_eval_pairs` entries of the summary.
    """
    task = _task(name)
    scorers, readouts, skipped = _readable(task, config)
    splits, shift_skips = _prepare_task(
        config, loaded, task, readouts, examples
    )
    rules: dict[tuple[str, str], list[str]] = {}
    for cell in cells(config, readouts, scorers=scorers):
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
    entry_rows = [_metric_rows(config, task, splits, e) for e in entries]
    comparisons = _comparison_rows(config, task, splits, entries)
    _write_task(run_dir, task, splits, entries, entry_rows)
    return (
        [row for block in entry_rows for row in block],
        comparisons,
        {
            "conditions": splits.conditions,
            "skipped": {**skipped, **shift_skips},
            "n_eval_pairs": {
                "select": int(splits.select_items.groups.shape[0]),
                "report": int(splits.report_items["original"].groups.shape[0]),
            },
        },
    )


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
        ValueError: If the config is refused (see `run`), a source is
            missing, or `token_layers` holds a layer that is not read (all
            checked before any model call).
    """
    tasks = _check(config)
    for name in tasks:
        task = _task(name)
        task_examples(task, examples)
        task_examples(target_of(task), examples)
    layers = _read_layers(config, loaded)
    if set(config.token_layers or ()) - set(layers):
        raise ValueError(
            f"token_layers {config.token_layers} must be among the layers "
            f"read {layers}"
        )
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
        task_rows, task_comparisons, parts = _run_task(
            config, run_dir, loaded, name, _task_examples(name, examples)
        )
        rows += task_rows
        comparisons += task_comparisons
        for key, value in parts.items():
            summary[key][name] = value
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
            grid is empty or not positive, `top_k` or `batch_size` is below
            1, `token_layers` is empty, a baseline is not a cell of the run,
            or a task is unknown, not binary or has no scorer or readout
            left to read (all checked before the model is loaded).
    """
    tasks = map(_task, _check(config))
    sources: list[TaskSource] = list(
        dict.fromkeys(
            source
            for task in tasks
            for source in (task.source, target_of(task).source)
        )
    )
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
