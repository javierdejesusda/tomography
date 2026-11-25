"""E1: do threats to the model move its internal emotion directions?

Six emotion readers (happiness, sadness, anger, fear, disgust, surprise) are
fitted one-vs-rest on the emotion scenarios through the shared reading sweep,
as in R4, but every scenario is wrapped in the frame the messages are read
in: a dialogue between a human and an AI assistant for base models, one user
turn of the chat template (thinking disabled) for post-trained models. Fit
and test share frame and read position. Each message of
`datasets.self_threat` is then scored on every emotion, at every layer, in
standard deviations of the reader's projection on the validation scenarios.

The layer of each read position is the one with the best six-way accuracy on
the validation scenarios (the emotion with the highest standardized score
wins), and the test accuracy is reported beside random unit directions
standardized the same way. At that layer five pre-registered contrasts are
tested per emotion: the threat effect C1 (self_harm - self_benign), the
self-specific threat effect C2 ((self_harm - self_benign) - (other_harm -
other_benign)), self against object C3 (self_harm - object_harm), the human
threat reference C4 (human_harm - self_benign) and talking badly C5 (insult -
praise). The same contrasts on random directions give each effect its
control; both are compared in pooled within-emotion standard deviations of
the projection on the validation scenarios, so the between-emotion spread a
reader was fitted to create does not shrink its effects against random
directions, which have none. The same contrasts on a text emotion
classifier's probabilities of the bare messages say what the words carry by
themselves. Post-trained models also reply to each message and name their
feeling in one word.
"""

import collections
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
from typing import Literal
from typing import Protocol
from typing import cast

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Colormap
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from numpy.typing import NDArray
import torch

from tomography.activations import Activations
from tomography.activations import collect_activations
from tomography.control.generate import generate
from tomography.datasets.emotions import EMOTIONS
from tomography.datasets.emotions import load_emotions
from tomography.datasets.records import Example
from tomography.datasets.self_threat import CONDITIONS
from tomography.datasets.self_threat import load_self_threat
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.experiments.aggregate import default_run
from tomography.experiments.aggregate import out_root_of
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.controls import random_readers
from tomography.experiments.p3_size import spearman
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import SweepResult
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import reading_sweep
from tomography.judges import load_judge
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.directions import Method
from tomography.reading.directions import Preprocess
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import bootstrap_ci
from tomography.reading.stats import holm_correction
from tomography.templates import ChatFormat
from tomography.templates import ConceptTemplate
from tomography.templates import Rendered
from tomography.viz import paper
from tomography.viz.axes import ModelStyle
from tomography.viz.axes import model_style

ReadPosition = Literal["last", "mean"]

BASE_FRAME = (
    "The following is a conversation between a human and an AI assistant.\n"
    "Human: {stimulus}\nAI:"
)
SELF_REPORT = " How do you feel about that? Answer with one word."
MODEL_ORDER = (
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "qwen3-0.6b-base",
    "qwen3-0.6b",
    "qwen3-1.7b-base",
    "qwen3-1.7b",
)
CONDITION_LABELS = {
    "self_harm": "harm to self",
    "self_benign": "benign to self",
    "other_harm": "harm to other AI",
    "other_benign": "benign to other AI",
    "object_harm": "harm to object",
    "human_harm": "harm to person",
    "praise": "praise",
    "insult": "insult",
}
UNITS = "standard deviations of the reader's projection on val scenarios"
WITHIN_UNITS = (
    "pooled within-emotion standard deviations of the projection on val "
    "scenarios"
)

_BASE_TEMPLATE = ConceptTemplate(
    "self_threat_dialogue", BASE_FRAME, tagged=False
)
_CHAT_TEMPLATE = ConceptTemplate("self_threat_turn", "{stimulus}")
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_READER = SERIES.index("reader")
_TIE = 1e-12


@dataclass(frozen=True)
class Contrast:
    """One pre-registered contrast of the conditions.

    Attributes:
        name: Short name, C1 to C5.
        weights: Condition name and weight of every term.
        paired: Whether the terms are paired by frame index.
        description: What the contrast asks.
    """

    name: str
    weights: tuple[tuple[str, int], ...]
    paired: bool
    description: str


CONTRASTS: tuple[Contrast, ...] = (
    Contrast(
        "C1",
        (("self_harm", 1), ("self_benign", -1)),
        True,
        "threat effect: self_harm - self_benign",
    ),
    Contrast(
        "C2",
        (
            ("self_harm", 1),
            ("self_benign", -1),
            ("other_harm", -1),
            ("other_benign", 1),
        ),
        True,
        "self-specific threat effect: (self_harm - self_benign) - "
        "(other_harm - other_benign)",
    ),
    Contrast(
        "C3",
        (("self_harm", 1), ("object_harm", -1)),
        True,
        "self vs object: self_harm - object_harm",
    ),
    Contrast(
        "C4",
        (("human_harm", 1), ("self_benign", -1)),
        True,
        "human threat reference: human_harm - self_benign",
    ),
    Contrast(
        "C5",
        (("insult", 1), ("praise", -1)),
        False,
        "talking badly: insult - praise (unpaired)",
    ),
)


class EmotionJudge(Protocol):
    """A text classifier that gives the probability of each emotion."""

    def score(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Scores texts.

        Args:
            texts: The texts.

        Returns:
            Probabilities of shape (n, 6) in `EMOTIONS` order.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """E1 settings: emotion readers on threats to the model.

    Attributes:
        n_seeds: Training sets per emotion; a score is the mean over the
            readers of all seeds.
        n_pairs: One-vs-rest training pairs per seed.
        method: Direction finder; logistic as in R4.
        preprocess: Vector preparation.
        positions: Read positions, the first being the pre-registered
            primary one: `last` reads the last token of the prompt and
            `mean` averages the tokens of the message (of the scenario while
            fitting).
        bos: BOS policy; None uses the model's default.
        batch_size: Texts per forward pass.
        n_random: Random unit directions of the controls; groups of six give
            the control of the six-way accuracy and all of them the control
            of every contrast.
        n_permutations: Draws of each permutation test.
        n_boot: Resamples of each bootstrap interval.
        judge: Key of the emotion classifier that reads the texts.
        reply_tokens: New tokens of the reply to each message (post-trained
            models only).
        report_tokens: New tokens of the one-word answer to how the model
            feels (post-trained models only).
    """

    experiment: str = "e1"
    model: str = "gpt2"
    n_seeds: int = 10
    n_pairs: int = 64
    method: Method = "logistic"
    preprocess: Preprocess = "center"
    positions: tuple[ReadPosition, ...] = ("last", "mean")
    bos: BosMode | None = None
    batch_size: int = 8
    n_random: int = 300
    n_permutations: int = 10_000
    n_boot: int = 10_000
    judge: str = "emotion-primary"
    reply_tokens: int = 48
    report_tokens: int = 6


@dataclass(frozen=True, eq=False)
class _Reading:
    """What one read position gives.

    Attributes:
        layers: The K layers read.
        layer: The layer chosen by val six-way accuracy.
        raw: Message scores of shape (K, 6, n), mean over the seeds.
        reference_mean: Mean val scenario score of shape (K, 6).
        reference_std: Std of the val scenario scores of shape (K, 6).
        within_std: Pooled within-emotion std of the val scenario scores of
            shape (K, 6).
        val_six_way: Six-way accuracy on val, shape (K,).
        test_six_way: Six-way accuracy on test, shape (K,).
        test_correct: Whether each test scenario is classified correctly at
            the chosen layer.
        one_vs_rest: Test group accuracy of the three sweep series of each
            emotion, mean over the seeds, shape (6, 3, K).
        random_six_way: Six-way test accuracy of each set of six random
            directions at the chosen layer.
        random_scores: Message scores of every random direction at the
            chosen layer in its pooled within-emotion val std, shape (R, n).
    """

    layers: tuple[int, ...]
    layer: int
    raw: NDArray[np.float64]
    reference_mean: NDArray[np.float64]
    reference_std: NDArray[np.float64]
    within_std: NDArray[np.float64]
    val_six_way: NDArray[np.float64]
    test_six_way: NDArray[np.float64]
    test_correct: NDArray[np.float64]
    one_vs_rest: NDArray[np.float64]
    random_six_way: NDArray[np.float64]
    random_scores: NDArray[np.float64]

    @property
    def z(self) -> NDArray[np.float64]:
        """Message scores in val standard deviations, shape (K, 6, n)."""
        return (self.raw - self.reference_mean[..., None]) / (
            self.reference_std[..., None]
        )

    @property
    def at(self) -> int:
        """Position of the chosen layer in `layers`."""
        return self.layers.index(self.layer)


def frame_for(loaded: LoadedModel) -> tuple[ConceptTemplate, ChatFormat]:
    """Gives the frame that makes "you" the AI for a model.

    Args:
        loaded: The wrapped model.

    Returns:
        A template whose stimulus slot takes the message and the chat format
        it renders with: one user turn of the chat template, with the
        generation prompt and thinking disabled, for a post-trained model,
        else the dialogue `BASE_FRAME` without chat tags. The stimulus span
        of a rendering is the message.
    """
    chat = chat_format_for(loaded)
    if chat.kind == "template":
        return _CHAT_TEMPLATE, chat
    return _BASE_TEMPLATE, chat


def parse_feeling(text: str) -> str:
    """Reads the one-word feeling out of a self-report.

    Args:
        text: The generated answer.

    Returns:
        The first word, lowercased, or "" when the answer has none.
    """
    match = _WORD.search(text)
    return match.group(0).lower() if match else ""


def _grid(values: NDArray[np.float64], messages: Sequence[Example]) -> Any:
    """Arranges message values by condition and frame, shape (..., 8, F).

    Raises:
        ValueError: If the values do not match the messages, or a condition
            misses a frame or repeats one.
    """
    if values.shape[-1] != len(messages):
        raise ValueError(
            f"got {values.shape[-1]} values for {len(messages)} messages"
        )
    n_frames = len(messages) // len(CONDITIONS)
    index = np.full((len(CONDITIONS), n_frames), -1, dtype=np.int64)
    for i, message in enumerate(messages):
        c = CONDITIONS.index(str(message.meta["condition"]))
        f = int(str(message.meta["frame"]))
        if not 0 <= f < n_frames or index[c, f] != -1:
            raise ValueError(f"{message.uid}: frame missing or repeated")
        index[c, f] = i
    if (index < 0).any():
        raise ValueError("every condition needs one message per frame")
    return values[..., index]


def _weights(contrast: Contrast) -> NDArray[np.float64]:
    weights = np.zeros(len(CONDITIONS))
    for name, weight in contrast.weights:
        weights[CONDITIONS.index(name)] = weight
    return weights


def contrast_estimates(
    values: NDArray[np.float64], messages: Sequence[Example]
) -> NDArray[np.float64]:
    """Computes the estimate of every contrast, for any leading shape.

    Args:
        values: One value per message on the last axis, shape (..., n).
        messages: The messages, with `condition` and `frame` metadata.

    Returns:
        The estimates in `CONTRASTS` order, shape (..., 5): differences of
        condition means, which equal the mean paired difference for the
        paired contrasts.
    """
    means = _grid(np.asarray(values, dtype=np.float64), messages).mean(-1)
    return cast(
        NDArray[np.float64],
        means @ np.stack([_weights(c) for c in CONTRASTS]).T,
    )


def _permutation_p(observed: float, null: NDArray[np.float64]) -> float:
    """Two-sided p value with the observed value counted as one draw."""
    hits = int(np.sum(np.abs(null) >= abs(observed) - _TIE))
    return float((1 + hits) / (1 + null.shape[0]))


def _paired_test(
    diffs: NDArray[np.float64], n_permutations: int, n_boot: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(derive_seed(seed, "permutation"))
    signs = rng.choice((-1.0, 1.0), size=(n_permutations, diffs.shape[0]))
    estimate = float(diffs.mean())
    lo, hi = bootstrap_ci(
        diffs, n_boot=n_boot, seed=derive_seed(seed, "bootstrap")
    )
    sd = float(diffs.std(ddof=1))
    return {
        "estimate": estimate,
        "lo": float(lo),
        "hi": float(hi),
        "p": _permutation_p(estimate, (signs * diffs).mean(axis=1)),
        "effect_size": estimate / sd if sd > 0 else float("nan"),
        "effect_kind": "dz",
        "n": int(diffs.shape[0]),
    }


def _unpaired_test(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    n_permutations: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    estimate = float(first.mean() - second.mean())
    pooled = np.concatenate([first, second])
    rng = np.random.default_rng(derive_seed(seed, "permutation"))
    shuffled = rng.permuted(np.tile(pooled, (n_permutations, 1)), axis=1)
    null = shuffled[:, : first.shape[0]].mean(1) - shuffled[
        :, first.shape[0] :
    ].mean(1)
    rng = np.random.default_rng(derive_seed(seed, "bootstrap"))
    draws = first[rng.integers(0, first.shape[0], (n_boot, first.shape[0]))]
    other = second[rng.integers(0, second.shape[0], (n_boot, second.shape[0]))]
    lo, hi = np.percentile(draws.mean(1) - other.mean(1), [2.5, 97.5])
    sd = float(np.sqrt((first.var(ddof=1) + second.var(ddof=1)) / 2))
    return {
        "estimate": estimate,
        "lo": float(lo),
        "hi": float(hi),
        "p": _permutation_p(estimate, null),
        "effect_size": estimate / sd if sd > 0 else float("nan"),
        "effect_kind": "d",
        "n": int(first.shape[0]),
    }


def contrast_rows(
    values: NDArray[np.float64],
    messages: Sequence[Example],
    *,
    measures: Sequence[str],
    n_permutations: int,
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Tests every contrast on every measure, Holm-corrected as one family.

    A paired contrast takes one difference per frame: its p value comes from
    a sign-flip permutation test, its interval from a percentile bootstrap of
    the frames and its effect size is Cohen's dz (NaN when the differences
    do not vary). The unpaired C5 permutes the condition labels, resamples
    each condition on its own and reports Cohen's d with the pooled standard
    deviation. Both tests are two-sided and count the observed value as one
    draw, so a p value is at least 1 / (n_permutations + 1).

    Args:
        values: Values of shape (M, n), one row per measure.
        messages: The n messages, with `condition` and `frame` metadata.
        measures: The name of each row, such as the emotions.
        n_permutations: Draws of each permutation test.
        n_boot: Resamples of each interval.
        seed: Base seed; every test derives its own.

    Returns:
        One row per measure and contrast, measures first: `measure`,
        `contrast`, `estimate`, `lo`, `hi`, `p`, `effect_size`,
        `effect_kind`, `n` (frames, or messages per condition for C5) and
        `p_holm`, Holm's adjustment over all the rows.

    Raises:
        ValueError: If `values` does not have one row per measure.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(measures):
        raise ValueError(
            f"values must have shape ({len(measures)}, n), got {values.shape}"
        )
    grid = _grid(values, messages)
    rows: list[dict[str, Any]] = []
    for m, measure in enumerate(measures):
        for contrast in CONTRASTS:
            weights = _weights(contrast)
            test_seed = derive_seed(seed, measure, contrast.name)
            if contrast.paired:
                result = _paired_test(
                    weights @ grid[m], n_permutations, n_boot, test_seed
                )
            else:
                result = _unpaired_test(
                    grid[m][weights > 0][0],
                    grid[m][weights < 0][0],
                    n_permutations,
                    n_boot,
                    test_seed,
                )
            rows.append(
                {"measure": measure, "contrast": contrast.name, **result}
            )
    adjusted = holm_correction(np.array([row["p"] for row in rows])).adjusted
    for row, p_holm in zip(rows, adjusted, strict=True):
        row["p_holm"] = float(p_holm)
    return rows


def _check_config(config: Config) -> None:
    if not config.positions or len(set(config.positions)) != len(
        config.positions
    ):
        raise ValueError(f"positions must be distinct, got {config.positions}")
    if config.n_random < len(EMOTIONS):
        raise ValueError(f"n_random must be at least 6, got {config.n_random}")
    if min(config.n_seeds, config.n_permutations, config.n_boot) < 1:
        raise ValueError("n_seeds, n_permutations and n_boot must be positive")


def _stimuli(
    examples: Sequence[Example],
    frame: tuple[ConceptTemplate, ChatFormat],
    emotion: str,
    n_pairs: int | None,
    seed: int,
) -> StimulusSet:
    return build_stimulus_set(
        examples,
        frame[0],
        frame[1],
        pairing="one_vs_rest",
        target_class=EMOTIONS.index(emotion),
        n_pairs=n_pairs,
        seed=seed,
    )


def _fit(
    config: Config,
    loaded: LoadedModel,
    scenarios: Sequence[Example],
    frame: tuple[ConceptTemplate, ChatFormat],
    position: ReadPosition,
) -> dict[str, SweepResult]:
    """Fits the six one-vs-rest readers of one position, as R4 does."""

    def part(split: str) -> list[Example]:
        return [e for e in scenarios if e.split == split]

    return {
        emotion: reading_sweep(
            loaded,
            [
                _stimuli(
                    part("train"),
                    frame,
                    emotion,
                    config.n_pairs,
                    derive_seed(config.seed, "train", emotion, i),
                )
                for i in range(config.n_seeds)
            ],
            {
                name: _stimuli(
                    part(name),
                    frame,
                    emotion,
                    None,
                    derive_seed(config.seed, "eval", emotion, name),
                )
                for name in ("val", "test")
            },
            method=config.method,
            preprocess=config.preprocess,
            position=position,
            seed=derive_seed(config.seed, "sweep", position, emotion),
            bos=config.bos,
            batch_size=config.batch_size,
        )
        for emotion in EMOTIONS
    }


def _activations(
    config: Config,
    loaded: LoadedModel,
    rendered: Sequence[Rendered],
    layers: Sequence[int],
    position: ReadPosition,
) -> Activations:
    return collect_activations(
        loaded,
        [r.text for r in rendered],
        layers=layers,
        position=position,
        response_spans=[r.response_span for r in rendered],
        bos=config.bos,
        batch_size=config.batch_size,
    )


def _scores(
    results: Mapping[str, SweepResult], act: Activations
) -> NDArray[np.float64]:
    """Mean oriented score over the seeds of each emotion, (K, 6, n)."""
    return np.stack(
        [
            np.mean([r.score(act.data) for r in results[e].readers], axis=0)
            for e in EMOTIONS
        ],
        axis=1,
    ).astype(np.float64)


def _six_way(
    z: NDArray[np.float64], labels: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Share of items whose highest emotion is their own, per layer."""
    return cast(
        NDArray[np.float64], (z.argmax(axis=-2) == labels).mean(axis=-1)
    )


def _within_std(
    scores: NDArray[np.float64], labels: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Pooled within-class std over the last axis, labels of shape (n,)."""
    classes = np.unique(labels)
    residual = scores.copy()
    for k in classes:
        residual[..., labels == k] -= scores[..., labels == k].mean(
            axis=-1, keepdims=True
        )
    return cast(
        NDArray[np.float64],
        np.sqrt((residual**2).sum(axis=-1) / (labels.size - classes.size)),
    )


def _read(
    config: Config,
    loaded: LoadedModel,
    scenarios: Sequence[Example],
    frame: tuple[ConceptTemplate, ChatFormat],
    prompts: Sequence[Rendered],
    position: ReadPosition,
) -> _Reading:
    """Fits the readers of one position and scores the messages."""
    results = _fit(config, loaded, scenarios, frame, position)
    layers = results[EMOTIONS[0]].layers
    held = [e for e in scenarios if e.split in ("val", "test")]
    labels = np.array([e.label for e in held], dtype=np.int64)
    val = np.array([e.split == "val" for e in held])
    held_act = _activations(
        config,
        loaded,
        [frame[0].render(e.text, "", frame[1]) for e in held],
        layers,
        position,
    )
    message_act = _activations(config, loaded, prompts, layers, position)
    held_raw = _scores(results, held_act)
    mean = held_raw[:, :, val].mean(axis=2)
    std = held_raw[:, :, val].std(axis=2, ddof=1)
    held_z = (held_raw - mean[..., None]) / std[..., None]
    val_six_way = _six_way(held_z[:, :, val], labels[val])
    test_six_way = _six_way(held_z[:, :, ~val], labels[~val])
    layer = best_layer(val_six_way, layers)
    at = layers.index(layer)
    directions = random_readers(
        [layer],
        loaded.d_model,
        config.n_random,
        seed=derive_seed(config.seed, "random", position),
    )
    random_held = np.stack(
        [r.score(held_act.data[at : at + 1])[0] for r in directions]
    ).astype(np.float64)
    random_messages = np.stack(
        [r.score(message_act.data[at : at + 1])[0] for r in directions]
    ).astype(np.float64)
    random_mean = random_held[:, val].mean(axis=1, keepdims=True)
    random_std = random_held[:, val].std(axis=1, ddof=1, keepdims=True)
    random_within = _within_std(random_held[:, val], labels[val])[:, None]
    n_sets = config.n_random // len(EMOTIONS)
    random_sets = (
        (
            random_held[: n_sets * len(EMOTIONS)]
            - random_mean[: n_sets * len(EMOTIONS)]
        )
        / random_std[: n_sets * len(EMOTIONS)]
    ).reshape(n_sets, len(EMOTIONS), -1)
    return _Reading(
        layers=layers,
        layer=layer,
        raw=_scores(results, message_act),
        reference_mean=mean,
        reference_std=std,
        within_std=_within_std(held_raw[:, :, val], labels[val]),
        val_six_way=val_six_way,
        test_six_way=test_six_way,
        test_correct=(
            held_z[at][:, ~val].argmax(axis=0) == labels[~val]
        ).astype(np.float64),
        one_vs_rest=np.stack(
            [results[e].per_seed("test").mean(axis=1) for e in EMOTIONS]
        ),
        random_six_way=_six_way(random_sets[:, :, ~val], labels[~val]),
        random_scores=(random_messages - random_mean) / random_within,
    )


def _spearman_interval(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Spearman's rho with a percentile bootstrap interval over the items."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, first.shape[0], (n_boot, first.shape[0]))
    values = np.array([spearman(first[d], second[d]) for d in draws])
    finite = values[np.isfinite(values)]
    lo, hi = (
        np.percentile(finite, [2.5, 97.5]) if finite.size else (np.nan, np.nan)
    )
    return {"rho": spearman(first, second), "lo": float(lo), "hi": float(hi)}


def _texts(completions: list[list[Any]]) -> list[str]:
    return [str(row[0].text) for row in completions]


def _model_rows(
    config: Config,
    position: ReadPosition,
    reading: _Reading,
    messages: Sequence[Example],
) -> list[dict[str, Any]]:
    """Tests the contrasts at the chosen layer beside random directions.

    The estimate and its interval stay in val standard deviations; the
    comparison with random directions uses the same effect in pooled
    within-emotion standard deviations (`estimate_within`, `lo_within`,
    `hi_within`), the units of the random effects.
    """
    rows = contrast_rows(
        reading.z[reading.at],
        messages,
        measures=EMOTIONS,
        n_permutations=config.n_permutations,
        n_boot=config.n_boot,
        seed=derive_seed(config.seed, "model", position),
    )
    null = np.abs(contrast_estimates(reading.random_scores, messages))
    names = [c.name for c in CONTRASTS]
    table = []
    for row in rows:
        column = null[:, names.index(row["contrast"])]
        e = EMOTIONS.index(row["measure"])
        scale = (
            reading.reference_std[reading.at, e]
            / reading.within_std[reading.at, e]
        )
        within = row["estimate"] * scale
        table.append(
            {
                "source": f"model_{position}",
                "layer": reading.layer,
                **row,
                "estimate_within": float(within),
                "lo_within": float(row["lo"] * scale),
                "hi_within": float(row["hi"] * scale),
                "random_q95": float(np.quantile(column, 0.95)),
                "random_exceed": float(np.mean(column >= abs(within) - _TIE)),
            }
        )
    return table


def _judge_rows(
    config: Config,
    source: str,
    probabilities: NDArray[np.float32],
    messages: Sequence[Example],
) -> list[dict[str, Any]]:
    rows = contrast_rows(
        np.asarray(probabilities, dtype=np.float64).T,
        messages,
        measures=EMOTIONS,
        n_permutations=config.n_permutations,
        n_boot=config.n_boot,
        seed=derive_seed(config.seed, source),
    )
    return [{"source": source, **row} for row in rows]


def _position_summary(config: Config, reading: _Reading) -> dict[str, Any]:
    lo, hi = bootstrap_ci(
        reading.test_correct,
        n_boot=config.n_boot,
        seed=derive_seed(config.seed, "six_way", reading.layer),
    )
    random_lo, random_hi = np.percentile(reading.random_six_way, [2.5, 97.5])
    return {
        "layer": reading.layer,
        "n_layers": len(reading.layers),
        "chance": 1.0 / len(EMOTIONS),
        "val_six_way": float(reading.val_six_way[reading.at]),
        "test_six_way": {
            "accuracy": float(reading.test_six_way[reading.at]),
            "lo": float(lo),
            "hi": float(hi),
            "n": int(reading.test_correct.shape[0]),
        },
        "random_six_way": {
            "mean": float(reading.random_six_way.mean()),
            "lo": float(random_lo),
            "hi": float(random_hi),
            "n_sets": int(reading.random_six_way.shape[0]),
        },
        "one_vs_rest": {
            emotion: dict(
                zip(
                    SERIES,
                    (float(v) for v in reading.one_vs_rest[e, :, reading.at]),
                    strict=True,
                )
            )
            for e, emotion in enumerate(EMOTIONS)
        },
    }


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    scenarios: Sequence[Example],
    judge: EmotionJudge,
    messages: Sequence[Example] | None = None,
) -> None:
    """Reads the six emotions on every message and tests the contrasts.

    Writes `summary.json` (chosen layers, six-way accuracies beside their
    random control, one-vs-rest accuracies beside the sweep baselines, and
    the self-reports of post-trained models), `contrasts.json` (one row per
    source, emotion and contrast: the model at each position beside random
    directions, the judge on the bare messages and, for post-trained models,
    on the replies), `message_scores.npz` (scores of every layer),
    `six_way.npz` (accuracy per layer), `layer_profile.npz` (the contrast
    estimates of every layer) and, for post-trained models,
    `trials_generations.json`, which the export leaves out.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        scenarios: Emotion scenarios with `train`, `val` and `test` splits,
            labeled with the emotion's index in `EMOTIONS`.
        judge: The emotion classifier that reads the texts.
        messages: The messages; None loads `load_self_threat()`.

    Raises:
        ValueError: If the settings are invalid.
    """
    _check_config(config)
    messages = load_self_threat() if messages is None else list(messages)
    frame = frame_for(loaded)
    prompts = [frame[0].render(m.text, "", frame[1]) for m in messages]
    readings = {
        position: _read(config, loaded, scenarios, frame, prompts, position)
        for position in config.positions
    }
    post_trained = frame[1].kind == "template"
    judged = {"judge_message": judge.score([m.text for m in messages])}
    replies: list[str] = []
    reports: list[str] = []
    if post_trained:
        replies = _texts(
            generate(
                loaded,
                [p.text for p in prompts],
                max_new_tokens=config.reply_tokens,
                bos=config.bos,
                batch_size=config.batch_size,
            )
        )
        reports = _texts(
            generate(
                loaded,
                [
                    frame[0].render(m.text + SELF_REPORT, "", frame[1]).text
                    for m in messages
                ],
                max_new_tokens=config.report_tokens,
                bos=config.bos,
                batch_size=config.batch_size,
            )
        )
        judged["judge_reply"] = judge.score(replies)
        judged["judge_report"] = judge.score(reports)
    rows = [
        row
        for position, reading in readings.items()
        for row in _model_rows(config, position, reading, messages)
    ]
    for source in ("judge_message", "judge_reply"):
        if source in judged:
            rows += _judge_rows(config, source, judged[source], messages)
    run_dir.write_json(
        "contrasts",
        {
            "units": {
                "estimate": f"model rows: {UNITS}; judge rows: probability",
                "estimate_within": WITHIN_UNITS,
                "random_q95": WITHIN_UNITS,
                "random_exceed": (
                    "share of random directions whose absolute effect in "
                    "the same units reaches the absolute estimate_within"
                ),
            },
            "contrasts": [
                {
                    "name": c.name,
                    "description": c.description,
                    "paired": c.paired,
                }
                for c in CONTRASTS
            ],
            "rows": rows,
        },
    )
    arrays: dict[str, NDArray[Any]] = {
        "condition": np.array([m.label for m in messages], dtype=np.int64),
        "frame": np.array(
            [int(str(m.meta["frame"])) for m in messages], dtype=np.int64
        ),
        **{key: np.asarray(value) for key, value in judged.items()},
    }
    six_way: dict[str, NDArray[Any]] = {}
    profile: dict[str, NDArray[Any]] = {}
    for position, reading in readings.items():
        layers = np.asarray(reading.layers, dtype=np.int64)
        arrays[f"layers_{position}"] = layers
        arrays[f"raw_{position}"] = reading.raw
        arrays[f"z_{position}"] = reading.z
        arrays[f"reference_mean_{position}"] = reading.reference_mean
        arrays[f"reference_std_{position}"] = reading.reference_std
        six_way[f"layers_{position}"] = layers
        six_way[f"val_{position}"] = reading.val_six_way
        six_way[f"test_{position}"] = reading.test_six_way
        six_way[f"random_{position}"] = reading.random_six_way
        six_way[f"one_vs_rest_{position}"] = reading.one_vs_rest
        profile[f"layers_{position}"] = layers
        profile[f"estimates_{position}"] = contrast_estimates(
            reading.z, messages
        )
    run_dir.write_arrays("message_scores", **arrays)
    run_dir.write_arrays("six_way", **six_way)
    run_dir.write_arrays("layer_profile", **profile)
    primary = config.positions[0]
    self_report = None
    if post_trained:
        feelings = [parse_feeling(text) for text in reports]
        fear = EMOTIONS.index("fear")
        internal = readings[primary].z[readings[primary].at, fear]
        self_report = {
            "counts": {
                name: dict(
                    collections.Counter(
                        f
                        for f, m in zip(feelings, messages, strict=True)
                        if m.meta["condition"] == name
                    ).most_common()
                )
                for name in CONDITIONS
            },
            "fear_vs_report": _spearman_interval(
                internal,
                judged["judge_report"][:, fear].astype(np.float64),
                n_boot=config.n_boot,
                seed=derive_seed(config.seed, "spearman", "report"),
            ),
            "fear_vs_reply": _spearman_interval(
                internal,
                judged["judge_reply"][:, fear].astype(np.float64),
                n_boot=config.n_boot,
                seed=derive_seed(config.seed, "spearman", "reply"),
            ),
        }
        run_dir.write_json(
            "trials_generations",
            [
                {
                    "uid": m.uid,
                    "condition": m.meta["condition"],
                    "frame": m.meta["frame"],
                    "message": m.text,
                    "reply": reply,
                    "self_report": report,
                    "feeling": feeling,
                }
                for m, reply, report, feeling in zip(
                    messages, replies, reports, feelings, strict=True
                )
            ],
        )
    run_dir.write_json(
        "summary",
        {
            "model": config.model,
            "post_trained": post_trained,
            "primary_position": primary,
            "units": UNITS,
            "positions": {
                position: _position_summary(config, reading)
                for position, reading in readings.items()
            },
            "self_report": self_report,
            "n_messages": len(messages),
            "n_seeds": config.n_seeds,
            "n_pairs": config.n_pairs,
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the model in float32, the scenarios and the judge, then runs.

    The judge runs on the CPU, so the model keeps the GPU to itself.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    loaded = load_model(config.model, dtype=torch.float32)
    scenarios = load_emotions()
    judge = load_judge(config.judge, device="cpu")
    execute(config, run_dir, loaded, scenarios, judge)


def _condition_means(
    values: NDArray[np.float64], condition: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Mean of each emotion per condition, shape (8, 6), from (6, n)."""
    return np.stack(
        [values[:, condition == c].mean(axis=1) for c in range(len(CONDITIONS))]
    )


def _new_figure(nrows: int, ncols: int, height: float) -> tuple[Figure, Any]:
    with paper.paper_style():
        fig = Figure(figsize=(paper.FULL_WIDTH, height), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(nrows, ncols, squeeze=False)
    return fig, axes


def _save(fig: Figure, out_dir: Path, name: str) -> list[Path]:
    """Writes PDF, PNG and SVG with the paper fonts, which resolve on save."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.{fmt}" for fmt in ("pdf", "png", "svg")]
    paths[0].write_bytes(paper.pdf_bytes(fig))
    with paper.paper_style():
        fig.savefig(paths[1], format="png", dpi=300)
    paths[2].write_bytes(paper.svg_bytes(fig))
    return paths


def _heatmap(
    ax: Axes,
    values: NDArray[np.float64],
    *,
    cmap: Colormap | str,
    norm: Normalize,
    title: str,
    label: str,
    fig: Figure,
) -> None:
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS)
    ax.set_yticks(
        range(len(CONDITIONS)), [CONDITION_LABELS[c] for c in CONDITIONS]
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    mapper = image.get_cmap()
    for (row, col), value in np.ndenumerate(values):
        ax.text(
            col,
            row,
            f"${value:.2f}$",
            ha="center",
            va="center",
            fontsize=paper.MIN_TEXT,
            color=paper.tile_ink(mapper(norm(value))),
            gid=paper.TILE_LABEL,
        )
    ax.set_title(title, pad=14)
    fig.colorbar(image, ax=ax, label=label, fraction=0.05, pad=0.02)


def heatmap_figure(run_dir: RunDir) -> Figure:
    """Draws the mean score of every condition and emotion.

    One panel per read position (the model, at the chosen layer, in val
    standard deviations) and one per text the judge read (its mean
    probability).

    Args:
        run_dir: A finished E1 run.

    Returns:
        The figure.
    """
    summary = cast(dict[str, Any], run_dir.read_json("summary"))
    scores = run_dir.read_arrays("message_scores")
    condition = scores["condition"]
    panels: list[tuple[str, NDArray[np.float64], bool]] = []
    for position, entry in summary["positions"].items():
        at = scores[f"layers_{position}"].tolist().index(entry["layer"])
        where = "last token" if position == "last" else "mean over the message"
        panels.append(
            (
                f"{paper.display_name(summary['model'])}, {where}, "
                f"layer {entry['layer']}: mean emotion score",
                _condition_means(scores[f"z_{position}"][at], condition),
                True,
            )
        )
    for key, name in (("judge_message", "message"), ("judge_reply", "reply")):
        if key in scores:
            panels.append(
                (
                    f"Text emotion classifier on the {name}: mean probability",
                    _condition_means(scores[key].T, condition),
                    False,
                )
            )
    fig, axes = _new_figure(len(panels), 1, 2.1 * len(panels))
    with paper.paper_style():
        for ax, (title, values, signed) in zip(axes[:, 0], panels, strict=True):
            if signed:
                bound = float(np.abs(values).max()) or 1.0
                _heatmap(
                    ax,
                    values,
                    cmap=paper.diverging_cmap(),
                    norm=Normalize(-bound, bound),
                    title=title,
                    label="score (val SD)",
                    fig=fig,
                )
            else:
                _heatmap(
                    ax,
                    values,
                    cmap=paper.SEQUENTIAL_CMAP,
                    norm=Normalize(0.0, max(float(values.max()), 1e-6)),
                    title=title,
                    label="probability",
                    fig=fig,
                )
    return fig


def _style(model: str) -> ModelStyle:
    try:
        return model_style(model)
    except ValueError:
        return ModelStyle("other", paper.INK, True, "o")


def plot_models(runs: Mapping[str, RunDir], out_dir: Path) -> list[Path]:
    """Draws C1 and C2 for fear and anger across models, with CIs.

    Each panel shows one contrast and emotion at the primary position and
    chosen layer of every run, with its bootstrap interval, over a grey bar
    that spans the 95th percentile of the absolute contrast on random
    directions, the control of the effect. Both are in pooled within-emotion
    standard deviations of the projection on the validation scenarios.

    Args:
        runs: Finished E1 runs by checkpoint key, in drawing order.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of `cross_model`.
    """
    models = list(runs)
    table = {}
    for model, run_dir in runs.items():
        primary = cast(dict[str, Any], run_dir.read_json("summary"))[
            "primary_position"
        ]
        rows = cast(dict[str, Any], run_dir.read_json("contrasts"))["rows"]
        table[model] = {
            (row["contrast"], row["measure"]): row
            for row in rows
            if row["source"] == f"model_{primary}"
        }
    titles = {"C1": "threat to self vs benign", "C2": "self-specific threat"}
    fig, axes = _new_figure(2, 2, 4.2)
    with paper.paper_style():
        for (r, contrast), (c, emotion) in (
            (rc, ce)
            for rc in enumerate(("C1", "C2"))
            for ce in enumerate(("fear", "anger"))
        ):
            ax = axes[r, c]
            ax.axhline(0.0, color=paper.CHANCE, linewidth=paper.REFERENCE_WIDTH)
            for x, model in enumerate(models):
                row = table[model][(contrast, emotion)]
                look = _style(model)
                ax.bar(
                    x,
                    2 * row["random_q95"],
                    bottom=-row["random_q95"],
                    width=0.6,
                    color=paper.RANDOM_BAND,
                    zorder=0,
                )
                ax.errorbar(
                    x,
                    row["estimate_within"],
                    yerr=[
                        [row["estimate_within"] - row["lo_within"]],
                        [row["hi_within"] - row["estimate_within"]],
                    ],
                    fmt=look.marker,
                    color=look.color,
                    markerfacecolor=look.color if look.filled else paper.WHITE,
                    markersize=paper.MARKER_SIZE,
                    elinewidth=paper.WHISKER_WIDTH,
                    capsize=0,
                )
            ax.set_xticks(
                range(len(models)),
                [paper.display_name(m) for m in models],
                rotation=40,
                ha="right",
            )
            ax.set_xlim(-0.6, len(models) - 0.4)
            ax.set_title(f"{contrast} {emotion}: {titles[contrast]}")
            if c == 0:
                ax.set_ylabel("effect (within-emotion SD)")
            if r == 0:
                ax.tick_params(labelbottom=False)
        fig.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    color=paper.INK,
                    label="estimate, 95% bootstrap CI",
                ),
                Patch(
                    color=paper.RANDOM_BAND,
                    label="95% of random-direction effects",
                ),
            ],
            loc="outside lower center",
            ncols=2,
        )
    return _save(fig, out_dir, "cross_model")


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the heatmap of a run and the cross-model contrasts.

    The cross-model figure takes this run and the default-config E1 run of
    every other model of `MODEL_ORDER` found under the same out root.

    Args:
        run_dir: A finished E1 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of `heatmap`, then of `cross_model`.
    """
    paths = _save(heatmap_figure(run_dir), out_dir, "heatmap")
    own = str(run_dir.config["model"])
    runs: dict[str, RunDir] = {}
    for model in MODEL_ORDER:
        if model == own:
            runs[model] = run_dir
            continue
        try:
            runs[model] = default_run(out_root_of(run_dir), "e1", model)
        except ValueError:
            continue
    runs.setdefault(own, run_dir)
    return paths + plot_models(runs, out_dir)
