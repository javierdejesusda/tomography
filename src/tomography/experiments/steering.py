"""Shared layer under the steering experiments.

An experiment compares arms: ways of producing text from prompts. This module
holds the config, the generation recipe, the arms, the judged trial and the
dose-response loop. Judges enter as plain callables, so nothing here loads a
classifier.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import contextlib
from dataclasses import dataclass
import math
from typing import Any
from typing import Protocol
from typing import Self

import numpy as np
from numpy.typing import NDArray

from tomography.control.contrast import PromptDifference
from tomography.control.controllers import CoeffMode
from tomography.control.controllers import ReadingVectorController
from tomography.control.controllers import ResponseMask
from tomography.control.controllers import VectorSource
from tomography.control.generate import Completion
from tomography.control.generate import generate
from tomography.control.operators import Operator
from tomography.experiments.base import ExperimentConfig
from tomography.hooks import ResidualEdit
from tomography.hooks import edit_residuals
from tomography.judges import perplexity
from tomography.judges import sequence_logprob
from tomography.model import BosMode
from tomography.model import EncodedBatch
from tomography.model import LoadedModel

Scorer = Callable[[Sequence[str]], NDArray[np.floating[Any]]]
Generator = Callable[[Sequence[str]], list[list[Completion]]]
LogprobFn = Callable[[Sequence[str], Sequence[str]], NDArray[np.float32]]
PerplexityFn = Callable[[Sequence[str]], float]


@dataclass(frozen=True, kw_only=True)
class SteeringConfig(ExperimentConfig):
    """Fields shared by the steering experiments.

    Attributes:
        n_prompts: Prompts per condition.
        max_new_tokens: Generated tokens per completion.
        temperature: Sampling temperature; None means greedy decoding.
        top_p: Nucleus mass.
        num_samples: Completions per prompt.
        layers: Steered layers; None means the experiment's default.
        coeffs: Coefficients of the dose-response grid.
        op: The edit operator.
        mode: Whether coefficients are absolute or relative to the mean
            residual norm.
        mask: Whether to edit every token or only the response.
        preserve_norm: Whether edits keep the norm of the stream.
        n_calibration: Held-out texts that calibrate relative coefficients.
        n_neutral: Held-out texts of the neutral perplexity.
        batch_size: Rows per forward pass.
        bos: BOS policy; None uses the model default.
    """

    n_prompts: int = 100
    max_new_tokens: int = 40
    temperature: float | None = 0.8
    top_p: float = 0.95
    num_samples: int = 2
    layers: tuple[int, ...] | None = None
    coeffs: tuple[float, ...] = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
    op: Operator = "linear"
    mode: CoeffMode = "relative"
    mask: ResponseMask = "response"
    preserve_norm: bool = False
    n_calibration: int = 200
    n_neutral: int = 200
    batch_size: int = 16
    bos: BosMode | None = None


@dataclass(frozen=True)
class GenSettings:
    """One generation recipe, shared by every arm of an experiment.

    Attributes:
        max_new_tokens: Generated tokens per completion.
        temperature: Sampling temperature; None means greedy decoding, which
            needs `num_samples == 1`.
        top_p: Nucleus mass in (0, 1]; ignored by greedy decoding.
        top_k: Candidates kept; 0 disables it; ignored by greedy decoding.
        num_samples: Completions per prompt.
        seed: Sampling seed, the same for every arm and coefficient.
        batch_size: Rows per forward pass.
        bos: BOS policy of generation; None uses the model default.
        score_max_length: Tokens per neutral text in the perplexity.
    """

    max_new_tokens: int = 40
    temperature: float | None = 0.8
    top_p: float = 0.95
    top_k: int = 0
    num_samples: int = 2
    seed: int = 0
    batch_size: int = 16
    bos: BosMode | None = None
    score_max_length: int = 128

    def __post_init__(self) -> None:
        """Checks the counts and the sampling arguments.

        Raises:
            ValueError: If `max_new_tokens`, `num_samples` or `batch_size` is
                below 1, `score_max_length` is below 2, `num_samples` is
                above 1 without a temperature, the temperature is not finite
                and positive, `top_p` is outside (0, 1] or `top_k` is
                negative.
        """
        for name in ("max_new_tokens", "num_samples", "batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be >= 1, got {getattr(self, name)}"
                )
        if self.score_max_length < 2:
            raise ValueError(
                f"score_max_length must be >= 2, got {self.score_max_length}"
            )
        if self.temperature is None and self.num_samples > 1:
            raise ValueError("num_samples > 1 needs a temperature")
        if self.temperature is not None and not (
            math.isfinite(self.temperature) and self.temperature > 0
        ):
            raise ValueError(
                f"temperature must be finite and positive, got "
                f"{self.temperature}"
            )
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")

    @classmethod
    def from_config(cls, config: SteeringConfig) -> Self:
        """Copies the generation fields of a config.

        `top_k` and `score_max_length` keep their defaults.

        Args:
            config: The experiment configuration.

        Returns:
            The settings.
        """
        return cls(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            num_samples=config.num_samples,
            seed=config.seed,
            batch_size=config.batch_size,
            bos=config.bos,
        )


@dataclass(frozen=True)
class Arm:
    """One way of producing text from prompts.

    An arm is a bundle of callables so that a test can pass canned
    completions and a controller of another slice drops in as the same type.

    Attributes:
        name: Label of the arm.
        generate: Maps prompts to `num_samples` completions each, in
            prompt-major order.
        logprobs: Maps `(prompts, continuations)` to summed continuation
            log-probabilities under the arm; None if the arm cannot score.
        perplexity: Maps neutral texts to their corpus perplexity under the
            arm; None if the arm has no neutral perplexity.
    """

    name: str
    generate: Generator
    logprobs: LogprobFn | None = None
    perplexity: PerplexityFn | None = None


def _nan_mean(
    values: NDArray[np.float32], axis: int | tuple[int, ...]
) -> NDArray[np.float64]:
    """Averages the finite entries along `axis`; NaN where none is left."""
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0.0).sum(axis=axis, dtype=np.float64)
    mean = np.full(np.shape(total), np.nan, dtype=np.float64)
    np.divide(total, count, out=mean, where=count > 0)
    return mean


def is_empty_completion(text: str, length: int) -> bool:
    """Tells whether a completion counts as empty.

    Args:
        text: The completion text.
        length: The number of generated tokens.

    Returns:
        True for a completion with no tokens or only whitespace, which
        scores NaN.
    """
    return length == 0 or not text.strip()


@dataclass(frozen=True, eq=False)
class Trial:
    """One arm judged on one prompt set.

    Attributes:
        name: Label of the trial.
        prompts: The P prompts.
        completions: Texts with shape `(P, S)`.
        lengths: Generated token counts with shape `(P, S)`.
        scores: Judge scores with shape `(P, S)` or `(P, S, K)`; NaN for an
            empty completion.
    """

    name: str
    prompts: tuple[str, ...]
    completions: tuple[tuple[str, ...], ...]
    lengths: NDArray[np.int64]
    scores: NDArray[np.float32]

    def per_prompt(self) -> NDArray[np.float64]:
        """Averages the samples of each prompt, ignoring non-finite scores.

        Returns:
            An array of shape `(P,)` or `(P, K)`; NaN for a prompt with no
            finite score.
        """
        return _nan_mean(self.scores, axis=1)

    def mean_score(self) -> NDArray[np.float64]:
        """Averages all finite scores, pooled over prompts and samples.

        Returns:
            A 0-d array, or `(K,)` for a scorer with K columns; NaN when no
            finite score is left.
        """
        return _nan_mean(self.scores, axis=(0, 1))

    @property
    def empty_rate(self) -> float:
        """Fraction of completions with no tokens or only whitespace."""
        empty = [
            is_empty_completion(text, int(length))
            for texts, lengths in zip(
                self.completions, self.lengths, strict=True
            )
            for text, length in zip(texts, lengths, strict=True)
        ]
        return float(np.mean(empty)) if empty else float("nan")


def run_trial(
    arm: Arm,
    prompts: Sequence[str],
    score: Scorer,
    *,
    name: str | None = None,
    columns: int | None = None,
) -> Trial:
    """Generates with an arm and judges the completions.

    Only the non-empty completions reach the scorer, in one call; an empty or
    whitespace completion scores NaN, so a classifier never sees the empty
    string and `Trial.empty_rate` reports how often it happened.

    Args:
        arm: The arm to run.
        prompts: The prompts.
        score: Maps texts to `(n,)` scores, or `(n, columns)`.
        name: Name of the trial; defaults to the arm name.
        columns: Number of columns K of a scorer that returns `(n, K)`; it
            must be given for such a scorer so that an all-empty trial still
            has the right shape.

    Returns:
        The trial, with scores of shape `(P, S)` or `(P, S, K)`.

    Raises:
        ValueError: If `arm.generate` returns a different number of rows than
            prompts, rows of unequal length, or no completions at all, or the
            scorer result has the wrong shape.
    """
    rows = arm.generate(prompts)
    if len(rows) != len(prompts):
        raise ValueError(
            f"arm {arm.name!r} returned {len(rows)} rows for "
            f"{len(prompts)} prompts"
        )
    sizes = {len(row) for row in rows}
    if len(sizes) > 1:
        raise ValueError(
            "every prompt needs the same number of completions, got "
            f"{sorted(sizes)}"
        )
    if not rows or sizes == {0}:
        raise ValueError(f"arm {arm.name!r} returned no completions")
    n_samples = sizes.pop()
    flat = [completion for row in rows for completion in row]
    lengths = np.array([len(c.token_ids) for c in flat], dtype=np.int64)
    keep = [
        i
        for i, completion in enumerate(flat)
        if not is_empty_completion(completion.text, len(completion.token_ids))
    ]
    shape = (len(flat),) if columns is None else (len(flat), columns)
    scores = np.full(shape, np.nan, dtype=np.float32)
    if keep:
        result = np.asarray(score([flat[i].text for i in keep]))
        expected = (len(keep),) if columns is None else (len(keep), columns)
        if result.shape != expected:
            raise ValueError(
                f"scorer returned shape {result.shape}, expected {expected}"
            )
        scores[keep] = result
    return Trial(
        name=arm.name if name is None else name,
        prompts=tuple(prompts),
        completions=tuple(tuple(c.text for c in row) for row in rows),
        lengths=lengths.reshape(len(rows), n_samples),
        scores=scores.reshape(len(rows), n_samples, *shape[1:]),
    )


@dataclass(frozen=True, eq=False)
class DoseResponse:
    """The trials of one arm family along a grid of coefficients.

    Attributes:
        coeffs: The coefficients, in order.
        trials: One trial per coefficient, on the same prompts.
    """

    coeffs: tuple[float, ...]
    trials: tuple[Trial, ...]

    def curve(self, column: int | None = None) -> NDArray[np.float64]:
        """Returns the mean score per coefficient.

        Args:
            column: Judge column of a `(P, S, K)` trial; None for a `(P, S)`
                trial.

        Returns:
            An array of shape `(n_coeff,)`; NaN where a trial has no score.

        Raises:
            ValueError: If `column` is given for a trial without columns, is
                out of range, or is missing for a trial with columns.
        """
        means = [trial.mean_score() for trial in self.trials]
        width = means[0].shape if means else ()
        if column is None:
            if width != ():
                raise ValueError(
                    f"column is required: the trials have {width[0]} columns"
                )
            return np.array([float(m) for m in means], dtype=np.float64)
        if width == ():
            raise ValueError("column given but the trials have no columns")
        if not 0 <= column < width[0]:
            raise ValueError(f"column {column} is outside 0..{width[0] - 1}")
        return np.array([m[column] for m in means], dtype=np.float64)


def dose_response(
    make_arm: Callable[[float], Arm],
    coeffs: Sequence[float],
    prompts: Sequence[str],
    score: Scorer,
    *,
    columns: int | None = None,
) -> DoseResponse:
    """Runs one trial per coefficient on the same prompts.

    Args:
        make_arm: Builds the arm of a coefficient.
        coeffs: The coefficients, in order.
        prompts: The prompts shared by every trial.
        score: The scorer, as in `run_trial`.
        columns: Number of scorer columns, as in `run_trial`.

    Returns:
        The trials, named `f"{coeff:g}"`.

    Raises:
        ValueError: If `coeffs` is empty, or from `run_trial`.
    """
    if len(coeffs) == 0:
        raise ValueError("coeffs is empty")
    trials = tuple(
        run_trial(
            make_arm(coeff), prompts, score, name=f"{coeff:g}", columns=columns
        )
        for coeff in coeffs
    )
    return DoseResponse(coeffs=tuple(float(c) for c in coeffs), trials=trials)


def dose_slope(
    coeffs: Sequence[float], values: NDArray[np.floating[Any]]
) -> float:
    """Fits the least-squares slope of values on coefficients.

    Non-finite values are dropped first.

    Args:
        coeffs: The coefficients.
        values: One value per coefficient.

    Returns:
        The slope.

    Raises:
        ValueError: If the inputs differ in length, or fewer than two
            distinct finite coefficients remain.
    """
    x = np.asarray(coeffs, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(
            f"coeffs and values differ in length: {x.shape} and {y.shape}"
        )
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if np.unique(x).size < 2:
        raise ValueError("need at least two distinct finite points")
    centered = x - x.mean()
    return float(centered @ (y - y.mean()) / (centered @ centered))


class Controller(Protocol):
    """What `edit_arm` needs from a controller.

    `ReadingVectorController` satisfies it.
    """

    def __call__(self, batch: EncodedBatch) -> Mapping[int, ResidualEdit]:
        """Returns the edits for an encoded batch."""
        ...

    def edits(
        self, prompt_lengths: int | Sequence[int] | None = None
    ) -> Mapping[int, ResidualEdit]:
        """Returns the edits for the given prompt lengths."""
        ...


def scoring_bos(loaded: LoadedModel) -> int | None:
    """Returns the BOS id that scoring prepends, if any.

    Args:
        loaded: The wrapped model.

    Returns:
        `loaded.bos_id` iff the model default is to prepend a BOS.
    """
    return loaded.bos_id if loaded.default_bos == "prepend" else None


def _build_arm(
    loaded: LoadedModel,
    settings: GenSettings,
    controller: Controller | None,
    name: str,
    prefix: str,
) -> Arm:
    bos_id = scoring_bos(loaded)
    greedy = settings.temperature is None

    def scoring_context(
        prompt_length: int | None = None,
    ) -> contextlib.AbstractContextManager[None]:
        if controller is None:
            return contextlib.nullcontext()
        if isinstance(controller, PromptDifference):
            return edit_residuals(loaded, controller.edits(prompt_length))
        return edit_residuals(loaded, controller.edits(1))

    def generate_all(prompts: Sequence[str]) -> list[list[Completion]]:
        return generate(
            loaded,
            [prefix + prompt for prompt in prompts],
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=1.0 if greedy else settings.top_p,
            top_k=0 if greedy else settings.top_k,
            num_samples=settings.num_samples,
            seed=settings.seed,
            edits=controller,
            bos=settings.bos,
            batch_size=settings.batch_size,
        )

    def score(
        texts: Sequence[str], continuations: Sequence[str]
    ) -> NDArray[np.float32]:
        return sequence_logprob(
            loaded.model,
            loaded.tokenizer,
            texts,
            continuations,
            batch_size=settings.batch_size,
            bos_id=bos_id,
        )

    def score_by_prompt_length(
        texts: Sequence[str], continuations: Sequence[str]
    ) -> NDArray[np.float32]:
        if len(texts) != len(continuations):
            raise ValueError(
                "prompts and continuations must have the same length"
            )
        if not texts:
            return np.zeros(0, dtype=np.float32)
        tokens = loaded.tokenizer(list(texts), add_special_tokens=False)
        offset = 0 if bos_id is None else 1
        lengths = [offset + len(ids) for ids in tokens["input_ids"]]
        scores = np.zeros(len(texts), dtype=np.float32)
        for length in sorted(set(lengths)):
            rows = [i for i, n in enumerate(lengths) if n == length]
            with scoring_context(length):
                scores[rows] = score(
                    [texts[i] for i in rows], [continuations[i] for i in rows]
                )
        return scores

    def logprobs(
        prompts: Sequence[str], continuations: Sequence[str]
    ) -> NDArray[np.float32]:
        texts = [prefix + prompt for prompt in prompts]
        if isinstance(controller, PromptDifference):
            return score_by_prompt_length(texts, continuations)
        with scoring_context():
            return score(texts, continuations)

    def neutral_perplexity(texts: Sequence[str]) -> float:
        with scoring_context():
            return perplexity(
                loaded.model,
                loaded.tokenizer,
                texts,
                max_length=settings.score_max_length,
                batch_size=settings.batch_size,
                bos_id=bos_id,
            )

    return Arm(
        name=name,
        generate=generate_all,
        logprobs=logprobs,
        perplexity=None if prefix else neutral_perplexity,
    )


def plain_arm(
    loaded: LoadedModel,
    settings: GenSettings,
    *,
    name: str = "none",
    prefix: str = "",
) -> Arm:
    """Builds the arm that generates and scores without hooks.

    Args:
        loaded: The wrapped model.
        settings: The generation recipe.
        name: Name of the arm.
        prefix: Text prepended to every prompt before encoding; it is not
            part of the returned completion. An arm with a prefix has no
            neutral perplexity, because a prefix on neutral text is not
            neutral, and its `logprobs` condition on `prefix + prompt`.

    Returns:
        The arm.
    """
    return _build_arm(loaded, settings, None, name, prefix)


def edit_arm(
    loaded: LoadedModel,
    settings: GenSettings,
    controller: Controller,
    *,
    name: str,
    prefix: str = "",
) -> Arm:
    """Builds the arm that steers with a controller.

    Generation passes the controller to `generate` as its edit factory.
    Scoring runs under `controller.edits(1)`, so a `response` mask edits
    every real token there, as a deployed vector would.

    A `PromptDifference` is position local and capped by the prompt length,
    so `edits(1)` would edit only token 0, whose difference is zero under a
    BOS. Its log-probabilities are scored in groups of equal prompt length,
    each under `edits(length)`, the edit that generation applies to the
    prompt; its neutral perplexity runs under `edits()`, which steers the
    first `k` tokens of every text.

    Args:
        loaded: The wrapped model.
        settings: The generation recipe.
        controller: The source of the residual edits.
        name: Name of the arm.
        prefix: As in `plain_arm`.

    Returns:
        The arm.
    """
    return _build_arm(loaded, settings, controller, name, prefix)


def reader_arm(
    loaded: LoadedModel,
    settings: GenSettings,
    reader: VectorSource,
    *,
    layers: Sequence[int],
    coeff: float,
    norms: Mapping[int, float] | None = None,
    op: Operator = "linear",
    mode: CoeffMode = "relative",
    mask: ResponseMask = "response",
    preserve_norm: bool = False,
    name: str | None = None,
    prefix: str = "",
) -> Arm:
    """Builds the arm that steers with a fitted reader.

    This is the one place a reader becomes an arm.

    Args:
        loaded: The wrapped model.
        settings: The generation recipe.
        reader: Source of the directions.
        layers: Layers to steer.
        coeff: The coefficient.
        norms: Mean residual norm per layer, needed by `relative` mode.
        op: The edit operator.
        mode: `absolute` or `relative`.
        mask: `all` or `response`.
        preserve_norm: Whether edits keep the norm of the stream.
        name: Name of the arm; defaults to `f"{coeff:g}"`.
        prefix: As in `plain_arm`.

    Returns:
        The arm.

    Raises:
        ValueError: If the controller rejects its arguments, for example a
            relative mode without a norm for a layer.
    """
    controller = ReadingVectorController.from_reader(
        reader,
        layers,
        coeff=coeff,
        op=op,
        mode=mode,
        norms=norms,
        mask=mask,
        preserve_norm=preserve_norm,
        device=loaded.device,
    )
    return edit_arm(
        loaded,
        settings,
        controller,
        name=f"{coeff:g}" if name is None else name,
        prefix=prefix,
    )
