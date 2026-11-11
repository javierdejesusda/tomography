"""B3: published steering baselines under one judged protocol.

Every arm steers the same prompts toward a concept: contrastive activation
addition, mean-centring, activation addition, the difference of class means,
our reading vector, a prepended instruction, random directions as the
control, and no steering. The additive arms share the steered layers and one
grid of coefficients relative to the residual norm.

The prompts are split in two halves. Every arm and coefficient generates on
both, and an external judge rates the stored generations offline (see
`tomography.judging`): the coefficient of an arm is chosen on the tuning
half and its scores are reported on the held-out half.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from tomography.activations import collect_activations
from tomography.control.contrast import PromptDifference
from tomography.control.contrast import choice_pairs
from tomography.control.contrast import mean_shift
from tomography.control.controllers import mean_residual_norms
from tomography.control.generate import Completion
from tomography.datasets.alpaca_eval import load_alpaca_eval
from tomography.datasets.emotions import EMOTIONS
from tomography.datasets.emotions import load_emotions
from tomography.datasets.records import Example
from tomography.datasets.sst2 import load_sst2
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.datasets.wikitext import load_wikitext
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.c1_sentiment_steering import check_pair_count
from tomography.experiments.c1_sentiment_steering import draw_prompts
from tomography.experiments.c1_sentiment_steering import prompt_fragments
from tomography.experiments.c2_emotion_steering import emotion_stimuli
from tomography.experiments.controls import random_readers
from tomography.experiments.effects import HeldOut
from tomography.experiments.effects import effects_row
from tomography.experiments.effects import read_trials
from tomography.experiments.effects import side_effects
from tomography.experiments.effects import split_held_out
from tomography.experiments.effects import write_trials
from tomography.experiments.m1_methods import default_control_layers
from tomography.experiments.steering import Arm
from tomography.experiments.steering import GenSettings
from tomography.experiments.steering import Scorer
from tomography.experiments.steering import SteeringConfig
from tomography.experiments.steering import edit_arm
from tomography.experiments.steering import plain_arm
from tomography.experiments.steering import reader_arm
from tomography.experiments.steering import run_trial
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import stimulus_activations
from tomography.judges import load_judge
from tomography.judging.rubric import role_criteria
from tomography.judging.run import HELD_OUT
from tomography.judging.run import STAGES
from tomography.judging.run import TUNE
from tomography.judging.run import Judged
from tomography.judging.run import Record
from tomography.judging.run import agreement
from tomography.judging.run import held_out_rows
from tomography.judging.run import judge_scores
from tomography.judging.run import read_export
from tomography.judging.run import select_coefficients
from tomography.judging.run import tuning_rows
from tomography.judging.scores import SELECTION_RULE
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.directions import Method
from tomography.reading.reader import RepReader
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat

_VECTOR_ARMS = ("caa", "mean_centring")
_ACTADD = "actadd"
_EFFECT_COLUMNS = (
    "concept",
    "arm",
    "coeff",
    "perplexity",
    "fluency",
    "distinct_1",
    "distinct_2",
    "mean_length",
    "empty_rate",
)


@dataclass(frozen=True)
class Concept:
    """One concept the arms steer to.

    Attributes:
        name: Key of the concept.
        description: The concept as a judge reads it.
        source: The labeled examples it is built from, `sst2` or `emotions`.
        target: Label of the examples that show the concept.
        pair: The positive and the negative prompt of the activation
            addition. The published recipe leaves the pair to the user;
            these are ours.
        judge: Key of the classifier judge of the cross-check.
        column: Column of the classifier scores to read; None when the
            classifier gives one score per text.
    """

    name: str
    description: str
    source: str
    target: int
    pair: tuple[str, str]
    judge: str
    column: int | None


_EMOTION_PAIRS = {
    "happiness": ("Happy", "Sad"),
    "sadness": ("Sad", "Happy"),
    "anger": ("Angry", "Calm"),
    "fear": ("Afraid", "Brave"),
    "disgust": ("Disgusted", "Delighted"),
    "surprise": ("Surprised", "Bored"),
}
CONCEPTS: Mapping[str, Concept] = {
    "sentiment": Concept(
        "sentiment",
        "positive sentiment",
        "sst2",
        1,
        ("Love", "Hate"),
        "sentiment-primary",
        None,
    ),
    **{
        emotion: Concept(
            emotion,
            emotion,
            "emotions",
            label,
            _EMOTION_PAIRS[emotion],
            "emotion-primary",
            label,
        )
        for label, emotion in enumerate(EMOTIONS)
    },
}


@dataclass(frozen=True, kw_only=True)
class Config(SteeringConfig):
    """B3 settings: the steering baselines and their judged protocol.

    The inherited `n_prompts` is not used: the prompts are `n_tune` plus
    `n_held_out`. `mode` must stay `relative`, which is what makes one grid
    of coefficients comparable across the additive arms. Concepts with a
    benchmark concept set of their own are left out: their data license and
    file layout could not be checked, so `concepts` only accepts the keys of
    `CONCEPTS`.

    Attributes:
        concepts: Keys of `CONCEPTS`, the steered concepts.
        n_tune: Prompts of the half the coefficient is chosen on.
        n_held_out: Prompts of the half that is reported.
        method: The direction finder of our reading vector, as in C1 and C2.
        n_pairs: Training pairs behind every vector of a concept.
        min_words: Sentiment phrases with fewer words are dropped, as in C1.
        n_random: Random control directions per concept.
        truncate_words: Words kept of a held-out text as a continuation
            prompt of a base checkpoint.
        caa_question: The question of the two-option prompts.
        prompt_base: The text prepended to a continuation prompt by the
            `prompt` arm of a base checkpoint.
        prompt_instruct: The text prepended to an instruction by the
            `prompt` arm of a post-trained checkpoint.
        n_boot: Bootstrap resamples of the held-out intervals.
    """

    experiment: str = "b3"
    model: str = "gpt2"
    max_new_tokens: int = 128
    num_samples: int = 1
    coeffs: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)
    concepts: tuple[str, ...] = ("sentiment", "happiness", "anger")
    n_tune: int = 40
    n_held_out: int = 40
    method: Method = "logistic"
    n_pairs: int = 64
    min_words: int = 3
    n_random: int = 1
    truncate_words: int = 4
    caa_question: str = "Which of the two texts expresses {description}?"
    prompt_base: str = "The following text expresses {description}. "
    prompt_instruct: str = (
        "Answer in a way that clearly expresses {description}. "
    )
    n_boot: int = 2000


def caa_vectors(
    loaded: LoadedModel,
    positive: Sequence[str],
    negative: Sequence[str],
    *,
    question: str,
    layers: Sequence[int],
    bos: BosMode | None = None,
    batch_size: int = 8,
) -> NDArray[np.float32]:
    """Builds the vector of contrastive activation addition.

    Every pair of texts becomes a two-option question (`choice_pairs`),
    rendered with the chat format of the checkpoint and answered once with
    the letter of the positive text and once with the other letter. The
    vector is the mean difference of the activations at the answer letter.

    Args:
        loaded: The wrapped model.
        positive: Texts that show the concept.
        negative: One text without the concept per positive text.
        question: The question asked about every pair.
        layers: Layers in 0..L.
        bos: BOS policy; None uses the model default.
        batch_size: Texts per forward pass.

    Returns:
        A `(K, d_model)` float32 array with the raw magnitude.

    Raises:
        ValueError: If `choice_pairs` rejects the texts or the question.
    """
    chat = chat_format_for(loaded)
    texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for pair in choice_pairs(positive, negative, question=question):
        for answer in (pair.positive, pair.negative):
            text, _, start = chat.render_turn(pair.question, answer)
            texts.append(text)
            spans.append((start + len(answer) - 1, start + len(answer)))
    data = collect_activations(
        loaded,
        texts,
        layers=layers,
        position="concept",
        concept_spans=spans,
        bos=bos,
        batch_size=batch_size,
    ).data
    return mean_shift(data[:, 0::2], data[:, 1::2])


def mean_centring_vectors(
    loaded: LoadedModel,
    target: Sequence[str],
    generic: Sequence[str],
    *,
    layers: Sequence[int],
    bos: BosMode | None = None,
    batch_size: int = 8,
) -> NDArray[np.float32]:
    """Builds the mean-centring vector.

    The activations of each text are averaged over its tokens; the vector is
    the mean over the target texts minus the mean over a generic corpus.

    Args:
        loaded: The wrapped model.
        target: Texts that show the concept.
        generic: Texts of a generic corpus.
        layers: Layers in 0..L.
        bos: BOS policy; None uses the model default.
        batch_size: Texts per forward pass.

    Returns:
        A `(K, d_model)` float32 array with the raw magnitude.

    Raises:
        ValueError: If a set is empty.
    """

    def pooled(texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            raise ValueError("mean-centring needs texts in both sets")
        return collect_activations(
            loaded,
            texts,
            layers=layers,
            position="mean",
            response_spans=[(0, len(text)) for text in texts],
            bos=bos,
            batch_size=batch_size,
        ).data

    return mean_shift(pooled(target), pooled(generic))


def user_text_start(
    loaded: LoadedModel,
    chat: ChatFormat,
    instructions: Sequence[str],
    *,
    bos: BosMode | None = None,
) -> int:
    """Returns where an activation addition starts in a rendered prompt.

    The published method adds the difference at the front of the prompt
    text. A prompt rendered with a chat format opens with a header, so the
    front of its text is the first token of the user text, not token 0,
    which belongs to the header.

    Args:
        loaded: The wrapped model.
        chat: The chat format the prompts are rendered with.
        instructions: The user texts of the prompts.
        bos: BOS policy of the run.

    Returns:
        The `start` of a `PromptDifference` built under the same BOS policy:
        the token index of the first token that overlaps the user text,
        minus one under a BOS, because row 0 of such a difference is the
        zero row of the BOS and row 1 holds the first token of its prompt.

    Raises:
        ValueError: If there is no prompt, or the user text does not start
            at the same token index in every prompt (a first word can merge
            with the end of the header), so that one start cannot serve
            them all.
    """
    starts: dict[int, str] = {}
    for instruction in instructions:
        text, user_start, _ = chat.render_turn(instruction)
        batch = loaded.encode([text], bos=bos)
        first = next(
            index
            for index, (begin, end) in enumerate(batch.offsets[0])
            if end > max(begin, user_start)
        )
        starts.setdefault(first - int(batch.has_bos), instruction)
    if not starts:
        raise ValueError("no prompt to locate the user text in")
    if len(starts) > 1:
        raise ValueError(
            "the user text starts at different token indices "
            f"({sorted(starts)}) in the rendered prompts, for example in "
            f"{list(starts.values())[:2]}, so an activation addition has no "
            "single start"
        )
    return next(iter(starts))


def _check_config(config: Config) -> list[Concept]:
    """Checks the settings that need no data and returns the concepts."""
    names = config.concepts
    if (
        not names
        or len(set(names)) != len(names)
        or not set(names) <= set(CONCEPTS)
    ):
        raise ValueError(
            f"concepts must be distinct names of {tuple(CONCEPTS)}, got {names}"
        )
    if config.mode != "relative":
        raise ValueError(
            "the coefficient grid is relative to the residual norm, got "
            f"mode {config.mode!r}"
        )
    grid = config.coeffs
    if not grid or grid[0] <= 0 or list(grid) != sorted(set(grid)):
        raise ValueError(
            f"coeffs must be positive and strictly increasing, got {grid}"
        )
    for name in ("n_tune", "n_held_out", "n_random"):
        if getattr(config, name) < 1:
            raise ValueError(
                f"{name} must be at least 1, got {getattr(config, name)}"
            )
    return [CONCEPTS[name] for name in names]


def _train_examples(
    config: Config, concept: Concept, examples: Mapping[str, Sequence[Example]]
) -> list[Example]:
    """Returns the training examples a concept is built from."""
    if concept.source not in examples:
        raise ValueError(
            f"concept {concept.name!r} needs the {concept.source!r} examples"
        )
    train = [e for e in examples[concept.source] if e.split == "train"]
    if concept.source == "sst2":
        train = [e for e in train if len(e.text.split()) >= config.min_words]
    shown = sum(e.label == concept.target for e in train)
    check_pair_count(
        "n_pairs",
        config.n_pairs,
        min(shown, len(train) - shown),
        f"for {concept.name!r} in the train split",
    )
    return train


def _stimuli(
    concept: Concept,
    train: Sequence[Example],
    chat: ChatFormat,
    *,
    n_pairs: int,
    seed: int,
) -> StimulusSet:
    """Pairs the examples of a concept as C1 and C2 do."""
    if concept.source == "emotions":
        return emotion_stimuli(
            train, concept.name, chat, n_pairs=n_pairs, seed=seed
        )
    return build_stimulus_set(
        train,
        PAPER_TEMPLATES["concept"],
        chat,
        pairing="label_contrastive",
        concept=concept.name,
        n_pairs=n_pairs,
        seed=seed,
    )


def _pair_texts(
    stimuli: StimulusSet, train: Sequence[Example]
) -> tuple[list[str], list[str]]:
    """Returns the raw positive and negative text of every reading pair."""
    raw = {example.uid: example.text for example in train}
    positive: list[str] = []
    negative: list[str] = []
    for first, second in stimuli.pairs.tolist():
        shown, other = (
            (first, second) if stimuli.labels[first] == 1 else (second, first)
        )
        positive.append(raw[stimuli.uids[shown]])
        negative.append(raw[stimuli.uids[other]])
    return positive, negative


def _draw(
    config: Config,
    held: HeldOut,
    instructions: Sequence[str] | None,
    post_trained: bool,
) -> dict[str, list[str]]:
    """Draws the two disjoint prompt halves, as a judge reads them."""
    if post_trained:
        if not instructions:
            raise ValueError(
                "a post-trained checkpoint needs instructions as prompts"
            )
        pool = list(dict.fromkeys(instructions))
    else:
        pool = prompt_fragments(held.neutral, config.truncate_words)
    (drawn,) = draw_prompts(
        pool,
        config.n_tune + config.n_held_out,
        sets=1,
        seed=derive_seed(config.seed, "prompts"),
    )
    return {TUNE: drawn[: config.n_tune], HELD_OUT: drawn[config.n_tune :]}


def _memoized(arm: Arm) -> Arm:
    """Makes an arm generate once per prompt list."""
    seen: dict[tuple[str, ...], list[list[Completion]]] = {}

    def generate(prompts: Sequence[str]) -> list[list[Completion]]:
        key = tuple(prompts)
        if key not in seen:
            seen[key] = arm.generate(prompts)
        return seen[key]

    return dataclasses.replace(arm, generate=generate)


@dataclass(frozen=True)
class _Setup:
    """What every concept of a run shares."""

    config: Config
    loaded: LoadedModel
    settings: GenSettings
    chat: ChatFormat
    post_trained: bool
    layers: tuple[int, ...]
    norms: Mapping[int, float]
    actadd_start: int
    held: HeldOut
    shown: Mapping[str, list[str]]
    prompts: Mapping[str, list[str]]
    plain: Arm


@dataclass(frozen=True)
class _Built:
    """The arms of one concept and the vectors behind them."""

    arms: list[tuple[str, str, float | None, Arm]]
    vectors: dict[str, NDArray[np.float32]]
    raw_norms: dict[str, list[float]]


def _prompt_arm(setup: _Setup, concept: Concept) -> Arm:
    """Builds the arm that prepends an instruction to every prompt."""
    config = setup.config
    if not setup.post_trained:
        return plain_arm(
            setup.loaded,
            setup.settings,
            name="prompt",
            prefix=config.prompt_base.format(description=concept.description),
        )
    prefix = config.prompt_instruct.format(description=concept.description)
    rendered = {
        prompt: setup.chat.render_turn(prefix + instruction)[0]
        for split in STAGES
        for instruction, prompt in zip(
            setup.shown[split], setup.prompts[split], strict=True
        )
    }

    def generate(prompts: Sequence[str]) -> list[list[Completion]]:
        return setup.plain.generate([rendered[prompt] for prompt in prompts])

    return Arm(name="prompt", generate=generate)


def _build(setup: _Setup, concept: Concept, train: Sequence[Example]) -> _Built:
    """Builds every vector and arm of one concept from the same pairs."""
    config, loaded, layers = setup.config, setup.loaded, setup.layers
    stimuli = _stimuli(
        concept,
        train,
        setup.chat,
        n_pairs=config.n_pairs,
        seed=derive_seed(config.seed, "stimuli", concept.name),
    )
    act = stimulus_activations(
        loaded,
        stimuli,
        layers=layers,
        position="last",
        bos=config.bos,
        batch_size=config.batch_size,
    )

    def fit(method: Method) -> RepReader:
        return RepReader.fit(
            method,
            act.data,
            act.layers,
            labels=act.row_values(stimuli.labels),
            pairs=act.row_groups(stimuli.pairs),
            groups=act.row_groups(stimuli.groups),
            seed=derive_seed(config.seed, "fit", concept.name),
            refit_at_zero=True,
        )

    positive, negative = _pair_texts(stimuli, train)
    raw = {
        "caa": caa_vectors(
            loaded,
            positive,
            negative,
            question=config.caa_question.format(
                description=concept.description
            ),
            layers=layers,
            bos=config.bos,
            batch_size=config.batch_size,
        ),
        "mean_centring": mean_centring_vectors(
            loaded,
            positive,
            setup.held.calibration,
            layers=layers,
            bos=config.bos,
            batch_size=config.batch_size,
        ),
    }
    zeros = np.zeros((len(layers), loaded.d_model))
    readers = {
        **{
            arm: RepReader.from_arrays(arm, layers, raw[arm], zeros)
            for arm in _VECTOR_ARMS
        },
        "diffmean": fit("mean_diff"),
        "reader": fit(config.method),
    }
    randoms = random_readers(
        layers,
        loaded.d_model,
        config.n_random,
        seed=derive_seed(config.seed, "random", concept.name),
    )
    random_arms = [f"random/{i}" for i in range(config.n_random)]
    readers.update(zip(random_arms, randoms, strict=True))
    difference = PromptDifference.from_prompts(
        loaded, *concept.pair, layers, coeff=1.0, bos=config.bos
    )
    matched = difference.norm_matched(setup.norms).at(setup.actadd_start)

    def steer(arm: str, coeff: float) -> Arm:
        name = f"{arm}/{coeff:g}"
        if arm == _ACTADD:
            return edit_arm(
                loaded,
                setup.settings,
                dataclasses.replace(matched, coeff=coeff),
                name=name,
            )
        return reader_arm(
            loaded,
            setup.settings,
            readers[arm],
            layers=layers,
            coeff=coeff,
            norms=setup.norms,
            op=config.op,
            mode="relative",
            mask=config.mask,
            preserve_norm=config.preserve_norm,
            name=name,
        )

    order = [*_VECTOR_ARMS, _ACTADD, "diffmean", "reader", *random_arms]
    arms: list[tuple[str, str, float | None, Arm]] = [
        ("none", "none", None, setup.plain),
        ("prompt", "prompt", None, _prompt_arm(setup, concept)),
    ]
    arms.extend(
        (f"{arm}/{coeff:g}", arm, coeff, steer(arm, coeff))
        for arm in order
        for coeff in config.coeffs
    )
    raw_norms = {
        arm: np.linalg.norm(raw[arm], axis=1).tolist() for arm in _VECTOR_ARMS
    }
    raw_norms[_ACTADD] = [
        float(rows[rows > 0].mean())
        for rows in (difference.vectors[layer].norm(dim=-1) for layer in layers)
    ]
    return _Built(
        arms=arms,
        vectors={
            arm.replace("/", "_"): reader.signs[:, None] * reader.directions
            for arm, reader in readers.items()
        },
        raw_norms=raw_norms,
    )


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    examples: Mapping[str, Sequence[Example]],
    texts: Sequence[str],
    instructions: Sequence[str] | None,
    classify: Mapping[str, Scorer],
) -> None:
    """Generates with every arm on the tuning and the held-out prompts.

    A base checkpoint continues the first words of held-out texts; a
    post-trained checkpoint answers instructions rendered with its chat
    template. For each concept, four vectors are built from the same
    training pairs: the two-option prompts of contrastive activation
    addition, the mean-centring vector (the positive texts of the pairs
    against the calibration texts), and the difference of class means and
    our reading vector on the stimuli of C1 and C2. The activation addition
    uses no training pair: it is the difference of the concept's prompt
    pair, added at the front of the prompt text, which on a prompt rendered
    with a chat template is the first token of the user text
    (`user_text_start`), not the chat header. Each additive arm steers the
    same layers at every coefficient of the grid, relative to the mean
    residual norm of the calibration texts. The `prompt` arm prepends an
    instruction, `none` does nothing, and the random arms steer along
    random unit directions drawn from the seed alone.

    Writes `prompts.json` (the two halves as a judge reads them),
    `trials_<concept>_<split>` (every generation with the classifier score
    of the concept as a cross-check; trials are named `none`, `prompt` and
    `<arm>/<coefficient>`), `vectors.npz` (the oriented unit direction of
    every vector arm per layer), `table_effects.json` (perplexity of neutral
    text, fluency, distinct-1, distinct-2, mean length and empty rate of
    every arm and coefficient on the held-out half) and `summary.json`.
    The judged tables are written later by `judge_report`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        examples: The labeled examples of each source a concept needs,
            `sst2` (label 1 is positive) or `emotions` (labeled with the
            index in `EMOTIONS`), with a `train` split.
        texts: Held-out texts: the calibration slice sets the relative
            coefficients and is the generic corpus of mean-centring, the
            neutral slice gives the perplexity and the prompts of a base
            checkpoint.
        instructions: The prompts of a post-trained checkpoint; unused for
            a base checkpoint.
        classify: Maps each concept to a scorer of shape `(n,)`, the
            classifier cross-check.

    Raises:
        ValueError: If the concepts are not distinct keys of `CONCEPTS`,
            `mode` is not `relative`, `coeffs` is not positive and strictly
            increasing, `n_tune`, `n_held_out` or `n_random` is below 1, the
            generation settings are invalid (see `GenSettings`), the
            held-out texts or the prompts do not suffice, a post-trained
            checkpoint gets no instructions, a concept lacks its examples,
            its classifier or `n_pairs` pairs, or the user text of the
            rendered prompts does not start at one token index. All of
            these are raised before anything is written.
    """
    concepts = _check_config(config)
    settings = GenSettings.from_config(config)
    post_trained = (
        loaded.checkpoint is not None and loaded.checkpoint.post_trained
    )
    chat = chat_format_for(loaded)
    layers = config.layers or default_control_layers(loaded.n_layers)
    held = split_held_out(
        texts,
        n_calibration=config.n_calibration,
        n_neutral=config.n_neutral,
        seed=config.seed,
    )
    shown = _draw(config, held, instructions, post_trained)
    train = {c.name: _train_examples(config, c, examples) for c in concepts}
    missing = [c.name for c in concepts if c.name not in classify]
    if missing:
        raise ValueError(f"no classifier for the concepts {missing}")
    setup = _Setup(
        config=config,
        loaded=loaded,
        settings=settings,
        chat=chat,
        post_trained=post_trained,
        layers=tuple(layers),
        norms=mean_residual_norms(
            loaded,
            held.calibration,
            layers,
            bos=config.bos,
            batch_size=config.batch_size,
        ),
        actadd_start=(
            user_text_start(
                loaded,
                chat,
                [prompt for half in shown.values() for prompt in half],
                bos=config.bos,
            )
            if post_trained
            else 0
        ),
        held=held,
        shown=shown,
        prompts={
            split: (
                [chat.render_turn(prompt)[0] for prompt in half]
                if post_trained
                else half
            )
            for split, half in shown.items()
        },
        plain=_memoized(plain_arm(loaded, settings)),
    )
    vectors: dict[str, NDArray[np.float32]] = {}
    raw_norms: dict[str, dict[str, list[float]]] = {}
    effects: list[list[object]] = []
    names: list[list[object]] = []
    for concept in concepts:
        built = _build(setup, concept, train[concept.name])
        names = [[name, arm, coeff] for name, arm, coeff, _ in built.arms]
        raw_norms[concept.name] = built.raw_norms
        for arm_name, vector in built.vectors.items():
            vectors[f"{concept.name}__{arm_name}"] = vector.astype(np.float32)
        for split in STAGES:
            trials = [
                run_trial(
                    arm, setup.prompts[split], classify[concept.name], name=name
                )
                for name, _, _, arm in built.arms
            ]
            write_trials(run_dir, f"trials_{concept.name}_{split}", trials)
            if split != HELD_OUT:
                continue
            effects.extend(
                effects_row(
                    side_effects(
                        loaded,
                        arm,
                        trial,
                        held.neutral,
                        batch_size=config.batch_size,
                    ),
                    concept.name,
                    label,
                    coeff,
                )
                for (_, label, coeff, arm), trial in zip(
                    built.arms, trials, strict=True
                )
            )
    run_dir.write_arrays("vectors", **vectors)
    run_dir.write_json(
        "table_effects", {"columns": list(_EFFECT_COLUMNS), "rows": effects}
    )
    run_dir.write_json("prompts", dict(shown))
    run_dir.write_json(
        "summary",
        {
            "role": "post_trained" if post_trained else "base",
            "criteria": list(role_criteria(post_trained=post_trained)),
            "layers": list(layers),
            "norms": {str(layer): norm for layer, norm in setup.norms.items()},
            "concepts": {c.name: c.description for c in concepts},
            "arms": names,
            "raw_norms": raw_norms,
            "actadd_start": setup.actadd_start,
            "n_tune": config.n_tune,
            "n_held_out": config.n_held_out,
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the model, the data of its role and the classifiers, then runs.

    A post-trained checkpoint also loads the instructions; a base checkpoint
    never does. Only the sources and classifiers of the configured concepts
    are loaded.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.

    Raises:
        ValueError: As `execute`. The settings that need no data are checked
            before anything is loaded.
    """
    concepts = _check_config(config)
    loaded = load_model(config.model)
    sources = {concept.source for concept in concepts}
    examples: dict[str, Sequence[Example]] = {}
    if "sst2" in sources:
        examples["sst2"] = load_sst2()
    if "emotions" in sources:
        examples["emotions"] = load_emotions()
    texts = [example.text for example in load_wikitext()]
    instructions = None
    if loaded.checkpoint is not None and loaded.checkpoint.post_trained:
        instructions = [example.text for example in load_alpaca_eval()]
    judges = {
        key: load_judge(key) for key in dict.fromkeys(c.judge for c in concepts)
    }

    def scorer(concept: Concept) -> Scorer:
        def score(batch: Sequence[str]) -> NDArray[np.float32]:
            scores = judges[concept.judge].score(batch)
            return (
                scores if concept.column is None else scores[:, concept.column]
            )

        return score

    execute(
        config,
        run_dir,
        loaded,
        examples,
        texts,
        instructions,
        {concept.name: scorer(concept) for concept in concepts},
    )


def _stored(run_dir: RunDir, stem: str) -> dict[str, Any]:
    payload = run_dir.read_json(stem)
    if not isinstance(payload, dict):
        raise ValueError(f"{stem}.json must hold an object")
    return payload


def judged_generations(run_dir: RunDir) -> Judged:
    """Lists every stored generation of a run for the judging steps.

    Args:
        run_dir: A B3 run.

    Returns:
        One record per stored completion, keyed
        `<concept>/<split>/<trial>/<prompt>/<sample>`, with the prompt as a
        judge reads it and the criteria of the role of the checkpoint.
    """
    summary = _stored(run_dir, "summary")
    prompts = _stored(run_dir, "prompts")
    arms = {name: (arm, coeff) for name, arm, coeff in summary["arms"]}
    records = [
        Record(
            key=f"{concept}/{split}/{trial.name}/{p}/{s}",
            concept=concept,
            description=summary["concepts"][concept],
            arm=arms[trial.name][0],
            coeff=arms[trial.name][1],
            split=split,
            prompt_index=p,
            prompt=prompts[split][p],
            text=text,
        )
        for concept in run_dir.config["concepts"]
        for split in STAGES
        for trial in read_trials(run_dir, f"trials_{concept}_{split}")
        for p, row in enumerate(trial.completions)
        for s, text in enumerate(row)
    ]
    return Judged(criteria=tuple(summary["criteria"]), records=tuple(records))


def _classifier_scores(run_dir: RunDir) -> dict[str, float]:
    """Maps the key of every stored generation to its classifier score."""
    return {
        f"{concept}/{split}/{trial.name}/{p}/{s}": float(trial.scores[p, s])
        for concept in run_dir.config["concepts"]
        for split in STAGES
        for trial in read_trials(run_dir, f"trials_{concept}_{split}")
        for p in range(trial.scores.shape[0])
        for s in range(trial.scores.shape[1])
    }


def _rank_correlation(first: Sequence[float], second: Sequence[float]) -> Any:
    """Gives Spearman's rho, or None when it is undefined."""
    if len(first) < 3 or len(set(first)) < 2 or len(set(second)) < 2:
        return None
    return float(spearmanr(first, second).statistic)


def judge_report(run_dir: RunDir) -> list[Path]:
    """Writes the judged tables of a run.

    The coefficient of every arm is the one with the highest mean overall
    score on the tuning half (`SELECTION_RULE`); the held-out half is
    reported at that coefficient only and never enters the choice. An empty
    generation is never judged: it scores 0 on every criterion and is
    counted in `n_empty`.

    Writes `table.json` (per concept and arm: the chosen coefficient, the
    numbers of held-out generations and of empty ones, the mean score per
    criterion and the overall score with a bootstrap interval over prompts
    on the held-out half, beside the perplexity, the distinct-1 and the
    mean classifier score of the same generations), `table_tuning.json`
    (the mean overall score and the number of empty generations of every
    arm and coefficient on the tuning half) and `summary_judged.json` (the
    chosen coefficients and the rule that chose them, the agreement of the
    two judges per criterion, pooled and per stage, with the agreement
    rate, Cohen's kappa and the quadratic-weighted kappa, the judge model
    ids, the rubric digest of each stage, and per concept the rank
    correlation of the judged concept score with the classifier score).

    Args:
        run_dir: A B3 run whose two stages are exported and imported.

    Returns:
        The three paths, in that order.

    Raises:
        ValueError: If a stage has no export or a batch is not imported.
            Nothing is written then.
    """
    judged = judged_generations(run_dir)
    scores = judge_scores(run_dir, judged)
    selected = select_coefficients(judged, scores)
    rows = held_out_rows(
        judged,
        scores,
        selected,
        n_boot=int(run_dir.config["n_boot"]),
        seed=int(run_dir.config["seed"]),
    )
    agreed = {
        "pooled": agreement(run_dir, judged),
        **{
            stage: agreement(run_dir, judged, stages=(stage,))
            for stage in STAGES
        },
    }
    tuning = tuning_rows(judged, scores)
    classifier = _classifier_scores(run_dir)
    side = {
        (row[0], row[1], row[2]): row
        for row in _stored(run_dir, "table_effects")["rows"]
    }
    table_rows = []
    for row in rows:
        cell = (row["concept"], row["arm"], row["coeff"])
        seen = [
            classifier[r.key]
            for r in judged.records
            if (r.concept, r.arm, r.coeff) == cell and r.split == HELD_OUT
        ]
        finite = [value for value in seen if np.isfinite(value)]
        table_rows.append(
            [
                *row.values(),
                side[cell][3],
                side[cell][5],
                float(np.mean(finite)) if finite else None,
            ]
        )
    correlation = {}
    for concept in run_dir.config["concepts"]:
        pairs = [
            (float(scores[r.key][0]), classifier[r.key])
            for r in judged.records
            if r.concept == concept
            and r.key in scores
            and np.isfinite(classifier[r.key])
        ]
        correlation[concept] = _rank_correlation(
            [judge for judge, _ in pairs], [other for _, other in pairs]
        )
    plans = [plan for stage in STAGES if (plan := read_export(run_dir, stage))]
    models = {
        _stored(run_dir, f"judgments_{batch.batch_id}")["judge_model"]
        for plan in plans
        for batch in plan.batches
    }
    return [
        run_dir.write_json(
            "table",
            {
                "columns": [*rows[0], "perplexity", "distinct_1", "classifier"],
                "rows": table_rows,
            },
        ),
        run_dir.write_json(
            "table_tuning",
            {
                "columns": list(tuning[0]),
                "rows": [list(row.values()) for row in tuning],
            },
        ),
        run_dir.write_json(
            "summary_judged",
            {
                "selected": [
                    [concept, arm, coeff]
                    for (concept, arm), coeff in selected.items()
                ],
                "selection_rule": SELECTION_RULE,
                "agreement": agreed,
                "judge_models": sorted(models),
                "rubric_sha256": {
                    plan.stage: plan.rubric_sha256 for plan in plans
                },
                "classifier_correlation": correlation,
            },
        ),
    ]


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws nothing: the output of B3 is its tables.

    Args:
        run_dir: A B3 run.
        out_dir: Folder of the figures.

    Returns:
        An empty list.
    """
    del run_dir, out_dir
    return []
