"""P9a: labeling known and unknown questions of one model.

Builds the pool of questions, labels each by what the model does on it
(closed book, few-shot, one greedy answer and several temperature samples),
demotes the known labels that a constant guess would explain, estimates the
training cutoff from the award questions and writes everything the later
P9 experiments read: the labeled items, the label tables, the guess control
and the gate that admits a model to the known versus unknown contrasts.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np

from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.intervals import write_interval_curves
from tomography.experiments.sweep import plot_curves_file
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import LabelRecipe
from tomography.knowledge import label_questions
from tomography.knowledge import write_hand_check
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.p9.guess import StratumGuess
from tomography.p9.guess import apply_guess_control
from tomography.p9.guess import stratum_guesses
from tomography.p9.pool import GROUPS
from tomography.p9.pool import LabeledItem
from tomography.p9.pool import PoolLoaders
from tomography.p9.pool import PoolSizes
from tomography.p9.pool import assign_award_kinds
from tomography.p9.pool import build_pool
from tomography.p9.pool import name_token_counter
from tomography.p9.pool import pool_from_records
from tomography.p9.pool import pool_records
from tomography.p9.probe import CutoffEstimate
from tomography.p9.probe import YearRate
from tomography.p9.probe import estimate_cutoff
from tomography.p9.probe import label_table
from tomography.p9.probe import popularity_table
from tomography.p9.probe import year_rates
from tomography.reading.metrics import wilson_ci

_COUNT_COLUMNS = ["n", "known_raw", "known", "unknown", "uncertain"]


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """P9a settings: the pool, the label rule and the guess control.

    Attributes:
        cutoff_year: Last year of the model's training data; None takes the
            probe estimate.
        n_popqa: PopQA questions per relation; None keeps all.
        n_synthal: SyntHal questions per relation; None keeps all.
        n_kuq: KUQ questions per category; None keeps all.
        n_falseqa: FalseQA questions; None keeps all.
        n_triviaqa: TriviaQA questions; None keeps all.
        n_nq_open: NQ-open questions; None keeps all.
        n_detail: Unknowable-detail questions.
        n_shots: How many of `knowledge.DEFAULT_SHOTS` open the prompt (the
            tiny test models have 128 positions and use 2).
        n_samples: Temperature samples per question.
        temperature: Sampling temperature.
        known_min_correct: Samples that must be correct, besides the greedy
            answer, for "known" (3 of 5; 4 is the robustness setting).
        max_new_tokens: Generation budget per answer.
        n_hand_check: Rows of the hand-check file.
        max_guess_accuracy: A stratum whose constant guess is right more
            often than this supplies no known items.
        never_known: PopQA relations that never supply known items.
        min_known: Guess-adjusted known items a model needs to enter the
            known versus unknown contrasts.
        popular_min: A PopQA question is popular when its popularity is at
            least this.
        popular_share: Share of a capped PopQA relation drawn from the
            popular questions first.
        batch_size: Prompts per generation call chunk.
        offline: Whether the datasets are read from the cache only.
    """

    experiment: str = "p9a"
    model: str = "gpt2"
    cutoff_year: int | None = 2017
    n_popqa: int | None = None
    n_synthal: int | None = None
    n_kuq: int | None = 200
    n_falseqa: int | None = 500
    n_triviaqa: int | None = 1000
    n_nq_open: int | None = 1000
    n_detail: int = 300
    n_shots: int = 15
    n_samples: int = 5
    temperature: float = 1.0
    known_min_correct: int = 3
    max_new_tokens: int = 16
    n_hand_check: int = 200
    max_guess_accuracy: float = 0.10
    never_known: tuple[str, ...] = ("sport",)
    min_known: int = 100
    popular_min: int = 10_000
    popular_share: float = 0.5
    batch_size: int = 8
    offline: bool = False


def _estimate(rates: list[YearRate]) -> CutoffEstimate | None:
    """The probe estimate, or None when the known rate shows no drop."""
    try:
        return estimate_cutoff(rates)
    except ValueError:
        return None


def _choose_cutoff(
    config: Config, estimate: CutoffEstimate | None
) -> tuple[int, str]:
    """The cutoff year and where it came from."""
    if config.cutoff_year is not None:
        return config.cutoff_year, "config"
    if estimate is not None:
        return estimate.year, "probe"
    raise ValueError("set cutoff_year: the probe found no drop")


def _write_tables(
    run_dir: RunDir,
    items: list[LabeledItem],
    guesses: dict[str, StratumGuess],
) -> None:
    run_dir.write_json(
        "table",
        {
            "columns": ["group", "kind", *_COUNT_COLUMNS],
            "rows": label_table(items),
        },
    )
    run_dir.write_json(
        "table_popularity",
        {
            "columns": ["relation", "popularity", *_COUNT_COLUMNS],
            "rows": popularity_table(items),
        },
    )
    run_dir.write_json(
        "table_guess",
        {
            "columns": [
                "stratum",
                "n",
                "modal",
                "modal_share",
                "guess_accuracy",
                "excluded",
                "eligible",
            ],
            "rows": [
                [
                    g.stratum,
                    g.n,
                    g.modal,
                    g.modal_share,
                    g.guess_accuracy,
                    g.excluded,
                    g.eligible,
                ]
                for g in guesses.values()
            ],
        },
    )


def _write_known_rate_curves(run_dir: RunDir, rates: list[YearRate]) -> None:
    bands = [wilson_ci(r.known, r.n) for r in rates]
    write_interval_curves(
        run_dir,
        [r.year for r in rates],
        np.array([[r.rate for r in rates]]),
        np.array([[lo for lo, _ in bands]]),
        np.array([[hi for _, hi in bands]]),
        series=["known rate"],
        xlabel="Year",
        ylabel="Known rate",
    )


def _counts(items: list[LabeledItem]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for group in GROUPS:
        members = [i for i in items if i.group == group]
        if members:
            counts[group] = {
                "known_raw": sum(m.label.label == "known" for m in members),
                **{
                    name: sum(m.contrast_label == name for m in members)
                    for name in ("known", "unknown", "uncertain")
                },
            }
    return counts


def _verdicts(items: list[LabeledItem]) -> dict[str, int]:
    verdicts: dict[str, int] = {}
    for item in items:
        if item.guess is not None:
            verdicts[item.guess] = verdicts.get(item.guess, 0) + 1
    return verdicts


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    loaders: PoolLoaders,
) -> None:
    """Labels the pool of one model and writes the files the study reads.

    Builds the pool, labels it in one `label_questions` call with the raw
    few-shot prompt (no chat template, also for post-trained models),
    applies the guess control, estimates the cutoff from the award
    questions and assigns the award kinds from it. Writes `items`,
    `table`, `table_popularity`, `table_guess`, `curves` (the known rate by
    year with Wilson bands, only when the pool has award questions),
    `hand_check.csv` (behavior-labeled items only) and `summary`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        loaders: The question sources.

    Raises:
        ValueError: If `config.cutoff_year` is None and the probe finds no
            drop in the known rate, or a generation argument is invalid.
    """
    sizes = PoolSizes(
        popqa_per_relation=config.n_popqa,
        synthal_per_relation=config.n_synthal,
        kuq_per_category=config.n_kuq,
        falseqa=config.n_falseqa,
        triviaqa=config.n_triviaqa,
        nq_open=config.n_nq_open,
        popular_min=config.popular_min,
        popular_share=config.popular_share,
    )
    pool = build_pool(
        loaders,
        sizes,
        count_tokens=name_token_counter(loaded),
        seed=config.seed,
    )
    recipe = LabelRecipe(
        n_samples=config.n_samples,
        temperature=config.temperature,
        known_min_correct=config.known_min_correct,
        max_new_tokens=config.max_new_tokens,
    )
    labels = label_questions(
        loaded,
        pool,
        recipe=recipe,
        shots=DEFAULT_SHOTS[: config.n_shots],
        seed=config.seed,
        batch_size=config.batch_size,
    )
    raw = [LabeledItem(e, lab) for e, lab in zip(pool, labels, strict=True)]
    guesses = stratum_guesses(
        raw,
        max_guess_accuracy=config.max_guess_accuracy,
        excluded=[f"popqa:{relation}" for relation in config.never_known],
    )
    controlled = apply_guess_control(raw, guesses)
    rates = year_rates(controlled)
    estimate = _estimate(rates)
    cutoff, source = _choose_cutoff(config, estimate)
    kinds = assign_award_kinds([i.example for i in controlled], cutoff)
    items = [
        dataclasses.replace(item, example=example)
        for item, example in zip(controlled, kinds, strict=True)
    ]
    run_dir.write_json("items", pool_records(items))
    _write_tables(run_dir, items, guesses)
    if rates:
        _write_known_rate_curves(run_dir, rates)
    behavior = [i for i in items if i.label.basis == "behavior"]
    n_hand_check = write_hand_check(
        run_dir.path / "hand_check.csv",
        [i.example for i in behavior],
        [i.label for i in behavior],
        n=config.n_hand_check,
        seed=config.seed,
    )
    n_known = sum(i.contrast_label == "known" for i in items)
    run_dir.write_json(
        "summary",
        {
            "cutoff": {
                "year": cutoff,
                "source": source,
                "estimate": None
                if estimate is None
                else dataclasses.asdict(estimate),
            },
            "n_items": len(items),
            "counts": _counts(items),
            "recipe": {**dataclasses.asdict(recipe), "n_shots": config.n_shots},
            "guess": {
                "n_known_raw": sum(i.label.label == "known" for i in items),
                "n_known": n_known,
                "verdicts": _verdicts(items),
                "max_guess_accuracy": config.max_guess_accuracy,
                "never_known": list(config.never_known),
            },
            "min_known": config.min_known,
            "known_contrasts": n_known >= config.min_known,
            "hand_check_rows": n_hand_check,
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the model and the datasets, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    loaders = PoolLoaders.default(
        offline=config.offline, seed=config.seed, n_detail=config.n_detail
    )
    execute(config, run_dir, load_model(config.model), loaders)


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the known rate by year.

    Args:
        run_dir: A finished P9a run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths, or an empty list when the pool had no
        award questions.
    """
    if not (run_dir.path / "curves.json").is_file():
        return []
    return plot_curves_file(run_dir, out_dir)


def load_labeled(
    run_path: str | Path, *, model: str | None = None
) -> tuple[list[LabeledItem], dict[str, Any]]:
    """Reads the labels of a finished P9a run.

    Args:
        run_path: The run directory.
        model: The model the caller is about to read; None skips the check.
            The known and unknown labels depend on the model that was asked,
            so another model's labels must not feed its study.

    Returns:
        The labeled items, with their guess verdicts, and the summary.

    Raises:
        ValueError: If the directory is not a run, the run is not
            finished (an unfinished labeling run must not feed a study), or
            `model` is given and is not the model of the run.
    """
    run_dir = RunDir.open(run_path)
    if run_dir.meta.get("finished_at") is None:
        raise ValueError(f"{run_path} is not a finished run")
    labeled_model = run_dir.config.get("model")
    if model is not None and labeled_model != model:
        raise ValueError(
            f"{run_path} labeled {labeled_model!r}, not {model!r}: the known "
            "and unknown labels depend on the model"
        )
    records = cast(list[dict[str, object]], run_dir.read_json("items"))
    summary = cast(dict[str, Any], run_dir.read_json("summary"))
    return pool_from_records(records), summary
