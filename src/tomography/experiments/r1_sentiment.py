"""R1: reading sentiment from GPT-2 on SST-2."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.datasets.sst2 import load_sst2
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import ReadingConfig
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import reading_sweep
from tomography.experiments.sweep import write_curves
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.metrics import best_layer
from tomography.templates import PAPER_TEMPLATES


@dataclass(frozen=True, kw_only=True)
class Config(ReadingConfig):
    """R1 settings: GPT-2 sentiment on SST-2 with the shared defaults.

    Attributes:
        min_words: SST-2 phrases with fewer words are dropped.
    """

    experiment: str = "r1"
    model: str = "gpt2"
    n_eval_pairs: int | None = 256
    min_words: int = 3


def select_examples(
    examples: Sequence[Example], *, split: Split, min_words: int = 1
) -> list[Example]:
    """Keeps the examples of one split that have enough words.

    Args:
        examples: The source examples.
        split: The split to keep.
        min_words: The minimum number of whitespace-separated words.

    Returns:
        The kept examples, in their original order.
    """
    return [
        e
        for e in examples
        if e.split == split and len(e.text.split()) >= min_words
    ]


def _stimuli(
    examples: Sequence[Example],
    loaded: LoadedModel,
    n_pairs: int | None,
    seed: int,
) -> StimulusSet:
    return build_stimulus_set(
        examples,
        PAPER_TEMPLATES["concept"],
        chat_format_for(loaded),
        pairing="label_contrastive",
        concept="sentiment",
        n_pairs=n_pairs,
        seed=seed,
    )


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    examples: Sequence[Example],
) -> None:
    """Reads sentiment on SST-2 train phrases and scores it on val sentences.

    Writes `curves` (`reader`, `random` and `shuffled` series on the val
    set), `readers.npz` (the fitted directions and signs, one per seed) and
    `summary.json`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        examples: SST-2 examples with `train` and `val` splits.
    """
    train = select_examples(examples, split="train", min_words=config.min_words)
    val = select_examples(examples, split="val", min_words=config.min_words)
    train_sets = [
        _stimuli(
            train, loaded, config.n_pairs, derive_seed(config.seed, "train", i)
        )
        for i in range(config.n_seeds)
    ]
    val_stimuli = _stimuli(
        val, loaded, config.n_eval_pairs, derive_seed(config.seed, "eval")
    )
    result = reading_sweep(
        loaded,
        train_sets,
        {"val": val_stimuli},
        method=config.method,
        preprocess=config.preprocess,
        position=config.position,
        eval_position=config.eval_position,
        layers=config.layers,
        seed=config.seed,
        bos=config.bos,
        batch_size=config.batch_size,
    )
    write_curves(
        run_dir,
        result.per_seed("val"),
        result.layers,
        series=SERIES,
        chance=0.5,
        n_layers=loaded.n_layers,
    )
    run_dir.write_arrays(
        "readers",
        layers=np.asarray(result.layers, dtype=np.int64),
        directions=np.stack([r.directions for r in result.readers]),
        signs=np.stack([r.signs for r in result.readers]),
    )
    curves = run_dir.read_arrays("curves")
    index = SERIES.index("reader")
    mean = curves["mean"][index]
    best = best_layer(mean, result.layers)
    at = result.layers.index(best)
    run_dir.write_json(
        "summary",
        {
            "best_layer": best,
            "best_accuracy": float(mean[at]),
            "ci": [
                float(curves["lo"][index, at]),
                float(curves["hi"][index, at]),
            ],
            "n_seeds": config.n_seeds,
            "n_pairs": config.n_pairs,
            "n_eval_pairs": int(val_stimuli.groups.shape[0]),
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads GPT-2 and SST-2, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    execute(config, run_dir, load_model(config.model), load_sst2())


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the R1 layer curves.

    Args:
        run_dir: A finished R1 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths.
    """
    return plot_curves_file(run_dir, out_dir)
