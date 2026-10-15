"""R3: reading morality from GPT-2 on ETHICS commonsense scenarios."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tomography.datasets.ethics import load_ethics_morality
from tomography.datasets.records import Example
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
from tomography.reading.directions import Method
from tomography.reading.metrics import best_layer
from tomography.templates import PAPER_TEMPLATES


@dataclass(frozen=True, kw_only=True)
class Config(ReadingConfig):
    """R3 settings: GPT-2 morality on ETHICS commonsense scenarios.

    The shared defaults apply except for `method` and `short_only`.
    `short_only` exists because the long ETHICS posts reach 2,857 GPT-2
    tokens (median 415, 236 over 900) against the 1,024 positions of GPT-2,
    while the short scenarios have a median of 13 and a maximum of 61 tokens.
    `method` is `logistic` instead of the paper's `pca_diff` because on short
    scenarios the principal component of the pair differences followed content
    and not morality (test accuracy 0.536 with 32 pairs and 0.500 with 128
    pairs, chance, against 0.659 and 0.732 for `logistic` with 3 seeds).

    Attributes:
        match_length: Pair scenarios only inside the same length class
            (`meta["is_short"]`), so length cannot separate the pair members.
            It only has an effect when `short_only` is off, because with only
            short scenarios there is a single length class.
        short_only: Keep only the short scenarios (`meta["is_short"] == 1`)
            before the split into train, val and test and before pairing.
            Examples without `meta["is_short"]` raise `KeyError` when it is on.
    """

    experiment: str = "r3"
    model: str = "gpt2"
    match_length: bool = True
    short_only: bool = True
    method: Method = "logistic"


def _stimuli(
    examples: Sequence[Example],
    loaded: LoadedModel,
    n_pairs: int | None,
    seed: int,
    match_length: bool,
) -> StimulusSet:
    strata = (
        [str(e.meta["is_short"]) for e in examples] if match_length else None
    )
    return build_stimulus_set(
        examples,
        PAPER_TEMPLATES["morality"],
        chat_format_for(loaded),
        pairing="label_contrastive",
        strata=strata,
        n_pairs=n_pairs,
        seed=seed,
    )


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    examples: Sequence[Example],
) -> None:
    """Reads morality on ETHICS train scenarios and scores val and test.

    Writes `curves` (the `reader`, `random` and `shuffled` series on the test
    split), `curves_val` (the same on the val split), `readers.npz` (the
    fitted directions and signs, one per seed) and `summary.json`. The layer
    is chosen on val, never on test. With `short_only` on only the short
    scenarios are used; otherwise, with `match_length` on, pairs never cross a
    length class.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        examples: ETHICS commonsense scenarios with `train`, `val` and `test`
            splits (label 1 = acceptable). With `short_only` or
            `match_length` on they need `meta["is_short"]`.

    Raises:
        KeyError: An example has no `meta["is_short"]` and `short_only` is
            on, or `short_only` is off and `match_length` is on.
    """
    if config.short_only:
        examples = [e for e in examples if e.meta["is_short"] == 1]
    by_split = {
        split: [e for e in examples if e.split == split]
        for split in ("train", "val", "test")
    }
    train_sets = [
        _stimuli(
            by_split["train"],
            loaded,
            config.n_pairs,
            derive_seed(config.seed, "train", i),
            config.match_length,
        )
        for i in range(config.n_seeds)
    ]
    eval_sets = {
        split: _stimuli(
            by_split[split],
            loaded,
            config.n_eval_pairs,
            derive_seed(config.seed, "eval", split),
            config.match_length,
        )
        for split in ("val", "test")
    }
    result = reading_sweep(
        loaded,
        train_sets,
        eval_sets,
        method=config.method,
        preprocess=config.preprocess,
        position=config.position,
        eval_position=config.eval_position,
        layers=config.layers,
        seed=config.seed,
        bos=config.bos,
        batch_size=config.batch_size,
    )
    for split, stem in (("test", "curves"), ("val", "curves_val")):
        write_curves(
            run_dir,
            result.per_seed(split),
            result.layers,
            series=SERIES,
            stem=stem,
            chance=0.5,
            n_layers=loaded.n_layers,
        )
    run_dir.write_arrays(
        "readers",
        layers=np.asarray(result.layers, dtype=np.int64),
        directions=np.stack([r.directions for r in result.readers]),
        signs=np.stack([r.signs for r in result.readers]),
    )
    index = SERIES.index("reader")
    test = run_dir.read_arrays("curves")
    val = run_dir.read_arrays("curves_val")
    best = best_layer(val["mean"][index], result.layers)
    at = result.layers.index(best)
    run_dir.write_json(
        "summary",
        {
            "best_layer": best,
            "val_accuracy": float(val["mean"][index, at]),
            "test_accuracy": float(test["mean"][index, at]),
            "ci": [
                float(test["lo"][index, at]),
                float(test["hi"][index, at]),
            ],
            "n_seeds": config.n_seeds,
            "n_pairs": config.n_pairs,
            "n_eval_pairs": {
                split: int(stimuli.groups.shape[0])
                for split, stimuli in eval_sets.items()
            },
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads GPT-2 and ETHICS commonsense, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    execute(config, run_dir, load_model(config.model), load_ethics_morality())


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the R3 layer curves of every evaluation split.

    Args:
        run_dir: A finished R3 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of every `curves*` file, sorted by stem.
    """
    stems = sorted(p.stem for p in run_dir.path.glob("curves*.npz"))
    paths: list[Path] = []
    for stem in stems:
        paths.extend(plot_curves_file(run_dir, out_dir, stem=stem))
    return paths
