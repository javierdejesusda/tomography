"""R7: reading truthfulness from GPT-2 on facts and Geometry of Truth."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tomography.datasets.facts import load_facts
from tomography.datasets.geometry_of_truth import SUBSETS
from tomography.datasets.geometry_of_truth import load_geometry_of_truth
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
from tomography.reading.metrics import best_layer
from tomography.templates import PAPER_TEMPLATES


@dataclass(frozen=True, kw_only=True)
class Config(ReadingConfig):
    """R7 settings: GPT-2 truthfulness with the shared defaults.

    Attributes:
        concept: The concept word the `concept` template puts in the prompt.
        ood_subsets: The Geometry of Truth subsets scored out of
            distribution with the readers fitted on the facts.
    """

    experiment: str = "r7"
    model: str = "gpt2"
    concept: str = "truthfulness"
    ood_subsets: tuple[str, ...] = SUBSETS


def _stimuli(
    examples: Sequence[Example],
    loaded: LoadedModel,
    concept: str,
    n_pairs: int | None,
    seed: int,
) -> StimulusSet:
    return build_stimulus_set(
        examples,
        PAPER_TEMPLATES["concept"],
        chat_format_for(loaded),
        pairing="label_contrastive",
        concept=concept,
        n_pairs=n_pairs,
        seed=seed,
    )


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    facts: Sequence[Example],
    truth_sets: Sequence[Example],
) -> None:
    """Reads truthfulness on fact train rows and scores it on every set.

    The readers are fitted once per seed on the `train` facts and scored,
    with no refit, on the fact `val` and `test` splits and on every
    out-of-distribution Geometry of Truth subset. Writes `curves` (the test
    set), one `curves_<name>` per other set, `readers.npz` and `summary.json`
    (the best-layer accuracy of each set and its accuracy at the layer
    chosen on val).

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        facts: Fact statements with `train`, `val` and `test` splits (label
            1 = true).
        truth_sets: Geometry of Truth examples, each with `meta["subset"]`.
            All examples of a subset are used, whatever their split.

    Raises:
        ValueError: If a name in `config.ood_subsets` matches no example.
    """
    subsets = {
        name: [e for e in truth_sets if e.meta.get("subset") == name]
        for name in config.ood_subsets
    }
    missing = [name for name, rows in subsets.items() if not rows]
    if missing:
        raise ValueError(f"no examples for subsets {missing}")
    train = [e for e in facts if e.split == "train"]
    train_sets = [
        _stimuli(
            train,
            loaded,
            config.concept,
            config.n_pairs,
            derive_seed(config.seed, "train", i),
        )
        for i in range(config.n_seeds)
    ]
    sources = {
        "val": [e for e in facts if e.split == "val"],
        "test": [e for e in facts if e.split == "test"],
        **subsets,
    }
    eval_sets = {
        name: _stimuli(
            rows,
            loaded,
            config.concept,
            config.n_eval_pairs,
            derive_seed(config.seed, "eval", name),
        )
        for name, rows in sources.items()
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
    index = SERIES.index("reader")
    curves = {}
    for name in eval_sets:
        stem = "curves" if name == "test" else f"curves_{name}"
        write_curves(
            run_dir,
            result.per_seed(name),
            result.layers,
            series=SERIES,
            stem=stem,
            chance=0.5,
            n_layers=loaded.n_layers,
        )
        curves[name] = run_dir.read_arrays(stem)
    run_dir.write_arrays(
        "readers",
        layers=np.asarray(result.layers, dtype=np.int64),
        directions=np.stack([r.directions for r in result.readers]),
        signs=np.stack([r.signs for r in result.readers]),
    )
    val_layer = best_layer(curves["val"]["mean"][index], result.layers)
    at_val = result.layers.index(val_layer)
    sets = {}
    for name, stimuli in eval_sets.items():
        mean = curves[name]["mean"][index]
        best = best_layer(mean, result.layers)
        at = result.layers.index(best)
        sets[name] = {
            "best_layer": best,
            "best_accuracy": float(mean[at]),
            "ci": [
                float(curves[name]["lo"][index, at]),
                float(curves[name]["hi"][index, at]),
            ],
            "accuracy_at_val_layer": float(mean[at_val]),
            "n_eval_pairs": int(stimuli.groups.shape[0]),
        }
    run_dir.write_json(
        "summary",
        {
            "val_layer": val_layer,
            "sets": sets,
            "n_seeds": config.n_seeds,
            "n_pairs": config.n_pairs,
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads GPT-2, the facts and Geometry of Truth, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    execute(
        config,
        run_dir,
        load_model(config.model),
        load_facts(),
        load_geometry_of_truth(subsets=config.ood_subsets),
    )


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the R7 layer curves of every evaluation set.

    Args:
        run_dir: A finished R7 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of every `curves*` file, sorted by stem.
    """
    stems = sorted(p.stem for p in run_dir.path.glob("curves*.npz"))
    paths: list[Path] = []
    for stem in stems:
        paths.extend(plot_curves_file(run_dir, out_dir, stem=stem))
    return paths
