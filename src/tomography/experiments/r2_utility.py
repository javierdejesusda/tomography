"""R2: reading utility (pleasantness) from GPT-2 on ETHICS."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tomography.activations import PositionStrategy
from tomography.analysis import pca_explained_variance
from tomography.datasets.ethics import load_ethics_utility
from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import Prepared
from tomography.experiments.sweep import ReadingConfig
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import stimulus_activations
from tomography.experiments.sweep import sweep_from_activations
from tomography.experiments.sweep import write_curves
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.directions import Method
from tomography.reading.metrics import best_layer
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat

_COMPONENTS = ("PC1", "PC2", "PC3")


@dataclass(frozen=True, kw_only=True)
class Config(ReadingConfig):
    """R2 settings: GPT-2 utility on ETHICS.

    The shared defaults apply except for the direction finder. The paper
    reads utility with the first principal component of the pair differences
    (`pca_diff`), which stays selectable. On GPT-2 it reads weakly even with
    matched pairs: with 3 seeds and 512 evaluation pairs its test accuracy
    is 0.546 (32 training pairs), 0.564 (128) and 0.559 (512), about the
    level of a random direction (0.568). A supervised `logistic` direction
    reaches 0.611, 0.683 (95% interval 0.678 to 0.691) and 0.705 (0.684 to
    0.723, layer 8), so R2 fits it by default. The explained-variance
    curves describe the pair differences and do not depend on the finder.

    Attributes:
        n_components: Components kept for the explained-variance curves; at
            least 3, and at most the smaller of the number of training pairs
            and the model width.
    """

    experiment: str = "r2"
    model: str = "gpt2"
    method: Method = "logistic"
    n_components: int = 10


def _stimuli(
    examples: Sequence[Example],
    chat_format: ChatFormat,
    n_pairs: int | None,
    seed: int,
) -> StimulusSet:
    return build_stimulus_set(
        examples,
        PAPER_TEMPLATES["utility"],
        chat_format,
        pairing="by_group",
        group_ids=[str(e.meta["pair"]) for e in examples],
        n_pairs=n_pairs,
        seed=seed,
    )


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    examples: Sequence[Example],
) -> None:
    """Reads utility on ETHICS train rows and scores it on val and test rows.

    Rows are split before pairing, and the two scenarios of one row form one
    evaluation group. The layer is chosen on val and reported on test.
    Writes `curves` (test) and `curves_val`, `variance` (the first three
    explained-variance ratios of the training pair differences),
    `readers.npz` and `summary.json`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        examples: ETHICS utility examples with `train`, `val` and `test`
            splits and the row id in `meta["pair"]`.

    Raises:
        ValueError: If `config.n_components` is below 3 or above the smaller
            of `config.n_pairs` and the model width.
    """
    limit = min(config.n_pairs, loaded.d_model)
    if not len(_COMPONENTS) <= config.n_components <= limit:
        raise ValueError(
            f"n_components must be in {len(_COMPONENTS)}..min(n_pairs, "
            f"d_model)={limit}, got {config.n_components}"
        )

    chat_format = chat_format_for(loaded)

    def part(split: Split) -> list[Example]:
        return [e for e in examples if e.split == split]

    train_sets = [
        _stimuli(
            part("train"),
            chat_format,
            config.n_pairs,
            derive_seed(config.seed, "train", i),
        )
        for i in range(config.n_seeds)
    ]
    eval_sets = {
        name: _stimuli(
            part(name),
            chat_format,
            config.n_eval_pairs,
            derive_seed(config.seed, "eval", name),
        )
        for name in ("val", "test")
    }
    selected = (
        tuple(range(1, loaded.n_layers + 1))
        if config.layers is None
        else config.layers
    )
    where_eval: PositionStrategy = (
        config.position
        if config.eval_position is None
        else config.eval_position
    )

    def collect(stimuli: StimulusSet, where: PositionStrategy) -> Prepared:
        act = stimulus_activations(
            loaded,
            stimuli,
            layers=selected,
            position=where,
            bos=config.bos,
            batch_size=config.batch_size,
        )
        return stimuli, act

    train = [collect(s, config.position) for s in train_sets]
    result = sweep_from_activations(
        train,
        {name: collect(s, where_eval) for name, s in eval_sets.items()},
        method=config.method,
        preprocess=config.preprocess,
        seed=config.seed,
    )
    write_curves(
        run_dir,
        result.per_seed("test"),
        result.layers,
        series=SERIES,
        chance=0.5,
        n_layers=loaded.n_layers,
    )
    write_curves(
        run_dir,
        result.per_seed("val"),
        result.layers,
        series=SERIES,
        chance=0.5,
        n_layers=loaded.n_layers,
        stem="curves_val",
    )
    variance = np.stack(
        [
            pca_explained_variance(
                act.data,
                pairs=act.row_groups(stimuli.pairs),
                n_components=config.n_components,
                preprocess=config.preprocess,
            )[:, : len(_COMPONENTS)]
            for stimuli, act in train
        ],
        axis=1,
    ).transpose(2, 1, 0)
    write_curves(
        run_dir,
        variance,
        result.layers,
        series=_COMPONENTS,
        stem="variance",
        ylabel="Explained variance",
        n_layers=loaded.n_layers,
    )
    run_dir.write_arrays(
        "readers",
        layers=np.asarray(result.layers, dtype=np.int64),
        directions=np.stack([r.directions for r in result.readers]),
        signs=np.stack([r.signs for r in result.readers]),
    )
    index = SERIES.index("reader")
    val = run_dir.read_arrays("curves_val")["mean"][index]
    test = run_dir.read_arrays("curves")
    best = best_layer(val, result.layers)
    at = result.layers.index(best)
    run_dir.write_json(
        "summary",
        {
            "best_layer": best,
            "val_accuracy": float(val[at]),
            "test_accuracy": float(test["mean"][index, at]),
            "ci": [
                float(test["lo"][index, at]),
                float(test["hi"][index, at]),
            ],
            "n_seeds": config.n_seeds,
            "n_pairs": config.n_pairs,
            "n_eval_pairs": {
                name: int(s.groups.shape[0]) for name, s in eval_sets.items()
            },
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads GPT-2 and ETHICS utility, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    execute(config, run_dir, load_model(config.model), load_ethics_utility())


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the R2 test curves, val curves and variance curves.

    Args:
        run_dir: A finished R2 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of the three figures.
    """
    return [
        path
        for stem in ("curves", "curves_val", "variance")
        for path in plot_curves_file(run_dir, out_dir, stem=stem)
    ]
