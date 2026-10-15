"""A1: geometry of the reading directions across layers, seeds and concepts.

A1 loads no model and reads no activations. It reads the `readers*.npz` files
of finished reading runs, found under its own out root.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from tomography.analysis import agreement_band
from tomography.analysis import layer_cosines
from tomography.analysis import mean_direction
from tomography.analysis import seed_agreement
from tomography.experiments.aggregate import ReaderSet
from tomography.experiments.aggregate import common_layers
from tomography.experiments.aggregate import depth_layer
from tomography.experiments.aggregate import gather_readers
from tomography.experiments.aggregate import label_stem
from tomography.experiments.aggregate import out_root_of
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.intervals import write_interval_curves
from tomography.experiments.matrices import plot_matrix_file
from tomography.experiments.matrices import write_matrix
from tomography.experiments.sweep import plot_curves_file
from tomography.reading.metrics import cosine_matrix


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """A1 settings.

    Attributes:
        sources: Experiments whose default runs on `model` are read.
        extra_runs: Finished runs that replace a default run.
        depth: Relative depth of the cross-concept matrix.
        n_boot: Not used by the agreement band, which is a closed-form
            interval; kept, at least 1, so existing configs and run hashes
            stay valid.
    """

    experiment: str = "a1"
    model: str = "gpt2"
    sources: tuple[str, ...] = ("r1", "r2", "r3", "r5", "r7")
    extra_runs: tuple[str, ...] = ()
    depth: float = 0.5
    n_boot: int = 2000


def execute(
    config: Config, run_dir: RunDir, readers: Mapping[str, ReaderSet]
) -> None:
    """Measures how the reading directions agree.

    Every concept uses its first `n_min` seeds, `n_min` being the smallest
    seed count among the concepts. Writes `matrix_layers_<label>` (the cosine
    between layers of one concept, averaged over seeds), `matrix_concepts`
    (the cosine between the mean directions of the concepts at the layer
    nearest `config.depth`), `curves_agreement` and `summary.json`. The
    curves hold the cosine of each seed to the other seeds per layer, whose
    mean is the mean pairwise cosine. Its band is a t-interval over the
    jackknife pseudo-values of that mean, clipped to [-1, 1], because the
    per-seed rows share pairs and are not independent. The summary holds
    `chance_abs_cosine`, the mean absolute cosine of two random directions,
    as the reference for `matrix_concepts`, and `source_runs`, the config
    hash of the run each concept was read from (None for a set built by
    hand).

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        readers: The reader sets by concept label, at most eight, read at
            the same layers of the same model.

    Raises:
        ValueError: If there is no set, there are too many, they were read
            at different layers or depths, a concept has fewer than three
            seeds, `config.depth` is outside (0, 1], `config.n_boot` is below
            1 or `config.seed` is negative.
    """
    if config.n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {config.n_boot}")
    if config.seed < 0:
        raise ValueError(f"seed must be non-negative, got {config.seed}")
    layers, n_layers, n_min = common_layers(readers)
    if n_min < 3:
        raise ValueError(f"A1 needs at least 3 seeds per concept, got {n_min}")
    layer = depth_layer(layers, config.depth, n_layers)
    at = layers.index(layer)
    labels = list(readers)
    vectors = [readers[label].vectors[:n_min] for label in labels]
    names = [str(k) for k in layers]
    layer_matrices = [layer_cosines(v) for v in vectors]
    concepts = cosine_matrix(np.stack([mean_direction(v)[at] for v in vectors]))
    agreement = np.stack([seed_agreement(v) for v in vectors])
    for label, matrix in zip(labels, layer_matrices, strict=True):
        write_matrix(
            run_dir,
            f"matrix_layers_{label_stem(label)}",
            matrix,
            rows=names,
            cols=names,
            title=f"Cosine between layers, {label}",
        )
    write_matrix(
        run_dir,
        "matrix_concepts",
        concepts,
        rows=labels,
        cols=labels,
        title=f"Cosine between concepts at layer {layer}",
    )
    bands = [agreement_band(v) for v in vectors]
    write_interval_curves(
        run_dir,
        layers,
        agreement.mean(axis=1),
        np.stack([lo for lo, _ in bands]),
        np.stack([hi for _, hi in bands]),
        series=labels,
        stem="curves_agreement",
        xlabel="Layer",
        ylabel="Cosine to the other seeds",
        chance=0.0,
        n_layers=n_layers,
        per_seed=agreement,
    )
    width = vectors[0].shape[2]
    run_dir.write_json(
        "summary",
        {
            "concepts": labels,
            "layer": layer,
            "depth": config.depth,
            "n_seeds": n_min,
            "chance_abs_cosine": math.sqrt(2.0 / (math.pi * width)),
            "agreement_at_layer": {
                label: float(agreement[i, :, at].mean())
                for i, label in enumerate(labels)
            },
            "source_runs": {
                label: readers[label].config_hash for label in labels
            },
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Gathers the readers of the source runs, then runs `execute`.

    The source runs are looked up under the out root of `run_dir`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    readers = gather_readers(
        out_root_of(run_dir), config.sources, config.model, config.extra_runs
    )
    execute(config, run_dir, readers)


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the agreement curves and every matrix of an A1 run.

    Args:
        run_dir: A finished A1 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of the curves and of every matrix.
    """
    paths = plot_curves_file(run_dir, out_dir, stem="curves_agreement")
    for file in sorted(run_dir.path.glob("matrix_*.npz")):
        paths.extend(
            plot_matrix_file(
                run_dir,
                out_dir,
                stem=file.stem,
                kind="diverging",
                vmin=-1.0,
                vmax=1.0,
            )
        )
    return paths
