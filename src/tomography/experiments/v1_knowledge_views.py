"""V1: the activations behind the knowledge readers of P9b.

P9b reports how well a direction separates known from unknown questions. This
experiment keeps the hidden states of the same questions on the CPU, so the
readers can be seen and not only scored. The inputs are two finished runs of
the model: the P9a labels and the P9b reading.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
from numpy.typing import NDArray

from tomography.experiments import p9b_reading
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.p9.pool import LabeledItem
from tomography.p9.reading import READING_METHODS
from tomography.p9.reading import TrainDraw
from tomography.p9.reading import draw_seed
from tomography.p9.reading import draw_train
from tomography.p9.unknowns import TypePool

READING_FILES = (
    "summary.json",
    "table.json",
    "transfer.npz",
    "transfer.json",
    "readers.npz",
    "readers.json",
)
COUNT_COLUMNS = (
    "n_positive_train",
    "n_negative_train",
    "n_positive_test",
    "n_negative_test",
)


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """V1 settings: the runs to follow and the cost of one forward pass.

    Everything else is read from the P9b run, so the questions, the draws and
    the template are those of its committed numbers.

    Attributes:
        labels_run: Path of a finished P9a run of the same model.
        reading_run: Path of a finished P9b run of the same model, read in
            full: the local copy that still holds `transfer.npz` and
            `readers.npz`, not an export.
        batch_size: Prompts per forward pass.
    """

    experiment: str = "v1"
    model: str = "gpt2"
    labels_run: str
    reading_run: str
    batch_size: int = 8


@dataclass(frozen=True, eq=False)
class CommittedReading:
    """What the reproduction reads from a finished P9b run.

    Attributes:
        config: The settings of the run: the template, the draws and the seed
            its numbers came from.
        layer: The layer the run chose, where its numbers are compared.
        types: The contrasts, in the order of the arrays below.
        layers: Layer index of each entry of the last axis of `test` and of
            the second axis of the readers.
        test: Shape (M, T, S, K): the test AUROC of each method, contrast,
            draw and layer, from `transfer.npz`.
        table: The row of each contrast in `table.json`, by column name.
        directions: Shape (T, K, d): the all-pairs reader of each contrast,
            fitted by the headline method.
        means: Shape (T, K, d): the vectors subtracted before projecting.
        signs: Shape (T, K): the orientation of each layer.
    """

    config: p9b_reading.Config
    layer: int
    types: tuple[str, ...]
    layers: tuple[int, ...]
    test: NDArray[np.float64]
    table: dict[str, dict[str, Any]]
    directions: NDArray[np.float32]
    means: NDArray[np.float32]
    signs: NDArray[np.float32]


def load_reading(path: str | Path, *, model: str) -> CommittedReading:
    """Reads a finished P9b run, with the files an export drops.

    Args:
        path: The run directory.
        model: The model the caller is about to read.

    Returns:
        The committed numbers and the published readers.

    Raises:
        ValueError: If the directory is not a finished P9b run of `model`,
            a file is missing (an export drops `transfer.npz` and
            `readers.npz`), the run read a template other than "fewshot" or
            methods other than `READING_METHODS`, or its files disagree on
            the contrasts, the methods or the layers.
    """
    run_dir = RunDir.open(path)
    if run_dir.meta.get("finished_at") is None:
        raise ValueError(f"{path} is not a finished run")
    if run_dir.meta.get("experiment") != "p9b":
        raise ValueError(f"{path} is not a P9b run")
    found = run_dir.config.get("model")
    if found != model:
        raise ValueError(f"{path} is a run of {found!r}, not of {model!r}")
    for name in READING_FILES:
        if not (run_dir.path / name).is_file():
            raise ValueError(
                f"{path} has no {name}: read the full run directory, not an "
                "export"
            )
    config = p9b_reading.Config.from_dict(run_dir.config)
    if config.template != "fewshot":
        raise ValueError(
            f"{path} read the template {config.template!r}, not 'fewshot'"
        )
    if config.methods != READING_METHODS:
        raise ValueError(
            f"{path} read the methods {config.methods}, not {READING_METHODS}"
        )
    summary = cast(dict[str, Any], run_dir.read_json("summary"))
    sidecar = cast(dict[str, Any], run_dir.read_json("transfer"))
    stored = cast(dict[str, Any], run_dir.read_json("readers"))
    table = cast(dict[str, Any], run_dir.read_json("table"))
    types = tuple(summary["types"])
    if tuple(sidecar["types"]) != types or tuple(stored["types"]) != types:
        raise ValueError(f"{path}: the files disagree on the contrasts")
    if tuple(sidecar["methods"]) != config.methods:
        raise ValueError(f"{path}: transfer.json disagrees on the methods")
    transfer = run_dir.read_arrays("transfer")
    readers = run_dir.read_arrays("readers")
    layers = tuple(int(layer) for layer in transfer["layers"])
    test = transfer["test"]
    shape = (len(types), len(layers))
    if (
        tuple(int(layer) for layer in readers["layers"]) != layers
        or test.shape[0] != len(config.methods)
        or (test.shape[1], test.shape[3]) != shape
        or readers["directions"].shape[:2] != shape
        or readers["means"].shape != readers["directions"].shape
        or readers["signs"].shape != shape
        or summary["layer"] not in layers
    ):
        raise ValueError(
            f"{path}: the arrays do not fit {len(types)} contrasts and "
            f"{len(layers)} layers"
        )
    rows = {
        row[0]: dict(zip(table["columns"], row, strict=True))
        for row in table["rows"]
    }
    if tuple(rows) != types:
        raise ValueError(f"{path}: table.json disagrees on the contrasts")
    return CommittedReading(
        config=config,
        layer=int(summary["layer"]),
        types=types,
        layers=layers,
        test=test,
        table=rows,
        directions=readers["directions"],
        means=readers["means"],
        signs=readers["signs"],
    )


def check_reading(
    reading: CommittedReading, pools: Mapping[str, TypePool], n_layers: int
) -> None:
    """Checks that the contrasts rebuilt from the labels are those of the run.

    The questions to extract and the draws to refit come from the pools, the
    numbers to compare with from the run, so both must describe the same
    contrasts, with the same items on each side of the train and test splits.

    Args:
        reading: The committed P9b run.
        pools: The usable contrasts rebuilt from the P9a labels.
        n_layers: The number of layers of the model, L.

    Raises:
        ValueError: If the contrasts differ, a contrast holds other counts
            than the table of the run, or the run did not read the layers
            1..L.
    """
    if tuple(pools) != reading.types:
        raise ValueError(
            f"the labels give the contrasts {list(pools)}, the P9b run read "
            f"{list(reading.types)}"
        )
    if reading.layers != tuple(range(1, n_layers + 1)):
        raise ValueError(
            f"the P9b run read the layers {list(reading.layers)}, this model "
            f"has the layers 1..{n_layers}"
        )
    for name, pool in pools.items():
        held = [
            len(items)
            for split in ("train", "test")
            for items in pool.split(split)
        ]
        row = reading.table[name]
        written = [row[column] for column in COUNT_COLUMNS]
        if held != written:
            raise ValueError(
                f"{name}: the labels hold {held} items per side of the "
                f"train and test splits, the P9b table {written}"
            )


def train_draws(
    pools: Mapping[str, TypePool], protocol: p9b_reading.Config
) -> dict[str, list[TrainDraw]]:
    """Draws again the train items every fit of the P9b run used.

    Args:
        pools: The usable contrasts.
        protocol: The settings of the P9b run: the draws, the pairs per fit
            and the seed.

    Returns:
        For each contrast, the draws `0..n_seeds - 1`, each with the seed
        and the pairs of the fit it repeats.
    """
    return {
        name: [
            draw_train(
                pool,
                n_pairs=protocol.n_pairs,
                seed=draw_seed(protocol.seed, s),
            )
            for s in range(protocol.n_seeds)
        ]
        for name, pool in pools.items()
    }


def select_items(
    pools: Mapping[str, TypePool], draws: Mapping[str, list[TrainDraw]]
) -> list[LabeledItem]:
    """Lists the questions to run through the model.

    Args:
        pools: The usable contrasts.
        draws: The draws of each contrast, see `train_draws`.

    Returns:
        Every test item of the contrasts and every train item of a draw,
        once each, in the order of the pools. No validation item is kept,
        because the layer is the one the P9b run chose.
    """
    wanted = {
        item.example.uid
        for by_draw in draws.values()
        for draw in by_draw
        for item in draw.items
    }
    selected: dict[str, LabeledItem] = {}
    for pool in pools.values():
        for item in (*pool.positive, *pool.negative):
            split = item.example.split
            if split == "test" or (
                split == "train" and item.example.uid in wanted
            ):
                selected.setdefault(item.example.uid, item)
    return list(selected.values())


def run(config: Config, run_dir: RunDir) -> None:
    """Runs the experiment.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.

    Raises:
        NotImplementedError: Always, until the run is written.
    """
    raise NotImplementedError


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws nothing yet: the views of a run come with its 2D outputs.

    Args:
        run_dir: A finished V1 run.
        out_dir: Folder of the figures.

    Returns:
        An empty list.
    """
    return []
