"""V1: the activations behind the knowledge readers of P9b.

P9b reports how well a direction separates known from unknown questions. This
experiment keeps the hidden states of the same questions on the CPU, so the
readers can be seen and not only scored. The inputs are two finished runs of
the model: the P9a labels and the P9b reading. Run it with
`CUDA_VISIBLE_DEVICES` set to the empty string, so that `meta.json` records
the CPU the numbers came from.
"""

from collections import Counter
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
from numpy.typing import NDArray
import torch

from tomography.experiments import p9b_reading
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.p9a_labels import load_labeled
from tomography.experiments.sweep import chat_format_for
from tomography.knowledge import DEFAULT_SHOTS
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.p9.pool import LabeledItem
from tomography.p9.reading import READING_METHODS
from tomography.p9.reading import SURFACE_BASELINES
from tomography.p9.reading import ItemActivations
from tomography.p9.reading import TrainDraw
from tomography.p9.reading import collect_item_activations
from tomography.p9.reading import draw_seed
from tomography.p9.reading import draw_train
from tomography.p9.reading import fit_draw
from tomography.p9.reading import p9_template
from tomography.p9.reading import surface_baselines
from tomography.p9.reading import type_auroc
from tomography.p9.reading import type_scores
from tomography.p9.unknowns import TypePool
from tomography.reading.metrics import auroc
from tomography.reading.reader import RepReader

READING_FILES = (
    "summary.json",
    "table.json",
    "transfer.npz",
    "transfer.json",
    "readers.npz",
    "readers.json",
)
TOLERANCE = 0.02
SURFACE_TOLERANCE = 1e-9
MAX_DISPLAYED = 300
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
    the template are those of its committed numbers. The seed of this run only
    mixes into the permutations of the shuffled-label control.

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
        or (test.shape[1], test.shape[2], test.shape[3])
        != (len(types), config.n_seeds, len(layers))
        or readers["directions"].shape[:2] != shape
        or readers["means"].shape != readers["directions"].shape
        or readers["signs"].shape != shape
        or summary["layer"] not in layers
    ):
        raise ValueError(
            f"{path}: the arrays do not fit {len(types)} contrasts, "
            f"{config.n_seeds} draws and {len(layers)} layers"
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


def shuffled_draw(draw: TrainDraw, *, seed: int) -> TrainDraw:
    """Permutes the labels of a draw inside each of its strata.

    Every stratum keeps its number of items on each side, so the control
    differs from the reader only in which item carries which label. The
    pairs are dropped, because they no longer join the two sides.

    Args:
        draw: The items and labels of a fit.
        seed: Seeds the permutations.

    Returns:
        The draw with permuted labels and no pairs.
    """
    rng = np.random.default_rng(seed)
    labels = draw.labels.copy()
    for code in np.unique(draw.strata):
        inside = np.flatnonzero(draw.strata == code)
        labels[inside] = rng.permutation(labels[inside])
    return dataclasses.replace(draw, labels=labels, pairs=None)


def refit_methods(
    acts: ItemActivations,
    pools: Mapping[str, TypePool],
    draws: Mapping[str, list[TrainDraw]],
    protocol: p9b_reading.Config,
) -> NDArray[np.float64]:
    """Fits every method on every draw and scores it on the test items.

    This is the fit of `tomography.p9.reading.reading_cube`, on the items of
    the draws instead of every train item.

    Args:
        acts: The activations of the test items and of the items of `draws`.
        pools: The usable contrasts.
        draws: The draws of each contrast, see `train_draws`.
        protocol: The settings of the P9b run: the methods, the preprocessing
            and the seed.

    Returns:
        Shape (M, T, S, K): the test AUROC of each method, contrast, draw and
        layer of `acts`.
    """
    test = np.zeros(
        (len(protocol.methods), len(pools), protocol.n_seeds, len(acts.layers))
    )
    for t, (name, pool) in enumerate(pools.items()):
        for s, draw in enumerate(draws[name]):
            for m, method in enumerate(protocol.methods):
                reader = fit_draw(
                    acts,
                    draw,
                    method=method,
                    preprocess=protocol.preprocess,
                    seed=draw_seed(protocol.seed, s),
                )
                test[m, t, s] = type_auroc(reader, acts, pool, "test")
    return test


def refit_shuffled(
    acts: ItemActivations,
    pools: Mapping[str, TypePool],
    draws: Mapping[str, list[TrainDraw]],
    protocol: p9b_reading.Config,
    *,
    seed: int,
) -> NDArray[np.float64]:
    """Fits the headline method on shuffled labels, the control of the reader.

    Each fit uses the items of a draw with their labels permuted inside the
    strata, with a seed derived from the seed of the draw and of the
    contrast. A reader that follows what the labels mean falls to chance.

    Args:
        acts: The activations of the test items and of the items of `draws`.
        pools: The usable contrasts.
        draws: The draws of each contrast, see `train_draws`.
        protocol: The settings of the P9b run.
        seed: The seed of the control, mixed into every permutation.

    Returns:
        Shape (T, S, K): the test AUROC of each contrast, draw and layer.
    """
    test = np.zeros((len(pools), protocol.n_seeds, len(acts.layers)))
    for t, (name, pool) in enumerate(pools.items()):
        for s, draw in enumerate(draws[name]):
            draw_state = draw_seed(protocol.seed, s)
            reader = fit_draw(
                acts,
                shuffled_draw(
                    draw,
                    seed=derive_seed(draw_state, "shuffled", name, seed),
                ),
                method=protocol.methods[0],
                preprocess=protocol.preprocess,
                seed=draw_state,
            )
            test[t, s] = type_auroc(reader, acts, pool, "test")
    return test


def displayed_items(
    items: Sequence[LabeledItem], *, cap: int = MAX_DISPLAYED
) -> list[LabeledItem]:
    """Picks the items a figure draws, whoever looks at the run.

    Args:
        items: One side of one contrast.
        cap: The most items to keep.

    Returns:
        The first `cap` items by ascending SHA-256 of the uid, so the choice
        depends on the uids only; all of them when there are fewer.
    """
    return sorted(
        items,
        key=lambda item: hashlib.sha256(item.example.uid.encode()).hexdigest(),
    )[:cap]


def compare_auroc(
    recomputed: NDArray[np.float64],
    committed: NDArray[np.float64],
    k: int,
    methods: Sequence[str],
    types: Sequence[str],
) -> dict[str, Any]:
    """Measures how far the recomputed AUROC is from the committed one.

    The run reproduces the numbers of the P9b run when the mean over the
    draws, at its chosen layer, differs by at most `TOLERANCE` for every
    contrast and method. A miss is reported, not raised.

    Args:
        recomputed: Shape (M, T, S, K): the test AUROC refitted here.
        committed: The same axes, from `transfer.npz`.
        k: Position of the chosen layer on the last axis.
        methods: The method of each entry of the first axis.
        types: The contrast of each entry of the second axis.

    Returns:
        The `tolerance`, whether the run is `within_tolerance`, the largest
        absolute difference (`max_abs_diff`) of a draw and of the mean over
        the draws, at the chosen layer and over all layers, and the
        difference of the means at the chosen layer for each contrast and
        method (`mean_diff_at_layer`).
    """
    draw_diff = np.abs(recomputed - committed)
    mean_diff = np.abs(recomputed.mean(axis=2) - committed.mean(axis=2))
    at_layer = float(mean_diff[..., k].max())
    return {
        "tolerance": TOLERANCE,
        "within_tolerance": at_layer <= TOLERANCE,
        "max_abs_diff": {
            "draw_at_layer": float(draw_diff[..., k].max()),
            "mean_at_layer": at_layer,
            "draw_all_layers": float(draw_diff.max()),
            "mean_all_layers": float(mean_diff.max()),
        },
        "mean_diff_at_layer": {
            type_name: {
                method: float(mean_diff[m, t, k])
                for m, method in enumerate(methods)
            }
            for t, type_name in enumerate(types)
        },
    }


def _same(recomputed: float | None, committed: float | None) -> bool:
    if recomputed is None or committed is None:
        return recomputed is None and committed is None
    return abs(recomputed - committed) <= SURFACE_TOLERANCE


def compare_surface(
    pools: Mapping[str, TypePool], reading: CommittedReading
) -> dict[str, dict[str, dict[str, Any]]]:
    """Recomputes the surface controls and sets them beside the committed ones.

    They need no model, so a difference points at the labels or the library
    versions, not at the activations.

    Args:
        pools: The usable contrasts.
        reading: The committed P9b run.

    Returns:
        For each contrast and each of `SURFACE_BASELINES`, the `recomputed`
        and the `committed` value (None where a contrast has none) and
        whether they are `equal` within `SURFACE_TOLERANCE`.
    """
    report: dict[str, dict[str, dict[str, Any]]] = {}
    for name, pool in pools.items():
        recomputed = surface_baselines(pool)
        report[name] = {
            control: {
                "recomputed": recomputed[control],
                "committed": reading.table[name][control],
                "equal": _same(
                    recomputed[control], reading.table[name][control]
                ),
            }
            for control in SURFACE_BASELINES
        }
    return report


@dataclass(frozen=True, eq=False)
class PublishedViews:
    """What the published readers give on the test items of each contrast.

    Attributes:
        auroc: Shape (T, K): the test AUROC of the published reader of each
            contrast at each layer.
        projections: The scores of the displayed items on the published
            reader at the chosen layer, float32, by `projection_<contrast>`.
            The displayed items of the positive side come first, then those
            of the negative side, as in `uids`.
        uids: The uids of the displayed items of each side, by contrast.
    """

    auroc: NDArray[np.float64]
    projections: dict[str, NDArray[np.float32]]
    uids: dict[str, dict[str, list[str]]]


def published_views(
    reading: CommittedReading,
    acts: ItemActivations,
    pools: Mapping[str, TypePool],
    *,
    cap: int = MAX_DISPLAYED,
) -> PublishedViews:
    """Scores the test items on the published reader of each contrast.

    The readers are the all-pairs ones of `readers.npz`, not refitted. Every
    test item counts in the AUROC; at most `cap` per side are kept for the
    projections.

    Args:
        reading: The committed P9b run.
        acts: The activations of the test items, over the layers of the run.
        pools: The usable contrasts.
        cap: The most displayed items per side of a contrast.

    Returns:
        The AUROC per layer, the projections at the layer the run chose and
        the uids of the displayed items.

    Raises:
        ValueError: If a test item has no activations.
    """
    k = reading.layers.index(reading.layer)
    aurocs = np.zeros((len(pools), len(reading.layers)))
    projections: dict[str, NDArray[np.float32]] = {}
    uids: dict[str, dict[str, list[str]]] = {}
    for t, (name, pool) in enumerate(pools.items()):
        reader = RepReader.from_arrays(
            reading.config.methods[0],
            reading.layers,
            reading.directions[t],
            reading.means[t],
            reading.signs[t],
        )
        scores, labels = type_scores(reader, acts, pool, "test")
        aurocs[t] = auroc(scores, labels)
        positive, negative = pool.split("test")
        shown = {
            "positive": displayed_items(positive, cap=cap),
            "negative": displayed_items(negative, cap=cap),
        }
        column = {
            item.example.uid: j for j, item in enumerate([*positive, *negative])
        }
        order = [
            column[item.example.uid]
            for item in (*shown["positive"], *shown["negative"])
        ]
        projections[f"projection_{name}"] = scores[k, order]
        uids[name] = {
            side: [item.example.uid for item in items]
            for side, items in shown.items()
        }
    return PublishedViews(aurocs, projections, uids)


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    items: Sequence[LabeledItem],
    *,
    known_contrasts: bool,
    reading: CommittedReading,
) -> None:
    """Extracts the activations of a P9b run and checks its numbers on them.

    Rebuilds the contrasts of the run from the labels, runs the test items
    and the train items of the committed draws through the model once, at
    the layers 1..L the P9b run read and with the shared few-shot prefix, then
    refits and rescores the committed draws of every method. The layers stay
    in the order and at the positions of the P9b run, because the random
    finder draws its direction from the position of a layer. The layer to
    compare at is the one the P9b run chose.

    Writes `projections.npz`: `layers` (K,), `test` (M, T, S, K) the test
    AUROC of each method, contrast, draw and layer refitted here, `shuffled`
    (T, S, K) the control of the headline method refitted on labels permuted
    inside the strata, `published` (T, K) the test AUROC of the published
    all-pairs reader of each contrast, and `projection_<contrast>` the scores
    of the displayed items on that reader at the chosen layer. Writes
    `projections.json`: the contrasts, the methods and the uids of the
    displayed items of each side (at most `MAX_DISPLAYED`, by ascending
    SHA-256 of the uid; no question text). Writes `summary`: the layer, the
    number of items extracted, the comparison with the committed AUROC
    (`reproduction`, see `compare_auroc`) and with the committed surface
    controls (`surface`, see `compare_surface`). A difference is reported,
    never raised.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        items: The labeled questions of the P9a run.
        known_contrasts: The P9a gate of the labels.
        reading: The committed P9b run, see `load_reading`.

    Raises:
        ValueError: If `batch_size` is below 1, the contrasts rebuilt from
            the labels differ from those of the run (see `check_reading`) or
            a surface control cannot be computed. Nothing runs through the
            model before these.
    """
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    protocol = reading.config
    pools, _ = p9b_reading.usable_pools(protocol, items, known_contrasts)
    check_reading(reading, pools, loaded.n_layers)
    surface = compare_surface(pools, reading)
    draws = train_draws(pools, protocol)
    selected = select_items(pools, draws)
    template = p9_template(protocol.template, DEFAULT_SHOTS[: protocol.n_shots])
    acts = collect_item_activations(
        loaded,
        selected,
        template,
        chat_format_for(loaded),
        layers=reading.layers,
        bos=protocol.bos,
        batch_size=config.batch_size,
        share_prefix=True,
    )
    test = refit_methods(acts, pools, draws, protocol)
    shuffled = refit_shuffled(acts, pools, draws, protocol, seed=config.seed)
    views = published_views(reading, acts, pools)
    run_dir.write_arrays(
        "projections",
        layers=np.asarray(reading.layers, dtype=np.int64),
        test=test,
        shuffled=shuffled,
        published=views.auroc,
        **views.projections,
    )
    run_dir.write_json(
        "projections",
        {
            "types": list(pools),
            "methods": list(protocol.methods),
            "max_per_side": MAX_DISPLAYED,
            "uids": views.uids,
        },
    )
    splits = Counter(item.example.split for item in selected)
    run_dir.write_json(
        "summary",
        {
            "layer": reading.layer,
            "template": protocol.template,
            "methods": list(protocol.methods),
            "types": list(pools),
            "n_items": {"train": splits["train"], "test": splits["test"]},
            "reproduction": compare_auroc(
                test,
                reading.test,
                reading.layers.index(reading.layer),
                protocol.methods,
                reading.types,
            ),
            "surface": surface,
            "surface_equal": all(
                entry["equal"]
                for by_control in surface.values()
                for entry in by_control.values()
            ),
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the labels, the P9b run and the model on the CPU, then runs it.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.

    Raises:
        ValueError: If either run is unfinished, of another model, or the
            P9b run lacks a file (see `load_labeled` and `load_reading`).
    """
    items, labels = load_labeled(config.labels_run, model=config.model)
    reading = load_reading(config.reading_run, model=config.model)
    execute(
        config,
        run_dir,
        load_model(config.model, device="cpu", dtype=torch.float32),
        items,
        known_contrasts=bool(labels["known_contrasts"]),
        reading=reading,
    )


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws nothing yet: the views of a run come with its 2D outputs.

    Args:
        run_dir: A finished V1 run.
        out_dir: Folder of the figures.

    Returns:
        An empty list.
    """
    return []
