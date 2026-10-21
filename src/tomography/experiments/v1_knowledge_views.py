"""V1: the activations behind the knowledge readers of P9b.

P9b reports how well a direction separates known from unknown questions. This
experiment keeps the hidden states of the same questions, so the readers can
be seen and not only scored: the 2D views of the questions the figures show,
each beside a text-only view of the same questions, and the published reader of
each contrast against the rest of the hidden state. The inputs are two
finished runs of the model: the P9a labels and the P9b reading. The model loads
in float32 on the default device, because the shared few-shot prefix matches
the uncached activations only in float32; `meta.json` records the device the
numbers came from.
"""

from collections import Counter
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
import gc
import hashlib
from pathlib import Path
from typing import Any
from typing import cast

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD
import torch

from tomography.analysis import pca_coordinates
from tomography.experiments import p9b_reading
from tomography.experiments.aggregate import open_finished
from tomography.experiments.aggregate import read_table
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.p9a_labels import load_labeled
from tomography.experiments.sweep import chat_format_for
from tomography.knowledge import DEFAULT_SHOTS
from tomography.model import MODELS
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
from tomography.p9.reading import split_activations
from tomography.p9.reading import surface_baselines
from tomography.p9.reading import type_scores
from tomography.p9.risk import word_vectorizer
from tomography.p9.unknowns import TypePool
from tomography.p9.unknowns import panel_of
from tomography.reading.metrics import auroc
from tomography.reading.reader import RepReader
from tomography.viz import style
from tomography.viz.geometry import POINT_SIZE

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
REST_ENERGY = 1e-12
MIN_SECOND_COMPONENT = 1e-9
SHOWN_ENTITY = ("obscure_real", "fictional_generated")
COUNT_COLUMNS = (
    "n_positive_train",
    "n_negative_train",
    "n_positive_test",
    "n_negative_test",
)
# How the check figure draws each side, see `_sides_of`: the code, color,
# marker, alpha and legend label, in the order the points are drawn.
SIDES = (
    (-1, style.INK_MUTED, "o", 0.4, "not in the contrast"),
    (0, style.CATEGORICAL[0], "o", 1.0, "positive side"),
    (1, style.CATEGORICAL[1], "s", 1.0, "negative side"),
)


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """V1 settings: the runs to follow and the cost of one forward pass.

    Everything else is read from the P9b run, so the questions, the draws and
    the template are those of its committed numbers. The seed of this run only
    mixes into the permutations of the shuffled-label control and into the
    randomized SVD of the text-only views, which are an LSA on uncentered
    TF-IDF and not a PCA.

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
            the layers of the readers.
        test: Shape (M, T, S, K): the test AUROC of each method, contrast,
            draw and layer, from `transfer.npz`.
        table: The row of each contrast in `table.json`, by column name.
        readers: The published reader of each contrast, the all-pairs one
            the headline method fitted, built from `readers.npz` when the
            run is loaded, so one that cannot score is refused before any
            model work.
    """

    config: p9b_reading.Config
    layer: int
    types: tuple[str, ...]
    layers: tuple[int, ...]
    test: NDArray[np.float64]
    table: dict[str, dict[str, Any]]
    readers: tuple[RepReader, ...]


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
            the contrasts, the methods or the layers, `table.json` lacks a
            column the checks read, or a published reader cannot score: a
            value that is not finite, a zero direction or a sign that is not
            -1 or +1.
    """
    run_dir = open_finished(
        path, model=model, files=READING_FILES, experiment="p9b"
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
    table = read_table(run_dir, "table")
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
    rows = {row["type"]: row for row in table}
    if tuple(rows) != types:
        raise ValueError(f"{path}: table.json disagrees on the contrasts")
    for column in (*COUNT_COLUMNS, *SURFACE_BASELINES):
        if any(column not in row for row in table):
            raise ValueError(f"{path}: table.json has no column {column!r}")
    for key in ("directions", "means", "signs"):
        if not np.isfinite(readers[key]).all():
            raise ValueError(
                f"{path}: readers.npz holds a {key} that is not finite"
            )
    published = []
    for t, name in enumerate(types):
        try:
            published.append(
                RepReader.from_arrays(
                    config.methods[0],
                    layers,
                    readers["directions"][t],
                    readers["means"][t],
                    readers["signs"][t],
                )
            )
        except ValueError as error:
            raise ValueError(
                f"{path}: the reader of {name}: {error}"
            ) from error
    return CommittedReading(
        config=config,
        layer=int(summary["layer"]),
        types=types,
        layers=layers,
        test=test,
        table=rows,
        readers=tuple(published),
    )


def check_reading(
    reading: CommittedReading, pools: Mapping[str, TypePool]
) -> None:
    """Checks that the contrasts rebuilt from the labels are those of the run.

    The questions to extract and the draws to refit come from the pools, the
    numbers to compare with from the run, so both must describe the same
    contrasts, with the same items on each side of the train and test splits.

    Args:
        reading: The committed P9b run.
        pools: The usable contrasts rebuilt from the P9a labels.

    Raises:
        ValueError: If the contrasts differ, a contrast holds other counts
            than the table of the run, or the run did not read the layers
            1..L in order.
    """
    if tuple(pools) != reading.types:
        raise ValueError(
            f"the labels give the contrasts {list(pools)}, the P9b run read "
            f"{list(reading.types)}"
        )
    if reading.layers != tuple(range(1, len(reading.layers) + 1)):
        raise ValueError(
            f"the P9b run read the layers {list(reading.layers)}, not 1..L "
            "in order"
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


def check_labels_run(config: Config, reading: CommittedReading) -> None:
    """Checks that the P9b run was read from the labels this run reads.

    The two paths are compared as directories, resolved from the working
    directory, so a relative and an absolute spelling of one run agree.

    Args:
        config: The experiment settings.
        reading: The committed P9b run.

    Raises:
        ValueError: If the P9a run the P9b run recorded is not `labels_run`.
    """
    recorded = reading.config.labels_run
    if Path(recorded).resolve() != Path(config.labels_run).resolve():
        raise ValueError(
            f"the P9b run was read from the labels in {recorded!r}, this run "
            f"reads the labels in {config.labels_run!r}"
        )


def check_model(reading: CommittedReading, loaded: LoadedModel) -> None:
    """Checks that the model has the layers and the width of the readers.

    It is the one check that needs the model, so it is the first thing done
    once the model is loaded and before anything runs through it.

    Args:
        reading: The committed P9b run.
        loaded: The wrapped model.

    Raises:
        ValueError: If the model has another number of layers than the run
            read, or another width than its readers.
    """
    if loaded.n_layers != len(reading.layers):
        raise ValueError(
            f"the P9b run read {len(reading.layers)} layers, this model has "
            f"{loaded.n_layers}"
        )
    width = reading.readers[0].directions.shape[-1]
    if loaded.d_model != width:
        raise ValueError(
            f"the readers of the P9b run have the width {width}, this model "
            f"has {loaded.d_model}"
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
        The draw with permuted labels and no pairs, under the seed of the
        fit.
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
        h, labels = split_activations(acts, pool, "test")
        for s, draw in enumerate(draws[name]):
            for m, method in enumerate(protocol.methods):
                reader = fit_draw(
                    acts, draw, method=method, preprocess=protocol.preprocess
                )
                test[m, t, s] = auroc(reader.score(h), labels)
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
        h, labels = split_activations(acts, pool, "test")
        for s, draw in enumerate(draws[name]):
            reader = fit_draw(
                acts,
                shuffled_draw(
                    draw,
                    seed=derive_seed(draw.seed, "shuffled", name, seed),
                ),
                method=protocol.methods[0],
                preprocess=protocol.preprocess,
            )
            test[t, s] = auroc(reader.score(h), labels)
    return test


def displayed_items(
    items: Sequence[LabeledItem], *, cap: int
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


def displayed_uids(
    pools: Mapping[str, TypePool], *, cap: int
) -> dict[str, dict[str, list[str]]]:
    """Names the items a figure draws for each side of each contrast.

    Only the test items are drawn, because the readers were fitted on the
    others.

    Args:
        pools: The usable contrasts.
        cap: The most items per side of a contrast.

    Returns:
        For each contrast, the uids of the `positive` and the `negative`
        side, see `displayed_items`.
    """
    shown: dict[str, dict[str, list[str]]] = {}
    for name, pool in pools.items():
        positive, negative = pool.split("test")
        shown[name] = {
            "positive": [
                i.example.uid for i in displayed_items(positive, cap=cap)
            ],
            "negative": [
                i.example.uid for i in displayed_items(negative, cap=cap)
            ],
        }
    return shown


def panel_contrasts(names: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Groups the contrasts a figure shows into its panels.

    The entity panel shows the two contrasts of `SHOWN_ENTITY`, the question
    panel every question contrast and the unknown panel every unknown-type
    one. The other entity contrasts are scored in the tables but not drawn.

    Args:
        names: The usable contrasts, in the order of the pools.

    Returns:
        The contrasts of each panel in the order given, panels in the order
        of `tomography.experiments.p9b_reading.PANELS`; a panel without a
        usable contrast is left out.
    """
    panels: dict[str, list[str]] = {panel: [] for panel in p9b_reading.PANELS}
    for name in names:
        panel = panel_of(name)
        if panel != "entity" or name in SHOWN_ENTITY:
            panels[panel].append(name)
    return {panel: tuple(shown) for panel, shown in panels.items() if shown}


def panel_union(
    contrasts: Iterable[str], uids: Mapping[str, Mapping[str, list[str]]]
) -> list[str]:
    """Lists the items a panel shows, each once.

    Args:
        contrasts: The contrasts of the panel, see `panel_contrasts`.
        uids: The displayed uids of each side of each contrast, see
            `displayed_uids`.

    Returns:
        The uids of the contrasts in order, the positive side before the
        negative one, keeping the first place of an item that several
        contrasts show.
    """
    shown: dict[str, None] = {}
    for name in contrasts:
        for side in ("positive", "negative"):
            shown.update(dict.fromkeys(uids[name][side]))
    return list(shown)


@dataclass(frozen=True)
class Panel:
    """One panel of the figures.

    Attributes:
        types: The contrasts it shows, see `panel_contrasts`.
        uids: The items it places, each once, see `panel_union`.
    """

    types: tuple[str, ...]
    uids: tuple[str, ...]


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
            of the negative side.
    """

    auroc: NDArray[np.float64]
    projections: dict[str, NDArray[np.float32]]


def published_views(
    reading: CommittedReading,
    acts: ItemActivations,
    pools: Mapping[str, TypePool],
    shown: Mapping[str, Mapping[str, list[str]]],
) -> PublishedViews:
    """Scores the test items on the published reader of each contrast.

    The readers are the all-pairs ones of `readers.npz`, not refitted. Every
    test item counts in the AUROC; only the displayed ones are kept for the
    projections.

    Args:
        reading: The committed P9b run.
        acts: The activations of the test items, over the layers of the run.
        pools: The usable contrasts.
        shown: The displayed uids of each side of each contrast, see
            `displayed_uids`.

    Returns:
        The AUROC per layer and the projections at the layer the run chose.

    Raises:
        ValueError: If a test item has no activations.
    """
    k = reading.layers.index(reading.layer)
    aurocs = np.zeros((len(pools), len(reading.layers)))
    projections: dict[str, NDArray[np.float32]] = {}
    for t, (name, pool) in enumerate(pools.items()):
        scores, labels = type_scores(reading.readers[t], acts, pool, "test")
        aurocs[t] = auroc(scores, labels)
        positive, negative = pool.split("test")
        column = {
            item.example.uid: j for j, item in enumerate([*positive, *negative])
        }
        order = [column[uid] for side in shown[name].values() for uid in side]
        projections[f"projection_{name}"] = scores[k, order]
    return PublishedViews(aurocs, projections)


def view_layers(layer: int, n_layers: int) -> tuple[int, ...]:
    """Names the layers the 2D views are drawn at.

    Layer 0 is left out: every prompt ends in the same token, so layer 0 at
    the last position is the embedding of that token and nothing of the
    question. For GPT-2 it is the token embedding plus the position
    embedding, so it varies only with the length of the prompt. Layer 1 shows
    what the model has done with a question before it has read much of it.

    Args:
        layer: The layer the P9b run chose.
        n_layers: The number of layers of the model, L.

    Returns:
        Layer 1, the chosen layer and the last layer, in that order and each
        once: a run that chose layer 1 or the last layer gets two views and
        not three.
    """
    return tuple(dict.fromkeys((1, layer, n_layers)))


def pca_views(
    acts: ItemActivations,
    layers: Sequence[int],
    panels: Mapping[str, Sequence[str]],
) -> dict[str, NDArray[Any]]:
    """Projects the items of each panel on their two leading components.

    The components are fitted at every layer on the items the panel shows
    and on no other, so the view is what a reader of the figure could
    compute from the points on it.

    Args:
        acts: The activations of at least the items of the panels.
        layers: The layers to draw, see `view_layers`.
        panels: The uids each panel shows, see `panel_union`.

    Returns:
        `view_layers` (V,), and for each panel `coordinates_<panel>` (V, n, 2)
        float32 in the order of its uids and `ratios_<panel>` (V, 2) float64,
        the share of the variance of the layer on each component.

    Raises:
        ValueError: If a panel item has no activations, or a layer of a
            panel has no variance (see `tomography.analysis.pca_coordinates`).
    """
    ks = [acts.layers.index(layer) for layer in layers]
    arrays: dict[str, NDArray[Any]] = {
        "view_layers": np.asarray(layers, dtype=np.int64)
    }
    for panel, uids in panels.items():
        rows = acts.rows_of(uids)
        coordinates, ratios = pca_coordinates(acts.data[np.ix_(ks, rows)])
        arrays[f"coordinates_{panel}"] = coordinates
        arrays[f"ratios_{panel}"] = ratios
    return arrays


def residual_axis(
    h: NDArray[Any], direction: NDArray[Any]
) -> NDArray[np.float32]:
    """Scores rows on their top principal component off a direction.

    The rows are centered, the part of each along `direction` is taken
    away, and the rows that are left are projected on their leading
    component. What a reader finds along `direction` is therefore not in the
    score, and the score cannot be explained by it.

    Args:
        h: Activations of shape (n, d).
        direction: A vector of shape (d,), not zero; its scale is ignored.

    Returns:
        The coordinate of each row on the leading component of the rest,
        shape (n,), signed as `tomography.analysis.pca_coordinates` does.

    Raises:
        ValueError: If `direction` is zero, or nothing but the direction
            varies across the rows: what is left of them holds less than
            `REST_ENERGY` of their centered energy.
    """
    rows = np.asarray(h, dtype=np.float64)
    rows = rows - rows.mean(axis=0)
    unit = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(unit)
    if norm == 0.0:
        raise ValueError("the direction is zero")
    unit = unit / norm
    centered = float(np.sum(rows**2))
    rows -= np.outer(rows @ unit, unit)
    if float(np.sum(rows**2)) <= REST_ENERGY * centered:
        raise ValueError("the rows vary along the direction alone")
    coordinates, _ = pca_coordinates(rows[None])
    return coordinates[0, :, 0]


def supervised_views(
    reading: CommittedReading,
    acts: ItemActivations,
    layers: Sequence[int],
    panels: Mapping[str, Panel],
) -> dict[str, NDArray[np.float32]]:
    """Sets the published reader of each contrast against the rest.

    For every contrast a panel shows, the x axis is the score of the items
    of the panel on the published reader of the contrast, the y axis the
    leading component of the same items once the direction of that reader
    is taken out (see `residual_axis`). All the items of the panel are
    placed, also those that belong to other contrasts of it.

    Args:
        reading: The committed P9b run, which holds the published readers.
        acts: The activations of at least the items of the panels, over the
            layers of the run.
        layers: The layers to draw, see `view_layers`.
        panels: The panels of the figures, see `Panel`.

    Returns:
        For each shown contrast, `supervised_<contrast>` of shape (V, n, 2)
        float32 with x then y, in the order of the uids of its panel.

    Raises:
        ValueError: If a panel item has no activations, or the items of a
            panel vary along the direction alone.
    """
    ks = [reading.layers.index(layer) for layer in layers]
    arrays: dict[str, NDArray[np.float32]] = {}
    for panel in panels.values():
        h = acts.data[:, acts.rows_of(panel.uids)]
        for name in panel.types:
            reader = reading.readers[reading.types.index(name)]
            x = reader.score(h)[ks]
            y = np.stack(
                [residual_axis(h[k], reader.directions[k]) for k in ks]
            )
            arrays[f"supervised_{name}"] = np.stack([x, y], axis=-1)
    return arrays


def text_view(
    texts: Sequence[str], *, seed: int
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Places questions by their words alone, the text-only control of a view.

    The features are those of the text classifier of the P9b table (see
    `tomography.p9.risk.word_vectorizer`): TF-IDF weights of the word
    1-grams and 2-grams seen in at least 2 of the texts. A truncated SVD
    keeps the two leading components. This is latent semantic
    analysis (LSA) on uncentered features, not a PCA: the first component
    mostly follows the mean of the features, and the shares of variance it
    returns are not comparable with the ratios of `pca_views`.

    Args:
        texts: The questions of a panel.
        seed: Seeds the randomized SVD.

    Returns:
        The coordinates of shape (n, 2) float32 and the share of the
        variance of the features each LSA component keeps, shape (2,)
        float64.

    Raises:
        ValueError: If the texts share too few words to give two components:
            fewer than two features are left, or the second singular value
            is below `MIN_SECOND_COMPONENT` of the first, as when the
            questions differ only by words that occur once.
    """
    features = word_vectorizer().fit_transform(texts)
    svd = TruncatedSVD(n_components=2, random_state=seed)
    with np.errstate(invalid="ignore"):
        coordinates = svd.fit_transform(features)
    first, second = svd.singular_values_
    if second <= MIN_SECOND_COMPONENT * first:
        raise ValueError("the texts do not give two components")
    return (
        np.asarray(coordinates, dtype=np.float32),
        np.asarray(svd.explained_variance_ratio_, dtype=np.float64),
    )


def text_views(
    questions: Mapping[str, str],
    panels: Mapping[str, Sequence[str]],
    *,
    seed: int,
) -> dict[str, NDArray[Any]]:
    """Fits the text-only view of each panel on the questions it shows.

    The text is read here and goes no further: the arrays hold coordinates
    only. The views are LSA on uncentered TF-IDF (see `text_view`), so the
    `ratios_<panel>` are not PCA ratios.

    Args:
        questions: The text of each question by uid.
        panels: The uids each panel shows, see `panel_union`.
        seed: The seed of the run; each panel derives its own.

    Returns:
        For each panel, `coordinates_<panel>` (n, 2) float32 and
        `ratios_<panel>` (2,) float64, in the order of its uids.

    Raises:
        ValueError: If a panel item has no question, or a panel gives no
            view (see `text_view`).
    """
    arrays: dict[str, NDArray[Any]] = {}
    for panel, uids in panels.items():
        for uid in uids:
            if uid not in questions:
                raise ValueError(f"no question for {uid!r}")
        coordinates, ratios = text_view(
            [questions[uid] for uid in uids],
            seed=derive_seed(seed, "text", panel),
        )
        arrays[f"coordinates_{panel}"] = coordinates
        arrays[f"ratios_{panel}"] = ratios
    return arrays


@dataclass(frozen=True, eq=False)
class Plan:
    """What a run needs before it touches the model.

    Attributes:
        pools: The usable contrasts rebuilt from the labels.
        draws: The draws of each contrast, see `train_draws`.
        selected: The questions to run through the model, see
            `select_items`.
        surface: The recomputed surface controls beside the committed ones,
            see `compare_surface`.
        shown: The uids of the displayed items of each side of each
            contrast, at most `MAX_DISPLAYED`, see `displayed_uids`.
        panels: The panels of the figures, see `Panel`.
        text: The text-only view of each panel, see `text_views`.
    """

    pools: dict[str, TypePool]
    draws: dict[str, list[TrainDraw]]
    selected: list[LabeledItem]
    surface: dict[str, dict[str, dict[str, Any]]]
    shown: dict[str, dict[str, list[str]]]
    panels: dict[str, Panel]
    text: dict[str, NDArray[Any]]


def prepare(
    config: Config,
    items: Sequence[LabeledItem],
    *,
    known_contrasts: bool,
    reading: CommittedReading,
) -> Plan:
    """Checks every input and lists the questions to extract, with no model.

    Args:
        config: The experiment settings.
        items: The labeled questions of the P9a run.
        known_contrasts: The P9a gate of the labels.
        reading: The committed P9b run, see `load_reading`.

    Returns:
        The plan of the run.

    Raises:
        ValueError: If `batch_size` is below 1, the P9b run was read from
            other labels (see `check_labels_run`), the contrasts rebuilt from
            the labels differ from those of the run (see `check_reading`), a
            surface control cannot be computed or a text view cannot be
            fitted (see `text_view`).
    """
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    check_labels_run(config, reading)
    pools, _ = p9b_reading.usable_pools(reading.config, items, known_contrasts)
    check_reading(reading, pools)
    surface = compare_surface(pools, reading)
    draws = train_draws(pools, reading.config)
    shown = displayed_uids(pools, cap=MAX_DISPLAYED)
    panels = {
        name: Panel(types, tuple(panel_union(types, shown)))
        for name, types in panel_contrasts(pools).items()
    }
    questions = {
        item.example.uid: item.example.question
        for pool in pools.values()
        for item in (*pool.positive, *pool.negative)
    }
    text = text_views(
        questions,
        {name: panel.uids for name, panel in panels.items()},
        seed=config.seed,
    )
    return Plan(
        pools,
        draws,
        select_items(pools, draws),
        surface,
        shown,
        panels,
        text,
    )


def run_plan(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    plan: Plan,
    reading: CommittedReading,
) -> None:
    """Extracts the activations of a P9b run and checks its numbers on them.

    Runs the test items and the train items of the committed draws through
    the model once, at the layers 1..L the P9b run read and with the shared
    few-shot prefix, then refits and rescores the committed draws of every
    method. The layers stay in the order and at the positions of the P9b run,
    because the random finder draws its direction from the position of a
    layer. The layer to compare at is the one the P9b run chose.

    Writes `projections.npz`: `layers` (K,), `test` (M, T, S, K) the test
    AUROC of each method, contrast, draw and layer refitted here, `shuffled`
    (T, S, K) the control of the headline method refitted on labels permuted
    inside the strata, `published` (T, K) the test AUROC of the published
    all-pairs reader of each contrast, and `projection_<contrast>` the scores
    of the displayed items on that reader at the chosen layer. Writes
    `projections.json`: the contrasts, the methods, the uids of the
    displayed items of each side (at most `MAX_DISPLAYED`, by ascending
    SHA-256 of the uid; no question text) and, for each panel of the figures,
    its contrasts and the uids of the items it shows (see `panel_union`).
    Writes `pca2d.npz`: the 2D views of each panel (see `pca_views`) and
    the supervised view of each shown contrast (see `supervised_views`).
    Writes `text2d.npz`: the text-only view of each panel (see
    `text_views`), an LSA on uncentered TF-IDF whose ratios are not PCA
    ratios; the questions are read to make it and are not stored.
    Writes `summary`: the layer, the number of items extracted, the dtype of
    this run (`extraction_dtype`, that of the loaded model) and the one the
    P9b run was extracted in (`reading_run_dtype`, the registry default of
    the model, since that run loaded it with no dtype; None for a model
    outside the registry), the layers and panels of the views, the
    comparison with the committed AUROC
    (`reproduction`, see `compare_auroc`) and with the committed surface
    controls (`surface`, see `compare_surface`). A difference is reported,
    never raised.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model. Its reference is dropped once the
            activations are extracted, so the refits and the views do not
            hold the model when the caller holds no other reference.
        plan: The checked inputs, see `prepare`.
        reading: The committed P9b run, see `load_reading`.

    Raises:
        ValueError: If the model does not fit the readers of the run (see
            `check_model`). Nothing runs through the model before that.
    """
    check_model(reading, loaded)
    protocol = reading.config
    pools, draws, selected = plan.pools, plan.draws, plan.selected
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
    n_layers, dtype = loaded.n_layers, loaded.dtype
    del loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    test = refit_methods(acts, pools, draws, protocol)
    shuffled = refit_shuffled(acts, pools, draws, protocol, seed=config.seed)
    views = published_views(reading, acts, pools, plan.shown)
    layers = view_layers(reading.layer, n_layers)
    run_dir.write_arrays(
        "pca2d",
        **pca_views(
            acts,
            layers,
            {name: panel.uids for name, panel in plan.panels.items()},
        ),
        **supervised_views(reading, acts, layers, plan.panels),
    )
    run_dir.write_arrays("text2d", **plan.text)
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
            "uids": plan.shown,
            "panels": {
                name: dataclasses.asdict(panel)
                for name, panel in plan.panels.items()
            },
        },
    )
    splits = Counter(item.example.split for item in selected)
    registered = MODELS.get(config.model)
    run_dir.write_json(
        "summary",
        {
            "layer": reading.layer,
            "template": protocol.template,
            "methods": list(protocol.methods),
            "types": list(pools),
            "n_items": {"train": splits["train"], "test": splits["test"]},
            "extraction_dtype": str(dtype).removeprefix("torch."),
            "reading_run_dtype": registered.dtype if registered else None,
            "views": {
                "layers": list(layers),
                "panels": {
                    name: {"types": panel.types, "n_items": len(panel.uids)}
                    for name, panel in plan.panels.items()
                },
            },
            "reproduction": compare_auroc(
                test,
                reading.test,
                reading.layers.index(reading.layer),
                protocol.methods,
                reading.types,
            ),
            "surface": plan.surface,
            "surface_equal": all(
                entry["equal"]
                for by_control in plan.surface.values()
                for entry in by_control.values()
            ),
        },
    )


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    items: Sequence[LabeledItem],
    *,
    known_contrasts: bool,
    reading: CommittedReading,
) -> None:
    """Checks the inputs, then extracts and writes a run (see `run_plan`).

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        items: The labeled questions of the P9a run.
        known_contrasts: The P9a gate of the labels.
        reading: The committed P9b run, see `load_reading`.

    Raises:
        ValueError: If an input is refused (see `prepare`) or the model does
            not fit the readers (see `check_model`). Nothing runs through the
            model before these.
    """
    plan = prepare(
        config, items, known_contrasts=known_contrasts, reading=reading
    )
    run_plan(config, run_dir, loaded, plan, reading)


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the labels and the P9b run, checks them, then loads the model.

    Every input is checked before the model is loaded, apart from the one
    check that needs the model (see `check_model`), which comes right after.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.

    Raises:
        ValueError: If either run is unfinished, of another model, or the
            P9b run lacks a file (see `load_labeled` and `load_reading`), or
            an input is refused (see `prepare`).
    """
    items, labels = load_labeled(config.labels_run, model=config.model)
    reading = load_reading(config.reading_run, model=config.model)
    plan = prepare(
        config,
        items,
        known_contrasts=bool(labels["known_contrasts"]),
        reading=reading,
    )
    # Not bound to a name here, so run_plan holds the one reference and the
    # model is freed once the activations are extracted.
    run_plan(
        config,
        run_dir,
        load_model(config.model, dtype=torch.float32),
        plan,
        reading,
    )


def _sides_of(
    uids: Sequence[str], sides: Sequence[Mapping[str, Sequence[str]]]
) -> NDArray[np.int64]:
    """Tells on which side of some contrasts each item lies.

    Args:
        uids: The items to place.
        sides: The uids of the positive and the negative side of each
            contrast, as the `projections.json` of a run holds them.

    Returns:
        For each item, 1 if any contrast puts it on its negative side, else
        0 if any puts it on its positive side, else -1.
    """
    negative = {uid for side in sides for uid in side["negative"]}
    positive = {uid for side in sides for uid in side["positive"]}
    return np.array(
        [
            1 if uid in negative else 0 if uid in positive else -1
            for uid in uids
        ],
        dtype=np.int64,
    )


def _scatter(ax: Axes, points: NDArray[Any], sides: NDArray[np.int64]) -> None:
    """Draws the points of a panel in one color and shape for each side.

    Args:
        ax: The axes to draw on.
        points: Coordinates of shape (n, 2).
        sides: The side of each point, see `_sides_of`: 0 positive, 1
            negative, -1 not in the contrast, drawn gray.
    """
    for code, color, marker, alpha, _ in SIDES:
        chosen = points[sides == code]
        ax.scatter(
            chosen[:, 0],
            chosen[:, 1],
            s=POINT_SIZE,
            color=color,
            marker=marker,
            alpha=alpha,
            edgecolors=style.SURFACE,
            linewidths=0.4,
            rasterized=True,
        )


def panel_figure(
    panel: str,
    sidecar: Mapping[str, Any],
    pca: Mapping[str, NDArray[Any]],
    text: Mapping[str, NDArray[Any]],
) -> Figure:
    """Draws the views of one panel, a check of the run and not a final figure.

    There is a row for each layer of the 2D views and a last row for the
    text-only view. The first column places every item of the panel on its two
    leading components (on the two LSA components of the words in the last
    row, which are not principal components). The other columns, one for each
    contrast of the panel, set the published reader of the contrast against
    the leading component of the rest; the text row has none. A point has one
    color and shape on the positive side of a contrast, another on the
    negative side, and is gray where the contrast does not show it. In the
    first column an item takes the negative side if any contrast puts it
    there.

    Args:
        panel: The panel to draw.
        sidecar: The `projections.json` of the run.
        pca: The arrays of `pca2d.npz`.
        text: The arrays of `text2d.npz`.

    Returns:
        The figure.
    """
    entry = sidecar["panels"][panel]
    types: list[str] = entry["types"]
    layers: list[int] = pca["view_layers"].tolist()
    shown = sidecar["uids"]
    every = _sides_of(entry["uids"], [shown[name] for name in types])
    own = {name: _sides_of(entry["uids"], [shown[name]]) for name in types}
    rows, columns = len(layers) + 1, 1 + len(types)
    with style.style_context():
        fig, axes = style.new_figure(
            rows, columns, width=max(style.TEXT_WIDTH, 1.8 * columns)
        )
        for v, layer in enumerate(layers):
            _scatter(axes[v, 0], pca[f"coordinates_{panel}"][v], every)
            axes[v, 0].set_ylabel(f"layer {layer}")
            for c, name in enumerate(types, start=1):
                _scatter(axes[v, c], pca[f"supervised_{name}"][v], own[name])
        _scatter(axes[-1, 0], text[f"coordinates_{panel}"], every)
        axes[-1, 0].set_ylabel("words only (LSA)")
        for ax in axes[-1, 1:]:
            ax.set_visible(False)
        axes[0, 0].set_title("two leading components")
        for c, name in enumerate(types, start=1):
            axes[0, c].set_title(name)
        # The legend names the two sides before the items outside them.
        keys = [
            Line2D(
                [], [], linestyle="", marker=marker, color=color, label=label
            )
            for _, color, marker, _, label in (*SIDES[1:], SIDES[0])
        ]
        fig.legend(handles=keys, loc="outside upper center", ncols=len(keys))
    return fig


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws a check figure of the views of each panel of a run.

    Args:
        run_dir: A finished V1 run.
        out_dir: Folder of the figures.

    Returns:
        The PNG of each panel, `views_<panel>.png`, in the order of the
        panels.

    Raises:
        FileNotFoundError: If the run has no views.
    """
    sidecar = cast(dict[str, Any], run_dir.read_json("projections"))
    pca = run_dir.read_arrays("pca2d")
    text = run_dir.read_arrays("text2d")
    paths: list[Path] = []
    for panel in sidecar["panels"]:
        paths += style.save_figure(
            panel_figure(panel, sidecar, pca, text),
            out_dir,
            f"views_{panel}",
            formats=("png",),
        )
    return paths
