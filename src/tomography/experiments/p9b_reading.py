"""P9b: reading known and unknown questions from the activations of a model.

The contrasts set known questions against unknown ones, and one kind of
unknown against another. The study consumes a finished P9a run: the
questions with their labels, and the gate `known_contrasts`, which says
whether the model has enough known questions left after the guess control.
Without them only the unknown-type contrasts are read, so a small model
still reads real obscure entities against invented ones.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np

from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.matrices import plot_matrix_file
from tomography.experiments.matrices import write_matrix
from tomography.experiments.p9a_labels import load_labeled
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import write_curves
from tomography.knowledge import DEFAULT_SHOTS
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.p9.pool import LabeledItem
from tomography.p9.reading import READING_METHODS
from tomography.p9.reading import ReadingCube
from tomography.p9.reading import check_pool
from tomography.p9.reading import choose_layer
from tomography.p9.reading import collect_item_activations
from tomography.p9.reading import p9_template
from tomography.p9.reading import reading_cube
from tomography.p9.unknowns import TypePool
from tomography.p9.unknowns import panel_of
from tomography.p9.unknowns import type_pools
from tomography.reading.directions import Method
from tomography.reading.directions import Preprocess
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import bootstrap_ci

KNOWLEDGE_YLABEL = "AUROC (known vs unknown)"
UNKNOWN_TYPE_YLABEL = "AUROC (unknown-type contrast)"
PANELS = ("entity", "question", "unknown")


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """P9b settings: reading the contrasts of a labeled pool.

    The first of `methods` is the headline: it fills the transfer matrix,
    the cosines, the readers, the panel curves and the interval of the
    table, and it chooses the layer. It is `logistic` because on GPT-2 the
    paper's `pca_diff` stays near chance on unmatched pairs.

    Attributes:
        labels_run: Path of a finished P9a run of the same model.
        template: How a question is rendered, one of
            `tomography.p9.reading.TEMPLATE_NAMES`.
        n_shots: Demonstrations of the "fewshot" prompt.
        types: The contrasts to read; None reads every contrast with enough
            items.
        methods: The direction finders, the headline first.
        preprocess: Vector preparation, see `fit_direction`.
        n_seeds: Draws of the pairs per contrast and method.
        n_pairs: Pairs per fit.
        layers: Layers to read; None reads 1..L.
        layer: Layer of the transfer and cosine matrices; None chooses it on
            the validation split.
        min_pairs: Fewest train items on each side of a contrast.
        min_eval: Fewest val items, and fewest test items, on each side.
        bos: BOS policy; None uses the model's default.
        batch_size: Prompts per forward pass.
    """

    experiment: str = "p9b"
    model: str = "gpt2"
    labels_run: str
    template: str = "fewshot"
    n_shots: int = 15
    types: tuple[str, ...] | None = None
    methods: tuple[Method, ...] = READING_METHODS
    preprocess: Preprocess = "center"
    n_seeds: int = 5
    n_pairs: int = 64
    layers: tuple[int, ...] | None = None
    layer: int | None = None
    min_pairs: int = 16
    min_eval: int = 10
    bos: BosMode | None = None
    batch_size: int = 8


def _usable_pools(
    config: Config, items: Sequence[LabeledItem], known_contrasts: bool
) -> tuple[dict[str, TypePool], dict[str, str]]:
    pools = type_pools(items, knowledge=known_contrasts)
    if config.types is not None:
        missing = [name for name in config.types if name not in pools]
        if missing:
            note = (
                "the p9a gate admits only the unknown_type contrasts for "
                "this model"
                if not known_contrasts
                else "a contrast needs items on both sides"
            )
            raise ValueError(
                f"unknown or empty types {missing}; available "
                f"{list(pools)}; {note}"
            )
        pools = {name: pools[name] for name in config.types}
    usable: dict[str, TypePool] = {}
    skipped: dict[str, str] = {}
    for name, pool in pools.items():
        reason = check_pool(
            pool, min_pairs=config.min_pairs, min_eval=config.min_eval
        )
        if reason is None:
            usable[name] = pool
        else:
            skipped[name] = reason
    if not usable:
        reasons = "; ".join(f"{n}: {r}" for n, r in skipped.items())
        raise ValueError(f"no usable type: {reasons or 'no contrast'}")
    return usable, skipped


def _write_curves(
    run_dir: RunDir,
    cube: ReadingCube,
    pools: dict[str, TypePool],
    n_layers: int,
    seed: int,
) -> list[str]:
    stems = []
    for t, name in enumerate(cube.types):
        stem = f"curves_{name}"
        ylabel = (
            UNKNOWN_TYPE_YLABEL
            if pools[name].contrast == "unknown_type"
            else KNOWLEDGE_YLABEL
        )
        write_curves(
            run_dir,
            cube.test[:, t],
            cube.layers,
            series=cube.methods,
            stem=stem,
            ylabel=ylabel,
            chance=0.5,
            n_layers=n_layers,
            seed=seed,
        )
        stems.append(stem)
    for panel in PANELS:
        members = [
            t for t, name in enumerate(cube.types) if panel_of(name) == panel
        ]
        if not members:
            continue
        write_curves(
            run_dir,
            cube.test[0][members],
            cube.layers,
            series=[cube.types[t] for t in members],
            stem=f"curves_{panel}",
            ylabel=UNKNOWN_TYPE_YLABEL
            if panel == "unknown"
            else KNOWLEDGE_YLABEL,
            chance=0.5,
            n_layers=n_layers,
            seed=seed,
        )
        stems.append(f"curves_{panel}")
    return stems


def _table_rows(
    cube: ReadingCube, pools: dict[str, TypePool], k: int, seed: int
) -> list[list[object]]:
    rows: list[list[object]] = []
    for t, name in enumerate(cube.types):
        pool = pools[name]
        positive_train, negative_train = pool.split("train")
        positive_test, negative_test = pool.split("test")
        lo, hi = bootstrap_ci(cube.test[0, t, :, k], n_boot=200, seed=seed)
        means = [
            float(cube.test[m, t, :, k].mean())
            for m in range(len(cube.methods))
        ]
        rows.append(
            [
                name,
                pool.contrast,
                len(positive_train),
                len(negative_train),
                len(positive_test),
                len(negative_test),
                means[0],
                float(lo),
                float(hi),
                *means[1:],
            ]
        )
    return rows


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    items: Sequence[LabeledItem],
    *,
    known_contrasts: bool,
) -> None:
    """Reads every usable contrast of a labeled pool and writes the results.

    Activations are collected once per distinct question of the usable
    contrasts. Writes `curves_<name>` per contrast (the AUROC of each method
    on the test items), `curves_entity`, `curves_question` and
    `curves_unknown` (the headline method of the contrasts of a panel, a
    panel without contrasts is not written), `matrix` (the transfer AUROC at
    the chosen layer, row = fit contrast, column = scored contrast),
    `matrix_cosine` (split-half cross-fit cosines, whose diagonal is the
    split-half stability of a contrast, see `reading_cube`), `transfer`
    (every AUROC and cosine), `readers` (the all-pairs directions of the
    headline method), `table` and `summary`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        items: The labeled questions of the P9a run.
        known_contrasts: The P9a gate: whether the contrasts that need a
            known side may be read.

    Raises:
        ValueError: If `n_seeds` is below 1, `methods` is empty, `n_pairs` is
            below 2 or `min_eval` is below 1, if the template is unknown,
            `config.types` names a contrast that does not exist (or one the
            gate excludes), no contrast is usable, or `config.layer` is not
            among the layers. Nothing runs through the model before these.
    """
    for message, ok in (
        ("n_seeds must be at least 1", config.n_seeds >= 1),
        ("methods must not be empty", bool(config.methods)),
        ("n_pairs must be at least 2", config.n_pairs >= 2),
        ("min_eval must be at least 1", config.min_eval >= 1),
    ):
        if not ok:
            raise ValueError(message)
    template = p9_template(config.template, DEFAULT_SHOTS[: config.n_shots])
    usable, skipped = _usable_pools(config, items, known_contrasts)
    layers = (
        tuple(range(1, loaded.n_layers + 1))
        if config.layers is None
        else config.layers
    )
    if config.layer is not None and config.layer not in layers:
        raise ValueError(
            f"layer {config.layer} is not among the layers read {layers}"
        )
    union = {
        i.example.uid: i
        for pool in usable.values()
        for i in (*pool.positive, *pool.negative)
    }
    acts = collect_item_activations(
        loaded,
        list(union.values()),
        template,
        chat_format_for(loaded),
        layers=layers,
        bos=config.bos,
        batch_size=config.batch_size,
    )
    cube = reading_cube(
        acts,
        usable,
        methods=config.methods,
        preprocess=config.preprocess,
        n_seeds=config.n_seeds,
        n_pairs=config.n_pairs,
        seed=config.seed,
    )
    layer = (
        config.layer
        if config.layer is not None
        else choose_layer(cube.val, cube.layers)
    )
    k = cube.layers.index(layer)
    curve_stems = _write_curves(
        run_dir, cube, usable, loaded.n_layers, config.seed
    )
    write_matrix(
        run_dir,
        "matrix",
        cube.transfer[:, k].mean(axis=0),
        rows=cube.types,
        cols=cube.types,
        title=f"Transfer AUROC, layer {layer}",
    )
    write_matrix(
        run_dir,
        "matrix_cosine",
        cube.cosine[k],
        rows=cube.types,
        cols=cube.types,
        title=f"Cosine, layer {layer}",
    )
    run_dir.write_arrays(
        "transfer",
        auroc=cube.transfer,
        cosine=cube.cosine,
        val=cube.val,
        test=cube.test,
        layers=np.asarray(cube.layers, dtype=np.int64),
    )
    run_dir.write_json(
        "transfer", {"types": list(cube.types), "methods": list(cube.methods)}
    )
    run_dir.write_arrays(
        "readers",
        directions=np.stack([cube.readers[n].directions for n in cube.types]),
        means=np.stack([cube.readers[n].means for n in cube.types]),
        signs=np.stack([cube.readers[n].signs for n in cube.types]),
        layers=np.asarray(cube.layers, dtype=np.int64),
    )
    run_dir.write_json("readers", {"types": list(cube.types)})
    run_dir.write_json(
        "table",
        {
            "columns": [
                "type",
                "contrast",
                "n_positive_train",
                "n_negative_train",
                "n_positive_test",
                "n_negative_test",
                cube.methods[0],
                "lo",
                "hi",
                *cube.methods[1:],
            ],
            "rows": _table_rows(cube, usable, k, config.seed),
        },
    )
    run_dir.write_json(
        "summary",
        {
            "layer": layer,
            "layer_source": "config"
            if config.layer is not None
            else "validation",
            "template": config.template,
            "methods": list(cube.methods),
            "types": list(cube.types),
            "contrasts": {n: usable[n].contrast for n in cube.types},
            "known_contrasts": known_contrasts,
            "skipped": skipped,
            "curve_stems": curve_stems,
            "matrix_stems": ["matrix", "matrix_cosine"],
            "best_layer_by_type": {
                n: best_layer(cube.val[0, t].mean(axis=0), cube.layers)
                for t, n in enumerate(cube.types)
            },
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the labeled pool and the model, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.

    Raises:
        ValueError: If `config.labels_run` is not a finished run, or is a
            run of another model.
    """
    items, summary = load_labeled(config.labels_run, model=config.model)
    execute(
        config,
        run_dir,
        load_model(config.model),
        items,
        known_contrasts=bool(summary["known_contrasts"]),
    )


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws every curve file and the two matrices of a P9b run.

    Args:
        run_dir: A finished P9b run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of each stem of `summary.curve_stems`
        and of `matrix` and `matrix_cosine`.
    """
    summary = cast(dict[str, Any], run_dir.read_json("summary"))
    paths: list[Path] = []
    for stem in summary["curve_stems"]:
        paths += plot_curves_file(run_dir, out_dir, stem=stem)
    paths += plot_matrix_file(
        run_dir, out_dir, stem="matrix", vmin=0.0, vmax=1.0
    )
    paths += plot_matrix_file(
        run_dir,
        out_dir,
        stem="matrix_cosine",
        kind="diverging",
        vmin=-1.0,
        vmax=1.0,
    )
    return paths
