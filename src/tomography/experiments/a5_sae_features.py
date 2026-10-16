"""A5: decomposition of the reading directions over SAE features.

A5 asks what a reading direction encodes by reading it against the decoder of
the residual sparse autoencoder of its layer. It reads no activations: the
directions come from the `readers*.npz` files of finished reading runs, and
the model is loaded only for its unembedding, final norm and tokenizer.
Decoder weights come from the project cache, one layer at a time.
"""

from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
import functools
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tomography.analysis import column_norms
from tomography.analysis import decoder_cosines
from tomography.analysis import dictionary_directions
from tomography.analysis import lens_logits
from tomography.analysis import mean_direction
from tomography.analysis import sparse_fit
from tomography.analysis import top_features
from tomography.analysis import top_overlap
from tomography.experiments.a2_logit_lens import lens_params
from tomography.experiments.aggregate import ReaderSet
from tomography.experiments.aggregate import common_layers
from tomography.experiments.aggregate import gather_readers
from tomography.experiments.aggregate import out_root_of
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.matrices import plot_matrix_file
from tomography.experiments.matrices import write_matrix
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import write_curves
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.metrics import best_layer
from tomography.sae import SAES
from tomography.sae import Decoders
from tomography.sae import SaeSpec
from tomography.sae import load_decoder
from tomography.sae import sae_layer

SHOWN_TOKENS = 5
_QUANTILE = 0.95
_MIN_SEEDS = 2

Lens = Callable[[NDArray[np.float32]], NDArray[np.float32]]


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """A5 settings.

    Attributes:
        sources: Experiments whose default runs on `model` are read.
        extra_runs: Finished runs that replace a default run.
        sae: Key of `SAES`, the autoencoder release to decompose over.
        layers: Reading layers to analyze; None means every layer of the
            runs that has an autoencoder (1..L).
        top_k: Features per concept and layer in the table, and the size of
            the id sets whose overlap across seeds is the stability.
        n_atoms: Atoms of the sparse fit, at most the residual width.
        n_random: Draws of the matched null per layer, at least 2.
        n_boot: Bootstrap resamples of the curve bands, at least 1.
    """

    experiment: str = "a5"
    model: str = "qwen3-1.7b-base"
    sources: tuple[str, ...] = ("r1", "r2", "r3", "r5", "r7")
    extra_runs: tuple[str, ...] = ()
    sae: str = "qwen-scope-1.7b-base-l0-50"
    layers: tuple[int, ...] | None = None
    top_k: int = 10
    n_atoms: int = 50
    n_random: int = 64
    n_boot: int = 2000


@dataclass(frozen=True)
class _Plan:
    layers: list[int]
    skipped: list[int]
    n_layers: int
    n_min: int


@dataclass(frozen=True, eq=False)
class _Layer:
    features: NDArray[np.int64]
    cosines: NDArray[np.float32]
    null_q95: NDArray[np.float64]
    omp_rank: NDArray[np.int64]
    tokens: list[list[list[str]]]
    excess: NDArray[np.float64]
    reading: NDArray[np.float64]
    null_curve: NDArray[np.float64]
    stability: NDArray[np.float64]
    null_mean: float
    null_std: float
    top1_q95: float
    isotropic_top1_q95: float
    norm_deviation: float


def _check_config(config: Config) -> None:
    """Refuses a config the analysis cannot use.

    Args:
        config: The experiment settings.

    Raises:
        ValueError: If `top_k` or `n_atoms` is below 1, `n_random` is below
            2, `n_boot` is below 1, `seed` is negative, `sae` is not in
            `SAES`, or `layers` is empty or repeats a layer.
    """
    if config.top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {config.top_k}")
    if config.n_atoms < 1:
        raise ValueError(f"n_atoms must be at least 1, got {config.n_atoms}")
    if config.n_random < 2:
        raise ValueError(f"n_random must be at least 2, got {config.n_random}")
    if config.n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {config.n_boot}")
    if config.seed < 0:
        raise ValueError(f"seed must be non-negative, got {config.seed}")
    if config.sae not in SAES:
        raise ValueError(
            f"unknown sae {config.sae!r}, expected one of {sorted(SAES)}"
        )
    if config.layers is not None:
        if not config.layers:
            raise ValueError("layers must not be empty")
        if len(set(config.layers)) != len(config.layers):
            raise ValueError(
                f"layers must not repeat a layer, got {config.layers}"
            )


def _check_readers(
    config: Config, readers: Mapping[str, ReaderSet], spec: SaeSpec
) -> _Plan:
    """Checks that the readers fit the autoencoder and picks the layers.

    Args:
        config: The experiment settings.
        readers: The reader sets by concept label.
        spec: The autoencoder release.

    Returns:
        The reading layers to analyze, the layers of the runs that have no
        autoencoder, the depth of the model and the smallest seed count.

    Raises:
        ValueError: If the sets cannot be compared (see `common_layers`),
            their depth or width differs from the autoencoder's, a concept has
            fewer than two seeds, `n_atoms` exceeds the smaller of the width
            and the number of features, `top_k` exceeds the number of
            features, a requested layer is outside 1..L or not in the runs,
            or none of the runs' layers has an autoencoder.
    """
    layers, n_layers, n_min = common_layers(readers)
    if n_layers != spec.n_layers:
        raise ValueError(
            f"the readers come from a model of depth {n_layers}, but "
            f"{spec.key} has {spec.n_layers} layers"
        )
    for label, reader_set in readers.items():
        width = reader_set.vectors.shape[2]
        if width != spec.d_model:
            raise ValueError(
                f"{label} has vectors of width {width}, but {spec.key} has "
                f"width {spec.d_model}"
            )
    if n_min < _MIN_SEEDS:
        raise ValueError(
            f"A5 needs at least {_MIN_SEEDS} seeds per concept, got {n_min}"
        )
    atom_limit = min(spec.d_model, spec.width)
    if config.n_atoms > atom_limit:
        raise ValueError(
            f"n_atoms must be at most {atom_limit}, the smaller of the width "
            f"and the number of features, got {config.n_atoms}"
        )
    if config.top_k > spec.width:
        raise ValueError(
            f"top_k must be at most the {spec.width} features, "
            f"got {config.top_k}"
        )
    for layer in config.layers or ():
        sae_layer(layer, spec)
        if layer not in layers:
            raise ValueError(
                f"layer {layer} was not read by the source runs, which "
                f"hold {list(layers)}"
            )
    covered = [layer for layer in layers if 1 <= layer <= spec.n_layers]
    if not covered:
        raise ValueError("no layer of the source runs has an autoencoder")
    analyzed = [
        layer for layer in covered if layer in (config.layers or layers)
    ]
    skipped = [layer for layer in layers if layer not in covered]
    return _Plan(analyzed, skipped, n_layers, n_min)


def _omp_ranks(
    features: NDArray[np.int64], atoms: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Gives each feature's 1-based position among the atoms, 0 if absent."""
    hits = atoms[:, None, :] == features[:, :, None]
    return np.where(hits.any(axis=-1), hits.argmax(axis=-1) + 1, 0)


def _promoted_tokens(
    decoder: NDArray[np.float32],
    features: NDArray[np.int64],
    toward: NDArray[np.float64],
    lens: Lens,
    decode: Callable[[int], str],
) -> list[list[list[str]]]:
    """Reads the signed feature columns through the lens.

    Returns:
        For each concept and rank, the `SHOWN_TOKENS` highest tokens of the
        lens of `toward * column`.
    """
    columns = decoder[:, features.ravel()].T * toward.reshape(-1, 1)
    logits = lens(columns.astype(np.float32)).reshape(*features.shape, -1)
    return [
        [
            [
                decode(int(t))
                for t in np.argsort(-row, kind="stable")[:SHOWN_TOKENS]
            ]
            for row in concept
        ]
        for concept in logits
    ]


def _decompose(
    config: Config,
    layer: int,
    decoder: NDArray[np.float32],
    seeds: NDArray[np.float64],
    means: NDArray[np.float64],
    lens: Lens,
    decode: Callable[[int], str],
) -> _Layer:
    n_concepts, n_seeds, width = seeds.shape
    k, n_random = config.top_k, config.n_random
    null = dictionary_directions(
        decoder,
        n_random,
        np.random.default_rng(derive_seed(config.seed, "dictionary", layer)),
    )
    isotropic = np.random.default_rng(
        derive_seed(config.seed, "isotropic", layer)
    ).standard_normal((n_random, width))
    fitted = np.concatenate([seeds.reshape(-1, width), means, null])
    explained, atoms = sparse_fit(fitted, decoder, config.n_atoms)
    cosines = decoder_cosines(fitted, decoder)
    seed_rows = slice(0, n_concepts * n_seeds)
    mean_rows = slice(seed_rows.stop, seed_rows.stop + n_concepts)
    null_rows = slice(mean_rows.stop, None)

    seed_top = top_features(cosines[seed_rows], k).reshape(
        n_concepts, n_seeds, k
    )
    null_magnitude = np.abs(cosines[null_rows])
    null_top = top_features(null_magnitude, k)
    stability = np.array(
        [top_overlap(top) for top in seed_top] + [top_overlap(null_top)]
    )
    ranked = np.take_along_axis(null_magnitude, null_top, axis=1)
    features = top_features(cosines[mean_rows], k)
    signed = np.take_along_axis(cosines[mean_rows], features, axis=1)
    toward = np.where(signed < 0.0, -1.0, 1.0)
    final = explained[:, -1]
    null_final = final[null_rows]
    null_mean = float(null_final.mean())
    isotropic_top1 = np.abs(decoder_cosines(isotropic, decoder)).max(axis=1)
    return _Layer(
        features=features,
        cosines=signed,
        null_q95=np.quantile(ranked, _QUANTILE, axis=0),
        omp_rank=_omp_ranks(features, atoms[mean_rows]),
        tokens=_promoted_tokens(decoder, features, toward, lens, decode),
        excess=final[seed_rows].reshape(n_concepts, n_seeds) - null_mean,
        reading=explained[mean_rows],
        null_curve=explained[null_rows].mean(axis=0),
        stability=stability,
        null_mean=null_mean,
        null_std=float(null_final.std(ddof=1)),
        top1_q95=float(np.quantile(ranked[:, 0], _QUANTILE)),
        isotropic_top1_q95=float(np.quantile(isotropic_top1, _QUANTILE)),
        norm_deviation=float(np.abs(column_norms(decoder) - 1.0).max()),
    )


def _table_rows(
    labels: list[str], layers: list[int], results: list[_Layer], top_k: int
) -> list[list[Any]]:
    return [
        [
            label,
            layer,
            r + 1,
            int(result.features[i, r]),
            float(result.cosines[i, r]),
            float(result.null_q95[r]),
            int(result.omp_rank[i, r]),
            result.tokens[i][r],
        ]
        for i, label in enumerate(labels)
        for layer, result in zip(layers, results, strict=True)
        for r in range(top_k)
    ]


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    readers: Mapping[str, ReaderSet],
    spec: SaeSpec,
    decoders: Decoders,
) -> None:
    """Decomposes the reading directions over the SAE features of each layer.

    Every concept uses its first `n_min` seeds, `n_min` being the smallest
    seed count among the concepts. At each analyzed reading layer `l` the
    decoder of autoencoder `l - 1` is loaded (our layer `l` is the output of
    block `l`, which autoencoder `l - 1` reads) and every direction is read
    against its unit columns. Everything is compared with a null drawn per
    layer: unit directions with the covariance of the decoder, which keeps
    the anisotropy of the dictionary, plus an isotropic Gaussian kept only as
    a reference for the top cosine.

    Writes `table.json` (the `top_k` features of each concept's mean
    direction by absolute cosine, with the sign, the null quantile of the
    same rank, the position among the atoms of the sparse fit and the five
    tokens the signed feature promotes through the lens), `curves_explained`
    (per seed, the fraction of the direction explained by `n_atoms` atoms
    minus the mean of the null), `matrix_stability` (the overlap of the top
    ids across seeds, with a null row), `explained.npz` (the fit of each mean
    direction after 1..`n_atoms` atoms and the null means) and `summary.json`.
    The summary lists the layers analyzed and the layers without an
    autoencoder (`skipped_layers`), the largest deviation of a decoder column
    norm from 1, the null statistics per layer, whether the model differs
    from the one the autoencoder was trained on (`transfer`) and, for every
    concept, the config hash of the run it was read from (`source_runs`).

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model; only its unembedding, final norm and
            tokenizer are used.
        readers: The reader sets by concept label, at most eight, read at the
            same layers of the same model.
        spec: The autoencoder release the decoders belong to.
        decoders: Returns the decoder of an autoencoder layer; called once
            per analyzed layer, in layer order.

    Raises:
        ValueError: If `top_k` or `n_atoms` is below 1, `n_random` is below
            2, `n_boot` is below 1, `seed` is negative, `sae` is unknown or
            `layers` is empty or repeats a layer, or `spec` is not the release
            `sae` names; if the readers cannot be
            compared, come from a model whose depth or width differs from the
            autoencoder's, hold fewer than two seeds for a concept, or do not
            cover a requested layer or have no layer with an autoencoder; or
            if `n_atoms` exceeds the smaller of the width and the number of
            features, or `top_k` the number of features. The model and the
            decoders are not touched before these checks.
    """
    _check_config(config)
    if spec.key != config.sae:
        raise ValueError(
            f"spec {spec.key!r} is not the release {config.sae!r} that the "
            "config names"
        )
    plan = _check_readers(config, readers, spec)
    labels = list(readers)
    unembedding, scale, center = lens_params(loaded)
    vectors = {label: readers[label].vectors[: plan.n_min] for label in labels}
    centers = {label: mean_direction(vectors[label]) for label in labels}
    read_layers = next(iter(readers.values())).layers

    def lens(directions: NDArray[np.float32]) -> NDArray[np.float32]:
        return lens_logits(directions, unembedding, scale=scale, center=center)

    def decode(token_id: int) -> str:
        return str(loaded.tokenizer.decode([token_id]))

    results: list[_Layer] = []
    for layer in plan.layers:
        at = read_layers.index(layer)
        results.append(
            _decompose(
                config,
                layer,
                decoders(sae_layer(layer, spec)),
                np.stack([vectors[label][:, at] for label in labels]),
                np.stack([centers[label][at] for label in labels]),
                lens,
                decode,
            )
        )
    run_dir.write_json(
        "table",
        {
            "columns": [
                "concept",
                "layer",
                "rank",
                "feature",
                "cos",
                "null_q95",
                "omp_rank",
                "tokens",
            ],
            "n_layers": plan.n_layers,
            "rows": _table_rows(labels, plan.layers, results, config.top_k),
        },
    )
    excess = np.stack([result.excess for result in results], axis=-1)
    write_curves(
        run_dir,
        excess,
        plan.layers,
        series=labels,
        stem="curves_explained",
        ylabel="Explained fraction above the matched null",
        chance=0.0,
        n_layers=plan.n_layers,
        n_boot=config.n_boot,
        seed=config.seed,
    )
    write_matrix(
        run_dir,
        "matrix_stability",
        np.stack([result.stability for result in results], axis=1),
        rows=[*labels, "null"],
        cols=[str(layer) for layer in plan.layers],
        title=f"Overlap of the top {config.top_k} features across seeds",
    )
    run_dir.write_arrays(
        "explained",
        reading=np.stack([result.reading for result in results], axis=1),
        null=np.stack([result.null_curve for result in results]),
        layers=np.asarray(plan.layers, dtype=np.int64),
    )
    mean_excess = excess.mean(axis=1)
    run_dir.write_json(
        "summary",
        {
            "concepts": labels,
            "layers": plan.layers,
            "skipped_layers": plan.skipped,
            "n_seeds": plan.n_min,
            "top_k": config.top_k,
            "n_atoms": config.n_atoms,
            "sae": {
                "key": spec.key,
                "repo_id": spec.repo_id,
                "revision": spec.revision,
                "k": spec.k,
                "width": spec.width,
                "model": spec.model,
            },
            "transfer": config.model != spec.model,
            "decoder_norm_deviation": max(r.norm_deviation for r in results),
            "null": {
                "dictionary": {
                    "explained_mean": [r.null_mean for r in results],
                    "explained_std": [r.null_std for r in results],
                    "top1_q95": [r.top1_q95 for r in results],
                },
                "isotropic": {
                    "top1_q95": [r.isotropic_top1_q95 for r in results]
                },
            },
            "best_layer": {
                label: best_layer(mean_excess[i], plan.layers)
                for i, label in enumerate(labels)
            },
            "source_runs": {
                label: readers[label].config_hash for label in labels
            },
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Gathers the readers of the source runs, loads the model and analyzes.

    The source runs are looked up under the out root of `run_dir`. The config
    and the readers are checked before the model is loaded or any decoder is
    downloaded.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.

    Raises:
        ValueError: If the config is unusable, the source runs cannot be
            gathered or the readers do not fit the autoencoder (see
            `execute`).
    """
    _check_config(config)
    readers = gather_readers(
        out_root_of(run_dir), config.sources, config.model, config.extra_runs
    )
    spec = SAES[config.sae]
    _check_readers(config, readers, spec)
    execute(
        config,
        run_dir,
        load_model(config.model),
        readers,
        spec,
        functools.partial(load_decoder, spec),
    )


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the explained-fraction curves and the stability matrix.

    Args:
        run_dir: A finished A5 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of `curves_explained` and of
        `matrix_stability`.
    """
    paths = plot_curves_file(run_dir, out_dir, stem="curves_explained")
    paths.extend(
        plot_matrix_file(
            run_dir,
            out_dir,
            stem="matrix_stability",
            kind="sequential",
            vmin=0.0,
            vmax=1.0,
        )
    )
    return paths
