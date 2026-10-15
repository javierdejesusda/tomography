"""Reading the finished runs that the aggregate experiments build on.

An aggregate experiment (A1, A2 and the study experiments) reads the results
of other experiments instead of collecting activations. Runs are found under
the aggregate's own out root, at the path of the default config of each
source on the requested model; a run with other settings is named explicitly.
There is no directory scan and no latest-run rule, so a changed default makes
a new hash and a stale run is never read silently.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
from typing import cast

import numpy as np
from numpy.typing import NDArray

from tomography.cli import load_experiment
from tomography.experiments.base import RunDir
from tomography.reading.metrics import best_layer

_READERS_STEM = re.compile(r"readers(_[A-Za-z0-9._-]+)?")
_TIE = 1e-9


@dataclass(frozen=True, eq=False)
class ReaderSet:
    """The reading directions of one finished run.

    Attributes:
        layers: The K layers the directions were fitted at.
        vectors: Unit directions of shape (N, K, d): seed, layer, width,
            oriented so that label 1 projects higher.
        n_layers: Depth L of the model the run read.
        config_hash: Config hash of the run, or None for a set built by hand.
    """

    layers: tuple[int, ...]
    vectors: NDArray[np.float64]
    n_layers: int
    config_hash: str | None = None


@dataclass(frozen=True, eq=False)
class Curves:
    """The arrays and sidecar of one `write_curves` pair.

    Attributes:
        x: The layer of each of the K points.
        mean: Series means, shape (S, K).
        lo: Lower band bounds, shape (S, K).
        hi: Upper band bounds, shape (S, K).
        per_seed: Values of each seed, shape (S, N, K).
        series: The name of each series.
        n_layers: Depth of the model, or None if the writer gave none.
        chance: Level of the chance line, or None if there is none.
    """

    x: NDArray[np.float64]
    mean: NDArray[np.float64]
    lo: NDArray[np.float64]
    hi: NDArray[np.float64]
    per_seed: NDArray[np.float64]
    series: tuple[str, ...]
    n_layers: int | None
    chance: float | None


@dataclass(frozen=True, eq=False)
class ReadingPoint:
    """One reading run at its selected layer.

    Attributes:
        layer: The selected layer.
        n_layers: Depth of the model the run read.
        values: Per-seed values of the reported split at `layer`, shape (N,).
        selected_on: The stem of the curves the layer was chosen on.
    """

    layer: int
    n_layers: int
    values: NDArray[np.float64]
    selected_on: str


def out_root_of(run_dir: RunDir) -> Path:
    """Returns the out root of a run.

    Args:
        run_dir: A run laid out as `<out_root>/<experiment>/<model>/<hash>`.

    Returns:
        The folder three levels above the run.
    """
    return run_dir.path.parents[2]


def finished_run(path: str | Path) -> RunDir:
    """Opens a run and checks that it finished.

    Args:
        path: A run directory.

    Returns:
        The run.

    Raises:
        ValueError: If `path` is not a run directory or the run has no
            `finished_at` stamp.
    """
    run_dir = RunDir.open(path)
    if run_dir.meta["finished_at"] is None:
        raise ValueError(f"{path} is not a finished run")
    return run_dir


def default_run(out_root: str | Path, experiment: str, model: str) -> RunDir:
    """Opens the finished run of an experiment's default config on a model.

    Args:
        out_root: Root folder of all runs.
        experiment: The experiment id, such as "r1".
        model: Key of the checkpoint the run read.

    Returns:
        The run at `RunDir.path_for` of the default config.

    Raises:
        ValueError: If the experiment is unknown, no run is at that path or
            the run did not finish.
    """
    config = load_experiment(experiment).config_cls.from_dict(
        {"experiment": experiment, "model": model}
    )
    path = RunDir.path_for(config, out_root=out_root)
    if not path.is_dir():
        raise ValueError(
            f"no run of {experiment} on {model} with the default config "
            f"at {path}"
        )
    return finished_run(path)


def resolve_runs(
    out_root: str | Path,
    experiments: Sequence[str],
    models: Sequence[str],
    extra_runs: Sequence[str] = (),
) -> dict[tuple[str, str], RunDir]:
    """Maps every requested (experiment, model) pair to its finished run.

    Args:
        out_root: Root folder of all runs.
        experiments: The experiment ids.
        models: The checkpoint keys.
        extra_runs: Finished runs with other settings. Each replaces the
            default run of its own experiment and model.

    Returns:
        The run of each pair, experiments first and models second.

    Raises:
        ValueError: If a run is missing or unfinished, two extra runs belong
            to one pair, or an extra run belongs to a pair that was not
            requested.
    """
    pairs = [
        (e, m)
        for e in dict.fromkeys(experiments)
        for m in dict.fromkeys(models)
    ]
    chosen: dict[tuple[str, str], RunDir] = {}
    for path in extra_runs:
        run_dir = finished_run(path)
        pair = (run_dir.config["experiment"], run_dir.config["model"])
        if pair not in pairs:
            raise ValueError(
                f"extra run {path} is a run of {pair[0]} on {pair[1]}, "
                "which was not requested"
            )
        if pair in chosen:
            raise ValueError(
                f"two extra runs of {pair[0]} on {pair[1]}: "
                f"{chosen[pair].path} and {path}"
            )
        chosen[pair] = run_dir
    return {
        pair: chosen[pair] if pair in chosen else default_run(out_root, *pair)
        for pair in pairs
    }


def oriented(
    directions: NDArray[Any], signs: NDArray[Any]
) -> NDArray[np.float64]:
    """Normalizes reading directions and applies their orientation.

    Args:
        directions: Directions of shape (N, K, d).
        signs: Orientation of each direction, shape (N, K), -1 or +1.

    Returns:
        Unit directions of shape (N, K, d) in float64, oriented so that the
        label-1 class projects higher.

    Raises:
        ValueError: If `signs` does not match the leading axes of
            `directions` or a direction is zero.
    """
    vectors = np.asarray(directions, dtype=np.float64)
    orientation = np.asarray(signs, dtype=np.float64)
    if orientation.shape != vectors.shape[:-1]:
        raise ValueError(
            f"signs must have shape {vectors.shape[:-1]}, "
            f"got {orientation.shape}"
        )
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if (norms == 0.0).any():
        raise ValueError("directions must not contain a zero vector")
    return np.asarray(vectors / norms * orientation[..., None])


def run_depth(run: RunDir) -> int:
    """Returns the depth of the model a run read.

    Args:
        run: A finished run with at least one `curves*.json` sidecar.

    Returns:
        The `n_layers` that every sidecar with a value agrees on.

    Raises:
        ValueError: If no sidecar holds an `n_layers` or they disagree.
    """
    depths: set[int] = set()
    for path in sorted(run.path.glob("curves*.json")):
        sidecar = cast(dict[str, Any], run.read_json(path.stem))
        if sidecar.get("n_layers") is not None:
            depths.add(int(sidecar["n_layers"]))
    if len(depths) != 1:
        raise ValueError(
            f"the curves sidecars of {run.path} must hold exactly one "
            f"n_layers, got {sorted(depths)}"
        )
    return depths.pop()


def read_readers(run: RunDir) -> dict[str, ReaderSet]:
    """Reads every readers file of a run.

    Args:
        run: A finished run that wrote `readers.npz` or
            `readers_<suffix>.npz`, with arrays `layers (K,)`,
            `directions (N, K, d)` and `signs (N, K)`.

    Returns:
        One set per file, in file-name order, labeled `<experiment>` for
        `readers.npz` and `<experiment>:<suffix>` for `readers_<suffix>.npz`.
        Other file names are ignored.

    Raises:
        ValueError: If the run has no readers file, the depth is not
            defined, or an array does not fit the others.
    """
    stems = sorted(
        path.stem
        for path in run.path.glob("readers*.npz")
        if _READERS_STEM.fullmatch(path.stem)
    )
    if not stems:
        raise ValueError(f"{run.path} has no readers file")
    n_layers = run_depth(run)
    experiment = run.config["experiment"]
    found: dict[str, ReaderSet] = {}
    for stem in stems:
        arrays = run.read_arrays(stem)
        layers = tuple(int(layer) for layer in arrays["layers"])
        vectors = oriented(arrays["directions"], arrays["signs"])
        if vectors.shape[1] != len(layers):
            raise ValueError(
                f"{stem}.npz holds {len(layers)} layers for "
                f"{vectors.shape[1]} directions per seed"
            )
        suffix = stem.removeprefix("readers").removeprefix("_")
        label = f"{experiment}:{suffix}" if suffix else experiment
        found[label] = ReaderSet(
            layers, vectors, n_layers, str(run.meta["config_hash"])
        )
    return found


def common_layers(
    readers: Mapping[str, ReaderSet], max_series: int = 8
) -> tuple[tuple[int, ...], int, int]:
    """Checks that reader sets can be compared and returns what they share.

    Args:
        readers: The sets by label.
        max_series: The most sets one figure can draw.

    Returns:
        The shared layers, the shared depth and the smallest seed count.

    Raises:
        ValueError: If there is no set, there are more than `max_series`, or
            the sets were read at different layers or depths.
    """
    if not readers:
        raise ValueError("no reader sets")
    if len(readers) > max_series:
        raise ValueError(
            f"got {len(readers)} reader sets, at most {max_series} fit a figure"
        )
    first = next(iter(readers.values()))
    for label, readers_of in readers.items():
        if readers_of.layers != first.layers:
            raise ValueError(
                f"{label} was read at layers {readers_of.layers}, "
                f"not {first.layers}"
            )
        if readers_of.n_layers != first.n_layers:
            raise ValueError(
                f"{label} read a model of depth {readers_of.n_layers}, "
                f"not {first.n_layers}"
            )
    n_min = min(r.vectors.shape[0] for r in readers.values())
    return first.layers, first.n_layers, n_min


def gather_readers(
    out_root: str | Path,
    sources: Sequence[str],
    model: str,
    extra_runs: Sequence[str] = (),
) -> dict[str, ReaderSet]:
    """Reads the reader sets of several source experiments on one model.

    Args:
        out_root: Root folder of all runs.
        sources: The source experiment ids.
        model: Key of the checkpoint the sources read.
        extra_runs: Finished runs with other settings that replace the
            default run of their experiment.

    Returns:
        The sets of every source, concatenated in source order.

    Raises:
        ValueError: If a run cannot be found, as `resolve_runs`, or a run has
            no readers, as `read_readers`.
    """
    runs = resolve_runs(out_root, sources, [model], extra_runs)
    gathered: dict[str, ReaderSet] = {}
    for run_dir in runs.values():
        gathered.update(read_readers(run_dir))
    return gathered


def read_curves(run: RunDir, stem: str = "curves") -> Curves:
    """Reads the arrays and the sidecar of a `write_curves` pair.

    A `write_dose_curves` pair has the same layout.

    Args:
        run: The run holding the pair.
        stem: File stem of the pair.

    Returns:
        The curves.
    """
    arrays = run.read_arrays(stem)
    sidecar = cast(dict[str, Any], run.read_json(stem))
    n_layers = sidecar["n_layers"]
    chance = sidecar["chance"]
    return Curves(
        x=arrays["x"].astype(np.float64),
        mean=arrays["mean"].astype(np.float64),
        lo=arrays["lo"].astype(np.float64),
        hi=arrays["hi"].astype(np.float64),
        per_seed=arrays["per_seed"].astype(np.float64),
        series=tuple(sidecar["series"]),
        n_layers=None if n_layers is None else int(n_layers),
        chance=None if chance is None else float(chance),
    )


def best_point(
    curves: Curves, series: str = "reader", select: Curves | None = None
) -> tuple[int, NDArray[np.float64]]:
    """Finds the best layer of a series and its value for every seed.

    Args:
        curves: The curves to report.
        series: Name of the series.
        select: Curves to choose the layer on, read at the same layers as
            `curves`; None chooses on `curves` itself.

    Returns:
        The layer with the highest mean in `select`, the first on ties, and
        the values of the N seeds of `curves` at that layer. Runs that share
        seeds compare through these paired values.

    Raises:
        ValueError: If the series is not in `curves` or in `select`, or
            `select` was read at other layers.
    """
    if series not in curves.series:
        raise ValueError(
            f"unknown series {series!r}, expected one of {curves.series}"
        )
    chooser = curves if select is None else select
    if series not in chooser.series:
        raise ValueError(
            f"unknown series {series!r} in select, expected one of "
            f"{chooser.series}"
        )
    if not np.array_equal(chooser.x, curves.x):
        raise ValueError(
            f"select was read at layers {chooser.x.tolist()}, "
            f"not {curves.x.tolist()}"
        )
    layers = [int(layer) for layer in curves.x]
    layer = best_layer(chooser.mean[chooser.series.index(series)], layers)
    index = curves.series.index(series)
    return layer, curves.per_seed[index, :, layers.index(layer)]


def reading_point(
    run: RunDir,
    series: str = "reader",
    report: str = "curves",
    select: str = "curves_val",
) -> ReadingPoint:
    """Applies one layer-selection rule to a reading run.

    The layer is chosen on the validation curves when the run writes them
    and on the reported curves otherwise, so a selection and a report on one
    split stay visible through `selected_on`.

    Args:
        run: A finished reading run.
        series: Name of the series.
        report: Stem of the curves to report.
        select: Stem of the curves to choose the layer on, when the run
            wrote them.

    Returns:
        The selected layer, the depth of the model, the per-seed values of
        the reported curves there and the stem the layer was chosen on.

    Raises:
        ValueError: If the series is unknown or the reported curves have no
            `n_layers`.
    """
    reported = read_curves(run, report)
    if reported.n_layers is None:
        raise ValueError(f"the {report} curves of {run.path} have no n_layers")
    chooser: Curves | None = None
    selected_on = report
    if (run.path / f"{select}.npz").is_file():
        chooser = read_curves(run, select)
        selected_on = select
    layer, values = best_point(reported, series, chooser)
    return ReadingPoint(layer, reported.n_layers, values, selected_on)


def relative_depth(layers: Sequence[int], n_layers: int) -> NDArray[np.float64]:
    """Converts layer indices to fractions of the model depth.

    Args:
        layers: Layer indices in 0..n_layers.
        n_layers: Depth L of the model.

    Returns:
        `layer / L` for every layer.

    Raises:
        ValueError: If a layer is outside 0..L.
    """
    values = np.asarray(layers, dtype=np.float64)
    if n_layers < 1 or (values < 0).any() or (values > n_layers).any():
        raise ValueError(
            f"every layer must be in 0..{n_layers}, got {list(layers)}"
        )
    return np.asarray(values / n_layers)


def depth_layer(layers: Sequence[int], depth: float, n_layers: int) -> int:
    """Picks the layer nearest a relative depth.

    Args:
        layers: The available layer indices.
        depth: Relative depth in (0, 1].
        n_layers: Depth L of the model.

    Returns:
        The layer nearest `depth * L`, the lower one on ties.

    Raises:
        ValueError: If `layers` is empty or `depth` is outside (0, 1].
    """
    if not 0.0 < depth <= 1.0:
        raise ValueError(f"depth must be in (0, 1], got {depth}")
    if len(layers) == 0:
        raise ValueError("layers is empty")
    values = np.asarray(layers, dtype=np.float64)
    distance = np.abs(values - depth * n_layers)
    return int(values[distance <= distance.min() + _TIE].min())


def label_stem(label: str) -> str:
    """Turns a reader-set label into a text usable in file names.

    Args:
        label: `<experiment>` or `<experiment>:<suffix>`.

    Returns:
        The label with the colon replaced by an underscore.
    """
    return label.replace(":", "_")
