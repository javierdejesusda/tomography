"""P10c: links of the range direction to knowledge and to hedging.

P10b reads a direction that separates questions whose true answer is a range
(aleatoric) from questions with one exact answer. Two links are measured on
finished runs, so no earlier config hash changes.

Knowledge: P9b reads directions that separate known from unknown questions
(epistemic). Are the two kinds of direction the same, opposite or unrelated in
one model? The cosine between the stored directions answers it layer by
layer, and so does the AUROC of each stored P9 reader on the P10 items.

Behavior: within one kind of question, does the P10b score of a test item
predict how often the model hedges (answers with a range) on the P10a run?
Across kinds the link is trivial, because the reader separates the kinds and
the behavior differs by kind.

The inputs `readers.npz` and `items.json` are local only (export drops them),
so this experiment runs where the full run directories exist.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Final
from typing import cast

import numpy as np
from numpy.typing import NDArray

from tomography.datasets.quantities import PEOPLE_FAMILY
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.p10b_reading import TEMPLATES
from tomography.p9.reading import READING_METHODS

READER_PARTS: Final = 2

Record = dict[str, Any]
Snapshots = dict[str, dict[str, dict[str, list[str]]]]


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """P10c settings: the finished runs to link and the reader to use.

    Attributes:
        p10b_run: Path of a finished P10b run of the same model.
        knowledge_run: Path of a finished P9b run of the same model, or None
            to skip the knowledge link.
        behavior_run: Path of a finished P10a run of the same model, or None
            to skip the behavior link.
        reader: The P10b reader, named "template/method".
        n_boot: Resamples of every bootstrap interval.
        n_random: Random unit directions of every random band.
        batch_size: Prompts per forward pass.
        offline: Whether the questions are read from the cache only.
    """

    experiment: str = "p10c"
    model: str = "gpt2"
    p10b_run: str
    knowledge_run: str | None = None
    behavior_run: str | None = None
    reader: str = "plain/logistic"
    n_boot: int = 1000
    n_random: int = 200
    batch_size: int = 8
    offline: bool = True

    def __post_init__(self) -> None:
        """Checks the settings that would act silently or fail late.

        Raises:
            ValueError: If neither `knowledge_run` nor `behavior_run` is
                set, `reader` is not a "template/method" name of P10b, or a
                count is below 1.
        """
        if self.knowledge_run is None and self.behavior_run is None:
            raise ValueError(
                "set at least one of knowledge_run and behavior_run"
            )
        parts = self.reader.split("/")
        if (
            len(parts) != READER_PARTS
            or parts[0] not in TEMPLATES
            or parts[1] not in READING_METHODS
        ):
            raise ValueError(
                f"reader must be 'template/method' with a template in "
                f"{TEMPLATES} and a method in {READING_METHODS}, got "
                f"{self.reader!r}"
            )
        for name in ("n_boot", "n_random", "batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")


@dataclass(frozen=True, eq=False)
class P10bRun:
    """What the links read from a finished P10b run.

    The saved direction is the one of draw 0 fitted on the full train set,
    while the saved item scores are the average over the draws of the score
    of the same reader at its chosen layer. Positive scores mean range, and
    the train items are scored in-sample.

    Attributes:
        path: The run directory, as given.
        config_hash: The hash of the run's config.
        config: The run's config as JSON values.
        reader: The reader read, named "template/method".
        layer: The layer chosen for the reader on the validation items.
        layers: The layer index of each row of `directions`.
        directions: The reader's unit directions, shape (K, d).
        signs: The reader's orientation per layer, shape (K,), +1 when a
            positive projection means range.
        items: Every item of the run: `uid`, `family`, `kind`, `split`,
            `sitelinks` and `scores` by reader name.
        snapshots: The Wikidata snapshot stamps by family and kind.
    """

    path: str
    config_hash: str
    config: Record
    reader: str
    layer: int
    layers: tuple[int, ...]
    directions: NDArray[np.float32]
    signs: NDArray[np.float32]
    items: tuple[Record, ...]
    snapshots: Snapshots


@dataclass(frozen=True, eq=False)
class P9bRun:
    """What the links read from a finished P9b run.

    Attributes:
        path: The run directory, as given.
        config_hash: The hash of the run's config.
        config: The run's config as JSON values.
        types: The contrast of each reader, in the order of the arrays.
        layers: The layer index of each layer slice.
        directions: Unit directions of shape (N, K, d), one reader per type.
        means: The vectors subtracted before projecting, shape (N, K, d).
        signs: The orientation per type and layer, shape (N, K), +1 when a
            positive projection means known.
    """

    path: str
    config_hash: str
    config: Record
    types: tuple[str, ...]
    layers: tuple[int, ...]
    directions: NDArray[np.float32]
    means: NDArray[np.float32]
    signs: NDArray[np.float32]


@dataclass(frozen=True, eq=False)
class P10aRun:
    """What the links read from a finished P10a run.

    Attributes:
        path: The run directory, as given.
        config_hash: The hash of the run's config.
        config: The run's config as JSON values.
        items: Every question with its parsed answers, by uid. The answers
            are keyed "condition/format/order".
        snapshots: The Wikidata snapshot stamps by family and kind.
    """

    path: str
    config_hash: str
    config: Record
    items: dict[str, Record]
    snapshots: Snapshots


def _open_finished(path: str | Path, *, model: str, files: list[str]) -> RunDir:
    """Opens a run and checks that it can feed a study of `model`.

    Args:
        path: The run directory.
        model: The model the caller is about to link.
        files: The files of the run that the caller reads.

    Returns:
        The run directory.

    Raises:
        ValueError: If the directory is not a run, the run is not finished,
            it is a run of another model or one of `files` is missing (an
            export drops the arrays and the items).
    """
    run_dir = RunDir.open(path)
    if run_dir.meta.get("finished_at") is None:
        raise ValueError(f"{path} is not a finished run")
    found = run_dir.config.get("model")
    if found != model:
        raise ValueError(f"{path} is a run of {found!r}, not of {model!r}")
    missing = [name for name in files if not (run_dir.path / name).is_file()]
    if missing:
        raise ValueError(
            f"{path} has no {missing[0]}: link a full run directory, not an "
            "export"
        )
    return run_dir


def load_p10b(path: str | Path, *, model: str, reader: str) -> P10bRun:
    """Reads a finished P10b run.

    Args:
        path: The run directory.
        model: The model the caller is about to link.
        reader: The reader to read, named "template/method".

    Returns:
        The run contents.

    Raises:
        ValueError: If the run cannot be linked (see `_open_finished`), it
            has no such reader or an item has no score of the reader.
    """
    run_dir = _open_finished(
        path,
        model=model,
        files=["readers.npz", "readers.json", "items.json", "summary.json"],
    )
    stored = cast(Record, run_dir.read_json("readers"))
    names = list(stored["readers"])
    if reader not in names:
        raise ValueError(f"{path} has no reader {reader!r}; it holds {names}")
    row = names.index(reader)
    arrays = run_dir.read_arrays("readers")
    summary = cast(Record, run_dir.read_json("summary"))
    layers = tuple(int(layer) for layer in arrays["layers"])
    key = f"{reader}/{stored['train_set']}"
    if key not in summary["layers"]:
        raise ValueError(f"{path} has no chosen layer for {key!r}")
    layer = int(summary["layers"][key])
    if layer not in layers:
        raise ValueError(f"{path} chose layer {layer}, not among {layers}")
    items = tuple(cast(list[Record], run_dir.read_json("items")))
    unscored = [i["uid"] for i in items if reader not in i["scores"]]
    if unscored:
        raise ValueError(
            f"{path}: item {unscored[0]!r} has no {reader!r} score"
        )
    return P10bRun(
        path=str(path),
        config_hash=str(run_dir.meta["config_hash"]),
        config=run_dir.config,
        reader=reader,
        layer=layer,
        layers=layers,
        directions=arrays["directions"][row],
        signs=arrays["signs"][row],
        items=items,
        snapshots=summary["snapshots"],
    )


def load_p9b(path: str | Path, *, model: str) -> P9bRun:
    """Reads a finished P9b run.

    Args:
        path: The run directory.
        model: The model the caller is about to link.

    Returns:
        The run contents.

    Raises:
        ValueError: If the run cannot be linked (see `_open_finished`) or its
            arrays disagree with its types and layers.
    """
    run_dir = _open_finished(
        path, model=model, files=["readers.npz", "readers.json"]
    )
    types = tuple(cast(Record, run_dir.read_json("readers"))["types"])
    arrays = run_dir.read_arrays("readers")
    layers = tuple(int(layer) for layer in arrays["layers"])
    shape = (len(types), len(layers))
    if (
        arrays["directions"].ndim != 3
        or arrays["directions"].shape[:2] != shape
        or arrays["means"].shape != arrays["directions"].shape
        or arrays["signs"].shape != shape
    ):
        raise ValueError(
            f"{path}: the arrays do not fit {len(types)} types and "
            f"{len(layers)} layers"
        )
    return P9bRun(
        path=str(path),
        config_hash=str(run_dir.meta["config_hash"]),
        config=run_dir.config,
        types=types,
        layers=layers,
        directions=arrays["directions"],
        means=arrays["means"],
        signs=arrays["signs"],
    )


def load_p10a(path: str | Path, *, model: str) -> P10aRun:
    """Reads a finished P10a run.

    Args:
        path: The run directory.
        model: The model the caller is about to link.

    Returns:
        The run contents.

    Raises:
        ValueError: If the run cannot be linked (see `_open_finished`).
    """
    run_dir = _open_finished(
        path, model=model, files=["items.json", "summary.json"]
    )
    records = cast(list[Record], run_dir.read_json("items"))
    summary = cast(Record, run_dir.read_json("summary"))
    return P10aRun(
        path=str(path),
        config_hash=str(run_dir.meta["config_hash"]),
        config=run_dir.config,
        items={r["uid"]: r for r in records},
        snapshots=summary["snapshots"],
    )


def check_links(
    p10b: P10bRun, p9b: P9bRun | None, p10a: P10aRun | None
) -> None:
    """Checks that the runs describe the same model and the same questions.

    A P9b run must read the layers and the hidden size of the P10b run. A
    P10a run must hold every test item of the P10b run (the people only when
    both runs have them) in the same split, and its Wikidata snapshots must
    match those of the P10b run, family and kind by kind, since a uid join is
    valid only when both runs drew the same items.

    Args:
        p10b: The P10b run.
        p9b: The P9b run, or None without a knowledge link.
        p10a: The P10a run, or None without a behavior link.

    Raises:
        ValueError: If the layers, the hidden size, the items, the splits or
            the snapshots differ.
    """
    if p9b is not None:
        if p9b.layers != p10b.layers:
            raise ValueError(
                f"the P9b run reads layers {p9b.layers}, the P10b run "
                f"{p10b.layers}"
            )
        if p9b.directions.shape[2] != p10b.directions.shape[1]:
            raise ValueError(
                f"the P9b run has hidden size {p9b.directions.shape[2]}, the "
                f"P10b run {p10b.directions.shape[1]}"
            )
    if p10a is not None:
        _check_behavior_run(p10b, p10a)


def _check_behavior_run(p10b: P10bRun, p10a: P10aRun) -> None:
    for record in p10b.items:
        if record["split"] != "test":
            continue
        uid = record["uid"]
        other = p10a.items.get(uid)
        if other is None:
            if record["family"] == PEOPLE_FAMILY:
                continue
            raise ValueError(
                f"the P10a run lacks the test item {uid!r} of the P10b run"
            )
        if other["split"] != record["split"]:
            raise ValueError(
                f"item {uid!r} is in the {other['split']} split of the P10a "
                f"run and in the {record['split']} split of the P10b run"
            )
    for family, kinds in p10b.snapshots.items():
        for kind, stamp in kinds.items():
            other_stamp = p10a.snapshots.get(family, {}).get(kind)
            if other_stamp is None:
                if family == PEOPLE_FAMILY:
                    continue
                raise ValueError(
                    f"the P10a run has no snapshot of {family}/{kind}"
                )
            if sorted(stamp["sha256"]) != sorted(other_stamp["sha256"]):
                raise ValueError(
                    f"the sha256 of the {family}/{kind} snapshot differs "
                    "between the P10a run and the P10b run"
                )
