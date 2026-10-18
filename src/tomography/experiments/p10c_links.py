"""P10c: links of the range direction to knowledge and to hedging.

P10b reads a direction that separates questions whose true answer is a range
(aleatoric) from questions with one exact answer. Two links are measured on
finished runs, so no earlier config hash changes.

Knowledge: P9b reads one direction per contrast of labeled questions. A
"knowledge" contrast has the known questions on its positive side and
unknown ones on its negative side. An "unknown_type" contrast sets one kind
of unknown against another, with the real or pre-cutoff questions on its
positive side. How is the range direction related to each of them in one
model? The cosine between the stored directions answers it layer by layer,
and so does the AUROC of each stored P9 reader on the P10 items. Every row
carries the kind of contrast of its P9 type.

Behavior: within one kind of question, does the P10b score of a test item
predict how often the model hedges (answers with a range) on the P10a run?
Across kinds the link is trivial, because the reader separates the kinds and
the behavior differs by kind.

The inputs `readers.npz` and `items.json` are local only (export drops them),
so this experiment runs where the full run directories exist.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import csv
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any
from typing import Final
from typing import cast

import numpy as np
from numpy.typing import NDArray

from tomography.activations import collect_activations
from tomography.datasets.quantities import FAMILIES
from tomography.datasets.quantities import PEOPLE_FAMILY
from tomography.datasets.quantities import QuantityQuestion
from tomography.datasets.quantities import load_people_heights
from tomography.datasets.quantities import load_quantities
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.p3_size import spearman
from tomography.experiments.p10a_quantities import CONDITIONS
from tomography.experiments.p10a_quantities import FORMATS
from tomography.experiments.p10b_reading import POOLED
from tomography.experiments.p10b_reading import TEMPLATES
from tomography.experiments.sweep import chat_format_for
from tomography.knowledge import DEFAULT_SHOTS
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.p9.reading import READING_METHODS
from tomography.p9.reading import p9_template
from tomography.p9.risk import auroc_ci
from tomography.reading.metrics import auroc
from tomography.reading.reader import RepReader
from tomography.templates import ChatFormat
from tomography.templates import ConceptTemplate
from tomography.viz.reading import curves_from_arrays
from tomography.viz.reading import plot_curves
from tomography.viz.style import BASELINE
from tomography.viz.style import COLUMN_WIDTH
from tomography.viz.style import INK_MUTED
from tomography.viz.style import INK_SECONDARY
from tomography.viz.style import PX
from tomography.viz.style import categorical_colors
from tomography.viz.style import new_figure
from tomography.viz.style import save_figure
from tomography.viz.style import style_context

READER_PARTS: Final = 2
COSINE_COLUMNS: Final = (
    "p9_type",
    "contrast",
    "layer",
    "cosine",
    "band_lo",
    "band_hi",
)
KNOWLEDGE_COLUMNS: Final = (
    "p9_type",
    "contrast",
    "layer",
    "items",
    "test_family",
    "n_range",
    "n_exact",
    "auroc",
    "lo",
    "hi",
    "rand_lo",
    "rand_hi",
)
BEHAVIOR_COLUMNS: Final = (
    "condition",
    "format",
    "decoding",
    "kind",
    "test_family",
    "n",
    "mean_hedge",
    "spearman",
    "lo",
    "hi",
    "constant",
)
DECODINGS: Final = ("greedy", "sampled")
KINDS: Final = ("range", "exact")
MIN_LINKED_ITEMS: Final = 3
MAX_SERIES: Final = 8
CONSTANT_TOLERANCE: Final = 1e-12
BAND_PERCENTILES: Final = (2.5, 97.5)
ALL_ITEMS: Final = "all"
TEST_ITEMS: Final = "test"
KNOWLEDGE_CONCEPT: Final = "knowledge"
CONTRASTS: Final = ("knowledge", "unknown_type")

Row = dict[str, object]
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
        types: The contrast each reader was fitted on, in the order of the
            arrays.
        contrasts: The kind of each contrast in `types`, one of `CONTRASTS`.
            A "knowledge" contrast has the known questions on its positive
            side. An "unknown_type" contrast has the real unknowns
            (real_vs_synthal, real_vs_generated) or the pre-cutoff unknowns
            (pre_vs_post_cutoff) on its positive side, see
            `tomography.p9.unknowns.TypePool`.
        layers: The layer index of each layer slice.
        directions: Unit directions of shape (N, K, d), one reader per type.
        means: The vectors subtracted before projecting, shape (N, K, d).
        signs: The orientation per type and layer, shape (N, K): a reader
            scores the positive side of its contrast higher when the
            direction is multiplied by the sign.
    """

    path: str
    config_hash: str
    config: Record
    types: tuple[str, ...]
    contrasts: tuple[str, ...]
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
        ValueError: If the run cannot be linked (see `_open_finished`), its
            summary gives no kind of contrast to a type or a kind that is not
            in `CONTRASTS`, or its arrays disagree with its types and layers.
    """
    run_dir = _open_finished(
        path,
        model=model,
        files=["readers.npz", "readers.json", "summary.json"],
    )
    types = tuple(cast(Record, run_dir.read_json("readers"))["types"])
    stored = cast(Record, run_dir.read_json("summary")).get("contrasts")
    if stored is None:
        raise ValueError(f"{path}: the summary has no contrasts")
    for name in types:
        if name not in stored:
            raise ValueError(
                f"{path}: the summary has no contrast for {name!r}"
            )
        if stored[name] not in CONTRASTS:
            raise ValueError(
                f"{path}: the contrast {stored[name]!r} of {name!r} is not "
                f"one of {CONTRASTS}"
            )
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
        contrasts=tuple(stored[name] for name in types),
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


def oriented(
    directions: NDArray[np.float32], signs: NDArray[np.float32]
) -> NDArray[np.float64]:
    """Turns the stored directions into the ones a score is a projection on.

    A reader scores `sign * (h - mean) . direction`, so the direction whose
    positive side is the positive class is `sign * direction`.

    Args:
        directions: Stored directions of shape (..., K, d).
        signs: Stored orientation of shape (..., K).

    Returns:
        The oriented directions as float64, shape (..., K, d).
    """
    return directions.astype(np.float64) * signs.astype(np.float64)[..., None]


def _unit_vector(vector: NDArray[np.float64], what: str) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{what} is a zero vector")
    return vector / norm


def _cosine_band(
    direction: NDArray[np.float64], *, n_random: int, seed: int
) -> tuple[float, float]:
    """Brackets the cosine of a direction with random unit vectors."""
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((n_random, direction.shape[0]))
    draws /= np.linalg.norm(draws, axis=1, keepdims=True)
    low, high = np.percentile(draws @ direction, BAND_PERCENTILES)
    return float(low), float(high)


def cosine_rows(
    p10b: P10bRun, p9b: P9bRun, *, n_random: int, seed: int
) -> list[Row]:
    """Measures the cosine of the range direction to each P9 direction.

    At each layer the cosine is taken between the oriented P10b direction
    (positive means range) and the oriented direction of a P9 reader
    (positive means the positive side of its contrast, see
    `P9bRun.contrasts`). The band holds the 2.5th and 97.5th percentiles of
    the cosine between the P10b direction and `n_random` random unit
    vectors, seeded per layer. A negative cosine outside the band means the
    range direction points toward the negative side of that contrast.

    The stored P10b direction is the one of draw 0 fitted on the full train
    set, not an average over the draws.

    Args:
        p10b: The P10b run, read with the reader to link.
        p9b: The P9b run.
        n_random: Random unit vectors per band.
        seed: Base seed of the random vectors.

    Returns:
        One row of `COSINE_COLUMNS` per P9 type and layer, types first.

    Raises:
        ValueError: If a direction is a zero vector.
    """
    range_side = oriented(p10b.directions, p10b.signs)
    p9_side = oriented(p9b.directions, p9b.signs)
    bands: list[tuple[float, float]] = []
    units: list[NDArray[np.float64]] = []
    for k, layer in enumerate(p10b.layers):
        unit = _unit_vector(
            range_side[k], f"the P10b direction at layer {layer}"
        )
        units.append(unit)
        bands.append(
            _cosine_band(
                unit, n_random=n_random, seed=derive_seed(seed, "cosine", layer)
            )
        )
    rows: list[Row] = []
    for t, name in enumerate(p9b.types):
        for k, layer in enumerate(p10b.layers):
            other = _unit_vector(
                p9_side[t, k], f"the P9 direction of {name} at layer {layer}"
            )
            rows.append(
                {
                    "p9_type": name,
                    "contrast": p9b.contrasts[t],
                    "layer": layer,
                    "cosine": float(units[k] @ other),
                    "band_lo": bands[k][0],
                    "band_hi": bands[k][1],
                }
            )
    return rows


def render_questions(
    items: Sequence[QuantityQuestion], p9b: P9bRun, chat: ChatFormat
) -> list[str]:
    """Renders each question as the P9b run rendered its questions.

    The template and the number of demonstrations come from the config of
    the P9b run, and the concept word of a "concept" template is the one P9b
    used, "knowledge".

    Args:
        items: The questions.
        p9b: The P9b run.
        chat: The chat format that wraps a "concept" prompt.

    Returns:
        One prompt per question, in order.

    Raises:
        ValueError: If the template of the P9b run is unknown or not a
            concept template.
    """
    template = p9_template(
        str(p9b.config["template"]), DEFAULT_SHOTS[: int(p9b.config["n_shots"])]
    )
    if not isinstance(template, ConceptTemplate):
        raise ValueError("P9 templates are concept templates")
    return [
        template.render(i.question, KNOWLEDGE_CONCEPT, chat).text for i in items
    ]


def collect_question_activations(
    loaded: LoadedModel,
    items: Sequence[QuantityQuestion],
    p9b: P9bRun,
    *,
    layers: Sequence[int],
    batch_size: int,
) -> NDArray[np.float32]:
    """Collects the activations at the last token of each rendered question.

    The prompts and the BOS policy are those of the P9b run, so a P9 reader
    sees the state it was fitted on.

    Args:
        loaded: The wrapped model.
        items: The questions.
        p9b: The P9b run.
        layers: The layers to read, as in the P9b run.
        batch_size: Prompts per forward pass.

    Returns:
        The activations, shape (K, n, d), one row per question in order.
    """
    texts = render_questions(items, p9b, chat_format_for(loaded))
    found = collect_activations(
        loaded,
        texts,
        layers=layers,
        position="last",
        bos=cast(BosMode | None, p9b.config.get("bos")),
        batch_size=batch_size,
    )
    return found.data


def _item_groups(
    items: Sequence[QuantityQuestion], labels: NDArray[np.int64]
) -> list[tuple[str, str, NDArray[np.int64]]]:
    """Lists the item set, family and rows of each group with both kinds."""
    groups: list[tuple[str, str, NDArray[np.int64]]] = []
    for subset in (ALL_ITEMS, TEST_ITEMS):
        for family in (*dict.fromkeys(i.family for i in items), POOLED):
            rows = np.array(
                [
                    n
                    for n, item in enumerate(items)
                    if (subset == ALL_ITEMS or item.split == "test")
                    and family in (POOLED, item.family)
                ],
                dtype=np.int64,
            )
            positives = int(labels[rows].sum())
            if 0 < positives < len(rows):
                groups.append((subset, family, rows))
    return groups


def knowledge_rows(
    data: NDArray[np.float32],
    items: Sequence[QuantityQuestion],
    p9b: P9bRun,
    *,
    n_boot: int,
    n_random: int,
    seed: int,
) -> list[Row]:
    """Scores the P10 items with every P9 reader and ranks range over exact.

    The AUROC has the range items as the positive class and the P9 score as
    stored, so a value below 0.5 means the range items score toward the
    negative side of the contrast.
    No P10 item trained a P9 reader, so the "all" rows use every split; the
    "test" rows use the test split only. The interval resamples items. The
    random band holds the 2.5th and 97.5th percentiles of the AUROC of
    `n_random` random unit directions at the layer, which carry a random
    sign, so the band is the spread of a reader that was not oriented on
    these labels. With a strong direction in the activations that spread is
    wide, so compare an AUROC with the band and never with 0.5.

    Args:
        data: Activations of the items, shape (K, n, d), at the layers of the
            P9b run.
        items: The questions, in the order of `data`.
        p9b: The P9b run.
        n_boot: Item bootstrap resamples of every interval.
        n_random: Random unit directions of every band.
        seed: Base seed of the bootstrap and of the random directions.

    Returns:
        One row of `KNOWLEDGE_COLUMNS` per P9 type, layer, item set and test
        family (each family, then "pooled"). A group without both kinds has
        no row.
    """
    labels = np.array([i.kind == "range" for i in items], dtype=np.int64)
    groups = _item_groups(items, labels)
    bands: dict[tuple[int, str, str], tuple[float, float]] = {}
    for k, layer in enumerate(p9b.layers):
        rng = np.random.default_rng(derive_seed(seed, "auroc", layer))
        draws = rng.standard_normal((n_random, data.shape[2]))
        draws /= np.linalg.norm(draws, axis=1, keepdims=True)
        for subset, family, rows in groups:
            projected = data[k][rows].astype(np.float64) @ draws.T
            low, high = np.percentile(
                auroc(projected.T, labels[rows]), BAND_PERCENTILES
            )
            bands[(layer, subset, family)] = (float(low), float(high))
    result: list[Row] = []
    for t, name in enumerate(p9b.types):
        reader = RepReader.from_arrays(
            name, p9b.layers, p9b.directions[t], p9b.means[t], p9b.signs[t]
        )
        scores = reader.score(data).astype(np.float64)
        for k, layer in enumerate(p9b.layers):
            for subset, family, rows in groups:
                value, low, high = auroc_ci(
                    scores[k][rows], labels[rows], n_boot=n_boot, seed=seed
                )
                n_range = int(labels[rows].sum())
                result.append(
                    {
                        "p9_type": name,
                        "contrast": p9b.contrasts[t],
                        "layer": layer,
                        "items": subset,
                        "test_family": family,
                        "n_range": n_range,
                        "n_exact": len(rows) - n_range,
                        "auroc": value,
                        "lo": low,
                        "hi": high,
                        "rand_lo": bands[(layer, subset, family)][0],
                        "rand_hi": bands[(layer, subset, family)][1],
                    }
                )
    return result


def hedge_share(
    answers: Mapping[str, Record], condition: str, fmt: str, decoding: str
) -> float:
    """Returns the share of range answers of one item in one condition.

    It is the definition of `_item_rates` in `p10a_quantities`: the share of
    the answers with the form "range" (the greedy answer for "greedy", the
    sampled ones for "sampled") in each cell of the condition and format,
    averaged over the cells, which are the shot orders of the few-shot
    format. The average is taken in exact fractions, so two items whose
    shares are equal compare equal whatever their cells hold, and the rank
    correlation ties them.

    Args:
        answers: The parsed answers of the item by "condition/format/order".
        condition: "free" or "forced".
        fmt: "fewshot" or "chat".
        decoding: One of `DECODINGS`.

    Returns:
        The mean over the cells of the share of range answers.

    Raises:
        ValueError: If the item has no cell of the condition and format or a
            cell has no answer of the decoding.
    """
    prefix = f"{condition}/{fmt}/"
    cells = [cell for key, cell in answers.items() if key.startswith(prefix)]
    if not cells:
        raise ValueError(f"no answers for {condition}/{fmt}")
    shares = []
    for cell in cells:
        given = [cell["greedy"]] if decoding == "greedy" else cell["samples"]
        if not given:
            raise ValueError(f"no {decoding} answer in {condition}/{fmt}")
        ranged = sum(answer["form"] == "range" for answer in given)
        shares.append(Fraction(ranged, len(given)))
    return float(sum(shares) / len(shares))


def _is_constant(values: NDArray[np.float64]) -> bool:
    """Tells whether the values differ only by rounding."""
    return bool(
        np.ptp(values) <= CONSTANT_TOLERANCE * max(1.0, np.abs(values).max())
    )


def _link(
    scores: NDArray[np.float64],
    shares: NDArray[np.float64],
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, bool]:
    """Returns the Spearman correlation, its interval and the constant flag."""
    if _is_constant(scores) or _is_constant(shares):
        return float("nan"), float("nan"), float("nan"), True
    draws = np.random.default_rng(seed).integers(
        0, len(scores), size=(n_boot, len(scores))
    )
    values = np.array([spearman(scores[d], shares[d]) for d in draws])
    values = values[np.isfinite(values)]
    low, high = (
        np.percentile(values, BAND_PERCENTILES)
        if values.size
        else (np.nan, np.nan)
    )
    return spearman(scores, shares), float(low), float(high), False


def behavior_rows(
    p10b: P10bRun, p10a: P10aRun, *, n_boot: int, seed: int
) -> list[Row]:
    """Correlates the P10b score with the hedge share within a kind.

    Only the test items of the P10b run that the P10a run also holds enter,
    since the train items are scored in-sample. Each row is one condition,
    format, decoding, kind and test family (each family, "pooled" over the
    families of the kind, and the people as their own row of the kind
    "exact"). Its Spearman correlation is between the P10b score of the
    reader and the hedge share of `hedge_share`, within the group, never
    across the kinds. The interval is an item bootstrap percentile
    interval. A group whose scores or hedge shares are all equal has no
    rank correlation, so it gets NaN and `constant` true, and a group of
    fewer than `MIN_LINKED_ITEMS` items has no row.

    Args:
        p10b: The P10b run, read with the reader to link.
        p10a: The P10a run.
        n_boot: Item bootstrap resamples of every interval.
        seed: Base seed of the bootstrap.

    Returns:
        One row of `BEHAVIOR_COLUMNS` per cell and group, in the order of
        the P10a conditions and formats.

    Raises:
        ValueError: If a linked item lacks a cell the others have.
    """
    members = [
        (record, p10a.items[record["uid"]])
        for record in p10b.items
        if record["split"] == "test" and record["uid"] in p10a.items
    ]
    present = {
        tuple(key.split("/")[:2])
        for _, other in members
        for key in other["answers"]
    }
    families = list(
        dict.fromkeys(
            r["family"] for r, _ in members if r["family"] != PEOPLE_FAMILY
        )
    )
    groups: list[tuple[str, str, list[int]]] = []
    for kind in KINDS:
        for family in (*families, POOLED):
            groups.append(
                (
                    kind,
                    family,
                    [
                        n
                        for n, (r, _) in enumerate(members)
                        if r["kind"] == kind
                        and r["family"] != PEOPLE_FAMILY
                        and family in (POOLED, r["family"])
                    ],
                )
            )
    groups.append(
        (
            "exact",
            PEOPLE_FAMILY,
            [
                n
                for n, (r, _) in enumerate(members)
                if r["family"] == PEOPLE_FAMILY
            ],
        )
    )
    scores = np.array(
        [r["scores"][p10b.reader] for r, _ in members], dtype=np.float64
    )
    rows: list[Row] = []
    for condition in CONDITIONS:
        for fmt in FORMATS:
            if (condition, fmt) not in present:
                continue
            for decoding in DECODINGS:
                shares = np.array(
                    [
                        hedge_share(other["answers"], condition, fmt, decoding)
                        for _, other in members
                    ],
                    dtype=np.float64,
                )
                for kind, family, picked in groups:
                    if len(picked) < MIN_LINKED_ITEMS:
                        continue
                    rho, low, high, constant = _link(
                        scores[picked],
                        shares[picked],
                        n_boot=n_boot,
                        seed=derive_seed(
                            seed, condition, fmt, decoding, kind, family
                        ),
                    )
                    rows.append(
                        {
                            "condition": condition,
                            "format": fmt,
                            "decoding": decoding,
                            "kind": kind,
                            "test_family": family,
                            "n": len(picked),
                            "mean_hedge": float(shares[picked].mean()),
                            "spearman": rho,
                            "lo": low,
                            "hi": high,
                            "constant": constant,
                        }
                    )
    return rows


def load_inputs(
    config: Config,
) -> tuple[P10bRun, P9bRun | None, P10aRun | None]:
    """Reads and cross-checks the runs a config links.

    Args:
        config: The experiment settings.

    Returns:
        The P10b run, and the P9b and P10a runs when the config names them.

    Raises:
        ValueError: If a run cannot be linked or the runs disagree, see
            `load_p10b`, `load_p9b`, `load_p10a` and `check_links`.
    """
    p10b = load_p10b(config.p10b_run, model=config.model, reader=config.reader)
    p9b = (
        None
        if config.knowledge_run is None
        else load_p9b(config.knowledge_run, model=config.model)
    )
    p10a = (
        None
        if config.behavior_run is None
        else load_p10a(config.behavior_run, model=config.model)
    )
    check_links(p10b, p9b, p10a)
    return p10b, p9b, p10a


def load_items(
    p10b: P10bRun, *, offline: bool
) -> tuple[list[QuantityQuestion], list[QuantityQuestion]]:
    """Rebuilds the questions of a P10b run from its config.

    Args:
        p10b: The P10b run.
        offline: Whether the questions are read from the cache only.

    Returns:
        The questions of the families, then the people (empty when the P10b
        run did not score them).
    """
    settings = p10b.config
    wanted = settings["families"]
    families = [f for f in FAMILIES if wanted is None or f.name in wanted]
    items = load_quantities(
        families=families,
        n_per_side=settings["n_per_side"],
        min_sitelinks=settings["min_sitelinks"],
        seed=settings["seed"],
        offline=offline,
    )
    people = load_people_heights(offline=offline) if settings["people"] else []
    return items, people


def check_items(
    items: Sequence[QuantityQuestion],
    people: Sequence[QuantityQuestion],
    p10b: P10bRun,
) -> None:
    """Checks that the rebuilt questions are those the P10b run scored.

    Args:
        items: The rebuilt questions of the families.
        people: The rebuilt people.
        p10b: The P10b run.

    Raises:
        ValueError: If the uids differ from those of `items.json` of the
            P10b run, or an item is in another split.
    """
    rebuilt = {i.uid: i for i in [*items, *people]}
    stored = {r["uid"]: r for r in p10b.items}
    missing = sorted(set(stored) - set(rebuilt))
    extra = sorted(set(rebuilt) - set(stored))
    if missing or extra:
        raise ValueError(
            "the rebuilt questions differ from those of the P10b run: "
            f"{len(missing)} missing (first {missing[:1]}), {len(extra)} "
            f"extra (first {extra[:1]})"
        )
    for uid, item in rebuilt.items():
        if item.split != stored[uid]["split"]:
            raise ValueError(
                f"item {uid!r} is in the {item.split} split of the rebuilt "
                f"questions and in the {stored[uid]['split']} split of the "
                "P10b run"
            )


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Row]) -> None:
    """Writes a CSV with unix line ends; None becomes an empty cell."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _counts(items: Sequence[QuantityQuestion]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for item in items:
        by_kind = counts.setdefault(item.family, {})
        by_kind[item.kind] = by_kind.get(item.kind, 0) + 1
    return counts


def _behavior_counts(rows: Sequence[Row]) -> dict[str, dict[str, int]]:
    """Counts the linked items of one cell by test family and kind."""
    counts: dict[str, dict[str, int]] = {}
    cell = ((r["condition"], r["format"], r["decoding"]) for r in rows)
    first = next(cell, None)
    for row in rows:
        if (row["condition"], row["format"], row["decoding"]) == first:
            by_kind = counts.setdefault(str(row["test_family"]), {})
            by_kind[str(row["kind"])] = cast(int, row["n"])
    return counts


def _at_layer(rows: Sequence[Row], layer: int, **where: object) -> list[Row]:
    return [
        r
        for r in rows
        if r["layer"] == layer and all(r[k] == v for k, v in where.items())
    ]


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel | None,
    items: Sequence[QuantityQuestion] | None,
    people: Sequence[QuantityQuestion] | None,
    p10b: P10bRun,
    p9b: P9bRun | None,
    p10a: P10aRun | None,
) -> None:
    """Links the range reader to the P9 readers and to the hedging.

    Writes `cosine.csv` and `knowledge.csv` when there is a knowledge link
    (see `cosine_rows` and `knowledge_rows`; the questions go through the
    model once), `behavior.csv` when there is a behavior link (see
    `behavior_rows`) and `summary.json`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model; None without a knowledge link.
        items: The questions of the families, as rebuilt from the P10b run;
            None without a knowledge link.
        people: The rebuilt people, which no P9 reader scores since they
            have no range side; None without a knowledge link.
        p10b: The P10b run.
        p9b: The P9b run; None without a knowledge link.
        p10a: The P10a run; None without a behavior link.

    Raises:
        ValueError: If there is a knowledge link but no model or no items.
    """
    path = run_dir.path
    cosine: list[Row] = []
    knowledge: list[Row] = []
    behavior: list[Row] = []
    if p9b is not None:
        if loaded is None or items is None:
            raise ValueError("the knowledge link needs the model and the items")
        cosine = cosine_rows(
            p10b, p9b, n_random=config.n_random, seed=config.seed
        )
        _write_csv(path / "cosine.csv", COSINE_COLUMNS, cosine)
        data = collect_question_activations(
            loaded,
            items,
            p9b,
            layers=p10b.layers,
            batch_size=config.batch_size,
        )
        knowledge = knowledge_rows(
            data,
            items,
            p9b,
            n_boot=config.n_boot,
            n_random=config.n_random,
            seed=config.seed,
        )
        _write_csv(path / "knowledge.csv", KNOWLEDGE_COLUMNS, knowledge)
    if p10a is not None:
        behavior = behavior_rows(
            p10b, p10a, n_boot=config.n_boot, seed=config.seed
        )
        _write_csv(path / "behavior.csv", BEHAVIOR_COLUMNS, behavior)
    types = [] if p9b is None else list(p9b.types)
    run_dir.write_json(
        "summary",
        {
            "model": config.model,
            "runs": {
                "p10b": _stamp(p10b),
                "p9b": None if p9b is None else _stamp(p9b),
                "p10a": None if p10a is None else _stamp(p10a),
            },
            "reader": config.reader,
            "layer": p10b.layer,
            "p9_types": types,
            "cosine": {
                name: {
                    key: _at_layer(cosine, p10b.layer, p9_type=name)[0][key]
                    for key in ("contrast", "cosine", "band_lo", "band_hi")
                }
                for name in types
            },
            "knowledge": {
                name: {
                    key: row[key]
                    for key in (
                        "contrast",
                        "n_range",
                        "n_exact",
                        "auroc",
                        "lo",
                        "hi",
                        "rand_lo",
                        "rand_hi",
                    )
                }
                for name in types
                for row in _at_layer(
                    knowledge,
                    p10b.layer,
                    p9_type=name,
                    items=ALL_ITEMS,
                    test_family=POOLED,
                )
            },
            "behavior": [r for r in behavior if r["test_family"] == POOLED],
            "counts": {
                "knowledge": _counts(items or []),
                "people": len(people or []),
                "behavior": _behavior_counts(behavior),
            },
            "config": config.to_dict(),
        },
    )


def _stamp(run: P10bRun | P9bRun | P10aRun) -> dict[str, str]:
    return {"path": run.path, "config_hash": run.config_hash}


def run(config: Config, run_dir: RunDir) -> None:
    """Reads the finished runs, rebuilds the questions and runs `execute`.

    Every input is checked before the model is loaded or a question is
    fetched, and a behavior link alone loads neither.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.

    Raises:
        ValueError: If a run cannot be linked, the runs disagree or the
            rebuilt questions are not those of the P10b run.
    """
    p10b, p9b, p10a = load_inputs(config)
    loaded = items = people = None
    if p9b is not None:
        items, people = load_items(p10b, offline=config.offline)
        check_items(items, people, p10b)
        loaded = load_model(config.model)
    execute(config, run_dir, loaded, items, people, p10b, p9b, p10a)


def _plot_by_type(
    rows: Sequence[dict[str, str]],
    out_dir: Path,
    *,
    stem: str,
    column: str,
    ylabel: str,
    band: tuple[str, str],
    band_label: str,
    ylim: tuple[float, float] | None,
) -> list[Path]:
    """Draws one value per layer for each P9 type over a shared band.

    There is one figure per eight types, since a figure has eight hues.
    """
    types = list(dict.fromkeys(r["p9_type"] for r in rows))
    paths: list[Path] = []
    for number, start in enumerate(range(0, len(types), MAX_SERIES), start=1):
        group = types[start : start + MAX_SERIES]
        by_type = [[r for r in rows if r["p9_type"] == t] for t in group]
        layers = np.array([float(r["layer"]) for r in by_type[0]])
        values = np.array([[float(r[column]) for r in t] for t in by_type])
        low = np.array([float(r[band[0]]) for r in by_type[0]])
        high = np.array([float(r[band[1]]) for r in by_type[0]])
        with style_context():
            fig, axes = new_figure()
            ax = axes[0, 0]
            ax.fill_between(
                layers,
                low,
                high,
                color=BASELINE,
                alpha=0.5,
                lw=0,
                label=band_label,
            )
            plot_curves(
                curves_from_arrays(layers, values, values, values),
                series=group,
                ylabel=ylabel,
                n_layers=int(layers.max()),
                ylim=ylim,
                legend=True,
                ax=ax,
            )
        name = stem if number == 1 else f"{stem}_{number}"
        paths += save_figure(fig, out_dir, name)
    return paths


def _plot_behavior(rows: Sequence[dict[str, str]], out_dir: Path) -> list[Path]:
    """Draws the pooled Spearman correlation of each kind with its interval.

    The bars are the sampled decoding, one group per condition and format.
    A group whose correlation is undefined has no bar.
    """
    pooled = [
        r
        for r in rows
        if r["test_family"] == POOLED and r["decoding"] == "sampled"
    ]
    cells = list(dict.fromkeys((r["condition"], r["format"]) for r in pooled))
    if not cells:
        return []
    colors = categorical_colors(len(KINDS))
    width = 0.8 / len(KINDS)
    with style_context():
        fig, axes = new_figure(width=COLUMN_WIDTH * max(1.0, len(cells) / 2))
        ax = axes[0, 0]
        for j, (kind, color) in enumerate(zip(KINDS, colors, strict=True)):
            found = [
                (i, r)
                for i, cell in enumerate(cells)
                for r in pooled
                if (r["condition"], r["format"], r["kind"]) == (*cell, kind)
                and r["constant"] != "True"
            ]
            x = np.array([i + (j - 0.5) * width for i, _ in found])
            value = np.array([float(r["spearman"]) for _, r in found])
            low = np.array([float(r["lo"]) for _, r in found])
            high = np.array([float(r["hi"]) for _, r in found])
            ax.bar(x, value, width=width, color=color, label=kind)
            ax.errorbar(
                x,
                value,
                yerr=[value - low, high - value],
                fmt="none",
                ecolor=INK_SECONDARY,
                elinewidth=PX,
                capsize=2,
            )
        ax.axhline(0.0, color=INK_MUTED, lw=PX)
        ax.set_xticks(np.arange(len(cells)), [f"{c}\n{f}" for c, f in cells])
        ax.set_ylabel("Spearman, P10b score vs hedge share")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=len(KINDS))
    return save_figure(fig, out_dir, "behavior_spearman")


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the figures whose CSV the run wrote.

    `cosine` shows the cosine to the range direction against the layer for
    each P9 type, over the band of random vectors. `knowledge_auroc` shows
    the AUROC of each P9 reader on all P10 items, pooled over the families,
    over the random-direction band; compare a curve with the band, never
    with 0.5. `behavior_spearman` shows, for the sampled decoding, the
    pooled Spearman correlation of the range score with the hedge share
    within each kind, with its bootstrap interval, for each condition and
    format. More than eight P9 types are split over `<stem>_2` and so on.

    Args:
        run_dir: A finished P10c run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of every figure.
    """
    path = run_dir.path
    paths: list[Path] = []
    if (path / "cosine.csv").is_file():
        paths += _plot_by_type(
            _read_csv(path / "cosine.csv"),
            out_dir,
            stem="cosine",
            column="cosine",
            ylabel="Cosine to the range direction",
            band=("band_lo", "band_hi"),
            band_label="random vectors, 95%",
            ylim=(-1.0, 1.0),
        )
    if (path / "knowledge.csv").is_file():
        pooled = [
            r
            for r in _read_csv(path / "knowledge.csv")
            if r["items"] == ALL_ITEMS and r["test_family"] == POOLED
        ]
        paths += _plot_by_type(
            pooled,
            out_dir,
            stem="knowledge_auroc",
            column="auroc",
            ylabel="AUROC (range vs exact)",
            band=("rand_lo", "rand_hi"),
            band_label="random directions, 95%",
            ylim=(0.0, 1.0),
        )
    if (path / "behavior.csv").is_file():
        paths += _plot_behavior(_read_csv(path / "behavior.csv"), out_dir)
    return paths
