"""P10b: reading a range against an exact answer from hidden states.

Each question asks for a quantity whose sourced answer is a range (the height
of a dog breed) or one exact value (the height of a building). A reader is
fitted on the activations of the questions of some families and scored on the
held-out questions of every family, which asks whether a direction that
separates the two kinds transfers across families. In every family the range
side is living and the exact side an artifact, so a set of well-known people
with one exact height is scored as a diagnostic: a reader of range against
exact calls them exact, a reader of living against artifact calls them range.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final
from typing import cast

import numpy as np
from numpy.typing import NDArray

from tomography.datasets.quantities import DEFAULT_MIN_SITELINKS
from tomography.datasets.quantities import FAMILIES
from tomography.datasets.quantities import PEOPLE_FAMILY
from tomography.datasets.quantities import QuantityQuestion
from tomography.experiments.base import ExperimentConfig
from tomography.p9.reading import READING_METHODS
from tomography.p9.reading import ItemActivations
from tomography.p9.reading import draw_seed
from tomography.p9.risk import TEXT_ANALYZERS
from tomography.p9.risk import auroc_ci
from tomography.p9.risk import orient_on_train
from tomography.p9.risk import text_scores
from tomography.pairing import label_contrastive_pairs
from tomography.reading.directions import Method
from tomography.reading.metrics import auroc
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import bootstrap_ci
from tomography.reading.reader import RepReader

TEMPLATES: Final = ("plain", "concept")
ALL_SET: Final = "all"
BASELINES: Final = ("text_word", "text_char", "length", "popularity")
POOLED: Final = "pooled"

Row = dict[str, object]


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """P10b settings: the questions, the prompts and the readers.

    Attributes:
        families: Names of the question families to read; None reads all.
        n_per_side: Largest number of items per family and kind.
        min_sitelinks: Smallest sitelink count of an entity.
        people: Whether to score the people heights as a diagnostic; it
            needs the "height" family.
        offline: Whether the questions are read from the cache only.
        templates: How a question is rendered: "plain" is the bare question,
            "concept" the paper's concept template wrapped in the chat
            format of post-trained models.
        concept: The concept word of the "concept" template.
        methods: The direction finders; "random" is the control.
        n_seeds: Draws of the pairs per template, method and train set.
        n_boot: Resamples of every bootstrap interval.
        batch_size: Prompts per forward pass.
    """

    experiment: str = "p10b"
    model: str = "gpt2"
    families: tuple[str, ...] | None = None
    n_per_side: int = 120
    min_sitelinks: int = DEFAULT_MIN_SITELINKS
    people: bool = True
    offline: bool = False
    templates: tuple[str, ...] = TEMPLATES
    concept: str = "variability"
    methods: tuple[str, ...] = READING_METHODS
    n_seeds: int = 5
    n_boot: int = 1000
    batch_size: int = 8

    def __post_init__(self) -> None:
        """Checks the settings that would act silently or fail late.

        Raises:
            ValueError: If `families`, `templates` or `methods` is empty,
                repeats a name or holds an unknown one, `concept` is empty
                or a count is below its minimum (`n_per_side`, `n_seeds`,
                `n_boot` and `batch_size` need 1, `min_sitelinks` needs 0).
        """
        if self.families is not None:
            _check_names("families", self.families, [f.name for f in FAMILIES])
        _check_names("templates", self.templates, TEMPLATES)
        _check_names("methods", self.methods, READING_METHODS)
        if not self.concept:
            raise ValueError("concept must not be empty")
        for name in ("n_per_side", "n_seeds", "n_boot", "batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.min_sitelinks < 0:
            raise ValueError("min_sitelinks must not be negative")


def _check_names(
    field: str, names: tuple[str, ...], allowed: Sequence[str]
) -> None:
    """Checks that a setting names at least one known option, each once.

    Args:
        field: The setting, for the message.
        names: The chosen options.
        allowed: The known options.

    Raises:
        ValueError: If `names` is empty, repeats an option or holds one that
            is not in `allowed`.
    """
    if not names:
        raise ValueError(f"{field} must not be empty")
    if len(set(names)) != len(names):
        raise ValueError(f"{field} repeats a name: {names}")
    unknown = sorted(set(names) - set(allowed))
    if unknown:
        raise ValueError(f"{field} names unknown {unknown}; known: {allowed}")


def train_sets(families: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Derives the sets of families a reader is fitted on.

    The sets are each single family, then, with three or more families,
    each leave-one-family-out set, then every family together as "all".
    With one family the single set is that family alone. A set is named by
    joining its family names with "+", and the families of a set follow the
    order of `FAMILIES`.

    Args:
        families: The selected family names.

    Returns:
        The family names of each set, by set name.
    """
    order = [f.name for f in FAMILIES]
    chosen = tuple(sorted(families, key=order.index))
    sets: dict[str, tuple[str, ...]] = {name: (name,) for name in chosen}
    if len(chosen) >= 3:
        for left_out in chosen:
            rest = tuple(name for name in chosen if name != left_out)
            sets["+".join(rest)] = rest
    if len(chosen) >= 2:
        sets[ALL_SET] = chosen
    return sets


def _labels(items: Sequence[QuantityQuestion]) -> NDArray[np.int64]:
    """Labels the items 1 for a range answer and 0 for an exact one."""
    return np.array([i.kind == "range" for i in items], dtype=np.int64)


def baseline_scores(
    items: Sequence[QuantityQuestion], families: Sequence[str]
) -> dict[str, NDArray[np.float64]]:
    """Scores the items from their surface form, without a model.

    The text classifiers (`tomography.p9.risk.text_scores`) are fitted on the
    questions of the train split of the given families. Length is the number
    of characters of the question and popularity is `log10(sitelinks + 1)`.
    Every score is oriented on the same train items, so a higher value means
    a range answer whichever way the surface feature points.

    Args:
        items: The questions to score, of any family and split.
        families: The families whose train items fit and orient the scores.

    Returns:
        The score of each item, in order, by name in `BASELINES`.

    Raises:
        ValueError: If the train items leave one kind out or no text feature.
    """
    train = [
        i
        for i, item in enumerate(items)
        if item.split == "train" and item.family in families
    ]
    labels = _labels([items[i] for i in train])
    questions = [item.question for item in items]
    raw: dict[str, NDArray[np.float64]] = {}
    for analyzer in TEXT_ANALYZERS:
        raw[f"text_{analyzer}"] = text_scores(
            [questions[i] for i in train],
            labels,
            questions,
            analyzer=analyzer,
        )
    raw["length"] = np.array([len(q) for q in questions], dtype=np.float64)
    raw["popularity"] = np.log10(
        np.array([i.sitelinks for i in items], dtype=np.float64) + 1.0
    )
    return {
        name: orient_on_train(raw[name][train], labels, raw[name])
        for name in BASELINES
    }


@dataclass(frozen=True, eq=False)
class Study:
    """Every result of the reading of the questions.

    Attributes:
        items: The questions in the order of the score arrays, the people
            last.
        families: The families read, in the order of `FAMILIES`.
        sets: The families of each train set, by set name.
        curves: One row per template, method, train set, test family and
            layer: the test AUROC averaged over the seeds, with a bootstrap
            interval over the seeds.
        transfer: One row per template, method, train set and test family at
            the layer chosen on the validation items of the train set.
        baselines: One row per baseline, train set and test family.
        people: The people diagnostic, see `people_row`; empty when it was
            not asked for.
        layers: The chosen layer, by "template/method/train_set".
        scores: The score of every item at the chosen layer, averaged over
            the seeds, by the same key.
        readers: The reader of the first draw fitted on the largest train
            set, by "template/method".
    """

    items: tuple[QuantityQuestion, ...]
    families: tuple[str, ...]
    sets: dict[str, tuple[str, ...]]
    curves: list[Row]
    transfer: list[Row]
    baselines: list[Row]
    people: list[Row]
    layers: dict[str, int]
    scores: dict[str, NDArray[np.float64]]
    readers: dict[str, RepReader]


@dataclass(frozen=True)
class Groups:
    """Row indices of the questions each metric ranks.

    Attributes:
        tests: The test items of each family and of all of them pooled.
        breeds: The test items of the range side of the "height" family.
        buildings: The test items of its exact side.
        people: The people items; empty unless the diagnostic is on.
    """

    tests: dict[str, NDArray[np.int64]]
    breeds: NDArray[np.int64]
    buildings: NDArray[np.int64]
    people: NDArray[np.int64]


def _indices(
    items: Sequence[QuantityQuestion],
    *,
    families: Sequence[str],
    split: str,
    kind: str | None = None,
) -> NDArray[np.int64]:
    return np.array(
        [
            n
            for n, item in enumerate(items)
            if item.family in families
            and item.split == split
            and (kind is None or item.kind == kind)
        ],
        dtype=np.int64,
    )


def _groups(
    items: Sequence[QuantityQuestion],
    families: Sequence[str],
    *,
    people: bool,
) -> Groups:
    tests = {
        name: _indices(items, families=[name], split="test")
        for name in families
    }
    tests[POOLED] = _indices(items, families=families, split="test")
    asked = people and "height" in families

    def pick(family: str, kind: str | None = None) -> NDArray[np.int64]:
        if not asked:
            return np.array([], dtype=np.int64)
        return _indices(items, families=[family], split="test", kind=kind)

    return Groups(
        tests=tests,
        breeds=pick("height", "range"),
        buildings=pick("height", "exact"),
        people=pick(PEOPLE_FAMILY),
    )


def _check_sides(
    items: Sequence[QuantityQuestion], families: Sequence[str]
) -> None:
    """Checks that every family has items of both kinds in every split.

    Raises:
        ValueError: If a family lacks a kind in a split.
    """
    for family in families:
        for split in ("train", "val", "test"):
            for kind in ("range", "exact"):
                found = _indices(
                    items, families=[family], split=split, kind=kind
                )
                if not found.size:
                    raise ValueError(
                        f"family {family!r} has no {kind} item in {split}"
                    )


def _fit_reader(
    data: NDArray[np.float32],
    items: Sequence[QuantityQuestion],
    labels: NDArray[np.int64],
    train: NDArray[np.int64],
    layers: Sequence[int],
    *,
    method: Method,
    seed: int,
) -> RepReader:
    """Fits a reader on the train items, pairing a range and an exact item.

    A pair never crosses two families. The reader orients each layer on the
    labels, so a higher score means a range answer.
    """
    y = labels[train]
    families = [items[n].family for n in train]
    codes = np.unique(families, return_inverse=True)[1].astype(np.int64)
    pairs = label_contrastive_pairs(y, seed=seed, strata=codes)
    used = np.unique(pairs)
    local = np.searchsorted(used, pairs)
    return RepReader.fit(
        method,
        data[:, train[used]],
        layers,
        labels=y[used],
        pairs=local,
        groups=local,
        preprocess="center",
        seed=seed,
    )


def _interval(
    score: NDArray[np.float64],
    positive: NDArray[np.int64],
    negative: NDArray[np.int64],
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rows = np.concatenate([positive, negative])
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    return auroc_ci(
        score[rows], labels.astype(np.int64), n_boot=n_boot, seed=seed
    )


def _by_kind(
    items: Sequence[QuantityQuestion], rows: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    is_range = np.array([items[n].kind == "range" for n in rows], dtype=bool)
    return rows[is_range], rows[~is_range]


def people_row(
    score: NDArray[np.float64],
    groups: Groups,
    *,
    n_boot: int,
    seed: int,
) -> Row:
    """Places the people between the dog breeds and the buildings.

    `breeds_vs_people` is the AUROC of the height range-side test items
    against the people: high when the people score like buildings (a reader
    of range against exact) and near 0.5 when they score like breeds.
    `people_vs_buildings` is the AUROC of the people against the height
    exact-side test items: near 0.5 when the people score like buildings and
    high when they score above them (a reader of living against artifact).
    `position` is the mean people score minus the mean buildings score over
    the mean breeds score minus the mean buildings score: 0 when the people
    read like buildings (exact) and 1 when they read like breeds.

    Args:
        score: The score of every item, higher meaning range.
        groups: The breeds, buildings and people to rank.
        n_boot: Item bootstrap resamples of each interval.
        seed: Seed of the bootstrap.

    Returns:
        The counts, the two AUROCs with their intervals and the position
        (None when the breeds and the buildings have the same mean score).
    """
    bp = _interval(
        score, groups.breeds, groups.people, n_boot=n_boot, seed=seed
    )
    pb = _interval(
        score, groups.people, groups.buildings, n_boot=n_boot, seed=seed
    )
    gap = score[groups.breeds].mean() - score[groups.buildings].mean()
    shift = score[groups.people].mean() - score[groups.buildings].mean()
    return {
        "n_breeds": len(groups.breeds),
        "n_buildings": len(groups.buildings),
        "n_people": len(groups.people),
        "breeds_vs_people": bp[0],
        "breeds_vs_people_lo": bp[1],
        "breeds_vs_people_hi": bp[2],
        "people_vs_buildings": pb[0],
        "people_vs_buildings_lo": pb[1],
        "people_vs_buildings_hi": pb[2],
        "position": None if gap == 0 else float(shift / gap),
    }


def _test_rows(
    score: NDArray[np.float64],
    items: Sequence[QuantityQuestion],
    groups: Groups,
    *,
    n_boot: int,
    seed: int,
) -> list[Row]:
    rows: list[Row] = []
    for name, held in groups.tests.items():
        positive, negative = _by_kind(items, held)
        value, lo, hi = _interval(
            score, positive, negative, n_boot=n_boot, seed=seed
        )
        rows.append(
            {
                "test_family": name,
                "n_range": len(positive),
                "n_exact": len(negative),
                "auroc": value,
                "lo": lo,
                "hi": hi,
            }
        )
    return rows


def read_study(
    config: Config,
    acts: Mapping[str, ItemActivations],
    items: Sequence[QuantityQuestion],
) -> Study:
    """Fits and scores a reader per template, method, train set and draw.

    Each reader is fitted on the train split of the families of its train
    set (see `train_sets`) and scores every item. Draw `s` of `n_seeds`
    pairs the train items with the seed `draw_seed(config.seed, s)`. The
    layer of a template, method and train set is the one with the best mean
    over the draws of the validation AUROC on the validation items of the
    train set's families; no test item chooses anything. The curves hold the
    test AUROC at every layer and the transfer rows the test AUROC of the
    draw-averaged score at the chosen layer, with an item bootstrap interval.
    The surface baselines (see `baseline_scores`) are fitted on the same
    train items. The people diagnostic (see `people_row`) needs
    `config.people`, the "height" family and the people items, the items of
    the family `PEOPLE_FAMILY`, which are test items and never train a
    reader.

    Args:
        config: The experiment settings.
        acts: The activations of every item, by template name.
        items: The questions of the families read, then the people.

    Returns:
        The study.

    Raises:
        ValueError: If no family of `config` has items, a family lacks a
            kind in a split, or an item has no activations.
    """
    wanted = config.families
    present = {item.family for item in items}
    families = tuple(
        f.name
        for f in FAMILIES
        if f.name in present and (wanted is None or f.name in wanted)
    )
    if not families:
        raise ValueError("no family to read")
    _check_sides(items, families)
    sets = train_sets(families)
    full = list(sets)[-1]
    groups = _groups(items, families, people=config.people)
    with_people = bool(groups.people.size)
    labels = _labels(items)
    seed, n_boot = config.seed, config.n_boot
    curves: list[Row] = []
    transfer: list[Row] = []
    baselines: list[Row] = []
    people: list[Row] = []
    chosen: dict[str, int] = {}
    scores: dict[str, NDArray[np.float64]] = {}
    readers: dict[str, RepReader] = {}
    for template in config.templates:
        found = acts[template]
        index = {uid: row for row, uid in enumerate(found.uids)}
        missing = [i.uid for i in items if i.uid not in index]
        if missing:
            raise ValueError(f"no activations for {missing[0]!r}")
        data = found.data[:, [index[i.uid] for i in items]]
        layers = found.layers
        for set_name, members in sets.items():
            train = _indices(items, families=members, split="train")
            val = _indices(items, families=members, split="val")
            for name in config.methods:
                per_test = {
                    test: np.zeros((config.n_seeds, len(layers)))
                    for test in groups.tests
                }
                val_auroc = np.zeros((config.n_seeds, len(layers)))
                total = np.zeros((len(layers), len(items)))
                for draw in range(config.n_seeds):
                    reader = _fit_reader(
                        data,
                        items,
                        labels,
                        train,
                        layers,
                        method=cast(Method, name),
                        seed=draw_seed(seed, draw),
                    )
                    if draw == 0 and set_name == full:
                        readers[f"{template}/{name}"] = reader
                    score = reader.score(data).astype(np.float64)
                    total += score
                    val_auroc[draw] = auroc(score[:, val], labels[val])
                    for test, held in groups.tests.items():
                        per_test[test][draw] = auroc(
                            score[:, held], labels[held]
                        )
                layer = best_layer(val_auroc.mean(axis=0), layers)
                key = f"{template}/{name}/{set_name}"
                chosen[key] = layer
                scores[key] = total[layers.index(layer)] / config.n_seeds
                identity: Row = {
                    "template": template,
                    "method": name,
                    "train_set": set_name,
                }
                for test, values in per_test.items():
                    lo, hi = bootstrap_ci(values, n_boot=n_boot, seed=seed)
                    curves += [
                        {
                            **identity,
                            "test_family": test,
                            "layer": at,
                            "auroc": float(values[:, k].mean()),
                            "lo": float(lo[k]),
                            "hi": float(hi[k]),
                        }
                        for k, at in enumerate(layers)
                    ]
                transfer += [
                    {**identity, "layer": layer, **row}
                    for row in _test_rows(
                        scores[key], items, groups, n_boot=n_boot, seed=seed
                    )
                ]
                if with_people:
                    people.append(
                        {
                            "source": "reader",
                            **identity,
                            "layer": layer,
                            **people_row(
                                scores[key], groups, n_boot=n_boot, seed=seed
                            ),
                        }
                    )
    for set_name, members in sets.items():
        for baseline, score in baseline_scores(items, members).items():
            baselines += [
                {"baseline": baseline, "train_set": set_name, **row}
                for row in _test_rows(
                    score, items, groups, n_boot=n_boot, seed=seed
                )
            ]
            if with_people:
                people.append(
                    {
                        "source": "baseline",
                        "template": None,
                        "method": baseline,
                        "train_set": set_name,
                        "layer": None,
                        **people_row(score, groups, n_boot=n_boot, seed=seed),
                    }
                )
    return Study(
        items=tuple(items),
        families=families,
        sets=sets,
        curves=curves,
        transfer=transfer,
        baselines=baselines,
        people=people,
        layers=chosen,
        scores=scores,
        readers=readers,
    )
