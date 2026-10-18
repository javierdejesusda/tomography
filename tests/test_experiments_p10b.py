"""Tests for the P10b reading experiment."""

from collections.abc import Callable
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.datasets.quantities import FAMILIES
from tomography.datasets.quantities import QuantityQuestion
from tomography.experiments.p10b_reading import BASELINES
from tomography.experiments.p10b_reading import Config
from tomography.experiments.p10b_reading import baseline_scores
from tomography.experiments.p10b_reading import read_study
from tomography.experiments.p10b_reading import train_sets
from tomography.p9.reading import ItemActivations
from tomography.reading.metrics import auroc

DEFAULT_HASH = "9d9170f5cccb"


def test_config_defaults_and_round_trip() -> None:
    config = Config()

    assert config.experiment == "p10b"
    assert config.model == "gpt2"
    assert config.seed == 0
    assert config.families is None
    assert config.n_per_side == 120
    assert config.min_sitelinks == 5
    assert config.people is True
    assert config.offline is False
    assert config.templates == ("plain", "concept")
    assert config.concept == "variability"
    assert config.methods == ("logistic", "pca_diff", "mean_diff", "random")
    assert config.n_seeds == 5
    assert config.n_boot == 1000
    assert config.batch_size == 8
    assert Config.from_dict(config.to_dict()) == config


def test_a_family_subset_round_trips() -> None:
    config = Config(families=("height", "duration"), people=False)

    assert Config.from_dict(config.to_dict()) == config


def test_the_default_config_hash_is_stable() -> None:
    assert Config().config_hash() == DEFAULT_HASH


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("families", ()),
        ("families", ("height", "height")),
        ("families", ("height", "no such family")),
        ("n_per_side", 0),
        ("n_seeds", 0),
        ("n_boot", 0),
        ("batch_size", 0),
        ("min_sitelinks", -1),
        ("concept", ""),
        ("templates", ()),
        ("templates", ("plain", "plain")),
        ("templates", ("fewshot",)),
        ("methods", ()),
        ("methods", ("logistic", "logistic")),
        ("methods", ("kmeans",)),
    ],
)
def test_bad_settings_raise_before_any_work(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        Config(**{field: value})


def test_one_family_is_its_own_train_set() -> None:
    assert train_sets(["height"]) == {"height": ("height",)}


def test_two_families_add_all_and_no_leave_one_out() -> None:
    assert train_sets(["height", "wingspan"]) == {
        "height": ("height",),
        "wingspan": ("wingspan",),
        "all": ("height", "wingspan"),
    }


def test_three_families_add_the_leave_one_out_pairs() -> None:
    names = [f.name for f in FAMILIES]

    sets = train_sets(names)

    assert list(sets) == [
        "height",
        "wingspan",
        "duration",
        "wingspan+duration",
        "height+duration",
        "height+wingspan",
        "all",
    ]
    assert sets["height+wingspan"] == ("height", "wingspan")
    assert sets["all"] == tuple(names)


def test_train_sets_follow_the_order_of_the_families_module() -> None:
    sets = train_sets(["duration", "height"])

    assert sets["all"] == ("height", "duration")


def _item(
    uid: str,
    family: str,
    kind: str,
    question: str,
    *,
    sitelinks: int = 10,
    split: str = "train",
) -> QuantityQuestion:
    return QuantityQuestion(
        uid=uid,
        family=family,
        kind=kind,  # type: ignore[arg-type]
        entity=uid,
        entity_id=f"Q{abs(hash(uid)) % 10000}",
        question=question,
        low=1.0,
        high=2.0 if kind == "range" else 1.0,
        unit="metres",
        sitelinks=sitelinks,
        split=split,  # type: ignore[arg-type]
        retrieved_on="2026-01-01",
        sha256="0" * 64,
    )


def _text_items(
    range_text: str, exact_text: str, *, range_links: int, exact_links: int
) -> list[QuantityQuestion]:
    items = []
    for split in ("train", "val", "test"):
        for i in range(12):
            items.append(
                _item(
                    f"height-range-{split}{i}",
                    "height",
                    "range",
                    f"{range_text} number {i}?",
                    sitelinks=range_links + i,
                    split=split,
                )
            )
            items.append(
                _item(
                    f"height-exact-{split}{i}",
                    "height",
                    "exact",
                    f"{exact_text} number {i}?",
                    sitelinks=exact_links + i,
                    split=split,
                )
            )
    return items


def _test_auroc(
    items: list[QuantityQuestion], scores: NDArray[np.float64]
) -> float:
    rows = [i for i, item in enumerate(items) if item.split == "test"]
    labels = np.array([items[i].kind == "range" for i in rows], dtype=np.int64)
    return float(auroc(scores[rows], labels))


def test_baselines_read_the_wording_the_length_and_the_popularity() -> None:
    items = _text_items(
        "How tall is a dog breed of the kind",
        "How tall is the building",
        range_links=1,
        exact_links=500,
    )

    scores = baseline_scores(items, ("height",))

    assert set(scores) == set(BASELINES)
    for name in BASELINES:
        assert scores[name].shape == (len(items),)
        assert _test_auroc(items, scores[name]) > 0.9, name


def test_baselines_are_oriented_on_the_train_items() -> None:
    items = _text_items(
        "Short",
        "A much longer question about the kind of thing",
        range_links=900,
        exact_links=1,
    )

    scores = baseline_scores(items, ("height",))

    assert _test_auroc(items, scores["length"]) > 0.9
    assert _test_auroc(items, scores["popularity"]) > 0.9


def test_baselines_fit_only_on_the_train_families() -> None:
    items = _text_items(
        "How tall is a dog breed of the kind",
        "How tall is the building",
        range_links=1,
        exact_links=500,
    )
    other = [
        _item(f"wingspan-{k}-{i}", "wingspan", k, f"Wing {k} {i}?", sitelinks=3)
        for k in ("range", "exact")
        for i in range(6)
    ]

    scores = baseline_scores([*items, *other], ("height",))

    assert scores["popularity"].shape == (len(items) + len(other),)
    assert _test_auroc(items, scores["popularity"][: len(items)]) > 0.9


WIDTH = 2048
LAYERS = (1, 2)
SPLITS = (("train", 16), ("val", 8), ("test", 16))
Shift = Callable[[QuantityQuestion], float]


def _study_items(
    families: Sequence[str] = ("height", "wingspan", "duration"),
    n_people: int = 40,
) -> list[QuantityQuestion]:
    items = [
        _item(
            f"{family}-{kind}-{split}{i}",
            family,
            kind,
            f"The {family} of the {kind} {split} thing {i}?",
            sitelinks=5 + i,
            split=split,
        )
        for family in families
        for kind in ("range", "exact")
        for split, n in SPLITS
        for i in range(n)
    ]
    items += [
        _item(
            f"person_height-exact-q{i}",
            "person_height",
            "exact",
            f"How tall is person {i}?",
            sitelinks=100 + i,
            split="test",
        )
        for i in range(n_people)
    ]
    return items


def _planted(
    items: Sequence[QuantityQuestion], shift: Shift
) -> ItemActivations:
    rng = np.random.default_rng(0)
    direction = np.random.default_rng(123).normal(size=WIDTH)
    direction /= np.linalg.norm(direction)
    data = rng.normal(0.0, 0.3, (len(LAYERS), len(items), WIDTH))
    for n, item in enumerate(items):
        data[:, n] += 3.0 * shift(item) * direction
    return ItemActivations(
        tuple(i.uid for i in items), LAYERS, data.astype(np.float32)
    )


def _by_range(item: QuantityQuestion) -> float:
    if item.family == "person_height":
        return -1.0
    return 1.0 if item.kind == "range" else -1.0


def _config(**changes: object) -> Config:
    base: dict[str, object] = {
        "templates": ("plain",),
        "n_seeds": 2,
        "n_boot": 40,
    }
    return Config(**{**base, **changes})


def _rows(
    rows: list[dict[str, object]], **where: object
) -> list[dict[str, object]]:
    return [r for r in rows if all(r[k] == v for k, v in where.items())]


def test_a_planted_direction_transfers_and_the_random_control_does_not() -> (
    None
):
    items = _study_items()
    study = read_study(_config(), {"plain": _planted(items, _by_range)}, items)

    control = []
    for row in study.transfer:
        if row["method"] == "random":
            control.append(float(row["auroc"]))
        else:
            assert float(row["auroc"]) > 0.9, row
    assert abs(float(np.mean(control)) - 0.5) < 0.1
    assert max(control) < 0.85


def test_curves_and_transfer_have_one_row_per_cell() -> None:
    items = _study_items()
    config = _config(methods=("logistic", "random"))
    study = read_study(config, {"plain": _planted(items, _by_range)}, items)

    n_sets = 7
    n_tests = 4
    assert len(study.transfer) == 2 * n_sets * n_tests
    assert len(study.curves) == 2 * n_sets * n_tests * len(LAYERS)
    assert set(study.transfer[0]) == {
        "template",
        "method",
        "train_set",
        "test_family",
        "layer",
        "n_range",
        "n_exact",
        "auroc",
        "lo",
        "hi",
    }
    assert set(study.curves[0]) == {
        "template",
        "method",
        "train_set",
        "test_family",
        "layer",
        "auroc",
        "lo",
        "hi",
    }
    tests = {r["test_family"] for r in study.transfer}
    assert tests == {"height", "wingspan", "duration", "pooled"}
    pooled = _rows(study.transfer, test_family="pooled")[0]
    assert (pooled["n_range"], pooled["n_exact"]) == (48, 48)
    one = _rows(study.transfer, test_family="height")[0]
    assert (one["n_range"], one["n_exact"]) == (16, 16)
    for row in study.transfer:
        assert float(row["lo"]) <= float(row["auroc"]) <= float(row["hi"])


def test_the_layer_is_chosen_on_the_validation_items_of_the_train_set() -> None:
    items = _study_items()
    acts = _planted(items, _by_range)
    data = acts.data.copy()
    data[0] = np.random.default_rng(1).normal(0.0, 0.3, data[0].shape)
    noisy_first = ItemActivations(acts.uids, acts.layers, data)

    study = read_study(
        _config(methods=("logistic",)), {"plain": noisy_first}, items
    )

    assert {r["layer"] for r in study.transfer} == {2}


def test_a_flipped_plant_still_reads_above_chance() -> None:
    items = _study_items()

    def flipped(item: QuantityQuestion) -> float:
        return -_by_range(item)

    study = read_study(_config(), {"plain": _planted(items, flipped)}, items)

    for row in study.transfer:
        if row["method"] != "random":
            assert float(row["auroc"]) > 0.9, row


def _diagnostic(
    rows: list[dict[str, object]], train_set: str
) -> dict[str, object]:
    return _rows(rows, source="reader", method="logistic", train_set=train_set)[
        0
    ]


def test_people_planted_with_the_exact_side_read_as_exact() -> None:
    items = _study_items()
    study = read_study(_config(), {"plain": _planted(items, _by_range)}, items)

    for train_set in ("all", "wingspan+duration"):
        row = _diagnostic(study.people, train_set)
        assert float(row["breeds_vs_people"]) > 0.9
        assert abs(float(row["people_vs_buildings"]) - 0.5) < 0.2
        assert abs(float(row["position"])) < 0.2


def test_people_planted_with_the_range_side_read_as_range() -> None:
    items = _study_items()

    def living(item: QuantityQuestion) -> float:
        return 1.0 if item.family == "person_height" else _by_range(item)

    study = read_study(_config(), {"plain": _planted(items, living)}, items)

    for train_set in ("all", "wingspan+duration"):
        row = _diagnostic(study.people, train_set)
        assert abs(float(row["breeds_vs_people"]) - 0.5) < 0.2
        assert float(row["people_vs_buildings"]) > 0.9
        assert abs(float(row["position"]) - 1.0) < 0.2


def test_the_people_diagnostic_covers_readers_and_baselines() -> None:
    items = _study_items()
    study = read_study(_config(), {"plain": _planted(items, _by_range)}, items)

    assert {r["source"] for r in study.people} == {"reader", "baseline"}
    names = {r["method"] for r in _rows(study.people, source="baseline")}
    assert names == set(BASELINES)
    row = _rows(study.people, source="reader")[0]
    assert row["layer"] in LAYERS
    assert (row["n_breeds"], row["n_buildings"], row["n_people"]) == (
        16,
        16,
        40,
    )
    assert float(row["breeds_vs_people_lo"]) <= float(row["breeds_vs_people"])
    assert float(row["people_vs_buildings_hi"]) >= float(
        row["people_vs_buildings"]
    )


def test_there_is_no_people_diagnostic_without_the_flag_or_the_family() -> None:
    items = _study_items()
    acts = {"plain": _planted(items, _by_range)}

    assert read_study(_config(people=False), acts, items).people == []

    no_height = [i for i in items if i.family != "height"]
    acts = {"plain": _planted(no_height, _by_range)}
    study = read_study(_config(), acts, no_height)
    assert study.people == []
    assert {r["train_set"] for r in study.transfer} == {
        "wingspan",
        "duration",
        "all",
    }


def test_baseline_rows_cover_every_train_set_and_test_family() -> None:
    items = _study_items()
    study = read_study(_config(), {"plain": _planted(items, _by_range)}, items)

    assert len(study.baselines) == len(BASELINES) * 7 * 4
    assert set(study.baselines[0]) == {
        "baseline",
        "train_set",
        "test_family",
        "n_range",
        "n_exact",
        "auroc",
        "lo",
        "hi",
    }
    assert {r["baseline"] for r in study.baselines} == set(BASELINES)
