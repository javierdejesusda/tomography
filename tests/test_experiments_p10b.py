"""Tests for the P10b reading experiment."""

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.datasets.quantities import FAMILIES
from tomography.datasets.quantities import QuantityQuestion
from tomography.experiments.p10b_reading import BASELINES
from tomography.experiments.p10b_reading import Config
from tomography.experiments.p10b_reading import baseline_scores
from tomography.experiments.p10b_reading import train_sets
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
