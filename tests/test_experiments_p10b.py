"""Tests for the P10b reading experiment."""

import pytest

from tomography.datasets.quantities import FAMILIES
from tomography.experiments.p10b_reading import Config
from tomography.experiments.p10b_reading import train_sets

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
