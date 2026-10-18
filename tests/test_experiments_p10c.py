"""Tests for the P10c links experiment."""

import pytest

from tomography.experiments.p10c_links import Config

PINNED_HASH = "ff466673e0c1"


def test_config_defaults_and_round_trip() -> None:
    config = Config(p10b_run="runs/p10b", knowledge_run="runs/p9b")

    assert config.experiment == "p10c"
    assert config.model == "gpt2"
    assert config.seed == 0
    assert config.p10b_run == "runs/p10b"
    assert config.knowledge_run == "runs/p9b"
    assert config.behavior_run is None
    assert config.reader == "plain/logistic"
    assert config.n_boot == 1000
    assert config.n_random == 200
    assert config.batch_size == 8
    assert config.offline is True
    assert Config.from_dict(config.to_dict()) == config


def test_both_links_round_trip() -> None:
    config = Config(
        p10b_run="a",
        knowledge_run="b",
        behavior_run="c",
        reader="concept/mean_diff",
    )

    assert Config.from_dict(config.to_dict()) == config


def test_the_p10b_run_is_required() -> None:
    with pytest.raises(ValueError, match="p10b_run"):
        Config.from_dict({"knowledge_run": "y"})


def test_the_config_hash_is_pinned() -> None:
    config = Config(p10b_run="x", knowledge_run="y")

    assert config.config_hash() == PINNED_HASH


def test_a_link_is_required() -> None:
    with pytest.raises(ValueError, match="knowledge_run"):
        Config(p10b_run="x")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_boot", 0),
        ("n_random", 0),
        ("batch_size", 0),
        ("reader", "plain"),
        ("reader", "plain/logistic/extra"),
        ("reader", "chat/logistic"),
        ("reader", "plain/kmeans"),
        ("reader", ""),
    ],
)
def test_bad_settings_raise_before_any_work(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        Config(p10b_run="x", knowledge_run="y", **{field: value})
