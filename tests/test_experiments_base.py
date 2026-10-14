import dataclasses
import hashlib
import json
from typing import Literal

import pytest

from tomography.experiments.base import ExperimentConfig


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Config(ExperimentConfig):
    experiment: str = "t1"
    model: str = "gpt2"
    layers: tuple[int, ...] = (0, 1)
    alpha: float = 0.5
    note: str | None = None
    mode: Literal["a", "b"] = "a"


@dataclasses.dataclass(frozen=True, kw_only=True)
class _RequiredConfig(ExperimentConfig):
    n_pairs: int


def test_to_dict_lists_every_field_with_json_values() -> None:
    config = _Config(seed=3)

    assert config.to_dict() == {
        "experiment": "t1",
        "model": "gpt2",
        "seed": 3,
        "layers": [0, 1],
        "alpha": 0.5,
        "note": None,
        "mode": "a",
    }


def test_config_hash_is_twelve_hex_of_sha256_over_canonical_json() -> None:
    config = ExperimentConfig(experiment="r1", model="gpt2", seed=0)

    canonical = b'{"experiment":"r1","model":"gpt2","seed":0}'
    expected = hashlib.sha256(canonical).hexdigest()[:12]
    assert config.config_hash() == expected


def test_config_hash_is_equal_for_equal_configs() -> None:
    assert _Config(seed=1).config_hash() == _Config(seed=1).config_hash()


@pytest.mark.parametrize(
    "changed",
    [
        {"seed": 1},
        {"model": "gpt2-medium"},
        {"layers": (0, 1, 2)},
        {"alpha": 0.25},
        {"note": "x"},
    ],
    ids=["seed", "model", "layers", "alpha", "note"],
)
def test_config_hash_changes_with_every_field(
    changed: dict[str, object],
) -> None:
    base = _Config()

    assert dataclasses.replace(base, **changed).config_hash() != (
        base.config_hash()
    )


def test_config_hash_rejects_non_finite_floats() -> None:
    config = _Config(alpha=float("nan"))

    with pytest.raises(ValueError, match=r"(?i)nan"):
        config.config_hash()


def test_config_hash_matches_the_hash_of_the_stored_json() -> None:
    config = _Config(layers=(2, 4))

    stored = json.loads(json.dumps(config.to_dict()))
    canonical = json.dumps(
        stored, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    expected = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    assert config.config_hash() == expected


def test_from_dict_round_trips_through_json() -> None:
    config = _Config(seed=2, layers=(3, 5), note="x", mode="b")

    restored = _Config.from_dict(json.loads(json.dumps(config.to_dict())))

    assert restored == config
    assert restored.layers == (3, 5)


def test_from_dict_uses_defaults_for_missing_optional_fields() -> None:
    config = _Config.from_dict({"seed": 4})

    assert config == _Config(seed=4)


def test_from_dict_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match=r"unknown.*lyers"):
        _Config.from_dict({"lyers": [1]})


def test_from_dict_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match=r"missing.*n_pairs"):
        _RequiredConfig.from_dict({"experiment": "t2", "model": "gpt2"})


@pytest.mark.parametrize(
    ("data", "field"),
    [
        ({"seed": "7"}, "seed"),
        ({"seed": True}, "seed"),
        ({"alpha": "high"}, "alpha"),
        ({"layers": 3}, "layers"),
        ({"layers": [1, "2"]}, "layers"),
        ({"note": 5}, "note"),
        ({"mode": "c"}, "mode"),
        ({"model": None}, "model"),
    ],
    ids=[
        "str-for-int",
        "bool-for-int",
        "str-for-float",
        "scalar-for-tuple",
        "mixed-tuple",
        "int-for-optional-str",
        "bad-literal",
        "none-for-str",
    ],
)
def test_from_dict_rejects_values_of_the_wrong_type(
    data: dict[str, object], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        _Config.from_dict(data)


def test_from_dict_accepts_an_int_for_a_float_field() -> None:
    assert _Config.from_dict({"alpha": 1}).alpha == 1


def test_from_dict_accepts_null_for_an_optional_field() -> None:
    assert _Config.from_dict({"note": None}).note is None
