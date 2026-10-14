from collections.abc import Callable
from collections.abc import Iterator
import contextlib
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import stat
import subprocess
import sys
from typing import Literal

import numpy as np
import pytest
import torch

import tomography
import tomography.experiments.base as experiments_base
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.base import seed_everything


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


ReplaceFails = Callable[[], contextlib.AbstractContextManager[None]]


@pytest.fixture
def replace_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> ReplaceFails:
    """Returns a context manager in which `os.replace` raises OSError."""

    @contextlib.contextmanager
    def patched() -> Iterator[None]:
        def refuse(src: object, dst: object) -> None:
            raise OSError("disk full")

        with monkeypatch.context() as scope:
            scope.setattr(os, "replace", refuse)
            yield

    return patched


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


def test_from_dict_turns_an_int_for_a_float_field_into_a_float() -> None:
    config = _Config.from_dict({"alpha": 1})

    assert isinstance(config.alpha, float)
    assert config.config_hash() == _Config(alpha=1.0).config_hash()


def test_from_dict_rejects_an_int_too_large_for_a_float_field() -> None:
    with pytest.raises(ValueError, match="alpha"):
        _Config.from_dict({"alpha": 10**400})


def test_from_dict_accepts_null_for_an_optional_field() -> None:
    assert _Config.from_dict({"note": None}).note is None


def test_derive_seed_is_a_stable_function_of_base_and_keys() -> None:
    assert derive_seed(7, "a", 1) == 3428137257
    assert derive_seed(7, "a", 1) == derive_seed(7, "a", 1)


def test_derive_seed_fits_in_an_unsigned_32_bit_integer() -> None:
    seeds = [derive_seed(base, "k") for base in range(200)]

    assert all(0 <= seed < 2**32 for seed in seeds)
    assert len(set(seeds)) == len(seeds)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ((1, ("a",)), (2, ("a",))),
        ((1, ("a",)), (1, ("b",))),
        ((1, ("a", "b")), (1, ("b", "a"))),
        ((1, (1,)), (1, ("1",))),
        ((1, ()), (1, ("a",))),
    ],
    ids=["base", "key", "order", "type", "length"],
)
def test_derive_seed_separates_distinct_inputs(
    first: tuple[int, tuple[str | int, ...]],
    second: tuple[int, tuple[str | int, ...]],
) -> None:
    assert derive_seed(first[0], *first[1]) != derive_seed(
        second[0], *second[1]
    )


def test_seed_everything_returns_the_generator_of_the_seed() -> None:
    generator = seed_everything(11)

    expected = np.random.default_rng(11)
    assert generator.integers(0, 10**9, 5).tolist() == (
        expected.integers(0, 10**9, 5).tolist()
    )


def test_seed_everything_seeds_every_global_source() -> None:
    seed_everything(5)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(5)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second
    assert torch.initial_seed() == 5


@pytest.mark.parametrize("seed", [-1, 2**32], ids=["negative", "too-large"])
def test_seed_everything_rejects_seeds_outside_uint32(seed: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        seed_everything(seed)


META_KEYS = {
    "experiment",
    "config_hash",
    "seed",
    "git_sha",
    "git_dirty",
    "device",
    "platform",
    "versions",
    "started_at",
    "finished_at",
}


def test_create_makes_the_directory_named_by_the_config(
    tmp_path: Path,
) -> None:
    config = _Config(seed=3)

    run = RunDir.create(config, out_root=tmp_path)

    assert run.path == tmp_path / "t1" / "gpt2" / config.config_hash()
    assert run.path.is_dir()


def test_create_writes_the_config_and_meta_files(tmp_path: Path) -> None:
    config = _Config(seed=3)

    run = RunDir.create(config, out_root=tmp_path)

    stored = json.loads((run.path / "config.json").read_text("utf-8"))
    meta = json.loads((run.path / "meta.json").read_text("utf-8"))
    assert stored == config.to_dict()
    assert set(meta) == META_KEYS
    assert meta["experiment"] == "t1"
    assert meta["config_hash"] == config.config_hash()
    assert meta["seed"] == 3
    assert meta["finished_at"] is None


def test_create_refuses_an_existing_directory_unless_exist_ok(
    tmp_path: Path,
) -> None:
    config = _Config()
    RunDir.create(config, out_root=tmp_path)

    with pytest.raises(FileExistsError):
        RunDir.create(config, out_root=tmp_path)

    again = RunDir.create(config, out_root=tmp_path, exist_ok=True)
    assert again.path.is_dir()


@pytest.mark.parametrize(
    ("experiment", "model"),
    [
        ("../x", "gpt2"),
        ("a/b", "gpt2"),
        ("R1", "gpt2"),
        ("", "gpt2"),
        ("t1", "../../x"),
        ("t1", "a\\b"),
        ("t1", ""),
        ("t1", ".hidden"),
    ],
    ids=[
        "experiment-parent",
        "experiment-slash",
        "experiment-uppercase",
        "experiment-empty",
        "model-parent",
        "model-backslash",
        "model-empty",
        "model-dot",
    ],
)
def test_create_rejects_ids_that_are_not_safe_path_parts(
    tmp_path: Path, experiment: str, model: str
) -> None:
    config = _Config(experiment=experiment, model=model)

    with pytest.raises(ValueError, match=r"experiment|model"):
        RunDir.create(config, out_root=tmp_path / "out")

    assert not (tmp_path / "out").exists()


def test_create_defaults_to_results_under_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    run = RunDir.create(_Config())

    assert run.path.resolve().is_relative_to(tmp_path.resolve() / "results")


def test_open_returns_the_run_with_its_config_and_meta(
    tmp_path: Path,
) -> None:
    config = _Config(layers=(1, 2))
    created = RunDir.create(config, out_root=tmp_path)

    opened = RunDir.open(created.path)

    assert opened.path == created.path
    assert opened.config == config.to_dict()
    assert opened.meta == created.meta


def test_open_rejects_a_directory_that_is_not_a_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run directory"):
        RunDir.open(tmp_path)


def _new_run(tmp_path: Path, *, seed: int = 0) -> RunDir:
    return RunDir.create(_Config(seed=seed), out_root=tmp_path)


def test_write_json_embeds_the_meta_and_config(tmp_path: Path) -> None:
    run = _new_run(tmp_path, seed=2)

    path = run.write_json("curves", {"series": ["a"], "chance": 0.5})

    document = json.loads(path.read_text("utf-8"))
    assert path == run.path / "curves.json"
    assert document["data"] == {"series": ["a"], "chance": 0.5}
    assert document["_meta"] == {**run.meta, "config": run.config}


def test_read_json_returns_the_payload_without_metadata(
    tmp_path: Path,
) -> None:
    run = _new_run(tmp_path)
    run.write_json("table", {"columns": ["x"], "rows": [[1], [2]]})

    assert run.read_json("table") == {"columns": ["x"], "rows": [[1], [2]]}


def test_write_json_turns_non_finite_floats_into_null(
    tmp_path: Path,
) -> None:
    run = _new_run(tmp_path)

    run.write_json("v", {"a": [1.0, float("nan")], "b": {"c": float("inf")}})

    assert run.read_json("v") == {"a": [1.0, None], "b": {"c": None}}


def test_write_json_converts_numpy_values(tmp_path: Path) -> None:
    run = _new_run(tmp_path)
    payload = {
        "scalar": np.float32(0.25),
        "count": np.int64(3),
        "flag": np.bool_(True),
        "vector": np.array([1.0, np.nan]),
        "pair": (1, 2),
    }

    run.write_json("n", payload)

    assert run.read_json("n") == {
        "scalar": 0.25,
        "count": 3,
        "flag": True,
        "vector": [1.0, None],
        "pair": [1, 2],
    }


def test_write_json_rejects_values_json_cannot_hold(tmp_path: Path) -> None:
    run = _new_run(tmp_path)

    with pytest.raises(TypeError):
        run.write_json("bad", {"x": {1, 2}})

    assert not (run.path / "bad.json").exists()


def test_failed_write_keeps_the_previous_file_and_leaves_no_temp(
    tmp_path: Path, replace_fails: ReplaceFails
) -> None:
    run = _new_run(tmp_path)
    run.write_json("v", {"x": 1})

    with replace_fails(), pytest.raises(OSError, match="disk full"):
        run.write_json("v", {"x": 2})

    assert run.read_json("v") == {"x": 1}
    assert sorted(path.name for path in run.path.iterdir()) == [
        "config.json",
        "meta.json",
        "v.json",
    ]


@pytest.mark.parametrize(
    "name",
    ["", "../x", "a/b", "a\\b", ".hidden", "config", "meta", "Config", "META"],
    ids=[
        "empty",
        "parent",
        "slash",
        "backslash",
        "dot",
        "config",
        "meta",
        "config-upper",
        "meta-upper",
    ],
)
def test_write_json_rejects_unsafe_or_reserved_names(
    tmp_path: Path, name: str
) -> None:
    run = _new_run(tmp_path)

    with pytest.raises(ValueError, match="name"):
        run.write_json(name, {"x": 1})


def test_read_json_rejects_a_file_without_embedded_metadata(
    tmp_path: Path,
) -> None:
    run = _new_run(tmp_path)
    (run.path / "plain.json").write_text('{"x": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="metadata"):
        run.read_json("plain")


def test_write_arrays_round_trips_dtype_shape_and_values(
    tmp_path: Path,
) -> None:
    run = _new_run(tmp_path)
    scores = np.arange(12, dtype=np.float32).reshape(3, 4)
    layers = np.array([0, 4, 8], dtype=np.int64)

    path = run.write_arrays("scan", scores=scores, layers=layers)
    loaded = run.read_arrays("scan")

    assert path == run.path / "scan.npz"
    assert sorted(loaded) == ["layers", "scores"]
    assert loaded["scores"].dtype == np.float32
    assert loaded["layers"].dtype == np.int64
    np.testing.assert_array_equal(loaded["scores"], scores)
    np.testing.assert_array_equal(loaded["layers"], layers)


def test_write_arrays_embeds_the_meta_and_config(tmp_path: Path) -> None:
    run = _new_run(tmp_path, seed=4)

    path = run.write_arrays("scan", x=np.zeros(2))

    with np.load(path, allow_pickle=False) as archive:
        marker = archive["_meta"]
    assert marker.shape == ()
    assert marker.dtype.kind == "U"
    assert json.loads(str(marker)) == {**run.meta, "config": run.config}


def test_write_arrays_rejects_object_arrays_and_leaves_no_file(
    tmp_path: Path,
) -> None:
    run = _new_run(tmp_path)
    objects = np.array([object()], dtype=object)

    with pytest.raises(ValueError, match="allow_pickle"):
        run.write_arrays("bad", x=objects)

    assert sorted(path.name for path in run.path.iterdir()) == [
        "config.json",
        "meta.json",
    ]


@pytest.mark.parametrize(
    "key",
    ["_meta", "file", "allow_pickle"],
    ids=["meta", "file", "allow-pickle"],
)
def test_write_arrays_rejects_reserved_array_names(
    tmp_path: Path, key: str
) -> None:
    run = _new_run(tmp_path)

    with pytest.raises(ValueError, match=key):
        run.write_arrays("scan", **{key: np.zeros(1)})


def test_write_arrays_rejects_unsafe_names(tmp_path: Path) -> None:
    run = _new_run(tmp_path)

    with pytest.raises(ValueError, match="name"):
        run.write_arrays("../scan", x=np.zeros(1))


def test_failed_array_write_keeps_the_previous_file(
    tmp_path: Path, replace_fails: ReplaceFails
) -> None:
    run = _new_run(tmp_path)
    run.write_arrays("scan", x=np.ones(2))

    with replace_fails(), pytest.raises(OSError, match="disk full"):
        run.write_arrays("scan", x=np.zeros(2))

    np.testing.assert_array_equal(run.read_arrays("scan")["x"], np.ones(2))
    assert not [p for p in run.path.iterdir() if p.name.endswith(".tmp")]


def test_read_arrays_rejects_a_file_without_embedded_metadata(
    tmp_path: Path,
) -> None:
    run = _new_run(tmp_path)
    np.savez(run.path / "plain.npz", x=np.zeros(1))

    with pytest.raises(ValueError, match="metadata"):
        run.read_arrays("plain")


def test_figure_path_points_into_the_figures_folder(tmp_path: Path) -> None:
    run = _new_run(tmp_path)

    path = run.figure_path("curves", "pdf")

    assert path == run.path / "figures" / "curves.pdf"
    assert path.parent.is_dir()


def test_figure_path_accepts_the_ext_keyword(tmp_path: Path) -> None:
    run = _new_run(tmp_path)

    path = run.figure_path("curves", ext="pdf")

    assert path == run.path / "figures" / "curves.pdf"
    assert path.parent.is_dir()


@pytest.mark.parametrize(
    ("name", "extension", "label"),
    [
        ("../curves", "pdf", "name"),
        ("", "pdf", "name"),
        ("curves", "", "extension"),
        ("curves", ".pdf", "extension"),
        ("curves", "p/df", "extension"),
    ],
    ids=["name-parent", "name-empty", "ext-empty", "ext-dot", "ext-slash"],
)
def test_figure_path_rejects_unsafe_names_and_extensions(
    tmp_path: Path, name: str, extension: str, label: str
) -> None:
    run = _new_run(tmp_path)

    with pytest.raises(ValueError, match=label):
        run.figure_path(name, extension)


def test_finalize_stamps_the_finish_time(tmp_path: Path) -> None:
    run = _new_run(tmp_path)
    assert run.meta["finished_at"] is None

    run.finalize()

    reopened = RunDir.open(run.path)
    started = datetime.datetime.fromisoformat(reopened.meta["started_at"])
    finished = datetime.datetime.fromisoformat(reopened.meta["finished_at"])
    assert started.tzinfo is not None
    assert finished >= started
    assert run.meta == reopened.meta


def test_meta_records_the_platform_and_library_versions(
    tmp_path: Path,
) -> None:
    meta = _new_run(tmp_path).meta

    versions = meta["versions"]
    assert meta["platform"] == platform.platform()
    assert versions["python"] == platform.python_version()
    assert versions["tomography"] == tomography.__version__
    assert versions["numpy"] == np.__version__
    assert versions["torch"] == torch.__version__
    assert set(versions) == {
        "python",
        "tomography",
        "torch",
        "transformers",
        "numpy",
        "scikit-learn",
        "datasets",
        "peft",
        "matplotlib",
        "plotly",
    }


def test_meta_skips_libraries_that_are_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_version = importlib.metadata.version

    def version(name: str) -> str:
        if name == "peft":
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", version)

    versions = _new_run(tmp_path).meta["versions"]

    assert "peft" not in versions
    assert versions["numpy"] == np.__version__


def test_meta_device_is_cpu_without_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert _new_run(tmp_path).meta["device"] == "cpu"


def test_meta_device_names_the_cuda_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Test GPU")

    assert _new_run(tmp_path).meta["device"] == "cuda:0 (Test GPU)"


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def source_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for variable in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(variable, raising=False)
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "module.py").write_text("x = 1", encoding="utf-8")
    _git(repo, "add", "module.py")
    _git(repo, "commit", "-q", "-m", "init")
    monkeypatch.setattr(experiments_base, "_SOURCE_DIR", repo)
    return repo


def test_meta_records_the_commit_of_a_clean_source_tree(
    tmp_path: Path, source_repo: Path
) -> None:
    meta = _new_run(tmp_path / "out").meta

    assert meta["git_sha"] == _git(source_repo, "rev-parse", "HEAD")
    assert meta["git_dirty"] is False


def test_meta_marks_a_modified_tracked_file_as_dirty(
    tmp_path: Path, source_repo: Path
) -> None:
    (source_repo / "module.py").write_text("x = 2", encoding="utf-8")

    assert _new_run(tmp_path / "out").meta["git_dirty"] is True


def test_meta_ignores_untracked_files_when_judging_dirtiness(
    tmp_path: Path, source_repo: Path
) -> None:
    (source_repo / "results.json").write_text("{}", encoding="utf-8")

    assert _new_run(tmp_path / "out").meta["git_dirty"] is False


def test_meta_git_values_are_null_outside_a_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "missing"))

    meta = _new_run(tmp_path / "out").meta

    assert meta["git_sha"] is None
    assert meta["git_dirty"] is None


def test_path_for_names_the_directory_without_creating_it(
    tmp_path: Path,
) -> None:
    config = _Config(seed=3)

    path = RunDir.path_for(config, out_root=tmp_path)

    assert path == tmp_path / "t1" / "gpt2" / config.config_hash()
    assert not path.exists()
    assert RunDir.create(config, out_root=tmp_path).path == path


def test_path_for_rejects_ids_that_are_not_safe_path_parts(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="experiment"):
        RunDir.path_for(_Config(experiment="../x"), out_root=tmp_path)


def test_create_leaves_no_directory_when_the_metadata_cannot_be_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> str:
        raise RuntimeError("no gpu driver")

    monkeypatch.setattr(experiments_base, "_device_name", broken)
    config = _Config()

    with pytest.raises(RuntimeError, match="no gpu driver"):
        RunDir.create(config, out_root=tmp_path)

    assert not RunDir.path_for(config, out_root=tmp_path).exists()


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows has no POSIX permission bits"
)
def test_written_files_follow_the_umask(tmp_path: Path) -> None:
    previous = os.umask(0o022)
    try:
        run = _new_run(tmp_path)
        json_path = run.write_json("v", {"x": 1})
        npz_path = run.write_arrays("a", x=np.zeros(1))
    finally:
        os.umask(previous)

    assert stat.S_IMODE(json_path.stat().st_mode) == 0o644
    assert stat.S_IMODE(npz_path.stat().st_mode) == 0o644
