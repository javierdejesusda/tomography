from collections.abc import Iterator
import importlib
from pathlib import Path
import subprocess
import sys

import pytest

from tomography.cli import main
import tomography.experiments as experiments_package
from tomography.experiments.base import RunDir

DUMMY = """
import dataclasses
import os

from tomography.experiments.base import ExperimentConfig


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    experiment: str = "{id}"
    model: str = "gpt2"
    n: int = 3
    tag: str = "x"
    layers: tuple[int, ...] = (0, 1)


def run(config, run_dir):
    if os.environ.get("TOMOGRAPHY_TEST_CRASH"):
        raise RuntimeError("boom")
    run_dir.write_json(
        "result",
        {{"n": config.n, "tag": config.tag, "layers": config.layers}},
    )


def plot(run_dir, out_dir):
    data = run_dir.read_json("result")
    path = out_dir / "result.txt"
    path.write_text(str(data["n"]), encoding="utf-8")
    return [path]
"""


@pytest.fixture
def experiments_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    folder = tmp_path / "experiments"
    folder.mkdir()
    monkeypatch.setattr(
        experiments_package,
        "__path__",
        [*experiments_package.__path__, str(folder)],
    )
    loaded_before = set(sys.modules)
    yield folder
    for name in set(sys.modules) - loaded_before:
        if name.startswith("tomography.experiments."):
            del sys.modules[name]
    importlib.invalidate_caches()


def _write_experiment(
    folder: Path, stem: str, source: str | None = None
) -> None:
    experiment_id = stem.partition("_")[0]
    text = DUMMY.format(id=experiment_id) if source is None else source
    (folder / f"{stem}.py").write_text(text, encoding="utf-8")
    importlib.invalidate_caches()


def _usage_error(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    with pytest.raises(SystemExit) as exit_info:
        main(list(argv))

    assert exit_info.value.code == 2
    return capsys.readouterr().err


def test_run_finds_the_module_by_prefix_and_runs_it(
    experiments_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_experiment(experiments_dir, "d1_dummy")
    out_root = tmp_path / "results"

    code = main(["run", "d1", "--out", str(out_root)])

    printed = Path(capsys.readouterr().out.strip())
    run = RunDir.open(printed)
    assert code == 0
    assert printed.parent.parent.parent == out_root
    assert run.read_json("result") == {"n": 3, "tag": "x", "layers": [0, 1]}
    assert run.meta["finished_at"] is not None


def test_run_imports_only_the_matching_module(
    experiments_dir: Path, tmp_path: Path
) -> None:
    _write_experiment(experiments_dir, "d1_dummy")
    _write_experiment(
        experiments_dir, "d2_broken", "raise RuntimeError('imported')"
    )

    assert main(["run", "d1", "--out", str(tmp_path / "results")]) == 0


def test_run_matches_the_whole_id_not_a_prefix_of_it(
    experiments_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_experiment(experiments_dir, "d10_other")
    _write_experiment(experiments_dir, "d1_dummy")

    main(["run", "d1", "--out", str(tmp_path / "results")])

    assert Path(capsys.readouterr().out.strip()).parts[-3] == "d1"


def test_run_rejects_an_unknown_experiment_and_lists_the_known_ones(
    experiments_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_experiment(experiments_dir, "d1_dummy")

    error = _usage_error(capsys, "run", "nope")
    assert "unknown experiment 'nope'" in error
    assert "d1" in error


def test_run_rejects_two_modules_with_the_same_id(
    experiments_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_experiment(experiments_dir, "d1_first")
    _write_experiment(experiments_dir, "d1_second")

    error = _usage_error(capsys, "run", "d1")
    assert "d1_first" in error
    assert "d1_second" in error


@pytest.mark.parametrize(
    "source",
    [
        "def run(config, run_dir):\n    pass\n",
        "class Config:\n    pass\n\n\ndef run(c, r):\n    pass\n"
        "\n\ndef plot(r, o):\n    return []\n",
    ],
    ids=["missing-names", "config-is-not-an-experiment-config"],
)
def test_run_rejects_a_module_that_breaks_the_contract(
    experiments_dir: Path, capsys: pytest.CaptureFixture[str], source: str
) -> None:
    _write_experiment(experiments_dir, "d1_bad", source)

    error = _usage_error(capsys, "run", "d1")
    assert "d1_bad" in error


def test_a_command_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    error = _usage_error(capsys)
    assert "command" in error


def test_run_applies_the_model_seed_and_overrides(
    experiments_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_experiment(experiments_dir, "d1_dummy")

    code = main(
        [
            "run",
            "d1",
            "--out",
            str(tmp_path),
            "--model",
            "gpt2-medium",
            "--seed",
            "5",
            "--set",
            "n=7",
            "--set",
            "tag=hello",
            "--set",
            "layers=[2, 3]",
        ]
    )

    run = RunDir.open(Path(capsys.readouterr().out.strip()))
    assert code == 0
    assert run.config == {
        "experiment": "d1",
        "model": "gpt2-medium",
        "seed": 5,
        "n": 7,
        "tag": "hello",
        "layers": [2, 3],
    }


def test_run_keeps_numeric_looking_text_for_string_fields(
    experiments_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_experiment(experiments_dir, "d1_dummy")

    main(["run", "d1", "--out", str(tmp_path), "--set", "tag=123"])

    run = RunDir.open(Path(capsys.readouterr().out.strip()))
    assert run.config["tag"] == "123"


def test_run_needs_a_model_when_the_config_has_no_default(
    experiments_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = DUMMY.format(id="d1").replace('model: str = "gpt2"', "model: str")
    _write_experiment(experiments_dir, "d1_dummy", source)

    error = _usage_error(capsys, "run", "d1", "--out", str(tmp_path))
    assert "model" in error

    assert main(["run", "d1", "--out", str(tmp_path), "--model", "gpt2"]) == 0


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("n", "KEY=VALUE"),
        ("nope=1", "nope"),
        ("n=abc", "expected"),
        ("n=1.5", "expected"),
        ("layers=3", "expected"),
        ("experiment=d2", "experiment"),
    ],
    ids=[
        "no-equals",
        "unknown-field",
        "not-a-number",
        "float-for-int",
        "scalar-for-tuple",
        "experiment-id",
    ],
)
def test_run_rejects_bad_overrides(
    experiments_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    override: str,
    message: str,
) -> None:
    _write_experiment(experiments_dir, "d1_dummy")

    error = _usage_error(
        capsys, "run", "d1", "--out", str(tmp_path / "out"), "--set", override
    )
    assert message in error
    assert not (tmp_path / "out").exists()


def test_run_writes_to_results_in_the_working_directory_by_default(
    experiments_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_experiment(experiments_dir, "d1_dummy")
    monkeypatch.chdir(tmp_path)

    main(["run", "d1"])

    printed = Path(capsys.readouterr().out.strip())
    assert printed.resolve().is_relative_to(tmp_path.resolve() / "results")


def test_run_refuses_to_overwrite_a_finished_run(
    experiments_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_experiment(experiments_dir, "d1_dummy")
    main(["run", "d1", "--out", str(tmp_path)])
    capsys.readouterr()

    error = _usage_error(capsys, "run", "d1", "--out", str(tmp_path))
    assert "already exists" in error


def test_run_leaves_a_crashed_run_unfinished_and_can_rerun_it(
    experiments_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_experiment(experiments_dir, "d1_dummy")
    monkeypatch.setenv("TOMOGRAPHY_TEST_CRASH", "1")

    with pytest.raises(RuntimeError, match="boom"):
        main(["run", "d1", "--out", str(tmp_path)])

    (run_path,) = tmp_path.glob("d1/*/*")
    assert RunDir.open(run_path).meta["finished_at"] is None

    monkeypatch.delenv("TOMOGRAPHY_TEST_CRASH")
    assert main(["run", "d1", "--out", str(tmp_path)]) == 0
    assert RunDir.open(run_path).meta["finished_at"] is not None


def _run_cli(
    capsys: pytest.CaptureFixture[str], *argv: str
) -> tuple[int, list[str]]:
    code = main(list(argv))
    return code, capsys.readouterr().out.split()


def _run_experiment(
    capsys: pytest.CaptureFixture[str], experiment: str, out: Path, *args: str
) -> Path:
    _, printed = _run_cli(capsys, "run", experiment, "--out", str(out), *args)
    return Path(printed[-1])


def test_plot_regenerates_the_figures_of_every_run_of_an_experiment(
    experiments_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_experiment(experiments_dir, "d1_dummy")
    first = _run_experiment(capsys, "d1", tmp_path, "--set", "n=1")
    second = _run_experiment(capsys, "d1", tmp_path, "--set", "n=2")

    code, printed = _run_cli(capsys, "plot", "d1", "--out", str(tmp_path))

    assert code == 0
    assert sorted(Path(line) for line in printed) == [
        first / "figures" / "result.txt",
        second / "figures" / "result.txt",
    ]
    assert (first / "figures" / "result.txt").read_text("utf-8") == "1"
    assert (second / "figures" / "result.txt").read_text("utf-8") == "2"


def test_plot_all_covers_every_experiment(
    experiments_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_experiment(experiments_dir, "d1_dummy")
    _write_experiment(experiments_dir, "d2_dummy")
    runs = [
        _run_experiment(capsys, "d1", tmp_path),
        _run_experiment(capsys, "d2", tmp_path),
    ]

    code, _ = _run_cli(capsys, "plot", "all", "--out", str(tmp_path))

    assert code == 0
    assert all((run / "figures" / "result.txt").is_file() for run in runs)


def test_plot_run_regenerates_only_that_run(
    experiments_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_experiment(experiments_dir, "d1_dummy")
    first = _run_experiment(capsys, "d1", tmp_path, "--set", "n=1")
    second = _run_experiment(capsys, "d1", tmp_path, "--set", "n=2")

    code, _ = _run_cli(capsys, "plot", "d1", "--run", str(first))

    assert code == 0
    assert (first / "figures" / "result.txt").is_file()
    assert not (second / "figures").exists()


def test_plot_skips_unfinished_runs_and_says_so(
    experiments_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_experiment(experiments_dir, "d1_dummy")
    done = _run_experiment(capsys, "d1", tmp_path, "--set", "n=1")
    monkeypatch.setenv("TOMOGRAPHY_TEST_CRASH", "1")
    with pytest.raises(RuntimeError, match="boom"):
        main(["run", "d1", "--out", str(tmp_path), "--set", "n=2"])
    capsys.readouterr()
    monkeypatch.delenv("TOMOGRAPHY_TEST_CRASH")

    assert main(["plot", "all", "--out", str(tmp_path)]) == 0

    captured = capsys.readouterr()
    assert (done / "figures" / "result.txt").is_file()
    assert "unfinished" in captured.err
    assert len(captured.out.split()) == 1


@pytest.mark.parametrize("target", ["d1", "all"], ids=["id", "all"])
def test_plot_needs_at_least_one_finished_run(
    experiments_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target: str,
) -> None:
    _write_experiment(experiments_dir, "d1_dummy")

    error = _usage_error(capsys, "plot", target, "--out", str(tmp_path))
    assert "no finished runs" in error


def test_plot_rejects_a_run_of_another_experiment(
    experiments_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_experiment(experiments_dir, "d1_dummy")
    _write_experiment(experiments_dir, "d2_dummy")
    other = _run_experiment(capsys, "d2", tmp_path)

    error = _usage_error(capsys, "plot", "d1", "--run", str(other))
    assert "d2" in error


def test_plot_rejects_a_folder_that_is_not_a_run(
    experiments_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_experiment(experiments_dir, "d1_dummy")

    error = _usage_error(capsys, "plot", "d1", "--run", str(tmp_path))
    assert "not a run directory" in error


def test_plot_all_reports_results_of_an_experiment_without_a_module(
    experiments_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_experiment(experiments_dir, "d1_dummy")
    _run_experiment(capsys, "d1", tmp_path)
    stale = experiments_dir / "d3_gone.py"
    _write_experiment(experiments_dir, "d3_gone")
    _run_experiment(capsys, "d3", tmp_path)
    stale.unlink()
    importlib.invalidate_caches()
    sys.modules.pop("tomography.experiments.d3_gone", None)

    error = _usage_error(capsys, "plot", "all", "--out", str(tmp_path))
    assert "unknown experiment 'd3'" in error


def test_python_dash_m_runs_the_command_line() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "tomography", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "run" in completed.stdout
    assert "plot" in completed.stdout


def test_python_dash_m_exits_with_the_code_of_main(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tomography",
            "plot",
            "d1",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "no finished runs" in completed.stderr
