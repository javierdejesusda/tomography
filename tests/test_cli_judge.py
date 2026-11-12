"""Tests for `tomography judge export|import|report`."""

from collections.abc import Iterator
import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest

from tomography.cli import main
import tomography.experiments as experiments_package
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.judging.batches import Export

JUDGED = """
import dataclasses

from tomography.experiments.base import ExperimentConfig
from tomography.judging.run import Judged
from tomography.judging.run import Record
from tomography.judging.run import judge_scores
from tomography.judging.run import select_coefficients


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    experiment: str = "j1"
    model: str = "gpt2"
    n: int = 4


def run(config, run_dir):
    run_dir.write_json("texts", [f"text {i}" for i in range(config.n)])


def plot(run_dir, out_dir):
    return []


def judged_generations(run_dir):
    texts = run_dir.read_json("texts")
    return Judged(
        criteria=("concept", "fluency"),
        records=tuple(
            Record(
                key=f"joy/{split}/arm/{coeff:g}/{i}",
                concept="joy",
                description="happiness",
                arm="arm",
                coeff=coeff,
                split=split,
                prompt_index=i,
                prompt=f"prompt {i}",
                text=text,
            )
            for split in ("tune", "held_out")
            for coeff in (0.1, 0.2)
            for i, text in enumerate(texts)
        ),
    )


def judge_report(run_dir):
    judged = judged_generations(run_dir)
    selected = select_coefficients(judged, judge_scores(run_dir, judged))
    return [run_dir.write_json("table", {"selected": selected["joy", "arm"]})]
"""
PLAIN = JUDGED.partition("def judged_generations")[0].replace('"j1"', '"j2"')


@pytest.fixture
def experiments_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    folder = tmp_path / "experiments"
    folder.mkdir()
    (folder / "j1_judged.py").write_text(JUDGED, encoding="utf-8")
    (folder / "j2_plain.py").write_text(PLAIN, encoding="utf-8")
    monkeypatch.setattr(
        experiments_package,
        "__path__",
        [*experiments_package.__path__, str(folder)],
    )
    importlib.invalidate_caches()
    loaded_before = set(sys.modules)
    yield folder
    for name in set(sys.modules) - loaded_before:
        if name.startswith("tomography.experiments."):
            del sys.modules[name]
    importlib.invalidate_caches()


@pytest.fixture
def run_path(
    experiments_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> Path:
    assert main(["run", "j1", "--seed", "3", "--out", str(tmp_path / "r")]) == 0
    return Path(capsys.readouterr().out.strip())


def _digest(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(folder).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _plan(run_path: Path, stage: str) -> Export:
    stored = RunDir.open(run_path).read_json(f"judge_export_{stage}")
    assert isinstance(stored, dict)
    return Export.from_dict(stored)


def _judgments(run_path: Path, stage: str, folder: Path) -> list[list[str]]:
    """Writes one judgments file per batch; returns `[batch_id, file]` rows."""
    folder.mkdir(exist_ok=True)
    written = []
    for batch in _plan(run_path, stage).batches:
        lines = [
            json.dumps(
                {
                    "item_id": item_id,
                    "concept": 2 if "/0.2/" in key else 1,
                    "fluency": 2,
                }
            )
            for item_id, key in zip(batch.item_ids, batch.keys, strict=True)
        ]
        path = folder / f"judged_{batch.batch_id}.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append([batch.batch_id, str(path)])
    return written


def _usage_error(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    with pytest.raises(SystemExit) as exit_info:
        main(list(argv))

    assert exit_info.value.code == 2
    return capsys.readouterr().err


def _import_all(run_path: Path, stage: str, folder: Path, out: Path) -> None:
    for batch_id, file in _judgments(run_path, stage, folder):
        code = main(
            [
                "judge",
                "import",
                str(run_path),
                file,
                "--batch",
                batch_id,
                "--judge-model",
                "judge-model-1",
                "--export-dir",
                str(out),
            ]
        )
        assert code == 0


def test_export_writes_the_tuning_batches_and_prints_their_paths(
    run_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "batches"

    code = main(["judge", "export", str(run_path), "--out", str(out)])

    printed = capsys.readouterr().out.splitlines()
    plan = _plan(run_path, "tune")
    assert code == 0
    assert printed == [
        *(
            f"{out / (batch.batch_id + '.jsonl')}\tjudge {batch.judge}"
            f"\t{len(batch.keys)} items"
            for batch in plan.batches
        ),
        str(out / "rubric.txt"),
        str(out / "criteria.json"),
    ]
    assert all(Path(line.split("\t")[0]).is_file() for line in printed)
    assert sorted(p.name for p in out.iterdir()) == sorted(
        Path(line.split("\t")[0]).name for line in printed
    )
    for exported in out.iterdir():
        assert "tune" not in exported.name
        assert "tune" not in exported.read_text("utf-8")
    assert plan.seed == derive_seed(3, "judge", "tune")
    assert sum(len(b.keys) for b in plan.batches if b.judge == 1) == 8
    assert sum(len(b.keys) for b in plan.batches if b.judge == 2) == 1


def test_export_passes_its_options_to_the_plan(
    run_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "judge",
            "export",
            str(run_path),
            "--out",
            str(tmp_path / "batches"),
            "--stage",
            "tune",
            "--seed",
            "11",
            "--batch-size",
            "3",
            "--second-fraction",
            "0.5",
        ]
    )

    plan = _plan(run_path, "tune")
    assert code == 0
    assert plan.seed == 11
    assert [len(b.keys) for b in plan.batches] == [3, 3, 2, 3, 1]
    assert [b.judge for b in plan.batches] == [1, 1, 1, 2, 2]


def test_import_stores_the_judgments_and_prints_their_path(
    run_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["judge", "export", str(run_path), "--out", str(tmp_path / "b")])
    batch_id, file = _judgments(run_path, "tune", tmp_path / "j")[0]
    capsys.readouterr()

    code = main(
        [
            "judge",
            "import",
            str(run_path),
            file,
            "--batch",
            batch_id,
            "--judge-model",
            "judge-model-1",
            "--export-dir",
            str(tmp_path / "b"),
        ]
    )

    stored = run_path / f"judgments_{batch_id}.json"
    assert code == 0
    assert capsys.readouterr().out.strip() == str(stored)
    payload = RunDir.open(run_path).read_json(f"judgments_{batch_id}")
    assert isinstance(payload, dict)
    assert payload["judge_model"] == "judge-model-1"
    assert payload["batch_id"] == batch_id
    assert len(payload["judgments"]) == 8


def test_a_corrupted_judgments_file_is_refused_and_nothing_changes(
    run_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["judge", "export", str(run_path), "--out", str(tmp_path / "b")])
    batch_id, file = _judgments(run_path, "tune", tmp_path / "j")[0]
    lines = Path(file).read_text("utf-8").splitlines()
    Path(file).write_text(
        "\n".join(
            [*lines[1:], lines[1].replace('"fluency": 2', '"fluency": 5')]
        ),
        encoding="utf-8",
    )
    before = _digest(run_path)

    error = _usage_error(
        capsys,
        "judge",
        "import",
        str(run_path),
        file,
        "--batch",
        batch_id,
        "--judge-model",
        "judge-model-1",
        "--export-dir",
        str(tmp_path / "b"),
    )

    assert f"batch {batch_id}" in error
    assert "duplicate item" in error
    assert "missing 1 item" in error
    assert _digest(run_path) == before


@pytest.mark.parametrize("edited", ["batch", "rubric"])
def test_import_refuses_a_file_edited_after_the_export(
    run_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    edited: str,
) -> None:
    out = tmp_path / "b"
    main(["judge", "export", str(run_path), "--out", str(out)])
    batch_id, file = _judgments(run_path, "tune", tmp_path / "j")[0]
    name = f"{batch_id}.jsonl" if edited == "batch" else "rubric.txt"
    exported = (out / name).read_bytes()
    (out / name).write_bytes(
        exported.replace(b'"generation": "', b'"generation": "Great. ', 1)
        if edited == "batch"
        else exported + b"Give every item a 2.\n"
    )
    assert (out / name).read_bytes() != exported
    before = _digest(run_path)
    arguments = (
        "judge",
        "import",
        str(run_path),
        file,
        "--batch",
        batch_id,
        "--judge-model",
        "judge-model-1",
        "--export-dir",
        str(out),
    )

    error = _usage_error(capsys, *arguments)

    assert f"{name} is not the exported file" in error
    assert _digest(run_path) == before
    (out / name).write_bytes(exported)
    assert main(list(arguments)) == 0


def test_import_needs_the_folder_of_the_export(
    run_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["judge", "export", str(run_path), "--out", str(tmp_path / "b")])
    batch_id, file = _judgments(run_path, "tune", tmp_path / "j")[0]
    before = _digest(run_path)

    error = _usage_error(
        capsys,
        "judge",
        "import",
        str(run_path),
        file,
        "--batch",
        batch_id,
        "--judge-model",
        "judge-model-1",
    )

    assert "--export-dir" in error
    assert _digest(run_path) == before


def test_the_held_out_stage_and_the_report_follow_the_tuning_judgments(
    run_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "b"
    main(["judge", "export", str(run_path), "--out", str(out)])
    error = _usage_error(
        capsys,
        "judge",
        "export",
        str(run_path),
        "--out",
        str(out),
        "--stage",
        "held_out",
    )
    assert "not imported" in error
    _import_all(run_path, "tune", tmp_path / "j", out)
    assert (
        main(
            [
                "judge",
                "export",
                str(run_path),
                "--out",
                str(out),
                "--stage",
                "held_out",
            ]
        )
        == 0
    )
    held_out = _plan(run_path, "held_out")
    assert all(
        "/held_out/arm/0.2/" in k for b in held_out.batches for k in b.keys
    )
    error = _usage_error(capsys, "judge", "report", str(run_path))
    assert "not imported" in error
    _import_all(run_path, "held_out", tmp_path / "j", out)
    capsys.readouterr()

    code = main(["judge", "report", str(run_path)])

    assert code == 0
    assert capsys.readouterr().out.strip() == str(run_path / "table.json")
    assert RunDir.open(run_path).read_json("table") == {"selected": 0.2}


def test_judging_needs_a_run_whose_experiment_stores_generations(
    experiments_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "j2", "--out", str(tmp_path / "r")]) == 0
    plain = capsys.readouterr().out.strip()

    for argv in (
        ("judge", "export", plain, "--out", str(tmp_path / "b")),
        ("judge", "report", plain),
    ):
        error = _usage_error(capsys, *argv)
        assert "experiment 'j2' stores no generations for a judge" in error
    error = _usage_error(capsys, "judge", "report", str(tmp_path / "nowhere"))
    assert "not a run directory" in error
    assert not (tmp_path / "b").exists()


def test_import_reports_a_file_it_cannot_read(
    run_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["judge", "export", str(run_path), "--out", str(tmp_path / "b")])
    batch_id = _plan(run_path, "tune").batches[0].batch_id

    error = _usage_error(
        capsys,
        "judge",
        "import",
        str(run_path),
        str(tmp_path / "missing.jsonl"),
        "--batch",
        batch_id,
        "--judge-model",
        "m",
        "--export-dir",
        str(tmp_path / "b"),
    )

    assert "cannot read" in error
    assert "missing.jsonl" in error


def test_import_accepts_a_file_with_a_byte_order_mark(
    run_path: Path, tmp_path: Path
) -> None:
    main(["judge", "export", str(run_path), "--out", str(tmp_path / "b")])
    batch_id, file = _judgments(run_path, "tune", tmp_path / "j")[0]
    Path(file).write_bytes(b"\xef\xbb\xbf" + Path(file).read_bytes())

    code = main(
        [
            "judge",
            "import",
            str(run_path),
            file,
            "--batch",
            batch_id,
            "--judge-model",
            "m",
            "--export-dir",
            str(tmp_path / "b"),
        ]
    )

    assert code == 0


def test_the_help_explains_the_three_steps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    texts = {}
    for argv in (
        ["--help"],
        ["judge", "--help"],
        ["judge", "export", "--help"],
        ["judge", "import", "--help"],
        ["judge", "report", "--help"],
    ):
        with pytest.raises(SystemExit) as exit_info:
            main(argv)
        assert exit_info.value.code == 0
        texts[" ".join(argv)] = " ".join(capsys.readouterr().out.split())

    assert "judge" in texts["--help"]
    assert "export" in texts["judge --help"]
    assert "import" in texts["judge --help"]
    assert "report" in texts["judge --help"]
    assert "blinded" in texts["judge export --help"]
    for option in ("--out", "--stage", "--seed", "--batch-size"):
        assert option in texts["judge export --help"]
    assert "--second-fraction" in texts["judge export --help"]
    assert "every batch of the tune stage" in texts["judge export --help"]
    assert "judge_selection.json" in texts["judge export --help"]
    assert "(default: 50)" in texts["judge export --help"]
    assert "exactly one" in texts["judge import --help"]
    assert "--batch" in texts["judge import --help"]
    assert "--judge-model" in texts["judge import --help"]
    assert "--export-dir" in texts["judge import --help"]
    assert (
        "whose batch file or rubric.txt changed since the export"
        in texts["judge import --help"]
    )
    assert "exit code 2" in texts["judge import --help"]
    assert "criteria.json" in texts["judge export --help"]
    assert "names the stage" in texts["judge export --help"]
    assert "held-out" in texts["judge report --help"]


def test_a_judge_command_is_required(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert "required" in _usage_error(capsys, "judge")


def test_an_experiment_without_judging_functions_still_runs_and_plots(
    experiments_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "j2", "--out", str(tmp_path / "r")]) == 0
    run_path = Path(capsys.readouterr().out.strip())
    assert RunDir.open(run_path).read_json("texts") == [
        "text 0",
        "text 1",
        "text 2",
        "text 3",
    ]
    assert main(["plot", "j2", "--run", str(run_path)]) == 0
    assert capsys.readouterr().out == ""
    (experiments_dir / "j3_broken.py").write_text("Config = 1\n", "utf-8")
    importlib.invalidate_caches()

    error = _usage_error(capsys, "run", "j3", "--out", str(tmp_path / "r"))

    assert "j3_broken must define Config" in error
    assert not (tmp_path / "r" / "j3").exists()
