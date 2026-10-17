"""Tests for the export of finished runs into a results tree."""

import dataclasses
import getpass
import io
import json
from pathlib import Path
import socket
import sys

import numpy as np
import pytest

from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.matrices import plot_matrix_file
from tomography.experiments.matrices import write_matrix
import tomography.export as export_module
from tomography.export import ExportError
from tomography.export import RunInfo
from tomography.export import export_runs
from tomography.export import is_exported


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Config(ExperimentConfig):
    experiment: str = "t1"
    model: str = "gpt2"
    tag: str = "x"
    source: str | None = None
    sources: tuple[str, ...] = ()


@pytest.fixture(autouse=True)
def _fixed_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(getpass, "getuser", lambda: "tester-user")
    monkeypatch.setattr(socket, "gethostname", lambda: "testhost-box.lan")


def _make_run(
    base: Path,
    *,
    finished: bool = True,
    nested: bool = False,
    git_sha: str | None = None,
    **fields: object,
) -> RunDir:
    out_root = base / "outer" / "inner" if nested else base
    run = RunDir.create(_Config(**fields), out_root=out_root)  # type: ignore[arg-type]
    run.write_json("summary", {"value": 1})
    if finished:
        run.finalize()
    if git_sha is not None:
        meta = json.loads((run.path / "meta.json").read_text("utf-8"))
        meta["git_sha"] = git_sha
        (run.path / "meta.json").write_text(json.dumps(meta), "utf-8")
    return run


def _relative_files(folder: Path) -> list[str]:
    return sorted(
        file.relative_to(folder).as_posix()
        for file in folder.rglob("*")
        if file.is_file()
    )


def _data(run_dir: Path, name: str) -> object:
    return json.loads((run_dir / name).read_text("utf-8"))["data"]


@pytest.mark.parametrize(
    "name",
    [
        "config.json",
        "meta.json",
        "summary.json",
        "table.json",
        "table_dose.json",
        "curves.json",
        "curves_val.npz",
        "matrix.npz",
        "readers.json",
        "hand_check.csv",
        "transfer_sentiment.npz",
        "transfer_sentiment.json",
    ],
)
def test_is_exported_keeps_the_files_the_plots_and_docs_need(
    name: str,
) -> None:
    assert is_exported(name)


@pytest.mark.parametrize(
    "name",
    [
        "readers.npz",
        "readers_triviaqa.npz",
        "sweep.npz",
        "sweep_a.npz",
        "items.json",
        "trials.json",
        "trials_sweep.npz",
        "transfer.npz",
        "signals_popqa.npz",
        "figures/curves.png",
        "figures/table.json",
        "run.log",
        "DONE.txt",
        "RUN_PATH.txt",
    ],
)
def test_is_exported_drops_raw_vectors_logs_and_figures(name: str) -> None:
    assert not is_exported(name)


def test_export_copies_finished_runs_in_both_nesting_styles(
    tmp_path: Path,
) -> None:
    flat = _make_run(tmp_path / "r1", tag="a")
    nested = _make_run(tmp_path / "r2", nested=True, experiment="t2", tag="b")
    dest = tmp_path / "dest"

    report = export_runs([tmp_path / "r1", tmp_path / "r2"], dest)

    flat_key = f"t1/gpt2/{flat.meta['config_hash']}"
    nested_key = f"t2/gpt2/{nested.meta['config_hash']}"
    assert sorted(run.key for run in report.exported) == [flat_key, nested_key]
    assert (dest / flat_key / "summary.json").is_file()
    assert (dest / nested_key / "config.json").is_file()
    assert report.total_bytes == sum(
        file.stat().st_size for file in dest.rglob("*") if file.is_file()
    )


def test_export_skips_unfinished_runs_and_reports_them(
    tmp_path: Path,
) -> None:
    done = _make_run(tmp_path / "r1", tag="a")
    open_run = _make_run(tmp_path / "r1", tag="b", finished=False)

    report = export_runs([tmp_path / "r1"], tmp_path / "dest")

    assert [run.key for run in report.exported] == [
        f"t1/gpt2/{done.meta['config_hash']}"
    ]
    assert report.skipped == [open_run.path]
    assert (
        not (tmp_path / "dest" / "t1" / "gpt2")
        .joinpath(open_run.meta["config_hash"])
        .exists()
    )


def test_export_applies_the_whitelist_and_blacklist(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "r1")
    for name in (
        "table_dose.json",
        "hand_check.csv",
        "readers.npz",
        "items.json",
        "trials_sweep.json",
        "run.log",
    ):
        (run.path / name).write_text("{}", "utf-8")
    (run.path / "figures").mkdir()
    (run.path / "figures" / "curves.png").write_bytes(b"png")
    run.write_arrays("curves_val", values=np.zeros(3))
    run.write_arrays("signals_x", values=np.zeros(3))

    export_runs([tmp_path / "r1"], tmp_path / "dest")

    exported = tmp_path / "dest" / "t1" / "gpt2" / run.meta["config_hash"]
    assert _relative_files(exported) == [
        "config.json",
        "curves_val.npz",
        "hand_check.csv",
        "meta.json",
        "summary.json",
        "table_dose.json",
    ]


def test_export_fails_on_a_collision_and_copies_nothing(
    tmp_path: Path,
) -> None:
    first = _make_run(tmp_path / "r1", git_sha="aaa111")
    _make_run(tmp_path / "r1", tag="other")
    second = _make_run(tmp_path / "r2", git_sha="bbb222")
    dest = tmp_path / "dest"

    with pytest.raises(ExportError) as error:
        export_runs([tmp_path / "r1", tmp_path / "r2"], dest)

    message = str(error.value)
    for text in (first.path, second.path, "aaa111", "bbb222"):
        assert str(text) in message
    assert not dest.exists()


def test_export_ignores_an_unfinished_duplicate(tmp_path: Path) -> None:
    _make_run(tmp_path / "r1")
    _make_run(tmp_path / "r2", finished=False)

    report = export_runs([tmp_path / "r1", tmp_path / "r2"], tmp_path / "dest")

    assert len(report.exported) == 1
    assert len(report.skipped) == 1


def test_reexport_of_identical_runs_changes_nothing(tmp_path: Path) -> None:
    _make_run(tmp_path / "r1")
    dest = tmp_path / "dest"
    export_runs([tmp_path / "r1"], dest)
    before = {
        file: file.stat().st_mtime_ns
        for file in dest.rglob("*")
        if file.is_file()
    }

    report = export_runs([tmp_path / "r1"], dest)

    assert [run.status for run in report.exported] == ["unchanged"]
    assert before == {
        file: file.stat().st_mtime_ns
        for file in dest.rglob("*")
        if file.is_file()
    }


def test_reexport_ignores_figures_generated_in_the_destination(
    tmp_path: Path,
) -> None:
    run = _make_run(tmp_path / "r1")
    dest = tmp_path / "dest"
    export_runs([tmp_path / "r1"], dest)
    figures = dest / "t1" / "gpt2" / run.meta["config_hash"] / "figures"
    figures.mkdir()
    (figures / "a.png").write_bytes(b"png")

    report = export_runs([tmp_path / "r1"], dest)

    assert [item.status for item in report.exported] == ["unchanged"]


def test_reexport_of_a_different_run_is_an_error(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "r1")
    dest = tmp_path / "dest"
    export_runs([tmp_path / "r1"], dest)
    run.write_json("summary", {"value": 2})

    with pytest.raises(ExportError, match=r"summary.json"):
        export_runs([tmp_path / "r1"], dest)

    exported = dest / "t1" / "gpt2" / run.meta["config_hash"]
    assert _data(exported, "summary.json") == {"value": 1}


def test_export_rewrites_paths_into_the_given_roots(tmp_path: Path) -> None:
    target = _make_run(tmp_path / "r1", tag="target")
    target_key = f"t1/gpt2/{target.meta['config_hash']}"
    nested_target = _make_run(tmp_path / "r2", nested=True, tag="deep")
    nested_key = f"t1/gpt2/{nested_target.meta['config_hash']}"
    unfinished = _make_run(tmp_path / "r1", tag="open", finished=False)
    unfinished_key = f"t1/gpt2/{unfinished.meta['config_hash']}"
    user = _make_run(
        tmp_path / "r1",
        tag="user",
        source=str(target.path),
        sources=(
            str(nested_target.path),
            str(target.path / "summary.json"),
            str(unfinished.path),
        ),
    )
    user.write_json(
        "table",
        {"rows": [["label", str(target.path)], {"run": str(target.path)}]},
    )

    export_runs([tmp_path / "r1", tmp_path / "r2"], tmp_path / "dest")

    exported = tmp_path / "dest" / "t1" / "gpt2" / user.meta["config_hash"]
    config = json.loads((exported / "config.json").read_text("utf-8"))
    assert config["source"] == f"results/{target_key}"
    assert config["sources"] == [
        f"results/{nested_key}",
        f"results/{target_key}/summary.json",
        unfinished_key,
    ]
    table = _data(exported, "table.json")
    assert table == {
        "rows": [
            ["label", f"results/{target_key}"],
            {"run": f"results/{target_key}"},
        ]
    }
    embedded = json.loads((exported / "table.json").read_text("utf-8"))
    assert embedded["_meta"]["config"]["source"] == f"results/{target_key}"


def test_export_rewrites_paths_outside_the_roots_and_warns(
    tmp_path: Path,
) -> None:
    _make_run(
        tmp_path / "r1",
        source="C:\\Users\\someone\\scratch\\old-run",
        sources=("/home/someone/runs/other", "relative/dir"),
    )

    report = export_runs([tmp_path / "r1"], tmp_path / "dest")

    (exported,) = (tmp_path / "dest").glob("t1/gpt2/*")
    config = json.loads((exported / "config.json").read_text("utf-8"))
    assert config["source"] == "<external>/old-run"
    assert config["sources"] == ["<external>/other", "relative/dir"]
    assert len(report.warnings) == 2
    assert "old-run" in report.warnings[0]


def test_export_names_an_outside_run_by_its_own_files(tmp_path: Path) -> None:
    outside = _make_run(tmp_path / "elsewhere", tag="far")
    _make_run(tmp_path / "r1", source=str(outside.path))

    report = export_runs([tmp_path / "r1"], tmp_path / "dest")

    (exported,) = (tmp_path / "dest").glob("t1/gpt2/*")
    config = json.loads((exported / "config.json").read_text("utf-8"))
    assert config["source"] == f"t1/gpt2/{outside.meta['config_hash']}"
    assert report.warnings == []


@pytest.mark.skipif(sys.platform != "win32", reason="backslash paths")
def test_export_rewrites_backslash_and_mixed_slash_paths(
    tmp_path: Path,
) -> None:
    target = _make_run(tmp_path / "r1", tag="target")
    key = f"t1/gpt2/{target.meta['config_hash']}"
    _make_run(
        tmp_path / "r1",
        tag="user",
        source=str(target.path).replace("/", "\\"),
        sources=(str(target.path).replace("\\", "/"),),
    )

    export_runs([tmp_path / "r1"], tmp_path / "dest")

    config = json.loads(
        _user_config(tmp_path / "dest", "user").read_text("utf-8")
    )
    assert config["source"] == f"results/{key}"
    assert config["sources"] == [f"results/{key}"]


def _user_config(dest: Path, tag: str) -> Path:
    for file in dest.glob("t1/gpt2/*/config.json"):
        if json.loads(file.read_text("utf-8"))["tag"] == tag:
            return file
    raise AssertionError(tag)


def test_export_rewrites_a_path_embedded_in_a_longer_string(
    tmp_path: Path,
) -> None:
    _make_run(tmp_path / "r1", tag="note", source="see C:/data/x/old-run now")

    export_runs([tmp_path / "r1"], tmp_path / "dest")

    config = json.loads(_user_config(tmp_path / "dest", "note").read_text())
    assert config["source"] == "see <external>/old-run now"


def test_export_rewrites_the_metadata_embedded_in_npz_files(
    tmp_path: Path,
) -> None:
    target = _make_run(tmp_path / "r1", tag="target")
    user = _make_run(tmp_path / "r1", tag="user", source=str(target.path))
    user.write_arrays("curves", values=np.arange(4.0), steps=np.arange(2))

    export_runs([tmp_path / "r1"], tmp_path / "dest")

    key = f"t1/gpt2/{target.meta['config_hash']}"
    exported = tmp_path / "dest" / "t1" / "gpt2" / user.meta["config_hash"]
    with np.load(exported / "curves.npz", allow_pickle=False) as archive:
        meta = json.loads(str(archive["_meta"]))
        values = archive["values"]
        steps = archive["steps"]
    assert meta["config"]["source"] == f"results/{key}"
    np.testing.assert_array_equal(values, np.arange(4.0))
    np.testing.assert_array_equal(steps, np.arange(2))
    copied = RunDir.open(exported)
    np.testing.assert_array_equal(
        copied.read_arrays("curves")["values"], np.arange(4.0)
    )


def test_export_keeps_npz_bytes_that_need_no_rewrite(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "r1")
    run.write_arrays("curves", values=np.arange(4.0))

    export_runs([tmp_path / "r1"], tmp_path / "dest")

    exported = tmp_path / "dest" / "t1" / "gpt2" / run.meta["config_hash"]
    assert (exported / "curves.npz").read_bytes() == (
        run.path / "curves.npz"
    ).read_bytes()


def test_export_refuses_a_file_that_still_holds_a_private_path(
    tmp_path: Path,
) -> None:
    run = _make_run(tmp_path / "r1")
    (run.path / "hand_check.csv").write_text(
        "id,path\n1,D:\\work\\data.csv\n", "utf-8"
    )
    dest = tmp_path / "dest"

    with pytest.raises(ExportError, match=r"hand_check.csv"):
        export_runs([tmp_path / "r1"], dest)

    assert not dest.exists()


@pytest.mark.parametrize(
    "leak", ["/home/Users/x", "owner: {user}"], ids=["users", "user-name"]
)
def test_export_refuses_the_substring_users_and_the_local_user_name(
    tmp_path: Path, leak: str
) -> None:
    run = _make_run(tmp_path / "r1")
    text = leak.format(user=getpass.getuser())
    (run.path / "hand_check.csv").write_text(text, "utf-8")

    with pytest.raises(ExportError, match=r"hand_check.csv"):
        export_runs([tmp_path / "r1"], tmp_path / "dest")


def test_export_refuses_a_private_segment_inside_npz_metadata(
    tmp_path: Path,
) -> None:
    run = _make_run(tmp_path / "r1")
    meta = {**run.meta, "note": "kept in x/Users/y"}
    buffer = io.BytesIO()
    np.savez(buffer, _meta=np.array(json.dumps(meta)), values=np.zeros(2))
    (run.path / "curves.npz").write_bytes(buffer.getvalue())
    dest = tmp_path / "dest"

    with pytest.raises(ExportError, match=r"curves.npz"):
        export_runs([tmp_path / "r1"], dest)

    assert not dest.exists()


def test_export_requires_existing_roots(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="not a directory"):
        export_runs([tmp_path / "missing"], tmp_path / "dest")


def test_an_exported_run_can_be_plotted(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "r1", experiment="a1", tag="plot")
    write_matrix(
        run,
        "matrix_concepts",
        np.eye(2),
        rows=("a", "b"),
        cols=("a", "b"),
        title="t",
    )
    dest = tmp_path / "dest"
    export_runs([tmp_path / "r1"], dest)
    exported = RunDir.open(dest / "a1" / "gpt2" / run.meta["config_hash"])

    paths = plot_matrix_file(
        exported, exported.path / "figures", stem="matrix_concepts"
    )

    assert len(paths) == 3
    assert all(path.is_file() for path in paths)


def test_a_p1_run_keeps_its_transfer_matrix_for_the_plot(
    tmp_path: Path,
) -> None:
    run = _make_run(tmp_path / "r1", experiment="p1", tag="transfer")
    write_matrix(
        run,
        "transfer_sentiment",
        np.eye(2),
        rows=("a", "b"),
        cols=("a", "b"),
        title="t",
    )
    (run.path / "transfer.npz").write_bytes(b"raw cube")
    dest = tmp_path / "dest"

    export_runs([tmp_path / "r1"], dest)

    exported = RunDir.open(dest / "p1" / "gpt2" / run.meta["config_hash"])
    assert not (exported.path / "transfer.npz").exists()
    paths = plot_matrix_file(
        exported,
        exported.path / "figures",
        stem="transfer_sentiment",
        vmin=0.0,
        vmax=1.0,
    )
    assert len(paths) == 3


def _leak_run(tmp_path: Path, text: str) -> Path:
    run = _make_run(tmp_path / "r1")
    (run.path / "hand_check.csv").write_text(text, "utf-8")
    return tmp_path / "dest"


def test_export_accepts_prose_that_mentions_users(tmp_path: Path) -> None:
    dest = _leak_run(tmp_path, "Users who asked twice, users table\n")

    report = export_runs([tmp_path / "r1"], dest)

    assert len(report.exported) == 1


@pytest.mark.parametrize(
    "text", ["x/Users/y", "x\\Users\\y"], ids=["slash", "backslash"]
)
def test_export_refuses_users_as_a_path_segment_with_a_snippet(
    tmp_path: Path, text: str
) -> None:
    dest = _leak_run(tmp_path, f"before {text} after")

    with pytest.raises(ExportError) as error:
        export_runs([tmp_path / "r1"], dest)

    assert "before" in str(error.value)
    assert "after" in str(error.value)
    assert not dest.exists()


@pytest.mark.parametrize(
    "text",
    [
        "kept in /home/alice/runs/x",
        "kept in /mnt/d/data/x",
        "at \\\\fileserver\\share\\runs",
        "on testhost-box today",
        "on TESTHOST-BOX.lan today",
        "by Tester-User today",
    ],
    ids=["posix", "wsl", "unc", "host-label", "host-full", "user"],
)
def test_export_refuses_posix_unc_host_and_user_leaks(
    tmp_path: Path, text: str
) -> None:
    dest = _leak_run(tmp_path, text)

    with pytest.raises(ExportError, match=r"hand_check\.csv"):
        export_runs([tmp_path / "r1"], dest)


def test_export_rewrites_embedded_posix_and_unc_paths(tmp_path: Path) -> None:
    source = "read /home/alice/data/old-run and \\\\srv1\\share\\x\\last"
    _make_run(tmp_path / "r1", tag="embedded", source=source)

    export_runs([tmp_path / "r1"], tmp_path / "dest")

    config = json.loads(
        _user_config(tmp_path / "dest", "embedded").read_text("utf-8")
    )
    assert config["source"] == "read <external>/old-run and <external>/last"


KEY = "p1/gpt2/abc123def456"


def _literal_rewriter() -> export_module._PathRewriter:
    path = "C:/work/res/p1/gpt2/abc123def456"
    run = RunInfo(Path(path), "p1", "gpt2", "abc123def456", None, True)
    index = {export_module._index_key(export_module._segments(path)): run}
    return export_module._PathRewriter(index, {KEY})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("C:\\work\\res\\p1\\gpt2\\abc123def456", f"results/{KEY}"),
        (
            "c:/WORK/res/p1/gpt2/abc123def456/curves.npz",
            f"results/{KEY}/curves.npz",
        ),
        ("/mnt/c/work/res/p1/gpt2/abc123def456", f"results/{KEY}"),
        ("/c/work/res/p1/gpt2/abc123def456/", f"results/{KEY}"),
        ("C:\\other\\place\\old-run", "<external>/old-run"),
        (
            "see file:///D:/other/place/old-run now",
            "see file:///<external>/old-run now",
        ),
        ("at \\\\srv1\\share\\old-run ok", "at <external>/old-run ok"),
        ("/home/alice/old-run", "<external>/old-run"),
        ("use C:\\a\\Jane Doe\\x for it", "use <external>/x for it"),
        ("C:\\a\\b and more", "<external>/b and more"),
        ("plain words", "plain words"),
    ],
)
def test_path_rewriter_maps_literal_paths_on_any_platform(
    text: str, expected: str
) -> None:
    assert _literal_rewriter().rewrite(text) == expected


def test_path_rewriter_leaves_no_space_path_tail() -> None:
    result = _literal_rewriter().rewrite("see C:\\Users\\Jane Doe\\runs\\x for")

    assert "Doe" not in result
    assert result == "see <external>/x for"


def test_export_refuses_a_drive_path_in_a_string_array_of_an_npz(
    tmp_path: Path,
) -> None:
    run = _make_run(tmp_path / "r1")
    run.write_arrays(
        "names",
        names=np.array(["fine", "D:\\private\\x"]),
        values=np.zeros(2),
    )
    dest = tmp_path / "dest"

    with pytest.raises(ExportError, match=r"names\.npz"):
        export_runs([tmp_path / "r1"], dest)

    assert not dest.exists()


def test_a_write_error_leaves_no_partial_run_and_export_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _make_run(tmp_path / "r1")
    dest = tmp_path / "dest"
    real = export_module._write_file
    calls: list[Path] = []

    def fail_on_second(target: Path, data: bytes) -> None:
        calls.append(target)
        if len(calls) == 2:
            raise OSError("disk full")
        real(target, data)

    with monkeypatch.context() as scope:
        scope.setattr(export_module, "_write_file", fail_on_second)
        with pytest.raises(ExportError, match="disk full"):
            export_runs([tmp_path / "r1"], dest)

    key = f"t1/gpt2/{run.meta['config_hash']}"
    assert not (dest / key).exists()
    assert not list(dest.rglob("*.tmp"))
    report = export_runs([tmp_path / "r1"], dest)
    assert [item.status for item in report.exported] == ["copied"]


def test_export_names_a_truncated_json_file(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "r1")
    (run.path / "table.json").write_text('{"data": [1, 2', "utf-8")

    with pytest.raises(ExportError, match=r"table\.json"):
        export_runs([tmp_path / "r1"], tmp_path / "dest")


def test_export_names_a_json_file_that_cannot_be_rewritten(
    tmp_path: Path,
) -> None:
    run = _make_run(tmp_path / "r1")
    (run.path / "table.json").write_text('{"data": [NaN, "C:/x/y/z"]}', "utf-8")

    with pytest.raises(ExportError, match=r"table\.json"):
        export_runs([tmp_path / "r1"], tmp_path / "dest")


def test_export_names_an_unreadable_npz_file(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "r1")
    (run.path / "curves.npz").write_bytes(b"not a zip")

    with pytest.raises(ExportError, match=r"curves\.npz"):
        export_runs([tmp_path / "r1"], tmp_path / "dest")
