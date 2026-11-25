"""The --shard option splits the suite into disjoint parts that cover it.

CI runs the suite as parallel jobs, one shard each. A shard holds whole test
files, so what the tests of a file share (a module-scoped fixture or a run
cached in a module global) is built in one shard only. The files are packed
by the time they took, as `--store-durations` recorded it.
"""

import json
from pathlib import Path
from typing import Any

import pytest

pytest_plugins = ("pytester",)

CONFTEST = Path(__file__).with_name("conftest.py")
TEMPLATE = """
import pytest


@pytest.mark.parametrize("n", range(4))
def test_function(n):
    pass


class TestGroup:
    def test_method(self):
        pass
"""
FILES = 10
COUNT = 3


@pytest.fixture
def suite(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeconftest(CONFTEST.read_text(encoding="utf-8"))
    pytester.makepyfile(**{f"test_file_{i}": TEMPLATE for i in range(FILES)})
    return pytester


def _collected(pytester: pytest.Pytester, *args: str) -> tuple[set[str], int]:
    """Returns the ids a run keeps and the number it deselects."""
    recorder = pytester.inline_run("--collect-only", *args)
    session = recorder.getcall("pytest_collection_finish").session
    deselected = sum(
        len(call.items) for call in recorder.getcalls("pytest_deselected")
    )
    return {item.nodeid for item in session.items}, deselected


def test_without_the_option_every_test_runs(suite: pytest.Pytester) -> None:
    ids, deselected = _collected(suite)
    assert len(ids) == 5 * FILES
    assert deselected == 0


def test_the_shards_are_disjoint_and_cover_the_suite(
    suite: pytest.Pytester,
) -> None:
    every, _ = _collected(suite)
    shards = [_collected(suite, f"--shard={i}/{COUNT}") for i in range(COUNT)]
    for ids, deselected in shards:
        assert deselected == len(every) - len(ids)
    assert sum(1 for ids, _ in shards if ids) > 1
    assert sum(len(ids) for ids, _ in shards) == len(every)
    assert set().union(*(ids for ids, _ in shards)) == every


def test_the_tests_of_a_file_stay_in_one_shard(
    suite: pytest.Pytester,
) -> None:
    shards = [
        _collected(suite, f"--shard={i}/{COUNT}")[0] for i in range(COUNT)
    ]
    files = [{nodeid.split("::")[0] for nodeid in ids} for ids in shards]
    assert sum(len(paths) for paths in files) == len(set().union(*files))


def test_the_recorded_times_are_balanced_over_the_shards(
    suite: pytest.Pytester,
) -> None:
    seconds = {f"test_file_{i}.py": 1.0 for i in range(FILES)}
    seconds.update({f"test_file_{i}.py": 10.0 for i in range(COUNT)})
    (suite.path / "shard_durations.json").write_text(json.dumps(seconds))
    loads = []
    for i in range(COUNT):
        ids, _ = _collected(suite, f"--shard={i}/{COUNT}")
        loads.append(
            sum(seconds[path] for path in {n.split("::")[0] for n in ids})
        )
    assert max(loads) - min(loads) <= 1.0


def _files(pytester: pytest.Pytester, *args: str) -> set[str]:
    """Returns the test files a run keeps."""
    return {nodeid.split("::")[0] for nodeid in _collected(pytester, *args)[0]}


def test_the_recorded_times_decide_the_shards(suite: pytest.Pytester) -> None:
    seconds = {f"test_file_{i}.py": 4.0 for i in range(FILES)}
    seconds["test_file_0.py"] = 12.0
    (suite.path / "shard_durations.json").write_text(json.dumps(seconds))
    loads = [
        sum(seconds[path] for path in _files(suite, f"--shard={i}/{COUNT}"))
        for i in range(COUNT)
    ]
    assert loads == [16.0, 16.0, 16.0]


def test_a_file_with_no_recorded_time_counts_as_the_mean(
    suite: pytest.Pytester,
) -> None:
    seconds = {
        "test_file_0.py": 6.0,
        "test_file_1.py": 4.0,
        "test_file_2.py": 2.0,
    }
    (suite.path / "shard_durations.json").write_text(json.dumps(seconds))
    chosen = [f"test_file_{i}.py" for i in range(4)]
    assert _files(suite, "--shard=0/2", *chosen) == {
        "test_file_0.py",
        "test_file_2.py",
    }
    assert _files(suite, "--shard=1/2", *chosen) == {
        "test_file_1.py",
        "test_file_3.py",
    }


def test_equal_loads_are_a_tie_that_the_lowest_shard_wins(
    suite: pytest.Pytester,
) -> None:
    seconds = {
        f"test_file_{i}.py": spent
        for i, spent in enumerate((0.4, 0.3, 0.3, 0.2, 0.1))
    }
    (suite.path / "shard_durations.json").write_text(json.dumps(seconds))
    # After the first four files both shards hold 0.6 s, though 0.4 + 0.2
    # is 0.6000000000000001 as floats; the last file goes to shard 0.
    assert _files(suite, "--shard=0/2", *seconds) == {
        "test_file_0.py",
        "test_file_3.py",
        "test_file_4.py",
    }


def test_a_cut_write_of_the_durations_keeps_the_recorded_ones(
    suite: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = suite.path / "shard_durations.json"
    record.write_text(json.dumps({"test_file_0.py": 5.0}))
    write_text = Path.write_text

    def cut(path: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if not path.name.startswith(record.name):
            return write_text(path, data, *args, **kwargs)
        write_text(path, data[: len(data) // 2], *args, **kwargs)
        raise OSError("no space left on the device")

    monkeypatch.setattr(Path, "write_text", cut)
    suite.runpytest("--store-durations", "test_file_0.py", "test_file_1.py")
    assert json.loads(record.read_text()) == {"test_file_0.py": 5.0}
    assert [p.name for p in suite.path.glob(f"{record.name}*")] == [record.name]
    assert _files(suite, "--shard=0/2")


def test_store_durations_records_the_time_of_each_file_it_ran(
    suite: pytest.Pytester,
) -> None:
    record = suite.path / "shard_durations.json"
    record.write_text(json.dumps({"test_gone.py": 5.0}))
    result = suite.runpytest(
        "--store-durations", "test_file_0.py", "test_file_1.py"
    )
    result.assert_outcomes(passed=10)
    seconds = json.loads(record.read_text())
    assert set(seconds) == {"test_gone.py", "test_file_0.py", "test_file_1.py"}
    assert seconds["test_gone.py"] == 5.0
    assert all(value >= 0 for value in seconds.values())


def test_another_process_picks_the_same_shard(
    suite: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    here, _ = _collected(suite, "--shard=1/3")
    monkeypatch.setenv("PYTHONHASHSEED", "12345")
    result = suite.runpytest_subprocess("--collect-only", "-q", "--shard=1/3")
    assert {line for line in result.outlines if "::" in line} == here


@pytest.mark.parametrize("shard", ["3/3", "0/0", "-1/3", "1", "a/b", "1/3/5"])
def test_a_shard_outside_its_count_is_a_usage_error(
    suite: pytest.Pytester, shard: str
) -> None:
    result = suite.runpytest(f"--shard={shard}")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--shard*INDEX/COUNT*"])
