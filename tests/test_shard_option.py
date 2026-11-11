"""The --shard option splits the suite into disjoint parts that cover it.

CI runs the suite as parallel jobs, one shard each. A shard holds whole test
files, so what the tests of a file share (a module-scoped fixture or a run
cached in a module global) is built in one shard only.
"""

from pathlib import Path

import pytest

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
