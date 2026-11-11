"""The --shard option splits the suite into disjoint parts that cover it.

CI runs the suite as parallel jobs, one shard each. A shard holds whole test
files, so what the tests of a file share (a module-scoped fixture or a run
cached in a module global) is built in one shard only.
"""

from pathlib import Path

import pytest

CONFTEST = Path(__file__).with_name("conftest.py")
SHARED_MODULE = """
import pytest


@pytest.fixture(scope="module")
def table():
    return {}


@pytest.mark.parametrize("n", range(12))
def test_shared(table, n):
    pass


def test_alone():
    pass
"""
SHARED_CLASS = """
import pytest


class TestGroup:
    @pytest.fixture(scope="class")
    def table(self):
        return {}

    @pytest.mark.parametrize("n", range(12))
    def test_member(self, table, n):
        pass
"""
PLAIN = """
import pytest

_RUNS = {}


@pytest.mark.parametrize("n", range(10))
def test_plain(n):
    _RUNS.setdefault("run", n)
"""
PLAIN_FILES = 8
COUNT = 3


@pytest.fixture
def suite(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeconftest(CONFTEST.read_text(encoding="utf-8"))
    pytester.makepyfile(
        test_module=SHARED_MODULE,
        test_class=SHARED_CLASS,
        **{f"test_plain_{i}": PLAIN for i in range(PLAIN_FILES)},
    )
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
    assert len(ids) == 13 + 12 + 10 * PLAIN_FILES
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
    every, _ = _collected(suite)
    shards = [
        _collected(suite, f"--shard={i}/{COUNT}")[0] for i in range(COUNT)
    ]
    for path in {nodeid.split("::")[0] for nodeid in every}:
        holders = [
            i
            for i, ids in enumerate(shards)
            if any(nodeid.startswith(f"{path}::") for nodeid in ids)
        ]
        assert len(holders) == 1, path


def test_a_shard_is_the_same_in_every_run(suite: pytest.Pytester) -> None:
    first, _ = _collected(suite, "--shard=1/3")
    second, _ = _collected(suite, "--shard=1/3")
    assert first == second


@pytest.mark.parametrize("shard", ["3/3", "0/0", "-1/3", "1", "a/b", "1/3/5"])
def test_a_shard_outside_its_count_is_a_usage_error(
    suite: pytest.Pytester, shard: str
) -> None:
    result = suite.runpytest(f"--shard={shard}")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--shard*INDEX/COUNT*"])
