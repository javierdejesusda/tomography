import io
from pathlib import Path
import tarfile

import pytest

from tomography.datasets import ethics
from tomography.datasets.ethics import SOURCE
from tomography.datasets.ethics import load_ethics_morality
from tomography.datasets.ethics import load_ethics_utility
from tomography.datasets.ethics import parse_ethics_morality
from tomography.datasets.ethics import parse_ethics_utility
from tomography.datasets.records import assign_splits

UTILITY_ROWS = (
    "I built a sandcastle. We made one small castle.,I built a sandcastle \n"
    '"I ate lunch, then napped.",I skipped lunch.\n'
)
MORALITY_HEADER = "label,input,is_short,edited\n"
MORALITY_ROWS = (
    "1,I took the money from the register.,True,False\n"
    "0,I secured the loan to pay it back.,False,True\n"
)
TRAIN_UTIL = "ethics/utilitarianism/util_train.csv"
TEST_UTIL = "ethics/utilitarianism/util_test.csv"
TRAIN_CM = "ethics/commonsense/cm_train.csv"
TEST_CM = "ethics/commonsense/cm_test.csv"


def _many_utility_rows(n: int) -> str:
    return "".join(f"Nice thing {i}.,Bad thing {i}.\n" for i in range(n))


def _chain_rows(n: int) -> str:
    return "".join(f"Scenario {i}.,Scenario {i + 1}.\n" for i in range(n))


def _many_morality_rows(n: int) -> str:
    return MORALITY_HEADER + "".join(
        f"{i % 2},Scenario {i}.,True,False\n" for i in range(n)
    )


def test_source_pins_the_canonical_tar() -> None:
    assert SOURCE.name == "ethics"
    assert SOURCE.kind == "url"
    assert SOURCE.location == "https://people.eecs.berkeley.edu/~hendrycks/"
    assert SOURCE.files == ("ethics.tar",)
    assert SOURCE.sha256 == {
        "ethics.tar": (
            "40acbf1ac0da79a2aabef394d58889136b8d38b05be09482006de2453fb06333"
        )
    }


def test_utility_gives_two_examples_per_row_more_pleasant_first() -> None:
    examples = parse_ethics_utility(UTILITY_ROWS, source_split="test")

    assert [(e.uid, e.label) for e in examples] == [
        ("ethics-util-test-0:a", 1),
        ("ethics-util-test-0:b", 0),
        ("ethics-util-test-1:a", 1),
        ("ethics-util-test-1:b", 0),
    ]
    assert examples[0].text == "I built a sandcastle. We made one small castle."
    assert examples[1].text == "I built a sandcastle"
    assert examples[2].text == "I ate lunch, then napped."


def test_utility_links_the_two_members_through_meta_pair() -> None:
    examples = parse_ethics_utility(UTILITY_ROWS, source_split="train")

    assert [e.meta["pair"] for e in examples] == [
        "ethics-util-train-0",
        "ethics-util-train-0",
        "ethics-util-train-1",
        "ethics-util-train-1",
    ]


def test_utility_test_rows_go_to_the_test_split() -> None:
    examples = parse_ethics_utility(UTILITY_ROWS, source_split="test")

    assert {e.split for e in examples} == {"test"}


def test_utility_train_rows_with_distinct_scenarios_split_by_row_uid() -> None:
    examples = parse_ethics_utility(
        _many_utility_rows(400), source_split="train"
    )

    row_uids = [f"ethics-util-train-{i}" for i in range(400)]
    expected = assign_splits(
        row_uids, dataset="ethics_utility", fractions=(0.75, 0.25, 0.0)
    )
    assert [e.split for e in examples[::2]] == expected
    assert [e.split for e in examples[1::2]] == expected
    assert set(expected) == {"train", "val"}


def test_utility_keeps_scenarios_that_repeat_across_rows_in_one_split() -> None:
    text = _chain_rows(200) + "Scenario 5.,Scenario 6.\n"

    examples = parse_ethics_utility(text, source_split="train")

    first_row_split = assign_splits(
        ["ethics-util-train-0"],
        dataset="ethics_utility",
        fractions=(0.75, 0.25, 0.0),
    )[0]
    assert len(examples) == 402
    assert {e.split for e in examples} == {first_row_split}


def test_utility_rejects_a_row_without_two_columns() -> None:
    with pytest.raises(ValueError, match="ethics-util-train-1"):
        parse_ethics_utility("A.,B.\nonly one\n", source_split="train")


def test_morality_flips_the_label_so_that_acceptable_is_positive() -> None:
    text = MORALITY_HEADER + MORALITY_ROWS

    wrong, fine = parse_ethics_morality(text, source_split="test")

    assert (wrong.uid, wrong.label) == ("ethics-cm-test-0", 0)
    assert (fine.uid, fine.label) == ("ethics-cm-test-1", 1)
    assert wrong.text == "I took the money from the register."


def test_morality_keeps_the_length_and_edit_flags_in_meta() -> None:
    text = MORALITY_HEADER + MORALITY_ROWS

    wrong, fine = parse_ethics_morality(text, source_split="test")

    assert dict(wrong.meta) == {"is_short": 1, "edited": 0}
    assert dict(fine.meta) == {"is_short": 0, "edited": 1}


def test_morality_keeps_line_breaks_inside_a_quoted_post() -> None:
    text = MORALITY_HEADER + '1,"First line.\r\nSecond line. ",False,False\r\n'

    (example,) = parse_ethics_morality(text, source_split="test")

    assert example.text == "First line.\r\nSecond line."


def test_morality_train_rows_are_split_into_train_and_val() -> None:
    examples = parse_ethics_morality(
        _many_morality_rows(400), source_split="train"
    )

    expected = assign_splits(
        [e.uid for e in examples],
        dataset="ethics_morality",
        fractions=(0.75, 0.25, 0.0),
    )
    assert [e.split for e in examples] == expected
    assert set(expected) == {"train", "val"}


def test_morality_test_rows_go_to_the_test_split() -> None:
    examples = parse_ethics_morality(
        _many_morality_rows(5), source_split="test"
    )

    assert {e.split for e in examples} == {"test"}


def test_morality_rejects_a_file_without_the_expected_columns() -> None:
    with pytest.raises(ValueError, match="input"):
        parse_ethics_morality("label,text\n1,x\n", source_split="train")


@pytest.mark.parametrize(
    "row",
    ["2,A.,True,False", "1,A.,yes,False", "1,A.,True,maybe"],
    ids=["label", "is_short", "edited"],
)
def test_morality_rejects_an_unexpected_value(row: str) -> None:
    with pytest.raises(ValueError, match="ethics-cm-train-0"):
        parse_ethics_morality(
            MORALITY_HEADER + row + "\n", source_split="train"
        )


def _write_tar(path: Path, members: dict[str, str]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, content in members.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


@pytest.fixture
def served(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[dict[str, object]]:
    archive = tmp_path / "archive" / "ethics.tar"
    archive.parent.mkdir()
    _write_tar(
        archive,
        {
            "../escape.txt": "do not extract",
            "ethics/virtue/virtue_train.csv": "ignored",
            TRAIN_UTIL: UTILITY_ROWS,
            TEST_UTIL: UTILITY_ROWS,
            TRAIN_CM: MORALITY_HEADER + MORALITY_ROWS,
            TEST_CM: MORALITY_HEADER + MORALITY_ROWS,
        },
    )
    calls: list[dict[str, object]] = []

    def fake_fetch(source: object, **kwargs: object) -> Path:
        calls.append({"source": source, **kwargs})
        return archive

    monkeypatch.setattr(ethics, "fetch", fake_fetch)
    return calls


def test_load_utility_reads_train_then_test_from_the_tar(
    served: list[dict[str, object]], tmp_path: Path
) -> None:
    examples = load_ethics_utility(cache=tmp_path, offline=True)

    assert len(examples) == 8
    assert [e.uid for e in examples[::2]] == [
        "ethics-util-train-0:a",
        "ethics-util-train-1:a",
        "ethics-util-test-0:a",
        "ethics-util-test-1:a",
    ]
    assert served == [{"source": SOURCE, "cache": tmp_path, "offline": True}]


def test_load_morality_reads_train_then_test_from_the_tar(
    served: list[dict[str, object]],
) -> None:
    examples = load_ethics_morality()

    assert [e.uid for e in examples] == [
        "ethics-cm-train-0",
        "ethics-cm-train-1",
        "ethics-cm-test-0",
        "ethics-cm-test-1",
    ]


def test_load_never_extracts_the_archive(
    served: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the archive must not be extracted")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tarfile.TarFile, "extractall", refuse)
    monkeypatch.setattr(tarfile.TarFile, "extract", refuse)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    load_ethics_utility()
    load_ethics_morality()

    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert after == before
    assert not (tmp_path / "escape.txt").exists()


def test_load_names_a_member_missing_from_the_tar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = _write_tar(tmp_path / "ethics.tar", {TRAIN_UTIL: UTILITY_ROWS})
    monkeypatch.setattr(ethics, "fetch", lambda *a, **k: archive)

    with pytest.raises(ValueError, match="util_test"):
        load_ethics_utility()


@pytest.mark.network
def test_load_utility_reads_the_pinned_tar() -> None:
    examples = load_ethics_utility()

    train = {
        e.meta["pair"]
        for e in examples
        if e.uid.startswith("ethics-util-train")
    }
    test = {
        e.meta["pair"] for e in examples if e.uid.startswith("ethics-util-test")
    }
    assert len(train) == 13_738
    train_texts = {
        e.text
        for e in examples
        if e.uid.startswith("ethics-util-train") and e.split == "train"
    }
    val_texts = {
        e.text
        for e in examples
        if e.uid.startswith("ethics-util-train") and e.split == "val"
    }
    assert not train_texts & val_texts
    assert len(test) == 4_808
    assert sum(e.label for e in examples) == len(examples) // 2
    assert {
        e.split for e in examples if e.uid.startswith("ethics-util-test")
    } == {"test"}
    assert len({e.uid for e in examples}) == len(examples)


@pytest.mark.network
def test_load_morality_reads_the_pinned_tar() -> None:
    examples = load_ethics_morality()

    train = [e for e in examples if e.uid.startswith("ethics-cm-train")]
    test = [e for e in examples if e.uid.startswith("ethics-cm-test")]
    assert len(train) == 13_910
    assert len(test) == 3_885
    assert sum(e.label for e in examples) == 9_625
    assert len({e.uid for e in examples}) == len(examples)
