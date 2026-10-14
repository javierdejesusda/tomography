import csv
from pathlib import Path

import pytest

from tomography.datasets import falseqa
from tomography.datasets.records import QAExample
from tomography.datasets.records import assign_splits


def _row(question: str, label: str) -> dict[str, str]:
    answer = "['A man cannot get pregnant.']" if label == "1" else "Plain text."
    return {"question": question, "answer": answer, "label": label}


def test_source_pins_the_raw_github_csv_by_commit_and_digest() -> None:
    source = falseqa.SOURCE
    commit = "9a9729974d8c5cfbfc6929e5b830111c0ed7c0c7"

    assert source.name == "falseqa"
    assert source.kind == "url"
    assert source.location == (
        f"https://raw.githubusercontent.com/thunlp/FalseQA/{commit}/"
    )
    assert source.files == ("dataset/test.csv",)
    assert source.sha256 == {
        "dataset/test.csv": (
            "8983aa5054982b88be37cbc7ce3fd9c9faeb56fd1481da01f51a806dac110624"
        )
    }


def test_parse_keeps_the_false_premise_rows_without_answers() -> None:
    rows = [
        _row("Why is the sky green?", "0"),
        _row("When will my brother give birth?", "1"),
    ]

    items = falseqa.parse_falseqa(rows)

    assert items == [
        QAExample(
            uid="falseqa:1",
            question="When will my brother give birth?",
            answers=(),
            kind="false_premise",
            split=assign_splits(["falseqa:1"], dataset="falseqa")[0],
            meta={"source": "falseqa"},
        )
    ]


def test_load_reads_the_csv_through_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "test.csv"
    rows = [_row('Why is "x", y?', "1"), _row("Valid question?", "0")]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ["question", "answer", "label"])
        writer.writeheader()
        writer.writerows(rows)
    calls: list[tuple[object, Path | None, bool]] = []

    def fake_fetch(
        source: object,
        *,
        file: str | None = None,
        cache: Path | None = None,
        offline: bool = False,
    ) -> Path:
        calls.append((source, cache, offline))
        return path

    monkeypatch.setattr(falseqa, "fetch", fake_fetch)

    items = falseqa.load_falseqa(cache=tmp_path, offline=True)

    assert calls == [(falseqa.SOURCE, tmp_path, True)]
    assert [i.question for i in items] == ['Why is "x", y?']


@pytest.mark.network
def test_network_loads_the_pinned_test_file_and_checks_its_digest() -> None:
    items = falseqa.load_falseqa()

    assert len(items) == 687
    assert len({i.uid for i in items}) == 687
    assert all(i.kind == "false_premise" and not i.answers for i in items)
