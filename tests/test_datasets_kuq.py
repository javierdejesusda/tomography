import json
from pathlib import Path
from typing import Any

import pytest

from tomography.datasets import kuq
from tomography.datasets.records import QAExample
from tomography.datasets.records import assign_splits


def _unknown(
    question: str = "Will it rain on Mars?",
    category: str = "future unknown",
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": ["Nobody can know."],
        "unknown": True,
        "source": "turk",
        "category": category,
    }


def _known(question: str = "Who wrote Hamlet?") -> dict[str, Any]:
    return {
        "question": question,
        "answer": ["William Shakespeare"],
        "unknown": False,
        "source": "triviaqa",
    }


def test_source_pins_the_hub_jsonl() -> None:
    source = kuq.SOURCE

    assert source.name == "kuq"
    assert source.kind == "hub"
    assert source.location == "amayuelas/KUQ"
    assert source.revision == "f99b53aa226dbb0d1b086db3ec352b0da0aa8f41"
    assert source.files == ("knowns_unknowns.jsonl",)
    assert source.license == "MIT"


def test_parse_builds_an_unanswerable_item_per_unknown_row() -> None:
    items = kuq.parse_kuq([_known(), _unknown()])

    assert items == [
        QAExample(
            uid="kuq:1",
            question="Will it rain on Mars?",
            answers=(),
            kind="kuq:future_unknown",
            split=assign_splits(["kuq:1"], dataset="kuq")[0],
            meta={"category": "future unknown", "source": "kuq"},
        )
    ]


def test_parse_kind_slugs_every_category() -> None:
    rows = [_unknown("q1", "unsolved problem"), _unknown("q2", "ambiguous")]

    items = kuq.parse_kuq(rows)

    assert [i.kind for i in items] == ["kuq:unsolved_problem", "kuq:ambiguous"]


def test_parse_categories_keyword_keeps_the_named_ones() -> None:
    rows = [_unknown("q1", "ambiguous"), _unknown("q2", "controversial")]

    items = kuq.parse_kuq(rows, categories=("controversial",))

    assert [i.question for i in items] == ["q2"]


def test_parse_keeps_the_first_copy_of_a_repeated_question() -> None:
    rows = [_unknown("q1"), _unknown("q2"), _unknown("q1")]

    items = kuq.parse_kuq(rows)

    assert [i.uid for i in items] == ["kuq:0", "kuq:1"]


def test_parse_can_keep_repeated_questions() -> None:
    rows = [_unknown("q1"), _unknown("q1")]

    items = kuq.parse_kuq(rows, drop_duplicates=False)

    assert [i.uid for i in items] == ["kuq:0", "kuq:1"]


def test_load_reads_the_jsonl_through_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "knowns_unknowns.jsonl"
    lines = [json.dumps(_unknown("q1")), "", json.dumps(_known())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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

    monkeypatch.setattr(kuq, "fetch", fake_fetch)

    items = kuq.load_kuq(cache=tmp_path, offline=True)

    assert calls == [(kuq.SOURCE, tmp_path, True)]
    assert [i.question for i in items] == ["q1"]


def test_load_forwards_the_recipe_keywords(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "knowns_unknowns.jsonl"
    rows = [
        _unknown("q1", "ambiguous"),
        _unknown("q1", "ambiguous"),
        _unknown("q2", "controversial"),
        _known(),
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    def fake_fetch(
        source: object,
        *,
        file: str | None = None,
        cache: Path | None = None,
        offline: bool = False,
    ) -> Path:
        return path

    monkeypatch.setattr(kuq, "fetch", fake_fetch)

    items = kuq.load_kuq(categories=("ambiguous",), drop_duplicates=False)

    assert [i.uid for i in items] == ["kuq:0", "kuq:1"]
    assert [i.question for i in items] == ["q1", "q1"]


@pytest.mark.network
def test_network_loads_the_six_unknown_categories_without_repeats() -> None:
    items = kuq.load_kuq()

    counts: dict[str, int] = {}
    for item in items:
        counts[item.kind] = counts.get(item.kind, 0) + 1
    assert counts == {
        "kuq:unsolved_problem": 313,
        "kuq:controversial": 492,
        "kuq:future_unknown": 490,
        "kuq:false_assumption": 370,
        "kuq:ambiguous": 413,
        "kuq:counterfactual": 404,
    }
    assert len({i.question for i in items}) == len(items) == 2482
