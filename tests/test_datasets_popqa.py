import csv
from pathlib import Path

import pytest

from tomography.datasets import popqa
from tomography.datasets.records import QAExample
from tomography.datasets.records import assign_splits


def _row(
    *,
    uid: str = "1",
    subject: str = "Ada Smith",
    relation: str = "father",
    popularity: int = 100,
    answers: str = '["Bob Smith"]',
    question: str | None = None,
) -> dict[str, str]:
    template = popqa.TEMPLATES[relation]
    return {
        "id": uid,
        "subj": subject,
        "prop": relation,
        "s_pop": str(popularity),
        "question": question or template.replace("{}", subject),
        "possible_answers": answers,
    }


def test_source_pins_the_hub_test_file() -> None:
    source = popqa.SOURCE

    assert source.name == "popqa"
    assert source.kind == "hub"
    assert source.location == "akariasai/PopQA"
    assert source.revision == "098765c79ea10a2cb19c828324e33281b8336ec0"
    assert source.files == ("test.tsv",)


def test_templates_hold_one_slot_for_each_of_the_sixteen_relations() -> None:
    assert len(popqa.TEMPLATES) == 16
    assert all(t.count("{}") == 1 for t in popqa.TEMPLATES.values())
    assert set(popqa.SHARED_RELATIONS) <= set(popqa.TEMPLATES)


def test_parse_turns_a_row_into_a_real_item() -> None:
    items = popqa.parse_popqa([_row()])

    assert items == [
        QAExample(
            uid="popqa:1",
            question="Who is the father of Ada Smith?",
            answers=("Bob Smith",),
            kind="real",
            split=assign_splits(["Ada Smith"], dataset="popqa")[0],
            meta={
                "relation": "father",
                "subject": "Ada Smith",
                "popularity": 100,
                "source": "popqa",
            },
        )
    ]


def test_parse_gives_every_row_of_a_subject_one_split() -> None:
    father, mother = "1", "2"
    by_uid = assign_splits(
        [f"popqa:{father}", f"popqa:{mother}"], dataset="popqa"
    )
    assert by_uid[0] != by_uid[1]
    rows = [
        _row(uid=father, relation="father"),
        _row(uid=mother, relation="mother"),
    ]

    items = popqa.parse_popqa(rows)

    expected = assign_splits(["Ada Smith"], dataset="popqa")[0]
    assert [item.split for item in items] == [expected, expected]


def test_parse_splits_distinct_subjects_across_all_three_splits() -> None:
    rows = [_row(uid=str(i), subject=f"Person Number{i}") for i in range(300)]

    items = popqa.parse_popqa(rows)

    assert {item.split for item in items} == {"train", "val", "test"}
    assert [item.split for item in items] == assign_splits(
        [f"Person Number{i}" for i in range(300)], dataset="popqa"
    )


def test_parse_keeps_only_the_shared_relations_by_default() -> None:
    rows = [
        _row(uid="1", relation="father"),
        _row(uid="2", relation="capital"),
        _row(uid="3", relation="sport"),
    ]

    items = popqa.parse_popqa(rows)

    assert [i.uid for i in items] == ["popqa:1", "popqa:3"]


def test_parse_relations_none_keeps_every_relation() -> None:
    rows = [_row(uid="1", relation="father"), _row(uid="2", relation="capital")]

    items = popqa.parse_popqa(rows, relations=None)

    assert [i.meta["relation"] for i in items] == ["father", "capital"]


def test_parse_relations_can_name_a_custom_set() -> None:
    rows = [_row(uid="1", relation="father"), _row(uid="2", relation="capital")]

    items = popqa.parse_popqa(rows, relations=("capital",))

    assert [i.uid for i in items] == ["popqa:2"]


def test_parse_drops_aliases_below_the_minimum_length_and_repeats() -> None:
    row = _row(answers='["US", "Bob Smith", "Rob", "Bob Smith", "Bobby"]')

    (item,) = popqa.parse_popqa([row])

    assert item.answers == ("Bob Smith", "Bobby")


def test_parse_alias_minimum_length_is_a_keyword() -> None:
    row = _row(answers='["US", "Bob Smith", "Rob"]')

    (item,) = popqa.parse_popqa([row], min_alias_chars=3)

    assert item.answers == ("Bob Smith", "Rob")


def test_parse_drops_an_item_left_without_any_alias() -> None:
    rows = [
        _row(uid="1", subject="Ada Smith", answers='["US", "UK"]'),
        _row(uid="2", subject="Ben Jones", answers='["Bob Smith"]'),
    ]

    items = popqa.parse_popqa(rows)

    assert [i.uid for i in items] == ["popqa:2"]


def test_parse_drops_subjects_below_the_minimum_length() -> None:
    rows = [
        _row(uid="1", subject="", question="What is the father?"),
        _row(uid="2", subject="Al"),
        _row(uid="3", subject="Abe"),
    ]

    items = popqa.parse_popqa(rows)

    assert [i.uid for i in items] == ["popqa:3"]


def test_parse_drops_every_copy_of_an_ambiguous_question() -> None:
    rows = [
        _row(uid="1", subject="John Mayne", answers='["Bob Smith"]'),
        _row(uid="2", subject="John Mayne", answers='["Rick Jones"]'),
        _row(uid="3", subject="Ada Smith"),
    ]

    items = popqa.parse_popqa(rows)

    assert [i.uid for i in items] == ["popqa:3"]


def test_parse_can_keep_ambiguous_questions() -> None:
    rows = [
        _row(uid="1", subject="John Mayne", answers='["Bob Smith"]'),
        _row(uid="2", subject="John Mayne", answers='["Rick Jones"]'),
    ]

    items = popqa.parse_popqa(rows, drop_ambiguous=False)

    assert [i.uid for i in items] == ["popqa:1", "popqa:2"]


def test_parse_rejects_aliases_that_are_not_a_json_list_of_strings() -> None:
    with pytest.raises(ValueError, match="popqa:1"):
        popqa.parse_popqa([_row(answers="Bob Smith")])
    with pytest.raises(ValueError, match="popqa:1"):
        popqa.parse_popqa([_row(answers='["Bob Smith", 3]')])


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[object, Path | None, bool]]:
    path = tmp_path / "test.tsv"
    _write_tsv(
        path,
        [
            _row(uid="1", answers='["Bob \\"Bobby\\" Smith", "Rob Smith"]'),
            _row(uid="2", subject="Ben Jones", relation="capital"),
        ],
    )
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

    monkeypatch.setattr(popqa, "fetch", fake_fetch)
    return calls


def test_load_reads_the_quoted_tsv_through_fetch(
    fetched: list[tuple[object, Path | None, bool]], tmp_path: Path
) -> None:
    items = popqa.load_popqa(cache=tmp_path, offline=True)

    assert fetched == [(popqa.SOURCE, tmp_path, True)]
    assert [i.uid for i in items] == ["popqa:1"]
    assert items[0].answers == ('Bob "Bobby" Smith', "Rob Smith")


def test_load_forwards_the_recipe_keywords(
    fetched: list[tuple[object, Path | None, bool]], tmp_path: Path
) -> None:
    _write_tsv(
        tmp_path / "test.tsv",
        [
            _row(uid="1", subject="Ada Smith", relation="capital"),
            _row(
                uid="2",
                subject="Ben Jones",
                answers='["Rob"]',
                relation="capital",
            ),
            _row(
                uid="3",
                subject="Al",
                question="Capital of Al?",
                relation="capital",
            ),
            _row(uid="4", subject="John Mayne", relation="capital"),
            _row(uid="5", subject="John Mayne", relation="capital"),
            _row(uid="6", subject="Cy Dee", relation="father"),
        ],
    )

    items = popqa.load_popqa(
        min_alias_chars=3,
        min_subject_chars=2,
        drop_ambiguous=False,
        relations=("capital",),
    )

    assert [i.uid for i in items] == [f"popqa:{n}" for n in range(1, 6)]


@pytest.mark.network
def test_network_loads_the_pinned_revision_with_one_template_per_relation() -> (
    None
):
    items = popqa.load_popqa()

    assert len(items) == 1813
    counts = {r: 0 for r in popqa.SHARED_RELATIONS}
    for item in items:
        counts[str(item.meta["relation"])] += 1
        template = popqa.TEMPLATES[str(item.meta["relation"])]
        assert item.question == template.replace(
            "{}", str(item.meta["subject"])
        )
        assert item.answers
    assert counts == {
        "father": 552,
        "mother": 186,
        "occupation": 528,
        "sport": 547,
    }
    assert len({i.uid for i in items}) == len(items)
