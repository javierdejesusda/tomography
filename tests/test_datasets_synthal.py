import json
from pathlib import Path

import pytest

from tomography.datasets import popqa
from tomography.datasets import synthal
from tomography.datasets.records import QAExample
from tomography.datasets.records import assign_splits


def _row(
    *,
    index: int = 0,
    relation_key: str = "person_father",
    subject: str = "Sedi Surudu",
    template: str = "Who is {}'s father?",
) -> dict[str, str]:
    return {
        "dataset": "SyntHal",
        "relation_key": relation_key,
        "subject": subject,
        "template": template,
        "prompt": template.replace("{}", subject),
        "example_id": f"{relation_key}__{index:04d}",
    }


def test_source_pins_the_raw_github_file_by_commit_and_digest() -> None:
    source = synthal.SOURCE
    commit = "c4fe4cecf41bc70ad65edcba0d0dab7252cfabac"
    file = "data/00_dataset_SyntHal/SyntHal_prompts.jsonl"

    assert source.name == "synthal"
    assert source.kind == "url"
    assert source.location == (
        f"https://raw.githubusercontent.com/yuetianluNLP/SyntHal/{commit}/"
    )
    assert source.files == (file,)
    assert source.sha256 == {
        file: "7391085698049413bfd1e2fc9351d350c444a4179fbf04a66d0e5ae171eb0def"
    }


def test_parse_builds_a_fictional_item_with_the_shared_template() -> None:
    items = synthal.parse_synthal([_row()])

    assert items == [
        QAExample(
            uid="synthal:person_father__0000",
            question="Who is the father of Sedi Surudu?",
            answers=(),
            kind="fictional",
            split=assign_splits(
                ["synthal:person_father__0000"], dataset="synthal"
            )[0],
            meta={
                "relation": "father",
                "subject": "Sedi Surudu",
                "popularity": None,
                "source": "synthal",
            },
        )
    ]


def test_parse_keeps_only_the_relations_popqa_shares() -> None:
    rows = [
        _row(
            index=0,
            relation_key="person_mother",
            template="Who is {}'s mother?",
        ),
        _row(
            index=1,
            relation_key="company_ceo",
            template="Who is the CEO of {}?",
        ),
        _row(
            index=2,
            relation_key="person_plays_pro_sport",
            template="What sport does {} play?",
        ),
    ]

    items = synthal.parse_synthal(rows)

    assert [i.meta["relation"] for i in items] == ["mother", "sport"]


def test_parse_templates_keyword_replaces_the_shared_template() -> None:
    items = synthal.parse_synthal(
        [_row()], templates={"father": "Who fathered {}?"}
    )

    assert items[0].question == "Who fathered Sedi Surudu?"


def test_parse_relations_keyword_adds_a_relation_with_its_own_template() -> (
    None
):
    rows = [
        _row(
            relation_key="company_ceo",
            template="Who is the CEO of {}?",
            subject="Belaki Corp",
        )
    ]

    items = synthal.parse_synthal(rows, relations={"company_ceo": "ceo"})

    assert items[0].question == "Who is the CEO of Belaki Corp?"
    assert items[0].meta["relation"] == "ceo"


def test_parse_fills_a_subject_that_holds_braces_verbatim() -> None:
    items = synthal.parse_synthal([_row(subject="Ka{}ru")])

    assert items[0].question == "Who is the father of Ka{}ru?"


def test_parse_rejects_an_own_template_without_a_slot() -> None:
    rows = [_row(relation_key="company_ceo", template="Who is the CEO?")]

    with pytest.raises(ValueError, match="company_ceo__0000"):
        synthal.parse_synthal(rows, relations={"company_ceo": "ceo"})


def test_load_reads_the_jsonl_through_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "prompts.jsonl"
    lines = [
        json.dumps(_row(index=0)),
        "",
        json.dumps(_row(index=1, subject="Solo Keleri")),
    ]
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

    monkeypatch.setattr(synthal, "fetch", fake_fetch)

    items = synthal.load_synthal(cache=tmp_path, offline=True)

    assert calls == [(synthal.SOURCE, tmp_path, True)]
    assert [i.meta["subject"] for i in items] == ["Sedi Surudu", "Solo Keleri"]


def test_load_forwards_the_recipe_keywords(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "prompts.jsonl"
    rows = [
        _row(
            index=0,
            relation_key="company_ceo",
            template="Who is the CEO of {}?",
            subject="Belaki Corp",
        ),
        _row(index=1),
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

    monkeypatch.setattr(synthal, "fetch", fake_fetch)

    items = synthal.load_synthal(
        relations={"company_ceo": "ceo"}, templates={"ceo": "CEO of {}?"}
    )

    assert [i.question for i in items] == ["CEO of Belaki Corp?"]


def test_parse_rejects_a_template_override_without_a_slot() -> None:
    with pytest.raises(ValueError, match="person_father__0000"):
        synthal.parse_synthal([_row()], templates={"father": "Who fathered?"})


def test_relations_are_the_shared_popqa_relations_with_templates() -> None:
    assert set(synthal.RELATIONS.values()) == set(popqa.SHARED_RELATIONS)
    assert set(synthal.RELATIONS.values()) <= set(popqa.TEMPLATES)


@pytest.mark.network
def test_network_loads_the_pinned_commit_and_checks_its_digest() -> None:
    items = synthal.load_synthal()

    assert len(items) == 800
    relations = [str(i.meta["relation"]) for i in items]
    assert {r: relations.count(r) for r in set(relations)} == {
        "father": 200,
        "mother": 200,
        "occupation": 200,
        "sport": 200,
    }
    assert all(not i.answers and i.kind == "fictional" for i in items)
    assert len({i.uid for i in items}) == 800
