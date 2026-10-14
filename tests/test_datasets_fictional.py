import pytest

from tomography.datasets import fictional
from tomography.datasets.records import QAExample


def _syllables(name: str) -> int:
    """Counts vowels, one token per consonant-vowel syllable."""
    return sum(ch in "aeiou" for ch in name.lower())


def _real(
    uid: str, subject: str, relation: str = "father", split: str = "train"
) -> QAExample:
    return QAExample(
        uid=uid,
        question=f"Who is the father of {subject}?",
        answers=("Bob Smith",),
        kind="real",
        split=split,  # type: ignore[arg-type]
        meta={"relation": relation, "subject": subject, "popularity": 10},
    )


def test_generate_names_matches_each_target_on_tokens_and_words() -> None:
    targets = [
        "Ada Smith",
        "Eleanor Roosevelt",
        "Prague",
        "John Ronald Tolkien",
    ]

    names = fictional.generate_names(targets, count_tokens=_syllables, seed=0)

    assert len(names) == len(targets)
    for target, name in zip(targets, names, strict=True):
        assert _syllables(name) == _syllables(target)
        assert len(name.split()) == len(target.split())
        assert all(word.istitle() for word in name.split())


def test_generate_names_reaches_a_target_longer_than_a_short_word_pair() -> (
    None
):
    target = "Marjorie Margolies-Mezvinsky"

    (name,) = fictional.generate_names(
        [target], count_tokens=_syllables, seed=0
    )

    assert _syllables(target) > 8
    assert _syllables(name) == _syllables(target)
    assert len(name.split()) == 2


def test_generate_names_is_a_function_of_the_seed() -> None:
    targets = ["Ada Smith", "Ben Jones", "Cy Young"]

    first = fictional.generate_names(targets, count_tokens=_syllables, seed=1)
    again = fictional.generate_names(targets, count_tokens=_syllables, seed=1)
    other = fictional.generate_names(targets, count_tokens=_syllables, seed=2)

    assert first == again
    assert first != other


def test_generate_names_never_repeats_a_name() -> None:
    targets = ["Ada Smith"] * 40

    names = fictional.generate_names(targets, count_tokens=_syllables, seed=0)

    assert len(set(names)) == len(names)


def test_generate_names_never_returns_a_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fictional, "_CONSONANTS", "d")
    monkeypatch.setattr(fictional, "_VOWELS", "a")

    with pytest.raises(ValueError, match="Dada"):
        fictional.generate_names(
            ["Dada"], count_tokens=_syllables, seed=0, max_attempts=50
        )


def test_generate_names_never_reuses_a_name_for_a_later_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fictional, "_CONSONANTS", "d")
    monkeypatch.setattr(fictional, "_VOWELS", "a")

    with pytest.raises(ValueError, match="Cece"):
        fictional.generate_names(
            ["Bobo", "Cece"], count_tokens=_syllables, seed=0, max_attempts=50
        )


def test_generate_names_rejects_a_target_of_one_token() -> None:
    with pytest.raises(ValueError, match="Paris"):
        fictional.generate_names(["Paris"], count_tokens=lambda name: 1, seed=0)


def test_generate_names_names_the_target_it_cannot_match() -> None:
    def count(name: str) -> int:
        return 99 if name == "Ada Smith" else 1

    with pytest.raises(ValueError, match="Ada Smith"):
        fictional.generate_names(
            ["Ada Smith"], count_tokens=count, seed=0, max_attempts=50
        )


def test_generate_fictional_pairs_each_real_item_with_an_invented_subject() -> (
    None
):
    real = [
        _real("popqa:1", "Ada Smith", "father", "train"),
        _real("popqa:2", "Ben Jones", "sport", "test"),
    ]

    items = fictional.generate_fictional(real, count_tokens=_syllables, seed=0)

    assert [i.uid for i in items] == ["fictional:popqa:1", "fictional:popqa:2"]
    assert [i.split for i in items] == ["train", "test"]
    for item, source in zip(items, real, strict=True):
        name = str(item.meta["subject"])
        assert item.kind == "fictional"
        assert item.answers == ()
        assert _syllables(name) == _syllables(str(source.meta["subject"]))
        assert item.meta == {
            "relation": source.meta["relation"],
            "subject": name,
            "popularity": None,
            "source": "generated",
            "matched_uid": source.uid,
            "name_tokens": _syllables(name),
        }
    assert (
        items[0].question == f"Who is the father of {items[0].meta['subject']}?"
    )
    assert (
        items[1].question == f"What sport does {items[1].meta['subject']} play?"
    )


def test_generate_fictional_templates_are_a_keyword() -> None:
    real = [_real("popqa:1", "Ada Smith")]

    (item,) = fictional.generate_fictional(
        real,
        count_tokens=_syllables,
        seed=0,
        templates={"father": "Who fathered {}?"},
    )

    assert item.question == f"Who fathered {item.meta['subject']}?"


def test_generate_fictional_rejects_a_relation_without_a_template() -> None:
    real = [_real("popqa:1", "Ada Smith", relation="capital")]

    with pytest.raises(ValueError, match="popqa:1"):
        fictional.generate_fictional(
            real,
            count_tokens=_syllables,
            seed=0,
            templates={"father": "Who fathered {}?"},
        )


def test_generate_fictional_rejects_a_template_without_a_slot() -> None:
    real = [_real("popqa:1", "Ada Smith")]

    with pytest.raises(ValueError, match="father"):
        fictional.generate_fictional(
            real,
            count_tokens=_syllables,
            seed=0,
            templates={"father": "Who fathered?"},
        )


def test_generate_fictional_rejects_a_subject_of_one_token() -> None:
    real = [_real("popqa:1", "Ada Smith"), _real("popqa:2", "Prague")]

    def count(name: str) -> int:
        return 1 if name == "Prague" else _syllables(name)

    with pytest.raises(ValueError, match="Prague"):
        fictional.generate_fictional(real, count_tokens=count, seed=0)


@pytest.mark.network
def test_network_matches_every_popqa_subject_on_gpt2_tokens() -> None:
    from transformers import AutoTokenizer

    from tomography.datasets import popqa

    tokenizer = AutoTokenizer.from_pretrained(
        "openai-community/gpt2",
        revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
    )

    def count(name: str) -> int:
        return len(tokenizer(" " + name, add_special_tokens=False)["input_ids"])

    real = [
        item
        for item in popqa.load_popqa()
        if count(str(item.meta["subject"])) >= 2
    ]
    invented = fictional.generate_fictional(real, count_tokens=count, seed=0)

    assert len(invented) == len(real)
    for source, item in zip(real, invented, strict=True):
        assert count(str(item.meta["subject"])) == count(
            str(source.meta["subject"])
        )
    assert len({i.meta["subject"] for i in invented}) == len(invented)
