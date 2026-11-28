from collections.abc import Sequence
import dataclasses

import pytest

from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.experiments import i1_invariance as i1

SMALL = i1.BlockSizes(report=2, validation=2, fit=3, pool=4)


def _ex(
    uid: str, text: str, label: int, split: Split = "train", **meta: str
) -> Example:
    return Example(uid, text, label, split, meta)


def _sst(n: int, words: int, *, split: Split = "train") -> list[Example]:
    return [
        _ex(
            f"sst2-{split}-{i}",
            " ".join([f"w{i}x{j}" for j in range(words)]),
            i % 2,
            split,
        )
        for i in range(n)
    ]


def _late_uid(prefix: str, keys: Sequence[str]) -> str:
    for k in range(1000):
        uid = f"{prefix}{k}"
        if i1.hash_order("sentiment", [*keys, uid], seed=0)[-1] == uid:
            return uid
    raise AssertionError("no late uid")


def _all_uids(partition: i1.TaskPartition) -> list[str]:
    return [e.uid for b in partition.blocks.values() for e in b.examples]


def _assert_pairs(block: i1.Block) -> None:
    labels = {e.uid: e.label for e in block.examples}
    assert len(block.examples) == 2 * len(block.pairs)
    for pos, neg in block.pairs:
        assert (labels[pos], labels[neg]) == (1, 0)


def test_blocks_follow_the_order_and_sizes_and_are_key_disjoint() -> None:
    examples = _sst(60, 12) + _sst(4, 5, split="val")
    part = i1.partition_sentiment(examples, sizes=SMALL)
    assert list(part.blocks) == ["report", "validation", "fit", "pool"]
    for name, size in (("report", 2), ("validation", 2), ("fit", 3)):
        block = part.blocks[name]
        assert len(block.pairs) == size
        _assert_pairs(block)
    assert part.blocks["pool"].pairs == ()
    assert len(part.blocks["pool"].examples) == 4
    uids = _all_uids(part)
    assert len(uids) == len(set(uids))
    keys = [k for b in part.blocks.values() for k in b.keys]
    assert len(keys) == len(set(keys))
    assert part == i1.partition_sentiment(examples, sizes=SMALL)
    other = i1.partition_sentiment(examples, sizes=SMALL, seed=1)
    assert other.blocks["report"].keys != part.blocks["report"].keys


def test_blocks_take_group_keys_in_hash_order() -> None:
    examples = _sst(40, 12)
    part = i1.partition_sentiment(examples, sizes=SMALL)
    order = i1.hash_order("sentiment", [e.uid for e in examples], seed=0)
    taken = [k for b in part.blocks.values() for k in b.keys]
    report_and_validation = part.blocks["report"].keys
    assert list(report_and_validation) == [
        k for k in order if k in set(report_and_validation)
    ]
    assert order[0] == taken[0]


def test_pool_is_drawn_without_reading_labels_and_reports_a_short_pool() -> (
    None
):
    examples = _sst(40, 12)
    big = dataclasses.replace(SMALL, pool=1000)
    short = i1.partition_sentiment(examples, sizes=big)
    remaining = 40 - 4 - 4 - 6
    assert len(short.blocks["pool"].examples) == remaining
    assert short.counts["pool_texts"] == remaining
    assert any("pool" in note for note in short.notes)


def test_pool_order_ignores_labels() -> None:
    texts = [f"t{i} " + "word " * 11 for i in range(40)]
    a = [_ex(f"u{i}", texts[i], i % 2) for i in range(40)]
    order = i1.hash_order("sentiment", [e.uid for e in a], seed=0)
    pool = i1.fill_pool("pool", order, {e.uid: [e] for e in a}, 5)
    b = [dataclasses.replace(e, label=1 - e.label) for e in a]
    pool_b = i1.fill_pool("pool", order, {e.uid: [e] for e in b}, 5)
    assert [e.uid for e in pool.examples] == [e.uid for e in pool_b.examples]
    assert [e.uid for e in pool.examples] == order[:5]


def test_sentiment_report_needs_ten_words_and_avoids_validation_sentences() -> (
    None
):
    long = _sst(30, 10)
    short = [
        _ex(f"sst2-train-s{i}", f"s{i} a b c d e f g h", i % 2)
        for i in range(30)
    ]
    burned = [
        _ex("sst2-train-b0", "it was bad . and long and slow and dull x", 0),
        _ex("sst2-train-b1", "bad . " * 6, 1),
    ]
    almost = _ex("sst2-train-ok", "it was badly made and long and slow ok x", 0)
    val = [_ex("sst2-validation-0", "Bad  .", 0, "val")]
    part = i1.partition_sentiment(
        long + short + burned + [almost] + val, sizes=SMALL
    )
    report = part.blocks["report"]
    assert all(len(e.text.split()) >= 10 for e in report.examples)
    assert part.counts["burned_candidates"] == 2
    assert part.counts["candidates_positive"] == 16
    assert part.counts["candidates_negative"] == 17
    report_uids = {e.uid for e in report.examples}
    assert not report_uids & {"sst2-train-b0", "sst2-train-b1"}
    assert not report_uids & {e.uid for e in short}


def test_sentiment_removes_items_contained_in_or_containing_reserved_text() -> (
    None
):
    base = _sst(40, 12)
    part = i1.partition_sentiment(base, sizes=SMALL)
    reserved = part.blocks["report"].examples[0].text
    words = reserved.split()
    keys = list(part.blocks["report"].keys)
    inside = _ex(_late_uid("sst2-train-in", keys), " ".join(words[2:6]), 1)
    around = _ex(_late_uid("sst2-train-out", keys), f"zz {reserved} zz", 0)
    partial = _ex("sst2-train-part", words[3][:-1] + " " + words[4], 1)
    again = i1.partition_sentiment(
        [*base, inside, around, partial], sizes=SMALL
    )
    assert again.blocks["report"] == part.blocks["report"]
    used = set(_all_uids(again))
    assert inside.uid not in used
    assert around.uid not in used
    assert again.counts["removed_by_report"] >= 2


def test_sentiment_removes_fit_and_pool_items_overlapping_validation() -> None:
    base = _sst(40, 12)
    part = i1.partition_sentiment(base, sizes=SMALL)
    text = part.blocks["validation"].examples[0].text
    uid = _late_uid(
        "sst2-train-sub",
        [*part.blocks["report"].keys, *part.blocks["validation"].keys],
    )
    sub = _ex(uid, " ".join(text.split()[1:4]), 1)
    again = i1.partition_sentiment([*base, sub], sizes=SMALL)
    assert again.blocks["validation"] == part.blocks["validation"]
    assert uid not in set(_all_uids(again))
    assert again.counts["removed_by_validation"] == 1


def test_contains_words_is_whole_word_and_normalized() -> None:
    assert i1.contains_words("It  was BAD . ok", "bad .")
    assert not i1.contains_words("it was badly .", "bad .")
    assert i1.contains_words("bad .", "Bad .")
    assert not i1.contains_words("bad", "bad .")


def _truth_fixture() -> tuple[list[Example], list[Example]]:
    common = [
        _ex("got-c-0", "Water is wet.", 1, subset="common_claim_true_false"),
        _ex("got-c-1", "Fire is cold.", 0, "test"),
        _ex("got-c-2", "water is  WET", 1),
        _ex("got-c-3", "Dogs bark.", 1, "train"),
        _ex("got-c-4", "dogs bark.", 1, "val"),
        _ex("got-c-5", "The sun is hot.", 1, "test"),
    ]
    common += [_ex(f"got-t-{i}", f"Train claim {i}.", i % 2) for i in range(30)]
    common += [
        _ex(f"got-x-{i}", f"Test claim {i}.", i % 2, "test") for i in range(6)
    ]
    facts = [_ex("facts-0", "the sun is hot", 1)]
    return common, facts


def test_truth_removes_facts_matches_and_cross_split_repeats() -> None:
    common, facts = _truth_fixture()
    part = i1.partition_truth(common, facts, sizes=SMALL)
    counts = part.counts
    assert counts["facts_matches"] == 1
    assert counts["train"] == 31
    assert counts["test"] == 7
    assert counts["test_true"] == 3
    uids = set(_all_uids(part))
    assert not uids & {"got-c-2", "got-c-3", "got-c-4", "got-c-5"}
    report = part.blocks["report"]
    assert all(e.split == "test" for e in report.examples)
    _assert_pairs(report)
    for name in ("validation", "fit", "pool"):
        assert all(e.split == "train" for e in part.blocks[name].examples)


def _negations() -> tuple[list[Example], dict[str, list[str]]]:
    rows: list[Example] = []
    cities: dict[str, list[str]] = {"cities": [], "neg_cities": []}
    for subset, negated in (("cities", False), ("neg_cities", True)):
        for i in range(12):
            city = f"C{i}"
            verb = "is not" if negated else "is"
            for j, country in enumerate(("A", "B")):
                label = int((j == 0) != negated)
                n = len(cities[subset])
                split: Split = "test" if i == 0 and j == 1 else "train"
                rows.append(
                    _ex(
                        f"got-{subset}-{n}",
                        f"The city of {city} {verb} in {country}.",
                        label,
                        split,
                        subset=subset,
                    )
                )
                cities[subset].append(city)
    for subset, verb in (("sp_en_trans", "means"), ("neg_sp_en_trans", "x")):
        for i in range(4):
            rows.append(
                _ex(
                    f"got-{subset}-{i}",
                    f"The Spanish word 'p{i}' {verb} 'w{i}'.",
                    i % 2,
                    subset=subset,
                )
            )
    rows.append(
        _ex(
            "got-sp_en_trans-4", "No quoted word here.", 1, subset="sp_en_trans"
        )
    )
    rows.append(_ex("got-cities-dup", "the city of C3 is in A", 1))
    rows.append(_ex("got-cities-dup2", "The city of C5 is in A.", 1, "val"))
    return rows, cities


def test_negation_subjects_come_from_the_city_column_or_quoted_word() -> None:
    rows, cities = _negations()
    keys, notes = i1.negation_subjects(rows, cities)
    assert keys["got-cities-0"] == keys["got-neg_cities-0"] == "city:C0"
    assert keys["got-sp_en_trans-1"] == keys["got-neg_sp_en_trans-1"]
    assert keys["got-sp_en_trans-1"] == "es:p1"
    assert keys["got-sp_en_trans-4"] == "uid:got-sp_en_trans-4"
    assert any("got-sp_en_trans-4" in note for note in notes)


def test_negation_report_uses_only_subjects_left_in_train() -> None:
    rows, cities = _negations()
    keys, _ = i1.negation_subjects(rows, cities)
    keys["got-cities-dup"] = "city:C3"
    keys["got-cities-dup2"] = "city:C5"
    part = i1.partition_negations(rows, keys, sizes=SMALL)
    report_keys = set(part.blocks["report"].keys)
    assert "city:C0" not in report_keys
    assert "city:C5" not in report_keys
    assert part.counts["eligible_subjects"] == 15
    seen: dict[str, str] = {}
    for name, block in part.blocks.items():
        for key in block.keys:
            assert seen.setdefault(key, name) == name
        for e in block.examples:
            assert keys[e.uid] in block.keys
    used = set(_all_uids(part))
    assert "got-cities-dup" not in used
    assert "got-cities-10" not in used
    assert part.counts["train_repeating_eval"] == 1
    assert part.counts["repeated_statements"] == 2


def _utility_rows() -> list[Example]:
    rows: list[Example] = []

    def row(split: Split, n: int, a: str, b: str) -> None:
        file = "test" if split == "test" else "train"
        pair = f"ethics-util-{file}-{n}"
        rows.append(_ex(f"{pair}:a", a, 1, split, pair=pair))
        rows.append(_ex(f"{pair}:b", b, 0, split, pair=pair))

    for n in range(20):
        row("train", n, f"tr{n}a", f"tr{n}b")
    row("val", 20, "va", "vb")
    row("test", 0, "tr0a", "ts0b")
    row("test", 1, "ts0b", "ts1b")
    row("test", 2, "vb", "ts2b")
    row("test", 3, "m2a", "m2b")
    row("test", 4, "m2b", "ts4b")
    for n in range(5, 12):
        row("test", n, f"ts{n}a", f"ts{n}b")
    row("test", 12, "ts11b", "ts12b")
    return rows


def test_utility_excludes_test_pairs_linked_to_train_or_m2_report() -> None:
    rows = _utility_rows()
    part = i1.partition_utility(
        rows, frozenset({"ethics-util-test-3"}), sizes=SMALL
    )
    assert part.counts["report_candidate_pairs"] == 8
    assert part.counts["report_candidate_groups"] == 7
    report = part.blocks["report"]
    _assert_pairs(report)
    pairs = {str(e.meta["pair"]) for e in report.examples}
    excluded = {f"ethics-util-test-{n}" for n in range(5)}
    assert not pairs & excluded
    groups = [{"ethics-util-test-11", "ethics-util-test-12"}]
    for group in groups:
        assert not pairs & group or group <= pairs
    for name in ("validation", "fit", "pool"):
        assert all(e.split == "train" for e in part.blocks[name].examples)


def test_linked_utility_groups_are_never_split() -> None:
    rows = _utility_rows()
    tiny = i1.BlockSizes(report=1, validation=1, fit=1, pool=2)
    part = i1.partition_utility(rows, frozenset(), sizes=tiny)
    owner: dict[str, str] = {}
    for name, block in part.blocks.items():
        for key in block.keys:
            assert owner.setdefault(key, name) == name


def test_m2_utility_report_pairs_are_test_rows_drawn_with_the_m2_seed() -> None:
    rows = _utility_rows()
    pairs = i1.m2_utility_report_pairs(rows, n_pairs=5)
    assert len(pairs) == 5
    assert all(p.startswith("ethics-util-test-") for p in pairs)
    assert pairs == i1.m2_utility_report_pairs(rows, n_pairs=5)
    seeds = {
        i1.m2_utility_report_pairs(rows, n_pairs=5, seed=s) for s in range(6)
    }
    assert len(seeds) > 1


def test_draws_are_seeded_permutations_of_the_fit_reservoir() -> None:
    part = i1.partition_sentiment(
        _sst(1200, 12), sizes=i1.BlockSizes(report=10, validation=10, fit=100)
    )
    fit = part.blocks["fit"].pairs
    draws = i1.draws(part)
    assert len(draws) == 10
    for draw in draws:
        assert len(draw) == 64
        assert len(set(draw)) == 64
        assert set(draw) <= set(fit)
    assert len({tuple(d) for d in draws}) == 10
    assert draws == i1.draws(part)


def test_item_ids_list_every_block_in_order() -> None:
    part = i1.partition_sentiment(_sst(40, 12), sizes=SMALL)
    ids = part.item_ids()
    assert list(ids) == ["report", "validation", "fit", "pool"]
    assert ids["fit"]["pairs"] == [list(p) for p in part.blocks["fit"].pairs]
    assert ids["pool"]["uids"] == [e.uid for e in part.blocks["pool"].examples]


def _count(partition: i1.TaskPartition, keys: Sequence[str]) -> dict[str, int]:
    return {k: partition.counts[k] for k in keys}


@pytest.mark.network
def test_real_partition_counts_match_the_preregistration() -> None:
    parts = i1.load_partitions()
    observed = {
        "truth": _count(parts["truth"], ("train", "test", "test_true")),
        "truth_negations": _count(
            parts["truth_negations"],
            (
                "eligible_subjects",
                "eligible_statements",
                "eligible_pairs",
                "train_distinct",
            ),
        ),
        "utility": _count(
            parts["utility"],
            ("report_candidate_pairs", "report_candidate_groups"),
        ),
        "sentiment": _count(
            parts["sentiment"],
            (
                "burned_candidates",
                "candidates_positive",
                "candidates_negative",
            ),
        ),
    }
    assert observed == {
        "truth": {"train": 2622, "test": 912, "test_true": 451},
        "truth_negations": {
            "eligible_subjects": 217,
            "eligible_statements": 652,
            "eligible_pairs": 326,
            "train_distinct": 2230,
        },
        "utility": {
            "report_candidate_pairs": 803,
            "report_candidate_groups": 608,
        },
        "sentiment": {
            "burned_candidates": 18,
            "candidates_positive": 12987,
            "candidates_negative": 12165,
        },
    }
    for part in parts.values():
        assert [
            len(part.blocks[b].pairs) for b in ("report", "validation")
        ] == [
            256,
            200,
        ]
        assert len(part.blocks["fit"].pairs) == 192
        assert len(part.blocks["pool"].examples) == 640
