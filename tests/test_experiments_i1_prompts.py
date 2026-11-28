from collections import Counter
from types import MappingProxyType

import pytest

from tomography.datasets.records import Example
from tomography.experiments import i1_invariance as i1
from tomography.templates import ChatFormat

PLAIN = ChatFormat("plain")


def _block(task: str, name: i1.BlockName, n: int) -> i1.Block:
    if name == "pool":
        texts = tuple(
            Example(f"{task}-{name}-{i}", f"pool text {i}", i % 2, "train")
            for i in range(n)
        )
        return i1.Block(name, tuple(e.uid for e in texts), texts, ())
    texts = tuple(
        Example(
            f"{task}-{name}-{i}-{side}",
            f"text {i} {side}",
            int(side == "a"),
            "train",
            {"pair": f"{task}-{name}-{i}"},
        )
        for i in range(n)
        for side in ("a", "b")
    )
    pairs = tuple((texts[2 * i].uid, texts[2 * i + 1].uid) for i in range(n))
    return i1.Block(name, tuple(p[0] for p in pairs), texts, pairs)


def _partitions(sizes: i1.BlockSizes) -> dict[str, i1.TaskPartition]:
    return {
        task: i1.TaskPartition(
            task,
            MappingProxyType(
                {
                    name: _block(task, name, getattr(sizes, name))
                    for name in i1.BLOCKS
                }
            ),
            MappingProxyType({}),
            (),
        )
        for task in i1.TASKS
    }


@pytest.fixture(scope="module")
def stage0() -> list[i1.Prompt]:
    return i1.stage0_prompts(_partitions(i1.PREREGISTERED_SIZES), PLAIN)


@pytest.fixture(scope="module")
def stage2() -> list[i1.Prompt]:
    return i1.stage2_prompts(_partitions(i1.PREREGISTERED_SIZES), PLAIN)


def test_stage0_counts_match_section_6_1(stage0: list[i1.Prompt]) -> None:
    per_task = Counter(p.task for p in stage0)
    assert per_task == {
        "sentiment": 9968,
        "truth": 9968,
        "truth_negations": 9968,
        "utility": 6096,
    }
    assert len(stage0) == 36000


def test_stage0_conditions_per_block(stage0: list[i1.Prompt]) -> None:
    def conditions(task: str, block: str) -> list[str]:
        seen: dict[str, None] = {}
        for p in stage0:
            if (p.task, p.block) == (task, block):
                seen.setdefault(p.condition)
        return list(seen)

    fit_concept = [
        "clean",
        "F-T1",
        "F-T2",
        "F-T3",
        "F-D1/random",
        "F-D2/random",
        "F-D3/random",
    ]
    assert conditions("sentiment", "fit") == fit_concept
    assert conditions("sentiment", "pool") == fit_concept
    assert conditions("utility", "pool") == [
        "clean",
        "F-D1/random",
        "F-D2/random",
        "F-D3/random",
    ]
    assert conditions("truth", "validation") == [
        "clean",
        "S-T1",
        "S-T2",
        "S-D1/random",
        "S-D1/against",
        "S-D2/random",
        "S-D2/against",
    ]
    assert conditions("utility", "validation") == [
        "clean",
        "S-D1/random",
        "S-D1/against",
        "S-D2/random",
        "S-D2/against",
    ]
    assert not [p for p in stage0 if p.block == "report"]


def test_against_never_reaches_fit_or_pool(
    stage0: list[i1.Prompt], stage2: list[i1.Prompt]
) -> None:
    for prompt in (*stage0, *stage2):
        if prompt.condition.endswith("/against"):
            assert prompt.block in ("validation", "report")


def test_stage0_renders_no_test_wording(stage0: list[i1.Prompt]) -> None:
    test_names = {
        w.name for w in i1.WORDINGS.values() if w.wording_set == "test"
    }
    for prompt in stage0:
        assert prompt.condition.split("/")[0] not in test_names


def test_prompts_carry_span_label_and_drawn_level(
    stage0: list[i1.Prompt],
) -> None:
    texts = {
        (task, e.uid): (e.text, e.label)
        for task, part in _partitions(i1.PREREGISTERED_SIZES).items()
        for block in part.blocks.values()
        for e in block.examples
    }
    for prompt in stage0:
        text, label = texts[(prompt.task, prompt.uid)]
        assert prompt.text[slice(*prompt.stimulus_span)] == text
        assert prompt.label == label
    pool = [
        p for p in stage0 if p.block == "pool" and p.condition == "F-D3/random"
    ]
    assert {p.level for p in pool} == {"weak", "strong"}
    example = Example(pool[0].uid, "x", pool[0].label, "train")
    assert pool[0].level == i1.decoy_level(
        i1.WORDINGS["F-D3"], example, "random"
    )
    levels = i1.pool_levels(stage0)
    assert levels["sentiment"]["F-D3"][pool[0].uid] == pool[0].level
    assert set(levels["utility"]) == {"F-D1", "F-D2", "F-D3"}
    assert len(levels["truth"]["F-D1"]) == 640


def test_stage2_counts_match_section_10(stage2: list[i1.Prompt]) -> None:
    per_task = Counter(p.task for p in stage2)
    assert per_task == {
        "sentiment": 13312,
        "truth": 13312,
        "truth_negations": 13312,
        "utility": 8704,
    }
    assert len(stage2) == 48640
    report = [p for p in stage2 if p.task == "truth" and p.block == "report"]
    assert list(dict.fromkeys(p.condition for p in report)) == [
        "clean",
        "M2-D/random",
        "M2-D/against",
        "T-D1/random",
        "T-D1/against",
        "T-D2/random",
        "T-D2/against",
        "T-D3/random",
        "T-D3/against",
        "P1",
        "P2",
        "P3",
    ]
    assert not [p for p in stage2 if p.block == "validation"]


def test_oracle_renders_each_wrapped_report_text_clean(
    stage2: list[i1.Prompt],
) -> None:
    oracle = i1.oracle_prompts(stage2, PLAIN)
    wrapped = [
        p for p in stage2 if p.block == "report" and p.condition != "clean"
    ]
    assert len(oracle) == len(wrapped)
    clean = {
        (p.task, p.uid): p.text
        for p in stage2
        if p.block == "report" and p.condition == "clean"
    }
    for source, prompt in zip(wrapped, oracle, strict=True):
        assert prompt.condition == f"oracle/{source.condition}"
        assert (prompt.task, prompt.uid, prompt.label) == (
            source.task,
            source.uid,
            source.label,
        )
        assert prompt.text == clean[(prompt.task, prompt.uid)]
        assert prompt.level is None
