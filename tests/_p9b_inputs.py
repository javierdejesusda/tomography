"""The planted labels the P9b and V1 tests read.

The pool holds thirteen contrasts, eleven of them usable at the settings of
the tests. A labels run of the pool stands in for a finished P9a run.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tomography.datasets.records import MetaValue
from tomography.datasets.records import QAExample
from tomography.datasets.records import Split
from tomography.experiments.base import RunDir
from tomography.experiments.p9a_labels import Config as LabelsConfig
from tomography.knowledge import ItemLabel
from tomography.knowledge import Label
from tomography.p9.pool import LabeledItem
from tomography.p9.pool import pool_records

SPLITS: tuple[Split, ...] = ("train", "train", "train", "val", "test")
LABELS: tuple[Label, ...] = (
    "known",
    "known",
    "unknown",
    "unknown",
    "known",
    "uncertain",
)
RELATIONS = ("father", "mother", "occupation")
USABLE = [
    "fictional_synthal",
    "fictional_generated",
    "obscure_real",
    "obscure_real_pop",
    "post_cutoff",
    "triviaqa_known",
    "unknowable_detail",
    "false_premise",
    "kuq_future_unknown",
    "real_vs_synthal",
    "real_vs_generated",
]


def items(*, post_awards_apart: bool = False) -> list[LabeledItem]:
    """A planted pool: eleven contrasts are usable, two are too small.

    With `post_awards_apart` the post-cutoff awards share no relation with
    the real ones, so `post_cutoff` has no pair to match.
    """
    planted: list[LabeledItem] = []

    def add(
        prefix: str,
        n: int,
        source: str,
        kind: str,
        label_of: Any,
        meta_of: Any,
    ) -> None:
        for i in range(n):
            uid = f"{prefix}:{i}"
            label: Label = label_of(i)
            correct = label == "known"
            example = QAExample(
                uid=uid,
                question=f"Who is {prefix} {i}?",
                answers=("gold",),
                kind=kind,
                split=SPLITS[i % 5],
                meta={"source": source, **meta_of(i)},
            )
            item_label = ItemLabel(
                uid=uid,
                label=label,
                basis="behavior",
                greedy="gold" if correct else "other",
                greedy_correct=correct,
                samples=("gold",) * 5 if correct else ("other",) * 5,
                n_correct=5 if correct else 0,
            )
            planted.append(LabeledItem(example, item_label))

    def unknown(_: int) -> Label:
        return "unknown"

    def relation(i: int) -> dict[str, MetaValue]:
        return {"relation": RELATIONS[i % 3], "popularity": None}

    add(
        "popqa",
        150,
        "popqa",
        "real",
        lambda i: LABELS[i % 6],
        lambda i: {
            "relation": RELATIONS[i % 3],
            "popularity": 10 ** (1 + i % 7),
        },
    )
    add("synthal", 50, "synthal", "fictional", unknown, relation)
    add("generated", 50, "generated", "fictional", unknown, relation)
    add(
        "award-real",
        40,
        "wikidata",
        "real",
        lambda i: "unknown" if i < 2 else "known",
        lambda i: {"relation": f"Award{i % 2}", "year": 1990 + i % 23},
    )
    add(
        "award-post",
        30,
        "wikidata",
        "post_cutoff",
        unknown,
        lambda i: {
            "relation": f"{'Post' if post_awards_apart else 'Award'}{i % 2}",
            "year": 2015 + i % 10,
        },
    )
    add(
        "detail",
        40,
        "wikidata",
        "unknowable_detail",
        unknown,
        lambda i: {"relation": "ordinary_date"},
    )
    add("falseqa", 40, "falseqa", "false_premise", unknown, lambda i: {})
    add(
        "kuq-future",
        40,
        "kuq",
        "kuq:future unknown",
        unknown,
        lambda i: {"category": "future unknown"},
    )
    add(
        "kuq-assumption",
        8,
        "kuq",
        "kuq:false assumption",
        unknown,
        lambda i: {"category": "false assumption"},
    )
    for source in ("triviaqa", "nq_open"):
        add(
            source,
            60,
            source,
            "open",
            lambda i: "known" if i % 2 == 0 else "unknown",
            lambda i: {},
        )
    return planted


def labels_run(
    tmp_path: Path,
    labeled: Sequence[LabeledItem],
    *,
    known_contrasts: bool = True,
    finish: bool = True,
) -> RunDir:
    """Writes a P9a run that holds `labeled` under `tmp_path / "labels"`."""
    run_dir = RunDir.create(
        LabelsConfig(model="tiny"), out_root=tmp_path / "labels"
    )
    run_dir.write_json("items", pool_records(labeled))
    run_dir.write_json("summary", {"known_contrasts": known_contrasts})
    if finish:
        run_dir.finalize()
    return run_dir
