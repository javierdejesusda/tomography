"""The item pool of the knowledge boundary study and its labels.

One pool holds the questions of one model: real entities, invented ones,
unanswerable questions and open-domain trivia. Each question carries the
behavioral label of `tomography.knowledge.label_questions`, and the guess
control of `tomography.p9.guess` may later demote a known label.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from typing import Self
from typing import cast
import zlib

import numpy as np

from tomography.datasets.falseqa import load_falseqa
from tomography.datasets.fictional import generate_fictional
from tomography.datasets.kuq import load_kuq
from tomography.datasets.nq_open import load_nq_open
from tomography.datasets.popqa import load_popqa
from tomography.datasets.records import MetaValue
from tomography.datasets.records import QAExample
from tomography.datasets.records import Split
from tomography.datasets.synthal import load_synthal
from tomography.datasets.triviaqa import load_triviaqa
from tomography.datasets.wikidata import load_post_cutoff
from tomography.datasets.wikidata import load_unknowable_detail
from tomography.knowledge import Basis
from tomography.knowledge import ItemLabel
from tomography.knowledge import Label
from tomography.model import LoadedModel

GROUPS: tuple[str, ...] = (
    "popqa",
    "synthal",
    "fictional",
    "award",
    "detail",
    "falseqa",
    "kuq",
    "triviaqa",
    "nq_open",
)

GuessVerdict = Literal["excluded_stratum", "guessable_stratum", "modal_answer"]
Loader = Callable[[], list[QAExample]]

_NO_CUTOFF = 9999

_SOURCE_GROUP = {
    "popqa": "popqa",
    "synthal": "synthal",
    "generated": "fictional",
    "kuq": "kuq",
    "falseqa": "falseqa",
    "triviaqa": "triviaqa",
    "nq_open": "nq_open",
}


def group_of(example: QAExample) -> str:
    """Names the pool group of a question.

    Args:
        example: A question whose meta holds "source".

    Returns:
        The group, one of `GROUPS`: the source name, except that generated
        names are "fictional" and Wikidata items are "detail" for the kind
        "unknowable_detail" and "award" otherwise.

    Raises:
        ValueError: If the source is missing or not one of the pool's.
    """
    source = example.meta.get("source")
    if source == "wikidata":
        return "detail" if example.kind == "unknowable_detail" else "award"
    if not isinstance(source, str) or source not in _SOURCE_GROUP:
        raise ValueError(f"{example.uid}: unknown source {source!r}")
    return _SOURCE_GROUP[source]


@dataclass(frozen=True)
class LabeledItem:
    """One question with the label of one model.

    Attributes:
        example: The question.
        label: The raw behavioral label.
        guess: Why a known label failed the guess control, or None when the
            label stands.

    Raises:
        ValueError: If the label belongs to another question or a verdict
            is set on a label that is not "known".
    """

    example: QAExample
    label: ItemLabel
    guess: GuessVerdict | None = None

    def __post_init__(self) -> None:
        """Checks that the label fits the question and the verdict."""
        if self.label.uid != self.example.uid:
            raise ValueError(
                f"label {self.label.uid!r} does not belong to "
                f"{self.example.uid!r}"
            )
        if self.guess is not None and self.label.label != "known":
            raise ValueError(
                f"{self.example.uid}: only a known label can fail the "
                "guess control"
            )

    @property
    def group(self) -> str:
        """The pool group of the question."""
        return group_of(self.example)

    @property
    def contrast_label(self) -> Label:
        """The label every contrast and count uses.

        "uncertain" when the guess control rejected a known label, else the
        raw label.
        """
        return "uncertain" if self.guess is not None else self.label.label


def subsample(
    items: Sequence[QAExample],
    n: int | None,
    *,
    seed: int,
    by: Callable[[QAExample], str] | None = None,
    prefer: Callable[[QAExample], bool] | None = None,
    share: float = 0.5,
) -> list[QAExample]:
    """Caps the number of questions per stratum with a seeded draw.

    A stratum larger than `n` keeps `n` questions chosen by a seeded
    permutation of its uid-sorted members, so the draw does not depend on
    the input order. The result keeps the order of `items`.

    With `prefer`, a capped stratum takes up to `round(share * n)` preferred
    questions first, fills the rest from the others, and tops up from the
    preferred ones when the others run short.

    Args:
        items: The questions.
        n: Questions kept per stratum; None keeps every question.
        seed: Seeds the draw.
        by: Stratum of a question; None puts all questions in one stratum.
        prefer: Marks the questions the draw favors, if any.
        share: Share of `n` drawn from the preferred questions first.

    Returns:
        The kept questions, in their original order.

    Raises:
        ValueError: If `n` is negative or `share` is outside [0, 1].
    """
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"share must be in [0, 1], got {share}")
    if n is None:
        return list(items)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    strata: dict[str, list[int]] = {}
    for index, example in enumerate(items):
        key = "" if by is None else by(example)
        strata.setdefault(key, []).append(index)
    keep: set[int] = set()
    for key, indices in strata.items():
        if len(indices) <= n:
            keep.update(indices)
            continue
        ordered = sorted(indices, key=lambda i: items[i].uid)
        rng = np.random.default_rng([seed, zlib.crc32(key.encode())])
        shuffled = [ordered[int(j)] for j in rng.permutation(len(ordered))]
        if prefer is None:
            keep.update(shuffled[:n])
            continue
        first = [i for i in shuffled if prefer(items[i])]
        rest = [i for i in shuffled if not prefer(items[i])]
        take = min(len(first), round(share * n))
        chosen = first[:take] + rest[: n - take]
        chosen += first[take : take + n - len(chosen)]
        keep.update(chosen)
    return [example for i, example in enumerate(items) if i in keep]


@dataclass(frozen=True)
class PoolSizes:
    """How many questions the pool keeps; None keeps every question.

    Attributes:
        popqa_per_relation: PopQA questions per relation.
        synthal_per_relation: SyntHal questions per relation.
        kuq_per_category: KUQ questions per category.
        falseqa: FalseQA questions.
        triviaqa: TriviaQA questions.
        nq_open: NQ-open questions.
        popular_min: A PopQA question is popular when its popularity is at
            least this.
        popular_share: Share of a capped PopQA relation drawn from the
            popular questions first.
    """

    popqa_per_relation: int | None = None
    synthal_per_relation: int | None = None
    kuq_per_category: int | None = 200
    falseqa: int | None = 500
    triviaqa: int | None = 1000
    nq_open: int | None = 1000
    popular_min: int = 10_000
    popular_share: float = 0.5


@dataclass(frozen=True)
class PoolLoaders:
    """The sources of the pool, injectable so tests read no dataset.

    Attributes:
        popqa: PopQA questions of the shared relations.
        synthal: SyntHal questions about invented entities.
        kuq: KUQ questions.
        falseqa: FalseQA questions with a false premise.
        triviaqa: TriviaQA questions.
        nq_open: NQ-open questions.
        awards: Wikidata award winners of every year, whose kind is
            reassigned by `assign_award_kinds`.
        details: Unknowable-detail questions about famous people.
    """

    popqa: Loader
    synthal: Loader
    kuq: Loader
    falseqa: Loader
    triviaqa: Loader
    nq_open: Loader
    awards: Loader
    details: Loader

    @classmethod
    def default(
        cls,
        *,
        offline: bool = False,
        seed: int = 0,
        n_detail: int = 300,
        cache: Path | None = None,
    ) -> Self:
        """Binds the dataset loaders of the study.

        The award loader gets a cutoff year above every year, so it labels
        every award question "real" and the kind is assigned later from the
        model's cutoff.

        Args:
            offline: Whether to read only what is already cached.
            seed: Seed of the unknowable-detail draw.
            n_detail: Number of unknowable-detail questions.
            cache: Cache directory; None means the shared cache.

        Returns:
            The loaders; nothing is read until one is called.
        """
        return cls(
            popqa=lambda: load_popqa(cache=cache, offline=offline),
            synthal=lambda: load_synthal(cache=cache, offline=offline),
            kuq=lambda: load_kuq(cache=cache, offline=offline),
            falseqa=lambda: load_falseqa(cache=cache, offline=offline),
            triviaqa=lambda: load_triviaqa(cache=cache, offline=offline),
            nq_open=lambda: load_nq_open(cache=cache, offline=offline),
            awards=lambda: load_post_cutoff(
                cutoff_year=_NO_CUTOFF, cache=cache, offline=offline
            ),
            details=lambda: load_unknowable_detail(
                seed=seed, limit=n_detail, cache=cache, offline=offline
            ),
        )


def name_token_counter(loaded: LoadedModel) -> Callable[[str], int]:
    """Counts the tokens of a name as this model sees it in a question.

    Args:
        loaded: The wrapped model.

    Returns:
        A function from a name to the token count of " " + name with the
        model's tokenizer and no special tokens.
    """

    def count(name: str) -> int:
        ids = loaded.tokenizer(" " + name, add_special_tokens=False)
        return len(ids["input_ids"])

    return count


def build_pool(
    loaders: PoolLoaders,
    sizes: PoolSizes,
    *,
    count_tokens: Callable[[str], int],
    seed: int = 0,
) -> list[QAExample]:
    """Assembles the questions of one model.

    The order is PopQA (capped per relation, popular questions favored by
    `sizes.popular_share`), the fictional counterparts of those PopQA
    questions, SyntHal, the award questions, the unknowable details,
    FalseQA, KUQ (capped per category), TriviaQA and NQ-open. A counterpart
    is generated only for a PopQA subject of at least two tokens: the
    generator cannot match a one-token name, and the PopQA question itself
    stays in the pool.

    Args:
        loaders: The sources.
        sizes: The caps.
        count_tokens: Token count of a name, as `name_token_counter` gives.
        seed: Seeds the caps and the invented names.

    Returns:
        The questions, in the order above.

    Raises:
        ValueError: If a question has an unknown source or a uid repeats.
    """
    popqa = subsample(
        loaders.popqa(),
        sizes.popqa_per_relation,
        seed=seed,
        by=_relation,
        prefer=lambda e: _is_popular(e, sizes.popular_min),
        share=sizes.popular_share,
    )
    matchable = [e for e in popqa if count_tokens(str(e.meta["subject"])) >= 2]
    pool = [
        *popqa,
        *generate_fictional(matchable, count_tokens=count_tokens, seed=seed),
        *subsample(
            loaders.synthal(),
            sizes.synthal_per_relation,
            seed=seed,
            by=_relation,
        ),
        *loaders.awards(),
        *loaders.details(),
        *subsample(loaders.falseqa(), sizes.falseqa, seed=seed),
        *subsample(
            loaders.kuq(), sizes.kuq_per_category, seed=seed, by=_category
        ),
        *subsample(loaders.triviaqa(), sizes.triviaqa, seed=seed),
        *subsample(loaders.nq_open(), sizes.nq_open, seed=seed),
    ]
    seen: set[str] = set()
    for example in pool:
        group_of(example)
        if example.uid in seen:
            raise ValueError(f"duplicate uid {example.uid!r}")
        seen.add(example.uid)
    return pool


def _relation(example: QAExample) -> str:
    return str(example.meta["relation"])


def _category(example: QAExample) -> str:
    return str(example.meta["category"])


def _is_popular(example: QAExample, popular_min: int) -> bool:
    popularity = example.meta.get("popularity")
    return isinstance(popularity, int) and popularity >= popular_min


def assign_award_kinds(
    items: Sequence[QAExample], cutoff_year: int
) -> list[QAExample]:
    """Sets the kind of every award question from the model's cutoff.

    Run after labeling, so one labeling pass serves any cutoff.

    Args:
        items: The questions.
        cutoff_year: Last year of the model's training data.

    Returns:
        The questions in order. An award question is replaced by a copy of
        kind "post_cutoff" when its year is after `cutoff_year` and "real"
        otherwise, with `meta["cutoff_year"]` set; others are returned
        unchanged.

    Raises:
        ValueError: If an award question has a year that is not an int.
    """
    out = []
    for example in items:
        if group_of(example) != "award":
            out.append(example)
            continue
        year = example.meta.get("year")
        if not isinstance(year, int):
            raise ValueError(f"{example.uid}: meta['year'] must be an int")
        out.append(
            dataclasses.replace(
                example,
                kind="post_cutoff" if year > cutoff_year else "real",
                meta={**example.meta, "cutoff_year": cutoff_year},
            )
        )
    return out


def pool_records(items: Sequence[LabeledItem]) -> list[dict[str, object]]:
    """Flattens labeled questions into JSON-able dicts.

    Args:
        items: The labeled questions.

    Returns:
        One dict per item with the question fields (`uid`, `question`,
        `answers`, `kind`, `split`, `meta`), the label fields (`label`,
        `basis`, `greedy`, `greedy_correct`, `samples`, `n_correct`) and
        `guess`.
    """
    return [
        {
            "uid": item.example.uid,
            "question": item.example.question,
            "answers": list(item.example.answers),
            "kind": item.example.kind,
            "split": item.example.split,
            "meta": dict(item.example.meta),
            "label": item.label.label,
            "basis": item.label.basis,
            "greedy": item.label.greedy,
            "greedy_correct": item.label.greedy_correct,
            "samples": list(item.label.samples),
            "n_correct": item.label.n_correct,
            "guess": item.guess,
        }
        for item in items
    ]


def pool_from_records(
    records: Sequence[Mapping[str, object]],
) -> list[LabeledItem]:
    """Rebuilds labeled questions from `pool_records`, possibly via JSON.

    Args:
        records: Dicts as `pool_records` returns, or their JSON round trip.

    Returns:
        The labeled questions, with `answers` and `samples` as tuples.
    """
    items = []
    for record in records:
        example = QAExample(
            uid=str(record["uid"]),
            question=str(record["question"]),
            answers=tuple(str(a) for a in cast(list[str], record["answers"])),
            kind=str(record["kind"]),
            split=cast(Split, record["split"]),
            meta=dict(cast(Mapping[str, MetaValue], record["meta"])),
        )
        label = ItemLabel(
            uid=example.uid,
            label=cast(Label, record["label"]),
            basis=cast(Basis, record["basis"]),
            greedy=str(record["greedy"]),
            greedy_correct=bool(record["greedy_correct"]),
            samples=tuple(str(s) for s in cast(list[str], record["samples"])),
            n_correct=int(cast(int, record["n_correct"])),
        )
        items.append(
            LabeledItem(example, label, cast(GuessVerdict, record["guess"]))
        )
    return items
