"""Tests for the reading half of the P9 knowledge boundary study."""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.activations import collect_activations
from tomography.datasets.records import MetaValue
from tomography.datasets.records import QAExample
from tomography.datasets.records import Split
from tomography.datasets.records import assign_splits
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import ItemLabel
from tomography.knowledge import Label
from tomography.knowledge import few_shot_prompt
from tomography.model import LoadedModel
from tomography.p9.pool import LabeledItem
import tomography.p9.reading as reading_module
from tomography.p9.reading import READING_METHODS
from tomography.p9.reading import TEMPLATE_NAMES
from tomography.p9.reading import WITHIN_BASELINES
from tomography.p9.reading import ItemActivations
from tomography.p9.reading import ReadingCube
from tomography.p9.reading import TrainDraw
from tomography.p9.reading import check_pool
from tomography.p9.reading import choose_layer
from tomography.p9.reading import collect_item_activations
from tomography.p9.reading import draw_seed
from tomography.p9.reading import draw_train
from tomography.p9.reading import fit_draw
from tomography.p9.reading import fit_type_reader
from tomography.p9.reading import p9_template
from tomography.p9.reading import reading_cube
from tomography.p9.reading import relation_groups
from tomography.p9.reading import render_items
from tomography.p9.reading import surface_baselines
from tomography.p9.reading import type_auroc
from tomography.p9.reading import type_scores
from tomography.p9.reading import within_relation_pairs
from tomography.p9.unknowns import TypePool
from tomography.p9.unknowns import relation_of
from tomography.p9.unknowns import stratum_name
from tomography.p9.unknowns import type_pools
from tomography.reading.directions import Method
from tomography.reading.metrics import auroc
from tomography.reading.reader import RepReader
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat

D = 256
SHIFT = 8.0
LAYERS = (1, 2, 3)
SIGNAL = (1, 2)
N_SEEDS = 3
N_PAIRS = 64
A_GROUP = [
    "obscure_real",
    "obscure_real_pop",
    "unknowable_detail",
    "false_premise",
    "kuq_false_assumption",
    "kuq_future_unknown",
]
B_GROUP = ["real_vs_synthal", "real_vs_generated", "pre_vs_post_cutoff"]
MIXED = ["fictional_synthal", "fictional_generated", "post_cutoff"]


def _labeled(
    uid: str,
    label: Label = "known",
    *,
    source: str = "triviaqa",
    kind: str = "open",
    split: Split = "train",
    **meta: MetaValue,
) -> LabeledItem:
    example = QAExample(
        uid=uid,
        question=f"question {uid}",
        answers=("gold",),
        kind=kind,
        split=split,
        meta={"source": source, **meta},
    )
    correct = label == "known"
    item_label = ItemLabel(
        uid=uid,
        label=label,
        basis="behavior",
        greedy="gold" if correct else "other",
        greedy_correct=correct,
        samples=("gold",) * 5 if correct else ("other",) * 5,
        n_correct=5 if correct else 0,
    )
    return LabeledItem(example, item_label)


def _specs() -> list[tuple[str, str, str, Label, dict[str, MetaValue]]]:
    rng = np.random.default_rng(7)
    relations = ("father", "mother", "occupation")
    specs: list[tuple[str, str, str, Label, dict[str, MetaValue]]] = []
    for i in range(720):
        draw = rng.random()
        label: Label = (
            "known" if draw < 0.5 else "unknown" if draw < 0.9 else "uncertain"
        )
        meta: dict[str, MetaValue] = {
            "relation": relations[i % 3],
            "popularity": int(10 ** rng.uniform(1, 6)),
        }
        specs.append((f"popqa:{i}", "popqa", "real", label, meta))
    for i in range(240):
        meta = {"relation": relations[i % 3], "popularity": None}
        specs.append((f"synthal:{i}", "synthal", "fictional", "unknown", meta))
        specs.append(
            (f"generated:{i}", "generated", "fictional", "unknown", meta)
        )
    for award in range(16):
        for year in range(1990, 2025):
            kind = "real" if year <= 2012 else "post_cutoff"
            known = kind == "real" and rng.random() < 0.6
            meta = {"relation": f"Award{award}", "year": year}
            specs.append(
                (
                    f"wikidata:award:{award}:{year}",
                    "wikidata",
                    kind,
                    "known" if known else "unknown",
                    meta,
                )
            )
    for i in range(200):
        specs.append(
            (
                f"wikidata:detail:{i}",
                "wikidata",
                "unknowable_detail",
                "unknown",
                {"relation": "ordinary_date"},
            )
        )
        specs.append(
            (f"falseqa:{i}", "falseqa", "false_premise", "unknown", {})
        )
        for category in ("false assumption", "future unknown"):
            specs.append(
                (
                    f"kuq:{category}:{i}",
                    "kuq",
                    f"kuq:{category}",
                    "unknown",
                    {"category": category},
                )
            )
    for i in range(320):
        for source in ("triviaqa", "nq_open"):
            label = "known" if rng.random() < 0.5 else "unknown"
            specs.append((f"{source}:{i}", source, "open", label, {}))
    return specs


def _planted_items() -> list[LabeledItem]:
    specs = _specs()
    splits = assign_splits([s[0] for s in specs], dataset="planted")
    return [
        _labeled(uid, label, source=source, kind=kind, split=split, **meta)
        for (uid, source, kind, label, meta), split in zip(
            specs, splits, strict=True
        )
    ]


def _planted_activations(
    items: Sequence[LabeledItem],
) -> tuple[ItemActivations, NDArray[np.float64], NDArray[np.float64]]:
    """Direction A on every known item, B on every real pre-cutoff entity."""
    rng = np.random.default_rng(3)
    basis = np.linalg.qr(rng.standard_normal((D, 2)))[0]
    a, b = basis[:, 0], basis[:, 1]
    data = rng.standard_normal((len(LAYERS), len(items), D)).astype(np.float32)
    known = np.array([i.contrast_label == "known" for i in items])
    entity = np.array(
        [
            i.group == "popqa"
            or (i.group == "award" and i.example.kind == "real")
            for i in items
        ]
    )
    data[1:] += (
        SHIFT * known[:, None] * a[None, :]
        + SHIFT * entity[:, None] * b[None, :]
    ).astype(np.float32)
    uids = tuple(i.example.uid for i in items)
    return ItemActivations(uids, LAYERS, data), a, b


@dataclasses.dataclass
class Planted:
    items: list[LabeledItem]
    acts: ItemActivations
    pools: dict[str, TypePool]
    cube: ReadingCube
    a: NDArray[np.float64]
    b: NDArray[np.float64]


@pytest.fixture(scope="module")
def planted() -> Planted:
    items = _planted_items()
    acts, a, b = _planted_activations(items)
    pools = type_pools(items)
    cube = reading_cube(acts, pools, n_seeds=N_SEEDS, n_pairs=N_PAIRS, seed=5)
    return Planted(items, acts, pools, cube, a, b)


def _row(cube: ReadingCube, name: str) -> int:
    return cube.types.index(name)


def test_template_names_and_methods() -> None:
    assert TEMPLATE_NAMES == ("fewshot", "concept", "plain")
    assert READING_METHODS == ("logistic", "pca_diff", "mean_diff", "random")


def _tagged_chat() -> ChatFormat:
    def render(messages: list[dict[str, str]]) -> str:
        return f"<user>{messages[0]['content']}<assistant>"

    return ChatFormat("template", render_fn=render)


@pytest.mark.parametrize(
    "chat",
    [ChatFormat("plain"), ChatFormat("none"), _tagged_chat()],
    ids=["plain", "none", "template"],
)
def test_the_fewshot_template_is_the_labeling_prompt(chat: ChatFormat) -> None:
    template = p9_template("fewshot")

    for question in ("Who is Ann Lee?", "What is 2 + 2?"):
        assert template.render(question, "knowledge", chat).text == (
            few_shot_prompt(question)
        )


def test_the_fewshot_template_takes_the_shots() -> None:
    template = p9_template("fewshot", DEFAULT_SHOTS[:2])

    text = template.render("Who?", "knowledge", ChatFormat("plain")).text

    assert text == few_shot_prompt("Who?", DEFAULT_SHOTS[:2])
    assert text.count("Q:") == 3


def test_the_concept_template_is_the_paper_template() -> None:
    template = p9_template("concept")

    assert template is PAPER_TEMPLATES["concept"]
    text = template.render("Who?", "knowledge", ChatFormat("plain")).text
    assert text == (
        "USER: Consider the amount of knowledge in the following:\nWho?\n"
        "The amount of knowledge is ASSISTANT: "
    )
    assert text.endswith("ASSISTANT: ")


def test_the_concept_template_follows_the_chat_format() -> None:
    template = p9_template("concept")

    text = template.render("Who?", "knowledge", _tagged_chat()).text

    assert text.startswith("<user>Consider the amount of knowledge")
    assert text.endswith("<assistant>")


@pytest.mark.parametrize(
    "chat", [ChatFormat("plain"), _tagged_chat()], ids=["plain", "template"]
)
def test_the_plain_template_is_the_bare_question(chat: ChatFormat) -> None:
    template = p9_template("plain")

    assert template.render("Who is Ann?", "knowledge", chat).text == (
        "Who is Ann?"
    )


def test_an_unknown_template_name_raises() -> None:
    with pytest.raises(ValueError, match="verbose"):
        p9_template("verbose")


def test_render_items_renders_every_question_in_order() -> None:
    items = [_labeled("a"), _labeled("b")]

    texts = render_items(items, p9_template("plain"), ChatFormat("plain"))

    assert texts == ["question a", "question b"]


def test_render_items_fills_the_concept_word() -> None:
    items = [_labeled("a")]

    default = render_items(items, p9_template("concept"), ChatFormat("plain"))
    other = render_items(
        items, p9_template("concept"), ChatFormat("plain"), concept="truth"
    )

    assert "amount of knowledge in" in default[0]
    assert "amount of truth in" in other[0]


def test_render_items_rejects_a_function_template() -> None:
    with pytest.raises(ValueError, match="concept"):
        render_items(
            [_labeled("a")], PAPER_TEMPLATES["honesty"], ChatFormat("plain")
        )


def test_item_activations_rows_follow_the_items() -> None:
    items = [_labeled("a"), _labeled("b"), _labeled("c")]
    acts = ItemActivations(
        ("a", "b", "c"), (1,), np.zeros((1, 3, 4), dtype=np.float32)
    )

    rows = acts.rows([items[2], items[0]])

    assert rows.tolist() == [2, 0]
    assert rows.dtype == np.int64
    assert acts.rows([]).tolist() == []


def test_item_activations_rows_name_the_first_missing_uid() -> None:
    acts = ItemActivations(("a",), (1,), np.zeros((1, 1, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="x-1"):
        acts.rows([_labeled("a"), _labeled("x-1"), _labeled("x-2")])


def test_item_activations_rows_of_follow_the_uids() -> None:
    acts = ItemActivations(
        ("a", "b", "c"), (1,), np.zeros((1, 3, 4), dtype=np.float32)
    )

    rows = acts.rows_of(["c", "a"])

    assert rows.tolist() == [2, 0]
    assert rows.dtype == np.int64
    assert acts.rows_of([]).tolist() == []


def test_item_activations_rows_of_name_the_first_missing_uid() -> None:
    acts = ItemActivations(("a",), (1,), np.zeros((1, 1, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="'x-1'"):
        acts.rows_of(["a", "x-1", "x-2"])


def _small_items() -> list[LabeledItem]:
    return [
        _labeled("alpha", "known"),
        _labeled("beta", "unknown"),
        _labeled("gamma", "known"),
        _labeled("delta", "unknown"),
        _labeled("epsilon", "known"),
    ]


def test_collect_item_activations_has_one_row_per_item(
    tiny_arch: LoadedModel,
) -> None:
    items = _small_items()
    template = p9_template("fewshot", DEFAULT_SHOTS[:2])

    acts = collect_item_activations(
        tiny_arch, items, template, ChatFormat("plain"), layers=(1, 2)
    )

    assert acts.uids == tuple(i.example.uid for i in items)
    assert acts.layers == (1, 2)
    assert acts.data.shape == (2, len(items), tiny_arch.d_model)
    assert acts.data.dtype == np.float32
    expected = collect_activations(
        tiny_arch,
        render_items(items, template, ChatFormat("plain")),
        layers=(1, 2),
        position="last",
    )
    np.testing.assert_array_equal(acts.data, expected.data)


def test_collect_item_activations_defaults_to_every_layer(
    tiny_arch: LoadedModel,
) -> None:
    acts = collect_item_activations(
        tiny_arch,
        _small_items(),
        p9_template("fewshot", DEFAULT_SHOTS[:2]),
        ChatFormat("plain"),
        layers=None,
    )

    assert acts.layers == tuple(range(tiny_arch.n_layers + 1))
    assert acts.data.shape[0] == tiny_arch.n_layers + 1


def test_collect_item_activations_does_not_depend_on_the_batch_size(
    tiny_arch: LoadedModel,
) -> None:
    kwargs: dict[str, Any] = {"layers": (2,)}
    args = (
        tiny_arch,
        _small_items(),
        p9_template("fewshot", DEFAULT_SHOTS[:2]),
        ChatFormat("plain"),
    )

    one = collect_item_activations(*args, batch_size=1, **kwargs)
    many = collect_item_activations(*args, batch_size=8, **kwargs)

    np.testing.assert_allclose(one.data, many.data, rtol=1e-4, atol=1e-3)


def test_collect_item_activations_forwards_the_reading_settings(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    real = reading_module.collect_activations

    def spy(loaded: LoadedModel, texts: Sequence[str], **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real(loaded, texts, **kwargs)

    monkeypatch.setattr(reading_module, "collect_activations", spy)

    collect_item_activations(
        tiny_arch,
        _small_items(),
        p9_template("plain"),
        ChatFormat("plain"),
        layers=(1,),
        bos="none",
        batch_size=2,
    )

    assert seen == {
        "layers": (1,),
        "position": "last",
        "bos": "none",
        "batch_size": 2,
        "share_prefix": False,
    }


def test_collect_item_activations_shares_the_prefix_when_asked(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    real = reading_module.collect_activations

    def spy(loaded: LoadedModel, texts: Sequence[str], **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real(loaded, texts, **kwargs)

    monkeypatch.setattr(reading_module, "collect_activations", spy)
    args = (
        tiny_arch,
        _small_items(),
        p9_template("fewshot", DEFAULT_SHOTS[:2]),
        ChatFormat("plain"),
    )

    shared = collect_item_activations(*args, layers=(1, 2), share_prefix=True)

    assert seen["share_prefix"] is True
    plain = collect_item_activations(*args, layers=(1, 2))
    np.testing.assert_allclose(shared.data, plain.data, rtol=1e-4, atol=1e-3)


def _pool_of(
    train: tuple[int, int] = (12, 12),
    val: tuple[int, int] = (3, 3),
    test: tuple[int, int] = (3, 3),
) -> TypePool:
    positive: list[LabeledItem] = []
    negative: list[LabeledItem] = []
    for split, (n_pos, n_neg) in (
        ("train", train),
        ("val", val),
        ("test", test),
    ):
        positive += [
            _labeled(f"p-{split}-{i}", "known", split=split)  # type: ignore[arg-type]
            for i in range(n_pos)
        ]
        negative += [
            _labeled(f"n-{split}-{i}", "unknown", split=split)  # type: ignore[arg-type]
            for i in range(n_neg)
        ]
    return TypePool("t", tuple(positive), tuple(negative), "none", "knowledge")


def test_check_pool_accepts_a_usable_pool() -> None:
    assert check_pool(_pool_of(), min_pairs=4, min_eval=3) is None


def test_check_pool_names_a_short_train_side() -> None:
    reason = check_pool(_pool_of(train=(2, 9)), min_pairs=3, min_eval=3)

    assert reason == "fewer than 3 train items on one side"


def test_check_pool_checks_val_before_test() -> None:
    pool = _pool_of(val=(1, 5), test=(0, 5))

    reason = check_pool(pool, min_pairs=4, min_eval=2)

    assert reason == "fewer than 2 val items on one side"


def test_check_pool_names_a_short_test_side() -> None:
    reason = check_pool(_pool_of(test=(5, 1)), min_pairs=4, min_eval=2)

    assert reason == "fewer than 2 test items on one side"


def test_check_pool_checks_train_first() -> None:
    pool = _pool_of(train=(1, 1), val=(0, 0))

    assert check_pool(pool, min_pairs=2, min_eval=2) == (
        "fewer than 2 train items on one side"
    )


def _stratified_pool(train: dict[str, tuple[int, int]]) -> TypePool:
    positive: list[LabeledItem] = []
    negative: list[LabeledItem] = []
    for relation, (n_pos, n_neg) in train.items():
        positive += [
            _labeled(f"p-{relation}-{i}", "known", relation=relation)
            for i in range(n_pos)
        ]
        negative += [
            _labeled(f"n-{relation}-{i}", "unknown", relation=relation)
            for i in range(n_neg)
        ]
    for split in ("val", "test"):
        positive += [
            _labeled(f"p-{split}-{i}", "known", split=split, relation="a")  # type: ignore[arg-type]
            for i in range(3)
        ]
        negative += [
            _labeled(f"n-{split}-{i}", "unknown", split=split, relation="a")  # type: ignore[arg-type]
            for i in range(3)
        ]
    return TypePool(
        "t", tuple(positive), tuple(negative), "relation", "knowledge"
    )


def test_check_pool_counts_the_pairs_the_strata_allow() -> None:
    apart = _stratified_pool({"a": (6, 0), "b": (0, 6)})
    mixed = _stratified_pool({"a": (6, 10), "b": (8, 4), "c": (2, 2)})

    assert check_pool(apart, min_pairs=4, min_eval=3) == (
        "fewer than 4 matched train pairs"
    )
    assert check_pool(mixed, min_pairs=12, min_eval=3) is None
    assert check_pool(mixed, min_pairs=13, min_eval=3) == (
        "fewer than 13 matched train pairs"
    )


def test_check_pool_names_the_short_side_before_the_matched_pairs() -> None:
    pool = _stratified_pool({"a": (1, 9), "b": (0, 9)})

    assert check_pool(pool, min_pairs=3, min_eval=3) == (
        "fewer than 3 train items on one side"
    )


def _uids_in_half(prefix: str, half: int, n: int) -> list[str]:
    uids: list[str] = []
    i = 0
    while len(uids) < n:
        uid = f"{prefix}-{i}"
        if reading_module._half_of(uid) == half:
            uids.append(uid)
        i += 1
    return uids


def test_check_pool_names_a_train_half_that_cannot_be_fitted() -> None:
    base = _pool_of(train=(0, 0))
    lopsided = dataclasses.replace(
        base,
        positive=(
            *(_labeled(u, "known") for u in _uids_in_half("p", 0, 8)),
            *base.positive,
        ),
        negative=(
            *(_labeled(u, "unknown") for u in _uids_in_half("n", 0, 8)),
            *base.negative,
        ),
    )

    assert check_pool(lopsided, min_pairs=4, min_eval=3) == (
        "fewer than 2 matched train pairs in one split half"
    )


def test_planted_pools_are_all_usable(planted: Planted) -> None:
    assert len(planted.pools) == 13
    for name, pool in planted.pools.items():
        assert check_pool(pool, min_pairs=8, min_eval=5) is None, name


def _spy_fit(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    real = RepReader.fit.__func__  # type: ignore[attr-defined]

    def spy(cls: type[RepReader], method: str, h: Any, layers: Any, **kw: Any):  # type: ignore[no-untyped-def]
        calls.append({"method": method, "h": h, "layers": layers, **kw})
        return real(cls, method, h, layers, **kw)

    monkeypatch.setattr(RepReader, "fit", classmethod(spy))
    return calls


def _uid_of_column(acts: ItemActivations, h: NDArray[np.float32]) -> list[str]:
    lookup = {acts.data[0, j].tobytes(): uid for j, uid in enumerate(acts.uids)}
    return [lookup[h[0, j].tobytes()] for j in range(h.shape[1])]


def test_fit_type_reader_returns_a_reader_over_the_layers(
    planted: Planted,
) -> None:
    reader = fit_type_reader(
        planted.acts, planted.pools["obscure_real"], n_pairs=16, seed=1
    )

    assert isinstance(reader, RepReader)
    assert reader.method == "logistic"
    assert reader.layers == LAYERS
    assert reader.directions.shape == (len(LAYERS), D)
    assert reader.signs.shape == (len(LAYERS),)


def test_fit_type_reader_scores_the_positive_side_higher(
    planted: Planted,
) -> None:
    pool = planted.pools["unknowable_detail"]
    reader = fit_type_reader(planted.acts, pool, n_pairs=32, seed=1)

    for split in ("val", "test"):
        positive, negative = pool.split(split)  # type: ignore[arg-type]
        rows = planted.acts.rows([*positive, *negative])
        scores = reader.score(planted.acts.data[:, rows])
        n_pos = len(positive)
        for k in SIGNAL:
            assert scores[k, :n_pos].mean() > scores[k, n_pos:].mean()


@pytest.mark.parametrize("method", ["logistic", "pca_diff", "mean_diff"])
def test_fit_type_reader_finds_the_planted_direction(
    planted: Planted, method: Any
) -> None:
    reader = fit_type_reader(
        planted.acts,
        planted.pools["obscure_real"],
        method=method,
        n_pairs=64,
        seed=2,
    )

    signed = reader.directions * reader.signs[:, None]
    for k in SIGNAL:
        assert signed[k] @ planted.a > 0.9
        assert abs(signed[k] @ planted.b) < 0.3


def test_fit_type_reader_trains_on_train_items_only(
    planted: Planted, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy_fit(monkeypatch)
    pool = planted.pools["obscure_real"]

    fit_type_reader(planted.acts, pool, n_pairs=None, seed=0)

    (call,) = calls
    by_uid = {i.example.uid: i for i in planted.items}
    uids = _uid_of_column(planted.acts, call["h"])
    assert {by_uid[u].example.split for u in uids} == {"train"}


def test_fit_type_reader_pairs_known_with_unknown_inside_a_stratum(
    planted: Planted, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy_fit(monkeypatch)
    pool = planted.pools["obscure_real_pop"]

    fit_type_reader(planted.acts, pool, n_pairs=None, seed=0)

    (call,) = calls
    by_uid = {i.example.uid: i for i in planted.items}
    uids = _uid_of_column(planted.acts, call["h"])
    pairs, labels = call["pairs"], call["labels"]
    assert pairs.shape[1] == 2
    assert np.array_equal(call["groups"], pairs)
    names = {u: stratum_name(by_uid[u], pool.strata) for u in uids}
    for first, second in pairs:
        assert labels[first] != labels[second]
        assert names[uids[first]] == names[uids[second]]
    assert sorted(pairs.ravel().tolist()) == list(range(len(uids)))


def test_fit_type_reader_keeps_the_first_n_pairs(
    planted: Planted, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy_fit(monkeypatch)
    pool = planted.pools["false_premise"]

    fit_type_reader(planted.acts, pool, n_pairs=None, seed=0)
    fit_type_reader(planted.acts, pool, n_pairs=7, seed=0)
    fit_type_reader(planted.acts, pool, n_pairs=10_000, seed=0)

    every, seven, many = (c["pairs"].shape[0] for c in calls)
    assert seven == 7
    assert every > 7
    assert many == every


def test_fit_type_reader_forwards_method_preprocess_and_seed(
    planted: Planted, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy_fit(monkeypatch)

    fit_type_reader(
        planted.acts,
        planted.pools["false_premise"],
        method="mean_diff",
        preprocess="unit",
        n_pairs=8,
        seed=11,
    )

    (call,) = calls
    assert call["method"] == "mean_diff"
    assert call["preprocess"] == "unit"
    assert call["seed"] == 11
    assert tuple(call["layers"]) == LAYERS


def test_fit_type_reader_is_reproducible_and_depends_on_the_seed(
    planted: Planted,
) -> None:
    pool = planted.pools["obscure_real"]

    one = fit_type_reader(planted.acts, pool, n_pairs=16, seed=4)
    again = fit_type_reader(planted.acts, pool, n_pairs=16, seed=4)
    other = fit_type_reader(planted.acts, pool, n_pairs=16, seed=5)

    np.testing.assert_array_equal(one.directions, again.directions)
    assert not np.array_equal(one.directions, other.directions)


def test_draw_train_holds_the_train_items_the_pairs_use(
    planted: Planted,
) -> None:
    pool = planted.pools["obscure_real_pop"]
    positive, negative = pool.split("train")
    sides = {i.example.uid: 1 for i in positive}
    sides |= {i.example.uid: 0 for i in negative}

    draw = draw_train(pool, n_pairs=16, seed=4)

    assert isinstance(draw, TrainDraw)
    assert draw.pairs is not None
    assert draw.pairs.shape == (16, 2)
    assert sorted(draw.pairs.ravel().tolist()) == list(range(len(draw.items)))
    assert len(draw.items) == 32
    uids = [i.example.uid for i in draw.items]
    assert draw.labels.tolist() == [sides[u] for u in uids]
    assert draw.labels.dtype == np.int64
    names = [stratum_name(i, pool.strata) for i in draw.items]
    assert len(draw.strata) == len(uids)
    assert len(set(zip(names, draw.strata.tolist(), strict=True))) == len(
        set(names)
    )
    for first, second in draw.pairs:
        assert draw.labels[first] != draw.labels[second]
        assert draw.strata[first] == draw.strata[second]


def test_draw_train_depends_on_the_seed_only(planted: Planted) -> None:
    pool = planted.pools["obscure_real"]

    one = draw_train(pool, n_pairs=16, seed=4)
    again = draw_train(pool, n_pairs=16, seed=4)
    other = draw_train(pool, n_pairs=16, seed=5)

    assert [i.example.uid for i in one.items] == [
        i.example.uid for i in again.items
    ]
    np.testing.assert_array_equal(one.pairs, again.pairs)
    assert [i.example.uid for i in one.items] != [
        i.example.uid for i in other.items
    ]


def test_fit_type_reader_needs_only_the_items_the_pairs_hold(
    planted: Planted,
) -> None:
    pool = planted.pools["obscure_real"]
    draw = draw_train(pool, n_pairs=16, seed=4)
    held = planted.acts.rows(draw.items)
    partial = ItemActivations(
        tuple(i.example.uid for i in draw.items),
        planted.acts.layers,
        planted.acts.data[:, held],
    )

    reader = fit_type_reader(partial, pool, n_pairs=16, seed=4)

    full = fit_type_reader(planted.acts, pool, n_pairs=16, seed=4)
    np.testing.assert_array_equal(reader.directions, full.directions)


def test_fit_draw_without_pairs_orients_on_the_labels(
    planted: Planted, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy_fit(monkeypatch)
    draw = draw_train(planted.pools["obscure_real"], n_pairs=16, seed=4)
    flipped = dataclasses.replace(draw, labels=1 - draw.labels, pairs=None)

    reader = fit_draw(planted.acts, flipped, seed=4)

    (call,) = calls
    assert call["pairs"] is None
    assert call["groups"] is None
    np.testing.assert_array_equal(call["labels"], 1 - draw.labels)
    assert reader.layers == LAYERS


def test_type_scores_hold_the_scores_and_labels_the_auroc_ranks(
    planted: Planted,
) -> None:
    pool = planted.pools["obscure_real"]
    reader = fit_type_reader(planted.acts, pool, n_pairs=16, seed=1)
    positive, negative = pool.split("test")

    scores, labels = type_scores(reader, planted.acts, pool, "test")

    assert scores.shape == (len(LAYERS), len(positive) + len(negative))
    assert labels.tolist() == [1] * len(positive) + [0] * len(negative)
    assert labels.dtype == np.int64
    np.testing.assert_array_equal(
        auroc(scores, labels), type_auroc(reader, planted.acts, pool, "test")
    )


def test_type_auroc_ranks_the_positive_side_on_one_split(
    planted: Planted,
) -> None:
    pool = planted.pools["obscure_real"]
    direction = np.tile(planted.a.astype(np.float32), (len(LAYERS), 1))
    reader = RepReader.from_arrays(
        "planted", LAYERS, direction, np.zeros_like(direction)
    )

    for split in ("val", "test"):
        values = type_auroc(reader, planted.acts, pool, split)  # type: ignore[arg-type]
        assert values.shape == (len(LAYERS),)
        assert values[1] > 0.95
        assert values[2] > 0.95
        assert abs(values[0] - 0.5) < 0.2
        positive, negative = pool.split(split)  # type: ignore[arg-type]
        rows = planted.acts.rows([*positive, *negative])
        labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
        expected = auroc(
            reader.score(planted.acts.data[:, rows]), labels.astype(np.int64)
        )
        np.testing.assert_allclose(values, expected)


def test_type_auroc_of_an_inverted_reader_is_one_minus_the_auroc(
    planted: Planted,
) -> None:
    pool = planted.pools["obscure_real"]
    direction = np.tile(planted.a.astype(np.float32), (len(LAYERS), 1))
    means = np.zeros_like(direction)
    forward = RepReader.from_arrays("planted", LAYERS, direction, means)
    backward = RepReader.from_arrays("planted", LAYERS, -direction, means)

    up = type_auroc(forward, planted.acts, pool, "test")
    down = type_auroc(backward, planted.acts, pool, "test")

    np.testing.assert_allclose(up + down, 1.0)


def test_type_auroc_without_one_side_raises(planted: Planted) -> None:
    pool = planted.pools["obscure_real"]
    only_positive = dataclasses.replace(pool, negative=())
    reader = fit_type_reader(planted.acts, pool, n_pairs=8, seed=0)

    with pytest.raises(ValueError, match="both classes"):
        type_auroc(reader, planted.acts, only_positive, "test")


def test_the_cube_has_the_documented_axes(planted: Planted) -> None:
    cube = planted.cube
    n_types, n_layers = len(planted.pools), len(LAYERS)

    assert cube.types == tuple(planted.pools)
    assert cube.methods == READING_METHODS
    assert cube.layers == LAYERS
    assert cube.val.shape == (4, n_types, N_SEEDS, n_layers)
    assert cube.test.shape == cube.val.shape
    assert cube.transfer.shape == (N_SEEDS, n_layers, n_types, n_types)
    assert cube.cosine.shape == (n_layers, n_types, n_types)
    assert list(cube.readers) == list(cube.types)
    for reader in cube.readers.values():
        assert reader.method == "logistic"
        assert reader.layers == LAYERS


def test_the_supervised_methods_read_each_contrast_at_the_signal_layers(
    planted: Planted,
) -> None:
    cube = planted.cube

    for m in range(3):
        for split in (cube.val, cube.test):
            at_signal = split[m][:, :, SIGNAL].mean(axis=1)
            at_noise = split[m][:, :, 0].mean(axis=1)
            assert at_signal.min() > 0.9, cube.methods[m]
            assert at_noise.max() < 0.75, cube.methods[m]


def test_the_random_series_is_the_floor(planted: Planted) -> None:
    cube = planted.cube

    for k in SIGNAL:
        gap = cube.test[0][:, :, k].mean() - cube.test[3][:, :, k].mean()
        assert gap >= 0.15


def test_transfer_follows_the_planted_directions(planted: Planted) -> None:
    cube = planted.cube
    k = 2
    mean = cube.transfer[:, k].mean(axis=0)
    rows = {name: _row(cube, name) for name in cube.types}

    for group in (A_GROUP, B_GROUP):
        for first in group:
            for second in group:
                assert mean[rows[first], rows[second]] > 0.9, (first, second)
    for first in A_GROUP:
        for second in B_GROUP:
            assert abs(mean[rows[first], rows[second]] - 0.5) < 0.2
            assert abs(mean[rows[second], rows[first]] - 0.5) < 0.2
    for mixed in MIXED:
        for name in cube.types:
            assert mean[rows[mixed], rows[name]] > 0.5
            assert mean[rows[name], rows[mixed]] > 0.5


def test_the_transfer_diagonal_is_the_test_auroc(planted: Planted) -> None:
    cube = planted.cube

    for t in range(len(cube.types)):
        np.testing.assert_allclose(
            cube.transfer[:, :, t, t], cube.test[0, t], atol=1e-12
        )


def test_cosines_follow_the_planted_directions(planted: Planted) -> None:
    cube = planted.cube
    k = 2
    rows = {name: _row(cube, name) for name in cube.types}

    np.testing.assert_allclose(cube.cosine[k], cube.cosine[k].T)
    for group in (A_GROUP, B_GROUP):
        for first in group:
            for second in group:
                assert cube.cosine[k, rows[first], rows[second]] > 0.75
    for first in A_GROUP:
        for second in B_GROUP:
            assert abs(cube.cosine[k, rows[first], rows[second]]) < 0.3


def test_the_cosines_are_symmetric_and_bounded(planted: Planted) -> None:
    cosine = planted.cube.cosine

    np.testing.assert_allclose(cosine, cosine.transpose(0, 2, 1))
    assert np.abs(cosine).max() <= 1.0


def test_the_diagonal_is_the_split_half_stability(planted: Planted) -> None:
    cube = planted.cube

    for t in range(len(cube.types)):
        for k in SIGNAL:
            assert cube.cosine[k, t, t] > 0.7
        assert abs(cube.cosine[0, t, t]) < 0.3


def _noise_case(
    seed: int, d: int = 1024, n_side: int = 60
) -> tuple[ItemActivations, dict[str, TypePool]]:
    """Pure noise, with the known side shared by contrasts A and B."""
    splits: tuple[Split, ...] = ("train", "train", "train", "val", "test")

    def side(prefix: str, label: Label) -> tuple[LabeledItem, ...]:
        return tuple(
            _labeled(f"{prefix}-{i}", label, split=splits[i % 5])
            for i in range(n_side)
        )

    shared, a, b = (
        side("x", "known"),
        side("a", "unknown"),
        side("b", "unknown"),
    )
    c, e = side("c", "known"), side("e", "unknown")
    pools = {
        "A": TypePool("A", shared, a, "none", "knowledge"),
        "B": TypePool("B", shared, b, "none", "knowledge"),
        "C": TypePool("C", c, e, "none", "knowledge"),
    }
    items = [*shared, *a, *b, *c, *e]
    rng = np.random.default_rng(100 + seed)
    data = rng.standard_normal((2, len(items), d)).astype(np.float32)
    uids = tuple(i.example.uid for i in items)
    return ItemActivations(uids, (1, 2), data), pools


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_contrasts_that_share_a_side_are_not_similar_on_noise(
    seed: int,
) -> None:
    acts, pools = _noise_case(seed)

    cube = reading_cube(
        acts, pools, methods=("logistic",), n_seeds=1, n_pairs=16, seed=seed
    )

    assert np.abs(cube.cosine[:, 0, 1]).max() < 0.1
    assert np.abs(cube.cosine[:, 0, 2]).max() < 0.1


def test_identical_contrasts_have_identical_cosines() -> None:
    acts, pools = _noise_case(0)
    twin = {"A": pools["A"], "A2": dataclasses.replace(pools["A"], name="A2")}

    cube = reading_cube(
        acts, twin, methods=("logistic",), n_seeds=1, n_pairs=16, seed=0
    )

    np.testing.assert_array_equal(cube.cosine[:, 0, 0], cube.cosine[:, 0, 1])
    np.testing.assert_array_equal(cube.cosine[:, 0, 0], cube.cosine[:, 1, 1])


def test_the_cosines_use_the_signed_directions(
    planted: Planted, monkeypatch: pytest.MonkeyPatch
) -> None:
    fitted: list[RepReader] = []
    real = reading_module.fit_type_reader

    def spy(*args: Any, **kw: Any) -> RepReader:
        reader = real(*args, **kw)
        if kw["n_pairs"] is None:
            fitted.append(reader)
        return reader

    monkeypatch.setattr(reading_module, "fit_type_reader", spy)
    names = ("obscure_real", "real_vs_synthal", "post_cutoff")
    pools = {name: planted.pools[name] for name in names}

    cube = reading_cube(
        planted.acts,
        pools,
        methods=("pca_diff",),
        n_seeds=1,
        n_pairs=16,
        seed=0,
    )

    n_types = len(names)
    first, second = fitted[-2 * n_types : -n_types], fitted[-n_types:]
    for half in (first, second):
        assert any((r.signs == -1).any() for r in half)
    for k in range(len(LAYERS)):
        a = np.stack([r.directions[k] * r.signs[k] for r in first])
        b = np.stack([r.directions[k] * r.signs[k] for r in second])
        cross = (a / np.linalg.norm(a, axis=1, keepdims=True)) @ (
            b / np.linalg.norm(b, axis=1, keepdims=True)
        ).T
        np.testing.assert_allclose(
            cube.cosine[k], (cross + cross.T) / 2, atol=1e-6
        )


def test_draw_seed_is_the_first_word_of_the_seed_sequence() -> None:
    expected = np.random.SeedSequence([7, 3]).generate_state(1)[0]

    assert draw_seed(7, 3) == int(expected)
    assert len({draw_seed(7, d) for d in range(8)}) == 8
    assert draw_seed(7, 3) != draw_seed(8, 3)


@pytest.mark.parametrize("seed", [0, 5])
def test_the_half_fits_do_not_reuse_a_draw_seed(
    planted: Planted, monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    calls: list[tuple[int | None, int]] = []
    real = reading_module.fit_type_reader

    def spy(*args: Any, **kw: Any) -> RepReader:
        calls.append((kw["n_pairs"], kw["seed"]))
        return real(*args, **kw)

    monkeypatch.setattr(reading_module, "fit_type_reader", spy)
    names = ("obscure_real", "real_vs_synthal")
    pools = {name: planted.pools[name] for name in names}

    reading_cube(
        planted.acts,
        pools,
        methods=("logistic",),
        n_seeds=3,
        n_pairs=16,
        seed=seed,
    )

    draw_seeds = {s for n_pairs, s in calls if n_pairs is not None}
    all_pairs = [s for n_pairs, s in calls if n_pairs is None]
    half_seeds = all_pairs[len(names) :]
    assert len(draw_seeds) == 3
    assert len(set(half_seeds)) == 2
    assert not draw_seeds & {*half_seeds, seed}


def test_the_cosines_are_deterministic_under_the_seed() -> None:
    acts, pools = _noise_case(0, d=64)

    def cosines(seed: int) -> NDArray[np.float64]:
        cube = reading_cube(
            acts, pools, methods=("logistic",), n_seeds=1, n_pairs=16, seed=seed
        )
        return cube.cosine

    np.testing.assert_array_equal(cosines(3), cosines(3))


def test_the_cosines_fit_on_train_items_only(
    planted: Planted, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy_fit(monkeypatch)
    pools = {
        name: planted.pools[name]
        for name in ("obscure_real", "real_vs_synthal")
    }

    reading_cube(
        planted.acts, pools, methods=("logistic",), n_seeds=1, n_pairs=8, seed=0
    )

    by_uid = {i.example.uid: i for i in planted.items}
    assert len(calls) == 8
    for call in calls:
        uids = _uid_of_column(planted.acts, call["h"])
        assert {by_uid[u].example.split for u in uids} == {"train"}


def test_the_readers_are_fitted_on_all_train_pairs_with_the_run_seed(
    planted: Planted,
) -> None:
    for name in ("obscure_real", "real_vs_generated"):
        expected = fit_type_reader(
            planted.acts, planted.pools[name], n_pairs=None, seed=5
        )
        got = planted.cube.readers[name]
        np.testing.assert_array_equal(got.directions, expected.directions)
        np.testing.assert_array_equal(got.signs, expected.signs)


def test_every_draw_is_a_fit_with_a_seed_derived_from_the_run_seed(
    planted: Planted,
) -> None:
    name, t = "unknowable_detail", _row(planted.cube, "unknowable_detail")
    for s in range(N_SEEDS):
        state = int(np.random.SeedSequence([5, s]).generate_state(1)[0])
        for m, method in enumerate(READING_METHODS):
            reader = fit_type_reader(
                planted.acts,
                planted.pools[name],
                method=method,
                n_pairs=N_PAIRS,
                seed=state,
            )
            for split, cube_split in (
                ("val", planted.cube.val),
                ("test", planted.cube.test),
            ):
                np.testing.assert_allclose(
                    cube_split[m, t, s],
                    type_auroc(
                        reader, planted.acts, planted.pools[name], split
                    ),  # type: ignore[arg-type]
                )
            if m == 0:
                for u, other in enumerate(planted.cube.types):
                    np.testing.assert_allclose(
                        planted.cube.transfer[s, :, t, u],
                        type_auroc(
                            reader, planted.acts, planted.pools[other], "test"
                        ),
                    )


def test_each_fitted_reader_scores_the_held_out_rows_once(
    planted: Planted, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    real = RepReader.score

    def spy(self: RepReader, h: Any) -> Any:
        seen.append(h)
        return real(self, h)

    monkeypatch.setattr(RepReader, "score", spy)
    names = ("obscure_real", "real_vs_synthal")
    pools = {name: planted.pools[name] for name in names}

    reading_cube(
        planted.acts,
        pools,
        methods=("logistic", "pca_diff"),
        n_seeds=2,
        n_pairs=16,
        seed=0,
    )

    held_out = {
        i.example.uid
        for pool in pools.values()
        for i in (*pool.positive, *pool.negative)
        if i.example.split != "train"
    }
    assert len(seen) == len(names) * 2 * 2
    assert all(h is seen[0] for h in seen)
    assert seen[0].shape[1] == len(held_out)


def test_the_draws_differ_and_the_cube_is_reproducible(
    planted: Planted,
) -> None:
    names = ("obscure_real", "real_vs_synthal")
    pools = {name: planted.pools[name] for name in names}
    kwargs: dict[str, Any] = {
        "methods": ("logistic", "pca_diff"),
        "n_seeds": 2,
        "n_pairs": 16,
    }

    one = reading_cube(planted.acts, pools, seed=1, **kwargs)
    again = reading_cube(planted.acts, pools, seed=1, **kwargs)
    other = reading_cube(planted.acts, pools, seed=2, **kwargs)

    assert one.methods == ("logistic", "pca_diff")
    np.testing.assert_array_equal(one.val, again.val)
    np.testing.assert_array_equal(one.test, again.test)
    np.testing.assert_array_equal(one.transfer, again.transfer)
    np.testing.assert_array_equal(one.cosine, again.cosine)
    assert not np.array_equal(one.test[0, :, 0, 0], one.test[0, :, 1, 0])
    assert not np.array_equal(one.test, other.test)


def test_the_first_method_is_the_headline(planted: Planted) -> None:
    pools = {"obscure_real": planted.pools["obscure_real"]}

    cube = reading_cube(
        planted.acts,
        pools,
        methods=("pca_diff", "logistic"),
        n_seeds=2,
        n_pairs=16,
        seed=1,
    )

    assert {reader.method for reader in cube.readers.values()} == {"pca_diff"}
    np.testing.assert_allclose(cube.transfer[:, :, 0, 0], cube.test[0, 0])


def test_the_preprocess_reaches_the_fits(
    planted: Planted, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy_fit(monkeypatch)
    pools = {"false_premise": planted.pools["false_premise"]}

    reading_cube(
        planted.acts,
        pools,
        methods=("logistic",),
        preprocess="unit",
        n_seeds=1,
        n_pairs=8,
        seed=0,
    )

    assert calls
    assert {call["preprocess"] for call in calls} == {"unit"}


def test_a_single_contrast_has_a_one_by_one_transfer(planted: Planted) -> None:
    pools = {"false_premise": planted.pools["false_premise"]}

    cube = reading_cube(
        planted.acts, pools, methods=("logistic",), n_seeds=1, n_pairs=8
    )

    assert cube.transfer.shape == (1, len(LAYERS), 1, 1)
    assert cube.cosine.shape == (len(LAYERS), 1, 1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"methods": ()}, "methods"),
        ({"n_seeds": 0}, "n_seeds"),
    ],
)
def test_reading_cube_rejects_empty_settings(
    planted: Planted, kwargs: dict[str, Any], match: str
) -> None:
    pools = {"false_premise": planted.pools["false_premise"]}

    with pytest.raises(ValueError, match=match):
        reading_cube(planted.acts, pools, **kwargs)


def test_reading_cube_rejects_no_contrast(planted: Planted) -> None:
    with pytest.raises(ValueError, match="contrast"):
        reading_cube(planted.acts, {})


def test_choose_layer_picks_the_best_mean_validation_layer(
    planted: Planted,
) -> None:
    assert choose_layer(planted.cube.val, planted.cube.layers) == 2


def test_choose_layer_returns_the_layer_index_not_the_position() -> None:
    val = np.zeros((2, 2, 3, 3))
    val[0, :, :, 1] = 0.9
    val[0, :, :, 0] = 0.6

    assert choose_layer(val, (4, 8, 12)) == 8


def test_choose_layer_averages_over_contrasts_and_draws() -> None:
    val = np.zeros((1, 2, 2, 2))
    val[0, 0, :, 0] = 1.0
    val[0, 1, :, 0] = 0.5
    val[0, 0, :, 1] = 0.9
    val[0, 1, :, 1] = 0.9

    assert choose_layer(val, (1, 2)) == 2


def test_choose_layer_breaks_a_tie_towards_the_first_layer() -> None:
    val = np.full((1, 2, 2, 3), 0.5)
    val[0, :, :, 1:] = 0.8

    assert choose_layer(val, (5, 6, 7)) == 6


def test_choose_layer_reads_the_first_method_only() -> None:
    val = np.zeros((2, 1, 1, 2))
    val[0, 0, 0] = [0.9, 0.5]
    val[1, 0, 0] = [0.1, 1.0]

    assert choose_layer(val, (1, 2)) == 1


def _baseline_pool(
    *,
    test_popularity: tuple[MetaValue, MetaValue] = (900, 12),
    train_popularity: MetaValue = 50,
    train_positive_popularity: MetaValue = 500,
) -> TypePool:
    def make(
        uid: str, question: str, label: Label, split: Split, popularity: Any
    ) -> LabeledItem:
        item = _labeled(uid, label, split=split, popularity=popularity)
        return dataclasses.replace(
            item, example=dataclasses.replace(item.example, question=question)
        )

    positive: list[LabeledItem] = []
    negative: list[LabeledItem] = []
    for split, n in (("train", 12), ("val", 4), ("test", 6)):
        for i in range(n):
            popularity_pos = (
                test_popularity[0]
                if split == "test"
                else train_positive_popularity
            )
            popularity_neg = (
                test_popularity[1] if split == "test" else train_popularity
            )
            positive.append(
                make(
                    f"p-{split}-{i}",
                    "who wrote the alpha novel",
                    "known",
                    split,  # type: ignore[arg-type]
                    popularity_pos,
                )
            )
            negative.append(
                make(
                    f"n-{split}-{i}",
                    "who wrote the beta novel",
                    "unknown",
                    split,  # type: ignore[arg-type]
                    popularity_neg,
                )
            )
    return TypePool("t", tuple(positive), tuple(negative), "none", "knowledge")


def test_surface_baselines_name_the_three_columns_in_order() -> None:
    assert list(surface_baselines(_baseline_pool()))[:3] == [
        "text_word",
        "text_char",
        "popularity",
    ]


def test_surface_baselines_read_a_one_word_difference() -> None:
    baselines = surface_baselines(_baseline_pool())

    assert baselines["text_word"] == pytest.approx(1.0)
    assert baselines["text_char"] == pytest.approx(1.0)
    assert baselines["popularity"] == pytest.approx(1.0)


def test_surface_baselines_fit_on_all_train_items_and_score_the_test_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _baseline_pool()
    calls: list[tuple[list[str], list[int], list[str], str]] = []
    real = reading_module.text_scores

    def spy(
        train_texts: Sequence[str],
        train_labels: NDArray[np.int64],
        test_texts: Sequence[str],
        *,
        analyzer: Any,
    ) -> NDArray[np.float64]:
        calls.append(
            (
                list(train_texts),
                train_labels.tolist(),
                list(test_texts),
                analyzer,
            )
        )
        return real(train_texts, train_labels, test_texts, analyzer=analyzer)

    monkeypatch.setattr(reading_module, "text_scores", spy)

    surface_baselines(pool)

    positive_train, negative_train = pool.split("train")
    positive_test, negative_test = pool.split("test")
    want_train = [
        i.example.question for i in (*positive_train, *negative_train)
    ]
    want_test = [i.example.question for i in (*positive_test, *negative_test)]
    assert [call[3] for call in calls] == ["word", "char"]
    for train_texts, train_labels, test_texts, _ in calls:
        assert train_texts == want_train
        assert train_labels == [1] * 12 + [0] * 12
        assert test_texts == want_test


def test_surface_baselines_popularity_is_null_when_a_test_item_has_none() -> (
    None
):
    pool = _baseline_pool(test_popularity=(900, None))

    baselines = surface_baselines(pool)

    assert baselines["popularity"] is None
    assert baselines["text_word"] == pytest.approx(1.0)


def test_surface_baselines_popularity_is_null_for_a_non_integer() -> None:
    pool = _baseline_pool(test_popularity=(900, 12.5))

    assert surface_baselines(pool)["popularity"] is None


def test_surface_baselines_popularity_is_null_without_the_field() -> None:
    pool = _pool_of()

    assert surface_baselines(pool)["popularity"] is None


def test_surface_baselines_popularity_reads_the_test_items_only() -> None:
    pool = _baseline_pool(train_popularity=None)

    assert surface_baselines(pool)["popularity"] == pytest.approx(1.0)


def test_surface_baselines_popularity_is_not_oriented_on_the_train_items() -> (
    None
):
    pool = _baseline_pool(train_positive_popularity=10, train_popularity=5000)

    assert surface_baselines(pool)["popularity"] == pytest.approx(1.0)


CAPITAL = "capital"
OCCUPATION = "occupation"
RELATION_COUNTS: dict[Split, tuple[int, int]] = {
    "train": (40, 8),
    "val": (20, 4),
    "test": (60, 6),
}
RELATION_POPULARITY = {CAPITAL: 100_000, OCCUPATION: 100}
RELATIONAL_TYPES = [
    "fictional_synthal",
    "fictional_generated",
    "obscure_real",
    "obscure_real_pop",
    "post_cutoff",
    "real_vs_synthal",
    "real_vs_generated",
    "pre_vs_post_cutoff",
]
# One test split of the confounded pool holds 66 x 66 pairs: 60 x 60 rank
# capital above occupation, and the 60 x 6 + 6 x 60 inside a relation tie.
POOLED_BASE_RATE_AUROC = (60 * 60 + (60 * 6 + 6 * 60) / 2) / (66 * 66)


def _relation_item(
    uid: str, label: Label, split: Split, relation: str
) -> LabeledItem:
    item = _labeled(
        uid,
        label,
        split=split,
        relation=relation,
        popularity=RELATION_POPULARITY[relation],
    )
    question = f"what is the {relation} of this person"
    return dataclasses.replace(
        item, example=dataclasses.replace(item.example, question=question)
    )


def _confounded_pool(*, apart: Sequence[Split] = ()) -> TypePool:
    """Capital questions are mostly known and occupation ones mostly unknown.

    The wording and the popularity of a question depend on its relation alone,
    so each of them separates the sides through the relation only. In a split
    of `apart` every positive item is a capital one and every negative item an
    occupation one, so the sides share no relation there.
    """
    positive: list[LabeledItem] = []
    negative: list[LabeledItem] = []
    for split, (major, minor) in RELATION_COUNTS.items():
        if split in apart:
            minor = 0
        for relation, n_positive, n_negative in (
            (CAPITAL, major, minor),
            (OCCUPATION, minor, major),
        ):
            positive += [
                _relation_item(
                    f"p-{split}-{relation}-{i}", "known", split, relation
                )
                for i in range(n_positive)
            ]
            negative += [
                _relation_item(
                    f"n-{split}-{relation}-{i}", "unknown", split, relation
                )
                for i in range(n_negative)
            ]
    return TypePool(
        "confounded", tuple(positive), tuple(negative), "none", "knowledge"
    )


def _relation_activations(pool: TypePool) -> ItemActivations:
    """Layer 1 is noise; layer 2 holds the relation of the item alone."""
    items = [*pool.positive, *pool.negative]
    rng = np.random.default_rng(11)
    data = np.zeros((2, len(items), 16), dtype=np.float32)
    data[0] = rng.standard_normal((len(items), 16))
    data[1, [relation_of(i) == CAPITAL for i in items], 0] = SHIFT
    return ItemActivations(tuple(i.example.uid for i in items), (1, 2), data)


def _map_items(
    pool: TypePool, change: Callable[[LabeledItem], LabeledItem]
) -> TypePool:
    return dataclasses.replace(
        pool,
        positive=tuple(change(i) for i in pool.positive),
        negative=tuple(change(i) for i in pool.negative),
    )


def _with_meta(item: LabeledItem, **meta: MetaValue) -> LabeledItem:
    example = dataclasses.replace(
        item.example, meta={**item.example.meta, **meta}
    )
    return dataclasses.replace(item, example=example)


def _without_relation(item: LabeledItem) -> LabeledItem:
    meta = {k: v for k, v in item.example.meta.items() if k != "relation"}
    return dataclasses.replace(
        item, example=dataclasses.replace(item.example, meta=meta)
    )


def test_relation_groups_follow_the_items_of_a_split() -> None:
    pool = _confounded_pool()

    test = relation_groups(pool, "test")
    val = relation_groups(pool, "val")

    assert test is not None
    assert test.tolist() == (
        [CAPITAL] * 60 + [OCCUPATION] * 6 + [CAPITAL] * 6 + [OCCUPATION] * 60
    )
    assert val is not None
    assert val.tolist() == (
        [CAPITAL] * 20 + [OCCUPATION] * 4 + [CAPITAL] * 4 + [OCCUPATION] * 20
    )


@pytest.mark.parametrize("split", ["val", "test"])
def test_relation_groups_need_a_relation_on_every_eval_item(
    split: Split,
) -> None:
    pool = _confounded_pool()
    lone = next(i for i in pool.positive if i.example.split == split)
    pool = _map_items(pool, lambda i: _without_relation(i) if i is lone else i)

    for asked in ("val", "test"):
        assert relation_groups(pool, asked) is None


def test_relation_groups_ignore_the_train_items() -> None:
    pool = _map_items(
        _confounded_pool(),
        lambda i: _without_relation(i) if i.example.split == "train" else i,
    )

    assert relation_groups(pool, "test") is not None
    assert relation_groups(pool, "val") is not None


def test_relation_groups_are_none_for_a_non_string_relation() -> None:
    pool = _map_items(_confounded_pool(), lambda i: _with_meta(i, relation=3))

    assert relation_groups(pool, "test") is None


def test_within_relation_pairs_count_the_pairs_inside_a_relation() -> None:
    pool = _confounded_pool()

    assert within_relation_pairs(pool, "test") == 60 * 6 + 6 * 60
    assert within_relation_pairs(pool, "val") == 20 * 4 + 4 * 20


def test_within_relation_pairs_are_zero_without_a_shared_relation() -> None:
    pool = _confounded_pool(apart=("test",))

    assert within_relation_pairs(pool, "test") == 0
    assert within_relation_pairs(pool, "val") == 20 * 4 + 4 * 20


def test_within_relation_pairs_are_none_without_relations() -> None:
    pool = _map_items(_confounded_pool(), _without_relation)

    assert within_relation_pairs(pool, "test") is None


def test_the_within_arrays_have_the_shape_of_the_pooled_ones(
    planted: Planted,
) -> None:
    cube = planted.cube

    assert cube.val_within.shape == cube.val.shape
    assert cube.test_within.shape == cube.test.shape


def test_only_the_relational_contrasts_have_within_values(
    planted: Planted,
) -> None:
    cube = planted.cube

    for t, name in enumerate(cube.types):
        for within in (cube.val_within[:, t], cube.test_within[:, t]):
            if name in RELATIONAL_TYPES:
                assert np.isfinite(within).all(), name
            else:
                assert np.isnan(within).all(), name


def test_the_within_reader_finds_the_planted_directions(
    planted: Planted,
) -> None:
    cube = planted.cube
    rows = [_row(cube, name) for name in RELATIONAL_TYPES]

    for m in range(3):
        for within in (cube.val_within, cube.test_within):
            at_signal = within[m][rows][:, :, SIGNAL].mean(axis=1)
            at_noise = within[m][rows][:, :, 0].mean(axis=1)
            assert at_signal.min() > 0.9, cube.methods[m]
            assert at_noise.max() < 0.75, cube.methods[m]


def test_inside_one_relation_the_within_auroc_is_the_pooled_auroc(
    planted: Planted,
) -> None:
    pool = _map_items(
        dataclasses.replace(planted.pools["obscure_real"], strata="none"),
        lambda i: _with_meta(i, relation="all"),
    )

    cube = reading_cube(
        planted.acts,
        {"one": pool},
        methods=("logistic", "pca_diff"),
        n_seeds=2,
        n_pairs=16,
        seed=1,
    )

    np.testing.assert_allclose(cube.val_within, cube.val, atol=1e-12)
    np.testing.assert_allclose(cube.test_within, cube.test, atol=1e-12)


def test_a_base_rate_gap_between_relations_inflates_the_pooled_auroc_only() -> (
    None
):
    pool = _confounded_pool()

    cube = reading_cube(
        _relation_activations(pool),
        {"confounded": pool},
        methods=("logistic",),
        n_seeds=2,
        n_pairs=64,
        seed=1,
    )

    assert cube.layers == (1, 2)
    np.testing.assert_allclose(cube.test[0, 0, :, 1], POOLED_BASE_RATE_AUROC)
    np.testing.assert_allclose(cube.test_within[0, 0, :, 1], 0.5)
    np.testing.assert_allclose(cube.val[0, 0, :, 1], 480 / 576)
    np.testing.assert_allclose(cube.val_within[0, 0, :, 1], 0.5)
    assert choose_layer(cube.val, cube.layers) == 2


def test_the_within_values_are_nan_where_a_split_shares_no_relation() -> None:
    pool = _confounded_pool(apart=("val",))

    cube = reading_cube(
        _relation_activations(pool),
        {"confounded": pool},
        methods=("logistic",),
        n_seeds=2,
        n_pairs=64,
        seed=1,
    )

    assert np.isnan(cube.val_within).all()
    assert np.isfinite(cube.test_within).all()
    assert np.isfinite(cube.val).all()
    assert choose_layer(cube.val, cube.layers) == 2


def test_the_pooled_arrays_do_not_depend_on_the_relations(
    planted: Planted,
) -> None:
    relational = dataclasses.replace(
        planted.pools["obscure_real"], strata="none"
    )
    flat = _map_items(relational, _without_relation)
    kwargs: dict[str, Any] = {
        "methods": ("logistic", "pca_diff"),
        "n_seeds": 2,
        "n_pairs": 16,
        "seed": 1,
    }

    with_relations = reading_cube(planted.acts, {"t": relational}, **kwargs)
    without = reading_cube(planted.acts, {"t": flat}, **kwargs)

    assert np.isfinite(with_relations.test_within).all()
    assert np.isnan(without.test_within).all()
    assert np.isnan(without.val_within).all()
    for name in ("val", "test", "transfer", "cosine"):
        np.testing.assert_array_equal(
            getattr(with_relations, name), getattr(without, name)
        )
    assert choose_layer(with_relations.val, with_relations.layers) == (
        choose_layer(without.val, without.layers)
    )


def test_surface_baselines_name_the_six_columns_in_order() -> None:
    assert list(surface_baselines(_confounded_pool())) == [
        "text_word",
        "text_char",
        "popularity",
        "text_word_within",
        "text_char_within",
        "popularity_within",
    ]
    assert WITHIN_BASELINES == (
        "text_word_within",
        "text_char_within",
        "popularity_within",
    )


def test_surface_baselines_fall_to_chance_inside_a_relation() -> None:
    baselines = surface_baselines(_confounded_pool())

    for name in ("text_word", "text_char", "popularity"):
        assert baselines[name] == pytest.approx(POOLED_BASE_RATE_AUROC), name
    for name in WITHIN_BASELINES:
        assert baselines[name] == pytest.approx(0.5), name


def test_surface_baselines_inside_a_relation_read_a_word_difference() -> None:
    pool = _map_items(
        _baseline_pool(),
        lambda i: _with_meta(
            i, relation="even" if int(i.example.uid[-1]) % 2 == 0 else "odd"
        ),
    )

    baselines = surface_baselines(pool)

    for name in WITHIN_BASELINES:
        assert baselines[name] == pytest.approx(1.0), name


def test_surface_baselines_inside_a_relation_are_null_without_relations() -> (
    None
):
    pool = _map_items(_confounded_pool(), _without_relation)

    baselines = surface_baselines(pool)

    for name in WITHIN_BASELINES:
        assert baselines[name] is None, name
    assert baselines["text_word"] == pytest.approx(POOLED_BASE_RATE_AUROC)


def test_surface_baselines_inside_a_relation_are_null_without_a_pair() -> None:
    baselines = surface_baselines(_confounded_pool(apart=("test",)))

    for name in WITHIN_BASELINES:
        assert baselines[name] is None, name
    assert baselines["popularity"] == pytest.approx(1.0)


def test_surface_baselines_popularity_within_follows_the_null_rule() -> None:
    pool = _confounded_pool()
    lone = next(i for i in pool.negative if i.example.split == "test")
    pool = _map_items(
        pool, lambda i: _with_meta(i, popularity=None) if i is lone else i
    )

    baselines = surface_baselines(pool)

    assert baselines["popularity"] is None
    assert baselines["popularity_within"] is None
    assert baselines["text_word_within"] == pytest.approx(0.5)


def _stem_item(
    uid: str,
    label: Label,
    split: Split,
    relation: str,
    question: str,
    popularity: int = 10,
) -> LabeledItem:
    item = _labeled(
        uid, label, split=split, relation=relation, popularity=popularity
    )
    return dataclasses.replace(
        item, example=dataclasses.replace(item.example, question=question)
    )


def test_surface_baselines_give_a_within_value_for_a_single_shared_pair() -> (
    None
):
    positive: list[LabeledItem] = []
    negative: list[LabeledItem] = []
    for i in range(12):
        relation = CAPITAL if i % 2 else OCCUPATION
        positive.append(
            _stem_item(
                f"p-train-{i}",
                "known",
                "train",
                relation,
                "who wrote the alpha novel",
                500,
            )
        )
        negative.append(
            _stem_item(
                f"n-train-{i}",
                "unknown",
                "train",
                relation,
                "who wrote the beta novel",
                50,
            )
        )
    for split in ("val", "test"):
        positive.append(
            _stem_item(
                f"p-{split}-c",
                "known",
                split,
                CAPITAL,
                "who wrote the alpha novel",
                900,
            )
        )
        negative.append(
            _stem_item(
                f"n-{split}-c",
                "unknown",
                split,
                CAPITAL,
                "who wrote the beta novel",
                12,
            )
        )
    positive.append(
        _stem_item(
            "p-test-o",
            "known",
            "test",
            OCCUPATION,
            "who wrote the alpha novel",
            900,
        )
    )
    pool = TypePool("t", tuple(positive), tuple(negative), "none", "knowledge")

    baselines = surface_baselines(pool)

    assert within_relation_pairs(pool, "test") == 1
    assert baselines["popularity_within"] == pytest.approx(1.0)
    assert baselines["text_word_within"] == pytest.approx(1.0)
    assert baselines["text_char_within"] == pytest.approx(1.0)


def test_surface_baselines_fit_the_within_text_on_the_train_items() -> None:
    def change(item: LabeledItem) -> LabeledItem:
        item = _with_meta(
            item,
            relation="even" if int(item.example.uid[-1]) % 2 == 0 else "odd",
        )
        if item.example.split != "test":
            return item
        word = "beta" if item.example.uid.startswith("p-") else "alpha"
        return dataclasses.replace(
            item,
            example=dataclasses.replace(
                item.example, question=f"who wrote the {word} novel"
            ),
        )

    baselines = surface_baselines(_map_items(_baseline_pool(), change))

    for name in ("text_word", "text_char", *WITHIN_BASELINES[:2]):
        assert baselines[name] == pytest.approx(0.0), name


def test_surface_baselines_keep_the_within_word_and_char_baselines_apart() -> (
    None
):
    positive: list[LabeledItem] = []
    negative: list[LabeledItem] = []
    for split, n in (("train", 12), ("val", 4), ("test", 6)):
        for i in range(n):
            relation = "even" if i % 2 == 0 else "odd"
            positive.append(
                _stem_item(
                    f"p-{split}-{i}",
                    "known",
                    split,
                    relation,
                    f"who wrote the novel qqa{split}{i}",
                )
            )
            negative.append(
                _stem_item(
                    f"n-{split}-{i}",
                    "unknown",
                    split,
                    relation,
                    f"who wrote the novel zzb{split}{i}",
                )
            )
    pool = TypePool("t", tuple(positive), tuple(negative), "none", "knowledge")

    baselines = surface_baselines(pool)

    assert baselines["text_word_within"] == pytest.approx(0.5)
    assert baselines["text_char_within"] == pytest.approx(1.0)


def test_reading_cube_within_arrays_match_a_per_relation_oracle(
    planted: Planted,
) -> None:
    pool = dataclasses.replace(planted.pools["obscure_real"], strata="none")
    methods: tuple[Any, ...] = ("logistic", "pca_diff")

    cube = reading_cube(
        planted.acts,
        {"t": pool},
        methods=methods,
        n_seeds=2,
        n_pairs=16,
        seed=1,
    )

    for split, held in (("val", cube.val_within), ("test", cube.test_within)):
        positive, negative = pool.split(split)
        items = [*positive, *negative]
        rows = planted.acts.rows(items)
        labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
        labels = labels.astype(np.int64)
        relation = np.array([relation_of(i) for i in items])
        for m, method in enumerate(methods):
            for s in range(2):
                reader = fit_type_reader(
                    planted.acts,
                    pool,
                    method=method,
                    n_pairs=16,
                    seed=draw_seed(1, s),
                )
                scores = reader.score(planted.acts.data[:, rows])
                for k in range(len(planted.acts.layers)):
                    total = 0.0
                    pairs = 0.0
                    for name in np.unique(relation):
                        here = relation == name
                        n_pos = int(labels[here].sum())
                        n_neg = int(here.sum()) - n_pos
                        if n_pos and n_neg:
                            total += (
                                n_pos
                                * n_neg
                                * float(auroc(scores[k][here], labels[here]))
                            )
                            pairs += n_pos * n_neg
                    assert held[m, 0, s, k] == pytest.approx(
                        total / pairs, abs=1e-12
                    )


# Taken from the fit that preceded the split into `draw_train` and `fit_draw`,
# on this fixture: the number of train items, a digest of their uids in order
# and the first pairs, by contrast and seed.
PRE_SPLIT_DRAWS = {
    ("obscure_real", 4): (
        32,
        "426552e26eae",
        [
            ("popqa:402", "popqa:426"),
            ("popqa:100", "popqa:307"),
            ("popqa:11", "popqa:437"),
        ],
    ),
    ("obscure_real", 5): (
        32,
        "bce0af3996bb",
        [
            ("popqa:275", "popqa:359"),
            ("popqa:376", "popqa:292"),
            ("popqa:718", "popqa:88"),
        ],
    ),
    ("obscure_real_pop", 4): (
        32,
        "acad56475012",
        [
            ("popqa:621", "popqa:660"),
            ("popqa:459", "popqa:9"),
            ("popqa:272", "popqa:302"),
        ],
    ),
    ("obscure_real_pop", 5): (
        32,
        "36a1a6d5bb50",
        [
            ("popqa:688", "popqa:157"),
            ("popqa:323", "popqa:620"),
            ("popqa:303", "popqa:165"),
        ],
    ),
    ("unknowable_detail", 4): (
        32,
        "effa2e552fd2",
        [
            ("wikidata:detail:6", "triviaqa:209"),
            ("wikidata:detail:63", "triviaqa:127"),
            ("nq_open:156", "wikidata:detail:103"),
        ],
    ),
    ("unknowable_detail", 5): (
        32,
        "6fedd25c9caa",
        [
            ("wikidata:detail:177", "nq_open:195"),
            ("wikidata:detail:132", "nq_open:272"),
            ("wikidata:detail:111", "triviaqa:199"),
        ],
    ),
}


# The sum of the directions and of the means and the signs of the reader of each
# method, from the same fit, with 16 pairs and seed 4.
PRE_SPLIT_READERS = {
    ("obscure_real", "logistic"): (-0.51105, 28.9746, [1.0, -1.0, 1.0]),
    ("obscure_real", "pca_diff"): (0.94213, 9.7139, [-1.0, -1.0, -1.0]),
    ("obscure_real", "mean_diff"): (-0.51879, 28.9746, [1.0, -1.0, 1.0]),
    ("obscure_real", "random"): (1.87111, 28.9746, [1.0, 1.0, 1.0]),
    ("unknowable_detail", "logistic"): (0.75258, -8.0474, [-1.0, -1.0, 1.0]),
    ("unknowable_detail", "pca_diff"): (-2.4015, -11.9654, [1.0, 1.0, 1.0]),
    ("unknowable_detail", "mean_diff"): (-0.3081, -8.0474, [-1.0, 1.0, 1.0]),
    ("unknowable_detail", "random"): (1.87111, -8.0474, [-1.0, -1.0, 1.0]),
}


def _digest(uids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(uids).encode()).hexdigest()[:12]


@pytest.mark.parametrize(("name", "seed"), list(PRE_SPLIT_DRAWS))
def test_draw_train_draws_what_the_fit_before_the_split_drew(
    planted: Planted, name: str, seed: int
) -> None:
    count, digest, first = PRE_SPLIT_DRAWS[name, seed]

    draw = draw_train(planted.pools[name], n_pairs=16, seed=seed)

    uids = [i.example.uid for i in draw.items]
    assert len(uids) == count
    assert _digest(uids) == digest
    assert draw.pairs is not None
    assert [(uids[a], uids[b]) for a, b in draw.pairs[:3]] == first


@pytest.mark.parametrize(("name", "method"), list(PRE_SPLIT_READERS))
def test_fit_type_reader_gives_the_numbers_of_the_fit_before_the_split(
    planted: Planted, name: str, method: Method
) -> None:
    total, centers, signs = PRE_SPLIT_READERS[name, method]

    reader = fit_type_reader(
        planted.acts, planted.pools[name], method=method, n_pairs=16, seed=4
    )

    assert float(reader.directions.sum()) == pytest.approx(total, abs=5e-3)
    assert float(reader.means.sum()) == pytest.approx(centers, abs=1e-2)
    assert reader.signs.tolist() == signs
