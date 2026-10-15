"""Reading known against unknown questions from the activations of a model.

One direction is fitted per contrast of `tomography.p9.unknowns`, on the
train items, and scored on held-out items by AUROC. Activations are collected
once per distinct question and pairs are formed by index afterwards, so a
question shared by many contrasts and draws is run through the model once.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tomography.activations import collect_activations
from tomography.datasets.records import Split
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import few_shot_prompt
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.p9.pool import LabeledItem
from tomography.p9.unknowns import TypePool
from tomography.p9.unknowns import stratum_name
from tomography.pairing import label_contrastive_pairs
from tomography.reading.directions import Method
from tomography.reading.directions import Preprocess
from tomography.reading.metrics import auroc
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import cosine_matrix
from tomography.reading.reader import RepReader
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat
from tomography.templates import ConceptTemplate
from tomography.templates import Template

TEMPLATE_NAMES: tuple[str, ...] = ("fewshot", "concept", "plain")
READING_METHODS: tuple[Method, ...] = (
    "logistic",
    "pca_diff",
    "mean_diff",
    "random",
)


def p9_template(
    name: str, shots: Sequence[tuple[str, str]] = DEFAULT_SHOTS
) -> Template:
    """Returns one of the three prompts a question is read under.

    All three are read at the last token. "fewshot" is the labeling prompt
    of the study (the demonstrations, the question and a bare "A:"), so the
    direction is read in the state that produces the labeled behavior. It
    ignores the chat format, as does "plain", the bare question. "concept" is
    the paper's concept template with the word "knowledge", wrapped in the
    chat format.

    Args:
        name: One of `TEMPLATE_NAMES`.
        shots: The (question, answer) demonstrations of "fewshot".

    Returns:
        The template.

    Raises:
        ValueError: If the name is not one of `TEMPLATE_NAMES`.
    """
    if name == "fewshot":
        return ConceptTemplate(
            "p9_fewshot", few_shot_prompt("{stimulus}", shots), tagged=False
        )
    if name == "concept":
        return PAPER_TEMPLATES["concept"]
    if name == "plain":
        return ConceptTemplate("p9_plain", "{stimulus}", tagged=False)
    raise ValueError(
        f"unknown template {name!r}, expected one of {TEMPLATE_NAMES}"
    )


def render_items(
    items: Sequence[LabeledItem],
    template: Template,
    chat: ChatFormat,
    *,
    concept: str = "knowledge",
) -> list[str]:
    """Renders the question of each item.

    Args:
        items: The labeled questions.
        template: A concept template, such as `p9_template` returns.
        chat: The chat format that wraps the question.
        concept: The concept word of the template.

    Returns:
        One prompt per item, in order.

    Raises:
        ValueError: If the template is not a concept template.
    """
    if not isinstance(template, ConceptTemplate):
        raise ValueError("P9 templates are concept templates")
    return [
        template.render(i.example.question, concept, chat).text for i in items
    ]


@dataclass(frozen=True, eq=False)
class ItemActivations:
    """The activations of distinct questions at the last token of a prompt.

    Attributes:
        uids: The question of each row of the second axis of `data`.
        layers: Layer index of each slice of the first axis.
        data: Array of shape (K, n, d_model).
    """

    uids: tuple[str, ...]
    layers: tuple[int, ...]
    data: NDArray[np.float32]

    def rows(self, items: Sequence[LabeledItem]) -> NDArray[np.int64]:
        """Finds the row of each item.

        Args:
            items: Questions whose activations were collected.

        Returns:
            The row of each item, in order, as int64.

        Raises:
            ValueError: If an item has no row; names the first one.
        """
        index = {uid: row for row, uid in enumerate(self.uids)}
        for item in items:
            if item.example.uid not in index:
                raise ValueError(f"no activations for {item.example.uid!r}")
        return np.array([index[i.example.uid] for i in items], dtype=np.int64)


def collect_item_activations(
    loaded: LoadedModel,
    items: Sequence[LabeledItem],
    template: Template,
    chat: ChatFormat,
    *,
    layers: Sequence[int] | None,
    concept: str = "knowledge",
    bos: BosMode | None = None,
    batch_size: int = 8,
) -> ItemActivations:
    """Collects the activations at the last token of each rendered question.

    Args:
        loaded: The wrapped model.
        items: Distinct labeled questions.
        template: A concept template, such as `p9_template` returns.
        chat: The chat format that wraps the question.
        layers: Layer indices in 0..L; None means all of them.
        concept: The concept word of the template.
        bos: BOS policy; None uses the model's default.
        batch_size: Prompts per forward pass.

    Returns:
        The activations, one row per item in order.

    Raises:
        ValueError: If the template is not a concept template.
    """
    texts = render_items(items, template, chat, concept=concept)
    acts = collect_activations(
        loaded,
        texts,
        layers=layers,
        position="last",
        bos=bos,
        batch_size=batch_size,
    )
    return ItemActivations(
        tuple(i.example.uid for i in items), acts.layers, acts.data
    )


def check_pool(pool: TypePool, *, min_pairs: int, min_eval: int) -> str | None:
    """Says why a contrast cannot be read, if it cannot.

    Args:
        pool: One contrast.
        min_pairs: Fewest train items on each side.
        min_eval: Fewest val items, and fewest test items, on each side.

    Returns:
        None for a usable pool, else the reason: the train split is checked
        first, then val, then test.
    """
    positive, negative = pool.split("train")
    if min(len(positive), len(negative)) < min_pairs:
        return f"fewer than {min_pairs} train items on one side"
    splits: tuple[Split, ...] = ("val", "test")
    for split in splits:
        positive, negative = pool.split(split)
        if min(len(positive), len(negative)) < min_eval:
            return f"fewer than {min_eval} {split} items on one side"
    return None


def _split_arrays(
    acts: ItemActivations, pool: TypePool, split: Split
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    positive, negative = pool.split(split)
    rows = acts.rows([*positive, *negative])
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    return rows, labels.astype(np.int64)


def fit_type_reader(
    acts: ItemActivations,
    pool: TypePool,
    *,
    method: Method = "logistic",
    preprocess: Preprocess = "center",
    n_pairs: int | None = 64,
    seed: int = 0,
) -> RepReader:
    """Fits a reader of one contrast on its train items.

    The train items are paired, one from each side and never across two
    strata, in a random order from `seed`; the first `n_pairs` pairs are
    kept and the fit uses the items they hold. A higher score means the
    positive side for every method: the reader orients each layer on the
    labels.

    Args:
        acts: The activations of at least the train items of the pool.
        pool: One contrast.
        method: The direction finder.
        preprocess: Vector preparation, see `fit_direction`.
        n_pairs: Pairs to keep; None keeps all of them.
        seed: Seeds the pairing and the stochastic finders.

    Returns:
        The reader, one direction per layer of `acts`.

    Raises:
        ValueError: If no pair can be formed or an item has no activations.
    """
    positive, negative = pool.split("train")
    rows, labels = _split_arrays(acts, pool, "train")
    names = [stratum_name(i, pool.strata) for i in (*positive, *negative)]
    codes = np.unique(names, return_inverse=True)[1].astype(np.int64)
    pairs = label_contrastive_pairs(labels, seed=seed, strata=codes)[:n_pairs]
    used = np.unique(pairs)
    local = np.searchsorted(used, pairs)
    return RepReader.fit(
        method,
        acts.data[:, rows[used]],
        acts.layers,
        labels=labels[used],
        pairs=local,
        groups=local,
        preprocess=preprocess,
        seed=seed,
    )


def type_auroc(
    reader: RepReader, acts: ItemActivations, pool: TypePool, split: Split
) -> NDArray[np.float64]:
    """Scores a reader on the items of one split of a contrast.

    The AUROC pools every item of the split, positive side = 1. It is not a
    paired accuracy, because the two sides differ in size. Chance is 0.5 and
    a value below it means the direction is inverted on held-out data.

    Args:
        reader: A fitted reader over the layers of `acts`.
        acts: The activations of at least the items of the split.
        pool: One contrast; need not be the one the reader was fitted on.
        split: "train", "val" or "test".

    Returns:
        The AUROC of each layer, shape (K,).

    Raises:
        ValueError: If a side of the split is empty.
    """
    rows, labels = _split_arrays(acts, pool, split)
    return auroc(reader.score(acts.data[:, rows]), labels)


@dataclass(frozen=True, eq=False)
class ReadingCube:
    """Every AUROC of the reading study.

    Attributes:
        types: The contrasts, in the order of the axes below.
        methods: The direction finders; the first is the headline.
        layers: Layer index of each entry of the last axis.
        val: Shape (M, T, S, K): the AUROC on the validation items of the
            same contrast, for each method, contrast, draw and layer.
        test: Shape (M, T, S, K): the same on the test items.
        transfer: Shape (S, K, T, T): the first method fitted on contrast i
            (row) and scored on the test items of contrast j (column).
        cosine: Shape (K, T, T): cosines of the signed directions fitted on
            all train pairs of each contrast, so every contrast points
            towards its positive side.
        readers: Those all-pairs readers, by contrast.
    """

    types: tuple[str, ...]
    methods: tuple[str, ...]
    layers: tuple[int, ...]
    val: NDArray[np.float64]
    test: NDArray[np.float64]
    transfer: NDArray[np.float64]
    cosine: NDArray[np.float64]
    readers: dict[str, RepReader]


def _draw_seed(seed: int, draw: int) -> int:
    sequence = np.random.SeedSequence([seed, draw])
    return int(sequence.generate_state(1)[0])


def reading_cube(
    acts: ItemActivations,
    pools: Mapping[str, TypePool],
    *,
    methods: Sequence[Method] = READING_METHODS,
    preprocess: Preprocess = "center",
    n_seeds: int = 5,
    n_pairs: int | None = 64,
    seed: int = 0,
) -> ReadingCube:
    """Fits and scores a reader per contrast, method and draw.

    Draw `s` of `n_seeds` samples other pairs; its random state is the first
    word of `SeedSequence([seed, s])`. Each fit is scored on the val and test
    items of its own contrast. The first method is also scored on the test
    items of every other contrast, which crosses the two kinds of contrast: a
    knowledge direction scored on an unknown-type contrast asks whether
    "known" is only "familiar entity". The cosines come from readers fitted
    on all train pairs with `seed`.

    Args:
        acts: The activations of every item of the pools.
        pools: The contrasts, by name.
        methods: The direction finders; the first is the headline.
        preprocess: Vector preparation, see `fit_direction`.
        n_seeds: Number of draws.
        n_pairs: Pairs per fit; None keeps all of them.
        seed: The run seed.

    Returns:
        The cube.

    Raises:
        ValueError: If there is no contrast, no method or no draw, or a
            contrast cannot be fitted or scored.
    """
    if not pools:
        raise ValueError("pools must hold at least one contrast")
    if not methods:
        raise ValueError("methods must not be empty")
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be at least 1, got {n_seeds}")
    names = tuple(pools)
    n_layers = len(acts.layers)
    val = np.zeros((len(methods), len(names), n_seeds, n_layers))
    test = np.zeros_like(val)
    transfer = np.zeros((n_seeds, n_layers, len(names), len(names)))
    splits: tuple[Split, ...] = ("val", "test")
    held = {
        split: {name: _split_arrays(acts, pools[name], split) for name in names}
        for split in splits
    }

    def score(
        reader: RepReader, split: Split, name: str
    ) -> NDArray[np.float64]:
        rows, labels = held[split][name]
        return auroc(reader.score(acts.data[:, rows]), labels)

    for t, name in enumerate(names):
        for s in range(n_seeds):
            for m, method in enumerate(methods):
                reader = fit_type_reader(
                    acts,
                    pools[name],
                    method=method,
                    preprocess=preprocess,
                    n_pairs=n_pairs,
                    seed=_draw_seed(seed, s),
                )
                val[m, t, s] = score(reader, "val", name)
                test[m, t, s] = score(reader, "test", name)
                if m == 0:
                    for u, other in enumerate(names):
                        transfer[s, :, t, u] = (
                            test[m, t, s]
                            if u == t
                            else score(reader, "test", other)
                        )
    full = {
        name: fit_type_reader(
            acts,
            pools[name],
            method=methods[0],
            preprocess=preprocess,
            n_pairs=None,
            seed=seed,
        )
        for name in names
    }
    cosine = np.stack(
        [
            cosine_matrix(
                np.stack(
                    [full[n].directions[k] * full[n].signs[k] for n in names]
                )
            )
            for k in range(n_layers)
        ]
    )
    return ReadingCube(
        names, tuple(methods), acts.layers, val, test, transfer, cosine, full
    )


def choose_layer(val: NDArray[np.float64], layers: Sequence[int]) -> int:
    """Picks the layer with the best validation AUROC of the first method.

    Args:
        val: The validation AUROC of a `ReadingCube`, shape (M, T, S, K).
        layers: The layer index of each entry of the last axis.

    Returns:
        The layer index (an element of `layers`, not a position) with the
        highest mean of `val[0]` over contrasts and draws; the first layer
        wins ties. No test item chooses anything.
    """
    return best_layer(val[0].mean(axis=(0, 1)), layers)
