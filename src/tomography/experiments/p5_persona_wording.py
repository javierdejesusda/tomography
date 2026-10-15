"""P5: does the honesty reader read honesty or the persona's wording?"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any
from typing import cast

from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from tomography.activations import PositionStrategy
from tomography.datasets.facts import load_facts
from tomography.datasets.geometry_of_truth import load_geometry_of_truth
from tomography.datasets.records import Example
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import Prepared
from tomography.experiments.sweep import ReadingConfig
from tomography.experiments.sweep import SweepResult
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import stimulus_activations
from tomography.experiments.sweep import sweep_from_activations
from tomography.experiments.sweep import write_curves
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.directions import Method
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import parse_groups
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat
from tomography.templates import ConceptTemplate
from tomography.templates import FunctionTemplate
from tomography.viz.style import INK_MUTED
from tomography.viz.style import PX
from tomography.viz.style import TEXT_WIDTH
from tomography.viz.style import categorical_colors
from tomography.viz.style import new_figure
from tomography.viz.style import save_figure
from tomography.viz.style import style_context

PERSONA_PAIRS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "paper": ("an honest", "an untruthful"),
        "synonym_0": ("a truthful", "a deceitful"),
        "synonym_1": ("a sincere", "a lying"),
        "unrelated_0": ("a tall", "a short"),
        "unrelated_1": ("an old", "a young"),
    }
)
CONTROLS: tuple[str, ...] = ("instruction_only", "fact_only", "random_label")
CONDITIONS: tuple[str, ...] = (*PERSONA_PAIRS, *CONTROLS)
LABEL_METHODS: tuple[Method, ...] = ("mean_diff", "logistic")
GOT_SUBSETS: tuple[str, ...] = (
    "cities",
    "neg_cities",
    "sp_en_trans",
    "larger_than",
    "common_claim_true_false",
)
PLAIN = ConceptTemplate("honesty_plain", "{stimulus}", tagged=False)
PLAIN_POSITION: PositionStrategy = "mean"
MAX_CONDITIONS = 8
_READER = SERIES.index("reader")
_RELATED = ("paper", "synonym_0", "synonym_1")
_UNRELATED = ("unrelated_0", "unrelated_1")


@dataclass(frozen=True, kw_only=True)
class Config(ReadingConfig):
    """P5 settings: which persona words the honesty reader depends on.

    Every condition changes one thing against the paper persona prompt, the
    persona words or the stimuli, and is read the way R5 reads honesty: a
    reader trained on the response tokens of true statements, then scored on
    plain true against false statements at the token mean. Accuracies are
    raw, so a reader whose first persona aligns with false statements
    scores below 0.5.

    Attributes:
        conditions: Names from `CONDITIONS`, at most `MAX_CONDITIONS`, and
            `paper` is required. `random_label` is skipped unless `method` is
            in `LABEL_METHODS`.
        got_subsets: Geometry of Truth subsets scored on their test split.
        n_boot: Number of bootstrap resamples of every interval.
    """

    experiment: str = "p5"
    model: str = "gpt2"
    method: Method = "logistic"
    position: PositionStrategy = "response"
    n_eval_pairs: int | None = 256
    conditions: tuple[str, ...] = CONDITIONS
    got_subsets: tuple[str, ...] = GOT_SUBSETS
    n_boot: int = 2000


def persona_template(condition: str) -> FunctionTemplate:
    """Returns the paper honesty template with the persona words replaced.

    Args:
        condition: A key of `PERSONA_PAIRS`.

    Returns:
        The template of the paper, its instruction text unchanged, with the
        first word of the pair as the label-1 persona and the second as the
        label-0 persona.

    Raises:
        ValueError: If `condition` is not a key of `PERSONA_PAIRS`.
    """
    if condition not in PERSONA_PAIRS:
        raise ValueError(
            f"{condition!r} has no persona words, expected one of "
            f"{tuple(PERSONA_PAIRS)}"
        )
    paper = PAPER_TEMPLATES["honesty"]
    assert isinstance(paper, FunctionTemplate)
    pos, neg = PERSONA_PAIRS[condition]
    return replace(
        paper,
        name=f"honesty_{condition}",
        pos_personas=(pos,),
        neg_personas=(neg,),
    )


def plain_stimuli(
    statements: Sequence[Example], *, n_pairs: int | None, seed: int
) -> StimulusSet:
    """Pairs plain true statements with plain false ones, with no persona.

    Args:
        statements: Statements labeled 1 for true and 0 for false.
        n_pairs: Number of pairs, capped at the number of true statements
            and at the number of false ones; None takes every pair.
        seed: Seed of the pairing.

    Returns:
        The `PLAIN` stimuli in the label-contrastive layout.

    Raises:
        ValueError: If the statements hold no true or no false one.
    """
    n_true = sum(e.label == 1 for e in statements)
    n_false = sum(e.label == 0 for e in statements)
    if not n_true or not n_false:
        raise ValueError(
            "plain stimuli need true and false statements, got "
            f"{n_true} true and {n_false} false"
        )
    return build_stimulus_set(
        statements,
        PLAIN,
        ChatFormat("none"),
        pairing="label_contrastive",
        n_pairs=None if n_pairs is None else min(n_pairs, n_true, n_false),
        seed=seed,
    )


def balanced_flip(
    labels: NDArray[np.int64], groups: NDArray[np.int64], *, seed: int
) -> NDArray[np.int64]:
    """Swaps the two labels of exactly half of the pairs.

    An independent shuffle per pair leaves a binomial imbalance of flipped
    pairs, and a label-fitting direction leaks the true one in proportion to
    it. Flipping exactly `n // 2` of the `n` pairs removes that first-order
    leak; an odd `n` leaves one pair, a leak of `1 / n`.

    Args:
        labels: Binary labels of shape (m,).
        groups: Row pairs of shape (n, 2), each with exactly one label-1
            member.
        seed: Seed of the choice of the swapped pairs.

    Returns:
        New labels of shape (m,): the pairs `np.random.default_rng(seed)
        .permutation(n)[: n // 2]` have their two labels exchanged, so every
        pair keeps exactly one label-1 member.

    Raises:
        ValueError: If a group does not have exactly two members, one of them
            labeled 1.
    """
    index = parse_groups(groups, labels)
    if not (index.valid.sum(axis=1) == 2).all():
        raise ValueError("balanced_flip needs groups of exactly two members")
    n = groups.shape[0]
    swapped = groups[np.random.default_rng(seed).permutation(n)[: n // 2]]
    flipped = labels.astype(np.int64, copy=True)
    flipped[swapped] = labels[swapped][:, ::-1]
    return flipped


def condition_stimuli(
    condition: str,
    statements: Sequence[Example],
    chat: ChatFormat,
    *,
    n_pairs: int | None,
    seed: int,
    train: bool,
) -> StimulusSet:
    """Builds the stimuli of one condition for one split.

    The persona conditions and `random_label` use the true statements only,
    and the persona plan depends on the statements and the seed alone, so
    with the same seed they draw the same statements. `instruction_only`
    renders the paper personas with an empty response, so every training set
    holds the same two texts: its readers are identical across seeds, its
    intervals have zero width and only its out-of-distribution columns carry
    information. `fact_only` drops the persona and pairs plain true with
    plain false statements.

    Args:
        condition: A name from `CONDITIONS`.
        statements: The statements of one split, true ones labeled 1.
        chat: The chat format of the persona prompts.
        n_pairs: Number of statements (pairs); None takes every one.
        seed: Seed of the draw.
        train: Whether the stimuli train a reader. Only `random_label`
            changes: its training labels are swapped in exactly half of the
            pairs (`balanced_flip`, with the seed
            `derive_seed(seed, "random_label")`), and its evaluation labels
            stay real.

    Returns:
        The stimuli of the condition.

    Raises:
        ValueError: If the condition is unknown or the statements do not fit
            the stimuli.
    """
    if condition == "fact_only":
        return plain_stimuli(statements, n_pairs=n_pairs, seed=seed)
    if condition not in CONDITIONS:
        raise ValueError(
            f"unknown condition {condition!r}, expected one of {CONDITIONS}"
        )
    true = [e for e in statements if e.label == 1]
    if condition == "instruction_only":
        true = [replace(e, text="") for e in true]
    template = (
        persona_template(condition)
        if condition in PERSONA_PAIRS
        else persona_template("paper")
    )
    stimuli = build_stimulus_set(
        true, template, chat, pairing="persona", n_pairs=n_pairs, seed=seed
    )
    if condition == "random_label" and train:
        labels = balanced_flip(
            stimuli.labels,
            stimuli.groups,
            seed=derive_seed(seed, "random_label"),
        )
        stimuli = replace(stimuli, labels=labels)
    return stimuli


def condition_position(
    condition: str, position: PositionStrategy
) -> PositionStrategy:
    """Returns the token positions a condition reads.

    Args:
        condition: A name from `CONDITIONS`.
        position: The position of the persona conditions.

    Returns:
        `last` for `instruction_only`, whose response is empty,
        `PLAIN_POSITION` for `fact_only` and `position` otherwise.
    """
    if condition == "instruction_only":
        return "last"
    if condition == "fact_only":
        return PLAIN_POSITION
    return position


def _check(
    config: Config, facts: Sequence[Example], got: Sequence[Example]
) -> dict[str, list[Example]]:
    conditions = config.conditions
    if not 1 <= len(conditions) <= MAX_CONDITIONS:
        raise ValueError(
            f"conditions must hold between 1 and {MAX_CONDITIONS} names, "
            f"got {len(conditions)}"
        )
    if len(set(conditions)) != len(conditions):
        raise ValueError(f"conditions must be distinct, got {conditions}")
    if "paper" not in conditions:
        raise ValueError(f"conditions must include 'paper', got {conditions}")
    if len(set(config.got_subsets)) != len(config.got_subsets):
        raise ValueError(
            f"got_subsets must be distinct, got {config.got_subsets}"
        )
    for split in ("train", "val", "test"):
        if not any(e.split == split and e.label == 1 for e in facts):
            raise ValueError(
                f"the {split} split of the facts has no true statements"
            )
    test_rows = {
        subset: [
            e
            for e in got
            if e.meta.get("subset") == subset and e.split == "test"
        ]
        for subset in config.got_subsets
    }
    missing = [subset for subset, rows in test_rows.items() if not rows]
    if missing:
        raise ValueError(
            f"no test statements for the Geometry of Truth subsets {missing}"
        )
    return test_rows


def _plan(
    config: Config,
    condition: str,
    by_split: Mapping[str, Sequence[Example]],
    chat: ChatFormat,
) -> tuple[list[StimulusSet], dict[str, StimulusSet]]:
    train = [
        condition_stimuli(
            condition,
            by_split["train"],
            chat,
            n_pairs=config.n_pairs,
            seed=derive_seed(config.seed, "train", i),
            train=True,
        )
        for i in range(config.n_seeds)
    ]
    if condition == "fact_only":
        return train, {}
    evals = {}
    for name, split in (("id_val", "val"), ("id", "test")):
        statements = by_split[split]
        n_true = sum(e.label == 1 for e in statements)
        evals[name] = condition_stimuli(
            condition,
            statements,
            chat,
            n_pairs=(
                None
                if config.n_eval_pairs is None
                else min(config.n_eval_pairs, n_true)
            ),
            seed=derive_seed(config.seed, "eval", "id", split),
            train=False,
        )
    return train, evals


def _prepare(
    config: Config,
    loaded: LoadedModel,
    layers: tuple[int, ...],
    stimuli: StimulusSet,
    position: PositionStrategy,
) -> Prepared:
    act = stimulus_activations(
        loaded,
        stimuli,
        layers=layers,
        position=position,
        bos=config.bos,
        batch_size=config.batch_size,
    )
    return stimuli, act


def _fit_condition(
    config: Config,
    loaded: LoadedModel,
    layers: tuple[int, ...],
    condition: str,
    plan: tuple[list[StimulusSet], dict[str, StimulusSet]],
    shared: Mapping[str, Prepared],
) -> SweepResult:
    train, evals = plan
    where = condition_position(condition, config.position)
    where_eval = condition_position(
        condition,
        config.position
        if config.eval_position is None
        else config.eval_position,
    )
    trained = [_prepare(config, loaded, layers, s, where) for s in train]
    own = {
        name: _prepare(config, loaded, layers, stimuli, where_eval)
        for name, stimuli in evals.items()
    }
    if condition == "fact_only":
        own = {"id_val": shared["facts_val"], "id": shared["facts"]}
    return sweep_from_activations(
        trained,
        {**own, **shared},
        method=config.method,
        preprocess=config.preprocess,
        seed=config.seed,
    )


def _skipped(config: Config) -> dict[str, str]:
    if "random_label" in config.conditions and (
        config.method not in LABEL_METHODS
    ):
        return {
            "random_label": (
                f"method {config.method!r} fits its direction without "
                "labels; flipped labels would only set its sign"
            )
        }
    return {}


def _table_rows(
    conditions: Sequence[str],
    curves: Mapping[str, Mapping[str, NDArray[Any]]],
    layers: Sequence[int],
    ood: Sequence[str],
) -> list[list[Any]]:
    def at_layer(name: str, c: int, layer: int) -> list[float]:
        at = layers.index(layer)
        return [float(curves[name][key][c, at]) for key in ("mean", "lo", "hi")]

    rows = []
    for c, condition in enumerate(conditions):
        id_layer = best_layer(curves["id_val"]["mean"][c], layers)
        ood_layer = best_layer(curves["facts_val"]["mean"][c], layers)
        rows.append(
            [
                condition,
                id_layer,
                *at_layer("id", c, id_layer),
                ood_layer,
                *(v for name in ood for v in at_layer(name, c, ood_layer)),
            ]
        )
    return rows


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    facts: Sequence[Example],
    got: Sequence[Example],
) -> None:
    """Reads honesty under every condition and scores the truth sets.

    `random_label` runs only when `config.method` is in `LABEL_METHODS`: a
    finder that fits its direction without the labels (`pca_diff` and the
    others) would only take its sign from the flipped labels, so the reader
    would be the paper reader up to sign and its row would read as a null.
    The condition is then named in `skipped` with its reason and has no
    series, reader, table row or scatter point.

    The shared evaluation sets are plain true against false statements read
    once at `PLAIN_POSITION` and reused by every condition: the facts val and
    test splits (`facts_val`, `facts`) and the test split of each Geometry of
    Truth subset (`got_<subset>`). Each condition trains `n_seeds` readers on
    the train split, scores its own in-distribution val and test sets (`id_val`
    and `id`, built like its training sets with the real labels; for
    `fact_only` they are the facts sets) and every shared set, and releases
    its activations before the next one.

    Writes `curves_<set>` for every set, with one series per condition and the
    accuracy of its reader at each layer, and `curves_cosine`, the cosine of
    each condition's reader with the paper reader of the same seed (absent
    when only `paper` runs). Also writes `readers_<condition>`, `table.json`
    and `summary.json`, which lists the conditions run and, under `skipped`,
    the reason of each skipped one. The table reports `id` at the layer that
    is best on the condition's `id_val` curve and every other set at the layer
    that is best on its `facts_val` curve, so no test set picks a layer.
    Accuracies are raw, not folded above 0.5.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        facts: Facts with `train`, `val` and `test` splits, label 1 for a
            true statement and 0 for a false one.
        got: Geometry of Truth statements; `meta["subset"]` names the subset
            and the `test` split of each subset in `config.got_subsets` is
            scored.

    Raises:
        ValueError: If `config.conditions` are not 1 to `MAX_CONDITIONS`
            distinct names of `CONDITIONS` including `paper`, the
            `got_subsets` repeat, a split of the facts has no true
            statement, a subset has no test statement, or a split cannot
            supply the requested pairs.
    """
    test_rows = _check(config, facts, got)
    skipped = _skipped(config)
    conditions = tuple(c for c in config.conditions if c not in skipped)
    by_split = {
        split: [e for e in facts if e.split == split]
        for split in ("train", "val", "test")
    }
    chat = chat_format_for(loaded)
    sets = {
        name: plain_stimuli(
            rows,
            n_pairs=config.n_eval_pairs,
            seed=derive_seed(config.seed, "eval", *keys),
        )
        for name, rows, keys in (
            ("facts_val", by_split["val"], ("facts", "val")),
            ("facts", by_split["test"], ("facts", "test")),
            *(
                (f"got_{subset}", rows, (subset,))
                for subset, rows in test_rows.items()
            ),
        )
    }
    plans = {c: _plan(config, c, by_split, chat) for c in conditions}
    layers = (
        tuple(range(1, loaded.n_layers + 1))
        if config.layers is None
        else config.layers
    )
    shared = {
        name: _prepare(config, loaded, layers, stimuli, PLAIN_POSITION)
        for name, stimuli in sets.items()
    }
    names = ("id_val", "id", *shared)
    accuracy: dict[str, list[NDArray[np.float64]]] = {n: [] for n in names}
    oriented = {}
    for condition in conditions:
        result = _fit_condition(
            config, loaded, layers, condition, plans[condition], shared
        )
        for name in names:
            accuracy[name].append(result.per_seed(name)[_READER])
        directions = np.stack([r.directions for r in result.readers])
        signs = np.stack([r.signs for r in result.readers])
        run_dir.write_arrays(
            f"readers_{condition}",
            layers=np.asarray(result.layers, dtype=np.int64),
            directions=directions,
            signs=signs,
        )
        oriented[condition] = signs[:, :, None] * directions
    for name in names:
        write_curves(
            run_dir,
            np.stack(accuracy[name]),
            layers,
            series=conditions,
            stem=f"curves_{name}",
            chance=0.5,
            n_layers=loaded.n_layers,
            n_boot=config.n_boot,
        )
    others = [c for c in conditions if c != "paper"]
    if others:
        write_curves(
            run_dir,
            np.stack(
                [
                    np.einsum("nkd,nkd->nk", oriented[c], oriented["paper"])
                    for c in others
                ]
            ).astype(np.float64),
            layers,
            series=others,
            stem="curves_cosine",
            ylabel="Cosine with the paper reader",
            chance=0.0,
            n_layers=loaded.n_layers,
            n_boot=config.n_boot,
        )
    ood = ["facts", *(f"got_{subset}" for subset in config.got_subsets)]
    curves = {name: run_dir.read_arrays(f"curves_{name}") for name in names}
    columns = [
        "condition",
        "id_layer",
        "id",
        "id_lo",
        "id_hi",
        "ood_layer",
        *(f"{name}{suffix}" for name in ood for suffix in ("", "_lo", "_hi")),
    ]
    rows = _table_rows(conditions, curves, layers, ood)
    run_dir.write_json("table", {"columns": columns, "rows": rows})
    by_condition = {
        row[0]: dict(zip(columns, row, strict=True)) for row in rows
    }
    summary: dict[str, Any] = {
        "conditions": list(conditions),
        "personas": {
            c: list(PERSONA_PAIRS[c]) for c in conditions if c in PERSONA_PAIRS
        },
        "got_subsets": list(config.got_subsets),
        "n_eval_pairs": {
            name: int(stimuli.groups.shape[0]) for name, stimuli in sets.items()
        },
        "n_seeds": config.n_seeds,
        "skipped": skipped,
    }
    related = [c for c in conditions if c in _RELATED]
    unrelated = [c for c in conditions if c in _UNRELATED]
    if unrelated:
        summary["lexical_gap"] = {
            name: float(
                np.mean([by_condition[c][name] for c in related])
                - np.mean([by_condition[c][name] for c in unrelated])
            )
            for name in ood
        }
    run_dir.write_json("summary", summary)


def run(config: Config, run_dir: RunDir) -> None:
    """Loads the model, the facts and Geometry of Truth, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    execute(
        config,
        run_dir,
        load_model(config.model),
        load_facts(),
        load_geometry_of_truth(subsets=config.got_subsets),
    )


def _persona_scatter(table: Mapping[str, Any]) -> Figure:
    rows = [
        dict(zip(table["columns"], row, strict=True)) for row in table["rows"]
    ]
    colors = categorical_colors(len(rows))
    with style_context():
        fig, axes = new_figure(width=TEXT_WIDTH, aspect=0.55)
        ax = axes[0, 0]
        for row, color in zip(rows, colors, strict=True):
            ax.hlines(
                row["facts"], row["id_lo"], row["id_hi"], color=color, lw=PX
            )
            ax.vlines(
                row["id"], row["facts_lo"], row["facts_hi"], color=color, lw=PX
            )
            ax.plot(
                row["id"],
                row["facts"],
                "o",
                color=color,
                label=row["condition"],
            )
        ax.axhline(0.5, color=INK_MUTED, lw=PX, label="_chance")
        ax.axvline(0.5, color=INK_MUTED, lw=PX, label="_chance")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("In-distribution accuracy")
        ax.set_ylabel("Facts accuracy")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    return fig


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the P5 curves and the persona confound scatter.

    Args:
        run_dir: A finished P5 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of every `curves_<set>` file in stem
        order, then of `persona_scatter`: each condition's in-distribution
        accuracy against its facts accuracy, both with their intervals and
        the chance level of each axis at 0.5.
    """
    stems = sorted(p.stem for p in run_dir.path.glob("curves*.npz"))
    paths = [
        path
        for stem in stems
        for path in plot_curves_file(run_dir, out_dir, stem=stem)
    ]
    table = cast(dict[str, Any], run_dir.read_json("table"))
    return paths + save_figure(
        _persona_scatter(table), out_dir, "persona_scatter"
    )
