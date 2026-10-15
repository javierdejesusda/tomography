"""Tests for the P5 persona wording experiment."""

import dataclasses
import gc
from pathlib import Path
from typing import Any
import weakref

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.cli import load_experiment
from tomography.datasets.facts import load_facts
from tomography.datasets.geometry_of_truth import load_geometry_of_truth
from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.datasets.sources import OfflineCacheMissError
from tomography.experiments import p5_persona_wording
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.p5_persona_wording import CONDITIONS
from tomography.experiments.p5_persona_wording import CONTROLS
from tomography.experiments.p5_persona_wording import GOT_SUBSETS
from tomography.experiments.p5_persona_wording import LABEL_METHODS
from tomography.experiments.p5_persona_wording import MAX_CONDITIONS
from tomography.experiments.p5_persona_wording import PERSONA_PAIRS
from tomography.experiments.p5_persona_wording import PLAIN
from tomography.experiments.p5_persona_wording import PLAIN_POSITION
from tomography.experiments.p5_persona_wording import Config
from tomography.experiments.p5_persona_wording import balanced_flip
from tomography.experiments.p5_persona_wording import condition_position
from tomography.experiments.p5_persona_wording import condition_stimuli
from tomography.experiments.p5_persona_wording import execute
from tomography.experiments.p5_persona_wording import persona_template
from tomography.experiments.p5_persona_wording import plain_stimuli
from tomography.experiments.p5_persona_wording import plot
from tomography.experiments.p5_persona_wording import run
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import SweepResult
from tomography.experiments.sweep import shuffle_within_groups
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.reader import RepReader
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat
from tomography.templates import ConceptTemplate
from tomography.templates import FunctionTemplate

WORDS = {
    "paper": ("an honest", "an untruthful"),
    "synonym_0": ("a truthful", "a deceitful"),
    "synonym_1": ("a sincere", "a lying"),
    "unrelated_0": ("a tall", "a short"),
    "unrelated_1": ("an old", "a young"),
}


PLAIN_CHAT = ChatFormat("plain")
PERSONA_CONDITIONS = (*WORDS, "random_label")


def _statements(
    count: int = 12, splits: tuple[Split, ...] = ("train", "val", "test")
) -> list[Example]:
    examples = []
    for split in splits:
        for r in range(count):
            body = "plain and long " * (r % 4 + 1)
            examples.append(
                Example(
                    uid=f"fact-{split}-{r}",
                    text=f"The fact number {r} is {body}",
                    label=1,
                    split=split,
                )
            )
            examples.append(
                Example(
                    uid=f"fact-{split}-{r}-false",
                    text=f"The claim number {r} is odd and {body}",
                    label=0,
                    split=split,
                )
            )
    return examples


def _got(
    subsets: tuple[str, ...] = ("cities", "larger_than"),
    splits: tuple[Split, ...] = ("train", "test"),
    count: int = 16,
) -> list[Example]:
    rows = []
    for subset in subsets:
        for split in splits:
            for r in range(count):
                rows.append(
                    Example(
                        uid=f"got-{subset}-{split}-{r}",
                        text=f"In {subset} item {r} the claim is "
                        f"{'big' if r % 2 else 'small'} and plain",
                        label=r % 2,
                        split=split,
                        meta={"subset": subset},
                    )
                )
    return rows


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        n_seeds=3,
        n_pairs=8,
        n_eval_pairs=4,
        got_subsets=("cities", "larger_than"),
        n_boot=50,
    )
    return dataclasses.replace(base, **overrides)


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "p5"
    assert config.model == "gpt2"
    assert config.method == "logistic"
    assert config.position == "response"
    assert config.n_eval_pairs == 256
    assert config.conditions == CONDITIONS
    assert config.got_subsets == GOT_SUBSETS
    assert config.n_boot == 2000
    assert Config.from_dict(config.to_dict()) == config


def test_the_conditions_are_the_personas_then_the_controls() -> None:
    assert dict(PERSONA_PAIRS) == WORDS
    assert CONTROLS == ("instruction_only", "fact_only", "random_label")
    assert (*WORDS, *CONTROLS) == CONDITIONS
    assert len(CONDITIONS) == MAX_CONDITIONS == 8
    assert ConceptTemplate("honesty_plain", "{stimulus}", tagged=False) == PLAIN
    assert PLAIN_POSITION == "mean"
    assert GOT_SUBSETS == (
        "cities",
        "neg_cities",
        "sp_en_trans",
        "larger_than",
        "common_claim_true_false",
    )


def test_the_paper_persona_words_are_those_of_the_paper_template() -> None:
    paper = PAPER_TEMPLATES["honesty"]
    assert isinstance(paper, FunctionTemplate)
    assert PERSONA_PAIRS["paper"] == (
        paper.pos_personas[0],
        paper.neg_personas[0],
    )


@pytest.mark.parametrize("condition", sorted(WORDS))
def test_a_persona_template_changes_only_the_persona_words(
    condition: str,
) -> None:
    paper = PAPER_TEMPLATES["honesty"]
    template = persona_template(condition)
    pos, neg = WORDS[condition]
    assert template.name == f"honesty_{condition}"
    assert template.pos_personas == (pos,)
    assert template.neg_personas == (neg,)
    assert template.user == paper.user
    assert template.tagged == paper.tagged


@pytest.mark.parametrize("condition", [*CONTROLS, "shouting"])
def test_a_persona_template_exists_for_persona_conditions_only(
    condition: str,
) -> None:
    with pytest.raises(ValueError, match=condition):
        persona_template(condition)


def test_plain_stimuli_pair_true_against_false_statements() -> None:
    statements = _statements(count=6, splits=("train",))
    stimuli = plain_stimuli(statements, n_pairs=4, seed=1)
    texts = {e.text for e in statements}
    assert set(stimuli.texts) <= texts
    assert stimuli.groups.shape == (4, 2)
    assert stimuli.labels.shape == (8,)
    for group in stimuli.groups:
        assert sorted(stimuli.labels[group]) == [0, 1]
    source = {e.uid: e for e in statements}
    for text, label, uid in zip(
        stimuli.texts, stimuli.labels, stimuli.uids, strict=True
    ):
        assert label == source[uid].label
        assert text == source[uid].text


def test_plain_stimuli_cap_the_pairs_and_pass_none_through() -> None:
    statements = [
        e
        for e in _statements(count=6, splits=("train",))
        if e.label == 1 or e.uid.endswith(("-0-false", "-1-false", "-2-false"))
    ]
    assert plain_stimuli(statements, n_pairs=10, seed=1).groups.shape[0] == 3
    assert plain_stimuli(statements, n_pairs=2, seed=1).groups.shape[0] == 2
    assert plain_stimuli(statements, n_pairs=None, seed=1).groups.shape[0] == 3


@pytest.mark.parametrize("label", [0, 1])
def test_plain_stimuli_need_both_truth_values(label: int) -> None:
    statements = [e for e in _statements() if e.label == label]
    with pytest.raises(ValueError, match="true and false"):
        plain_stimuli(statements, n_pairs=4, seed=1)


def _persona(condition: str, **kwargs: Any) -> Any:
    settings: dict[str, Any] = {"n_pairs": 8, "seed": 3, "train": True}
    settings.update(kwargs)
    return condition_stimuli(condition, _statements(), PLAIN_CHAT, **settings)


def test_persona_conditions_draw_the_same_true_statements() -> None:
    reference = _persona("paper")
    true_uids = {e.uid for e in _statements() if e.label == 1}
    assert set(reference.uids) <= true_uids
    assert len(set(reference.uids)) == 8
    for condition in PERSONA_CONDITIONS:
        assert _persona(condition).uids == reference.uids
    assert _persona("paper", seed=4).uids != reference.uids


@pytest.mark.parametrize("condition", sorted(WORDS))
def test_a_persona_condition_renders_its_own_words(condition: str) -> None:
    stimuli = _persona(condition)
    pos, neg = WORDS[condition]
    for text, label in zip(stimuli.texts, stimuli.labels, strict=True):
        assert (pos if label == 1 else neg) in text
        assert (neg if label == 1 else pos) not in text
    assert stimuli.groups.shape == (8, 2)


@pytest.mark.parametrize("n_pairs", [12, 11])
def test_random_label_swaps_half_of_the_training_pairs(n_pairs: int) -> None:
    real = _persona("paper", n_pairs=n_pairs)
    flipped = _persona("random_label", n_pairs=n_pairs)
    assert flipped.texts == real.texts
    assert real.groups.shape[0] == n_pairs
    first = real.groups[:, 0]
    swapped = flipped.labels[first] != real.labels[first]
    seed = derive_seed(3, "random_label")
    chosen = np.random.default_rng(seed).permutation(n_pairs)[: n_pairs // 2]
    assert sorted(np.flatnonzero(swapped).tolist()) == sorted(chosen.tolist())
    for group in flipped.groups:
        assert sorted(flipped.labels[group]) == [0, 1]
    evaluation = _persona("random_label", n_pairs=n_pairs, train=False)
    np.testing.assert_array_equal(evaluation.labels, real.labels)


def _pair_labels(n: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    groups = np.arange(2 * n, dtype=np.int64).reshape(n, 2)
    labels = np.empty(2 * n, dtype=np.int64)
    labels[0::2] = np.arange(n) % 2
    labels[1::2] = 1 - labels[0::2]
    return labels, groups


@pytest.mark.parametrize("n", [1, 2, 6, 7])
def test_balanced_flip_swaps_exactly_half_of_the_pairs(n: int) -> None:
    labels, groups = _pair_labels(n)
    before = labels.copy()
    flipped = balanced_flip(labels, groups, seed=5)
    np.testing.assert_array_equal(labels, before)
    swapped = flipped[groups[:, 0]] != labels[groups[:, 0]]
    assert swapped.sum() == n // 2
    for group in groups:
        assert sorted(flipped[group]) == [0, 1]
    chosen = np.random.default_rng(5).permutation(n)[: n // 2]
    assert sorted(np.flatnonzero(swapped).tolist()) == sorted(chosen.tolist())


def test_balanced_flip_is_deterministic_and_the_seed_picks_the_pairs() -> None:
    labels, groups = _pair_labels(20)
    first = balanced_flip(labels, groups, seed=1)
    np.testing.assert_array_equal(first, balanced_flip(labels, groups, seed=1))
    for seed in (2, 3, 4):
        other = balanced_flip(labels, groups, seed=seed)
        assert not np.array_equal(other, first)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        pytest.param("three", "two members", id="three-members"),
        pytest.param("one", "two members", id="one-member"),
        pytest.param("padded", "two members", id="padded-column"),
        pytest.param("no_positive", "one label-1 member", id="no-positive"),
        pytest.param("two_positive", "one label-1 member", id="two-positive"),
    ],
)
def test_balanced_flip_needs_pairs_with_one_positive_member(
    case: str, message: str
) -> None:
    labels, groups = _pair_labels(4)
    if case == "three":
        groups = np.arange(12, dtype=np.int64).reshape(4, 3)
        labels = np.tile(np.array([1, 0, 0], dtype=np.int64), 4)
    elif case == "one":
        groups = groups.copy()
        groups[1, 1] = -1
    elif case == "padded":
        pad = np.full((4, 1), -1, dtype=np.int64)
        groups = np.concatenate([groups, pad], axis=1)
    elif case == "no_positive":
        labels[groups[2]] = 0
    else:
        labels[groups[2]] = 1
    with pytest.raises(ValueError, match=message):
        balanced_flip(labels, groups, seed=0)


def test_balanced_flip_leaks_less_than_a_per_pair_shuffle() -> None:
    n, d = 32, 64
    rng = np.random.default_rng(0)
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)
    shift = 2.0 * rng.normal(size=(n, d))
    labels, groups = _pair_labels(n)
    h = np.empty((1, 2 * n, d))
    sign = (2 * labels[0::2] - 1)[:, None]
    h[0, 0::2] = shift + sign * direction + 0.3 * rng.normal(size=(n, d))
    h[0, 1::2] = shift - sign * direction + 0.3 * rng.normal(size=(n, d))

    def read(fitted: Any) -> Any:
        reader = RepReader.fit(
            "mean_diff",
            h,
            (1,),
            labels=fitted,
            pairs=groups,
            groups=groups,
            preprocess="center",
        )
        return reader.directions[0]

    paper = read(labels)
    seeds = range(20)

    def overlap(make: Any) -> float:
        return float(np.mean([abs(read(make(s)) @ paper) for s in seeds]))

    balanced = overlap(lambda s: balanced_flip(labels, groups, seed=s))
    shuffled = overlap(lambda s: shuffle_within_groups(labels, groups, seed=s))
    assert balanced < 0.5 * shuffled
    assert balanced < 0.2


def test_instruction_only_is_the_persona_instruction_alone() -> None:
    stimuli = _persona("instruction_only")
    reference = _persona("paper")
    assert stimuli.uids == reference.uids
    assert len(set(stimuli.texts)) == 2
    assert all(
        span is not None and span[0] == span[1]
        for span in stimuli.response_spans
    )
    assert not any(
        e.text in text for e in _statements() for text in stimuli.texts
    )
    for text, label in zip(stimuli.texts, stimuli.labels, strict=True):
        assert ("honest" if label == 1 else "untruthful") in text
    np.testing.assert_array_equal(stimuli.labels, reference.labels)


def test_fact_only_trains_on_plain_true_against_false_statements() -> None:
    stimuli = _persona("fact_only")
    expected = plain_stimuli(_statements(), n_pairs=8, seed=3)
    assert stimuli.texts == expected.texts
    assert stimuli.uids == expected.uids
    np.testing.assert_array_equal(stimuli.labels, expected.labels)
    assert {e.label for e in _statements() if e.uid in stimuli.uids} == {0, 1}
    assert not any(
        word in text for text in stimuli.texts for word in ("honest", "USER")
    )


def test_an_unknown_condition_raises() -> None:
    with pytest.raises(ValueError, match="shouting"):
        _persona("shouting")


@pytest.mark.parametrize("position", ["last", "response", "mean", "concept"])
def test_condition_positions(position: Any) -> None:
    assert condition_position("instruction_only", position) == "last"
    assert condition_position("fact_only", position) == "mean"
    for condition in (*WORDS, "random_label"):
        assert condition_position(condition, position) == position


def _execute(
    loaded: LoadedModel,
    out_root: Path,
    *,
    facts: list[Example] | None = None,
    got: list[Example] | None = None,
    **overrides: Any,
) -> RunDir:
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=out_root)
    execute(
        config,
        run_dir,
        loaded,
        _statements() if facts is None else facts,
        _got() if got is None else got,
    )
    return run_dir


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"conditions": ()}, "between 1 and 8", id="none"),
        pytest.param(
            {"conditions": (*CONDITIONS, "paper")},
            "between 1 and 8",
            id="nine",
        ),
        pytest.param(
            {"conditions": ("paper", "paper")}, "distinct", id="repeated"
        ),
        pytest.param(
            {"conditions": ("paper", "shouting")}, "shouting", id="unknown"
        ),
        pytest.param({"conditions": ("unrelated_0",)}, "paper", id="no-paper"),
        pytest.param(
            {"got_subsets": ("cities", "cities")},
            "distinct",
            id="repeated-subset",
        ),
        pytest.param(
            {"got_subsets": ("cities", "neg_cities")},
            "neg_cities",
            id="subset-without-rows",
        ),
        pytest.param({"n_pairs": 50}, "n_pairs=50", id="too-many-pairs"),
    ],
)
def test_bad_inputs_fail_before_any_forward(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    message: str,
) -> None:
    def nothing(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("nothing may run before the checks")

    monkeypatch.setattr(p5_persona_wording, "stimulus_activations", nothing)
    with pytest.raises(ValueError, match=message):
        _execute(tiny_arch, tmp_path, **overrides)


def test_a_subset_with_no_test_statements_raises(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nothing(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("nothing may run before the checks")

    monkeypatch.setattr(p5_persona_wording, "stimulus_activations", nothing)
    got = _got(splits=("train", "val"))
    with pytest.raises(ValueError, match="no test statements"):
        _execute(tiny_arch, tmp_path, got=got)


def test_facts_without_a_split_fail_before_any_forward(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nothing(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("nothing may run before the checks")

    monkeypatch.setattr(p5_persona_wording, "stimulus_activations", nothing)
    facts = _statements(splits=("train", "test"))
    with pytest.raises(ValueError, match="val split"):
        _execute(tiny_arch, tmp_path, facts=facts)


EVALS = ("id", "id_val", "facts", "facts_val", "got_cities", "got_larger_than")
_RUNS: dict[str, RunDir] = {}


@pytest.fixture
def ran(
    tiny_arch: LoadedModel, tmp_path_factory: pytest.TempPathFactory
) -> RunDir:
    """The finished default run of the architecture, built once."""
    key = type(tiny_arch.model).__name__
    if key not in _RUNS:
        _RUNS[key] = _execute(tiny_arch, tmp_path_factory.mktemp(key))
    return _RUNS[key]


def _oriented(run_dir: RunDir, condition: str) -> Any:
    readers = run_dir.read_arrays(f"readers_{condition}")
    return readers["signs"][:, :, None] * readers["directions"]


def test_execute_writes_the_files(ran: RunDir) -> None:
    expected = {
        "config.json",
        "meta.json",
        "table.json",
        "summary.json",
        "curves_cosine.npz",
        "curves_cosine.json",
        *(f"curves_{e}.{ext}" for e in EVALS for ext in ("npz", "json")),
        *(f"readers_{c}.npz" for c in CONDITIONS),
    }
    assert {p.name for p in ran.path.iterdir()} == expected
    assert len(expected) == 26


@pytest.mark.parametrize("name", EVALS)
def test_each_curves_file_holds_one_series_per_condition(
    ran: RunDir, tiny_arch: LoadedModel, name: str
) -> None:
    n_layers = tiny_arch.n_layers
    curves = ran.read_arrays(f"curves_{name}")
    assert curves["x"].tolist() == list(range(1, n_layers + 1))
    assert curves["mean"].shape == (len(CONDITIONS), n_layers)
    assert curves["per_seed"].shape == (len(CONDITIONS), 3, n_layers)
    assert curves["lo"].shape == curves["hi"].shape == curves["mean"].shape
    assert ((curves["per_seed"] >= 0.0) & (curves["per_seed"] <= 1.0)).all()
    sidecar = ran.read_json(f"curves_{name}")
    assert isinstance(sidecar, dict)
    assert sidecar["series"] == list(CONDITIONS)
    assert sidecar["chance"] == 0.5
    assert sidecar["n_layers"] == n_layers


def test_the_readers_files_use_the_reading_layout(
    ran: RunDir, tiny_arch: LoadedModel
) -> None:
    n_layers = tiny_arch.n_layers
    for condition in CONDITIONS:
        readers = ran.read_arrays(f"readers_{condition}")
        assert readers["layers"].dtype == np.int64
        assert readers["layers"].tolist() == list(range(1, n_layers + 1))
        directions = readers["directions"]
        assert directions.shape[:2] == (3, n_layers)
        np.testing.assert_allclose(
            np.linalg.norm(directions, axis=-1), 1.0, rtol=1e-4
        )
        assert readers["signs"].shape == (3, n_layers)
        assert set(np.unique(readers["signs"])) <= {-1.0, 1.0}


def test_the_readers_files_hold_the_fitted_readers(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results: list[SweepResult] = []
    real = p5_persona_wording.sweep_from_activations

    def spy(*args: Any, **kwargs: Any) -> SweepResult:
        result = real(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(p5_persona_wording, "sweep_from_activations", spy)
    conditions = ("paper", "synonym_0")
    run_dir = _execute(tiny_arch, tmp_path, conditions=conditions)
    assert len(results) == len(conditions)
    for condition, result in zip(conditions, results, strict=True):
        saved = run_dir.read_arrays(f"readers_{condition}")
        directions = np.stack([r.directions for r in result.readers])
        signs = np.stack([r.signs for r in result.readers])
        assert saved["layers"].tolist() == list(result.layers)
        np.testing.assert_array_equal(saved["directions"], directions)
        np.testing.assert_array_equal(saved["signs"], signs)


def test_the_cosine_curves_compare_each_reader_with_the_paper_reader(
    ran: RunDir, tiny_arch: LoadedModel
) -> None:
    others = [c for c in CONDITIONS if c != "paper"]
    curves = ran.read_arrays("curves_cosine")
    assert curves["x"].tolist() == list(range(1, tiny_arch.n_layers + 1))
    assert curves["per_seed"].shape == (7, 3, tiny_arch.n_layers)
    sidecar = ran.read_json("curves_cosine")
    assert isinstance(sidecar, dict)
    assert sidecar["series"] == others
    assert sidecar["chance"] == 0.0
    paper = _oriented(ran, "paper")
    for i, condition in enumerate(others):
        expected = np.einsum("nkd,nkd->nk", _oriented(ran, condition), paper)
        np.testing.assert_allclose(
            curves["per_seed"][i], expected, rtol=1e-4, atol=1e-5
        )


def test_instruction_only_reads_its_own_text_perfectly(ran: RunDir) -> None:
    index = CONDITIONS.index("instruction_only")
    for name in ("id", "id_val"):
        per_seed = ran.read_arrays(f"curves_{name}")["per_seed"]
        np.testing.assert_array_equal(per_seed[index], 1.0)


def test_fact_only_reads_the_facts_as_its_own_distribution(
    ran: RunDir,
) -> None:
    index = CONDITIONS.index("fact_only")
    for own, shared in (("id", "facts"), ("id_val", "facts_val")):
        np.testing.assert_array_equal(
            ran.read_arrays(f"curves_{own}")["per_seed"][index],
            ran.read_arrays(f"curves_{shared}")["per_seed"][index],
        )


OOD = ("facts", "got_cities", "got_larger_than")


def _table(run_dir: RunDir) -> dict[str, dict[str, Any]]:
    table = run_dir.read_json("table")
    assert isinstance(table, dict)
    columns = table["columns"]
    return {
        row[0]: dict(zip(columns, row, strict=True)) for row in table["rows"]
    }


def _first_best(values: Any, layers: list[int]) -> int:
    return layers[int(np.argmax(values))]


def test_the_table_has_one_row_per_condition_and_the_documented_columns(
    ran: RunDir,
) -> None:
    table = ran.read_json("table")
    assert isinstance(table, dict)
    assert table["columns"] == [
        "condition",
        "id_layer",
        "id",
        "id_lo",
        "id_hi",
        "ood_layer",
        *(f"{name}{suffix}" for name in OOD for suffix in ("", "_lo", "_hi")),
    ]
    assert [row[0] for row in table["rows"]] == list(CONDITIONS)
    assert all(len(row) == len(table["columns"]) for row in table["rows"])


def test_the_table_reads_each_condition_at_its_val_layers(
    ran: RunDir, tiny_arch: LoadedModel
) -> None:
    layers = list(range(1, tiny_arch.n_layers + 1))
    rows = _table(ran)
    for c, condition in enumerate(CONDITIONS):
        row = rows[condition]
        val = ran.read_arrays("curves_id_val")["mean"][c]
        assert row["id_layer"] == _first_best(val, layers)
        at = layers.index(row["id_layer"])
        curves = ran.read_arrays("curves_id")
        assert row["id"] == pytest.approx(curves["mean"][c, at])
        assert row["id_lo"] == pytest.approx(curves["lo"][c, at])
        assert row["id_hi"] == pytest.approx(curves["hi"][c, at])
        val = ran.read_arrays("curves_facts_val")["mean"][c]
        assert row["ood_layer"] == _first_best(val, layers)
        at = layers.index(row["ood_layer"])
        for name in OOD:
            curves = ran.read_arrays(f"curves_{name}")
            assert row[name] == pytest.approx(curves["mean"][c, at])
            assert row[f"{name}_lo"] == pytest.approx(curves["lo"][c, at])
            assert row[f"{name}_hi"] == pytest.approx(curves["hi"][c, at])


def _peak(name: str, c: int, n_layers: int) -> int:
    if name == "id_val":
        return c % n_layers
    if name in ("id", "facts_val"):
        return (c + 1) % n_layers
    return c % n_layers


def _planted(name: str, c: int, series: int, n_layers: int) -> Any:
    accuracy = np.zeros((3, n_layers))
    for seed in range(3):
        accuracy[seed] = 0.4 - 0.05 * series + 0.01 * seed
    accuracy[:, _peak(name, c, n_layers)] += 0.4
    return accuracy


def _plant_sweep(monkeypatch: pytest.MonkeyPatch, n_layers: int) -> None:
    real = p5_persona_wording.sweep_from_activations
    calls: list[int] = []

    def spy(*args: Any, **kwargs: Any) -> SweepResult:
        result = real(*args, **kwargs)
        c = len(calls)
        calls.append(c)
        accuracy = np.zeros(result.accuracy.shape)
        for e, name in enumerate(result.eval_names):
            for series in range(accuracy.shape[1]):
                accuracy[e, series] = _planted(name, c, series, n_layers)
        return dataclasses.replace(result, accuracy=accuracy)

    monkeypatch.setattr(p5_persona_wording, "sweep_from_activations", spy)


def test_no_layer_is_chosen_on_a_test_set(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    n_layers = tiny_arch.n_layers
    assert n_layers >= 3
    _plant_sweep(monkeypatch, n_layers)
    run_dir = _execute(tiny_arch, tmp_path)
    layers = list(range(1, n_layers + 1))
    reader = SERIES.index("reader")
    for name in EVALS:
        per_seed = run_dir.read_arrays(f"curves_{name}")["per_seed"]
        for c in range(len(CONDITIONS)):
            np.testing.assert_allclose(
                per_seed[c], _planted(name, c, reader, n_layers)
            )
    rows = _table(run_dir)
    for c, condition in enumerate(CONDITIONS):
        row = rows[condition]
        assert row["id_layer"] == layers[c % n_layers]
        assert row["ood_layer"] == layers[(c + 1) % n_layers]
        planted = _planted("id", c, reader, n_layers)
        assert row["id"] == pytest.approx(planted[:, c % n_layers].mean())
        for name in OOD:
            planted = _planted(name, c, reader, n_layers)
            at = (c + 1) % n_layers
            assert row[name] == pytest.approx(planted[:, at].mean())
            assert row[name] != pytest.approx(planted.max(axis=1).mean())


def _record_conditions(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    calls: list[Any] = []
    real = p5_persona_wording.condition_stimuli

    def spy(
        condition: str,
        statements: Any,
        chat: Any,
        *,
        n_pairs: int | None,
        seed: int,
        train: bool,
    ) -> Any:
        result = real(
            condition,
            statements,
            chat,
            n_pairs=n_pairs,
            seed=seed,
            train=train,
        )
        splits = {e.split for e in statements}
        calls.append((condition, train, seed, n_pairs, splits, result))
        return result

    monkeypatch.setattr(p5_persona_wording, "condition_stimuli", spy)
    return calls


def test_the_conditions_draw_their_stimuli_per_seed_and_split(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_conditions(monkeypatch)
    config = _config()
    _execute(tiny_arch, tmp_path)
    expected: list[tuple[str, bool, int, int | None, set[str]]] = []
    for condition in config.conditions:
        expected += [
            (
                condition,
                True,
                derive_seed(config.seed, "train", i),
                config.n_pairs,
                {"train"},
            )
            for i in range(config.n_seeds)
        ]
        if condition != "fact_only":
            expected += [
                (
                    condition,
                    False,
                    derive_seed(config.seed, "eval", "id", split),
                    config.n_eval_pairs,
                    {split},
                )
                for split in ("val", "test")
            ]
    assert sorted((c[:5] for c in calls), key=repr) == sorted(
        expected, key=repr
    )


def test_every_persona_condition_trains_on_the_same_statements(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_conditions(monkeypatch)
    _execute(tiny_arch, tmp_path)
    for i in range(3):
        train = {
            c[0]: c[5].uids
            for c in calls
            if c[1] and c[2] == derive_seed(_config().seed, "train", i)
        }
        for condition in PERSONA_CONDITIONS:
            assert train[condition] == train["paper"]
        assert train["fact_only"] != train["paper"]


def test_the_evaluation_pairs_are_capped_at_the_true_statements(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_conditions(monkeypatch)
    _execute(tiny_arch, tmp_path, n_eval_pairs=256)
    evals = [c for c in calls if not c[1]]
    assert evals
    assert {c[3] for c in evals} == {12}


def test_a_none_evaluation_size_takes_every_pair(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_conditions(monkeypatch)
    _execute(tiny_arch, tmp_path, n_eval_pairs=None)
    assert {c[3] for c in calls if not c[1]} == {None}


def test_the_in_distribution_sets_are_capped_at_their_own_split(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts = (
        _statements(count=12, splits=("train",))
        + _statements(count=5, splits=("val",))
        + _statements(count=7, splits=("test",))
    )
    calls = _record_conditions(monkeypatch)
    _execute(
        tiny_arch,
        tmp_path,
        facts=facts,
        conditions=("paper",),
        n_eval_pairs=256,
    )
    sizes = {next(iter(c[4])): c[3] for c in calls if not c[1]}
    assert sizes == {"val": 5, "test": 7}
    for c in calls:
        if not c[1]:
            assert c[5].groups.shape[0] == c[3]


def _expected_positions(config: Config) -> list[str]:
    where = (
        config.position
        if config.eval_position is None
        else (config.eval_position)
    )
    positions = ["mean"] * (2 + len(config.got_subsets))
    for condition in config.conditions:
        train = condition_position(condition, config.position)
        positions += [train] * config.n_seeds
        if condition != "fact_only":
            positions += [condition_position(condition, where)] * 2
    return positions


@pytest.mark.parametrize("eval_position", [None, "last"])
def test_the_shared_sets_are_read_once_at_the_token_mean(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    eval_position: Any,
) -> None:
    positions: list[str] = []
    real = p5_persona_wording.stimulus_activations

    def spy(loaded: Any, stimuli: Any, **kwargs: Any) -> Any:
        positions.append(kwargs["position"])
        return real(loaded, stimuli, **kwargs)

    monkeypatch.setattr(p5_persona_wording, "stimulus_activations", spy)
    config = _config(eval_position=eval_position)
    _execute(tiny_arch, tmp_path, eval_position=eval_position)
    assert positions == _expected_positions(config)
    assert len(positions) == 4 + 8 * 3 + 7 * 2


def test_the_activations_of_a_condition_are_released_before_the_next(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs: list[Any] = []
    real = p5_persona_wording.stimulus_activations
    config = _config()
    sizes = [
        config.n_seeds + (0 if c == "fact_only" else 2)
        for c in config.conditions
    ]
    starts = {4 + sum(sizes[:i]) for i in range(1, len(sizes))}
    alive_at_start: list[int] = []

    def spy(loaded: Any, stimuli: Any, **kwargs: Any) -> Any:
        if len(refs) in starts:
            gc.collect()
            alive_at_start.append(sum(r() is not None for r in refs[4:]))
        act = real(loaded, stimuli, **kwargs)
        refs.append(weakref.ref(act))
        return act

    monkeypatch.setattr(p5_persona_wording, "stimulus_activations", spy)
    _execute(tiny_arch, tmp_path)
    assert len(alive_at_start) == len(config.conditions) - 1
    assert alive_at_start == [0] * len(alive_at_start)


def test_the_summary_describes_the_run(ran: RunDir) -> None:
    summary = ran.read_json("summary")
    assert isinstance(summary, dict)
    assert set(summary) == {
        "conditions",
        "personas",
        "got_subsets",
        "n_eval_pairs",
        "lexical_gap",
        "n_seeds",
        "skipped",
    }
    assert summary["skipped"] == {}
    assert summary["conditions"] == list(CONDITIONS)
    assert summary["personas"] == {
        name: list(words) for name, words in WORDS.items()
    }
    assert summary["got_subsets"] == ["cities", "larger_than"]
    assert summary["n_eval_pairs"] == {
        "facts_val": 4,
        "facts": 4,
        "got_cities": 4,
        "got_larger_than": 4,
    }
    assert summary["n_seeds"] == 3


@pytest.mark.parametrize("method", ["pca_diff", "pca_raw", "kmeans", "random"])
@pytest.mark.parametrize(
    "conditions",
    [("paper", "synonym_0", "random_label"), ("paper", "random_label")],
)
def test_random_label_is_skipped_under_a_method_that_ignores_labels(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    conditions: tuple[str, ...],
    method: Any,
) -> None:
    assert method not in LABEL_METHODS
    calls = _record_conditions(monkeypatch)
    run_dir = _execute(
        tiny_arch, tmp_path, method=method, conditions=conditions
    )
    run = [c for c in conditions if c != "random_label"]
    assert "random_label" not in {c[0] for c in calls}
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["conditions"] == run
    assert summary["skipped"] == {
        "random_label": f"method '{method}' fits its direction without "
        "labels; flipped labels would only set its sign"
    }
    names = {p.name for p in run_dir.path.iterdir()}
    assert "readers_random_label.npz" not in names
    assert {n for n in names if n.startswith("readers_")} == {
        f"readers_{c}.npz" for c in run
    }
    for name in EVALS:
        sidecar = run_dir.read_json(f"curves_{name}")
        assert isinstance(sidecar, dict)
        assert sidecar["series"] == run
        assert run_dir.read_arrays(f"curves_{name}")["mean"].shape[0] == len(
            run
        )
    assert list(_table(run_dir)) == run
    assert (run_dir.path / "curves_cosine.npz").exists() == (len(run) > 1)
    if len(run) > 1:
        cosine = run_dir.read_json("curves_cosine")
        assert isinstance(cosine, dict)
        assert cosine["series"] == run[1:]


def test_the_skipped_condition_has_no_point_in_the_scatter(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _execute(
        tiny_arch,
        tmp_path,
        method="pca_diff",
        conditions=("paper", "random_label"),
    )
    saved: dict[str, Any] = {}
    real = p5_persona_wording.save_figure

    def spy(fig: Any, out_dir: Any, name: str, **kwargs: Any) -> Any:
        saved[name] = fig
        return real(fig, out_dir, name, **kwargs)

    monkeypatch.setattr(p5_persona_wording, "save_figure", spy)
    plot(run_dir, tmp_path / "figs")
    ax = saved["persona_scatter"].axes[0]
    labels = {line.get_label() for line in ax.lines}
    assert "paper" in labels
    assert "random_label" not in labels


@pytest.mark.parametrize("n_pairs", [8, 7])
def test_random_label_runs_under_mean_diff_with_balanced_flips(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n_pairs: int,
) -> None:
    sweeps: list[tuple[Any, Any]] = []
    real = p5_persona_wording.sweep_from_activations

    def spy(train: Any, evaluation: Any, **kwargs: Any) -> Any:
        sweeps.append((train, dict(evaluation)))
        return real(train, evaluation, **kwargs)

    monkeypatch.setattr(p5_persona_wording, "sweep_from_activations", spy)
    run_dir = _execute(
        tiny_arch,
        tmp_path,
        method="mean_diff",
        n_pairs=n_pairs,
        conditions=("paper", "random_label"),
    )
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["skipped"] == {}
    assert summary["conditions"] == ["paper", "random_label"]
    assert (run_dir.path / "readers_random_label.npz").is_file()
    assert list(_table(run_dir)) == ["paper", "random_label"]
    (paper_train, paper_eval), (flipped_train, flipped_eval) = sweeps
    assert len(flipped_train) == 3
    for (real_set, _), (flipped_set, _) in zip(
        paper_train, flipped_train, strict=True
    ):
        first = real_set.groups[:, 0]
        swapped = flipped_set.labels[first] != real_set.labels[first]
        assert swapped.sum() == n_pairs // 2
    for name in ("id", "id_val"):
        np.testing.assert_array_equal(
            flipped_eval[name][0].labels, paper_eval[name][0].labels
        )


@pytest.mark.parametrize(
    ("n_eval_pairs", "expected"),
    [(None, (12, 12, 8, 8)), (256, (12, 12, 8, 8)), (6, (6, 6, 6, 6))],
)
def test_the_shared_sets_are_capped_at_the_pairs_they_allow(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    n_eval_pairs: int | None,
    expected: tuple[int, ...],
) -> None:
    run_dir = _execute(
        tiny_arch,
        tmp_path,
        conditions=("paper", "fact_only"),
        n_eval_pairs=n_eval_pairs,
    )
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    names = ("facts_val", "facts", "got_cities", "got_larger_than")
    assert summary["n_eval_pairs"] == dict(zip(names, expected, strict=True))


def test_the_lexical_gap_is_the_wording_effect_on_each_truth_set(
    ran: RunDir,
) -> None:
    summary = ran.read_json("summary")
    assert isinstance(summary, dict)
    rows = _table(ran)
    related = ("paper", "synonym_0", "synonym_1")
    unrelated = ("unrelated_0", "unrelated_1")
    assert set(summary["lexical_gap"]) == set(OOD)
    for name in OOD:
        expected = np.mean([rows[c][name] for c in related]) - np.mean(
            [rows[c][name] for c in unrelated]
        )
        assert summary["lexical_gap"][name] == pytest.approx(expected)


def test_the_lexical_gap_averages_the_conditions_that_ran(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(
        tiny_arch, tmp_path, conditions=("synonym_1", "paper", "unrelated_1")
    )
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    rows = _table(run_dir)
    for name in OOD:
        related = (rows["paper"][name] + rows["synonym_1"][name]) / 2
        expected = related - rows["unrelated_1"][name]
        assert summary["lexical_gap"][name] == pytest.approx(expected)


@pytest.mark.parametrize(
    "conditions", [("paper", "fact_only"), ("paper", "synonym_0")]
)
def test_there_is_no_lexical_gap_without_both_kinds_of_wording(
    tiny_arch: LoadedModel, tmp_path: Path, conditions: tuple[str, ...]
) -> None:
    run_dir = _execute(tiny_arch, tmp_path, conditions=conditions)
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert "lexical_gap" not in summary


def test_the_paper_condition_alone_writes_no_cosine_curves(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(tiny_arch, tmp_path, conditions=("paper",))
    assert not (run_dir.path / "curves_cosine.npz").exists()
    assert not (run_dir.path / "curves_cosine.json").exists()
    assert (run_dir.path / "readers_paper.npz").is_file()
    assert list(_table(run_dir)) == ["paper"]
    sidecar = run_dir.read_json("curves_id")
    assert isinstance(sidecar, dict)
    assert sidecar["series"] == ["paper"]


def test_the_conditions_keep_the_order_they_are_given(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    conditions = ("fact_only", "unrelated_0", "paper")
    run_dir = _execute(tiny_arch, tmp_path, conditions=conditions)
    sidecar = run_dir.read_json("curves_facts")
    assert isinstance(sidecar, dict)
    assert sidecar["series"] == list(conditions)
    cosine = run_dir.read_json("curves_cosine")
    assert isinstance(cosine, dict)
    assert cosine["series"] == ["fact_only", "unrelated_0"]
    assert list(_table(run_dir)) == list(conditions)


def test_the_shared_sets_pair_plain_statements_with_their_own_seeds(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    real = p5_persona_wording.plain_stimuli

    def spy(statements: Any, *, n_pairs: int | None, seed: int) -> Any:
        calls.append(
            (
                {e.split for e in statements},
                {e.meta.get("subset") for e in statements},
                n_pairs,
                seed,
            )
        )
        return real(statements, n_pairs=n_pairs, seed=seed)

    monkeypatch.setattr(p5_persona_wording, "plain_stimuli", spy)
    config = _config()
    _execute(tiny_arch, tmp_path, conditions=("paper",))
    base, n = config.seed, config.n_eval_pairs
    assert calls == [
        ({"val"}, {None}, n, derive_seed(base, "eval", "facts", "val")),
        ({"test"}, {None}, n, derive_seed(base, "eval", "facts", "test")),
        ({"test"}, {"cities"}, n, derive_seed(base, "eval", "cities")),
        (
            {"test"},
            {"larger_than"},
            n,
            derive_seed(base, "eval", "larger_than"),
        ),
    ]


def test_run_loads_the_model_the_facts_and_the_subsets(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts = _statements(count=4)
    got = _got()
    asked: list[Any] = []
    seen: list[Any] = []

    def load_rows(*, subsets: Any) -> list[Example]:
        asked.append(subsets)
        return got

    def record(
        config: Config,
        run_dir: RunDir,
        loaded: LoadedModel,
        statements: Any,
        rows: Any,
    ) -> None:
        seen.append((loaded, statements, rows))

    monkeypatch.setattr(p5_persona_wording, "load_facts", lambda: facts)
    monkeypatch.setattr(p5_persona_wording, "load_geometry_of_truth", load_rows)
    monkeypatch.setattr(p5_persona_wording, "load_model", lambda key: tiny_arch)
    monkeypatch.setattr(p5_persona_wording, "execute", record)
    config = _config(got_subsets=("cities",))
    run(config, RunDir.create(config, out_root=tmp_path))
    assert asked == [("cities",)]
    assert seen == [(tiny_arch, facts, got)]


def test_the_reading_settings_come_from_the_config(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweeps: list[Any] = []
    activations: list[Any] = []
    curves: list[Any] = []
    real_sweep = p5_persona_wording.sweep_from_activations
    real_activations = p5_persona_wording.stimulus_activations
    real_curves = p5_persona_wording.write_curves

    def sweep_spy(*args: Any, **kwargs: Any) -> Any:
        sweeps.append(kwargs)
        return real_sweep(*args, **kwargs)

    def activations_spy(loaded: Any, stimuli: Any, **kwargs: Any) -> Any:
        activations.append(kwargs)
        return real_activations(loaded, stimuli, **kwargs)

    def curves_spy(*args: Any, **kwargs: Any) -> Any:
        curves.append(kwargs)
        return real_curves(*args, **kwargs)

    monkeypatch.setattr(p5_persona_wording, "sweep_from_activations", sweep_spy)
    monkeypatch.setattr(
        p5_persona_wording, "stimulus_activations", activations_spy
    )
    monkeypatch.setattr(p5_persona_wording, "write_curves", curves_spy)
    settings: dict[str, Any] = {
        "method": "mean_diff",
        "preprocess": "unit",
        "bos": "none",
        "batch_size": 3,
        "layers": (1, 3),
        "n_boot": 17,
        "seed": 5,
        "conditions": ("paper", "fact_only"),
    }
    run_dir = _execute(tiny_arch, tmp_path, **settings)
    assert {(s["method"], s["preprocess"], s["seed"]) for s in sweeps} == {
        ("mean_diff", "unit", 5)
    }
    assert {(a["bos"], a["batch_size"]) for a in activations} == {("none", 3)}
    assert {a["layers"] for a in activations} == {(1, 3)}
    assert {c["n_boot"] for c in curves} == {17}
    assert run_dir.read_arrays("curves_id")["x"].tolist() == [1.0, 3.0]


def test_fact_only_evaluates_on_the_shared_facts_activations(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evals: dict[str, Any] = {}
    real = p5_persona_wording.sweep_from_activations

    def spy(train: Any, evaluation: Any, **kwargs: Any) -> Any:
        key = "fact_only" if len(evals) else "paper"
        evals[key] = dict(evaluation)
        return real(train, evaluation, **kwargs)

    monkeypatch.setattr(p5_persona_wording, "sweep_from_activations", spy)
    _execute(tiny_arch, tmp_path, conditions=("paper", "fact_only"))
    paper, fact_only = evals["paper"], evals["fact_only"]
    assert list(fact_only) == list(paper)
    assert fact_only["id"] is fact_only["facts"]
    assert fact_only["id_val"] is fact_only["facts_val"]
    assert paper["id"] is not paper["facts"]
    assert paper["id_val"] is not paper["facts_val"]
    for name in ("facts", "facts_val", "got_cities", "got_larger_than"):
        assert fact_only[name] is paper[name]


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("p5")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_plot_writes_every_curves_pair_and_the_persona_scatter(
    ran: RunDir, tmp_path: Path
) -> None:
    paths = plot(ran, tmp_path / "figs")
    stems = [
        "curves_cosine",
        "curves_facts",
        "curves_facts_val",
        "curves_got_cities",
        "curves_got_larger_than",
        "curves_id",
        "curves_id_val",
        "persona_scatter",
    ]
    assert [p.stem for p in paths if p.suffix == ".pdf"] == stems
    assert sorted(p.name for p in paths) == sorted(
        f"{stem}.{ext}" for stem in stems for ext in ("pdf", "png", "svg")
    )
    assert len(paths) == 24
    assert all(p.stat().st_size > 0 for p in paths)
    assert sorted(p.name for p in (tmp_path / "figs").iterdir()) == sorted(
        p.name for p in paths
    )


def test_the_persona_scatter_puts_each_condition_at_its_two_accuracies(
    ran: RunDir, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved: dict[str, Any] = {}
    real = p5_persona_wording.save_figure

    def spy(fig: Any, out_dir: Any, name: str, **kwargs: Any) -> Any:
        saved[name] = fig
        return real(fig, out_dir, name, **kwargs)

    monkeypatch.setattr(p5_persona_wording, "save_figure", spy)
    plot(ran, tmp_path)
    ax = saved["persona_scatter"].axes[0]
    rows = _table(ran)
    points = {
        line.get_label(): line
        for line in ax.lines
        if line.get_label() in CONDITIONS
    }
    assert list(points) == list(CONDITIONS)
    segments = [c.get_segments() for c in ax.collections]
    assert len(segments) == 2 * len(CONDITIONS)
    for i, condition in enumerate(CONDITIONS):
        row = rows[condition]
        x, y = points[condition].get_data()
        assert (x[0], y[0]) == pytest.approx((row["id"], row["facts"]))
        np.testing.assert_allclose(
            segments[2 * i][0],
            [[row["id_lo"], row["facts"]], [row["id_hi"], row["facts"]]],
        )
        np.testing.assert_allclose(
            segments[2 * i + 1][0],
            [[row["id"], row["facts_lo"]], [row["id"], row["facts_hi"]]],
        )
    chance = [
        line
        for line in ax.lines
        if line.get_label() not in CONDITIONS
        and 0.5 in (*line.get_xdata(), *line.get_ydata())
    ]
    assert len(chance) == 2
    assert ax.get_xlabel() == "In-distribution accuracy"
    assert ax.get_ylabel() == "Facts accuracy"


def test_plot_of_the_paper_condition_alone_has_no_cosine_figure(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(tiny_arch, tmp_path, conditions=("paper",))
    paths = plot(run_dir, tmp_path / "figs")
    stems = {p.stem for p in paths}
    assert "curves_cosine" not in stems
    assert "persona_scatter" in stems
    assert len(paths) == 7 * 3


@pytest.mark.slow
def test_p5_on_gpt2_reads_the_persona_conditions(tmp_path: Path) -> None:
    config = Config(
        n_seeds=2,
        n_pairs=32,
        conditions=("paper", "unrelated_0", "fact_only"),
    )
    try:
        facts = load_facts(offline=True)
        got = load_geometry_of_truth(subsets=config.got_subsets, offline=True)
    except OfflineCacheMissError:
        pytest.skip("the facts or Geometry of Truth cache is missing")
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, load_model("gpt2"), facts, got)
    table = run_dir.read_json("table")
    assert isinstance(table, dict)
    for row in table["rows"]:
        print(
            dict(
                zip(
                    table["columns"],
                    [round(v, 3) if isinstance(v, float) else v for v in row],
                    strict=True,
                )
            )
        )
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    print(summary)
    rows = _table(run_dir)
    assert rows["paper"]["id"] > 0.9
    for row in rows.values():
        for name in ("id", "facts", *(f"got_{s}" for s in config.got_subsets)):
            assert 0.0 <= row[name] <= 1.0
