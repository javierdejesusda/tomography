"""Tests for the P10b reading experiment."""

from collections.abc import Callable
from collections.abc import Sequence
import csv
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.cli import load_experiment
from tomography.datasets.quantities import FAMILIES
from tomography.datasets.quantities import QuantityQuestion
from tomography.experiments import p10b_reading
from tomography.experiments.base import RunDir
from tomography.experiments.p10b_reading import BASELINES
from tomography.experiments.p10b_reading import Config
from tomography.experiments.p10b_reading import baseline_scores
from tomography.experiments.p10b_reading import collect_question_activations
from tomography.experiments.p10b_reading import execute
from tomography.experiments.p10b_reading import plot
from tomography.experiments.p10b_reading import read_study
from tomography.experiments.p10b_reading import run
from tomography.experiments.p10b_reading import train_sets
from tomography.experiments.sweep import chat_format_for
from tomography.model import LoadedModel
from tomography.p9.reading import ItemActivations
from tomography.reading.metrics import auroc
from tomography.reading.reader import RepReader

DEFAULT_HASH = "9d9170f5cccb"


def test_config_defaults_and_round_trip() -> None:
    config = Config()

    assert config.experiment == "p10b"
    assert config.model == "gpt2"
    assert config.seed == 0
    assert config.families is None
    assert config.n_per_side == 120
    assert config.min_sitelinks == 5
    assert config.people is True
    assert config.offline is False
    assert config.templates == ("plain", "concept")
    assert config.concept == "variability"
    assert config.methods == ("logistic", "pca_diff", "mean_diff", "random")
    assert config.n_seeds == 5
    assert config.n_boot == 1000
    assert config.batch_size == 8
    assert Config.from_dict(config.to_dict()) == config


def test_a_family_subset_round_trips() -> None:
    config = Config(families=("height", "duration"), people=False)

    assert Config.from_dict(config.to_dict()) == config


def test_the_default_config_hash_is_stable() -> None:
    assert Config().config_hash() == DEFAULT_HASH


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("families", ()),
        ("families", ("height", "height")),
        ("families", ("height", "no such family")),
        ("n_per_side", 0),
        ("n_seeds", 0),
        ("n_boot", 0),
        ("batch_size", 0),
        ("min_sitelinks", -1),
        ("concept", ""),
        ("templates", ()),
        ("templates", ("plain", "plain")),
        ("templates", ("fewshot",)),
        ("methods", ()),
        ("methods", ("logistic", "logistic")),
        ("methods", ("kmeans",)),
    ],
)
def test_bad_settings_raise_before_any_work(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        Config(**{field: value})


def test_one_family_is_its_own_train_set() -> None:
    assert train_sets(["height"]) == {"height": ("height",)}


def test_two_families_add_all_and_no_leave_one_out() -> None:
    assert train_sets(["height", "wingspan"]) == {
        "height": ("height",),
        "wingspan": ("wingspan",),
        "all": ("height", "wingspan"),
    }


def test_three_families_add_the_leave_one_out_pairs() -> None:
    names = [f.name for f in FAMILIES]

    sets = train_sets(names)

    assert list(sets) == [
        "height",
        "wingspan",
        "duration",
        "wingspan+duration",
        "height+duration",
        "height+wingspan",
        "all",
    ]
    assert sets["height+wingspan"] == ("height", "wingspan")
    assert sets["all"] == tuple(names)


def test_train_sets_follow_the_order_of_the_families_module() -> None:
    sets = train_sets(["duration", "height"])

    assert sets["all"] == ("height", "duration")


def _item(
    uid: str,
    family: str,
    kind: str,
    question: str,
    *,
    sitelinks: int = 10,
    split: str = "train",
) -> QuantityQuestion:
    return QuantityQuestion(
        uid=uid,
        family=family,
        kind=kind,  # type: ignore[arg-type]
        entity=uid,
        entity_id=f"Q{abs(hash(uid)) % 10000}",
        question=question,
        low=1.0,
        high=2.0 if kind == "range" else 1.0,
        unit="metres",
        sitelinks=sitelinks,
        split=split,  # type: ignore[arg-type]
        retrieved_on="2026-01-01",
        sha256="0" * 64,
    )


def _text_items(
    range_text: str, exact_text: str, *, range_links: int, exact_links: int
) -> list[QuantityQuestion]:
    items = []
    for split in ("train", "val", "test"):
        for i in range(12):
            items.append(
                _item(
                    f"height-range-{split}{i}",
                    "height",
                    "range",
                    f"{range_text} number {i}?",
                    sitelinks=range_links + i,
                    split=split,
                )
            )
            items.append(
                _item(
                    f"height-exact-{split}{i}",
                    "height",
                    "exact",
                    f"{exact_text} number {i}?",
                    sitelinks=exact_links + i,
                    split=split,
                )
            )
    return items


def _test_auroc(
    items: list[QuantityQuestion], scores: NDArray[np.float64]
) -> float:
    rows = [i for i, item in enumerate(items) if item.split == "test"]
    labels = np.array([items[i].kind == "range" for i in rows], dtype=np.int64)
    return float(auroc(scores[rows], labels))


def test_baselines_read_the_wording_the_length_and_the_popularity() -> None:
    items = _text_items(
        "How tall is a dog breed of the kind",
        "How tall is the building",
        range_links=1,
        exact_links=500,
    )

    scores = baseline_scores(items, ("height",))

    assert set(scores) == set(BASELINES)
    for name in BASELINES:
        assert scores[name].shape == (len(items),)
        assert _test_auroc(items, scores[name]) > 0.9, name


def test_baselines_are_oriented_on_the_train_items() -> None:
    items = _text_items(
        "Short",
        "A much longer question about the kind of thing",
        range_links=900,
        exact_links=1,
    )

    scores = baseline_scores(items, ("height",))

    assert _test_auroc(items, scores["length"]) > 0.9
    assert _test_auroc(items, scores["popularity"]) > 0.9


def test_baselines_fit_only_on_the_train_families() -> None:
    items = _text_items(
        "How tall is a dog breed of the kind",
        "How tall is the building",
        range_links=1,
        exact_links=500,
    )
    other = [
        _item(f"wingspan-{k}-{i}", "wingspan", k, f"Wing {k} {i}?", sitelinks=3)
        for k in ("range", "exact")
        for i in range(6)
    ]

    scores = baseline_scores([*items, *other], ("height",))

    assert scores["popularity"].shape == (len(items) + len(other),)
    assert _test_auroc(items, scores["popularity"][: len(items)]) > 0.9


WIDTH = 2048
LAYERS = (1, 2)
SPLITS = (("train", 16), ("val", 8), ("test", 16))
Shift = Callable[[QuantityQuestion], float]


def _study_items(
    families: Sequence[str] = ("height", "wingspan", "duration"),
    n_people: int = 40,
) -> list[QuantityQuestion]:
    items = [
        _item(
            f"{family}-{kind}-{split}{i}",
            family,
            kind,
            f"The {family} of the {kind} {split} thing {i}?",
            sitelinks=5 + i,
            split=split,
        )
        for family in families
        for kind in ("range", "exact")
        for split, n in SPLITS
        for i in range(n)
    ]
    items += [
        _item(
            f"person_height-exact-q{i}",
            "person_height",
            "exact",
            f"How tall is person {i}?",
            sitelinks=100 + i,
            split="test",
        )
        for i in range(n_people)
    ]
    return items


def _planted(
    items: Sequence[QuantityQuestion], shift: Shift
) -> ItemActivations:
    rng = np.random.default_rng(0)
    direction = np.random.default_rng(123).normal(size=WIDTH)
    direction /= np.linalg.norm(direction)
    data = rng.normal(0.0, 0.3, (len(LAYERS), len(items), WIDTH))
    for n, item in enumerate(items):
        data[:, n] += 3.0 * shift(item) * direction
    return ItemActivations(
        tuple(i.uid for i in items), LAYERS, data.astype(np.float32)
    )


def _by_range(item: QuantityQuestion) -> float:
    if item.family == "person_height":
        return -1.0
    return 1.0 if item.kind == "range" else -1.0


def _config(**changes: object) -> Config:
    base: dict[str, object] = {
        "templates": ("plain",),
        "n_seeds": 2,
        "n_boot": 40,
    }
    return Config(**{**base, **changes})


def _rows(
    rows: list[dict[str, object]], **where: object
) -> list[dict[str, object]]:
    return [r for r in rows if all(r[k] == v for k, v in where.items())]


def test_a_planted_direction_transfers_and_the_random_control_does_not() -> (
    None
):
    items = _study_items()
    study = read_study(_config(), {"plain": _planted(items, _by_range)}, items)

    control = []
    for row in study.transfer:
        if row["method"] == "random":
            control.append(float(row["auroc"]))
        else:
            assert float(row["auroc"]) > 0.9, row
    assert float(np.mean(control)) < 0.75
    assert max(control) < 0.95


def test_curves_and_transfer_have_one_row_per_cell() -> None:
    items = _study_items()
    config = _config(methods=("logistic", "random"))
    study = read_study(config, {"plain": _planted(items, _by_range)}, items)

    n_sets = 7
    n_tests = 4
    assert len(study.transfer) == 2 * n_sets * n_tests
    assert len(study.curves) == 2 * n_sets * n_tests * len(LAYERS)
    assert set(study.transfer[0]) == {
        "template",
        "method",
        "train_set",
        "test_family",
        "layer",
        "n_range",
        "n_exact",
        "auroc",
        "lo",
        "hi",
    }
    assert set(study.curves[0]) == {
        "template",
        "method",
        "train_set",
        "test_family",
        "layer",
        "auroc",
        "lo",
        "hi",
    }
    tests = {r["test_family"] for r in study.transfer}
    assert tests == {"height", "wingspan", "duration", "pooled"}
    pooled = _rows(study.transfer, test_family="pooled")[0]
    assert (pooled["n_range"], pooled["n_exact"]) == (48, 48)
    one = _rows(study.transfer, test_family="height")[0]
    assert (one["n_range"], one["n_exact"]) == (16, 16)
    for row in study.transfer:
        assert float(row["lo"]) <= float(row["auroc"]) <= float(row["hi"])


def test_the_layer_is_chosen_on_the_validation_items_of_the_train_set() -> None:
    items = _study_items()
    acts = _planted(items, _by_range)
    data = acts.data.copy()
    data[0] = np.random.default_rng(1).normal(0.0, 0.3, data[0].shape)
    noisy_first = ItemActivations(acts.uids, acts.layers, data)

    study = read_study(
        _config(methods=("logistic",)), {"plain": noisy_first}, items
    )

    assert {r["layer"] for r in study.transfer} == {2}


def test_a_flipped_plant_still_reads_above_chance() -> None:
    items = _study_items()

    def flipped(item: QuantityQuestion) -> float:
        return -_by_range(item)

    study = read_study(_config(), {"plain": _planted(items, flipped)}, items)

    for row in study.transfer:
        if row["method"] != "random":
            assert float(row["auroc"]) > 0.9, row


def _diagnostic(
    rows: list[dict[str, object]], train_set: str
) -> dict[str, object]:
    return _rows(rows, source="reader", method="logistic", train_set=train_set)[
        0
    ]


def test_people_planted_with_the_exact_side_read_as_exact() -> None:
    items = _study_items()
    study = read_study(_config(), {"plain": _planted(items, _by_range)}, items)

    for train_set in ("all", "wingspan+duration"):
        row = _diagnostic(study.people, train_set)
        assert float(row["breeds_vs_people"]) > 0.9
        assert abs(float(row["people_vs_buildings"]) - 0.5) < 0.2
        assert abs(float(row["position"])) < 0.2


def test_people_planted_with_the_range_side_read_as_range() -> None:
    items = _study_items()

    def living(item: QuantityQuestion) -> float:
        return 1.0 if item.family == "person_height" else _by_range(item)

    study = read_study(_config(), {"plain": _planted(items, living)}, items)

    for train_set in ("all", "wingspan+duration"):
        row = _diagnostic(study.people, train_set)
        assert abs(float(row["breeds_vs_people"]) - 0.5) < 0.2
        assert float(row["people_vs_buildings"]) > 0.9
        assert abs(float(row["position"]) - 1.0) < 0.2


def test_the_people_diagnostic_covers_readers_and_baselines() -> None:
    items = _study_items()
    study = read_study(_config(), {"plain": _planted(items, _by_range)}, items)

    assert {r["source"] for r in study.people} == {"reader", "baseline"}
    names = {r["method"] for r in _rows(study.people, source="baseline")}
    assert names == set(BASELINES)
    row = _rows(study.people, source="reader")[0]
    assert row["layer"] in LAYERS
    assert (row["n_breeds"], row["n_buildings"], row["n_people"]) == (
        16,
        16,
        40,
    )
    assert float(row["breeds_vs_people_lo"]) <= float(row["breeds_vs_people"])
    assert float(row["people_vs_buildings_hi"]) >= float(
        row["people_vs_buildings"]
    )


def test_there_is_no_people_diagnostic_without_the_flag_or_the_family() -> None:
    items = _study_items()
    acts = {"plain": _planted(items, _by_range)}

    assert read_study(_config(people=False), acts, items).people == []

    no_height = [i for i in items if i.family != "height"]
    acts = {"plain": _planted(no_height, _by_range)}
    study = read_study(_config(), acts, no_height)
    assert study.people == []
    assert {r["train_set"] for r in study.transfer} == {
        "wingspan",
        "duration",
        "all",
    }


def test_baseline_rows_cover_every_train_set_and_test_family() -> None:
    items = _study_items()
    study = read_study(_config(), {"plain": _planted(items, _by_range)}, items)

    assert len(study.baselines) == len(BASELINES) * 7 * 4
    assert set(study.baselines[0]) == {
        "baseline",
        "train_set",
        "test_family",
        "n_range",
        "n_exact",
        "auroc",
        "lo",
        "hi",
    }
    assert {r["baseline"] for r in study.baselines} == set(BASELINES)


def _tiny_items(n_people: int = 6) -> list[QuantityQuestion]:
    items = [
        _item(
            f"{family}-{kind}-{split}{i}",
            family,
            kind,
            f"What is the {family} of the {kind} {split} thing number {i}?",
            sitelinks=5 + 3 * i,
            split=split,
        )
        for family in ("height", "wingspan", "duration")
        for kind in ("range", "exact")
        for split, n in (("train", 6), ("val", 3), ("test", 5))
        for i in range(n)
    ]
    return items + _study_items(n_people=n_people)[-n_people:]


def _tiny_config(**changes: object) -> Config:
    return _config(
        model="tiny", templates=("plain", "concept"), batch_size=4, **changes
    )


def _executed(loaded: LoadedModel, tmp_path: Path, **changes: object) -> RunDir:
    config = _tiny_config(**changes)
    run_dir = RunDir.create(config, out_root=tmp_path)
    items = _tiny_items()
    execute(
        config,
        run_dir,
        loaded,
        [i for i in items if i.family != "person_height"],
        [i for i in items if i.family == "person_height"],
    )
    return run_dir


def _csv(run_dir: RunDir, name: str) -> list[dict[str, str]]:
    path = run_dir.path / name
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_prompts_are_the_bare_question_or_the_concept_template(
    tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, list[str]] = {}

    def spy(loaded: LoadedModel, texts: Sequence[str], **kwargs: Any) -> Any:
        seen["texts"] = list(texts)
        seen["layers"] = list(kwargs["layers"])
        assert kwargs["position"] == "last"
        return SimpleNamespace(
            layers=tuple(kwargs["layers"]),
            data=np.zeros((len(kwargs["layers"]), len(texts), 32), np.float32),
        )

    monkeypatch.setattr(p10b_reading, "collect_activations", spy)
    items = _tiny_items()[:3]
    chat = chat_format_for(tiny_gpt2)

    plain = collect_question_activations(
        tiny_gpt2, items, "plain", chat, concept="variability", batch_size=4
    )
    assert seen["texts"] == [i.question for i in items]
    assert plain.uids == tuple(i.uid for i in items)
    assert plain.layers == (1, 2, 3)

    collect_question_activations(
        tiny_gpt2, items, "concept", chat, concept="variability", batch_size=4
    )
    assert all(
        "variability" in t and i.question in t
        for t, i in zip(seen["texts"], items, strict=True)
    )
    assert seen["texts"][0] != items[0].question


def test_execute_writes_every_file(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _executed(tiny_gpt2, tmp_path)

    for name in (
        "curves.csv",
        "transfer.csv",
        "baselines.csv",
        "people.csv",
        "readers.npz",
        "readers.json",
        "items.json",
        "summary.json",
    ):
        assert (run_dir.path / name).is_file(), name
    transfer = _csv(run_dir, "transfer.csv")
    assert list(transfer[0]) == [
        "template",
        "method",
        "train_set",
        "test_family",
        "layer",
        "n_range",
        "n_exact",
        "auroc",
        "lo",
        "hi",
    ]
    assert len(transfer) == 2 * 4 * 7 * 4
    assert list(_csv(run_dir, "baselines.csv")[0]) == [
        "baseline",
        "train_set",
        "test_family",
        "n_range",
        "n_exact",
        "auroc",
        "lo",
        "hi",
    ]
    people = _csv(run_dir, "people.csv")
    assert {r["source"] for r in people} == {"reader", "baseline"}
    assert {r["breeds_vs_people"] != "" for r in people} == {True}


def test_the_readers_of_the_full_train_set_are_saved(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _executed(tiny_gpt2, tmp_path)

    names = run_dir.read_json("readers")["readers"]  # type: ignore[index]
    arrays = run_dir.read_arrays("readers")

    assert names == [
        f"{t}/{m}"
        for t in ("plain", "concept")
        for m in ("logistic", "pca_diff", "mean_diff", "random")
    ]
    assert arrays["directions"].shape == (8, 3, 32)
    assert arrays["means"].shape == (8, 3, 32)
    assert arrays["signs"].shape == (8, 3)
    assert arrays["layers"].tolist() == [1, 2, 3]
    assert run_dir.read_json("readers")["train_set"] == "all"  # type: ignore[index]


def test_items_hold_the_averaged_score_of_the_full_readers(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _executed(tiny_gpt2, tmp_path)

    records = run_dir.read_json("items")
    items = _tiny_items()

    assert [r["uid"] for r in records] == [i.uid for i in items]  # type: ignore[index, union-attr]
    first = records[0]  # type: ignore[index]
    assert set(first) == {
        "uid",
        "family",
        "kind",
        "split",
        "sitelinks",
        "scores",
    }
    assert set(first["scores"]) == {
        f"{t}/{m}"
        for t in ("plain", "concept")
        for m in ("logistic", "pca_diff", "mean_diff", "random")
    }


def test_the_summary_reports_the_counts_the_layers_and_the_diagnostic(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _executed(tiny_gpt2, tmp_path)

    summary = run_dir.read_json("summary")
    transfer = _csv(run_dir, "transfer.csv")

    assert summary["model"] == "tiny"  # type: ignore[index]
    assert summary["counts"]["height"] == {  # type: ignore[index]
        "range": {"train": 6, "val": 3, "test": 5},
        "exact": {"train": 6, "val": 3, "test": 5},
    }
    assert summary["n_people"] == 6  # type: ignore[index]
    assert set(summary["snapshots"]) == {  # type: ignore[index]
        "height",
        "wingspan",
        "duration",
        "person_height",
    }
    assert summary["snapshots"]["height"]["range"]["sha256"] == ["0" * 64]  # type: ignore[index]
    assert summary["config"]["model"] == "tiny"  # type: ignore[index]
    assert summary["layers"]["plain/logistic/all"] in (1, 2, 3)  # type: ignore[index]
    pooled = next(
        r
        for r in transfer
        if (r["template"], r["method"], r["train_set"], r["test_family"])
        == ("plain", "logistic", "all", "pooled")
    )
    assert summary["pooled_all"]["plain/logistic"] == pytest.approx(  # type: ignore[index]
        float(pooled["auroc"])
    )
    held_out = summary["held_out"]["wingspan+duration"]  # type: ignore[index]
    assert held_out["test_family"] == "height"
    assert set(held_out["auroc"]) == set(summary["pooled_all"])  # type: ignore[index]
    people = summary["people"]  # type: ignore[index]
    assert set(people) == {"all", "wingspan+duration"}
    assert set(people["all"]["plain/logistic"]) == {
        "breeds_vs_people",
        "people_vs_buildings",
        "position",
    }


def test_there_is_no_people_file_without_the_people(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _executed(tiny_gpt2, tmp_path, people=False)

    assert not (run_dir.path / "people.csv").exists()
    assert run_dir.read_json("summary")["people"] == {}  # type: ignore[index]
    assert len(run_dir.read_json("items")) == len(_tiny_items()) - 6  # type: ignore[arg-type]


def test_run_passes_offline_to_both_loaders_and_skips_the_people_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loads: dict[str, dict[str, Any]] = {}
    sent: dict[str, Any] = {}
    items = _tiny_items()

    def fake_quantities(**kwargs: Any) -> list[QuantityQuestion]:
        loads["quantities"] = kwargs
        return [i for i in items if i.family != "person_height"]

    def fake_people(**kwargs: Any) -> list[QuantityQuestion]:
        loads["people"] = kwargs
        return [i for i in items if i.family == "person_height"]

    def fake_execute(*args: Any) -> None:
        sent["args"] = args

    monkeypatch.setattr(p10b_reading, "load_quantities", fake_quantities)
    monkeypatch.setattr(p10b_reading, "load_people_heights", fake_people)
    monkeypatch.setattr(p10b_reading, "load_model", lambda key: key)
    monkeypatch.setattr(p10b_reading, "execute", fake_execute)
    config = _tiny_config(
        families=("wingspan", "height"),
        n_per_side=7,
        min_sitelinks=9,
        offline=True,
        seed=4,
    )
    run_dir = RunDir.create(config, out_root=tmp_path)

    run(config, run_dir)

    quantities = loads["quantities"]
    assert [f.name for f in quantities["families"]] == ["height", "wingspan"]
    assert quantities["offline"] is True
    assert quantities["n_per_side"] == 7
    assert quantities["min_sitelinks"] == 9
    assert quantities["seed"] == 4
    assert loads["people"]["offline"] is True
    assert sent["args"][2] == "tiny"
    assert len(sent["args"][4]) == 6

    loads.clear()
    quiet = _tiny_config(people=False, offline=True)
    run(quiet, RunDir.create(quiet, out_root=tmp_path))
    assert "people" not in loads
    assert sent["args"][4] == []


def test_the_experiment_is_registered() -> None:
    experiment = load_experiment("p10b")

    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_plot_draws_a_curve_and_a_heatmap_per_template(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _executed(tiny_gpt2, tmp_path)

    paths = plot(run_dir, tmp_path / "figs")

    assert sorted(p.name for p in paths) == sorted(
        f"{stem}.{ext}"
        for stem in (
            "curves_plain",
            "curves_concept",
            "transfer_plain",
            "transfer_concept",
        )
        for ext in ("pdf", "png", "svg")
    )
    assert all(p.is_file() for p in paths)


Signal = Callable[[QuantityQuestion, int], float]


def _shaped(
    items: Sequence[QuantityQuestion], signal: Signal, *, seed: int = 0
) -> ItemActivations:
    """Plants `signal(item, layer position)` along one direction."""
    rng = np.random.default_rng(seed)
    direction = np.random.default_rng(123).normal(size=WIDTH)
    direction /= np.linalg.norm(direction)
    data = rng.normal(0.0, 0.3, (len(LAYERS), len(items), WIDTH))
    for n, item in enumerate(items):
        for k in range(len(LAYERS)):
            data[k, n] += 3.0 * signal(item, k) * direction
    return ItemActivations(
        tuple(i.uid for i in items), LAYERS, data.astype(np.float32)
    )


def _sign(item: QuantityQuestion) -> float:
    return 1.0 if item.kind == "range" else -1.0


def _row(rows: list[dict[str, object]], **where: object) -> dict[str, object]:
    (row,) = _rows(rows, **where)
    return row


def test_a_held_out_family_never_trains_its_reader() -> None:
    items = _study_items()

    def only_height(item: QuantityQuestion, k: int) -> float:
        return _sign(item) if item.family == "height" else 0.0

    config = _config(methods=("logistic",))
    study = read_study(config, {"plain": _shaped(items, only_height)}, items)

    row = _row(
        study.transfer,
        train_set="wingspan+duration",
        test_family="height",
    )
    assert float(row["auroc"]) < 0.75

    heights = [
        n
        for n, i in enumerate(items)
        if i.family == "height" and i.split == "train"
    ]
    base = _shaped(items, only_height)
    scrambled = np.array(base.data)
    scrambled[:, heights] = np.random.default_rng(9).normal(
        0.0, 1.0, scrambled[:, heights].shape
    )
    again = read_study(
        config,
        {"plain": ItemActivations(base.uids, base.layers, scrambled)},
        items,
    )
    key = "plain/logistic/wingspan+duration"
    kept = np.setdiff1d(np.arange(len(items)), heights)
    np.testing.assert_array_equal(
        study.scores[key][kept], again.scores[key][kept]
    )


def _layer_trap(item: QuantityQuestion, k: int) -> float:
    """Layer 1 informs on val and not on test, layer 2 the reverse."""
    if item.split == "train":
        return _sign(item)
    informative = item.split == ("val" if k == 0 else "test")
    return _sign(item) if informative else 0.0


def test_the_layer_is_never_chosen_on_the_test_items() -> None:
    items = _study_items()
    study = read_study(
        _config(methods=("logistic",)),
        {"plain": _shaped(items, _layer_trap)},
        items,
    )

    assert {r["layer"] for r in study.transfer} == {1}


def test_the_curves_are_computed_on_the_test_items() -> None:
    items = _study_items()
    study = read_study(
        _config(methods=("logistic",)),
        {"plain": _shaped(items, _layer_trap)},
        items,
    )

    where = {"train_set": "all", "test_family": "pooled"}
    first = _row(study.curves, layer=1, **where)
    second = _row(study.curves, layer=2, **where)
    assert float(first["auroc"]) < 0.7
    assert float(second["auroc"]) > 0.9


def test_a_pair_never_crosses_two_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[Any, Any]] = []
    real = p10b_reading.label_contrastive_pairs

    def spy(labels: Any, **kwargs: Any) -> Any:
        pairs = real(labels, **kwargs)
        seen.append((kwargs.get("strata"), pairs))
        return pairs

    monkeypatch.setattr(p10b_reading, "label_contrastive_pairs", spy)
    items = _study_items()

    read_study(
        _config(methods=("pca_diff",)),
        {"plain": _planted(items, _by_range)},
        items,
    )

    assert seen
    for strata, pairs in seen:
        assert strata is not None
        assert (strata[pairs[:, 0]] == strata[pairs[:, 1]]).all()
    assert any(len(set(strata.tolist())) == 3 for strata, _ in seen)


def _marked(
    range_word: str,
    exact_word: str,
    *,
    split: str,
    n: int,
    long: bool = False,
) -> list[QuantityQuestion]:
    pad = " and some more words to make this question much longer" * 3
    return [
        _item(
            f"height-{kind}-{split}{i}",
            "height",
            kind,
            f"{word} thing {i}?" + (pad if kind == "range" and long else ""),
            split=split,
        )
        for kind, word in (("range", range_word), ("exact", exact_word))
        for i in range(n)
    ]


def test_the_baselines_never_see_the_val_or_test_labels() -> None:
    train = _marked("alpha", "beta", split="train", n=12)
    held = _marked("beta", "alpha", split="test", n=12) + _marked(
        "beta", "alpha", split="val", n=12
    )
    flipped = [
        dataclasses.replace(i, kind="exact" if i.kind == "range" else "range")
        for i in held
    ]

    first = baseline_scores([*train, *held], ("height",))
    second = baseline_scores([*train, *flipped], ("height",))

    for name in BASELINES:
        np.testing.assert_array_equal(first[name], second[name])
    assert _test_auroc([*train, *held], first["text_word"]) < 0.5


def test_the_baselines_are_oriented_on_the_train_items_only() -> None:
    train = [
        dataclasses.replace(i, sitelinks=100 if i.kind == "range" else 2)
        for i in _marked("alpha", "beta", split="train", n=12, long=True)
    ]
    test = [
        dataclasses.replace(i, sitelinks=2 if i.kind == "range" else 100)
        for i in _marked("alpha", "beta", split="test", n=60, long=False)
    ]
    test = [
        dataclasses.replace(
            i,
            question=i.question
            + (" with a very long tail" * 6 if i.kind == "exact" else ""),
        )
        for i in test
    ]
    items = [*train, *test]

    scores = baseline_scores(items, ("height",))

    assert _test_auroc(items, scores["length"]) < 0.5
    assert _test_auroc(items, scores["popularity"]) < 0.5


def test_the_selected_layer_averages_every_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = _study_items()
    acts = _planted(items, _by_range)
    direction = np.random.default_rng(123).normal(size=WIDTH)
    other = np.random.default_rng(7).normal(size=WIDTH)
    first_draw = p10b_reading.draw_seed(0, 0)

    def stub(
        data: Any,
        items: Any,
        labels: Any,
        train: Any,
        layers: Any,
        *,
        method: Any,
        seed: int,
        fit_seed: int,
    ) -> RepReader:
        informative = 0 if seed == first_draw else 1
        vectors = np.stack([other, other])
        vectors[informative] = direction
        return RepReader.from_arrays(
            "stub", layers, vectors, np.zeros_like(vectors)
        )

    monkeypatch.setattr(p10b_reading, "_fit_reader", stub)

    study = read_study(
        _config(methods=("logistic",), n_seeds=3),
        {"plain": acts},
        items,
    )

    assert {r["layer"] for r in study.transfer} == {2}


def test_the_random_control_differs_across_train_sets_and_templates() -> None:
    items = _study_items()
    one_layer = _shaped(items, lambda item, k: _sign(item))
    one_layer = ItemActivations(
        one_layer.uids, one_layer.layers[:1], one_layer.data[:1]
    )
    config = _config(
        templates=("plain", "concept"),
        methods=("random",),
        n_seeds=1,
        n_boot=10,
    )

    study = read_study(
        config, {"plain": one_layer, "concept": one_layer}, items
    )

    def correlation(a: str, b: str) -> float:
        return abs(float(np.corrcoef(study.scores[a], study.scores[b])[0, 1]))

    assert correlation("plain/random/height", "plain/random/wingspan") < 0.9
    assert correlation("plain/random/all", "concept/random/all") < 0.9
