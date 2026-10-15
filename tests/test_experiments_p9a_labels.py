"""Tests for the P9a labeling experiment."""

from collections.abc import Callable
import csv
import dataclasses
from pathlib import Path
import time
from typing import Any

import numpy as np
import pytest

from tomography.cli import load_experiment
from tomography.datasets.records import QAExample
from tomography.datasets.records import assign_splits
from tomography.datasets.sources import OfflineCacheMissError
from tomography.experiments import p9a_labels
from tomography.experiments.base import RunDir
from tomography.experiments.p9a_labels import Config
from tomography.experiments.p9a_labels import execute
from tomography.experiments.p9a_labels import load_labeled
from tomography.experiments.p9a_labels import plot
from tomography.experiments.p9a_labels import run
from tomography.knowledge import DEFAULT_SHOTS
from tomography.knowledge import ItemLabel
from tomography.knowledge import LabelRecipe
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.p9.pool import GROUPS
from tomography.p9.pool import PoolLoaders
from tomography.p9.pool import group_of

RELATIONS = ("father", "mother", "occupation", "sport")
POPULARITY = (50_000, 5_000, 100, 20_000)
YEARS = range(2005, 2017)
FIELDS = ("known_raw", "known", "unknown", "uncertain")
CAPS = {
    "n_popqa": 4,
    "n_synthal": 4,
    "n_kuq": 2,
    "n_falseqa": 3,
    "n_triviaqa": 6,
    "n_nq_open": 4,
}


def _make(prefix: str, n: int, source: str, **fields: Any) -> list[QAExample]:
    """Builds `n` synthetic questions; `fields` hold per-index callables."""
    uids = [f"{prefix}:{i}" for i in range(n)]
    splits = assign_splits(uids, dataset=prefix)
    return [
        QAExample(
            uid=uid,
            question=fields["question"](i),
            answers=fields["answers"](i) if "answers" in fields else (),
            kind=fields["kind"](i),
            split=split,
            meta={"source": source, **fields["meta"](i)},
        )
        for i, (uid, split) in enumerate(zip(uids, splits, strict=True))
    ]


def _loaders(*, awards: bool = True) -> PoolLoaders:
    """A small pool; the sport answers are all the same constant."""
    popqa = _make(
        "popqa",
        32,
        "popqa",
        question=lambda i: f"Who is the {RELATIONS[i % 4]} of Person {i}?",
        answers=lambda i: (
            ("football",) if RELATIONS[i % 4] == "sport" else (f"Answer {i}",)
        ),
        kind=lambda i: "real",
        meta=lambda i: {
            "relation": RELATIONS[i % 4],
            "subject": f"Person Number{i}",
            "popularity": POPULARITY[(i // 4) % 4],
        },
    )
    synthal = _make(
        "synthal",
        8,
        "synthal",
        question=lambda i: f"Who is the {RELATIONS[i % 4]} of Zed Qux{i}?",
        kind=lambda i: "fictional",
        meta=lambda i: {
            "relation": RELATIONS[i % 4],
            "subject": f"Zed Qux{i}",
            "popularity": None,
        },
    )
    winners = _make(
        "wikidata:award",
        2 * len(YEARS),
        "wikidata",
        question=lambda i: f"Who won the Award{i // 12} in {2005 + i % 12}?",
        answers=lambda i: (f"Laureate {i}",),
        kind=lambda i: "real",
        meta=lambda i: {
            "relation": f"Award{i // 12}",
            "year": 2005 + i % 12,
            "popularity": None,
        },
    )
    details = _make(
        "wikidata:detail",
        4,
        "wikidata",
        question=lambda i: f"What did Person{i} eat on 3 May 1850?",
        kind=lambda i: "unknowable_detail",
        meta=lambda i: {"relation": "ordinary_date", "popularity": 5_000},
    )
    falseqa = _make(
        "falseqa",
        6,
        "falseqa",
        question=lambda i: f"Why did the moon landing happen in {1900 + i}?",
        kind=lambda i: "false_premise",
        meta=lambda i: {},
    )
    kuq = _make(
        "kuq",
        8,
        "kuq",
        question=lambda i: f"What will happen in {2100 + i}?",
        kind=lambda i: "kuq:future" if i < 4 else "kuq:assumption",
        meta=lambda i: {"category": "future" if i < 4 else "assumption"},
    )
    trivia = _make(
        "triviaqa",
        12,
        "triviaqa",
        question=lambda i: f"Trivia question {i}?",
        answers=lambda i: (f"T{i}",),
        kind=lambda i: "open",
        meta=lambda i: {},
    )
    nq = _make(
        "nq_open",
        8,
        "nq_open",
        question=lambda i: f"NQ question {i}?",
        answers=lambda i: (f"N{i}",),
        kind=lambda i: "open",
        meta=lambda i: {},
    )
    return PoolLoaders(
        popqa=lambda: popqa,
        synthal=lambda: synthal,
        kuq=lambda: kuq,
        falseqa=lambda: falseqa,
        triviaqa=lambda: trivia,
        nq_open=lambda: nq,
        awards=lambda: winners if awards else [],
        details=lambda: details,
    )


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        n_popqa=None,
        n_kuq=None,
        n_falseqa=None,
        n_triviaqa=None,
        n_nq_open=None,
        n_shots=2,
        max_new_tokens=4,
        n_hand_check=12,
        cutoff_year=2010,
    )
    return dataclasses.replace(base, **overrides)


def _plant(
    loaded: LoadedModel,
    examples: list[QAExample],
    *,
    recipe: LabelRecipe,
    shots: tuple[tuple[str, str], ...],
    seed: int,
    batch_size: int,
) -> list[ItemLabel]:
    """Labels by rule: popular PopQA and early awards are known."""
    labels = []
    for example in examples:
        if not example.answers:
            labels.append(
                ItemLabel(
                    example.uid, "unknown", "construction", "", False, (), 0
                )
            )
            continue
        gold = example.answers[0]
        group = group_of(example)
        if group == "popqa":
            popularity = example.meta["popularity"]
            assert isinstance(popularity, int)
            known = example.meta["relation"] == "sport" or popularity >= 10_000
            uncertain = not known and popularity >= 1_000
        elif group == "award":
            year = example.meta["year"]
            assert isinstance(year, int)
            known = year <= 2010
            uncertain = False
        else:
            known, uncertain = False, False
        if known:
            label = ItemLabel(
                example.uid, "known", "behavior", gold, True, (gold,) * 5, 5
            )
        elif uncertain:
            label = ItemLabel(
                example.uid,
                "uncertain",
                "behavior",
                gold,
                True,
                (gold, *("wrong",) * 4),
                1,
            )
        else:
            label = ItemLabel(
                example.uid,
                "unknown",
                "behavior",
                "wrong",
                False,
                ("wrong",) * 5,
                0,
            )
        labels.append(label)
    return labels


def _plant_constant_father(
    loaded: LoadedModel, examples: list[QAExample], **kwargs: Any
) -> list[ItemLabel]:
    """As `_plant`, but every father answer is "John" (known if gold)."""
    labels = _plant(loaded, examples, **kwargs)
    planted = []
    for example, label in zip(examples, labels, strict=True):
        if example.meta.get("relation") != "father" or not example.answers:
            planted.append(label)
            continue
        hit = example.answers == ("John",)
        planted.append(
            ItemLabel(
                example.uid,
                "known" if hit else "unknown",
                "behavior",
                "John",
                hit,
                ("John",) * 5,
                5 if hit else 0,
            )
        )
    return planted


def _regold(
    loaders: PoolLoaders,
    field: str,
    gold: Callable[[QAExample], tuple[str, ...] | None],
) -> PoolLoaders:
    """Swaps the gold answers of a source where `gold` returns answers."""
    load: Callable[[], list[QAExample]] = getattr(loaders, field)

    def regolded() -> list[QAExample]:
        return [
            e
            if (answers := gold(e)) is None
            else dataclasses.replace(e, answers=answers)
            for e in load()
        ]

    return dataclasses.replace(loaders, **{field: regolded})


def _planted_run(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    plant: Callable[..., list[ItemLabel]] = _plant,
    loaders: PoolLoaders | None = None,
    **overrides: Any,
) -> RunDir:
    monkeypatch.setattr(p9a_labels, "label_questions", plant)
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, loaders or _loaders())
    return run_dir


def _summary(run_dir: RunDir) -> dict[str, Any]:
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    return summary


def _records(run_dir: RunDir) -> list[dict[str, Any]]:
    records = run_dir.read_json("items")
    assert isinstance(records, list)
    return records


def _hand_check(run_dir: RunDir) -> str:
    return (run_dir.path / "hand_check.csv").read_text(encoding="utf-8")


def _group(record: dict[str, Any]) -> str:
    source = record["meta"]["source"]
    if source == "wikidata":
        return "detail" if record["kind"] == "unknowable_detail" else "award"
    return {"generated": "fictional"}.get(source, source)


_REAL_RUNS: dict[str, RunDir] = {}


@pytest.fixture
def real_run(
    tiny_arch: LoadedModel, tmp_path_factory: pytest.TempPathFactory
) -> RunDir:
    """One labeling run per architecture with the real labeler."""
    key = type(tiny_arch.model).__name__
    if key not in _REAL_RUNS:
        config = _config()
        out = tmp_path_factory.mktemp(f"p9a-{key}")
        run_dir = RunDir.create(config, out_root=out)
        execute(config, run_dir, tiny_arch, _loaders())
        _REAL_RUNS[key] = run_dir
    return _REAL_RUNS[key]


def test_config_defaults_and_round_trip() -> None:
    config = Config()

    assert config.experiment == "p9a"
    assert config.model == "gpt2"
    assert config.cutoff_year == 2017
    assert config.n_popqa is None
    assert config.n_synthal is None
    assert config.n_kuq == 200
    assert config.n_falseqa == 500
    assert config.n_triviaqa == 1000
    assert config.n_nq_open == 1000
    assert config.n_detail == 300
    assert config.n_shots == 15
    assert config.n_samples == 5
    assert config.temperature == 1.0
    assert config.known_min_correct == 3
    assert config.max_new_tokens == 16
    assert config.n_hand_check == 200
    assert config.max_guess_accuracy == 0.10
    assert config.never_known == ("sport",)
    assert config.min_known == 100
    assert config.popular_min == 10_000
    assert config.popular_share == 0.5
    assert config.batch_size == 8
    assert config.offline is False
    assert Config.from_dict(config.to_dict()) == config


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("p9a")

    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_execute_writes_the_files_of_a_run(real_run: RunDir) -> None:
    for name in (
        "items.json",
        "table.json",
        "table_popularity.json",
        "table_guess.json",
        "curves.npz",
        "curves.json",
        "hand_check.csv",
        "summary.json",
    ):
        assert (real_run.path / name).is_file(), name


def test_execute_labels_every_pool_item_once(real_run: RunDir) -> None:
    records = _records(real_run)
    summary = _summary(real_run)

    uids = [r["uid"] for r in records]
    assert len(set(uids)) == len(uids) == summary["n_items"]
    assert {r["label"] for r in records} <= {"known", "unknown", "uncertain"}


def test_summary_counts_add_up_to_the_pool(real_run: RunDir) -> None:
    summary = _summary(real_run)
    records = _records(real_run)

    assert sorted(summary["counts"]) == sorted(
        g for g in GROUPS if any(_group(r) == g for r in records)
    )
    for group, cell in summary["counts"].items():
        n = sum(_group(r) == group for r in records)
        assert cell["known"] + cell["unknown"] + cell["uncertain"] == n
        assert cell["known"] <= cell["known_raw"]
    assert sum(
        sum(c[k] for k in ("known", "unknown", "uncertain"))
        for c in summary["counts"].values()
    ) == len(records)


def test_summary_records_the_recipe_and_the_gate(real_run: RunDir) -> None:
    summary = _summary(real_run)

    assert summary["recipe"] == {
        "n_samples": 5,
        "temperature": 1.0,
        "known_min_correct": 3,
        "max_new_tokens": 4,
        "stop": "\n",
        "n_shots": 2,
    }
    assert summary["min_known"] == 100
    assert summary["known_contrasts"] is False
    assert summary["guess"]["n_known"] <= summary["guess"]["n_known_raw"]
    assert summary["guess"]["never_known"] == ["sport"]
    assert summary["guess"]["max_guess_accuracy"] == 0.10


def test_award_kinds_follow_the_cutoff(real_run: RunDir) -> None:
    summary = _summary(real_run)

    assert summary["cutoff"]["year"] == 2010
    assert summary["cutoff"]["source"] == "config"
    awards = [r for r in _records(real_run) if _group(r) == "award"]
    assert {r["kind"] for r in awards} == {"real", "post_cutoff"}
    for record in awards:
        year = record["meta"]["year"]
        assert record["kind"] == ("post_cutoff" if year > 2010 else "real")
        assert record["meta"]["cutoff_year"] == 2010


def test_hand_check_holds_only_behavior_based_rows(real_run: RunDir) -> None:
    with (real_run.path / "hand_check.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_uid = {r["uid"]: r for r in _records(real_run)}

    assert 0 < len(rows) <= 12
    assert len(rows) == _summary(real_run)["hand_check_rows"]
    for row in rows:
        assert by_uid[row["uid"]]["basis"] == "behavior"
        assert row["human_label"] == ""


def test_curves_hold_the_known_rate_by_year_with_wilson_bands(
    real_run: RunDir,
) -> None:
    curves = real_run.read_arrays("curves")
    sidecar = real_run.read_json("curves")

    assert curves["x"].tolist() == [float(y) for y in YEARS]
    assert curves["mean"].shape == (1, len(YEARS))
    assert np.all(curves["lo"] <= curves["mean"])
    assert np.all(curves["mean"] <= curves["hi"])
    assert sidecar == {
        "series": ["known rate"],
        "xlabel": "Year",
        "ylabel": "Known rate",
        "chance": None,
        "n_layers": None,
    }


def test_plot_draws_the_known_rate_by_year(
    real_run: RunDir, tmp_path: Path
) -> None:
    paths = plot(real_run, tmp_path)

    assert sorted(p.name for p in paths) == [
        "curves.pdf",
        "curves.png",
        "curves.svg",
    ]
    assert all(p.stat().st_size > 0 for p in paths)


def test_the_tables_have_the_documented_columns(real_run: RunDir) -> None:
    table = real_run.read_json("table")
    popularity = real_run.read_json("table_popularity")
    guess = real_run.read_json("table_guess")

    assert table["columns"] == ["group", "kind", "n", *FIELDS]
    assert popularity["columns"] == ["relation", "popularity", "n", *FIELDS]
    assert guess["columns"] == [
        "stratum",
        "n",
        "modal",
        "modal_share",
        "guess_accuracy",
        "excluded",
        "eligible",
    ]
    assert sum(row[2] for row in table["rows"]) == _summary(real_run)["n_items"]


def test_execute_forwards_the_labeling_settings(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def spy(
        loaded: LoadedModel, examples: list[QAExample], **kwargs: Any
    ) -> list[ItemLabel]:
        calls.append({"loaded": loaded, "n": len(examples), **kwargs})
        return _plant(loaded, examples, **kwargs)

    monkeypatch.setattr(p9a_labels, "label_questions", spy)
    config = _config(
        n_samples=4,
        known_min_correct=2,
        temperature=0.7,
        max_new_tokens=6,
        batch_size=5,
        seed=9,
    )
    run_dir = RunDir.create(config, out_root=tmp_path)

    execute(config, run_dir, tiny_arch, _loaders())

    (call,) = calls
    assert call["loaded"] is tiny_arch
    assert call["n"] == _summary(run_dir)["n_items"]
    assert call["recipe"] == LabelRecipe(
        n_samples=4, temperature=0.7, known_min_correct=2, max_new_tokens=6
    )
    assert call["shots"] == DEFAULT_SHOTS[:2]
    assert call["seed"] == 9
    assert call["batch_size"] == 5


def test_the_guess_control_demotes_the_constant_sport_answers(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _planted_run(tiny_arch, tmp_path, monkeypatch)
    summary = _summary(run_dir)
    records = _records(run_dir)

    sport = [
        r
        for r in records
        if r["meta"].get("relation") == "sport"
        and r["meta"]["source"] == "popqa"
    ]
    assert sport
    assert all(r["label"] == "known" for r in sport)
    assert {r["guess"] for r in sport} == {"excluded_stratum"}
    guess = summary["guess"]
    assert guess["verdicts"] == {"excluded_stratum": len(sport)}
    assert guess["n_known_raw"] - guess["n_known"] == len(sport)
    assert guess["n_known"] < guess["n_known_raw"]
    popqa = summary["counts"]["popqa"]
    assert popqa["known_raw"] - popqa["known"] == len(sport)
    assert popqa["uncertain"] >= len(sport)


def test_table_guess_has_one_row_per_stratum(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _planted_run(tiny_arch, tmp_path, monkeypatch)

    rows = {row[0]: row for row in run_dir.read_json("table_guess")["rows"]}

    assert sorted(rows) == [
        "award:Award0",
        "award:Award1",
        "nq_open",
        "popqa:father",
        "popqa:mother",
        "popqa:occupation",
        "popqa:sport",
        "triviaqa",
    ]
    sport = rows["popqa:sport"]
    assert sport[1] == 8
    assert (sport[2], sport[5], sport[6]) == ("football", True, False)
    assert rows["popqa:father"][6] is True


def test_the_known_raw_column_keeps_the_label_before_the_control(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _planted_run(tiny_arch, tmp_path, monkeypatch)

    rows = run_dir.read_json("table")["rows"]
    popqa = next(r for r in rows if r[0] == "popqa")

    assert popqa[3] - popqa[4] == 8
    assert popqa[2] == popqa[4] + popqa[5] + popqa[6]
    pop_rows = run_dir.read_json("table_popularity")["rows"]
    assert sum(r[2] for r in pop_rows) == popqa[2]
    assert sum(r[3] - r[4] for r in pop_rows) == 8


@pytest.mark.parametrize(
    ("min_known", "expected"),
    [(0, True), (1, True), (24, True), (25, False), (100, False)],
    ids=["zero", "one", "exactly", "one_more", "default"],
)
def test_known_contrasts_is_the_gate_on_guess_adjusted_knowns(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    min_known: int,
    expected: bool,
) -> None:
    run_dir = _planted_run(
        tiny_arch, tmp_path, monkeypatch, min_known=min_known
    )
    summary = _summary(run_dir)

    assert summary["min_known"] == min_known
    assert summary["guess"]["n_known"] == 24
    assert summary["known_contrasts"] is expected


@pytest.mark.parametrize(
    ("never_known", "excluded", "verdicts"),
    [
        ((), [], {"guessable_stratum": 8}),
        (
            ("father",),
            ["popqa:father"],
            {"excluded_stratum": 4, "guessable_stratum": 8},
        ),
    ],
    ids=["none", "father"],
)
def test_never_known_names_the_excluded_strata(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    never_known: tuple[str, ...],
    excluded: list[str],
    verdicts: dict[str, int],
) -> None:
    run_dir = _planted_run(
        tiny_arch, tmp_path, monkeypatch, never_known=never_known
    )
    rows = run_dir.read_json("table_guess")["rows"]

    assert sorted(row[0] for row in rows if row[5]) == excluded
    guess = _summary(run_dir)["guess"]
    assert guess["never_known"] == list(never_known)
    assert guess["verdicts"] == verdicts


@pytest.mark.parametrize(
    ("limit", "eligible", "verdict"),
    [(0.10, False, "guessable_stratum"), (0.3, True, "modal_answer")],
    ids=["default", "raised"],
)
def test_max_guess_accuracy_sets_the_guessable_limit(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: float,
    eligible: bool,
    verdict: str,
) -> None:
    loaders = _regold(
        _loaders(),
        "popqa",
        lambda e: ("John",) if e.uid in ("popqa:0", "popqa:4") else None,
    )
    run_dir = _planted_run(
        tiny_arch,
        tmp_path,
        monkeypatch,
        plant=_plant_constant_father,
        loaders=loaders,
        max_guess_accuracy=limit,
    )
    rows = {row[0]: row for row in run_dir.read_json("table_guess")["rows"]}
    known_fathers = [
        r
        for r in _records(run_dir)
        if r["meta"]["source"] == "popqa"
        and r["meta"]["relation"] == "father"
        and r["label"] == "known"
    ]

    father = rows["popqa:father"]
    assert (father[1], father[2], father[4]) == (8, "john", 0.25)
    assert father[6] is eligible
    assert len(known_fathers) == 2
    assert {r["guess"] for r in known_fathers} == {verdict}


def test_the_curves_follow_the_guess_adjusted_labels(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaders = _regold(
        _loaders(),
        "awards",
        lambda e: ("Marie Curie",) if e.meta["relation"] == "Award0" else None,
    )
    run_dir = _planted_run(tiny_arch, tmp_path, monkeypatch, loaders=loaders)
    awards = [r for r in _records(run_dir) if _group(r) == "award"]
    years = sorted({r["meta"]["year"] for r in awards})
    by_year = [[r for r in awards if r["meta"]["year"] == y] for y in years]
    adjusted = [
        sum(r["label"] == "known" and r["guess"] is None for r in rs) / len(rs)
        for rs in by_year
    ]
    raw = [sum(r["label"] == "known" for r in rs) / len(rs) for rs in by_year]
    curves = run_dir.read_arrays("curves")

    assert adjusted != raw
    assert curves["x"].tolist() == [float(y) for y in years]
    assert curves["mean"][0].tolist() == pytest.approx(adjusted)


def test_the_probe_supplies_the_cutoff_when_the_config_has_none(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _planted_run(tiny_arch, tmp_path, monkeypatch, cutoff_year=None)
    cutoff = _summary(run_dir)["cutoff"]

    assert cutoff["source"] == "probe"
    assert cutoff["year"] == 2010
    assert cutoff["estimate"]["year"] == 2010
    assert cutoff["estimate"]["rate_before"] > cutoff["estimate"]["rate_after"]
    assert cutoff["estimate"]["gain"] > 0
    awards = [r for r in _records(run_dir) if _group(r) == "award"]
    assert {r["meta"]["cutoff_year"] for r in awards} == {2010}


def test_the_config_cutoff_wins_over_the_estimate(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _planted_run(tiny_arch, tmp_path, monkeypatch, cutoff_year=2013)
    cutoff = _summary(run_dir)["cutoff"]

    assert (cutoff["year"], cutoff["source"]) == (2013, "config")
    assert cutoff["estimate"]["year"] == 2010


def test_without_a_cutoff_and_without_a_drop_execute_raises(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p9a_labels, "label_questions", _plant)
    config = _config(cutoff_year=None)
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match="set cutoff_year"):
        execute(config, run_dir, tiny_arch, _loaders(awards=False))


def test_the_summary_estimate_is_null_without_a_drop(
    real_run: RunDir,
) -> None:
    assert _summary(real_run)["cutoff"]["estimate"] is None


def test_a_pool_without_awards_writes_no_curves(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p9a_labels, "label_questions", _plant)
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)

    execute(config, run_dir, tiny_arch, _loaders(awards=False))

    assert not (run_dir.path / "curves.json").exists()
    assert not (run_dir.path / "curves.npz").exists()
    assert plot(run_dir, tmp_path / "figs") == []


def test_a_seed_gives_the_same_pool_and_hand_check_twice(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _planted_run(tiny_arch, tmp_path / "a", monkeypatch, seed=3, **CAPS)
    second = _planted_run(
        tiny_arch, tmp_path / "b", monkeypatch, seed=3, **CAPS
    )

    assert _records(first) == _records(second)
    assert _hand_check(first) == _hand_check(second)


def test_the_seed_draws_the_capped_pool(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    three = _planted_run(tiny_arch, tmp_path / "a", monkeypatch, seed=3, **CAPS)
    four = _planted_run(tiny_arch, tmp_path / "b", monkeypatch, seed=4, **CAPS)

    assert {r["uid"] for r in _records(three)} != {
        r["uid"] for r in _records(four)
    }


def test_the_seed_draws_the_hand_check_rows(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    three = _planted_run(tiny_arch, tmp_path / "a", monkeypatch, seed=3)
    four = _planted_run(tiny_arch, tmp_path / "b", monkeypatch, seed=4)

    def behavior(run_dir: RunDir) -> list[dict[str, Any]]:
        return [r for r in _records(run_dir) if r["basis"] == "behavior"]

    assert behavior(three) == behavior(four)
    assert _hand_check(three) != _hand_check(four)


def test_load_labeled_reads_a_finished_run(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _planted_run(tiny_arch, tmp_path, monkeypatch)
    run_dir.finalize()

    items, summary = load_labeled(run_dir.path)

    records = _records(run_dir)
    assert [i.example.uid for i in items] == [r["uid"] for r in records]
    assert [i.guess for i in items] == [r["guess"] for r in records]
    assert summary == _summary(run_dir)
    assert any(i.guess == "excluded_stratum" for i in items)
    assert (
        sum(i.contrast_label == "known" for i in items)
        == (summary["guess"]["n_known"])
    )


def test_load_labeled_refuses_an_unfinished_run(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _planted_run(tiny_arch, tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="is not a finished run"):
        load_labeled(run_dir.path)


def test_run_loads_the_model_and_the_default_loaders(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []
    bound: dict[str, Any] = {}

    def fake_load_model(key: str) -> LoadedModel:
        asked.append(key)
        return tiny_arch

    def fake_default(cls: type[PoolLoaders], **kwargs: Any) -> PoolLoaders:
        bound.update(kwargs)
        return _loaders()

    monkeypatch.setattr(p9a_labels, "load_model", fake_load_model)
    monkeypatch.setattr(PoolLoaders, "default", classmethod(fake_default))
    monkeypatch.setattr(p9a_labels, "label_questions", _plant)
    config = _config(offline=True, n_detail=11, seed=4)
    run_dir = RunDir.create(config, out_root=tmp_path)

    run(config, run_dir)

    assert asked == ["tiny"]
    assert bound == {"offline": True, "seed": 4, "n_detail": 11}
    assert (run_dir.path / "summary.json").is_file()


@pytest.mark.slow
def test_p9a_on_gpt2_labels_a_small_pool(tmp_path: Path) -> None:
    config = Config(
        n_popqa=5,
        n_kuq=5,
        n_falseqa=5,
        n_triviaqa=5,
        n_nq_open=5,
        n_detail=5,
        n_hand_check=20,
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    loaders = PoolLoaders.default(offline=True, n_detail=config.n_detail)
    loaded = load_model("gpt2")
    start = time.perf_counter()
    try:
        execute(config, run_dir, loaded, loaders)
    except OfflineCacheMissError as error:
        pytest.skip(str(error))
    seconds = time.perf_counter() - start

    summary = _summary(run_dir)
    print(summary["counts"])
    print(summary["guess"], summary["cutoff"])
    print(f"{seconds / summary['n_items']:.2f} seconds per item")
    assert summary["n_items"] > 0
