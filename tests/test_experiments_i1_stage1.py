"""Exercises I1 stage 1 with a synthetic stage 0 cache."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tomography.cli import main
from tomography.experiments import i1_invariance as i1
from tomography.experiments.base import RunDir

FIT = (
    "clean",
    "F-T1",
    "F-T2",
    "F-T3",
    "F-D1/random",
    "F-D2/random",
    "F-D3/random",
)
SCORE = (
    "clean",
    "S-T1",
    "S-T2",
    "S-D1/random",
    "S-D1/against",
    "S-D2/random",
    "S-D2/against",
)
ARMS = ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8")


def test_stage0_retains_its_config_identity() -> None:
    assert i1.Config().to_dict() == {
        "experiment": "i1",
        "model": "qwen3-0.6b-base",
        "seed": 0,
        "batch_size": 8,
    }


def _cache(root: Path, *, adverse: bool, leaked: str = "") -> RunDir:
    cache = RunDir.create(i1.Config(), out_root=root)
    records = {}
    levels: dict[str, Any] = {}
    for task in i1.TASKS:
        fit_pairs = [[f"f{n}p", f"f{n}n"] for n in range(6)]
        val_pairs = [[f"v{n}p", f"v{n}n"] for n in range(3)]
        pool = [f"u{n}" for n in range(12)]
        blocks = {
            "report": {
                "keys": ["untouched"],
                "uids": ["not-cached"],
                "pairs": [["not-cached", "also-not-cached"]],
            },
            "validation": {
                "keys": [pair[0] for pair in val_pairs],
                "uids": [uid for pair in val_pairs for uid in pair[::-1]],
                "pairs": val_pairs,
            },
            "fit": {
                "keys": [pair[0] for pair in fit_pairs],
                "uids": [uid for pair in fit_pairs for uid in pair[::-1]],
                "pairs": fit_pairs,
            },
            "pool": {"keys": pool, "uids": pool, "pairs": []},
        }
        records[task] = {
            "blocks": blocks,
            "draws": [fit_pairs[:4], fit_pairs[2:]],
            "counts": {"pool_texts": len(pool)},
            "notes": [],
        }
        fit = (FIT[0], *FIT[4:]) if task == "utility" else FIT
        score = (SCORE[0], *SCORE[3:]) if task == "utility" else SCORE
        rows = []
        signal = []
        levels[task] = {condition: {} for condition in ("F-D1", "F-D2", "F-D3")}
        for block, names in (
            ("fit", fit),
            ("pool", fit),
            ("validation", score),
        ):
            for name in names:
                word, _, rule = name.partition("/")
                for uid in blocks[block]["uids"]:
                    label = int(uid.endswith("p"))
                    level = None
                    if rule:
                        level = (
                            "high" if label != (rule == "against") else "low"
                        )
                        if block == "pool":
                            levels[task][word][uid] = level
                    condition = name
                    if (
                        task == "sentiment"
                        and block == leaked
                        and name == "F-D1/random"
                    ):
                        condition = "F-D1/against"
                    rows.append(
                        {
                            "block": block,
                            "uid": uid,
                            "label": label,
                            "condition": condition,
                            "level": level,
                        }
                    )
                    value = 12 if block == "pool" else 8 * (2 * label - 1)
                    signal.append(
                        -value if adverse and name == "S-D1/against" else value
                    )
        rng = np.random.default_rng(15)
        stim = rng.normal(0, 0.05, (2, len(rows), 6)).astype(np.float32)
        last = rng.normal(0, 0.05, (2, len(rows), 6)).astype(np.float32)
        labelled = np.array([row["block"] != "pool" for row in rows])
        # Standardization would amplify random coordinates in the tiny draw.
        stim[:, labelled, 1:] = 0
        last[:, labelled, 1:] = 0
        stim[:, :, 0] += signal
        last[:, :, 0] += signal
        stim[0] = 0
        cache.write_json(f"prompts_{task}", rows)
        cache.write_arrays(
            f"activations_{task}",
            stimulus=stim,
            last=last,
            layers=np.array([12, 13]),
            last_token=np.full(len(rows), 15),
            stimulus_token=np.full(len(rows), 10),
        )
    cache.write_json("partitions", records)
    cache.write_json("pool_levels", levels)
    return cache


@pytest.fixture(scope="module")
def tournament(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[bool, int], RunDir]:
    root = tmp_path_factory.mktemp("tournament")
    caches = {
        flag: _cache(root / f"cache-{flag}", adverse=flag)
        for flag in (False, True)
    }
    runs = {}

    def execute(adverse: bool, workers: int) -> RunDir:
        if (adverse, workers) not in runs:
            cache = caches[adverse]
            out = root / f"out-{adverse}-{workers}"
            config = i1.Config.from_dict(
                {
                    "stage": 1,
                    "input_run": str(cache.path),
                    "go": "synthetic test",
                    "seed": 17,
                    "workers": workers,
                }
            )
            assert (
                main(
                    [
                        "run",
                        "i1",
                        "--out",
                        str(out),
                        "--seed",
                        "17",
                        "--set",
                        "stage=1",
                        "--set",
                        f"input_run={cache.path}",
                        "--set",
                        "go=synthetic test",
                        "--set",
                        f"workers={workers}",
                    ]
                )
                == 0
            )
            runs[adverse, workers] = RunDir.open(
                RunDir.path_for(config, out_root=out)
            )
        return runs[adverse, workers]

    return execute


def _data(run: RunDir, name: str) -> dict[str, Any]:
    payload = run.read_json(name)
    assert isinstance(payload, dict)
    return payload


def test_stage1_writes_scores_selection_gate_and_fit_audit(tournament) -> None:
    run = tournament(True, 1)
    assert run.meta["finished_at"] is not None
    results = _data(run, "stage1")
    selection = results["selection"]
    assert selection["winner"] == "A1"
    assert set(selection["eligible"]) == set(ARMS)
    assert selection["mean_gain"] == dict.fromkeys(ARMS, 0.0)
    for task, entries in results["draws"].items():
        assert selection["best_baseline"][task] == "A1"
        assert selection["endpoint"][task] == dict.fromkeys(ARMS, 0.0)
        assert selection["gain"][task] == dict.fromkeys(ARMS, 0.0)
        conditions = ("clean", *SCORE[3:]) if task == "utility" else SCORE
        assert selection["auroc"][task] == {
            arm: {name: float(name != "S-D1/against") for name in conditions}
            for arm in ARMS
        }
        assert [entry["draw"] for entry in entries] == [0, 1]
        assert entries[0]["seed"] != entries[1]["seed"]
        for entry in entries:
            assert entry["label_budget"] == 8
            assert entry["pool_texts"] == 12
            assert entry["validation_pairs"] == 3
            assert entry["fit_renderings"] == (4 if task == "utility" else 7)
            arms = entry["arms"]
            assert set(arms) == set(ARMS)
            for arm in ARMS:
                record = arms[arm]
                assert record["criterion"] == 1.0
                assert record["fits"] == (5 if arm in ("A5", "A6") else 9)
                assert record["fallbacks"] == (9 if arm in ("A7", "A8") else 0)
                assert record["refit_fallbacks"] == int(arm in ("A7", "A8"))
            assert arms["A3"]["layers"] == {"last": 12}
            assert arms["A4"]["layers"] == {"stim": 13}
            assert (
                arms["A5"]["layers"]
                == arms["A6"]["layers"]
                == {"stim": 13, "last": 12}
            )
            assert set(arms["A6"]["params"]) == {"lambda", "gamma"}
            assert set(arms["A8"]["params"]) == {"alpha"}
    assert set(results["draws"]) == set(i1.TASKS)
    assert (
        len(
            {
                entry["seed"]
                for entries in results["draws"].values()
                for entry in entries
            }
        )
        == 8
    )
    assert results["bootstrap_seed"] == 17
    assert results["n_resamples"] == 10_000
    assert results["bootstrap"] == [0.0] * 10_000
    gate = results["gate"]
    assert gate["comparator"] == "A3"
    assert gate["statistic"] == 0.0
    assert gate["interval"] == [0.0, 0.0]
    assert gate["passed"] is False
    assert len(gate["reasons"]) == 3
    log = _data(run, "run_log")
    assert log["go"] == "synthetic test"
    assert Path(log["stage0_cache"]).is_dir()
    assert log["gate"] == {"winner": "A1", **gate}


def test_worker_count_preserves_every_result(tournament) -> None:
    serial = tournament(True, 1)
    parallel = tournament(True, 2)
    assert _data(serial, "stage1") == _data(parallel, "stage1")
    assert _data(serial, "run_log") == _data(parallel, "run_log")


def test_against_activations_change_scoring_but_not_tuning(tournament) -> None:
    adverse = _data(tournament(True, 1), "stage1")
    neutral = _data(tournament(False, 1), "stage1")
    assert adverse["draws"] == neutral["draws"]
    assert adverse["selection"]["endpoint"]["sentiment"]["A6"] == 0.0
    assert neutral["selection"]["endpoint"]["sentiment"]["A6"] == 1.0


@pytest.mark.parametrize("block", ["fit", "pool"])
def test_stage1_rejects_against_in_fit_or_pool_cache(
    tmp_path: Path, block: str
) -> None:
    cache = _cache(tmp_path / "cache", adverse=True, leaked=block)
    config = i1.Config.from_dict(
        {
            "stage": 1,
            "input_run": str(cache.path),
            "go": "synthetic test",
        }
    )
    output = RunDir.create(config, out_root=tmp_path / "out")
    with pytest.raises(ValueError, match=rf"{block}.*against"):
        i1.run(config, output)
    assert _data(output, "run_log")["go"] == "synthetic test"
    assert "gate" not in _data(output, "run_log")
    assert not (output.path / "stage1.json").exists()


def test_stage1_requires_explicit_go_before_reading_cache(
    tmp_path: Path,
) -> None:
    config = i1.Config.from_dict({"stage": 1, "input_run": "missing"})
    output = RunDir.create(config, out_root=tmp_path)
    with pytest.raises(ValueError, match="explicit go"):
        i1.run(config, output)
    assert not (output.path / "run_log.json").exists()


@pytest.mark.parametrize("state", ["missing", "failed", "malformed", "passed"])
def test_stage2_requires_a_stored_pass(tmp_path: Path, state: str) -> None:
    source = RunDir.create(i1.Config.from_dict({"stage": 1}), out_root=tmp_path)
    if state != "missing":
        source.write_json(
            "run_log",
            {
                "gate": {
                    "passed": {
                        "failed": False,
                        "malformed": "true",
                        "passed": True,
                    }[state],
                    "winner": "A6",
                },
            },
        )
    config = i1.Config.from_dict({"stage": 2, "input_run": str(source.path)})
    output = RunDir.create(config, out_root=tmp_path)
    if state == "passed":
        with pytest.raises(NotImplementedError, match="stage 2"):
            i1.run(config, output)
    else:
        with pytest.raises(ValueError, match="stored passed stage 1 gate"):
            i1.run(config, output)
    assert not (output.path / "run_log.json").exists()
