from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from tomography.activations import Activations
from tomography.activations import collect_activations
from tomography.datasets.records import Example
from tomography.experiments import i1_invariance as i1
from tomography.experiments.base import RunDir
from tomography.model import LoadedModel
from tomography.templates import ChatFormat

TINY = i1.BlockSizes(report=1, validation=1, fit=1, pool=2)


def _block(task: str, name: i1.BlockName, n: int) -> i1.Block:
    if name == "pool":
        texts = tuple(
            Example(f"{task}-pool-{i}", f"the pool {i} is calm", 0, "train")
            for i in range(n)
        )
        return i1.Block(name, tuple(e.uid for e in texts), texts, ())
    texts = tuple(
        Example(
            f"{task}-{name}-{i}{side}",
            f"a {side} film about {i} dogs",
            int(side == "a"),
            "train",
            {"pair": f"{task}-{name}-{i}"},
        )
        for i in range(n)
        for side in ("a", "b")
    )
    pairs = tuple((texts[2 * i].uid, texts[2 * i + 1].uid) for i in range(n))
    return i1.Block(name, tuple(p[0] for p in pairs), texts, pairs)


def _partitions() -> dict[str, i1.TaskPartition]:
    return {
        task: i1.TaskPartition(
            task,
            MappingProxyType(
                {n: _block(task, n, getattr(TINY, n)) for n in i1.BLOCKS}
            ),
            MappingProxyType({"pool_texts": 2}),
            ("a note",),
        )
        for task in i1.TASKS
    }


def _run_dir(root: Path) -> RunDir:
    return RunDir.create(i1.Config(), out_root=root)


def test_candidate_layers_cover_relative_depth_0_4_to_1() -> None:
    assert i1.candidate_layers(28) == tuple(range(12, 29))
    assert i1.candidate_layers(16) == tuple(range(7, 17))
    assert i1.candidate_layers(3) == (2, 3)
    assert len(i1.candidate_layers(28)) == 17


def _short_prompts() -> list[i1.Prompt]:
    prompts = []
    for task in ("truth", "utility"):
        for i in range(5):
            stimulus = f"the {i} dog"
            text = f"ok {stimulus} is it"
            start = text.index(stimulus)
            prompts.append(
                i1.Prompt(
                    task,
                    "pool",
                    f"{task}-{i}",
                    i % 2,
                    "F-D1/random",
                    text,
                    (start, start + len(stimulus)),
                    "low",
                )
            )
    return prompts


def test_cache_holds_both_positions_at_every_candidate_layer(
    reading_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _run_dir(tmp_path)
    prompts = _short_prompts()
    i1.encode_prompts(reading_arch, prompts, run_dir, batch_size=2)
    layers = i1.candidate_layers(reading_arch.n_layers)
    assert not list(run_dir.path.glob("activations_sentiment*"))
    for task in ("truth", "utility"):
        rows = [p for p in prompts if p.task == task]
        arrays = run_dir.read_arrays(f"activations_{task}")
        shape = (len(layers), len(rows), reading_arch.d_model)
        assert arrays["last"].shape == shape
        assert arrays["stimulus"].shape == shape
        assert arrays["last"].dtype == np.float32
        assert arrays["stimulus"].dtype == np.float32
        assert tuple(arrays["layers"]) == layers
        index: Any = run_dir.read_json(f"prompts_{task}")
        assert index == [
            {
                "block": p.block,
                "uid": p.uid,
                "label": p.label,
                "condition": p.condition,
                "level": p.level,
            }
            for p in rows
        ]
        assert np.all(arrays["stimulus_token"] < arrays["last_token"])
        for row, prompt in enumerate(rows):
            for position, spans, key in (
                ("last", None, "last"),
                ("concept", [prompt.stimulus_span], "stimulus"),
            ):
                one = collect_activations(
                    reading_arch,
                    [prompt.text],
                    layers=layers,
                    position=position,
                    concept_spans=spans,
                )
                np.testing.assert_allclose(
                    arrays[key][:, row], one.data[:, 0], rtol=1e-4, atol=1e-3
                )
                assert arrays[f"{key}_token"][row] == one.row_token[0]


def test_stage0_encodes_the_stage0_prompt_set_in_float32_rows(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def record(
        loaded: LoadedModel, texts: list[str], **kwargs: Any
    ) -> Activations:
        calls.append({"texts": texts, **kwargs})
        n, k = len(texts), len(kwargs["layers"])
        return Activations(
            data=np.zeros((k, n, loaded.d_model), dtype=np.float32),
            layers=tuple(kwargs["layers"]),
            position=kwargs["position"],
            row_stimulus=np.arange(n),
            row_rank=np.zeros(n, dtype=np.int64),
            row_token=np.arange(n),
            row_token_id=np.zeros(n, dtype=np.int64),
            n_stimuli=n,
        )

    monkeypatch.setattr(i1, "collect_activations", record)
    run_dir = _run_dir(tmp_path)
    parts = _partitions()
    i1.encode_stage0(tiny_qwen3, parts, run_dir, batch_size=3)
    prompts = i1.stage0_prompts(parts, ChatFormat("plain"))
    assert len(calls) == 2 * len(i1.TASKS)
    for task, (last, stimulus) in zip(
        i1.TASKS, zip(calls[::2], calls[1::2], strict=True), strict=True
    ):
        rows = [p for p in prompts if p.task == task]
        assert last["texts"] == stimulus["texts"] == [p.text for p in rows]
        assert last["position"] == "last"
        assert stimulus["position"] == "concept"
        assert stimulus["concept_spans"] == [p.stimulus_span for p in rows]
        assert last["layers"] == i1.candidate_layers(tiny_qwen3.n_layers)
        assert last["batch_size"] == 3
    levels: Any = run_dir.read_json("pool_levels")
    assert levels == i1.pool_levels(prompts)


def test_partitions_are_written_before_any_encoding(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("encoding started")

    monkeypatch.setattr(i1, "collect_activations", fail)
    run_dir = _run_dir(tmp_path)
    parts = _partitions()
    with pytest.raises(RuntimeError, match="encoding started"):
        i1.encode_stage0(tiny_qwen3, parts, run_dir)
    written: Any = run_dir.read_json("partitions")
    assert written["sentiment"]["blocks"] == parts["sentiment"].item_ids()
    assert written["sentiment"]["counts"] == {"pool_texts": 2}
    assert written["utility"]["notes"] == ["a note"]
    assert written["truth"]["draws"] == [
        [list(p) for p in draw] for draw in i1.draws(parts["truth"], n_pairs=1)
    ]


def test_partitions_record_the_first_128_pairs_of_each_draw(
    tiny_qwen3: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("encoding started")

    monkeypatch.setattr(i1, "collect_activations", fail)
    run_dir = _run_dir(tmp_path)
    parts = _partitions()
    blocks = {**parts["truth"].blocks, "fit": _block("truth", "fit", 130)}
    parts["truth"] = replace(parts["truth"], blocks=MappingProxyType(blocks))
    with pytest.raises(RuntimeError, match="encoding started"):
        i1.encode_stage0(tiny_qwen3, parts, run_dir)
    written: Any = run_dir.read_json("partitions")
    stored = written["truth"]["draws"]
    longest = i1.draws(parts["truth"], n_pairs=128)
    assert stored == [[list(p) for p in draw] for draw in longest]
    assert [draw[:64] for draw in stored] == [
        [list(p) for p in draw] for draw in i1.draws(parts["truth"])
    ]


def test_stage0_needs_a_recorded_go_before_loading_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(**kwargs: object) -> None:
        raise RuntimeError("partitions loaded")

    monkeypatch.setattr(i1, "load_partitions", fail)
    config = i1.Config()
    run_dir = RunDir.create(config, out_root=tmp_path / "without")
    with pytest.raises(ValueError, match="go"):
        i1.run(config, run_dir)
    assert not (run_dir.path / "run_log.json").exists()
    config = i1.Config(go="test go")
    run_dir = RunDir.create(config, out_root=tmp_path / "with")
    with pytest.raises(RuntimeError, match="partitions loaded"):
        i1.run(config, run_dir)
    assert run_dir.read_json("run_log") == {"go": "test go"}


def test_stage0_runs_only_on_the_preregistered_checkpoint(
    tmp_path: Path,
) -> None:
    config = i1.Config(model="qwen3-1.7b-base")
    with pytest.raises(ValueError, match=r"qwen3-0\.6b-base"):
        i1.run(config, RunDir.create(config, out_root=tmp_path))
    assert i1.Config().model == "qwen3-0.6b-base"
    assert i1.plot(_run_dir(tmp_path / "b"), tmp_path) == []
