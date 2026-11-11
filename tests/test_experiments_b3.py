"""Tests for the B3 steering baselines experiment."""

from collections.abc import Sequence
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.activations import collect_activations
from tomography.cli import load_experiment
from tomography.control.contrast import choice_pairs
from tomography.control.contrast import mean_shift
from tomography.datasets.emotions import EMOTIONS
from tomography.datasets.records import Example
from tomography.experiments import b3_steering_baselines as b3
from tomography.experiments.b3_steering_baselines import CONCEPTS
from tomography.experiments.b3_steering_baselines import Config
from tomography.experiments.b3_steering_baselines import caa_vectors
from tomography.experiments.b3_steering_baselines import execute
from tomography.experiments.b3_steering_baselines import judge_report
from tomography.experiments.b3_steering_baselines import judged_generations
from tomography.experiments.b3_steering_baselines import mean_centring_vectors
from tomography.experiments.b3_steering_baselines import plot
from tomography.experiments.b3_steering_baselines import run
from tomography.experiments.base import RunDir
from tomography.experiments.effects import read_trials
from tomography.experiments.sweep import chat_format_for
from tomography.judging.batches import Export
from tomography.judging.run import export_stage
from tomography.judging.run import import_judgments
from tomography.model import MODELS
from tomography.model import LoadedModel

WORDS = ("good", "movie", "text", "dog", "fox", "true", "short", "happy")
ARMS = ("caa", "mean_centring", "actadd", "diffmean", "reader", "random/0")
COEFFS = (0.1, 0.4)
CONCEPT_NAMES = ("sentiment", "anger")
DASHES = (chr(0x2014), chr(0x2013))


def _phrase(rng: np.random.Generator, words: int) -> str:
    return " ".join(rng.choice(WORDS, words))


def _examples() -> dict[str, list[Example]]:
    rng = np.random.default_rng(5)
    sst2 = [
        Example(
            uid=f"sst2-{i}",
            text=_phrase(rng, 3),
            label=i % 2,
            split="train" if i % 5 else "val",
            meta={},
        )
        for i in range(60)
    ]
    emotions = [
        Example(
            uid=f"{emotion}-{i}",
            text=_phrase(rng, 3),
            label=label,
            split="train" if i % 5 else "val",
            meta={"emotion": emotion},
        )
        for label, emotion in enumerate(EMOTIONS)
        for i in range(20)
    ]
    return {"sst2": sst2, "emotions": emotions}


def _texts(n: int = 40) -> list[str]:
    return [_phrase(np.random.default_rng(100 + i), 12) for i in range(n)]


def _instructions(n: int = 12) -> list[str]:
    return [f"Is the {WORDS[i % 8]} {WORDS[(i // 8) % 8]}?" for i in range(n)]


def _vowels(texts: Sequence[str]) -> NDArray[np.float32]:
    return np.array(
        [sum(ch in "aeiou" for ch in t) / max(len(t), 1) for t in texts],
        dtype=np.float32,
    )


CLASSIFY = {name: _vowels for name in CONCEPT_NAMES}


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        concepts=CONCEPT_NAMES,
        n_tune=3,
        n_held_out=3,
        n_pairs=6,
        coeffs=COEFFS,
        n_calibration=10,
        n_neutral=6,
        max_new_tokens=4,
        truncate_words=3,
        caa_question="Is it {description}?",
        batch_size=8,
        n_boot=50,
    )
    return dataclasses.replace(base, **overrides)


def _post_trained(loaded: LoadedModel) -> LoadedModel:
    return dataclasses.replace(loaded, checkpoint=MODELS["qwen3-1.7b"])


def _execute(
    loaded: LoadedModel, root: Path, **overrides: Any
) -> tuple[Config, RunDir]:
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=root)
    execute(
        config,
        run_dir,
        loaded,
        _examples(),
        _texts(),
        _instructions(),
        CLASSIFY,
    )
    return config, run_dir


def _trial_names() -> list[str]:
    return [
        "none",
        "prompt",
        *(f"{arm}/{coeff:g}" for arm in ARMS for coeff in COEFFS),
    ]


def _digest(run_dir: RunDir, pattern: str = "*") -> str:
    digest = hashlib.sha256()
    for path in sorted(run_dir.path.glob(pattern)):
        if path.is_file() and path.name not in ("meta.json", "config.json"):
            digest.update(path.name.encode())
            if path.suffix == ".json":
                stored = json.loads(path.read_bytes())["data"]
                digest.update(json.dumps(stored).encode())
                continue
            arrays = run_dir.read_arrays(path.stem)
            for name in sorted(arrays):
                digest.update(name.encode())
                digest.update(arrays[name].tobytes())
    return digest.hexdigest()


def test_the_default_config_follows_the_protocol() -> None:
    config = Config()

    assert config.experiment == "b3"
    assert config.max_new_tokens == 128
    assert config.num_samples == 1
    assert (config.n_tune, config.n_held_out) == (40, 40)
    assert config.concepts == ("sentiment", "happiness", "anger")
    assert config.coeffs == (0.05, 0.1, 0.2, 0.4, 0.8)
    assert config.mode == "relative"
    assert config.method == "logistic"
    assert config.n_random == 1
    assert config.config_hash() == "ece096f9abd1"


def test_the_registry_describes_sentiment_and_every_emotion() -> None:
    assert set(CONCEPTS) == {"sentiment", *EMOTIONS}
    assert CONCEPTS["sentiment"].source == "sst2"
    assert CONCEPTS["sentiment"].description == "positive sentiment"
    assert CONCEPTS["sentiment"].judge == "sentiment-primary"
    for label, emotion in enumerate(EMOTIONS):
        concept = CONCEPTS[emotion]
        assert (concept.source, concept.target) == ("emotions", label)
        assert concept.description == emotion
        assert (concept.judge, concept.column) == ("emotion-primary", label)
        assert concept.pair[0] != concept.pair[1]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mode": "absolute"}, "relative"),
        ({"coeffs": (0.0, 0.1)}, "coeffs"),
        ({"coeffs": (0.2, 0.1)}, "coeffs"),
        ({"coeffs": ()}, "coeffs"),
        ({"concepts": ()}, "concepts"),
        ({"concepts": ("sentiment", "sentiment")}, "concepts"),
        ({"concepts": ("serenity",)}, "concepts"),
        ({"n_tune": 0}, "n_tune"),
        ({"n_held_out": 0}, "n_held_out"),
        ({"n_random": 0}, "n_random"),
        ({"num_samples": 2, "temperature": None}, "num_samples"),
        ({"n_pairs": 500}, "n_pairs"),
        ({"n_tune": 30}, "prompts"),
    ],
)
def test_a_setting_the_protocol_cannot_run_is_refused_before_any_write(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _execute(tiny_gpt2, tmp_path, **overrides)

    (run_path,) = tmp_path.glob("*/*/*")
    assert sorted(p.name for p in run_path.iterdir()) == [
        "config.json",
        "meta.json",
    ]


def test_a_base_run_stores_every_arm_on_two_disjoint_prompt_halves(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    config, run_dir = _execute(tiny_gpt2, tmp_path)

    prompts = run_dir.read_json("prompts")
    assert isinstance(prompts, dict)
    assert sorted(prompts) == ["held_out", "tune"]
    assert (len(prompts["tune"]), len(prompts["held_out"])) == (3, 3)
    assert not set(prompts["tune"]) & set(prompts["held_out"])
    assert len(set(prompts["tune"]) | set(prompts["held_out"])) == 6
    assert all(len(p.split()) == 3 for half in prompts.values() for p in half)
    for concept in CONCEPT_NAMES:
        for split in ("tune", "held_out"):
            trials = read_trials(run_dir, f"trials_{concept}_{split}")
            assert [t.name for t in trials] == _trial_names()
            for trial in trials:
                assert list(trial.prompts) == prompts[split]
                assert trial.scores.shape == (3, 1)
                assert trial.scores.dtype == np.float32
                texts = [row[0] for row in trial.completions]
                filled = [bool(text.strip()) for text in texts]
                expected = _vowels(texts)
                assert np.isfinite(trial.scores[:, 0]).tolist() == filled
                np.testing.assert_allclose(
                    trial.scores[filled, 0], expected[filled]
                )
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["role"] == "base"
    assert summary["criteria"] == ["concept", "fluency"]
    assert summary["layers"] == [1, 2]
    assert sorted(summary["norms"]) == ["1", "2"]
    assert all(norm > 0 for norm in summary["norms"].values())
    assert summary["concepts"] == {
        "sentiment": "positive sentiment",
        "anger": "anger",
    }
    assert summary["arms"] == [
        ["none", "none", None],
        ["prompt", "prompt", None],
        *([f"{arm}/{c:g}", arm, c] for arm in ARMS for c in COEFFS),
    ]
    assert (summary["n_tune"], summary["n_held_out"]) == (3, 3)
    assert config.concepts == CONCEPT_NAMES


def test_an_unsteered_generation_does_not_depend_on_the_concept(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    _, run_dir = _execute(tiny_gpt2, tmp_path)

    first = read_trials(run_dir, "trials_sentiment_tune")[0]
    second = read_trials(run_dir, "trials_anger_tune")[0]

    assert first.name == second.name == "none"
    assert first.completions == second.completions
    steered = read_trials(run_dir, "trials_sentiment_tune")[2:]
    assert any(t.completions != first.completions for t in steered)


def test_a_post_trained_run_renders_instructions_with_the_chat_template(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    loaded = _post_trained(tiny_qwen3)
    _, run_dir = _execute(loaded, tmp_path)

    prompts = run_dir.read_json("prompts")
    assert isinstance(prompts, dict)
    shown = [*prompts["tune"], *prompts["held_out"]]
    assert len(set(shown)) == 6
    assert set(shown) <= set(_instructions())
    chat = chat_format_for(loaded)
    trials = read_trials(run_dir, "trials_anger_held_out")
    assert [t.name for t in trials] == _trial_names()
    assert list(trials[0].prompts) == [
        chat.render_turn(instruction)[0] for instruction in prompts["held_out"]
    ]
    assert trials[0].prompts[0] != prompts["held_out"][0]
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["role"] == "post_trained"
    assert summary["criteria"] == ["concept", "instruction", "fluency"]
    judged = judged_generations(run_dir)
    assert judged.criteria == ("concept", "instruction", "fluency")
    assert {r.prompt for r in judged.records} == set(shown)


def test_a_post_trained_run_needs_instructions(
    tiny_qwen3: LoadedModel, tmp_path: Path
) -> None:
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)

    with pytest.raises(ValueError, match="instructions"):
        execute(
            config,
            run_dir,
            _post_trained(tiny_qwen3),
            _examples(),
            _texts(),
            None,
            CLASSIFY,
        )


def test_the_caa_vector_is_the_mean_answer_letter_difference(
    tiny_arch: LoadedModel,
) -> None:
    positive = ["good movie", "happy dog", "true text"]
    negative = ["short fox", "dog text", "fox movie"]
    question = "Is it happy?"

    vectors = caa_vectors(
        tiny_arch, positive, negative, question=question, layers=(1, 3)
    )

    chat = chat_format_for(tiny_arch)
    texts = [
        chat.render_turn(pair.question, answer)[0]
        for pair in choice_pairs(positive, negative, question=question)
        for answer in (pair.positive, pair.negative)
    ]
    assert all(t.endswith(("(A", "(B")) for t in texts)
    last = collect_activations(
        tiny_arch, texts, layers=(1, 3), position="last"
    ).data
    assert vectors.shape == (2, tiny_arch.d_model)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(
        vectors, mean_shift(last[:, 0::2], last[:, 1::2]), atol=1e-5
    )
    np.testing.assert_allclose(
        vectors, (last[:, 0::2] - last[:, 1::2]).mean(axis=1), atol=1e-4
    )


def test_the_mean_centring_vector_is_the_target_mean_minus_a_generic_mean(
    tiny_arch: LoadedModel,
) -> None:
    target = ["happy dog", "good true movie"]
    generic = ["the fox", "short text", "dog"]

    vectors = mean_centring_vectors(tiny_arch, target, generic, layers=(2,))

    def token_mean(text: str) -> NDArray[np.float32]:
        rows = collect_activations(
            tiny_arch,
            [text],
            layers=(2,),
            position="response",
            response_spans=[(0, len(text))],
        ).data[0]
        mean: NDArray[np.float32] = rows.mean(axis=0)
        return mean

    expected = np.mean([token_mean(t) for t in target], axis=0) - np.mean(
        [token_mean(t) for t in generic], axis=0
    )
    assert vectors.shape == (1, tiny_arch.d_model)
    np.testing.assert_allclose(vectors[0], expected, atol=1e-4)


def test_every_vector_arm_stores_a_unit_direction_per_layer(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    _, run_dir = _execute(tiny_gpt2, tmp_path)

    vectors = run_dir.read_arrays("vectors")
    names = [
        f"{concept}__{arm}"
        for concept in CONCEPT_NAMES
        for arm in ("caa", "mean_centring", "diffmean", "reader", "random_0")
    ]
    assert sorted(vectors) == sorted(names)
    for name in names:
        assert vectors[name].shape == (2, tiny_gpt2.d_model)
        assert vectors[name].dtype == np.float32
        np.testing.assert_allclose(
            np.linalg.norm(vectors[name], axis=1), 1.0, atol=1e-5
        )
    assert not np.allclose(vectors["anger__caa"], vectors["anger__diffmean"])
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert sorted(summary["raw_norms"]["anger"]) == [
        "actadd",
        "caa",
        "mean_centring",
    ]
    assert all(
        len(norms) == 2 and min(norms) > 0
        for norms in summary["raw_norms"]["anger"].values()
    )


def test_the_side_effects_cover_every_arm_of_the_held_out_half(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    _, run_dir = _execute(tiny_gpt2, tmp_path)

    table = run_dir.read_json("table_effects")
    assert isinstance(table, dict)
    assert table["columns"] == [
        "concept",
        "arm",
        "coeff",
        "perplexity",
        "fluency",
        "distinct_1",
        "distinct_2",
        "mean_length",
        "empty_rate",
    ]
    rows = {(r[0], r[1], r[2]): r for r in table["rows"]}
    assert list(rows) == [
        (concept, arm, coeff)
        for concept in CONCEPT_NAMES
        for arm, coeff in [
            ("none", None),
            ("prompt", None),
            *((arm, c) for arm in ARMS for c in COEFFS),
        ]
    ]
    for (_, arm, _), row in rows.items():
        assert (row[3] is None) == (arm == "prompt")
        assert row[3] is None or row[3] > 1.0
        assert 0.0 <= row[5] <= 1.0
    none = rows["sentiment", "none", None]
    assert rows["anger", "none", None][3:] == none[3:]
    assert rows["sentiment", "reader", 0.4][3] != none[3]
    assert rows["sentiment", "actadd", 0.4][3] != none[3]


def test_one_seed_gives_the_same_bytes_and_another_seed_other_prompts(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    _, first = _execute(tiny_gpt2, tmp_path / "a")
    _, again = _execute(tiny_gpt2, tmp_path / "b")
    _, other = _execute(tiny_gpt2, tmp_path / "c", seed=1)

    assert _digest(first) == _digest(again)
    assert first.read_json("prompts") != other.read_json("prompts")
    assert first.read_arrays("vectors")["anger__caa"].dtype == np.float32


def test_the_judged_generations_cover_every_stored_completion(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    _, run_dir = _execute(tiny_gpt2, tmp_path)

    judged = judged_generations(run_dir)

    prompts = run_dir.read_json("prompts")
    assert isinstance(prompts, dict)
    assert judged.criteria == ("concept", "fluency")
    assert len(judged.records) == 2 * 2 * len(_trial_names()) * 3
    assert judged.cells() == [
        (concept, arm)
        for concept in CONCEPT_NAMES
        for arm in ("none", "prompt", *ARMS)
    ]
    record = next(
        r for r in judged.records if r.key == "anger/held_out/caa/0.4/2/0"
    )
    trial = read_trials(run_dir, "trials_anger_held_out")[3]
    assert trial.name == "caa/0.4"
    assert (record.concept, record.description) == ("anger", "anger")
    assert (record.arm, record.coeff) == ("caa", 0.4)
    assert (record.split, record.prompt_index) == ("held_out", 2)
    assert record.prompt == prompts["held_out"][2]
    assert record.text == trial.completions[2][0]
    grid_free = {r.arm for r in judged.records if r.coeff is None}
    assert grid_free == {"none", "prompt"}


def _plan(run_dir: RunDir, stage: str) -> Export:
    stored = run_dir.read_json(f"judge_export_{stage}")
    assert isinstance(stored, dict)
    return Export.from_dict(stored)


def _judge(run_dir: RunDir, stage: str, out: Path) -> None:
    """Exports a stage and imports planted judgments for every batch.

    The planted judge prefers coefficient 0.1 for `caa` and 0.4 for every
    other arm, and finds every text fluent.
    """
    export_stage(
        run_dir, judged_generations(run_dir), stage, out, batch_size=40
    )
    for batch in _plan(run_dir, stage).batches:
        lines = []
        for item_id, key in zip(batch.item_ids, batch.keys, strict=True):
            best = "/0.1/" if "/caa/" in key else "/0.4/"
            lines.append(
                json.dumps(
                    {
                        "item_id": item_id,
                        "concept": 2 if best in key else 1,
                        "fluency": 2,
                    }
                )
            )
        import_judgments(
            run_dir,
            batch.batch_id,
            "\n".join(lines),
            judge_model=f"judge-model-{batch.judge}",
        )


def test_the_report_needs_the_judgments_of_both_stages(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    _, run_dir = _execute(tiny_gpt2, tmp_path / "r")

    with pytest.raises(ValueError, match="stage 'tune' has no export"):
        judge_report(run_dir)
    _judge(run_dir, "tune", tmp_path / "tune")
    with pytest.raises(ValueError, match="stage 'held_out' has no export"):
        judge_report(run_dir)

    assert not (run_dir.path / "table.json").exists()
    assert plot(run_dir, tmp_path / "figures") == []


def test_the_report_selects_on_tuning_and_reports_the_held_out_half(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    _, run_dir = _execute(tiny_gpt2, tmp_path / "r")
    stored = _digest(run_dir, "trials_*")
    _judge(run_dir, "tune", tmp_path / "tune")
    _judge(run_dir, "held_out", tmp_path / "held_out")

    paths = judge_report(run_dir)

    assert [p.name for p in paths] == [
        "table.json",
        "table_tuning.json",
        "summary_judged.json",
    ]
    assert _digest(run_dir, "trials_*") == stored
    table = run_dir.read_json("table")
    assert isinstance(table, dict)
    assert table["columns"] == [
        "concept",
        "arm",
        "coeff",
        "n_prompts",
        "n_items",
        "concept_score",
        "fluency_score",
        "overall",
        "overall_lo",
        "overall_hi",
        "perplexity",
        "distinct_1",
        "classifier",
    ]
    rows = {
        (r[0], r[1]): dict(zip(table["columns"], r, strict=True))
        for r in table["rows"]
    }
    assert list(rows) == judged_generations(run_dir).cells()
    effects = run_dir.read_json("table_effects")
    assert isinstance(effects, dict)
    side = {(r[0], r[1], r[2]): r for r in effects["rows"]}
    for (concept, arm), row in rows.items():
        wanted = (
            None
            if arm in ("none", "prompt")
            else (0.1 if arm == "caa" else 0.4)
        )
        assert row["coeff"] == wanted
        assert row["n_prompts"] == 3
        assert row["overall_lo"] <= row["overall"] <= row["overall_hi"]
        assert row["perplexity"] == side[concept, arm, wanted][3]
        assert row["distinct_1"] == side[concept, arm, wanted][5]
        name = arm if wanted is None else f"{arm}/{wanted:g}"
        trial = next(
            t
            for t in read_trials(run_dir, f"trials_{concept}_held_out")
            if t.name == name
        )
        if np.isfinite(trial.scores).any():
            assert row["classifier"] == pytest.approx(
                float(np.nanmean(trial.scores))
            )
        else:
            assert row["classifier"] is None
    full = [r for r in rows.values() if r["n_items"] == 3]
    assert full
    steered = [r for r in full if r["coeff"] is not None]
    assert all(r["concept_score"] == 2.0 for r in steered)
    assert all(r["overall"] == 2.0 for r in steered)
    assert all(r["fluency_score"] == 2.0 for r in full)
    tuning = run_dir.read_json("table_tuning")
    assert isinstance(tuning, dict)
    assert tuning["columns"] == [
        "concept",
        "arm",
        "coeff",
        "n_items",
        "overall",
    ]
    assert len(tuning["rows"]) == 2 * len(_trial_names())
    judged = run_dir.read_json("summary_judged")
    assert isinstance(judged, dict)
    assert judged["judge_models"] == ["judge-model-1", "judge-model-2"]
    assert len(judged["rubric_sha256"]) == 64
    assert sorted(judged["agreement"]) == ["concept", "fluency"]
    assert judged["agreement"]["concept"]["rate"] == 1.0
    assert judged["agreement"]["concept"]["n"] >= 1
    assert sorted(judged["classifier_correlation"]) == ["anger", "sentiment"]
    assert judged["selected"] == [
        [concept, arm, rows[concept, arm]["coeff"]] for concept, arm in rows
    ]


def test_the_experiment_is_found_by_its_id_with_the_judging_functions() -> None:
    experiment = load_experiment("b3")

    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot
    assert callable(b3.judged_generations)
    assert callable(b3.judge_report)


def test_run_loads_only_what_the_role_needs(
    tiny_gpt2: LoadedModel,
    tiny_qwen3: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = {"gpt2": tiny_gpt2, "qwen3-1.7b": _post_trained(tiny_qwen3)}
    calls: list[str] = []

    def loader(name: str, value: Any) -> Any:
        def load(*args: Any, **kwargs: Any) -> Any:
            calls.append(name if not args else f"{name}:{args[0]}")
            return value(*args) if callable(value) else value

        return load

    class Judge:
        def __init__(self, key: str) -> None:
            self.columns = 6 if key.startswith("emotion") else 0

        def score(self, texts: Sequence[str]) -> NDArray[np.float32]:
            scores = _vowels(texts)
            if not self.columns:
                return scores
            return np.tile(scores[:, None], (1, self.columns))

    examples = _examples()
    monkeypatch.setattr(b3, "load_model", loader("model", loaded.__getitem__))
    monkeypatch.setattr(b3, "load_sst2", loader("sst2", examples["sst2"]))
    monkeypatch.setattr(
        b3, "load_emotions", loader("emotions", examples["emotions"])
    )
    monkeypatch.setattr(
        b3,
        "load_wikitext",
        loader(
            "wikitext",
            [Example(f"w{i}", t, -1, "test") for i, t in enumerate(_texts())],
        ),
    )
    monkeypatch.setattr(
        b3,
        "load_alpaca_eval",
        loader(
            "alpaca_eval",
            [
                Example(f"a{i}", t, -1, "test")
                for i, t in enumerate(_instructions())
            ],
        ),
    )
    monkeypatch.setattr(b3, "load_judge", loader("judge", Judge))

    base = _config(model="gpt2", concepts=("sentiment",))
    run(base, RunDir.create(base, out_root=tmp_path))
    assert calls == [
        "model:gpt2",
        "sst2",
        "wikitext",
        "judge:sentiment-primary",
    ]
    calls.clear()
    tuned = _config(model="qwen3-1.7b", concepts=("anger",))
    run_dir = RunDir.create(tuned, out_root=tmp_path)
    run(tuned, run_dir)
    assert calls == [
        "model:qwen3-1.7b",
        "emotions",
        "wikitext",
        "alpaca_eval",
        "judge:emotion-primary",
    ]
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["role"] == "post_trained"


def test_the_module_names_no_person_vendor_or_dash() -> None:
    source = Path(b3.__file__).read_text("utf-8")

    assert not any(dash in source for dash in DASHES)
    for word in ("openai", "anthropic", "claude", "gpt-4"):
        assert word not in source.lower()
