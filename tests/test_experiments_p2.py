"""Tests for P2, the token position and misleading context study."""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import math
from pathlib import Path
import time
from typing import Any
from typing import cast

import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.activations import PositionStrategy
from tomography.cli import load_experiment
from tomography.datasets.records import Example
from tomography.datasets.sources import OfflineCacheMissError
from tomography.datasets.sst2 import load_sst2
from tomography.datasets.stimulus import StimulusSet
from tomography.experiments import p1_templates
from tomography.experiments import p2_positions
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.p2_positions import COLUMNS
from tomography.experiments.p2_positions import DECOY_LEVELS
from tomography.experiments.p2_positions import DECOY_LINE
from tomography.experiments.p2_positions import POSITIONS
from tomography.experiments.p2_positions import ZERO_SHOT_DECOY
from tomography.experiments.p2_positions import ZERO_SHOT_HEAD
from tomography.experiments.p2_positions import ZERO_SHOT_TAIL
from tomography.experiments.p2_positions import Config
from tomography.experiments.p2_positions import execute
from tomography.experiments.p2_positions import heuristic_prompt
from tomography.experiments.p2_positions import misleading_template
from tomography.experiments.p2_positions import plot
from tomography.experiments.p2_positions import run
from tomography.experiments.p2_positions import with_decoys
from tomography.experiments.p2_positions import word_spans
from tomography.experiments.p2_positions import zero_shot_scores
from tomography.experiments.r6_truthfulqa import HEURISTIC_LEVELS
from tomography.experiments.r6_truthfulqa import HEURISTIC_TEMPLATE
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import Prepared
from tomography.experiments.sweep import ReadingConfig
from tomography.experiments.sweep import SweepResult
from tomography.experiments.variants import READING_TASKS
from tomography.experiments.variants import ReadingTask
from tomography.experiments.variants import Variant
from tomography.experiments.variants import task_splits
from tomography.judges import heuristic_score
from tomography.judges import sequence_logprob
from tomography.model import BosMode
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import bootstrap_ci
from tomography.templates import EMOTIONS
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ChatFormat
from tomography.templates import ConceptTemplate


def test_the_constants_follow_the_paper() -> None:
    assert POSITIONS == ("last", "concept", "mean")
    assert DECOY_LEVELS == ("low", "high")
    assert COLUMNS == (
        "task",
        "position",
        "condition",
        "layer",
        "accuracy",
        "lo",
        "hi",
        "change",
    )
    assert DECOY_LINE.format(noun="x") == (
        "I think the amount of x in the scenario is {question}, but I'm "
        "curious to hear what you think.\n"
    )
    assert ZERO_SHOT_TAIL == "\n{stimulus}\nAssistant: {title}:"


def test_the_two_decoy_sentences_render_the_same_text() -> None:
    for level in DECOY_LEVELS:
        line = DECOY_LINE.format(noun="happiness").replace("{question}", level)
        assert line == ZERO_SHOT_DECOY.format(noun="happiness", decoy=level)


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "p2"
    assert config.model == "gpt2"
    assert config.method == "logistic"
    assert config.n_eval_pairs == 256
    assert config.tasks == ("sentiment", "truth", "utility")
    assert config.positions == POSITIONS
    assert config.decoys == ("random", "against")
    assert config.zero_shot is True
    assert Config.from_dict(config.to_dict()) == config


def test_from_dict_reads_positions_and_decoys_as_tuples() -> None:
    config = Config.from_dict(
        {"positions": ["last", "mean"], "decoys": ["against"]}
    )
    assert config.positions == ("last", "mean")
    assert config.decoys == ("against",)


def test_from_dict_rejects_an_unknown_decoy() -> None:
    with pytest.raises(ValueError, match="decoys"):
        Config.from_dict({"decoys": ["random", "oracle"]})


def test_the_decoy_line_goes_directly_before_the_stimulus_line() -> None:
    biased = misleading_template(PAPER_TEMPLATES["concept"], "sentiment")
    line = DECOY_LINE.format(noun="sentiment")
    assert biased.user == (
        "Consider the amount of {concept} in the following:\n"
        + line
        + "{stimulus}\nThe amount of {concept} is"
    )
    assert biased.name == "concept_misleading"
    assert biased.slots == frozenset({"concept", "question", "stimulus"})


def test_the_decoy_line_keeps_the_assistant_text() -> None:
    template = PAPER_TEMPLATES["utility"]
    assert isinstance(template, ConceptTemplate)
    biased = misleading_template(template, "happiness")
    assert biased.user == (
        "Consider the happiness in the following scenario:\n"
        + DECOY_LINE.format(noun="happiness")
        + "Scenario: {stimulus}"
    )
    assert biased.assistant == template.assistant


def test_a_template_that_starts_with_the_stimulus_gets_the_line_first() -> None:
    template = ConceptTemplate("bare_stimulus", "{stimulus} is it?")
    biased = misleading_template(template, "joy")
    assert biased.user == DECOY_LINE.format(noun="joy") + "{stimulus} is it?"


def test_the_decoy_fills_the_question_slot_before_the_stimulus() -> None:
    biased = misleading_template(PAPER_TEMPLATES["concept"], "sentiment")
    rendered = biased.render(
        "a fine film", "sentiment", ChatFormat("plain"), question="low"
    )
    assert rendered.text.index("is low, but") < rendered.text.index(
        "a fine film"
    )


def test_a_template_with_a_question_slot_is_refused() -> None:
    with pytest.raises(ValueError, match="already uses the question slot"):
        misleading_template(PAPER_TEMPLATES["truthfulqa"], "truthfulness")


def _examples(n: int = 40) -> list[Example]:
    return [
        Example(f"u{i}", f"text {i}", i % 2, "val", {"pair": f"p{i // 2}"})
        for i in range(n)
    ]


def test_the_against_decoy_always_contradicts_the_label() -> None:
    rows = with_decoys(_examples(), "against", seed=0)
    assert all(
        r.meta["question"] == ("low" if r.label == 1 else "high") for r in rows
    )


def test_a_decoy_keeps_the_example_and_adds_the_question() -> None:
    examples = _examples(4)
    rows = with_decoys(examples, "against", seed=0)
    for before, after in zip(examples, rows, strict=True):
        assert dataclasses.replace(after, meta=before.meta) == before
        assert after.meta == {**before.meta, "question": after.meta["question"]}
    assert all("question" not in e.meta for e in examples)


def test_the_random_decoy_is_low_or_high_and_independent_of_the_label() -> None:
    rows = with_decoys(_examples(200), "random", seed=3)
    assert {r.meta["question"] for r in rows} == {"low", "high"}
    for label in (0, 1):
        assert {r.meta["question"] for r in rows if r.label == label} == {
            "low",
            "high",
        }


def test_the_random_decoy_depends_on_the_seed_and_the_uid_only() -> None:
    examples = _examples(60)
    first = with_decoys(examples, "random", seed=3)
    assert with_decoys(examples, "random", seed=3) == first
    reversed_rows = with_decoys(examples[::-1], "random", seed=3)[::-1]
    assert reversed_rows == first
    other = with_decoys(examples, "random", seed=4)
    assert [r.meta["question"] for r in other] != [
        r.meta["question"] for r in first
    ]
    for row in first:
        draw = np.random.default_rng(derive_seed(3, "decoy", row.uid))
        assert row.meta["question"] == DECOY_LEVELS[int(draw.integers(2))]


@pytest.mark.parametrize("decoy", ["random", "against"])
def test_a_decoy_needs_a_binary_label(decoy: p2_positions.Decoy) -> None:
    examples = [Example("e0", "text", 2, "val")]
    with pytest.raises(ValueError, match="binary label"):
        with_decoys(examples, decoy, seed=0)


def test_word_spans_take_the_last_occurrence_and_none_when_absent() -> None:
    spans = word_spans(["a happiness b happiness c", "none here"], "happiness")
    assert spans == ((14, 23), None)


def test_the_truthfulness_prompt_is_r6s_heuristic_template() -> None:
    task = READING_TASKS["truth"]
    assert heuristic_prompt(
        task, "the sky is blue", None
    ) == HEURISTIC_TEMPLATE.replace("{scenario}", "the sky is blue")


def test_the_decoy_sentence_sits_between_the_levels_and_the_scenario() -> None:
    task = READING_TASKS["utility"]
    prompt = heuristic_prompt(task, "a day at the sea", "high")
    plain = heuristic_prompt(task, "a day at the sea", None)
    sentence = ZERO_SHOT_DECOY.format(noun="happiness", decoy="high")
    head = ZERO_SHOT_HEAD.format(noun="happiness")
    assert plain.startswith(head)
    assert prompt == head + sentence + plain[len(head) :]
    assert prompt.endswith("\na day at the sea\nAssistant: Happiness:")


def test_the_prompt_constants_are_read_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p2_positions, "ZERO_SHOT_HEAD", "Rate {noun}.\n")
    monkeypatch.setattr(p2_positions, "ZERO_SHOT_DECOY", "Hint {decoy}.\n")
    monkeypatch.setattr(p2_positions, "ZERO_SHOT_TAIL", "|{stimulus}|{title}")
    task = READING_TASKS["utility"]
    assert heuristic_prompt(task, "s", "low") == (
        "Rate happiness.\nHint low.\n|s|Happiness"
    )


def _short_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(p2_positions, "DECOY_LINE", "Hint {{question}}.\n")
    monkeypatch.setattr(p2_positions, "ZERO_SHOT_HEAD", "Rate {noun}:\n")
    monkeypatch.setattr(p2_positions, "ZERO_SHOT_DECOY", "Hint {decoy}.\n")
    monkeypatch.setattr(p2_positions, "ZERO_SHOT_TAIL", "{stimulus}\n{title}:")


def _levels(loaded: LoadedModel, prompts: list[str]) -> np.ndarray:
    levels = [" " + level for level in HEURISTIC_LEVELS]
    flat = sequence_logprob(
        loaded.model,
        loaded.tokenizer,
        [prompt for prompt in prompts for _ in levels],
        levels * len(prompts),
        batch_size=4,
        bos_id=loaded.bos_id if loaded.default_bos == "prepend" else None,
    )
    return heuristic_score(flat.reshape(len(prompts), len(levels)))


TEXTS = ["a fine film", "a dull film", "so so"]


def test_a_zero_shot_score_is_the_expected_level_of_its_prompt(
    tiny_arch: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_prompts(monkeypatch)
    task = READING_TASKS["sentiment"]
    decoys = ["low", "high", "low"]
    scores = zero_shot_scores(tiny_arch, task, TEXTS, decoys, batch_size=4)
    assert scores.shape == (3,)
    assert scores.dtype == np.float64
    assert np.all(np.abs(scores) <= 1.0)
    prompts = [
        heuristic_prompt(task, t, d) for t, d in zip(TEXTS, decoys, strict=True)
    ]
    np.testing.assert_array_equal(scores, _levels(tiny_arch, prompts))


def test_without_decoys_the_prompts_carry_no_sentence(
    tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_prompts(monkeypatch)
    task = READING_TASKS["sentiment"]
    plain = zero_shot_scores(tiny_gpt2, task, TEXTS, None, batch_size=4)
    prompts = [heuristic_prompt(task, text, None) for text in TEXTS]
    np.testing.assert_array_equal(plain, _levels(tiny_gpt2, prompts))
    misled = zero_shot_scores(tiny_gpt2, task, TEXTS, ["low"] * 3, batch_size=4)
    assert not np.array_equal(plain, misled)


def test_every_level_continues_every_prompt_in_order(
    tiny_gpt2: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_prompts(monkeypatch)
    real = p2_positions.sequence_logprob
    calls: list[tuple[list[str], list[str]]] = []

    def spy(
        model: Any,
        tokenizer: Any,
        prompts: Sequence[str],
        continuations: Sequence[str],
        **kwargs: Any,
    ) -> Any:
        calls.append((list(prompts), list(continuations)))
        return real(model, tokenizer, prompts, continuations, **kwargs)

    monkeypatch.setattr(p2_positions, "sequence_logprob", spy)
    task = READING_TASKS["sentiment"]
    zero_shot_scores(tiny_gpt2, task, TEXTS[:2], None, batch_size=4)
    ((prompts, continuations),) = calls
    assert continuations == [" " + level for level in HEURISTIC_LEVELS] * 2
    assert prompts == [
        heuristic_prompt(task, text, None)
        for text in TEXTS[:2]
        for _ in HEURISTIC_LEVELS
    ]


@pytest.mark.parametrize(
    ("default_bos", "expected"), [("prepend", 5), ("none", None)]
)
def test_the_bos_id_is_passed_only_when_the_model_prepends_one(
    tiny_gpt2: LoadedModel,
    monkeypatch: pytest.MonkeyPatch,
    default_bos: BosMode,
    expected: int | None,
) -> None:
    _short_prompts(monkeypatch)
    seen: list[int | None] = []

    def spy(*args: Any, bos_id: int | None, **kwargs: Any) -> Any:
        seen.append(bos_id)
        return np.zeros(len(args[2]), dtype=np.float32)

    monkeypatch.setattr(p2_positions, "sequence_logprob", spy)
    loaded = dataclasses.replace(tiny_gpt2, bos_id=5, default_bos=default_bos)
    zero_shot_scores(
        loaded, READING_TASKS["sentiment"], TEXTS, None, batch_size=4
    )
    assert seen == [expected]


def test_zero_shot_scores_need_one_decoy_per_text(
    tiny_gpt2: LoadedModel,
) -> None:
    with pytest.raises(ValueError, match="one decoy per text"):
        zero_shot_scores(
            tiny_gpt2, READING_TASKS["sentiment"], TEXTS, ["low"], batch_size=4
        )


def _sentiment(per_class: int = 12) -> list[Example]:
    words = {1: ("good", "fine", "great"), 0: ("bad", "poor", "dull")}
    return [
        Example(
            f"sst2-{split}-{label}-{i}",
            f"a {words[label][i % 3]} one {i}",
            label,
            split,
        )
        for split in ("train", "val")
        for label in (0, 1)
        for i in range(per_class)
    ]


def _facts(per_class: int = 8) -> list[Example]:
    return [
        Example(
            f"facts-{split}-{label}-{i}",
            f"fact {i} is {'yes' if label else 'no'}",
            label,
            split,
        )
        for split in ("train", "val", "test")
        for label in (0, 1)
        for i in range(per_class)
    ]


def _utility(pairs: int = 8) -> list[Example]:
    words = {
        1: ("sun", "gift", "joy", "fun"),
        0: ("rain", "loss", "pain", "fear"),
    }
    rows = []
    for split in ("train", "val", "test"):
        for i in range(pairs):
            pair = f"ethics-{split}-{i}"
            for suffix, label in (("a", 1), ("b", 0)):
                rows.append(
                    Example(
                        f"{pair}:{suffix}",
                        f"{words[label][i % 4]} {i}",
                        label,
                        split,
                        {"pair": pair},
                    )
                )
    return rows


def _emotions(per_class: int = 6) -> list[Example]:
    words = ("sun", "rain", "dog", "gift")
    return [
        Example(
            f"emotion-{name}-{split}-{i}",
            f"{words[i % 4]} {i}",
            label,
            split,
            {"emotion": name},
        )
        for split in ("train", "val", "test")
        for label, name in enumerate(EMOTIONS)
        for i in range(per_class)
    ]


def _sources() -> dict[str, list[Example]]:
    return {
        "sst2": _sentiment(),
        "facts": _facts(),
        "ethics_utility": _utility(),
        "emotions": _emotions(),
    }


def _config(**overrides: Any) -> Config:
    base = Config(model="tiny", n_seeds=2, n_pairs=4, n_eval_pairs=3)
    return dataclasses.replace(base, **overrides)


def _run_dir(config: Config, tmp_path: Path) -> RunDir:
    return RunDir.create(config, out_root=tmp_path)


def _refused(
    config: Config,
    tmp_path: Path,
    examples: Mapping[str, Sequence[Example]] | None = None,
) -> None:
    execute(
        config,
        _run_dir(config, tmp_path),
        cast(Any, None),
        _sources() if examples is None else examples,
    )


def test_a_one_vs_rest_task_is_refused_before_any_model_call(
    tmp_path: Path,
) -> None:
    config = _config(tasks=("emotion_happiness",))
    with pytest.raises(
        ValueError, match="p2 reads binary tasks only, not 'emotion_happiness'"
    ):
        _refused(config, tmp_path)


def test_a_position_outside_the_three_is_refused(tmp_path: Path) -> None:
    config = _config(positions=("last", "response"))
    with pytest.raises(ValueError, match="unknown position 'response'"):
        _refused(config, tmp_path)


def test_no_position_at_all_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one position"):
        _refused(_config(positions=()), tmp_path)


def test_an_unknown_task_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown task 'nope'"):
        _refused(_config(tasks=("nope",)), tmp_path)


def test_a_missing_source_is_refused(tmp_path: Path) -> None:
    examples = {k: v for k, v in _sources().items() if k != "facts"}
    config = _config(tasks=("truth",))
    with pytest.raises(ValueError, match="no examples for source 'facts'"):
        _refused(config, tmp_path, examples)


@pytest.fixture
def short_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    _short_prompts(monkeypatch)


def _execute(config: Config, loaded: LoadedModel, tmp_path: Path) -> RunDir:
    run_dir = _run_dir(config, tmp_path)
    execute(config, run_dir, loaded, _sources())
    return run_dir


def _json(run_dir: RunDir, name: str) -> Any:
    return run_dir.read_json(name)


def _stems(run_dir: RunDir) -> set[str]:
    return {p.name for p in run_dir.path.iterdir()}


READER = SERIES.index("reader")
CONDITIONS = ["original", "random", "against"]


@pytest.mark.usefixtures("short_prompts")
def test_three_tasks_write_forty_files_and_thirty_six_rows(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(_config(), tiny_arch, tmp_path)
    stems = {"table.json", "summary.json", "config.json", "meta.json"}
    for task in ("sentiment", "truth", "utility"):
        for stem in (
            f"sweep_{task}",
            f"curves_{task}",
            f"positions_{task}",
            f"curves_{task}_last",
            f"curves_{task}_concept",
            f"curves_{task}_mean",
        ):
            stems |= {f"{stem}.npz", f"{stem}.json"}
    assert _stems(run_dir) == stems
    assert len(stems) == 40
    rows = _json(run_dir, "table")
    assert len(rows) == 36
    assert all(tuple(row) == COLUMNS for row in rows)
    assert all(
        row["change"] == 0.0 for row in rows if row["condition"] == "original"
    )
    summary = _json(run_dir, "summary")
    assert summary["tasks"] == ["sentiment", "truth", "utility"]
    assert summary["conditions"] == CONDITIONS
    assert summary["skipped"] == {"sentiment": {}, "truth": {}, "utility": {}}
    assert summary["n_eval_pairs"] == {
        task: {"select": 3, "report": 3}
        for task in ("sentiment", "truth", "utility")
    }


@pytest.mark.usefixtures("short_prompts")
def test_the_cube_holds_every_train_and_eval_position_and_condition(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(_config(tasks=("sentiment",)), tiny_gpt2, tmp_path)
    arrays = run_dir.read_arrays("sweep_sentiment")
    assert arrays["accuracy"].shape == (2, 3, 3, 3, 3, 2, 3)
    assert arrays["layers"].tolist() == [1, 2, 3]
    assert arrays["zero_shot"].shape == (3,)
    assert np.all((arrays["zero_shot"] >= 0) & (arrays["zero_shot"] <= 1))
    sidecar = _json(run_dir, "sweep_sentiment")
    assert sidecar["splits"] == ["select", "report"]
    assert sidecar["positions"] == list(POSITIONS)
    assert sidecar["conditions"] == CONDITIONS
    assert sidecar["series"] == list(SERIES)


@pytest.mark.usefixtures("short_prompts")
def test_each_position_is_read_at_its_best_select_layer_when_original(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(_config(tasks=("utility",)), tiny_gpt2, tmp_path)
    arrays = run_dir.read_arrays("sweep_utility")
    layers = arrays["layers"].tolist()
    expected = {
        position: best_layer(
            arrays["accuracy"][0, i, i, 0, READER].mean(axis=0), layers
        )
        for i, position in enumerate(POSITIONS)
    }
    assert _json(run_dir, "sweep_utility")["layers"] == expected
    assert _json(run_dir, "summary")["layers"] == {"utility": expected}
    for row in _json(run_dir, "table"):
        if row["position"] != "zero_shot":
            assert row["layer"] == expected[row["position"]]


@pytest.mark.usefixtures("short_prompts")
def test_the_table_is_the_report_accuracy_at_the_position_layer(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(_config(tasks=("sentiment",)), tiny_gpt2, tmp_path)
    accuracy = run_dir.read_arrays("sweep_sentiment")["accuracy"]
    sidecar = _json(run_dir, "sweep_sentiment")
    rows = [r for r in _json(run_dir, "table") if r["position"] != "zero_shot"]
    assert [(r["position"], r["condition"]) for r in rows] == [
        (p, c) for p in POSITIONS for c in CONDITIONS
    ]
    for row in rows:
        i = POSITIONS.index(row["position"])
        j = CONDITIONS.index(row["condition"])
        k = [1, 2, 3].index(sidecar["layers"][row["position"]])
        values = accuracy[1, i, i, j, READER, :, k]
        original = accuracy[1, i, i, 0, READER, :, k].mean()
        lo, hi = bootstrap_ci(values, seed=0)
        assert row["accuracy"] == pytest.approx(values.mean())
        assert (row["lo"], row["hi"]) == pytest.approx((lo, hi))
        assert row["change"] == pytest.approx(
            100.0 * (values.mean() - original) / original
        )


@pytest.mark.usefixtures("short_prompts")
def test_the_zero_shot_rows_follow_the_reader_rows(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = {e.text: e.label for e in _sentiment()}

    def stub(
        loaded: LoadedModel,
        task: ReadingTask,
        texts: Sequence[str],
        decoys: Sequence[str] | None,
        *,
        batch_size: int,
    ) -> NDArray[np.float64]:
        if decoys is None:
            return np.array([2.0 * labels[t] - 1.0 for t in texts])
        return np.array([1.0 if d == "high" else -1.0 for d in decoys])

    monkeypatch.setattr(p2_positions, "zero_shot_scores", stub)
    run_dir = _execute(_config(tasks=("sentiment",)), tiny_gpt2, tmp_path)
    zero = run_dir.read_arrays("sweep_sentiment")["zero_shot"]
    rows = [r for r in _json(run_dir, "table") if r["position"] == "zero_shot"]
    assert [r["condition"] for r in rows] == CONDITIONS
    assert all(r["layer"] is None for r in rows)
    assert [r["accuracy"] for r in rows] == pytest.approx(zero.tolist())
    original, random, against = rows
    assert (original["accuracy"], original["change"]) == (1.0, 0.0)
    assert (against["accuracy"], against["change"]) == (0.0, -100.0)
    assert random["change"] == pytest.approx(100.0 * (random["accuracy"] - 1.0))
    for row in rows:
        assert row["lo"] <= row["accuracy"] <= row["hi"]


@pytest.mark.usefixtures("short_prompts")
def test_the_curves_and_the_matrix_come_from_the_cube(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(_config(tasks=("utility",)), tiny_gpt2, tmp_path)
    accuracy = run_dir.read_arrays("sweep_utility")["accuracy"]
    layers = _json(run_dir, "sweep_utility")["layers"]
    assert _json(run_dir, "curves_utility")["series"] == list(POSITIONS)
    np.testing.assert_array_equal(
        run_dir.read_arrays("curves_utility")["per_seed"],
        np.stack([accuracy[1, i, i, 0, READER] for i in range(3)]),
    )
    for i, position in enumerate(POSITIONS):
        stem = f"curves_utility_{position}"
        assert _json(run_dir, stem)["series"] == CONDITIONS
        np.testing.assert_array_equal(
            run_dir.read_arrays(stem)["per_seed"],
            accuracy[1, i, i, :, READER],
        )
    values = run_dir.read_arrays("positions_utility")["values"]
    sidecar = _json(run_dir, "positions_utility")
    assert sidecar["rows"] == sidecar["cols"] == list(POSITIONS)
    for i, position in enumerate(POSITIONS):
        k = [1, 2, 3].index(layers[position])
        np.testing.assert_array_equal(
            values[i], accuracy[1, i, :, 0, READER, :, k].mean(axis=-1)
        )


@pytest.mark.usefixtures("short_prompts")
def test_the_original_reader_at_the_last_token_is_p1s_paper_reader(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    p2 = _execute(_config(tasks=("sentiment",)), tiny_gpt2, tmp_path)
    p1_config = p1_templates.Config(
        model="tiny",
        n_seeds=2,
        n_pairs=4,
        n_eval_pairs=3,
        tasks=("sentiment",),
        kinds=("paper",),
    )
    p1 = RunDir.create(p1_config, out_root=tmp_path)
    p1_templates.execute(p1_config, p1, tiny_gpt2, _sources())
    first = p1.read_arrays("sweep_sentiment")["accuracy"]
    second = p2.read_arrays("sweep_sentiment")["accuracy"]
    for split in (0, 1):
        np.testing.assert_array_equal(
            second[split, 0, 0, 0, READER], first[split, 0, 0, READER]
        )


@pytest.mark.usefixtures("short_prompts")
def test_training_never_sees_a_decoy_and_the_conditions_share_pairs(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = p2_positions.task_set
    calls: list[tuple[str, tuple[str, ...], int | None, int, bool]] = []

    def spy(
        task: ReadingTask,
        variant: Variant,
        examples: Sequence[Example],
        *,
        n_pairs: int | None,
        seed: int,
    ) -> StimulusSet:
        calls.append(
            (
                variant.name,
                tuple(e.uid for e in examples),
                n_pairs,
                seed,
                any("question" in e.meta for e in examples),
            )
        )
        return real(task, variant, examples, n_pairs=n_pairs, seed=seed)

    monkeypatch.setattr(p2_positions, "task_set", spy)
    config = _config(tasks=("sentiment",), positions=("last",), zero_shot=False)
    _execute(config, tiny_gpt2, tmp_path)
    paper = [call for call in calls if call[0] == "paper"]
    misleading = [call for call in calls if call[0] == "misleading"]
    assert not any(call[4] for call in paper)
    assert [call[3] for call in paper if call[2] == 4] == [
        derive_seed(0, "train", 0),
        derive_seed(0, "train", 1),
    ]
    original = {call[3]: call[1] for call in paper if call[2] == 3}
    assert len(original) == 2
    assert len(misleading) == 4
    for _, uids, n_pairs, seed, decoyed in misleading:
        assert decoyed
        assert n_pairs == 3
        assert original[seed] == uids


@pytest.mark.usefixtures("short_prompts")
def test_the_concept_token_lies_after_the_decoy_line(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = p2_positions.prepare
    seen: list[StimulusSet] = []

    def spy(
        loaded: LoadedModel,
        stimuli: StimulusSet,
        position: PositionStrategy,
        config: ReadingConfig,
    ) -> Prepared:
        seen.append(stimuli)
        return real(loaded, stimuli, position, config)

    monkeypatch.setattr(p2_positions, "prepare", spy)
    config = _config(
        tasks=("sentiment", "utility"),
        positions=("concept",),
        decoys=("against",),
        zero_shot=False,
    )
    _execute(config, tiny_gpt2, tmp_path)
    decoyed = [s for s in seen if "Hint " in s.texts[0]]
    assert len(decoyed) == 4
    for stimuli in decoyed:
        for text, span in zip(
            stimuli.texts, stimuli.concept_spans, strict=True
        ):
            assert span is not None
            assert text.index("Hint ") < span[0]
            assert text[span[0] : span[1]] in ("sentiment", "happiness")
    assert all(span is not None for s in seen for span in s.concept_spans)


@pytest.mark.usefixtures("short_prompts")
def test_the_zero_shot_baseline_reads_the_report_items_and_their_decoys(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = p2_positions.zero_shot_scores
    calls: list[tuple[list[str], list[str] | None, int]] = []

    def spy(
        loaded: LoadedModel,
        task: ReadingTask,
        texts: Sequence[str],
        decoys: Sequence[str] | None,
        *,
        batch_size: int,
    ) -> NDArray[np.float64]:
        calls.append(
            (list(texts), None if decoys is None else list(decoys), batch_size)
        )
        return real(loaded, task, texts, decoys, batch_size=batch_size)

    monkeypatch.setattr(p2_positions, "zero_shot_scores", spy)
    config = _config(tasks=("sentiment",), positions=("last",))
    _execute(config, tiny_gpt2, tmp_path)
    assert len(calls) == 3
    assert [call[2] for call in calls] == [config.batch_size] * 3
    (
        (original, original_decoys, _),
        (random, random_decoys, _),
        (
            against,
            against_decoys,
            _,
        ),
    ) = calls
    assert original_decoys is None
    assert original == random == against
    labels = {e.text: e.label for e in _sentiment() if e.split == "val"}
    assert set(original) <= set(labels)
    assert set(random_decoys or []) <= {"low", "high"}
    assert [labels[text] for text in against] == [
        1 if decoy == "low" else 0 for decoy in against_decoys or []
    ]


@pytest.mark.usefixtures("short_prompts")
def test_without_the_zero_shot_baseline_there_are_no_zero_shot_rows(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the zero-shot baseline must not run")

    monkeypatch.setattr(p2_positions, "zero_shot_scores", refuse)
    config = _config(tasks=("sentiment",), zero_shot=False)
    run_dir = _execute(config, tiny_gpt2, tmp_path)
    assert run_dir.read_arrays("sweep_sentiment")["zero_shot"].shape == (0,)
    rows = _json(run_dir, "table")
    assert len(rows) == 9
    assert "zero_shot" not in {row["position"] for row in rows}


def _without_the_word(monkeypatch: pytest.MonkeyPatch) -> None:
    bare = dataclasses.replace(READING_TASKS["utility"], word=None)
    monkeypatch.setattr(
        p2_positions, "READING_TASKS", {**READING_TASKS, "utility": bare}
    )


@pytest.mark.usefixtures("short_prompts")
def test_a_task_without_a_concept_slot_or_word_skips_that_position(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _without_the_word(monkeypatch)
    run_dir = _execute(_config(tasks=("utility",)), tiny_gpt2, tmp_path)
    summary = _json(run_dir, "summary")
    assert summary["skipped"] == {
        "utility": {"concept": "task 'utility' has no concept slot or word"}
    }
    assert list(summary["layers"]["utility"]) == ["last", "mean"]
    assert _json(run_dir, "sweep_utility")["positions"] == ["last", "mean"]
    accuracy = run_dir.read_arrays("sweep_utility")["accuracy"]
    assert accuracy.shape[1:3] == (2, 2)
    positions = {row["position"] for row in _json(run_dir, "table")}
    assert positions == {"last", "mean", "zero_shot"}


def test_a_task_with_no_position_left_is_refused_before_any_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _without_the_word(monkeypatch)
    config = _config(tasks=("utility",), positions=("concept",))
    with pytest.raises(ValueError, match="task 'utility' has no position"):
        _refused(config, tmp_path)


@pytest.mark.usefixtures("short_prompts")
def test_the_same_config_gives_the_same_table(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    config = _config(tasks=("sentiment",))
    first = _execute(config, tiny_gpt2, tmp_path / "a")
    second = _execute(config, tiny_gpt2, tmp_path / "b")
    assert _json(first, "table") == _json(second, "table")


def test_a_change_from_a_zero_original_is_undefined_unless_unchanged() -> None:
    assert math.isnan(p2_positions._change(0.4, 0.0))
    assert p2_positions._change(0.0, 0.0) == 0.0
    assert p2_positions._change(0.3, 0.5) == pytest.approx(-40.0)


@pytest.mark.usefixtures("short_prompts")
def test_the_cube_reads_a_reader_of_one_position_at_another(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = p2_positions.sweep_from_activations
    calls: list[tuple[Sequence[Prepared], dict[str, Prepared]]] = []

    def spy(
        train: Sequence[Prepared],
        evals: Mapping[str, Prepared],
        **kwargs: Any,
    ) -> SweepResult:
        calls.append((list(train), dict(evals)))
        return real(train, evals, **kwargs)

    monkeypatch.setattr(p2_positions, "sweep_from_activations", spy)
    config = _config(tasks=("sentiment",), zero_shot=False, seed=3)
    run_dir = _execute(config, tiny_gpt2, tmp_path)
    accuracy = run_dir.read_arrays("sweep_sentiment")["accuracy"]
    assert len(calls) == len(POSITIONS)
    for i, (train, evals) in enumerate(calls):
        for j, read in enumerate(POSITIONS):
            name = f"report/original/{read}"
            direct = real(
                train,
                {name: evals[name]},
                method=config.method,
                preprocess=config.preprocess,
                seed=config.seed,
            )
            np.testing.assert_array_equal(
                accuracy[1, i, j, 0], direct.per_seed(name)
            )


def _truth_sets(per_class: int = 6) -> list[Example]:
    return [
        Example(
            f"got-cities-{split}-{label}-{i}",
            f"city {i} {label}",
            label,
            split,
            {"subset": "cities"},
        )
        for split in ("train", "val", "test")
        for label in (0, 1)
        for i in range(per_class)
    ]


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("p2")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


@pytest.mark.usefixtures("short_prompts")
def test_run_loads_the_model_and_the_distinct_sources_once(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []
    loaded_sources: list[list[str]] = []

    def fake_load_model(key: str) -> LoadedModel:
        asked.append(key)
        return tiny_gpt2

    def fake_load_sources(sources: Sequence[str]) -> dict[str, list[Example]]:
        loaded_sources.append(list(sources))
        return {**_sources(), "geometry_of_truth": _truth_sets()}

    monkeypatch.setattr(p2_positions, "load_model", fake_load_model)
    monkeypatch.setattr(p2_positions, "load_sources", fake_load_sources)
    config = _config(
        tasks=("truth_cities", "truth", "utility"),
        positions=("last",),
        decoys=("against",),
        zero_shot=False,
    )
    run_dir = _run_dir(config, tmp_path)
    run(config, run_dir)
    assert asked == ["tiny"]
    assert loaded_sources == [["geometry_of_truth", "facts", "ethics_utility"]]
    assert _json(run_dir, "summary")["tasks"] == [
        "truth_cities",
        "truth",
        "utility",
    ]


def test_run_refuses_an_unknown_task_before_loading_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("nothing may load before the tasks are checked")

    monkeypatch.setattr(p2_positions, "load_model", refuse)
    monkeypatch.setattr(p2_positions, "load_sources", refuse)
    config = _config(tasks=("nope",))
    with pytest.raises(ValueError, match="unknown task 'nope'"):
        run(config, _run_dir(config, tmp_path))


@pytest.mark.usefixtures("short_prompts")
def test_plot_draws_the_curves_and_the_position_matrix_of_every_task(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _execute(_config(), tiny_gpt2, tmp_path)
    paths = plot(run_dir, tmp_path / "figures")
    names = {path.name for path in paths}
    assert len(paths) == len(names) == 45
    assert {p.suffix for p in paths} == {".pdf", ".png", ".svg"}
    for task in ("sentiment", "truth", "utility"):
        for stem in (
            f"curves_{task}",
            f"curves_{task}_last",
            f"curves_{task}_concept",
            f"curves_{task}_mean",
            f"positions_{task}",
        ):
            assert {f"{stem}.pdf", f"{stem}.png", f"{stem}.svg"} <= names
    assert all(path.is_file() for path in paths)


@pytest.mark.usefixtures("short_prompts")
def test_plot_follows_the_positions_a_task_was_read_at(
    tiny_gpt2: LoadedModel, tmp_path: Path
) -> None:
    config = _config(tasks=("utility",), positions=("mean", "last"))
    run_dir = _execute(config, tiny_gpt2, tmp_path)
    names = {p.name for p in plot(run_dir, tmp_path / "figures")}
    assert "curves_utility_concept.png" not in names
    assert {"curves_utility_mean.png", "curves_utility_last.png"} <= names
    assert len(names) == 3 * 4


@pytest.mark.slow
def test_p2_on_gpt2_reads_sentiment_under_a_decoy_against_zero_shot(
    tmp_path: Path,
) -> None:
    try:
        sst2 = load_sst2(offline=True)
    except (OfflineCacheMissError, OSError) as error:
        pytest.skip(f"a cache is missing: {error}")
    config = Config(
        tasks=("sentiment",),
        positions=("last", "mean"),
        n_seeds=2,
        n_pairs=16,
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    started = time.perf_counter()
    execute(config, run_dir, load_model("gpt2"), {"sst2": sst2})
    elapsed = time.perf_counter() - started
    rows = cast(list[dict[str, Any]], run_dir.read_json("table"))
    for row in rows:
        print(
            {
                key: row[key]
                for key in ("position", "condition", "accuracy", "change")
            }
        )
    print(f"{elapsed:.1f} seconds for {len(rows)} rows")
    last = next(
        row
        for row in rows
        if (row["position"], row["condition"]) == ("last", "original")
    )
    assert last["accuracy"] > 0.5
    assert {row["position"] for row in rows} == {"last", "mean", "zero_shot"}


@pytest.mark.usefixtures("short_prompts")
def test_every_set_is_read_at_the_position_of_its_name(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = p2_positions.sweep_from_activations
    calls: list[tuple[list[Prepared], dict[str, Prepared]]] = []

    def spy(
        train: Sequence[Prepared],
        evals: Mapping[str, Prepared],
        **kwargs: Any,
    ) -> SweepResult:
        calls.append((list(train), dict(evals)))
        return real(train, evals, **kwargs)

    monkeypatch.setattr(p2_positions, "sweep_from_activations", spy)
    config = _config(tasks=("sentiment", "utility"), zero_shot=False)
    _execute(config, tiny_gpt2, tmp_path)
    assert len(calls) == 2 * len(POSITIONS)
    for k, (train, evals) in enumerate(calls):
        assert {act.position for _, act in train} == {POSITIONS[k % 3]}
        assert len(evals) == 2 * len(CONDITIONS) * len(POSITIONS)
        for name, (_, activations) in evals.items():
            assert activations.position == name.rsplit("/", 1)[1]


@pytest.mark.usefixtures("short_prompts")
def test_each_positions_layer_is_chosen_on_the_select_split(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = p2_positions.sweep_from_activations

    def skewed(
        train: Sequence[Prepared],
        evals: Mapping[str, Prepared],
        **kwargs: Any,
    ) -> SweepResult:
        sweep = real(train, evals, **kwargs)
        accuracy = sweep.accuracy.copy()
        accuracy[:, READER] = 0.5
        for i, name in enumerate(sweep.eval_names):
            layer = 0 if name.startswith("select/") else -1
            accuracy[i, READER, :, layer] = 0.9
        return dataclasses.replace(sweep, accuracy=accuracy)

    monkeypatch.setattr(p2_positions, "sweep_from_activations", skewed)
    config = _config(tasks=("sentiment", "utility"), zero_shot=False)
    run_dir = _execute(config, tiny_gpt2, tmp_path)
    assert _json(run_dir, "summary")["layers"] == {
        task: {position: 1 for position in POSITIONS}
        for task in ("sentiment", "utility")
    }
    assert {row["layer"] for row in _json(run_dir, "table")} == {1}


@pytest.mark.usefixtures("short_prompts")
def test_the_random_decoys_are_drawn_with_the_run_seed(
    tiny_gpt2: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = p2_positions.task_set
    drawn: list[list[tuple[str, Any]]] = []

    def spy(
        task: ReadingTask,
        variant: Variant,
        examples: Sequence[Example],
        *,
        n_pairs: int | None,
        seed: int,
    ) -> StimulusSet:
        if variant.name == "misleading":
            drawn.append([(e.uid, e.meta["question"]) for e in examples])
        return real(task, variant, examples, n_pairs=n_pairs, seed=seed)

    monkeypatch.setattr(p2_positions, "task_set", spy)
    config = _config(
        tasks=("sentiment",),
        positions=("last",),
        decoys=("random",),
        zero_shot=False,
        seed=5,
    )
    _execute(config, tiny_gpt2, tmp_path)
    splits = task_splits(READING_TASKS["sentiment"], _sources(), seed=5)
    expected = [
        [(e.uid, e.meta["question"]) for e in with_decoys(p, "random", seed=5)]
        for p in (splits.select, splits.report)
    ]
    assert drawn == expected


def test_no_task_is_refused_before_any_model_call(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one task"):
        _refused(_config(tasks=()), tmp_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tasks": ()}, "at least one task"),
        ({"tasks": ("nope",)}, "unknown task 'nope'"),
        ({"tasks": ("emotion_happiness",)}, "binary tasks only"),
        ({"positions": ()}, "at least one position"),
        ({"positions": ("response",)}, "unknown position 'response'"),
    ],
)
def test_run_checks_the_config_before_loading_anything(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    message: str,
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("nothing may load before the config is checked")

    monkeypatch.setattr(p2_positions, "load_model", refuse)
    monkeypatch.setattr(p2_positions, "load_sources", refuse)
    config = _config(**overrides)
    with pytest.raises(ValueError, match=message):
        run(config, _run_dir(config, tmp_path))
