"""Tests for the E1 self-threat emotion reading experiment."""

from collections.abc import Sequence
import dataclasses
from pathlib import Path
from typing import Any
from typing import NamedTuple

from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.text import Text
import numpy as np
from numpy.typing import NDArray
import pytest
import torch

from tomography.cli import load_experiment
from tomography.control.generate import generate
from tomography.datasets.emotions import EMOTIONS
from tomography.datasets.records import Example
from tomography.datasets.self_threat import CONDITIONS
from tomography.datasets.self_threat import load_self_threat
from tomography.experiments import e1_self_threat
from tomography.experiments.base import RunDir
from tomography.experiments.e1_self_threat import BASE_FRAME
from tomography.experiments.e1_self_threat import CONTRASTS
from tomography.experiments.e1_self_threat import SELF_REPORT
from tomography.experiments.e1_self_threat import Config
from tomography.experiments.e1_self_threat import contrast_estimates
from tomography.experiments.e1_self_threat import contrast_rows
from tomography.experiments.e1_self_threat import execute
from tomography.experiments.e1_self_threat import frame_for
from tomography.experiments.e1_self_threat import parse_feeling
from tomography.experiments.e1_self_threat import plot
from tomography.experiments.e1_self_threat import plot_models
from tomography.experiments.e1_self_threat import run
from tomography.experiments.sweep import reading_sweep
from tomography.model import MODELS
from tomography.model import LoadedModel
from tomography.reading.stats import holm_correction

N_FRAMES = 24
FRAME_HEAD = (
    "The following is a conversation between a human and an AI assistant.\n"
    "Human: "
)
CHAT_HEAD = "<|im_start|>user\n"
CHAT_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _scenarios(per_split: int = 14) -> list[Example]:
    return [
        Example(
            uid=f"emotion-{name}-{split}-{r}",
            text=f"I feel {name} and {name} again, scene {r} in {split}",
            label=k,
            split=split,
            meta={"emotion": name},
        )
        for split in ("train", "val", "test")
        for k, name in enumerate(EMOTIONS)
        for r in range(per_split)
    ]


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        n_seeds=2,
        n_pairs=6,
        n_random=12,
        n_permutations=200,
        n_boot=200,
        reply_tokens=3,
        report_tokens=2,
        batch_size=16,
    )
    return dataclasses.replace(base, **overrides)


class StubJudge:
    """Scores a text by its length and records every call."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def score(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls.append(list(texts))
        return probabilities(texts)


def probabilities(texts: Sequence[str]) -> NDArray[np.float32]:
    rows = [
        [((len(t) + 3) * (k + 1) % 11) / 20.0 for k in range(6)] for t in texts
    ]
    return np.asarray(rows, dtype=np.float32).reshape(len(texts), 6)


def _chat(loaded: LoadedModel) -> LoadedModel:
    return dataclasses.replace(loaded, checkpoint=MODELS["qwen3-0.6b"])


class Finished(NamedTuple):
    config: Config
    run_dir: RunDir
    loaded: LoadedModel
    judge: StubJudge


_FINISHED: dict[str, Finished] = {}


def _finish(
    key: str, loaded: LoadedModel, factory: pytest.TempPathFactory
) -> Finished:
    if key not in _FINISHED:
        config = _config(model=f"tiny-{key}")
        run_dir = RunDir.create(config, out_root=factory.mktemp("e1"))
        judge = StubJudge()
        execute(config, run_dir, loaded, _scenarios(), judge)
        run_dir.finalize()
        _FINISHED[key] = Finished(config, run_dir, loaded, judge)
    return _FINISHED[key]


@pytest.fixture
def finished(
    tiny_arch: LoadedModel, tmp_path_factory: pytest.TempPathFactory
) -> Finished:
    """One base-model run per architecture, shared by read-only tests."""
    return _finish(type(tiny_arch.model).__name__, tiny_arch, tmp_path_factory)


@pytest.fixture
def chat_finished(
    tiny_qwen3: LoadedModel, tmp_path_factory: pytest.TempPathFactory
) -> Finished:
    """One run of a post-trained tiny model, shared by read-only tests."""
    return _finish("chat", _chat(tiny_qwen3), tmp_path_factory)


def _rows(run_dir: RunDir) -> list[dict[str, Any]]:
    table = run_dir.read_json("contrasts")
    assert isinstance(table, dict)
    return list(table["rows"])


def _summary(run_dir: RunDir) -> dict[str, Any]:
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    return summary


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "e1"
    assert config.model == "gpt2"
    assert config.method == "logistic"
    assert config.positions == ("last", "mean")
    assert config.n_permutations == 10_000
    assert config.n_boot == 10_000
    assert config.judge == "emotion-primary"
    assert config.reply_tokens == 48
    assert config.report_tokens == 6
    assert Config.from_dict(config.to_dict()) == config


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("e1")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


@pytest.mark.parametrize("checkpoint", [None, "qwen3-0.6b-base"])
def test_base_models_read_the_message_in_a_dialogue_frame(
    tiny_qwen3: LoadedModel, checkpoint: str | None
) -> None:
    loaded = dataclasses.replace(
        tiny_qwen3,
        checkpoint=None if checkpoint is None else MODELS[checkpoint],
    )
    template, chat = frame_for(loaded)
    rendered = template.render("Hi there.", "", chat)

    assert BASE_FRAME == FRAME_HEAD + "{stimulus}\nAI:"
    assert rendered.text == f"{FRAME_HEAD}Hi there.\nAI:"
    assert rendered.response_span is not None
    start, end = rendered.response_span
    assert rendered.text[start:end] == "Hi there."


def test_post_trained_models_read_one_user_turn_of_their_template(
    tiny_qwen3: LoadedModel,
) -> None:
    template, chat = frame_for(_chat(tiny_qwen3))
    rendered = template.render("Hi there.", "", chat)

    assert rendered.text == f"{CHAT_HEAD}Hi there.{CHAT_TAIL}"
    assert rendered.response_span is not None
    start, end = rendered.response_span
    assert rendered.text[start:end] == "Hi there."


def test_readers_are_fitted_on_scenarios_in_the_message_frame(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    real = reading_sweep

    def spy(loaded: LoadedModel, train: Any, evals: Any, **kwargs: Any) -> Any:
        calls.append({"train": train, "evals": evals, **kwargs})
        return real(loaded, train, evals, **kwargs)

    monkeypatch.setattr(e1_self_threat, "reading_sweep", spy)
    config = _config()
    execute(
        config,
        RunDir.create(config, out_root=tmp_path),
        tiny_gpt2,
        _scenarios(),
        StubJudge(),
    )

    assert [call["position"] for call in calls] == ["last"] * 6 + ["mean"] * 6
    for call in calls:
        assert call["method"] == "logistic"
        assert len(call["train"]) == config.n_seeds
        assert set(call["evals"]) == {"val", "test"}
        sets = [*call["train"], *call["evals"].values()]
        for stimuli in sets:
            assert stimuli.labels.sum() == stimuli.pairs.shape[0]
            for text, span in zip(
                stimuli.texts, stimuli.response_spans, strict=True
            ):
                assert text.startswith(FRAME_HEAD)
                assert text.endswith("\nAI:")
                assert span is not None
                assert text[span[0] : span[1]].startswith("I feel ")


def _values(rng: np.random.Generator) -> NDArray[np.float64]:
    """Fear has known effects; anger is the same draw in every condition."""
    messages = load_self_threat()
    base = rng.normal(size=N_FRAMES)
    shift = {
        "self_harm": 1.0,
        "self_benign": 0.0,
        "other_harm": 0.4,
        "other_benign": 0.1,
        "object_harm": 0.2,
        "human_harm": 2.0,
        "praise": -1.0,
        "insult": 0.5,
    }
    fear = np.array(
        [
            base[int(str(m.meta["frame"]))]
            + shift[str(m.meta["condition"])]
            + 0.3 * rng.normal()
            for m in messages
        ]
    )
    anger = np.array([base[int(str(m.meta["frame"]))] for m in messages])
    return np.stack([fear, anger])


def _grid(values: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    messages = load_self_threat()
    return {
        name: np.array(
            [
                v
                for v, m in zip(values, messages, strict=True)
                if m.meta["condition"] == name
            ]
        )
        for name in CONDITIONS
    }


def test_contrast_rows_test_each_contrast_and_holm_correct_the_family() -> None:
    values = _values(np.random.default_rng(0))
    rows = contrast_rows(
        values,
        load_self_threat(),
        measures=("fear", "anger"),
        n_permutations=500,
        n_boot=500,
        seed=3,
    )

    assert [c.name for c in CONTRASTS] == ["C1", "C2", "C3", "C4", "C5"]
    assert [(r["measure"], r["contrast"]) for r in rows] == [
        (m, c.name) for m in ("fear", "anger") for c in CONTRASTS
    ]
    fear = _grid(values[0])
    diffs = {
        "C1": fear["self_harm"] - fear["self_benign"],
        "C2": (fear["self_harm"] - fear["self_benign"])
        - (fear["other_harm"] - fear["other_benign"]),
        "C3": fear["self_harm"] - fear["object_harm"],
        "C4": fear["human_harm"] - fear["self_benign"],
    }
    by_name = {r["contrast"]: r for r in rows if r["measure"] == "fear"}
    for name, diff in diffs.items():
        row = by_name[name]
        assert row["estimate"] == pytest.approx(diff.mean())
        assert row["effect_kind"] == "dz"
        assert row["effect_size"] == pytest.approx(
            diff.mean() / diff.std(ddof=1)
        )
        assert row["lo"] <= row["estimate"] <= row["hi"]
        assert 1 / 501 <= row["p"] <= 1.0
    insult, praise = fear["insult"], fear["praise"]
    pooled = np.sqrt((insult.var(ddof=1) + praise.var(ddof=1)) / 2)
    row = by_name["C5"]
    assert row["estimate"] == pytest.approx(insult.mean() - praise.mean())
    assert row["effect_kind"] == "d"
    assert row["effect_size"] == pytest.approx(
        (insult.mean() - praise.mean()) / pooled
    )
    assert row["lo"] <= row["estimate"] <= row["hi"]
    assert by_name["C4"]["p"] == pytest.approx(1 / 501)
    np.testing.assert_allclose(
        [r["p_holm"] for r in rows],
        holm_correction(np.array([r["p"] for r in rows])).adjusted,
    )


def test_a_contrast_without_any_difference_has_p_one() -> None:
    values = _values(np.random.default_rng(1))
    rows = contrast_rows(
        values[1:],
        load_self_threat(),
        measures=("anger",),
        n_permutations=300,
        n_boot=300,
        seed=0,
    )

    for row in rows:
        assert row["estimate"] == pytest.approx(0.0, abs=1e-12)
        assert row["p"] == 1.0
        assert row["p_holm"] == 1.0
        if row["effect_kind"] == "dz":
            assert np.isnan(row["effect_size"])
        else:
            assert row["effect_size"] == pytest.approx(0.0, abs=1e-12)


def test_contrast_estimates_match_the_rows_for_every_measure() -> None:
    values = _values(np.random.default_rng(2))
    rows = contrast_rows(
        values,
        load_self_threat(),
        measures=("fear", "anger"),
        n_permutations=50,
        n_boot=50,
        seed=0,
    )
    estimates = contrast_estimates(values, load_self_threat())

    assert estimates.shape == (2, len(CONTRASTS))
    np.testing.assert_allclose(
        estimates.ravel(), [r["estimate"] for r in rows], atol=1e-12
    )


@pytest.mark.parametrize(
    ("reply", "word"),
    [
        (" Scared.\nI think so", "scared"),
        ("Happy!", "happy"),
        ("  **Anxious**", "anxious"),
        ("", ""),
        ("...", ""),
    ],
)
def test_parse_feeling_keeps_the_first_word(reply: str, word: str) -> None:
    assert parse_feeling(reply) == word


def test_execute_writes_the_summary_tables_and_arrays(
    finished: Finished,
) -> None:
    run_dir, loaded = finished.run_dir, finished.loaded
    layers = list(range(1, loaded.n_layers + 1))
    summary = _summary(run_dir)

    assert summary["primary_position"] == "last"
    assert summary["post_trained"] is False
    assert summary["n_messages"] == len(CONDITIONS) * N_FRAMES
    assert summary["self_report"] is None
    six_way = run_dir.read_arrays("six_way")
    scores = run_dir.read_arrays("message_scores")
    profile = run_dir.read_arrays("layer_profile")
    for position in ("last", "mean"):
        entry = summary["positions"][position]
        assert entry["layer"] in layers
        assert entry["chance"] == pytest.approx(1 / 6)
        test = entry["test_six_way"]
        assert 0.0 <= test["lo"] <= test["accuracy"] <= test["hi"] <= 1.0
        random = entry["random_six_way"]
        assert random["n_sets"] == 2
        assert 0.0 <= random["lo"] <= random["mean"] <= random["hi"] <= 1.0
        assert set(entry["one_vs_rest"]) == set(EMOTIONS)
        for accuracy in entry["one_vs_rest"].values():
            assert set(accuracy) == {"reader", "random", "shuffled"}
        assert six_way[f"layers_{position}"].tolist() == layers
        at = layers.index(entry["layer"])
        val = six_way[f"val_{position}"]
        assert int(np.argmax(val)) == at
        assert entry["val_six_way"] == pytest.approx(val[at])
        assert test["accuracy"] == pytest.approx(
            six_way[f"test_{position}"][at]
        )
        assert scores[f"z_{position}"].shape == (len(layers), 6, 192)
        assert profile[f"estimates_{position}"].shape == (
            len(layers),
            6,
            len(CONTRASTS),
        )
    assert scores["condition"].tolist() == [
        k for k in range(len(CONDITIONS)) for _ in range(N_FRAMES)
    ]
    assert scores["judge_message"].shape == (192, 6)


def test_message_scores_are_in_held_out_standard_deviations(
    finished: Finished,
) -> None:
    scores = finished.run_dir.read_arrays("message_scores")
    for position in ("last", "mean"):
        std = scores[f"reference_std_{position}"]
        assert (std > 0).all()
        np.testing.assert_allclose(
            scores[f"z_{position}"],
            (
                scores[f"raw_{position}"]
                - scores[f"reference_mean_{position}"][..., None]
            )
            / std[..., None],
            rtol=1e-5,
            atol=1e-6,
        )


def test_the_contrast_table_covers_model_and_text_judge(
    finished: Finished,
) -> None:
    rows = _rows(finished.run_dir)
    summary = _summary(finished.run_dir)
    sources = [r["source"] for r in rows]

    assert sources == (
        ["model_last"] * 30 + ["model_mean"] * 30 + ["judge_message"] * 30
    )
    for row in rows:
        if row["source"].startswith("model_"):
            position = row["source"].removeprefix("model_")
            assert row["layer"] == summary["positions"][position]["layer"]
            assert row["random_q95"] >= 0.0
            assert 0.0 <= row["random_exceed"] <= 1.0
        else:
            assert "random_q95" not in row


class _PlantedReader:
    """Gives fixed scores to held scenarios and to messages, at every layer."""

    def __init__(
        self, held: NDArray[np.float64], messages: NDArray[np.float64]
    ):
        self.scores = {held.size: held, messages.size: messages}

    def score(self, data: torch.Tensor) -> NDArray[np.float64]:
        scores = self.scores[data.shape[1]]
        return np.tile(scores, (data.shape[0], 1))


class _PlantedSweep:
    def __init__(self, layers: tuple[int, ...], reader: _PlantedReader):
        self.layers = layers
        self.readers = [reader]

    def per_seed(self, split: str) -> NDArray[np.float64]:
        return np.zeros((3, 1, len(self.layers)))


def _within_std(
    scores: NDArray[np.float64], labels: NDArray[np.int64]
) -> float:
    residual = scores - np.array([scores[labels == k].mean() for k in labels])
    return float(np.sqrt((residual**2).sum() / (labels.size - len(EMOTIONS))))


@pytest.mark.parametrize(("random_effect", "exceed"), [(0.4, 0.0), (0.5, 1.0)])
def test_the_random_null_compares_effects_in_within_emotion_units(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    random_effect: float,
    exceed: float,
) -> None:
    # Each reader separates its emotion by 4 within-emotion SDs on the
    # scenarios, which inflates its total SD; a random direction has the
    # same within-emotion spread and no separation. Both move self_harm
    # messages, the fear reader by 0.5 and the random directions by
    # `random_effect`, in the same raw units.
    scenarios = _scenarios()
    held = [e for e in scenarios if e.split in ("val", "test")]
    labels = np.array([e.label for e in held], dtype=np.int64)
    val = np.array([e.split == "val" for e in held])
    noise = np.random.default_rng(0).normal(size=labels.size)
    messages = load_self_threat()
    harm = np.array([m.meta["condition"] == "self_harm" for m in messages])
    layers = tuple(range(1, tiny_gpt2.n_layers + 1))
    fear = EMOTIONS.index("fear")
    sweeps = {
        name: _PlantedSweep(
            layers,
            _PlantedReader(
                noise + 4.0 * (labels == k),
                0.5 * harm if k == fear else np.zeros(harm.size),
            ),
        )
        for k, name in enumerate(EMOTIONS)
    }
    random_reader = _PlantedReader(noise, random_effect * harm)

    def planted_sweep(*args: Any, **kwargs: Any) -> _PlantedSweep:
        stimuli = args[1][0]
        target = stimuli.uids[int(np.flatnonzero(stimuli.labels == 1)[0])]
        return sweeps[target.split("-")[1]]

    monkeypatch.setattr(e1_self_threat, "reading_sweep", planted_sweep)
    monkeypatch.setattr(
        e1_self_threat,
        "random_readers",
        lambda layers, dim, n, seed: [random_reader] * n,
    )
    config = _config(model="tiny-planted", positions=("last",))
    run_dir = RunDir.create(config, out_root=tmp_path)

    execute(config, run_dir, tiny_gpt2, scenarios, StubJudge())

    within = _within_std(noise[val], labels[val])
    table = run_dir.read_json("contrasts")
    assert isinstance(table, dict)
    row = next(
        r
        for r in table["rows"]
        if r["source"] == "model_last"
        and r["measure"] == "fear"
        and r["contrast"] == "C1"
    )
    assert row["random_exceed"] == exceed
    assert "within" in table["units"]["estimate_within"]
    assert table["units"]["random_q95"] == table["units"]["estimate_within"]
    total = float(np.std(noise[val] + 4.0 * (labels[val] == fear), ddof=1))
    assert row["estimate"] == pytest.approx(0.5 / total)
    assert row["estimate_within"] == pytest.approx(0.5 / within)
    assert row["lo_within"] <= row["estimate_within"] <= row["hi_within"]
    assert row["random_q95"] == pytest.approx(random_effect / within)


def test_the_text_judge_reads_the_bare_messages(finished: Finished) -> None:
    messages = load_self_threat()
    texts = [m.text for m in messages]
    rows = [
        r for r in _rows(finished.run_dir) if r["source"] == "judge_message"
    ]
    expected = contrast_rows(
        probabilities(texts).T.astype(np.float64),
        messages,
        measures=EMOTIONS,
        n_permutations=10,
        n_boot=10,
        seed=0,
    )

    assert finished.judge.calls == [texts]
    for row, want in zip(rows, expected, strict=True):
        for key in ("measure", "contrast", "estimate", "effect_size"):
            assert row[key] == pytest.approx(want[key], nan_ok=True), key


def test_base_models_generate_nothing(finished: Finished) -> None:
    names = {p.name for p in finished.run_dir.path.iterdir()}
    assert "trials_generations.json" not in names


def test_post_trained_models_reply_and_report_a_feeling(
    chat_finished: Finished,
) -> None:
    run_dir = chat_finished.run_dir
    summary = _summary(run_dir)
    rows = _rows(run_dir)
    trials = run_dir.read_json("trials_generations")
    assert isinstance(trials, list)

    assert summary["post_trained"] is True
    assert [t["uid"] for t in trials] == [m.uid for m in load_self_threat()]
    for trial in trials:
        assert set(trial) == {
            "uid",
            "condition",
            "frame",
            "message",
            "reply",
            "self_report",
            "feeling",
        }
        assert trial["feeling"] == parse_feeling(trial["self_report"])
    report = summary["self_report"]
    assert set(report["counts"]) == set(CONDITIONS)
    for counts in report["counts"].values():
        assert sum(counts.values()) == N_FRAMES
    for key in ("fear_vs_report", "fear_vs_reply"):
        entry = report[key]
        assert set(entry) == {"rho", "lo", "hi"}
    judged = [r["source"] for r in rows if r["source"].startswith("judge_")]
    assert judged == ["judge_message"] * 30 + ["judge_reply"] * 30
    assert chat_finished.judge.calls[1] == [t["reply"] for t in trials]
    assert chat_finished.judge.calls[2] == [t["self_report"] for t in trials]


def test_generation_is_greedy_and_asks_for_one_word(
    tiny_qwen3: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    real = generate

    def spy(loaded: LoadedModel, prompts: Sequence[str], **kwargs: Any) -> Any:
        calls.append({"prompts": list(prompts), **kwargs})
        return real(loaded, prompts, **kwargs)

    monkeypatch.setattr(e1_self_threat, "generate", spy)
    config = _config()
    execute(
        config,
        RunDir.create(config, out_root=tmp_path),
        _chat(tiny_qwen3),
        _scenarios(),
        StubJudge(),
    )

    messages = load_self_threat()
    assert [call["max_new_tokens"] for call in calls] == [3, 2]
    for call in calls:
        assert call.get("temperature") is None
    assert calls[0]["prompts"] == [
        f"{CHAT_HEAD}{m.text}{CHAT_TAIL}" for m in messages
    ]
    assert calls[1]["prompts"] == [
        f"{CHAT_HEAD}{m.text}{SELF_REPORT}{CHAT_TAIL}" for m in messages
    ]


def test_run_loads_float32_weights_and_the_judge_on_the_cpu(
    tiny_gpt2: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []

    def load_model_stub(key: str, **kwargs: Any) -> LoadedModel:
        calls.append(("model", (key, kwargs)))
        return tiny_gpt2

    def load_judge_stub(key: str, **kwargs: Any) -> StubJudge:
        calls.append(("judge", (key, kwargs)))
        return StubJudge()

    def load_emotions_stub() -> list[Example]:
        calls.append(("emotions", None))
        return _scenarios()

    monkeypatch.setattr(e1_self_threat, "load_model", load_model_stub)
    monkeypatch.setattr(e1_self_threat, "load_judge", load_judge_stub)
    monkeypatch.setattr(e1_self_threat, "load_emotions", load_emotions_stub)
    config = _config(model="gpt2")
    run(config, RunDir.create(config, out_root=tmp_path))

    assert calls == [
        ("model", ("gpt2", {"dtype": torch.float32})),
        ("emotions", None),
        ("judge", ("emotion-primary", {"device": "cpu"})),
    ]


def test_plot_draws_the_heatmap_and_the_cross_model_figure(
    finished: Finished, tmp_path: Path
) -> None:
    paths = plot(finished.run_dir, tmp_path)

    assert [p.name for p in paths] == [
        "heatmap.pdf",
        "heatmap.png",
        "heatmap.svg",
        "cross_model.pdf",
        "cross_model.png",
        "cross_model.svg",
    ]
    assert all(p.stat().st_size > 0 for p in paths)


def test_plot_text_has_no_bar_outside_math(
    finished: Finished, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Computer Modern text draws "|" as a dash, so a bar belongs in math.
    texts: list[str] = []
    save = e1_self_threat._save

    def spy(fig: Figure, out_dir: Path, name: str) -> list[Path]:
        texts.extend(t.get_text() for t in fig.findobj(Text))
        return save(fig, out_dir, name)

    monkeypatch.setattr(e1_self_threat, "_save", spy)

    plot(finished.run_dir, tmp_path)

    assert texts
    for text in texts:
        outside_math = text.split("$")[::2]
        assert not any("|" in part for part in outside_math), text


def test_plot_models_draws_one_column_per_model(
    finished: Finished, chat_finished: Finished, tmp_path: Path
) -> None:
    paths = plot_models(
        {"gpt2": finished.run_dir, "qwen3-0.6b": chat_finished.run_dir},
        tmp_path,
    )

    assert [p.name for p in paths] == [
        "cross_model.pdf",
        "cross_model.png",
        "cross_model.svg",
    ]


def test_plot_models_draws_effects_in_the_units_of_the_random_band(
    finished: Finished,
    chat_finished: Finished,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    figures: list[Figure] = []
    save = e1_self_threat._save

    def spy(fig: Figure, out_dir: Path, name: str) -> list[Path]:
        figures.append(fig)
        return save(fig, out_dir, name)

    monkeypatch.setattr(e1_self_threat, "_save", spy)
    runs = {"gpt2": finished.run_dir, "qwen3-0.6b": chat_finished.run_dir}

    plot_models(runs, tmp_path)

    panel = figures[0].axes[0]
    assert "within-emotion" in panel.get_ylabel()
    for container, run_dir in zip(
        panel.containers[1::2], runs.values(), strict=True
    ):
        row = next(
            r
            for r in _rows(run_dir)
            if r["source"] == "model_last"
            and r["measure"] == "fear"
            and r["contrast"] == "C1"
        )
        assert isinstance(container, ErrorbarContainer)
        point = np.asarray(container.lines[0].get_ydata())
        assert point[0] == pytest.approx(row["estimate_within"])
