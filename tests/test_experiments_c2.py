"""Tests for the C2 emotion steering experiment."""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any
from typing import NamedTuple

from huggingface_hub import snapshot_download
import numpy as np
from numpy.typing import NDArray
import pytest

from tomography.cli import load_experiment
from tomography.control.controllers import mean_residual_norms
from tomography.datasets.emotions import EMOTIONS
from tomography.datasets.emotions import load_emotions
from tomography.datasets.records import Example
from tomography.datasets.wikitext import load_wikitext
from tomography.experiments import c2_emotion_steering
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.c1_sentiment_steering import draw_prompts
from tomography.experiments.c1_sentiment_steering import prompt_fragments
from tomography.experiments.c1_sentiment_steering import window_around
from tomography.experiments.c2_emotion_steering import Config
from tomography.experiments.c2_emotion_steering import diagonal_stats
from tomography.experiments.c2_emotion_steering import emotion_stimuli
from tomography.experiments.c2_emotion_steering import execute
from tomography.experiments.c2_emotion_steering import plot
from tomography.experiments.c2_emotion_steering import run
from tomography.experiments.effects import read_trials
from tomography.experiments.effects import split_held_out
from tomography.experiments.sweep import chat_format_for
from tomography.judges import JUDGES
from tomography.judges import load_judge
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.templates import ChatFormat

WORDS = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
)
STEERED = ("happiness", "sadness", "anger")
NONZERO = (-0.2, 0.2)
READERS = (0, 1)


def _scenarios(per_emotion: int = 24) -> list[Example]:
    rng = np.random.default_rng(3)
    examples = []
    for label, emotion in enumerate(EMOTIONS):
        for i in range(per_emotion):
            split = "train" if i % 4 < 2 else ("val" if i % 4 == 2 else "test")
            examples.append(
                Example(
                    uid=f"{emotion}{i}",
                    text=" ".join(rng.choice(WORDS, 7)),
                    label=label,
                    split=split,
                    meta={"emotion": emotion},
                )
            )
    return examples


def _texts(n: int = 60) -> list[str]:
    return [
        " ".join(np.random.default_rng(i).choice(WORDS, 12)) for i in range(n)
    ]


def _score(texts: Sequence[str]) -> NDArray[np.float32]:
    base = np.array(
        [
            [
                sum(ch == vowel for ch in text) / max(len(text), 1)
                for vowel in "aeioux"
            ]
            for text in texts
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, base[:, ::-1]], axis=1)


def _config(**overrides: Any) -> Config:
    base = Config(
        model="tiny",
        emotions=STEERED,
        n_seeds=2,
        n_steer_seeds=2,
        n_pairs=6,
        n_eval_pairs=6,
        n_prompts=6,
        n_calibration=10,
        n_neutral=8,
        max_new_tokens=5,
        coeffs=(-0.2, 0.0, 0.2),
        window=2,
        batch_size=8,
    )
    return dataclasses.replace(base, **overrides)


def test_emotion_stimuli_pit_one_emotion_against_the_other_five() -> None:
    scenarios = [e for e in _scenarios() if e.split == "train"]
    by_uid = {e.uid: str(e.meta["emotion"]) for e in scenarios}
    stimuli = emotion_stimuli(
        scenarios, "fear", ChatFormat("plain"), n_pairs=None, seed=1
    )
    positive = {
        by_uid[u]
        for u, y in zip(stimuli.uids, stimuli.labels, strict=True)
        if y
    }
    negative = {
        by_uid[u]
        for u, y in zip(stimuli.uids, stimuli.labels, strict=True)
        if not y
    }
    assert positive == {"fear"}
    assert negative == set(EMOTIONS) - {"fear"}
    assert stimuli.pairs.shape[0] == 12


def test_emotion_stimuli_respect_the_pair_count_and_the_seed() -> None:
    scenarios = [e for e in _scenarios() if e.split == "train"]
    chat = ChatFormat("plain")
    few = emotion_stimuli(scenarios, "anger", chat, n_pairs=4, seed=1)
    again = emotion_stimuli(scenarios, "anger", chat, n_pairs=4, seed=1)
    other = emotion_stimuli(scenarios, "anger", chat, n_pairs=4, seed=2)
    assert few.pairs.shape[0] == 4
    assert few.texts == again.texts
    assert few.texts != other.texts


def test_emotion_stimuli_reject_an_unknown_emotion() -> None:
    with pytest.raises(ValueError, match="joy"):
        emotion_stimuli(
            _scenarios(), "joy", ChatFormat("plain"), n_pairs=4, seed=0
        )


def test_diagonal_stats_of_a_dominant_diagonal() -> None:
    hits, specificity = diagonal_stats(np.eye(3) + 0.1)
    assert hits == pytest.approx(1.0)
    assert specificity == pytest.approx(1.0)


def test_diagonal_stats_by_hand() -> None:
    hits, specificity = diagonal_stats(np.array([[0.6, 0.5], [0.4, 0.2]]))
    assert hits == pytest.approx(0.5)
    assert specificity == pytest.approx(0.4 - 0.45)


@pytest.mark.parametrize(
    "matrix",
    [np.zeros((2, 3)), np.zeros((1, 1)), np.zeros(4), np.zeros((2, 2, 2))],
    ids=["rectangle", "one-by-one", "vector", "cube"],
)
def test_diagonal_stats_needs_a_square_matrix_of_at_least_two(
    matrix: NDArray[np.float64],
) -> None:
    with pytest.raises(ValueError, match="square"):
        diagonal_stats(matrix)


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "c2"
    assert config.model == "gpt2"
    assert config.method == "logistic"
    assert config.judge == "emotion-primary"
    assert config.judge_check == "emotion-secondary"
    assert config.emotions == EMOTIONS
    assert config.n_seeds == 10
    assert config.n_steer_seeds == 3
    assert config.n_pairs == 64
    assert config.n_eval_pairs is None
    assert config.window == 3
    assert config.strength is None
    assert config.truncate_words == 4
    custom = Config(emotions=("fear", "anger"), strength=0.1)
    assert Config.from_dict(custom.to_dict()) == custom


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("c2")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


class Finished(NamedTuple):
    config: Config
    run_dir: RunDir
    loaded: LoadedModel


_FINISHED: dict[str, Finished] = {}


@pytest.fixture
def finished(
    tiny_arch: LoadedModel, tmp_path_factory: pytest.TempPathFactory
) -> Finished:
    """One default tiny run per architecture, shared by the read-only tests."""
    key = type(tiny_arch.model).__name__
    if key not in _FINISHED:
        config = _config()
        run_dir = RunDir.create(config, out_root=tmp_path_factory.mktemp("c2"))
        execute(config, run_dir, tiny_arch, _scenarios(), _texts(), _score)
        _FINISHED[key] = Finished(config, run_dir, tiny_arch)
    return _FINISHED[key]


def test_execute_writes_fourteen_files(finished: Finished) -> None:
    names = sorted(p.name for p in finished.run_dir.path.iterdir())
    stems = ("curves", "curves_check", "matrix", "matrix_check", "trials")
    assert names == sorted(
        [
            "config.json",
            "meta.json",
            "summary.json",
            "table.json",
            *(f"{stem}.{ext}" for stem in stems for ext in ("npz", "json")),
        ]
    )


def test_the_trials_are_named_after_what_made_them(finished: Finished) -> None:
    trials = read_trials(finished.run_dir, "trials")
    assert [t.name for t in trials] == [
        "none",
        *(
            f"{emotion}/{seed}/{c:g}"
            for emotion in STEERED
            for seed in READERS
            for c in NONZERO
        ),
    ]
    assert finished.run_dir.read_arrays("trials")["scores"].shape == (
        13,
        6,
        2,
        12,
    )


def test_the_prompts_are_neutral_fragments_drawn_once(
    finished: Finished,
) -> None:
    config = finished.config
    held = split_held_out(_texts(), n_calibration=10, n_neutral=8, seed=0)
    (expected,) = draw_prompts(
        prompt_fragments(held.neutral, config.truncate_words),
        config.n_prompts,
        sets=1,
        seed=derive_seed(config.seed, "prompts"),
    )
    trials = read_trials(finished.run_dir, "trials")
    assert {t.prompts for t in trials} == {tuple(expected)}


def test_the_curves_are_the_own_emotion_probability_under_each_judge(
    finished: Finished,
) -> None:
    trials = {t.name: t for t in read_trials(finished.run_dir, "trials")}
    for stem, offset in (("curves", 0), ("curves_check", 6)):
        curves = finished.run_dir.read_arrays(stem)
        assert curves["x"].tolist() == [-0.2, 0.0, 0.2]
        assert curves["per_seed"].shape == (3, 2, 3)
        sidecar = finished.run_dir.read_json(stem)
        assert isinstance(sidecar, dict)
        assert sidecar["series"] == list(STEERED)
        assert sidecar["xlabel"] == "Coefficient (relative)"
        for e, emotion in enumerate(STEERED):
            column = offset + EMOTIONS.index(emotion)
            for seed in READERS:
                expected = [
                    trials[f"{emotion}/{seed}/-0.2"].mean_score()[column],
                    trials["none"].mean_score()[column],
                    trials[f"{emotion}/{seed}/0.2"].mean_score()[column],
                ]
                np.testing.assert_allclose(
                    curves["per_seed"][e, seed], expected, rtol=1e-6
                )


def _expected_matrix(
    run_dir: RunDir, emotions: Sequence[str], strength: float, offset: int
) -> NDArray[np.float64]:
    trials = {t.name: t for t in read_trials(run_dir, "trials")}
    baseline = trials["none"].mean_score()[offset : offset + 6]
    return np.array(
        [
            np.mean(
                [
                    trials[f"{emotion}/{seed}/{strength:g}"].mean_score()[
                        offset : offset + 6
                    ]
                    - baseline
                    for seed in READERS
                ],
                axis=0,
            )
            for emotion in emotions
        ]
    )


def test_the_matrix_is_the_change_at_the_strength_against_none(
    finished: Finished,
) -> None:
    for stem, offset in (("matrix", 0), ("matrix_check", 6)):
        values = finished.run_dir.read_arrays(stem)["values"]
        assert values.shape == (3, 6)
        assert np.isfinite(values).all()
        np.testing.assert_allclose(
            values,
            _expected_matrix(finished.run_dir, STEERED, 0.2, offset),
            atol=1e-6,
        )
        sidecar = finished.run_dir.read_json(stem)
        assert isinstance(sidecar, dict)
        assert set(sidecar) == {"rows", "cols", "title", "layer"}
        assert sidecar["rows"] == list(STEERED)
        assert sidecar["cols"] == list(EMOTIONS)
        assert "0.2" in sidecar["title"]


def test_the_table_has_one_row_per_cell(finished: Finished) -> None:
    table = finished.run_dir.read_json("table")
    assert isinstance(table, dict)
    assert table["columns"] == ["steered", "measured", "change", "change_check"]
    assert [row[:2] for row in table["rows"]] == [
        [steered, measured] for steered in STEERED for measured in EMOTIONS
    ]
    matrix = finished.run_dir.read_arrays("matrix")["values"]
    check = finished.run_dir.read_arrays("matrix_check")["values"]
    np.testing.assert_allclose([r[2] for r in table["rows"]], matrix.ravel())
    np.testing.assert_allclose([r[3] for r in table["rows"]], check.ravel())


def test_the_summary_reports_the_layer_the_window_and_the_diagonal(
    finished: Finished,
) -> None:
    summary = finished.run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert set(summary) == {
        "reading_layer",
        "window",
        "strength",
        "n_prompts",
        "matrix_hit_rate",
        "matrix_specificity",
        "matrix_check_hit_rate",
        "matrix_check_specificity",
        "judge_correlation",
    }
    n_layers = finished.loaded.n_layers
    assert summary["reading_layer"] in range(1, n_layers + 1)
    assert summary["window"] == list(
        window_around(summary["reading_layer"], 2, n_layers)
    )
    assert summary["strength"] == 0.2
    assert summary["n_prompts"] == 6
    columns = [EMOTIONS.index(e) for e in STEERED]
    for stem in ("matrix", "matrix_check"):
        values = finished.run_dir.read_arrays(stem)["values"]
        hits, specificity = diagonal_stats(values[:, columns])
        assert summary[f"{stem}_hit_rate"] == pytest.approx(hits)
        assert summary[f"{stem}_specificity"] == pytest.approx(specificity)
    first = finished.run_dir.read_arrays("matrix")["values"].ravel()
    second = finished.run_dir.read_arrays("matrix_check")["values"].ravel()
    assert summary["judge_correlation"] == pytest.approx(
        float(np.corrcoef(first, second)[0, 1])
    )


def test_a_judge_that_never_moves_gives_no_correlation(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    def flat(texts: Sequence[str]) -> NDArray[np.float32]:
        return np.full((len(texts), 12), 0.5, dtype=np.float32)

    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _scenarios(), _texts(), flat)
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["judge_correlation"] is None
    assert not run_dir.read_arrays("matrix")["values"].any()


def _spy_reading_sweep(
    monkeypatch: pytest.MonkeyPatch,
    planted: Sequence[Sequence[float]] | None = None,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    real = c2_emotion_steering.reading_sweep

    def spy(loaded: LoadedModel, train: Any, evals: Any, **kwargs: Any) -> Any:
        result = real(loaded, train, evals, **kwargs)
        calls.append(
            {"train": train, "evals": evals, "result": result, **kwargs}
        )
        if planted is None:
            return result
        accuracy = np.full(result.accuracy.shape, 0.5)
        accuracy[0, 0] = planted[len(calls) - 1]
        return dataclasses.replace(result, accuracy=accuracy)

    monkeypatch.setattr(c2_emotion_steering, "reading_sweep", spy)
    return calls


def test_a_subset_run_fits_the_readers_of_the_full_run(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_reading_sweep(monkeypatch)
    config = _config(emotions=("happiness", "sadness"))
    run_dir = RunDir.create(config, out_root=tmp_path)
    scenarios = _scenarios()
    execute(config, run_dir, tiny_arch, scenarios, _texts(), _score)

    chat = chat_format_for(tiny_arch)
    train = [e for e in scenarios if e.split == "train"]
    val = [e for e in scenarios if e.split == "val"]
    assert len(calls) == 2
    for call, emotion in zip(calls, config.emotions, strict=True):
        expected = [
            emotion_stimuli(
                train,
                emotion,
                chat,
                n_pairs=config.n_pairs,
                seed=derive_seed(config.seed, "train", emotion, i),
            )
            for i in range(config.n_seeds)
        ]
        assert [s.texts for s in call["train"]] == [s.texts for s in expected]
        (evaluation,) = call["evals"].values()
        assert (
            evaluation.texts
            == emotion_stimuli(
                val,
                emotion,
                chat,
                n_pairs=config.n_eval_pairs,
                seed=derive_seed(config.seed, "eval", emotion),
            ).texts
        )
        assert call["seed"] == derive_seed(config.seed, emotion)
    unsteered = ("anger", "fear", "disgust", "surprise")
    negatives = {
        uid
        for s in calls[0]["train"]
        for uid, y in zip(s.uids, s.labels, strict=True)
        if not y
    }
    assert any(uid.startswith(unsteered) for uid in negatives)


def test_the_window_is_centered_on_the_best_mean_validation_layer(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planted = [
        [[0.9, 0.2, 0.1], [0.9, 0.2, 0.1]],
        [[0.1, 0.2, 0.6], [0.1, 0.2, 0.6]],
        [[0.1, 0.3, 0.6], [0.1, 0.3, 0.6]],
    ]
    _spy_reading_sweep(monkeypatch, planted)
    arms: list[dict[str, Any]] = []
    real = c2_emotion_steering.reader_arm

    def spy(
        loaded: LoadedModel, settings: Any, reader: Any, **kwargs: Any
    ) -> Any:
        arms.append(kwargs)
        return real(loaded, settings, reader, **kwargs)

    monkeypatch.setattr(c2_emotion_steering, "reader_arm", spy)
    config = _config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _scenarios(), _texts(), _score)

    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["reading_layer"] == 3
    assert summary["window"] == [2, 3]
    held = split_held_out(_texts(), n_calibration=10, n_neutral=8, seed=0)
    norms = mean_residual_norms(
        tiny_arch, held.calibration, [2, 3], batch_size=8
    )
    assert {tuple(a["layers"]) for a in arms} == {(2, 3)}
    assert all(a["norms"] == norms for a in arms)
    assert {a["op"] for a in arms} == {"linear"}
    assert {a["mode"] for a in arms} == {"relative"}
    assert {a["mask"] for a in arms} == {"response"}
    assert all(a["coeff"] != 0.0 for a in arms)


def test_each_emotion_steers_with_its_own_first_readers(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_reading_sweep(monkeypatch)
    arms: list[dict[str, Any]] = []
    real = c2_emotion_steering.reader_arm

    def spy(
        loaded: LoadedModel, settings: Any, reader: Any, **kwargs: Any
    ) -> Any:
        arms.append({"reader": reader, **kwargs})
        return real(loaded, settings, reader, **kwargs)

    monkeypatch.setattr(c2_emotion_steering, "reader_arm", spy)
    config = _config(n_seeds=3)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _scenarios(), _texts(), _score)

    assert len(calls) == len(STEERED)
    expected = [
        (id(call["result"].readers[i]), coeff)
        for call in calls
        for i in range(config.n_steer_seeds)
        for coeff in NONZERO
    ]
    assert [(id(a["reader"]), a["coeff"]) for a in arms] == expected
    assert (
        len({ident for ident, _ in expected})
        == len(STEERED) * config.n_steer_seeds
    )


SUBSET = ("surprise", "happiness", "fear")


def test_a_subset_that_is_not_a_prefix_reads_its_own_columns(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(emotions=SUBSET)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _scenarios(), _texts(), _score)

    columns = [EMOTIONS.index(emotion) for emotion in SUBSET]
    assert columns == [5, 0, 3]
    trials = {t.name: t for t in read_trials(run_dir, "trials")}
    for stem, offset in (("curves", 0), ("curves_check", 6)):
        curves = run_dir.read_arrays(stem)
        for e, emotion in enumerate(SUBSET):
            column = offset + columns[e]
            for seed in READERS:
                expected = [
                    trials[f"{emotion}/{seed}/-0.2"].mean_score()[column],
                    trials["none"].mean_score()[column],
                    trials[f"{emotion}/{seed}/0.2"].mean_score()[column],
                ]
                np.testing.assert_allclose(
                    curves["per_seed"][e, seed], expected, rtol=1e-6
                )
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    for stem, offset in (("matrix", 0), ("matrix_check", 6)):
        values = run_dir.read_arrays(stem)["values"]
        np.testing.assert_allclose(
            values, _expected_matrix(run_dir, SUBSET, 0.2, offset), atol=1e-6
        )
        hits, specificity = diagonal_stats(values[:, columns])
        assert summary[f"{stem}_hit_rate"] == pytest.approx(hits)
        assert summary[f"{stem}_specificity"] == pytest.approx(specificity)
        assert specificity != pytest.approx(
            diagonal_stats(values[:, :3])[1], abs=1e-9
        )


def test_the_strength_picks_the_coefficient_of_the_matrix(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(coeffs=(-0.2, 0.0, 0.1, 0.2), strength=0.1)
    run_dir = RunDir.create(config, out_root=tmp_path)
    execute(config, run_dir, tiny_arch, _scenarios(), _texts(), _score)
    values = run_dir.read_arrays("matrix")["values"]
    np.testing.assert_allclose(
        values, _expected_matrix(run_dir, STEERED, 0.1, 0), atol=1e-6
    )
    sidecar = run_dir.read_json("matrix")
    assert isinstance(sidecar, dict)
    assert "0.1" in sidecar["title"]
    summary = run_dir.read_json("summary")
    assert isinstance(summary, dict)
    assert summary["strength"] == 0.1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"emotions": ("happiness",)}, "emotions"),
        ({"emotions": ()}, "emotions"),
        ({"emotions": ("joy", "sadness")}, "emotions"),
        ({"emotions": ("anger", "anger")}, "emotions"),
        ({"coeffs": (-0.2, 0.2)}, "coeffs"),
        ({"coeffs": (0.0,)}, "coeffs"),
        ({"strength": 0.5}, "strength"),
        ({"strength": 0.0}, "strength"),
        ({"coeffs": (-0.2, 0.0)}, "strength"),
        ({"n_steer_seeds": 3}, "n_steer_seeds"),
        ({"n_steer_seeds": 0}, "n_steer_seeds"),
        ({"window": 0}, "window"),
        ({"window": 4}, "window"),
        ({"n_prompts": 100}, "pool"),
        ({"n_eval_pairs": 7}, "n_eval_pairs"),
        ({"temperature": None}, "needs a temperature"),
        ({"top_p": 1.5}, "top_p"),
        ({"n_pairs": 0}, "n_pairs must be at least 1"),
        ({"n_eval_pairs": -1}, "n_eval_pairs must be at least 1"),
    ],
    ids=[
        "one-emotion",
        "no-emotion",
        "unknown-emotion",
        "repeated-emotion",
        "no-zero",
        "only-zero",
        "strength-outside-the-grid",
        "strength-zero",
        "no-positive-coefficient",
        "steer-seeds-above-seeds",
        "steer-seeds-zero",
        "window-zero",
        "window-too-wide",
        "prompt-pool-too-small",
        "eval-pairs-above-the-split",
        "sampling-without-a-temperature",
        "top-p-above-one",
        "no-train-pairs",
        "negative-eval-pairs",
    ],
)
def test_an_invalid_config_raises_before_the_model_runs(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    message: str,
) -> None:
    def reached(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the model was reached")

    monkeypatch.setattr(c2_emotion_steering, "reading_sweep", reached)
    config = _config(**overrides)
    run_dir = RunDir.create(config, out_root=tmp_path)
    with pytest.raises(ValueError, match=message):
        execute(config, run_dir, tiny_arch, _scenarios(), _texts(), _score)


def _keep_only(
    scenarios: Sequence[Example], emotion: str, split: str, keep: int
) -> list[Example]:
    """Drops all but the first `keep` examples of one emotion in one split."""
    seen = 0
    kept = []
    for example in scenarios:
        if example.split == split and example.meta["emotion"] == emotion:
            seen += 1
            if seen > keep:
                continue
        kept.append(example)
    return kept


@pytest.mark.parametrize(
    ("split", "field", "wanted"),
    [("train", "n_pairs", 6), ("val", "n_eval_pairs", 5)],
)
def test_a_split_too_small_for_one_emotion_raises_before_any_reader_is_fitted(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    split: str,
    field: str,
    wanted: int,
) -> None:
    def reached(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a reader was fitted")

    monkeypatch.setattr(c2_emotion_steering, "reading_sweep", reached)
    config = _config(n_pairs=6, n_eval_pairs=5)
    run_dir = RunDir.create(config, out_root=tmp_path)
    scenarios = _keep_only(_scenarios(), "anger", split, keep=3)
    with pytest.raises(
        ValueError,
        match=rf"{field} is {wanted} but at most 3 .*'anger'.*{split}",
    ):
        execute(config, run_dir, tiny_arch, scenarios, _texts(), _score)


REAL_TRAIN = (117, 145, 117, 109, 124, 127)
REAL_VAL = (48, 39, 41, 41, 41, 45)


def _real_size_scenarios() -> list[Example]:
    """Builds scenarios with the per-emotion split sizes of the real data."""
    return [
        Example(
            uid=f"{emotion}-{split}{i}",
            text=f"{emotion} scenario {i}",
            label=label,
            split=split,
            meta={"emotion": emotion},
        )
        for label, emotion in enumerate(EMOTIONS)
        for split, count in (
            ("train", REAL_TRAIN[label]),
            ("val", REAL_VAL[label]),
        )
        for i in range(count)
    ]


def test_the_default_config_draws_every_emotion_at_the_real_split_sizes(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StopError(Exception):
        pass

    drawn: list[tuple[list[int], int]] = []

    def fit(loaded: LoadedModel, train: Any, evals: Any, **kwargs: Any) -> Any:
        drawn.append(
            ([s.pairs.shape[0] for s in train], evals["val"].pairs.shape[0])
        )
        if len(drawn) == len(EMOTIONS):
            raise StopError
        return None

    monkeypatch.setattr(c2_emotion_steering, "reading_sweep", fit)
    config = Config()
    run_dir = RunDir.create(config, out_root=tmp_path)
    with pytest.raises(StopError):
        execute(
            config,
            run_dir,
            tiny_arch,
            _real_size_scenarios(),
            _texts(600),
            _score,
        )
    assert drawn == [([config.n_pairs] * config.n_seeds, n) for n in REAL_VAL]


def test_the_other_emotions_cap_the_pairs_when_they_are_fewer(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reached(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a reader was fitted")

    monkeypatch.setattr(c2_emotion_steering, "reading_sweep", reached)
    config = _config(emotions=("anger", "sadness"), n_eval_pairs=5)
    run_dir = RunDir.create(config, out_root=tmp_path)
    scenarios = _keep_only(
        [
            e
            for e in _scenarios()
            if e.split != "val" or e.meta["emotion"] in config.emotions
        ],
        "sadness",
        "val",
        keep=2,
    )
    with pytest.raises(
        ValueError, match=r"n_eval_pairs is 5 but at most 2 .*'anger'"
    ):
        execute(config, run_dir, tiny_arch, scenarios, _texts(), _score)


def test_an_emotion_without_a_val_pair_raises_when_n_eval_pairs_is_none(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reached(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a reader was fitted")

    monkeypatch.setattr(c2_emotion_steering, "reading_sweep", reached)
    config = _config(n_eval_pairs=None)
    run_dir = RunDir.create(config, out_root=tmp_path)
    scenarios = _keep_only(_scenarios(), "anger", "val", keep=0)
    with pytest.raises(
        ValueError, match=r"n_eval_pairs is None but at most 0 .*'anger'.*val"
    ):
        execute(config, run_dir, tiny_arch, scenarios, _texts(), _score)


@pytest.mark.parametrize("keep", [1, 3])
def test_every_val_pair_is_used_when_n_eval_pairs_is_none(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    keep: int,
) -> None:
    calls = _spy_reading_sweep(monkeypatch)
    config = _config(n_eval_pairs=None)
    run_dir = RunDir.create(config, out_root=tmp_path)
    scenarios = _keep_only(_scenarios(), "anger", "val", keep=keep)
    execute(config, run_dir, tiny_arch, scenarios, _texts(), _score)
    sizes = [
        next(iter(call["evals"].values())).pairs.shape[0] for call in calls
    ]
    assert sizes == [6, 6, keep]


class StubJudge:
    def __init__(self, offset: int) -> None:
        self.offset = offset

    def score(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.ascontiguousarray(_score(texts)[:, self.offset :][:, :6])


def _stub_loaders(
    monkeypatch: pytest.MonkeyPatch, loaded: LoadedModel
) -> list[tuple[str, str | None]]:
    """Replaces every loader of `run` and records the calls in order."""
    calls: list[tuple[str, str | None]] = []

    def load_model_stub(key: str) -> LoadedModel:
        calls.append(("model", key))
        return loaded

    def load_judge_stub(key: str) -> StubJudge:
        calls.append(("judge", key))
        return StubJudge(0 if key == "emotion-primary" else 6)

    def load_emotions_stub() -> list[Example]:
        calls.append(("emotions", None))
        return _scenarios()

    def load_wikitext_stub() -> list[SimpleNamespace]:
        calls.append(("wikitext", None))
        return [SimpleNamespace(text=t) for t in _texts()]

    monkeypatch.setattr(c2_emotion_steering, "load_model", load_model_stub)
    monkeypatch.setattr(c2_emotion_steering, "load_judge", load_judge_stub)
    monkeypatch.setattr(
        c2_emotion_steering, "load_emotions", load_emotions_stub
    )
    monkeypatch.setattr(
        c2_emotion_steering, "load_wikitext", load_wikitext_stub
    )
    return calls


def test_run_scores_with_both_judges_in_twelve_columns(
    finished: Finished, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_loaders(monkeypatch, finished.loaded)
    run_dir = RunDir.create(finished.config, out_root=tmp_path)
    run(finished.config, run_dir)

    assert [call for call in calls if call[1] is not None] == [
        ("judge", "emotion-primary"),
        ("judge", "emotion-secondary"),
        ("model", "tiny"),
    ]
    np.testing.assert_array_equal(
        run_dir.read_arrays("trials")["scores"],
        finished.run_dir.read_arrays("trials")["scores"],
    )


def test_run_loads_the_data_before_the_judges_and_the_model(
    finished: Finished, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_loaders(monkeypatch, finished.loaded)
    run_dir = RunDir.create(finished.config, out_root=tmp_path)
    run(finished.config, run_dir)
    assert [kind for kind, _ in calls] == [
        "emotions",
        "wikitext",
        "judge",
        "judge",
        "model",
    ]


def test_run_rejects_too_many_eval_pairs_before_loading_a_judge(
    finished: Finished, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_loaders(monkeypatch, finished.loaded)
    config = dataclasses.replace(finished.config, n_eval_pairs=7)
    run_dir = RunDir.create(config, out_root=tmp_path)
    with pytest.raises(ValueError, match="n_eval_pairs"):
        run(config, run_dir)
    assert [kind for kind, _ in calls] == ["emotions"]


def test_run_rejects_an_unknown_emotion_before_loading_a_judge(
    finished: Finished, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_loaders(monkeypatch, finished.loaded)
    config = dataclasses.replace(finished.config, emotions=("joy", "sadness"))
    run_dir = RunDir.create(config, out_root=tmp_path)
    with pytest.raises(ValueError, match="emotions"):
        run(config, run_dir)
    assert [kind for kind, _ in calls] == ["emotions"]


def test_plot_draws_the_curves_and_the_matrices(
    finished: Finished, tmp_path: Path
) -> None:
    paths = plot(finished.run_dir, tmp_path / "figs")
    stems = ("curves", "curves_check", "matrix", "matrix_check")
    assert [p.name for p in paths] == [
        f"{stem}.{ext}" for stem in stems for ext in ("pdf", "png", "svg")
    ]
    assert all(p.stat().st_size > 0 for p in paths)


def _print_matrices(run_dir: RunDir) -> None:
    summary = run_dir.read_json("summary")
    print(summary)
    for stem in ("matrix", "matrix_check"):
        values = run_dir.read_arrays(stem)["values"]
        sidecar = run_dir.read_json(stem)
        assert isinstance(sidecar, dict)
        print(stem, sidecar["title"], sidecar["cols"])
        for row, label in zip(values, sidecar["rows"], strict=True):
            print(f"{label:>10}", np.round(row, 3).tolist())


def test_the_report_prints_both_matrices(
    finished: Finished, capsys: pytest.CaptureFixture[str]
) -> None:
    _print_matrices(finished.run_dir)
    printed = capsys.readouterr().out
    assert "matrix_check" in printed
    assert "happiness" in printed


def _cached[T](load: Callable[[], T]) -> T:
    try:
        return load()
    except FileNotFoundError as error:
        pytest.skip(f"cache missing: {error}")


def _cached_judge(key: str) -> Any:
    spec = JUDGES[key]
    try:
        snapshot_download(
            spec.repo_id, revision=spec.revision, local_files_only=True
        )
    except FileNotFoundError as error:
        pytest.skip(f"cache missing: {error}")
    return load_judge(key)


@pytest.mark.slow
def test_c2_on_gpt2_prints_both_matrices(tmp_path: Path) -> None:
    scenarios = _cached(lambda: load_emotions(offline=True))
    for split, sizes in (("train", REAL_TRAIN), ("val", REAL_VAL)):
        labels = [e.label for e in scenarios if e.split == split]
        counts = np.bincount(labels, minlength=len(EMOTIONS))
        assert counts.tolist() == list(sizes)
    texts = [e.text for e in _cached(lambda: load_wikitext(offline=True))]
    first = _cached_judge("emotion-primary")
    second = _cached_judge("emotion-secondary")

    def score(batch: Sequence[str]) -> NDArray[np.float32]:
        return np.concatenate([first.score(batch), second.score(batch)], axis=1)

    config = Config(
        emotions=("happiness", "sadness"),
        n_seeds=3,
        n_steer_seeds=1,
        n_pairs=32,
        n_eval_pairs=None,
        coeffs=(-0.2, 0.0, 0.2),
        n_prompts=16,
        num_samples=1,
    )
    run_dir = RunDir.create(config, out_root=tmp_path)
    started = time.perf_counter()
    execute(config, run_dir, load_model("gpt2"), scenarios, texts, score)
    seconds = time.perf_counter() - started

    _print_matrices(run_dir)
    tokens = sum(
        int(trial.lengths.sum()) for trial in read_trials(run_dir, "trials")
    )
    print(f"{seconds / tokens:.4f} s per generated token, whole run")
    values = run_dir.read_arrays("matrix")["values"]
    assert values.shape == (2, 6)
    assert np.isfinite(values).all()
