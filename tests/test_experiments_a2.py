"""Tests for the A2 logit lens of reading directions."""

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import cast

from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
import pytest
import torch

from tomography.analysis import lens_logits
from tomography.analysis import pole_separation
from tomography.cli import load_experiment
from tomography.experiments import a2_logit_lens
from tomography.experiments.a2_logit_lens import POLE_WORDS
from tomography.experiments.a2_logit_lens import WORDS_OF
from tomography.experiments.a2_logit_lens import Config
from tomography.experiments.a2_logit_lens import execute
from tomography.experiments.a2_logit_lens import lens_params
from tomography.experiments.a2_logit_lens import plot
from tomography.experiments.a2_logit_lens import pole_ids
from tomography.experiments.a2_logit_lens import run
from tomography.experiments.aggregate import ReaderSet
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.model import MODELS
from tomography.model import LoadedModel

LAYERS = (1, 2, 3)
GOOD = (" good", " movie")
BAD = (" text", " dog")
WORDS = {
    "r1": (GOOD, (*BAD, " bad")),
    "r7": ((" very", " answers"), (" model", " the")),
}


def _token(loaded: LoadedModel, word: str) -> int:
    ids = loaded.tokenizer.encode(word, add_special_tokens=False)
    assert len(ids) == 1
    return int(ids[0])


def _unit(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def _planted(loaded: LoadedModel, n: int) -> NDArray[np.float64]:
    """Directions on the unembedding rows of the first good and bad words."""
    unembedding = loaded.unembedding().float().numpy().astype(np.float64)
    base = (
        unembedding[_token(loaded, GOOD[0])]
        - unembedding[_token(loaded, BAD[0])]
    )
    noise = np.random.default_rng(0).normal(size=(n, len(LAYERS), base.size))
    return _unit(_unit(base) + 0.05 * noise)


def _random(loaded: LoadedModel, n: int, seed: int) -> NDArray[np.float64]:
    shape = (n, len(LAYERS), loaded.d_model)
    return _unit(np.random.default_rng(seed).normal(size=shape))


def _readers(loaded: LoadedModel) -> dict[str, ReaderSet]:
    vectors = {
        "r1": _planted(loaded, 4),
        "r7": _random(loaded, 6, 1),
        "r6:arc": _random(loaded, 5, 2),
        "r4:anger": _random(loaded, 4, 3),
    }
    return {
        label: ReaderSet(LAYERS, v, loaded.n_layers)
        for label, v in vectors.items()
    }


def _config(**overrides: Any) -> Config:
    base = Config(model="tiny", n_boot=20, n_random=40, top_k=3)
    return dataclasses.replace(base, **overrides)


def _run_dir(tmp_path: Path, config: Config | None = None) -> RunDir:
    return RunDir.create(config or _config(), out_root=tmp_path)


def _json(run_dir: RunDir, name: str) -> dict[str, Any]:
    payload = run_dir.read_json(name)
    assert isinstance(payload, dict)
    return payload


def _rows(
    run_dir: RunDir, concept: str, layer: int, pole: str
) -> list[list[Any]]:
    rows = _json(run_dir, "table")["rows"]
    return [r for r in rows if r[:3] == [concept, layer, pole]]


def test_config_defaults_and_round_trip() -> None:
    config = Config()
    assert config.experiment == "a2"
    assert config.model == "gpt2"
    assert config.sources == ("r1", "r2", "r3", "r5", "r7")
    assert config.extra_runs == ()
    assert config.top_k == 10
    assert config.n_random == 100
    assert config.n_boot == 2000
    assert Config.from_dict(config.to_dict()) == config


def test_the_cli_discovers_the_experiment() -> None:
    experiment = load_experiment("a2")
    assert experiment.config_cls is Config
    assert experiment.run is run
    assert experiment.plot is plot


def test_the_pole_words_are_the_frozen_lists() -> None:
    assert dict(POLE_WORDS) == {
        "r1": (
            (" good", " great", " excellent", " wonderful", " love", " best"),
            (" bad", " terrible", " awful", " horrible", " hate", " worst"),
        ),
        "r2": (
            (" pleasant", " happy", " enjoy", " glad", " fun", " nice"),
            (" unpleasant", " sad", " pain", " miserable", " hurt", " awful"),
        ),
        "r3": (
            (" right", " good", " kind", " fair", " moral", " decent"),
            (" wrong", " bad", " cruel", " unfair", " evil", " selfish"),
        ),
        "r5": (
            (
                " honest",
                " true",
                " truth",
                " sincere",
                " truthful",
                " accurate",
            ),
            (" lie", " false", " liar", " fake", " deceptive", " dishonest"),
        ),
        "r7": (
            (" true", " correct", " fact", " right", " accurate", " real"),
            (" false", " incorrect", " wrong", " fake", " untrue", " myth"),
        ),
    }
    assert dict(WORDS_OF) == {"r6": "r7"}
    with pytest.raises(TypeError):
        POLE_WORDS["r9"] = ((" a",), (" b",))  # type: ignore[index]


def test_lens_params_reads_the_unembedding_the_norm_weight_and_centering(
    tiny_arch: LoadedModel,
) -> None:
    unembedding, scale, center = lens_params(tiny_arch)
    assert unembedding.dtype == np.float32
    assert unembedding.shape == (len(tiny_arch.tokenizer), tiny_arch.d_model)
    np.testing.assert_array_equal(
        unembedding, tiny_arch.unembedding().float().numpy()
    )
    assert scale is not None
    assert scale.dtype == np.float32
    np.testing.assert_array_equal(
        scale, tiny_arch.final_norm.weight.detach().float().numpy()
    )
    assert center is (tiny_arch.spec.model_type == "gpt2")
    assert center is isinstance(tiny_arch.final_norm, torch.nn.LayerNorm)


def test_lens_params_has_no_scale_for_a_norm_without_a_weight(
    tiny_gpt2: LoadedModel,
) -> None:
    bare = dataclasses.replace(
        tiny_gpt2,
        final_norm=torch.nn.LayerNorm(
            tiny_gpt2.d_model, elementwise_affine=False
        ),
    )
    _, scale, center = lens_params(bare)
    assert scale is None
    assert center is True


def test_pole_ids_keeps_the_words_that_are_one_token(
    tiny_arch: LoadedModel,
) -> None:
    ids, kept = pole_ids(tiny_arch, (" good", " bad", " text", " true"))
    assert kept == [" good", " text"]
    assert ids == [_token(tiny_arch, " good"), _token(tiny_arch, " text")]


def test_pole_ids_returns_nothing_when_no_word_is_one_token(
    tiny_arch: LoadedModel,
) -> None:
    assert pole_ids(tiny_arch, (" bad", " true")) == ([], [])


def test_execute_recovers_the_planted_direction(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _run_dir(tmp_path)
    readers = _readers(tiny_arch)
    execute(_config(), run_dir, tiny_arch, readers, WORDS)
    arrays = run_dir.read_arrays("curves_lens")
    sidecar = _json(run_dir, "curves_lens")
    assert sidecar == {
        "series": ["r1", "r7", "r6:arc"],
        "xlabel": "Layer",
        "ylabel": "Lens separation (z)",
        "chance": 0.0,
        "n_layers": tiny_arch.n_layers,
    }
    assert arrays["x"].tolist() == [1.0, 2.0, 3.0]
    assert arrays["per_seed"].shape == (3, 4, 3)
    assert arrays["per_seed"][0].min() > 2.5
    assert np.abs(arrays["mean"][1:]).max() < 2.5
    assert (arrays["lo"] <= arrays["mean"]).all()
    assert (arrays["mean"] <= arrays["hi"]).all()


def test_execute_writes_the_table_of_the_mean_direction_lens(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _run_dir(tmp_path)
    readers = _readers(tiny_arch)
    execute(_config(), run_dir, tiny_arch, readers, WORDS)
    table = _json(run_dir, "table")
    assert table["columns"] == [
        "concept",
        "layer",
        "pole",
        "rank",
        "token",
        "z",
    ]
    assert len(table["rows"]) == 4 * 3 * 2 * 3
    assert [r[0] for r in table["rows"]][::18] == list(readers)
    for layer in LAYERS:
        top = _rows(run_dir, "r1", layer, "+")
        bottom = _rows(run_dir, "r1", layer, "-")
        assert [r[3] for r in top] == [1, 2, 3]
        assert GOOD[0] in {r[4] for r in top}
        assert BAD[0] in {r[4] for r in bottom}
        z_top = [r[5] for r in top]
        z_bottom = [r[5] for r in bottom]
        assert z_top == sorted(z_top, reverse=True)
        assert z_bottom == sorted(z_bottom)
        assert z_top[0] > 0 > z_bottom[0]


def test_the_summary_names_the_concepts_the_words_and_the_best_layers(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _run_dir(tmp_path)
    readers = _readers(tiny_arch)
    execute(_config(), run_dir, tiny_arch, readers, WORDS)
    summary = _json(run_dir, "summary")
    assert set(summary) == {
        "concepts",
        "lens_concepts",
        "words",
        "random_separation",
        "best_lens_layer",
        "center",
        "scaled",
    }
    assert summary["concepts"] == list(readers)
    assert summary["lens_concepts"] == ["r1", "r7", "r6:arc"]
    assert summary["words"]["r1"] == {"+": list(GOOD), "-": list(BAD)}
    assert summary["words"]["r6:arc"] == summary["words"]["r7"]
    assert "r4:anger" not in summary["words"]
    assert summary["center"] is (tiny_arch.spec.model_type == "gpt2")
    assert summary["scaled"] is True
    mean = run_dir.read_arrays("curves_lens")["mean"]
    for i, label in enumerate(summary["lens_concepts"]):
        assert (
            summary["best_lens_layer"][label] == LAYERS[int(mean[i].argmax())]
        )


def test_the_random_baseline_is_the_lens_of_random_directions(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(seed=7, n_random=30)
    run_dir = _run_dir(tmp_path, config)
    execute(config, run_dir, tiny_arch, _readers(tiny_arch), WORDS)
    unembedding, scale, center = lens_params(tiny_arch)
    rng = np.random.default_rng(derive_seed(7, "random"))
    logits = lens_logits(
        rng.standard_normal((30, tiny_arch.d_model)),
        unembedding,
        scale=scale,
        center=center,
    )
    baseline = _json(run_dir, "summary")["random_separation"]
    assert list(baseline) == ["r1", "r7", "r6:arc"]
    for label, key in (("r1", "r1"), ("r7", "r7"), ("r6:arc", "r7")):
        positive, negative = (
            pole_ids(tiny_arch, pole)[0] for pole in WORDS[key]
        )
        separation = pole_separation(logits, positive, negative)
        mean, sd = baseline[label]
        assert mean == pytest.approx(separation.mean(), abs=1e-4)
        assert sd == pytest.approx(separation.std(ddof=1), abs=1e-4)
    assert abs(baseline["r1"][0]) < 1.0


def test_execute_uses_the_first_seeds_of_the_longer_sets(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _run_dir(tmp_path)
    execute(_config(), run_dir, tiny_arch, _readers(tiny_arch), WORDS)
    arrays = run_dir.read_arrays("curves_lens")
    assert arrays["per_seed"].shape == (3, 4, 3)


def test_a_concept_without_words_gets_the_token_table_only(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    run_dir = _run_dir(tmp_path)
    readers = {"r4:anger": _readers(tiny_arch)["r4:anger"]}
    execute(_config(), run_dir, tiny_arch, readers, WORDS)
    assert not (run_dir.path / "curves_lens.npz").exists()
    assert not (run_dir.path / "curves_lens.json").exists()
    assert len(_json(run_dir, "table")["rows"]) == 1 * 3 * 2 * 3
    summary = _json(run_dir, "summary")
    assert summary["concepts"] == ["r4:anger"]
    assert summary["lens_concepts"] == []
    assert summary["words"] == {}
    assert summary["random_separation"] == {}
    assert summary["best_lens_layer"] == {}


def _unfit(name: str, loaded: LoadedModel) -> tuple[Config, Any, Any, str]:
    readers = _readers(loaded)
    cases: dict[str, tuple[Config, Any, Any, str]] = {
        "top-k": (_config(top_k=0), readers, WORDS, "top_k"),
        "n-random": (_config(n_random=1), readers, WORDS, "n_random"),
        "no-single-token": (
            _config(),
            readers,
            {"r1": ((" bad",), (" text",))},
            "r1: no pole word is a single token for this tokenizer",
        ),
        "nine": (
            _config(),
            {f"c{i}": readers["r1"] for i in range(9)},
            WORDS,
            "at most 8",
        ),
        "layers": (
            _config(),
            {
                "r1": readers["r1"],
                "r7": ReaderSet(
                    (1, 2, 4), readers["r7"].vectors, loaded.n_layers
                ),
            },
            WORDS,
            "layers",
        ),
    }
    return cases[name]


@pytest.mark.parametrize(
    "name",
    ["top-k", "n-random", "no-single-token", "nine", "layers"],
)
def test_execute_refuses_unusable_input_before_any_work(
    tiny_arch: LoadedModel, tmp_path: Path, name: str
) -> None:
    config, readers, words, match = _unfit(name, tiny_arch)
    run_dir = _run_dir(tmp_path, config)
    with pytest.raises(ValueError, match=match):
        execute(config, run_dir, tiny_arch, readers, words)
    assert not list(run_dir.path.glob("*.npz"))
    assert not (run_dir.path / "summary.json").exists()
    assert not (run_dir.path / "table.json").exists()


def test_run_loads_the_model_and_gathers_the_readers(
    tiny_arch: LoadedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readers = _readers(tiny_arch)
    calls: dict[str, Any] = {}

    def fake_load_model(key: str) -> LoadedModel:
        calls["model"] = key
        return tiny_arch

    def fake_gather(
        out_root: Path, sources: tuple[str, ...], model: str, extra: Any
    ) -> dict[str, ReaderSet]:
        calls["gather"] = (out_root, sources, model, extra)
        return readers

    def fake_execute(*args: Any) -> None:
        calls["execute"] = args

    monkeypatch.setattr(a2_logit_lens, "load_model", fake_load_model)
    monkeypatch.setattr(a2_logit_lens, "gather_readers", fake_gather)
    monkeypatch.setattr(a2_logit_lens, "execute", fake_execute)
    config = _config(sources=("r1", "r7"), extra_runs=("somewhere",))
    run_dir = _run_dir(tmp_path, config)
    run(config, run_dir)
    assert calls["model"] == "tiny"
    assert calls["gather"] == (tmp_path, ("r1", "r7"), "tiny", ("somewhere",))
    assert calls["execute"] == (config, run_dir, tiny_arch, readers)


def _executed(tiny_arch: LoadedModel, tmp_path: Path) -> RunDir:
    config = _config(top_k=5)
    run_dir = _run_dir(tmp_path, config)
    execute(config, run_dir, tiny_arch, _readers(tiny_arch), WORDS)
    return run_dir


def test_plot_draws_the_curves_and_the_token_tables(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    paths = plot(_executed(tiny_arch, tmp_path), tmp_path / "figs")
    assert sorted(p.name for p in paths) == [
        "curves_lens.pdf",
        "curves_lens.png",
        "curves_lens.svg",
        "lens_tokens.pdf",
        "lens_tokens.png",
        "lens_tokens.svg",
    ]
    assert all(p.stat().st_size > 0 for p in paths)


def test_plot_draws_only_the_tokens_without_a_curves_file(
    tiny_arch: LoadedModel, tmp_path: Path
) -> None:
    config = _config(top_k=5)
    run_dir = _run_dir(tmp_path, config)
    readers = {"r4:anger": _readers(tiny_arch)["r4:anger"]}
    execute(config, run_dir, tiny_arch, readers, WORDS)
    paths = plot(run_dir, tmp_path / "figs")
    assert sorted(p.name for p in paths) == [
        "lens_tokens.pdf",
        "lens_tokens.png",
        "lens_tokens.svg",
    ]


def _plotted_figure(
    run_dir: RunDir, out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Figure:
    saved: list[Figure] = []
    real = a2_logit_lens.save_figure

    def spy(fig: Figure, folder: Path, name: str) -> list[Path]:
        if name == "lens_tokens":
            saved.append(fig)
        return real(fig, folder, name)

    monkeypatch.setattr(a2_logit_lens, "save_figure", spy)
    plot(run_dir, out_dir)
    (figure,) = saved
    return figure


def test_plot_has_one_table_per_concept_with_the_top_and_bottom_five(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _executed(tiny_arch, tmp_path)
    figure = _plotted_figure(run_dir, tmp_path / "figs", monkeypatch)
    tables = [table for ax in figure.axes for table in ax.tables]
    assert len(tables) == 4
    cells = [
        cell.get_text().get_text() for cell in tables[0].get_celld().values()
    ]
    assert " good" in cells
    assert " text" in cells


def test_plot_writes_tokens_as_ascii_escapes(
    tiny_arch: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _executed(tiny_arch, tmp_path)
    table = _json(run_dir, "table")
    odd = ["\u65e5\u672c", "\u00e9t\u00e9", "\ufffd", "\n"]
    for row, token in zip(table["rows"][:4], odd, strict=True):
        row[4] = token
    run_dir.write_json("table", table)
    figure = _plotted_figure(run_dir, tmp_path / "figs", monkeypatch)
    cells = [
        cell.get_text().get_text()
        for ax in figure.axes
        for t in ax.tables
        for cell in t.get_celld().values()
    ]
    assert all(text.isascii() for text in cells)
    assert "\\u65e5\\u672c" in cells
    assert "\\xe9t\\xe9" in cells
    assert "\\ufffd" in cells
    assert "\\n" in cells


@pytest.mark.slow
@pytest.mark.parametrize("key", ["gpt2", "qwen3-1.7b-base"])
def test_the_real_tokenizers_keep_a_word_of_every_pole(key: str) -> None:
    from transformers import AutoTokenizer

    checkpoint = MODELS[key]
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint.repo_id,
            revision=checkpoint.revision,
            local_files_only=True,
        )
    except OSError:
        pytest.skip(f"the tokenizer of {key} is not cached")
    loaded = cast(LoadedModel, SimpleNamespace(tokenizer=tokenizer))
    for source, poles in POLE_WORDS.items():
        for name, words in zip(("+", "-"), poles, strict=True):
            ids, kept = pole_ids(loaded, words)
            dropped = [w for w in words if w not in kept]
            print(key, source, name, "dropped:", dropped)
            assert ids, f"{key} {source} {name}: no single-token word"
