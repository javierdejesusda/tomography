"""R4: reading six emotions from GPT-2 on the emotion scenarios."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
from numpy.typing import NDArray

from tomography.activations import collect_activations
from tomography.analysis import tsne_embed
from tomography.datasets.emotions import EMOTIONS
from tomography.datasets.emotions import load_emotions
from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import ReadingConfig
from tomography.experiments.sweep import SweepResult
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import reading_sweep
from tomography.experiments.sweep import write_curves
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.metrics import best_layer
from tomography.reading.metrics import cosine_matrix
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ConceptTemplate
from tomography.viz.geometry import plot_embedding
from tomography.viz.geometry import plot_matrix
from tomography.viz.style import save_figure
from tomography.viz.style import style_context

_READER = SERIES.index("reader")
_EMBED_LAYERS = 6


@dataclass(frozen=True, kw_only=True)
class Config(ReadingConfig):
    """R4 settings: GPT-2 emotions with the shared defaults.

    Attributes:
        emotions: The emotions to read, a subset of `EMOTIONS`.
        embed_layers: Layers of the t-SNE grid; None means about six evenly
            spaced layers.
        embed_points: Number of stimuli in the t-SNE grid.
        perplexity: The t-SNE perplexity; it must be below `embed_points`.
    """

    experiment: str = "r4"
    model: str = "gpt2"
    emotions: tuple[str, ...] = EMOTIONS
    embed_layers: tuple[int, ...] | None = None
    embed_points: int = 600
    perplexity: float = 30.0


def _stimuli(
    scenarios: Sequence[Example],
    loaded: LoadedModel,
    emotion: str,
    n_pairs: int | None,
    seed: int,
) -> StimulusSet:
    return build_stimulus_set(
        scenarios,
        PAPER_TEMPLATES["emotion_concept"],
        chat_format_for(loaded),
        pairing="one_vs_rest",
        target_class=EMOTIONS.index(emotion),
        concept=emotion,
        n_pairs=n_pairs,
        seed=seed,
    )


def _direction(result: SweepResult, layer: int) -> NDArray[np.float64]:
    at = result.layers.index(layer)
    return np.mean(
        [r.signs[at] * r.directions[at] for r in result.readers], axis=0
    )


def _embed(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    scenarios: Sequence[Example],
    sweep_layers: tuple[int, ...],
) -> None:
    pool = [
        e
        for e in scenarios
        if e.split == "test" and e.meta["emotion"] in config.emotions
    ]
    rng = np.random.default_rng(derive_seed(config.seed, "embed"))
    chosen = [
        pool[i] for i in rng.permutation(len(pool))[: config.embed_points]
    ]
    chat = chat_format_for(loaded)
    template = PAPER_TEMPLATES["emotion_concept"]
    assert isinstance(template, ConceptTemplate)
    rendered = [template.render(e.text, "emotion", chat) for e in chosen]
    if config.embed_layers is None:
        spaced = np.linspace(0, len(sweep_layers) - 1, _EMBED_LAYERS)
        layers = tuple(
            sweep_layers[i] for i in np.unique(spaced.round().astype(int))
        )
    else:
        layers = config.embed_layers
    act = collect_activations(
        loaded,
        [r.text for r in rendered],
        layers=layers,
        position=config.position,
        response_spans=[r.response_span for r in rendered],
        concept_spans=[r.concept_span for r in rendered],
        bos=config.bos,
        batch_size=config.batch_size,
    )
    class_ids = np.array(
        [EMOTIONS.index(str(e.meta["emotion"])) for e in chosen],
        dtype=np.int64,
    )
    run_dir.write_arrays(
        "embedding",
        xy=tsne_embed(
            act.data,
            perplexity=config.perplexity,
            seed=derive_seed(config.seed, "tsne"),
        ),
        labels=act.row_values(class_ids),
        layers=np.asarray(act.layers, dtype=np.int64),
    )
    run_dir.write_json("embedding", {"classes": list(EMOTIONS)})


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    scenarios: Sequence[Example],
) -> None:
    """Reads each emotion one-vs-rest and compares the six directions.

    Scenarios are split before pairing. Per emotion, the layer is chosen on
    val and reported on test. Writes `curves_<emotion>` (test), `matrix`
    (cosine similarity of the directions at the best mean val layer),
    `embedding` (t-SNE of test scenarios) and `summary.json`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        scenarios: Emotion scenarios with `train`, `val` and `test` splits,
            labeled with the emotion's index in `EMOTIONS`.

    Raises:
        ValueError: If `config.emotions` is empty, repeats an emotion or
            names one outside `EMOTIONS`.
    """
    emotions = config.emotions
    if (
        not emotions
        or len(set(emotions)) != len(emotions)
        or not set(emotions) <= set(EMOTIONS)
    ):
        raise ValueError(
            f"emotions must be a non-empty set of {EMOTIONS}, got {emotions}"
        )

    def part(split: Split) -> list[Example]:
        return [e for e in scenarios if e.split == split]

    results: dict[str, SweepResult] = {}
    eval_sizes: dict[str, dict[str, int]] = {}
    for emotion in emotions:
        train_sets = [
            _stimuli(
                part("train"),
                loaded,
                emotion,
                config.n_pairs,
                derive_seed(config.seed, "train", emotion, i),
            )
            for i in range(config.n_seeds)
        ]
        eval_sets = {
            name: _stimuli(
                part(name),
                loaded,
                emotion,
                config.n_eval_pairs,
                derive_seed(config.seed, "eval", emotion, name),
            )
            for name in ("val", "test")
        }
        result = reading_sweep(
            loaded,
            train_sets,
            eval_sets,
            method=config.method,
            preprocess=config.preprocess,
            position=config.position,
            eval_position=config.eval_position,
            layers=config.layers,
            seed=derive_seed(config.seed, "sweep", emotion),
            bos=config.bos,
            batch_size=config.batch_size,
        )
        write_curves(
            run_dir,
            result.per_seed("test"),
            result.layers,
            series=SERIES,
            stem=f"curves_{emotion}",
            chance=0.5,
            n_layers=loaded.n_layers,
        )
        results[emotion] = result
        eval_sizes[emotion] = {
            name: int(s.groups.shape[0]) for name, s in eval_sets.items()
        }
    layers = results[emotions[0]].layers
    val = {
        e: r.per_seed("val")[_READER].mean(axis=0) for e, r in results.items()
    }
    matrix_layer = best_layer(np.mean(list(val.values()), axis=0), layers)
    run_dir.write_arrays(
        "matrix",
        values=cosine_matrix(
            np.stack([_direction(results[e], matrix_layer) for e in emotions])
        ),
    )
    run_dir.write_json(
        "matrix",
        {
            "rows": list(emotions),
            "cols": list(emotions),
            "title": (
                f"Cosine similarity of the emotion directions at layer "
                f"{matrix_layer}"
            ),
        },
    )
    _embed(config, run_dir, loaded, scenarios, layers)
    summary = {}
    for emotion in emotions:
        test = run_dir.read_arrays(f"curves_{emotion}")
        best = best_layer(val[emotion], layers)
        at = layers.index(best)
        summary[emotion] = {
            "best_layer": best,
            "val_accuracy": float(val[emotion][at]),
            "test_accuracy": float(test["mean"][_READER, at]),
            "ci": [
                float(test["lo"][_READER, at]),
                float(test["hi"][_READER, at]),
            ],
        }
    run_dir.write_json(
        "summary",
        {
            "matrix_layer": matrix_layer,
            "emotions": summary,
            "n_seeds": config.n_seeds,
            "n_pairs": config.n_pairs,
            "n_eval_pairs": eval_sizes,
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads GPT-2 and the emotion scenarios, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    execute(config, run_dir, load_model(config.model), load_emotions())


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the R4 curves, direction similarity matrix and t-SNE grid.

    Args:
        run_dir: A finished R4 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths: one curves figure per emotion, then the
        `matrix` and `embedding` figures.
    """
    emotions = run_dir.config["emotions"]
    paths = [
        path
        for emotion in emotions
        for path in plot_curves_file(run_dir, out_dir, stem=f"curves_{emotion}")
    ]
    matrix = run_dir.read_arrays("matrix")
    matrix_info = cast(dict[str, Any], run_dir.read_json("matrix"))
    with style_context():
        fig = plot_matrix(
            matrix["values"],
            rows=matrix_info["rows"],
            cols=matrix_info["cols"],
            title=matrix_info["title"],
            kind="diverging",
            vmin=-1.0,
            vmax=1.0,
            cbar_label="Cosine similarity",
        )
    paths += save_figure(fig, out_dir, "matrix")
    embedding = run_dir.read_arrays("embedding")
    embedding_info = cast(dict[str, Any], run_dir.read_json("embedding"))
    with style_context():
        fig = plot_embedding(
            embedding["xy"],
            embedding["labels"],
            embedding["layers"],
            classes=embedding_info["classes"],
        )
    return paths + save_figure(fig, out_dir, "embedding")
