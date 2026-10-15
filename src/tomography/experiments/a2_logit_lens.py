"""A2: logit lens of the reading directions.

A2 reads each reading direction through the linear part of the final norm and
the unembedding, and asks which tokens it promotes. It reads no activations:
the directions come from the `readers*.npz` files of finished reading runs,
and the model is loaded only for its unembedding, final norm and tokenizer.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from typing import cast

from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
import numpy as np
from numpy.typing import NDArray
import torch

from tomography.analysis import lens_logits
from tomography.analysis import mean_direction
from tomography.analysis import pole_separation
from tomography.analysis import standardize
from tomography.analysis import top_tokens
from tomography.experiments.aggregate import ReaderSet
from tomography.experiments.aggregate import common_layers
from tomography.experiments.aggregate import depth_layer
from tomography.experiments.aggregate import gather_readers
from tomography.experiments.aggregate import out_root_of
from tomography.experiments.base import ExperimentConfig
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import write_curves
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.metrics import best_layer
from tomography.viz import style
from tomography.viz.style import save_figure

PoleWords = tuple[tuple[str, ...], tuple[str, ...]]

POLE_WORDS: Mapping[str, PoleWords] = MappingProxyType(
    {
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
)
WORDS_OF: Mapping[str, str] = MappingProxyType({"r6": "r7"})

SHOWN_TOKENS = 5
QUARTILES = (0.25, 0.5, 0.75, 1.0)
RANK_WIDTH = 0.08


@dataclass(frozen=True, kw_only=True)
class Config(ExperimentConfig):
    """A2 settings.

    Attributes:
        sources: Experiments whose default runs on `model` are read.
        extra_runs: Finished runs that replace a default run.
        top_k: Tokens per pole, layer and concept in the table.
        n_random: Random directions of the baseline.
        n_boot: Bootstrap resamples of the separation bands.
    """

    experiment: str = "a2"
    model: str = "gpt2"
    sources: tuple[str, ...] = ("r1", "r2", "r3", "r5", "r7")
    extra_runs: tuple[str, ...] = ()
    top_k: int = 10
    n_random: int = 100
    n_boot: int = 2000


@dataclass(frozen=True)
class _Poles:
    positive: list[int]
    negative: list[int]
    kept_positive: list[str]
    kept_negative: list[str]


def lens_params(
    loaded: LoadedModel,
) -> tuple[NDArray[np.float32], NDArray[np.float32] | None, bool]:
    """Returns what the lens needs from a model.

    Args:
        loaded: The wrapped model.

    Returns:
        The unembedding of shape (V, d) as float32, the weight of the final
        norm (None when it has none) and whether the norm centers its input,
        which a LayerNorm does and an RMSNorm does not.
    """
    unembedding = loaded.unembedding().detach().float().cpu().numpy()
    weight = getattr(loaded.final_norm, "weight", None)
    scale = None if weight is None else weight.detach().float().cpu().numpy()
    return unembedding, scale, isinstance(loaded.final_norm, torch.nn.LayerNorm)


def pole_ids(
    loaded: LoadedModel, words: Sequence[str]
) -> tuple[list[int], list[str]]:
    """Keeps the words that the tokenizer writes as one token.

    Args:
        loaded: The wrapped model.
        words: Words with their leading space.

    Returns:
        The token ids and the words kept, in the order given.
    """
    ids: list[int] = []
    kept: list[str] = []
    for word in words:
        encoded = loaded.tokenizer.encode(word, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(int(encoded[0]))
            kept.append(word)
    return ids, kept


def _poles(
    loaded: LoadedModel,
    labels: Sequence[str],
    words: Mapping[str, PoleWords],
) -> dict[str, _Poles]:
    poles: dict[str, _Poles] = {}
    for label in labels:
        experiment = label.partition(":")[0]
        source = words.get(WORDS_OF.get(experiment, experiment))
        if source is None:
            continue
        positive, kept_positive = pole_ids(loaded, source[0])
        negative, kept_negative = pole_ids(loaded, source[1])
        if not positive or not negative:
            raise ValueError(
                f"{label}: no pole word is a single token for this tokenizer"
            )
        poles[label] = _Poles(positive, negative, kept_positive, kept_negative)
    return poles


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    readers: Mapping[str, ReaderSet],
    words: Mapping[str, PoleWords] = POLE_WORDS,
) -> None:
    """Reads the reading directions through the logit lens.

    Every concept uses its first `n_min` seeds, `n_min` being the smallest
    seed count among the concepts. A concept with pole words gets the
    separation of its poles at every layer for every seed (`curves_lens`),
    compared with random directions in the summary; the lens of a random
    direction does not depend on the layer. Every concept, with or without
    words, gets the highest and lowest tokens of the lens of its mean
    direction at every layer (`table.json`).

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model; only its unembedding, final norm and
            tokenizer are used.
        readers: The reader sets by concept label, at most eight, read at
            the same layers of the same model.
        words: The pole words by source experiment. A label
            `<experiment>:<suffix>` takes the words of `<experiment>`,
            through `WORDS_OF`.

    Raises:
        ValueError: If `top_k` is below 1, `n_random` is below 2, the
            readers cannot be compared, or a concept with words has an empty
            pole after the single-token filter.
    """
    if config.top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {config.top_k}")
    if config.n_random < 2:
        raise ValueError(f"n_random must be at least 2, got {config.n_random}")
    layers, n_layers, n_min = common_layers(readers)
    labels = list(readers)
    poles = _poles(loaded, labels, words)
    unembedding, scale, center = lens_params(loaded)
    vectors = {label: readers[label].vectors[:n_min] for label in labels}

    def lens(directions: NDArray[np.float64]) -> NDArray[np.float32]:
        return lens_logits(directions, unembedding, scale=scale, center=center)

    separation = {
        label: np.stack(
            [
                pole_separation(
                    lens(vectors[label][:, k]), p.positive, p.negative
                )
                for k in range(len(layers))
            ],
            axis=1,
        )
        for label, p in poles.items()
    }
    rows: list[list[Any]] = []
    for label in labels:
        means = mean_direction(vectors[label])
        for k, layer in enumerate(layers):
            z = standardize(lens(means[k]))
            top, bottom = top_tokens(z, config.top_k)
            for pole, ids in (("+", top), ("-", bottom)):
                for rank, token_id in enumerate(ids, start=1):
                    token = loaded.tokenizer.decode([int(token_id)])
                    rows.append(
                        [label, layer, pole, rank, token, float(z[token_id])]
                    )
    baseline: dict[str, list[float]] = {}
    if poles:
        rng = np.random.default_rng(derive_seed(config.seed, "random"))
        random_logits = lens(
            rng.standard_normal((config.n_random, unembedding.shape[1]))
        )
        for label, p in poles.items():
            drawn = pole_separation(random_logits, p.positive, p.negative)
            baseline[label] = [float(drawn.mean()), float(drawn.std(ddof=1))]
        write_curves(
            run_dir,
            np.stack(list(separation.values())),
            layers,
            series=list(separation),
            stem="curves_lens",
            ylabel="Lens separation (z)",
            chance=0.0,
            n_layers=n_layers,
            n_boot=config.n_boot,
            seed=config.seed,
        )
    run_dir.write_json(
        "table",
        {
            "columns": ["concept", "layer", "pole", "rank", "token", "z"],
            "n_layers": n_layers,
            "rows": rows,
        },
    )
    run_dir.write_json(
        "summary",
        {
            "concepts": labels,
            "lens_concepts": list(poles),
            "words": {
                label: {"+": p.kept_positive, "-": p.kept_negative}
                for label, p in poles.items()
            },
            "random_separation": baseline,
            "best_lens_layer": {
                label: best_layer(values.mean(axis=0), layers)
                for label, values in separation.items()
            },
            "center": center,
            "scaled": scale is not None,
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Gathers the readers of the source runs, loads the model and analyzes.

    The source runs are looked up under the out root of `run_dir`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    readers = gather_readers(
        out_root_of(run_dir), config.sources, config.model, config.extra_runs
    )
    execute(config, run_dir, load_model(config.model), readers)


def _ascii(token: str) -> str:
    return token.encode("unicode_escape").decode("ascii")


def _token_figure(table: Mapping[str, Any]) -> Figure:
    rows = table["rows"]
    concepts = list(dict.fromkeys(row[0] for row in rows))
    layers = sorted({row[1] for row in rows})
    shown = sorted(
        {depth_layer(layers, q, table["n_layers"]) for q in QUARTILES}
    )
    tokens = {tuple(row[:4]): row[4] for row in rows}
    count = min(SHOWN_TOKENS, max(row[3] for row in rows))
    with style.style_context():
        fig, axes = style.new_figure(
            nrows=len(concepts), width=style.TEXT_WIDTH, aspect=0.25
        )
        for ax, concept in zip(axes[:, 0], concepts, strict=True):
            body = [
                [f"{pole}{rank}"]
                + [
                    _ascii(tokens[(concept, layer, pole, rank)])
                    for layer in shown
                ]
                for pole in "+-"
                for rank in range(1, count + 1)
            ]
            ax.axis("off")
            ax.set_title(concept, loc="left")
            grid = ax.table(
                cellText=body,
                colLabels=[""] + [f"layer {layer}" for layer in shown],
                colWidths=[RANK_WIDTH]
                + [(1.0 - RANK_WIDTH) / len(shown)] * len(shown),
                cellLoc="left",
                bbox=Bbox.from_bounds(0.0, 0.0, 1.0, 1.0),
            )
            grid.auto_set_font_size(False)
            grid.set_fontsize(6)
            for cell in grid.get_celld().values():
                cell.set_edgecolor(style.GRID)
                cell.set_linewidth(style.PX)
    return fig


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the separation curves and the token tables of an A2 run.

    The token table has one panel per concept with the top and bottom five
    tokens at the layers nearest the quartiles of the depth. Token text is
    written with escapes, so a byte-level or CJK token never needs a glyph
    the font lacks.

    Args:
        run_dir: A finished A2 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths of `curves_lens`, when the run wrote it,
        and of `lens_tokens`.
    """
    paths: list[Path] = []
    if (run_dir.path / "curves_lens.npz").is_file():
        paths.extend(plot_curves_file(run_dir, out_dir, stem="curves_lens"))
    table = cast(dict[str, Any], run_dir.read_json("table"))
    paths.extend(save_figure(_token_figure(table), out_dir, "lens_tokens"))
    return paths
