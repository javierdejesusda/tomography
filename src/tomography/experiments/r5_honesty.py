"""R5: reading honesty from GPT-2 on true statements."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np

from tomography.activations import PositionStrategy
from tomography.datasets.facts import load_facts
from tomography.datasets.records import Example
from tomography.datasets.records import Split
from tomography.datasets.stimulus import StimulusSet
from tomography.datasets.stimulus import build_stimulus_set
from tomography.experiments.base import RunDir
from tomography.experiments.base import derive_seed
from tomography.experiments.sweep import SERIES
from tomography.experiments.sweep import Prepared
from tomography.experiments.sweep import ReadingConfig
from tomography.experiments.sweep import SweepResult
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import stimulus_activations
from tomography.experiments.sweep import sweep_from_activations
from tomography.experiments.sweep import write_curves
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.directions import Method
from tomography.reading.metrics import best_layer
from tomography.templates import PAPER_TEMPLATES
from tomography.templates import ConceptTemplate
from tomography.viz.style import save_figure
from tomography.viz.style import style_context
from tomography.viz.tokens import plot_scan

_PERSONAS = ("honest", "untruthful")
_READER = SERIES.index("reader")
_PLAIN = ConceptTemplate("honesty_plain", "{stimulus}", tagged=False)
_TRANSFER_POSITION: PositionStrategy = "mean"


@dataclass(frozen=True, kw_only=True)
class Config(ReadingConfig):
    """R5 settings: GPT-2 honesty on facts.

    The readers are trained on the response tokens of true statements shown
    under an honest and an untruthful persona. Scoring held-out statements
    under the same two personas is only a sanity check: the two renderings
    differ in the persona instruction itself, so any direction separates
    them. On GPT-2 the reader scores 1.0 in every layer with either finder,
    a random direction reaches 0.928 at layer 1 (0.58 to 0.93 across
    layers) and the shuffled null sits at 0.494. The headline accuracy is
    therefore the transfer: the same readers, with no refit, score plain
    true and false statements without any persona instruction, paired one
    true with one false, on the mean of their response tokens.

    The paper reads functions with the first principal component of the pair
    differences (`pca_diff`), which stays selectable. R5 fits a supervised
    `logistic` direction by default, as R1 to R4 and R7 do.

    Attributes:
        n_scan_statements: Number of fixed test statements drawn as token by
            layer scans.
    """

    experiment: str = "r5"
    model: str = "gpt2"
    method: Method = "logistic"
    position: PositionStrategy = "response"
    n_scan_statements: int = 4


def _stimuli(
    statements: Sequence[Example],
    loaded: LoadedModel,
    n_pairs: int | None,
    seed: int,
) -> StimulusSet:
    return build_stimulus_set(
        statements,
        PAPER_TEMPLATES["honesty"],
        chat_format_for(loaded),
        pairing="persona",
        n_pairs=n_pairs,
        seed=seed,
    )


def _plain_stimuli(
    statements: Sequence[Example],
    loaded: LoadedModel,
    n_pairs: int | None,
    seed: int,
) -> StimulusSet:
    return build_stimulus_set(
        statements,
        _PLAIN,
        chat_format_for(loaded),
        pairing="label_contrastive",
        n_pairs=n_pairs,
        seed=seed,
    )


def _prepare(
    config: Config,
    loaded: LoadedModel,
    layers: tuple[int, ...],
    stimuli: StimulusSet,
    position: PositionStrategy,
) -> Prepared:
    act = stimulus_activations(
        loaded,
        stimuli,
        layers=layers,
        position=position,
        bos=config.bos,
        batch_size=config.batch_size,
    )
    return stimuli, act


def _write_scans(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    result: SweepResult,
    statements: Sequence[Example],
) -> None:
    stimuli = _stimuli(
        statements, loaded, None, derive_seed(config.seed, "scan")
    )
    act = stimulus_activations(
        loaded,
        stimuli,
        layers=result.layers,
        position="response",
        bos=config.bos,
        batch_size=config.batch_size,
    )
    scores = result.readers[0].score(act.data)
    for i, statement in enumerate(statements):
        for persona, label in zip(_PERSONAS, (1, 0), strict=True):
            stimulus = next(
                s
                for s, (uid, y) in enumerate(
                    zip(stimuli.uids, stimuli.labels, strict=True)
                )
                if uid == statement.uid and y == label
            )
            rows = np.flatnonzero(act.row_stimulus == stimulus)
            stem = f"scan_{i}_{persona}"
            run_dir.write_arrays(
                stem,
                scores=scores[:, rows],
                layers=np.asarray(result.layers, dtype=np.int64),
            )
            run_dir.write_json(
                stem,
                {
                    "tokens": [
                        loaded.tokenizer.decode([int(t)])
                        for t in act.row_token_id[rows]
                    ],
                    "mask_first": False,
                },
            )


def _report(
    run_dir: RunDir, result: SweepResult, stem: str, val_name: str
) -> dict[str, Any]:
    val = result.per_seed(val_name)[_READER].mean(axis=0)
    curves = run_dir.read_arrays(stem)
    best = best_layer(val, result.layers)
    at = result.layers.index(best)
    return {
        "best_layer": best,
        "val_accuracy": float(val[at]),
        "test_accuracy": float(curves["mean"][_READER, at]),
        "ci": [
            float(curves["lo"][_READER, at]),
            float(curves["hi"][_READER, at]),
        ],
    }


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    statements: Sequence[Example],
) -> None:
    """Reads honesty on true statements and scores held-out statements.

    Statements are split before pairing, so no statement is both read and
    scored. The readers are fitted on the true statements, each shown under
    an honest and an untruthful persona with the response tokens as rows.
    They are scored on held-out statements in two ways, with no refit. The
    transfer sets pair a true and a false plain statement and read the mean
    of its tokens; this is the headline. The in-template sets pair the two
    persona renderings of a true statement and are a sanity check. The layer
    is chosen on val and reported on test. Writes `curves_transfer` and
    `curves` (test), one `scan_<i>_<persona>` token by layer scan for each
    of the first `n_scan_statements` true test statements, and
    `summary.json`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        statements: Facts with `train`, `val` and `test` splits, label 1 for
            a true statement and 0 for a false one.

    Raises:
        ValueError: If `config.n_scan_statements` exceeds the number of true
            test statements, or a split lacks true or false statements for
            the transfer pairs.
    """

    def part(split: Split, *, only_true: bool) -> list[Example]:
        return [
            e
            for e in statements
            if e.split == split and (e.label == 1 or not only_true)
        ]

    test = part("test", only_true=True)
    if config.n_scan_statements > len(test):
        raise ValueError(
            f"n_scan_statements={config.n_scan_statements} exceeds the "
            f"{len(test)} true test statements"
        )
    train_sets = [
        _stimuli(
            part("train", only_true=True),
            loaded,
            config.n_pairs,
            derive_seed(config.seed, "train", i),
        )
        for i in range(config.n_seeds)
    ]
    persona_sets = {
        name: _stimuli(
            part(name, only_true=True),
            loaded,
            config.n_eval_pairs,
            derive_seed(config.seed, "eval", name),
        )
        for name in ("val", "test")
    }
    transfer_sets = {
        f"transfer_{name}": _plain_stimuli(
            part(name, only_true=False),
            loaded,
            config.n_eval_pairs,
            derive_seed(config.seed, "eval", "transfer", name),
        )
        for name in ("val", "test")
    }
    layers = (
        tuple(range(1, loaded.n_layers + 1))
        if config.layers is None
        else config.layers
    )
    where_eval = (
        config.position
        if config.eval_position is None
        else config.eval_position
    )
    result = sweep_from_activations(
        [
            _prepare(config, loaded, layers, s, config.position)
            for s in train_sets
        ],
        {
            **{
                name: _prepare(config, loaded, layers, s, where_eval)
                for name, s in persona_sets.items()
            },
            **{
                name: _prepare(config, loaded, layers, s, _TRANSFER_POSITION)
                for name, s in transfer_sets.items()
            },
        },
        method=config.method,
        preprocess=config.preprocess,
        seed=config.seed,
    )
    for stem, name in (
        ("curves", "test"),
        ("curves_transfer", "transfer_test"),
    ):
        write_curves(
            run_dir,
            result.per_seed(name),
            result.layers,
            series=SERIES,
            stem=stem,
            chance=0.5,
            n_layers=loaded.n_layers,
        )
    _write_scans(
        config, run_dir, loaded, result, test[: config.n_scan_statements]
    )
    sizes = {
        "transfer": {
            name: int(transfer_sets[f"transfer_{name}"].groups.shape[0])
            for name in ("val", "test")
        },
        "in_template": {
            name: int(persona_sets[name].groups.shape[0])
            for name in ("val", "test")
        },
    }
    run_dir.write_json(
        "summary",
        {
            **_report(run_dir, result, "curves_transfer", "transfer_val"),
            "in_template": _report(run_dir, result, "curves", "val"),
            "n_seeds": config.n_seeds,
            "n_pairs": config.n_pairs,
            "n_eval_pairs": sizes,
            "n_scan": config.n_scan_statements,
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads GPT-2 and the facts, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    execute(config, run_dir, load_model(config.model), load_facts())


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the R5 curves and the token by layer scans.

    Args:
        run_dir: A finished R5 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths: `curves_transfer`, `curves`, then the
        scans in statement order, honest before untruthful.
    """
    paths = [
        path
        for stem in ("curves_transfer", "curves")
        for path in plot_curves_file(run_dir, out_dir, stem=stem)
    ]
    summary = cast(dict[str, Any], run_dir.read_json("summary"))
    for i in range(summary["n_scan"]):
        for persona in _PERSONAS:
            stem = f"scan_{i}_{persona}"
            arrays = run_dir.read_arrays(stem)
            sidecar = cast(dict[str, Any], run_dir.read_json(stem))
            with style_context():
                fig = plot_scan(
                    arrays["scores"],
                    arrays["layers"],
                    tokens=sidecar["tokens"],
                    mask_first=sidecar["mask_first"],
                )
            paths += save_figure(fig, out_dir, stem)
    return paths
