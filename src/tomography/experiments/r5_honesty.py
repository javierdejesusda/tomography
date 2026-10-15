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
from tomography.experiments.sweep import ReadingConfig
from tomography.experiments.sweep import SweepResult
from tomography.experiments.sweep import chat_format_for
from tomography.experiments.sweep import plot_curves_file
from tomography.experiments.sweep import reading_sweep
from tomography.experiments.sweep import stimulus_activations
from tomography.experiments.sweep import write_curves
from tomography.model import LoadedModel
from tomography.model import load_model
from tomography.reading.metrics import best_layer
from tomography.templates import PAPER_TEMPLATES
from tomography.viz.style import save_figure
from tomography.viz.style import style_context
from tomography.viz.tokens import plot_scan

_PERSONAS = ("honest", "untruthful")


@dataclass(frozen=True, kw_only=True)
class Config(ReadingConfig):
    """R5 settings: GPT-2 honesty on true statements.

    Attributes:
        n_scan_statements: Number of fixed test statements drawn as token by
            layer scans.
    """

    experiment: str = "r5"
    model: str = "gpt2"
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


def execute(
    config: Config,
    run_dir: RunDir,
    loaded: LoadedModel,
    statements: Sequence[Example],
) -> None:
    """Reads honesty on true statements and scores held-out statements.

    Statements are split before pairing, so no statement is both read and
    scored. Each appears under an honest and an untruthful persona and the
    response tokens are the rows. The layer is chosen on val and reported on
    test. Writes `curves` (test), one `scan_<i>_<persona>` token by layer
    scan for each of the first `n_scan_statements` test statements and
    `summary.json`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
        loaded: The wrapped model.
        statements: True statements with `train`, `val` and `test` splits.

    Raises:
        ValueError: If `config.n_scan_statements` exceeds the number of test
            statements.
    """

    def part(split: Split) -> list[Example]:
        return [e for e in statements if e.split == split]

    test = part("test")
    if config.n_scan_statements > len(test):
        raise ValueError(
            f"n_scan_statements={config.n_scan_statements} exceeds the "
            f"{len(test)} test statements"
        )
    train_sets = [
        _stimuli(
            part("train"),
            loaded,
            config.n_pairs,
            derive_seed(config.seed, "train", i),
        )
        for i in range(config.n_seeds)
    ]
    eval_sets = {
        name: _stimuli(
            part(name),
            loaded,
            config.n_eval_pairs,
            derive_seed(config.seed, "eval", name),
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
        seed=config.seed,
        bos=config.bos,
        batch_size=config.batch_size,
    )
    write_curves(
        run_dir,
        result.per_seed("test"),
        result.layers,
        series=SERIES,
        chance=0.5,
        n_layers=loaded.n_layers,
    )
    _write_scans(
        config, run_dir, loaded, result, test[: config.n_scan_statements]
    )
    index = SERIES.index("reader")
    val = result.per_seed("val")[index].mean(axis=0)
    curves = run_dir.read_arrays("curves")
    best = best_layer(val, result.layers)
    at = result.layers.index(best)
    run_dir.write_json(
        "summary",
        {
            "best_layer": best,
            "val_accuracy": float(val[at]),
            "test_accuracy": float(curves["mean"][index, at]),
            "ci": [
                float(curves["lo"][index, at]),
                float(curves["hi"][index, at]),
            ],
            "n_seeds": config.n_seeds,
            "n_pairs": config.n_pairs,
            "n_eval_pairs": {
                name: int(s.groups.shape[0]) for name, s in eval_sets.items()
            },
            "n_scan": config.n_scan_statements,
        },
    )


def run(config: Config, run_dir: RunDir) -> None:
    """Loads GPT-2 and the true statements, then runs `execute`.

    Args:
        config: The experiment settings.
        run_dir: The run to write into.
    """
    statements = [e for e in load_facts() if e.label == 1]
    execute(config, run_dir, load_model(config.model), statements)


def plot(run_dir: RunDir, out_dir: Path) -> list[Path]:
    """Draws the R5 curves and the token by layer scans.

    Args:
        run_dir: A finished R5 run.
        out_dir: Folder of the figures.

    Returns:
        The PDF, PNG and SVG paths: `curves`, then the scans in statement
        order, honest before untruthful.
    """
    paths = plot_curves_file(run_dir, out_dir)
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
