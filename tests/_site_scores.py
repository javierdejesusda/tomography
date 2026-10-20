"""Independent readers of one reading score from the raw result files.

Used by the table tests; they never call the site loaders they check.
"""

import json
from pathlib import Path
import runpy

import numpy as np

_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
data, npz, run = _R["data"], _R["npz"], _R["run"]
CONFIGS = (
    Path(__file__).resolve().parents[1]
    / "src/tomography/site/data/model_configs.json"
)


def n_layers(model: str) -> int:
    """Number of decoder layers from the stored model config."""
    config = json.loads(CONFIGS.read_text("utf-8"))[model]
    return int(config.get("n_layer") or config["num_hidden_layers"])


def random_at(path: Path, stem: str, layer: int) -> float:
    """Mean random-direction accuracy at a layer from a curves file."""
    series = data(path, f"{stem}.json")["series"]
    mean = npz(path, f"{stem}.npz")["mean"]
    return float(mean[series.index("random")][layer - 1])


def score(concept: str, model: str) -> tuple[int, float, float, int]:
    """Layer, reader, random and test pairs of a concept on a checkpoint.

    Concepts: sentiment_r1, sentiment, utility, morality, truth, honesty,
    template.
    """
    if concept == "sentiment":
        path = run("p3", model)
        meta = data(path, "sizes_sentiment.json")
        arrays = npz(path, "sizes_sentiment.npz")
        k = int(np.flatnonzero(arrays["sizes"] == 64)[0])
        col = int(np.flatnonzero(arrays["layers"] == meta["layer"])[0])
        acc = arrays["accuracy"]
        return (
            int(meta["layer"]),
            float(acc[k, 1, 0, :, col].mean()),
            float(acc[k, 1, 1, :, col].mean()),
            214,
        )
    if concept == "sentiment_r1":
        path = run("r1", model)
        s = data(path, "summary.json")
        layer = s["best_layer"]
        return (
            layer,
            float(s["best_accuracy"]),
            random_at(path, "curves", layer),
            int(s["n_eval_pairs"]),
        )
    if concept == "template":
        path = run("r5", model)
        s = data(path, "summary.json")
        block = s["in_template"]
        layer = block["best_layer"]
        return (
            layer,
            float(block["test_accuracy"]),
            random_at(path, "curves", layer),
            int(s["n_eval_pairs"]["in_template"]["test"]),
        )
    exp = {"utility": "r2", "morality": "r3", "truth": "r7", "honesty": "r5"}[
        concept
    ]
    path = run(exp, model)
    s = data(path, "summary.json")
    if exp == "r7":
        layer = s["val_layer"]
        reader = s["sets"]["test"]["accuracy_at_val_layer"]
        n = s["sets"]["test"]["n_eval_pairs"]
    elif exp == "r5":
        layer = s["best_layer"]
        reader = s["test_accuracy"]
        n = s["n_eval_pairs"]["transfer"]["test"]
    else:
        layer = s["best_layer"]
        reader = s["test_accuracy"]
        n = s["n_eval_pairs"]["test"]
    stem = "curves_transfer" if exp == "r5" else "curves"
    return int(layer), float(reader), random_at(path, stem, layer), int(n)
