"""Parameter counts, test sizes and display names of the page.

Parameter counts are computed from the model configs.

`data/model_configs.json` stores the architecture fields of each pinned
checkpoint exactly as its `config.json` at that revision gives them. Counts
include the embeddings, the positional table of GPT-2 and every norm.
"""

import json
from pathlib import Path
from typing import Any

from tomography.site.manifest import Manifest

CONFIGS = Path(__file__).parent / "data" / "model_configs.json"


def _all() -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = json.loads(
        CONFIGS.read_text(encoding="utf-8")
    )
    return data


def model_config(key: str) -> dict[str, Any]:
    """Returns the stored architecture fields of a checkpoint.

    Args:
        key: Checkpoint key such as `gpt2-xl` or `qwen3-4b`.

    Returns:
        The config fields, with `repo_id` and `revision`.

    Raises:
        KeyError: If the checkpoint is unknown.
    """
    return _all()[key]


def _gpt2(config: dict[str, Any]) -> int:
    width = config["n_embd"]
    inner = config.get("n_inner") or 4 * width
    attention = 3 * width * width + 3 * width + width * width + width
    mlp = width * inner + inner + inner * width + width
    layer = attention + mlp + 4 * width
    embeddings = config["vocab_size"] * width + config["n_positions"] * width
    return int(embeddings + config["n_layer"] * layer + 2 * width)


def _qwen3(config: dict[str, Any]) -> int:
    width = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head = config["head_dim"]
    attention = (
        width * heads * head
        + 2 * width * kv_heads * head
        + heads * head * width
        + 2 * head
    )
    mlp = 3 * width * config["intermediate_size"]
    layer = attention + mlp + 2 * width
    total = config["vocab_size"] * width + config["num_hidden_layers"] * layer
    total += width
    if not config["tie_word_embeddings"]:
        total += config["vocab_size"] * width
    return int(total)


def param_count(key: str) -> int:
    """Counts the parameters of a checkpoint from its model config.

    Args:
        key: Checkpoint key.

    Returns:
        The parameter count, embeddings included.

    Raises:
        KeyError: If the checkpoint is unknown.
        ValueError: If the architecture has no formula.
    """
    config = model_config(key)
    kind = config["model_type"]
    if kind == "gpt2":
        return _gpt2(config)
    if kind == "qwen3":
        return _qwen3(config)
    raise ValueError(f"No parameter formula for {kind!r}")


def label(count: float) -> str:
    """Writes a parameter count with a unit.

    Args:
        count: Number of parameters.

    Returns:
        Text such as `124M` or `1.72B`.
    """
    if count >= 1e9:
        return f"{count / 1e9:.3g}B" if count >= 1e10 else f"{count / 1e9:.2f}B"
    return f"{count / 1e6:.0f}M"


REPORT_PAIRS = {"sentiment": 214, "truth": 59, "utility": 256}

DISPLAY = {
    "gpt2": "GPT-2 small",
    "gpt2-medium": "GPT-2 medium",
    "gpt2-large": "GPT-2 large",
    "gpt2-xl": "GPT-2 XL",
    "qwen3-0.6b-base": "Qwen3-0.6B-Base",
    "qwen3-0.6b": "Qwen3-0.6B",
    "qwen3-1.7b-base": "Qwen3-1.7B-Base",
    "qwen3-1.7b": "Qwen3-1.7B",
    "qwen3-4b-base": "Qwen3-4B-Base",
    "qwen3-4b": "Qwen3-4B",
    "pca_diff": "PCA of paired differences (paper)",
    "pca_raw": "PCA of activations",
    "mean_diff": "mean difference (mass-mean probe)",
    "logistic": "logistic regression",
    "kmeans": "k-means",
    "random": "random direction",
    "lat": "knowledge direction",
    "triviaqa": "TriviaQA",
    "kuq": "KUQ",
    "synthal": "SyntHal",
    "falseqa": "FalseQA",
    "popqa": "PopQA",
    "nq_open": "NQ-open",
    "wikidata": "Wikidata",
    "wikitext103": "WikiText-103",
    "cities": "cities",
    "neg_cities": "negated cities",
    "sp_en_trans": "Spanish to English",
    "neg_sp_en_trans": "negated Spanish to English",
    "larger_than": "larger than",
    "smaller_than": "smaller than",
    "cities_cities_conj": "cities conjunction",
    "cities_cities_disj": "cities disjunction",
    "common_claim_true_false": "common claims",
}


def display(key: str) -> str:
    """Writes a stored key as the page names it.

    Args:
        key: A checkpoint, finder, dataset, set or wording key.

    Returns:
        The display name; an unlisted key has its underscores written as
        spaces.
    """
    return DISPLAY.get(key, key.replace("_", " "))


def eval_pairs(
    manifest: Manifest, exp: str, model: str, which: str | None = None
) -> int:
    """Reads the number of test pairs a run stored for its reading.

    Args:
        manifest: The run resolver.
        exp: `r2`, `r3`, `r5`, `r6` or `r7`.
        model: Checkpoint key.
        which: For `r7` a set name (default the test set), for `r5` either
            `transfer` (default) or `in_template`.

    Returns:
        The number of test pairs.

    Raises:
        FileNotFoundError: If the cell has no run.
        ValueError: If the experiment stores no such size.
    """
    run = manifest.run(exp, model)
    if run is None:
        raise FileNotFoundError(f"results/{exp}/{model} has no run")
    data = json.loads((run / "summary.json").read_text(encoding="utf-8"))[
        "data"
    ]
    if exp in ("r2", "r3"):
        return int(data["n_eval_pairs"]["test"])
    if exp == "r7":
        return int(data["sets"][which or "test"]["n_eval_pairs"])
    if exp == "r5":
        return int(data["n_eval_pairs"][which or "transfer"]["test"])
    if exp == "r6":
        return int(data["n_test"])
    raise ValueError(f"no test size for {exp!r}")
