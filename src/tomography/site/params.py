"""Parameter counts of the checkpoints, computed from their model configs.

`data/model_configs.json` stores the architecture fields of each pinned
checkpoint exactly as its `config.json` at that revision gives them. Counts
include the embeddings, the positional table of GPT-2 and every norm.
"""

import json
from pathlib import Path
from typing import Any

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
