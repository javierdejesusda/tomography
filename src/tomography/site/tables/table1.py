"""Table 1: Models.

Every checkpoint that has a run under `results/`, with its role, depth,
width, parameter count and the pinned revision of its weights.
"""

from typing import Any

from tomography.site import params
from tomography.site import store
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.viz import paper


def _layers(config: dict[str, Any]) -> int:
    key = "n_layer" if config["model_type"] == "gpt2" else "num_hidden_layers"
    return int(config[key])


def _width(config: dict[str, Any]) -> int:
    key = "n_embd" if config["model_type"] == "gpt2" else "hidden_size"
    return int(config[key])


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    models = sorted(
        {m for e in manifest.experiments() for m in manifest.models(e)},
        key=store.model_rank,
    )
    rows: list[list[str]] = []
    counts: list[int] = []
    for model in models:
        config = params.model_config(model)
        count = params.param_count(model)
        counts.append(count)
        rows.append(
            [
                paper.display_name(model),
                "post-trained" if store.is_post_trained(model) else "base",
                str(_layers(config)),
                str(_width(config)),
                f"{count:,}",
                str(config["revision"])[:12],
            ]
        )
    numbers = {
        "checkpoints": float(len(models)),
        "smallest": float(min(counts)),
        "largest": float(max(counts)),
    }
    claim = (
        f"The {len(models)} checkpoints range from {params.label(min(counts))}"
        f" to {params.label(max(counts))} parameters, in four GPT-2 sizes and "
        "three Qwen3 sizes with a base and a post-trained model each."
    )
    detail = (
        "Parameters include the embeddings, the positional table of GPT-2 and "
        "every norm, and are computed from the stored model configuration. "
        "Layers are decoder blocks. Revision is the first 12 characters of "
        "the pinned commit of the weights on the Hugging Face Hub."
    )
    return TableOutput(
        claim,
        detail,
        ["Model", "Role", "Layers", "Hidden size", "Parameters", "Revision"],
        rows,
        "llrrrl",
        [],
        numbers,
    )
