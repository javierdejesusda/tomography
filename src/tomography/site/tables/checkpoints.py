"""Table 1: Models.

Every checkpoint that has a run under `results/`, with its role, parameter
count, layers and hidden size. The Hub repository and the license are in
Table A1; the pinned revisions stay in the code and the methodology document.
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
    models = manifest.checkpoints()
    rows: list[list[str]] = []
    counts: list[int] = []
    for model in models:
        config = params.model_config(model)
        count = params.param_count(model)
        counts.append(count)
        gpt2 = not model.startswith("qwen3")
        if gpt2:
            role = "GPT-2"
        else:
            role = "post-trained" if store.is_post_trained(model) else "base"
        rows.append(
            [
                paper.display_name(model),
                role,
                f"{count:,}",
                str(_layers(config)),
                str(_width(config)),
            ]
        )
    numbers = {
        "checkpoints": float(len(models)),
        "smallest": float(min(counts)),
        "largest": float(max(counts)),
    }
    claim = (
        f"The {len(models)} checkpoints range from {params.label(min(counts))}"
        f" to {params.label(max(counts))} parameters."
    )
    detail = (
        "Counts include the embeddings and every norm; layers are decoder "
        "blocks. Post-trained Qwen3 checkpoints run with their chat template "
        "and thinking disabled."
    )
    return TableOutput(
        claim,
        detail,
        [
            "Model",
            "Role",
            "Parameters",
            "Layers",
            "Hidden size",
        ],
        rows,
        "llrrr",
        [],
        numbers,
        row_groups=(0, sum(not m.startswith("qwen3") for m in models)),
    )
