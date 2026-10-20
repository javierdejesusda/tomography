"""Checkpoint keys of the results and their order on the page."""

MODEL_ORDER = (
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "qwen3-0.6b-base",
    "qwen3-0.6b",
    "qwen3-1.7b-base",
    "qwen3-1.7b",
    "qwen3-4b-base",
    "qwen3-4b",
)


def is_post_trained(model: str) -> bool:
    """Tells whether a checkpoint key names a post-trained model.

    Args:
        model: Checkpoint key such as `qwen3-1.7b` or `gpt2-xl`.

    Returns:
        True for the Qwen3 post-trained checkpoints.
    """
    return model.startswith("qwen3") and not model.endswith("-base")


def model_rank(model: str) -> tuple[int, str]:
    """Sorts checkpoint keys in paper order, unknown keys last.

    Args:
        model: Checkpoint key.

    Returns:
        A sort key.
    """
    if model in MODEL_ORDER:
        return MODEL_ORDER.index(model), model
    return len(MODEL_ORDER), model
