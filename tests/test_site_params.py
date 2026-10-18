import json
from pathlib import Path

import pytest

from tomography.model import MODELS
from tomography.site import params

RESULTS = Path(__file__).resolve().parents[1] / "results"
P6 = RESULTS / "p6" / "gpt2" / "25ee6de331d4" / "table.json"


def test_gpt2_counts_match_the_counts_stored_by_p6() -> None:
    table = json.loads(P6.read_text(encoding="utf-8"))["data"]
    columns = table["columns"]
    stored = {
        row[columns.index("model")]: row[columns.index("params")]
        for row in table["rows"]
    }
    assert set(stored) == {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
    for model, count in stored.items():
        assert params.param_count(model) == count, model


def test_every_registered_checkpoint_has_a_config_at_its_pinned_revision() -> (
    None
):
    for key, spec in MODELS.items():
        config = params.model_config(key)
        assert config["revision"] == spec.revision, key
        assert config["repo_id"] == spec.repo_id, key


@pytest.mark.parametrize("key", ["qwen3-0.6b-base", "qwen3-1.7b"])
def test_qwen3_counts_match_an_instantiated_model(key: str) -> None:
    import torch
    from transformers import AutoConfig
    from transformers import AutoModelForCausalLM

    spec = MODELS[key]
    config = AutoConfig.from_pretrained(spec.repo_id, revision=spec.revision)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    expected = sum(p.numel() for p in model.parameters())
    assert params.param_count(key) == expected


def test_counts_grow_with_size_inside_each_family() -> None:
    gpt2 = [params.param_count(k) for k in MODELS if k.startswith("gpt2")]
    assert sorted(gpt2) == sorted(set(gpt2))
    assert params.param_count("qwen3-0.6b") == params.param_count(
        "qwen3-0.6b-base"
    )
    assert (
        params.param_count("qwen3-0.6b")
        < params.param_count("qwen3-1.7b")
        < params.param_count("qwen3-4b")
    )


def test_label_is_a_short_count_with_a_unit() -> None:
    assert params.label(124439808) == "124M"
    assert params.label(1_720_000_000) == "1.72B"


def test_unknown_checkpoint_is_rejected() -> None:
    with pytest.raises(KeyError):
        params.param_count("llama-7b")
