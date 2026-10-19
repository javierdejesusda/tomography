from pathlib import Path

import pytest

from tomography import model
from tomography.site import params
from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
SPEC = "checkpoints"


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table(SPEC, Manifest(RESULTS))


def test_columns(output: registry.TableOutput) -> None:
    assert output.columns == [
        "Model",
        "Family",
        "Role",
        "Parameters",
        "Layers",
        "Hidden size",
        "Hub ID",
        "Revision",
        "License",
    ]
    assert output.align == "lllrrrlll"
    assert output.row_groups == (0, 4)


def test_one_row_per_checkpoint_found_in_the_results(
    output: registry.TableOutput,
) -> None:
    manifest = Manifest(RESULTS)
    models = sorted(
        {m for e in manifest.experiments() for m in manifest.models(e)}
    )
    assert len(output.rows) == len(models) == 10
    assert {r[0] for r in output.rows} == {
        paper.display_name(m) for m in models
    }


def test_rows_carry_the_stored_architecture_and_revision(
    output: registry.TableOutput,
) -> None:
    by_name = {r[0]: r for r in output.rows}
    xl = by_name["GPT-2 XL"]
    config = params.model_config("gpt2-xl")
    pin = model.MODELS["gpt2-xl"]
    assert xl[1] == "GPT-2"
    assert xl[2] == "GPT-2"
    assert xl[3] == f"{params.param_count('gpt2-xl'):,}"
    assert xl[4] == str(config["n_layer"])
    assert xl[5] == str(config["n_embd"])
    assert xl[6] == pin.repo_id
    assert xl[7] == pin.revision[:12]
    assert xl[8] == pin.license
    post = by_name["Qwen3-4B"]
    assert post[1] == "Qwen3"
    assert post[2] == "post-trained"
    assert post[4] == str(params.model_config("qwen3-4b")["num_hidden_layers"])
    assert post[6] == "Qwen/Qwen3-4B"
    assert by_name["Qwen3-4B-Base"][2] == "base"


def test_the_caption_states_the_revision_and_the_post_training_difference(
    output: registry.TableOutput,
) -> None:
    text = output.claim + " " + output.detail
    assert "pinned commit" in text
    assert "chat template" in text
    assert "thinking disabled" in text
    assert "distillation" in text


def test_the_claim_range_is_the_smallest_and_largest_count(
    output: registry.TableOutput,
) -> None:
    counts = [params.param_count(m) for m in ("gpt2", "qwen3-4b")]
    assert output.numbers["smallest"] == counts[0]
    assert output.numbers["largest"] == counts[1]
    assert output.numbers["checkpoints"] == 10
    assert params.label(counts[0]) in output.claim
    assert params.label(counts[1]) in output.claim
    assert output.claim.endswith(".")
