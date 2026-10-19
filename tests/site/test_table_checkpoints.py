from pathlib import Path

import pytest

from tomography.site import params
from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[2] / "results"
SPEC = "checkpoints"


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table(SPEC, Manifest(RESULTS))


def test_columns(output: registry.TableOutput) -> None:
    assert output.columns == [
        "Model",
        "Role",
        "Layers",
        "Hidden size",
        "Parameters",
        "Revision",
    ]
    assert output.align == "llrrrl"


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
    assert xl[1] == "base"
    assert xl[2] == str(config["n_layer"])
    assert xl[3] == str(config["n_embd"])
    assert xl[4] == f"{params.param_count('gpt2-xl'):,}"
    assert xl[5] == config["revision"][:12]
    post = by_name["Qwen3-4B"]
    assert post[1] == "post-trained"
    assert by_name["Qwen3-4B-Base"][1] == "base"
    assert post[2] == str(params.model_config("qwen3-4b")["num_hidden_layers"])


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
