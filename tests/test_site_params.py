import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from tomography.model import MODELS
from tomography.site import params
from tomography.site.manifest import Manifest
from tomography.site.store import MODEL_ORDER
from tomography.viz import paper

_RESULTS = runpy.run_path(str(Path(__file__).with_name("_results.py")))
npz = _RESULTS["npz"]
run = _RESULTS["run"]

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


SETS = {
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


def test_report_pairs_are_the_verified_sizes() -> None:
    assert params.REPORT_PAIRS == {
        "sentiment": 214,
        "truth": 59,
        "utility": 256,
    }


@pytest.mark.parametrize("model", MODEL_ORDER)
def test_every_p3_report_accuracy_is_a_multiple_of_one_over_2n(
    model: str,
) -> None:
    path = run("p3", model)
    for task, n in params.REPORT_PAIRS.items():
        accuracy = npz(path, f"sizes_{task}.npz")["accuracy"]
        report = accuracy[:, 1]
        scaled = report * 2 * n
        assert np.allclose(scaled, np.round(scaled), atol=1e-6), (model, task)


@pytest.mark.parametrize("model", MODEL_ORDER)
def test_every_p4_per_seed_value_is_a_multiple_of_one_over_2n(
    model: str,
) -> None:
    path = run("p4", model)
    for task, n in params.REPORT_PAIRS.items():
        per_seed = npz(path, f"curves_{task}_methods.npz")["per_seed"]
        scaled = per_seed * 2 * n
        assert np.allclose(scaled, np.round(scaled), atol=1e-6), (model, task)


@pytest.mark.parametrize("model", MODEL_ORDER)
def test_the_truth_select_split_has_53_pairs(model: str) -> None:
    accuracy = npz(run("p3", model), "sizes_truth.npz")["accuracy"]
    scaled = accuracy[:, 0] * 2 * 53
    assert np.allclose(scaled, np.round(scaled), atol=1e-6)


def test_eval_pairs_read_the_sizes_stored_with_each_run() -> None:
    manifest = Manifest(RESULTS)
    assert params.eval_pairs(manifest, "r2", "gpt2") == 4808
    assert params.eval_pairs(manifest, "r3", "gpt2") == 983
    assert params.eval_pairs(manifest, "r7", "gpt2") == 59
    assert params.eval_pairs(manifest, "r7", "gpt2", "cities") == 748
    assert params.eval_pairs(manifest, "r7", "gpt2", "sp_en_trans") == 177
    assert params.eval_pairs(manifest, "r5", "gpt2") == 59
    assert params.eval_pairs(manifest, "r5", "gpt2", "in_template") == 72
    assert params.eval_pairs(manifest, "r6", "gpt2") == 173


def test_display_names_follow_table_one_and_the_page_rules() -> None:
    for model in MODEL_ORDER:
        assert params.display(model) == paper.display_name(model)
    assert params.display("pca_diff") == "PCA of paired differences (paper)"
    assert params.display("lat") == "knowledge direction"
    assert params.display("no_template") == "no template"
    assert params.display("nq_open") == "NQ-open"
    for key, name in SETS.items():
        assert params.display(key) == name


def test_no_display_name_has_an_underscore_or_a_dash_character() -> None:
    banned = {"_", chr(0x2014), chr(0x2013)}
    assert not any(ch in v for v in params.DISPLAY.values() for ch in banned)
