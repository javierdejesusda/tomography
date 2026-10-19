import json
from pathlib import Path

import pytest

from tomography.site import manifest as manifest_module

RESULTS = Path(__file__).resolve().parents[2] / "results"


def _run(
    root: Path,
    exp: str,
    model: str,
    digest: str,
    started: str = "2026-10-01T00:00:00+00:00",
    **config: object,
) -> Path:
    path = root / exp / model / digest
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "meta.json").write_text(
        json.dumps({"started_at": started, "config": config}),
        encoding="utf-8",
    )
    return path


def test_single_run_is_found_and_models_are_listed_in_paper_order(
    tmp_path: Path,
) -> None:
    _run(tmp_path, "r1", "qwen3-4b", "aaa")
    _run(tmp_path, "r1", "gpt2", "bbb")
    found = manifest_module.Manifest(tmp_path)
    assert found.models("r1") == ["gpt2", "qwen3-4b"]
    assert found.run("r1", "gpt2") == tmp_path / "r1" / "gpt2" / "bbb"
    assert found.run("r1", "gpt2-xl") is None
    assert found.models("r9") == []


def test_a_second_hash_without_a_rule_fails_the_build(tmp_path: Path) -> None:
    _run(tmp_path, "r1", "gpt2", "aaa")
    _run(tmp_path, "r1", "gpt2", "bbb")
    with pytest.raises(manifest_module.AmbiguousRunError, match="r1/gpt2"):
        manifest_module.Manifest(tmp_path).run("r1", "gpt2")


def test_p1_takes_the_logistic_run_with_the_most_tasks(tmp_path: Path) -> None:
    _run(tmp_path, "p1", "gpt2", "a1", method="logistic", tasks=["sentiment"])
    full = _run(
        tmp_path,
        "p1",
        "gpt2",
        "a2",
        method="logistic",
        tasks=["sentiment", "truth", "emotion"],
    )
    _run(
        tmp_path,
        "p1",
        "gpt2",
        "a3",
        method="pca_diff",
        tasks=["sentiment", "truth", "emotion"],
    )
    assert manifest_module.Manifest(tmp_path).run("p1", "gpt2") == full
    assert manifest_module.Manifest(tmp_path).run(
        "p1", "gpt2", method="pca_diff"
    ) == (tmp_path / "p1" / "gpt2" / "a3")


def test_p1_with_two_equal_logistic_runs_is_ambiguous(tmp_path: Path) -> None:
    for digest in ("a1", "a2"):
        _run(tmp_path, "p1", "gpt2", digest, method="logistic", tasks=["s"])
    with pytest.raises(manifest_module.AmbiguousRunError):
        manifest_module.Manifest(tmp_path).run("p1", "gpt2")


def test_p9d_is_selected_by_template(tmp_path: Path) -> None:
    few = _run(tmp_path, "p9d", "qwen3-1.7b", "f1", template="fewshot")
    con = _run(tmp_path, "p9d", "qwen3-1.7b", "c1", template="concept")
    found = manifest_module.Manifest(tmp_path)
    assert found.run("p9d", "qwen3-1.7b") == few
    assert found.run("p9d", "qwen3-1.7b", template="concept") == con


def test_rerun_experiments_take_the_latest_started_run(tmp_path: Path) -> None:
    _run(
        tmp_path, "m1", "qwen3-1.7b", "old", started="2026-10-17T10:00:00+00:00"
    )
    new = _run(
        tmp_path, "m1", "qwen3-1.7b", "new", started="2026-10-17T23:00:00+00:00"
    )
    assert manifest_module.Manifest(tmp_path).run("m1", "qwen3-1.7b") == new


def test_every_experiment_and_model_of_the_committed_results_resolves() -> None:
    found = manifest_module.Manifest(RESULTS)
    count = 0
    for exp in found.experiments():
        for model in found.models(exp):
            assert found.run(exp, model) is not None, (exp, model)
            count += 1
    assert count > 100
