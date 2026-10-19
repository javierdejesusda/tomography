import json
import math
from pathlib import Path
import runpy
from typing import Any

from matplotlib.image import AxesImage
import pytest

from tomography.site import registry
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

_RESULTS = runpy.run_path(str(Path(__file__).with_name("_results.py")))
data = _RESULTS["data"]
models = _RESULTS["models"]
run = _RESULTS["run"]

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODELS = models("p1")
ROLES = ("gpt2", "base", "post")


def _role(model: str) -> str:
    if not model.startswith("qwen3"):
        return "gpt2"
    return "base" if model.endswith("-base") else "post"


def _rows(model: str) -> list[dict[str, Any]]:
    table = data(run("p1", model), "table.json")
    return [r for r in table if r["task"] == "sentiment"]


def _n_eval(model: str) -> int:
    config = json.loads((run("p1", model) / "config.json").read_text("utf-8"))
    return int(config["n_eval_pairs"])


def _span(model: str) -> float:
    values = [r["from_paper"] for r in _rows(model)]
    return max(values) - min(values)


def _noise(model: str) -> float:
    rows = _rows(model)
    p = sum(r["from_paper"] for r in rows) / len(rows)
    return stats.expected_range(len(rows)) * math.sqrt(
        p * (1 - p) / _n_eval(model)
    )


def _bare(model: str) -> dict[str, Any]:
    (row,) = [r for r in _rows(model) if r["variant"] == "bare"]
    return row


def _paper(model: str) -> dict[str, Any]:
    (row,) = [r for r in _rows(model) if r["variant"] == "paper"]
    return row


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure("wording", Manifest(RESULTS))


def test_the_csv_holds_every_wording_row_of_every_checkpoint(
    output: registry.FigureOutput,
) -> None:
    lines = output.csv.strip().splitlines()
    head = lines[0].split(",")
    assert head == [
        "model",
        "role",
        "variant",
        "kind",
        "family",
        "from_paper",
        "to_paper",
        "cosine_paper",
        "random",
        "stability",
    ]
    assert len(lines) - 1 == sum(len(_rows(m)) for m in MODELS)
    for line in lines[1:]:
        rec = dict(zip(head, line.split(","), strict=True))
        (row,) = [
            r for r in _rows(rec["model"]) if r["variant"] == rec["variant"]
        ]
        for column in ("from_paper", "to_paper", "cosine_paper", "random"):
            assert float(rec[column]) == pytest.approx(row[column], abs=1e-9)
        assert rec["role"] == _role(rec["model"])


def test_spans_are_per_role_and_noise_comes_from_the_expected_range(
    output: registry.FigureOutput,
) -> None:
    for role in ROLES:
        spans = [_span(m) for m in MODELS if _role(m) == role]
        if not spans:
            continue
        assert output.numbers[f"span_{role}_min"] == pytest.approx(min(spans))
        assert output.numbers[f"span_{role}_max"] == pytest.approx(max(spans))
    assert output.numbers["noise_max"] == pytest.approx(
        max(_noise(m) for m in MODELS)
    )
    assert output.numbers["n_checkpoints"] == len(MODELS)


def test_the_bare_wording_numbers_are_recomputed(
    output: registry.FigureOutput,
) -> None:
    cos = [_bare(m)["cosine_paper"] for m in MODELS]
    reads = [_bare(m)["to_paper"] for m in MODELS]
    random = [_paper(m)["random"] for m in MODELS]
    assert output.numbers["bare_cos_min"] == pytest.approx(min(cos))
    assert output.numbers["bare_cos_max"] == pytest.approx(max(cos))
    assert output.numbers["bare_to_paper_min"] == pytest.approx(min(reads))
    assert output.numbers["bare_to_paper_max"] == pytest.approx(max(reads))
    assert output.numbers["bare_random_min"] == pytest.approx(min(random))
    assert output.numbers["bare_random_max"] == pytest.approx(max(random))


def test_the_plan_values_hold_for_the_eight_checkpoints_it_checked() -> None:
    spans = {
        "gpt2": 0.061,
        "gpt2-medium": 0.113,
        "gpt2-large": 0.136,
        "gpt2-xl": 0.204,
        "qwen3-0.6b-base": 0.249,
        "qwen3-0.6b": 0.180,
        "qwen3-1.7b-base": 0.086,
        "qwen3-1.7b": 0.181,
    }
    cos = {
        "gpt2": 0.179,
        "gpt2-medium": 0.279,
        "gpt2-large": 0.274,
        "gpt2-xl": 0.276,
        "qwen3-0.6b-base": 0.110,
        "qwen3-0.6b": 0.075,
        "qwen3-1.7b-base": 0.137,
        "qwen3-1.7b": 0.121,
    }
    for model, span in spans.items():
        assert round(_span(model), 3) == pytest.approx(span)
        assert round(_bare(model)["cosine_paper"], 3) == pytest.approx(
            cos[model]
        )
    assert stats.expected_range(16) == pytest.approx(3.532, abs=1e-3)


def test_the_caption_quotes_the_spans_the_noise_and_the_bare_numbers(
    output: registry.FigureOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    n = output.numbers
    for role in ROLES:
        if f"span_{role}_min" in n:
            assert f"{n[f'span_{role}_min']:.3f}" in text
            assert f"{n[f'span_{role}_max']:.3f}" in text
    for name in (
        "noise_max",
        "bare_cos_min",
        "bare_cos_max",
        "bare_to_paper_min",
        "bare_to_paper_max",
        "bare_random_min",
        "bare_random_max",
    ):
        assert f"{n[name]:.3f}" in text, name
    assert f"on all {len(MODELS)} checkpoints" in text
    assert "Figure A4" in text
    assert "Table A6" in text


def test_the_figure_has_a_strip_and_a_scatter_and_no_heat_map(
    output: registry.FigureOutput,
) -> None:
    fig = output.figure
    assert len(fig.axes) == 2
    assert fig.get_size_inches()[0] == pytest.approx(paper.FULL_WIDTH)
    assert fig.get_size_inches()[1] == pytest.approx(2.8)
    assert not fig.findobj(AxesImage)


def test_the_guards_are_green(output: registry.FigureOutput) -> None:
    assert checks.guard_problems(output.figure) == []
    assert checks.problems(output.figure) == []


def test_the_bare_reader_is_compared_with_the_random_direction_per_checkpoint(
    output: registry.FigureOutput,
) -> None:
    above = sum(_bare(m)["to_paper"] > _paper(m)["random"] for m in MODELS)
    assert output.numbers["bare_above"] == above
    assert above == len(MODELS)
