import json
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.text import Text
import numpy as np
import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[2] / "results"
SPEC = "a_wording"
DROPPED = "no_trailing_space"


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _matrix(model: str) -> np.ndarray:
    run = Manifest(RESULTS).run("p1", model)
    assert run is not None
    labels = json.loads((run / "transfer_sentiment.json").read_text("utf-8"))[
        "data"
    ]
    with np.load(run / "transfer_sentiment.npz") as npz:
        values = np.asarray(npz["values"], dtype=float)
    keep = [i for i, v in enumerate(labels["rows"]) if v != DROPPED]
    return values[np.ix_(keep, keep)]


def _cells(ax: Axes) -> list[Text]:
    return [t for t in ax.texts if not t.get_text().startswith("$")]


def _models() -> list[str]:
    manifest = Manifest(RESULTS)
    return [m for m in manifest.models("p1") if manifest.run("p1", m)]


def test_one_annotated_matrix_per_logistic_p1_model(
    output: registry.FigureOutput,
) -> None:
    matrices = [ax for ax in output.figure.axes if ax.images]
    assert len(matrices) == len(_models())
    assert output.figure.get_figwidth() == pytest.approx(paper.FULL_WIDTH)
    for ax in matrices:
        assert ax.images[0].get_array().shape == (16, 16)
        assert len(_cells(ax)) == 256


def test_every_cell_is_annotated_with_its_stored_value(
    output: registry.FigureOutput,
) -> None:
    matrices = [ax for ax in output.figure.axes if ax.images]
    for ax, model in zip(matrices, _models(), strict=True):
        values = _matrix(model)
        shown = np.array(
            [
                float("1" if t.get_text() == "1." else "0" + t.get_text())
                for t in _cells(ax)
            ]
        ).reshape(16, 16)
        assert shown == pytest.approx(values, abs=0.005 + 1e-9)


def test_the_claim_numbers_match_the_stored_matrices(
    output: registry.FigureOutput,
) -> None:
    lows = [float(_matrix(m).min()) for m in _models()]
    assert output.numbers["lowest_min"] == pytest.approx(min(lows))
    assert output.numbers["lowest_max"] == pytest.approx(max(lows))
    assert f"{min(lows):.3f}" in output.claim
    assert f"{max(lows):.3f}" in output.claim
    assert min(lows) > 0.5
    assert output.claim.endswith(".")


def test_no_layout_faults(output: registry.FigureOutput) -> None:
    assert checks.problems(output.figure) == []
