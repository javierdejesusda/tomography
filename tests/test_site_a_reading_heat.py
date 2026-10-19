from pathlib import Path
import runpy

from matplotlib.image import AxesImage
import numpy as np
import pytest

from tomography.site import registry
from tomography.site.figs import reading as reading_fig
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

_RESULTS = runpy.run_path(str(Path(__file__).with_name("_results.py")))
models = _RESULTS["models"]

RESULTS = Path(__file__).resolve().parents[1] / "results"
CONCEPTS = {
    "sentiment": "p4",
    "utility": "r2",
    "morality": "r3",
    "truth": "r7",
}


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure("a_reading_heat", Manifest(RESULTS))


def test_the_maps_are_registered_in_the_appendix_after_the_layer_curves() -> (
    None
):
    appendix = registry.APPENDIX_FIGURES
    assert appendix.index("a_reading_heat") == appendix.index("a_layers") + 1
    assert ("figure", "a_reading_heat") not in registry.missing()


def test_one_heat_row_per_checkpoint_and_concept(
    output: registry.FigureOutput,
) -> None:
    images = output.figure.findobj(AxesImage)
    assert len(images) == sum(len(models(e)) for e in CONCEPTS.values())
    for image in images:
        assert image.get_clim() == pytest.approx((0.0, 0.6))


def test_every_row_is_the_reader_minus_random_curve(
    output: registry.FigureOutput,
) -> None:
    cells = {
        c: reading_fig.cells_of(Manifest(RESULTS), c, e)
        for c, e in CONCEPTS.items()
    }
    drawn = sorted(
        round(float(np.asarray(im.get_array()).sum()), 6)
        for im in output.figure.findobj(AxesImage)
    )
    want = sorted(
        round(float(cell.curve.selectivity.sum()), 6)
        for per in cells.values()
        for cell in per.values()
    )
    assert drawn == pytest.approx(want)


def test_the_caption_points_to_the_curves_of_figure_4(
    output: registry.FigureOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    assert "[[fig:reading]]" in text
    assert "black tick" in text
    assert "colour bar" in text


def test_the_guards_are_green(output: registry.FigureOutput) -> None:
    fig = output.figure
    assert checks.guard_problems(fig, appendix=True) == []
    assert checks.problems(fig) == []
    assert fig.get_size_inches()[0] == pytest.approx(paper.FULL_WIDTH)


def test_the_onset_key_uses_the_words_of_figure_4(
    output: registry.FigureOutput,
) -> None:
    (legend,) = output.figure.legends
    texts = [t.get_text() for t in legend.get_texts()]
    assert "onset (earliest layer above 0.1)" in texts
