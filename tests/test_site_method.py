from pathlib import Path

from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text
import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

ROOT = Path(__file__).resolve().parents[1]
SPEC = "method"


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(ROOT / "results"))


def _texts(output: registry.FigureOutput) -> list[str]:
    return [t.get_text() for t in output.figure.findobj(Text) if t.get_text()]


def test_the_steering_equation_matches_the_operator_and_coefficient_code() -> (
    None
):
    operators = (ROOT / "src/tomography/control/operators.py").read_text(
        "utf-8"
    )
    controllers = (ROOT / "src/tomography/control/controllers.py").read_text(
        "utf-8"
    )
    assert '`"linear"`: `h + c v`' in operators
    assert "return self.coeff * self.norms[layer]" in controllers
    assert "Unit-norm directions" in controllers


def test_the_schematic_draws_the_equation_the_code_implements(
    output: registry.FigureOutput,
) -> None:
    assert any(
        "h'_\\ell = h_\\ell + c\\,r_\\ell\\,v_\\ell" in t
        for t in _texts(output)
    )
    assert any("unit norm" in t for t in _texts(output))


def test_the_schematic_has_boxes_for_every_stage_and_the_random_baseline(
    output: registry.FigureOutput,
) -> None:
    boxes = [
        p
        for ax in output.figure.axes
        for p in ax.patches
        if isinstance(p, FancyBboxPatch)
    ]
    assert len(boxes) == 6
    assert any(p.get_linestyle() == "--" for p in boxes)


def test_the_claim_has_no_numbers_and_is_one_sentence(
    output: registry.FigureOutput,
) -> None:
    assert output.numbers == {}
    assert output.claim.endswith(".")
    assert not any(ch.isdigit() for ch in output.claim)


def test_the_figure_is_text_width(output: registry.FigureOutput) -> None:
    assert output.figure.get_figwidth() == pytest.approx(paper.FULL_WIDTH)


def test_no_layout_faults(output: registry.FigureOutput) -> None:
    assert checks.problems(output.figure) == []
