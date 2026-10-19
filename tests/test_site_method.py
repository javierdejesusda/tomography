from pathlib import Path

from matplotlib.patches import FancyArrowPatch
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
        r"h'_\ell = h_\ell + c\,r_\ell\,v_\ell" in t for t in _texts(output)
    )
    assert any(
        "3 consecutive layers" in " ".join(t.split()) for t in _texts(output)
    )


def test_the_five_stages_are_named_in_order(
    output: registry.FigureOutput,
) -> None:
    joined = " ".join(" ".join(_texts(output)).split())
    stages = [
        "pair",
        "residual stream at layer",
        "find a unit direction",
        "read",
        "steer",
    ]
    positions = [joined.index(s) for s in stages]
    assert positions == sorted(positions)
    for text in (
        "PCA of paired differences (paper)",
        "logistic regression",
        "random unit direction",
        "last token",
    ):
        assert text in joined


def test_reading_is_a_pairwise_comparison_not_a_threshold(
    output: registry.FigureOutput,
) -> None:
    joined = " ".join(_texts(output)).lower()
    assert "threshold" not in joined
    assert "projects higher" in joined
    assert "dashed box" not in " ".join([output.claim, output.detail]).lower()


def test_no_text_has_an_underscore_and_the_guards_are_green(
    output: registry.FigureOutput,
) -> None:
    assert all("_" not in t or "$" in t for t in _texts(output))
    assert checks.guard_problems(output.figure) == []


def test_the_arrows_are_straight_and_never_cross(
    output: registry.FigureOutput,
) -> None:
    arrows = [
        p
        for ax in output.figure.axes
        for p in ax.patches
        if isinstance(p, FancyArrowPatch)
    ]
    assert len(arrows) == 4
    boxes = [a.get_path().get_extents() for a in arrows]
    for i, a in enumerate(boxes):
        assert a.height < 0.03  # a head, not a slope
        for b in boxes[i + 1 :]:
            assert not a.overlaps(b)


def test_the_schematic_has_no_rows_and_no_numbers(
    output: registry.FigureOutput,
) -> None:
    assert output.csv == ""
    assert output.numbers == {}
    assert not any(ch.isdigit() for ch in output.claim)


def test_the_caption_describes_the_pairwise_read_and_the_baseline(
    output: registry.FigureOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    assert "positive member projects higher" in text
    assert "constant score sits at 0.5" in text
    assert "oriented on the training pairs" in text
    assert "illustrative" in text


def test_the_figure_is_5_5_by_2_1_inches(
    output: registry.FigureOutput,
) -> None:
    assert output.figure.get_figwidth() == pytest.approx(paper.FULL_WIDTH)
    assert output.figure.get_figheight() == pytest.approx(2.1)


def test_no_layout_faults(output: registry.FigureOutput) -> None:
    assert checks.problems(output.figure) == []
