"""Checks that the WP2 figures use each measure shape for its own family."""

from pathlib import Path
import runpy
from typing import Any

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import paper

_L = runpy.run_path(str(Path(__file__).with_name("_legends.py")))
RESULTS = Path(__file__).resolve().parents[1] / "results"
M = paper.MEASURE_MARKERS
DIRECTION = {M["direction"]}
RANDOM = {M["random"]}
TEXT = set(M["text"])
OWN = {
    "steering": DIRECTION,
    "knowledge_reading": DIRECTION
    | TEXT
    | {M["popularity"], M["probe"], M["answer_probability"]},
    "knowledge_steering": DIRECTION | RANDOM,
    "approximate": DIRECTION | RANDOM,
    "base_post": DIRECTION,
    "a_knowledge_matrix": DIRECTION | TEXT | RANDOM,
    "a_dose_runs": DIRECTION,
    "a_dose_types": DIRECTION,
    "a_links": set(),
}
MEASURES: set[str] = {
    m for v in M.values() for m in ((v,) if isinstance(v, str) else v)
}


@pytest.fixture(scope="module")
def built() -> dict[str, registry.FigureOutput]:
    manifest = Manifest(RESULTS)
    return {k: registry.build_figure(k, manifest) for k in OWN}


def _markers(fig: Figure) -> set[Any]:
    lines = [ln for ax in fig.axes for ln in ax.get_lines()]
    for leg in [*fig.legends, *(a.get_legend() for a in fig.axes)]:
        if leg is not None:
            lines.extend(h for h in leg.legend_handles if isinstance(h, Line2D))
    return {
        str(ln.get_marker())
        for ln in lines
        if str(ln.get_marker()) not in ("None", "", " ")
    }


@pytest.mark.parametrize("key", list(OWN))
def test_a_measure_shape_appears_only_for_its_own_family(
    built: dict[str, registry.FigureOutput], key: str
) -> None:
    drawn = _markers(built[key].figure)
    foreign = (drawn & MEASURES) - OWN[key]
    assert foreign == set(), (key, foreign)


@pytest.mark.parametrize("key", list(OWN))
def test_no_legend_symbol_means_two_things(
    built: dict[str, registry.FigureOutput], key: str
) -> None:
    assert _L["marker_clashes"](built[key].figure) == []


def test_known_and_unknown_questions_are_a_set_pair_not_a_measure() -> None:
    sets = paper.QUESTION_SET_MARKERS
    assert sets["unknown"] == M["direction"]
    assert sets["known"] in paper.CATEGORY_MARKERS


@pytest.mark.parametrize(
    "key", ["knowledge_steering", "a_dose_runs", "a_dose_types"]
)
def test_known_questions_use_the_question_set_marker(
    built: dict[str, registry.FigureOutput], key: str
) -> None:
    drawn = _markers(built[key].figure)
    assert paper.QUESTION_SET_MARKERS["known"] in drawn
    assert not drawn & TEXT
