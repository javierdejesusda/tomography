from pathlib import Path

import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest

RESULTS = Path(__file__).resolve().parents[1] / "results"
SPEC = registry.FIGURES[2]


@pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="Figure 3 is not built yet"
)
def test_figure_3_builds_from_the_committed_results() -> None:
    output = registry.build_figure(SPEC, Manifest(RESULTS))
    assert output.claim.endswith(".")
