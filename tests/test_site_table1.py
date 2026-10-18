from pathlib import Path

import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest

RESULTS = Path(__file__).resolve().parents[1] / "results"
SPEC = registry.TABLES[0]


@pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="Table 1 is not built yet"
)
def test_table_1_builds_from_the_committed_results() -> None:
    output = registry.build_table(SPEC, Manifest(RESULTS))
    assert output.claim.endswith(".")
