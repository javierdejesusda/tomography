from pathlib import Path

import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest

RESULTS = Path(__file__).resolve().parents[1] / "results"
SPEC = registry.TABLES[5]


@pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="Table 6 is not built yet"
)
def test_table_6_builds_from_the_committed_results() -> None:
    output = registry.build_table(SPEC, Manifest(RESULTS))
    assert output.claim.endswith(".")
