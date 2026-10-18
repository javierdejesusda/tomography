from pathlib import Path

import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest

RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="Figure A1 is not built yet"
)
def test_figure_a1_builds_from_the_committed_results() -> None:
    output = registry.build_figure(registry.APPENDIX[0], Manifest(RESULTS))
    assert output.claim.endswith(".")


@pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="Figure A2 is not built yet"
)
def test_figure_a2_builds_from_the_committed_results() -> None:
    output = registry.build_figure(registry.APPENDIX[1], Manifest(RESULTS))
    assert output.claim.endswith(".")


@pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="Figure A3 is not built yet"
)
def test_figure_a3_builds_from_the_committed_results() -> None:
    output = registry.build_figure(registry.APPENDIX[2], Manifest(RESULTS))
    assert output.claim.endswith(".")


@pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="Figure A4 is not built yet"
)
def test_figure_a4_builds_from_the_committed_results() -> None:
    output = registry.build_figure(registry.APPENDIX[3], Manifest(RESULTS))
    assert output.claim.endswith(".")


@pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="Figure A5 is not built yet"
)
def test_figure_a5_builds_from_the_committed_results() -> None:
    output = registry.build_figure(registry.APPENDIX[4], Manifest(RESULTS))
    assert output.claim.endswith(".")


@pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="Figure A6 is not built yet"
)
def test_figure_a6_builds_from_the_committed_results() -> None:
    output = registry.build_figure(registry.APPENDIX[5], Manifest(RESULTS))
    assert output.claim.endswith(".")
