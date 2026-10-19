from pathlib import Path
import re

import pytest

SITE = Path(__file__).resolve().parents[1] / "src" / "tomography" / "site"
MODULES = (
    "figs/knowledge_reading.py",
    "figs/approximate.py",
    "tables/approximate.py",
)


@pytest.mark.parametrize("module", MODULES)
def test_caption_numbers_use_the_rounded_zero_helper(module: str) -> None:
    source = (SITE / module).read_text(encoding="utf-8")
    assert not re.findall(r"\{[^{}]*:\.\d+f\}", source), module
