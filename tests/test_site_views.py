from pathlib import Path
import runpy

import pytest

from tomography.site import views
from tomography.site.manifest import Manifest

_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
ROOT, models = _R["ROOT"], _R["models"]


def test_a_run_that_follows_another_reading_run_is_refused() -> None:
    class Moved(Manifest):
        def run(self, exp: str, model: str, **select: str) -> Path | None:
            other = "gpt2-large" if exp == "p9b" else model
            return super().run(exp, other, **select)

    with pytest.raises(ValueError, match="follows"):
        views.load(Moved(ROOT), "gpt2-xl")


def test_a_checkpoint_without_a_run_has_no_views() -> None:
    assert "qwen3-4b" not in models("v1")
    assert views.load(Manifest(ROOT), "qwen3-4b") is None
