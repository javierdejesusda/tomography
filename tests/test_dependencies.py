import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "numpy",
        "torch",
        "transformers",
        "datasets",
        "peft",
        "sklearn",
        "matplotlib",
        "plotly",
    ],
)
def test_runtime_dependency_imports_cleanly(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module.__name__ == module_name
