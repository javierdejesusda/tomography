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
    importlib.import_module(module_name)


@pytest.mark.slow
def test_default_sync_gives_working_cuda_torch() -> None:
    import torch

    assert torch.version.cuda is not None
    assert torch.cuda.is_available()
