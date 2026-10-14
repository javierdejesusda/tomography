import importlib
import importlib.resources
import tomllib

import pytest

import tomography


def test_version_matches_pyproject(pytestconfig: pytest.Config) -> None:
    pyproject = pytestconfig.rootpath / "pyproject.toml"
    with pyproject.open("rb") as pyproject_file:
        declared_version = tomllib.load(pyproject_file)["project"]["version"]

    assert tomography.__version__ == declared_version


def test_package_has_py_typed_marker() -> None:
    marker = importlib.resources.files("tomography").joinpath("py.typed")

    assert marker.is_file()


@pytest.mark.parametrize(
    "subpackage",
    ["datasets", "reading", "control", "experiments", "viz"],
)
def test_subpackage_imports(subpackage: str) -> None:
    importlib.import_module(f"tomography.{subpackage}")
