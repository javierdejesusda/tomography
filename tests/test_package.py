import importlib.resources
import pathlib
import tomllib

import tomography

PYPROJECT_PATH = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_matches_pyproject() -> None:
    with PYPROJECT_PATH.open("rb") as pyproject_file:
        declared_version = tomllib.load(pyproject_file)["project"]["version"]

    assert tomography.__version__ == declared_version


def test_package_ships_py_typed_marker() -> None:
    marker = importlib.resources.files("tomography").joinpath("py.typed")

    assert marker.is_file()
