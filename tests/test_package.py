import importlib.metadata
import importlib.resources

import tomography


def test_version_matches_installed_distribution_metadata() -> None:
    assert tomography.__version__ == importlib.metadata.version("tomography")


def test_package_ships_py_typed_marker() -> None:
    marker = importlib.resources.files("tomography").joinpath("py.typed")

    assert marker.is_file()
