from pathlib import Path

import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest

RESULTS = Path(__file__).resolve().parents[1] / "results"
ALL = (*registry.FIGURES, *registry.TABLES, *registry.APPENDIX)


def test_the_paper_has_nine_figures_eight_tables_and_six_appendix_figures() -> (
    None
):
    assert len(registry.FIGURES) == 9
    assert len(registry.TABLES) == 8
    assert len(registry.APPENDIX) == 6
    assert [s.number for s in registry.FIGURES] == [
        str(i) for i in range(1, 10)
    ]
    assert [s.number for s in registry.APPENDIX] == [
        f"A{i}" for i in range(1, 7)
    ]


def test_keys_and_modules_are_unique() -> None:
    assert len({s.key for s in ALL}) == len(ALL)
    assert len({s.module for s in ALL}) == len(ALL)


@pytest.mark.parametrize("spec", ALL, ids=[s.key for s in ALL])
def test_every_entry_names_a_module_with_a_build_function(
    spec: registry.Spec,
) -> None:
    assert callable(registry.load(spec).build)


def test_every_source_experiment_exists_in_the_results() -> None:
    manifest = Manifest(RESULTS)
    experiments = set(manifest.experiments())
    for spec in ALL:
        assert set(spec.sources) <= experiments, spec.key
