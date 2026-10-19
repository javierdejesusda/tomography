import importlib

import pytest

from tomography.site import registry

FIGURE_KEYS = (
    "teaser",
    "method",
    "finders",
    "reading",
    "wording",
    "truth",
    "steering",
    "knowledge_reading",
    "knowledge_steering",
    "approximate",
    "base_post",
)
APPENDIX_FIGURE_KEYS = (
    "a_layers",
    "a_finders_gpt2",
    "a_finders_qwen3",
    "a_wording",
    "a_knowledge_matrix",
    "a_dose_runs",
    "a_dose_types",
    "a_links",
)
TABLE_KEYS = (
    "checkpoints",
    "notation",
    "finders",
    "reading",
    "truth",
    "steering",
    "knowledge",
    "approximate",
)
APPENDIX_TABLE_KEYS = (
    "assets",
    "finders_all",
    "reading_full",
    "truth_sets",
    "base_post",
    "wordings",
    "sizes",
    "prompts",
)


def test_the_paper_has_the_final_figures_and_tables_in_order() -> None:
    assert registry.MAIN_FIGURES == FIGURE_KEYS
    assert registry.APPENDIX_FIGURES == APPENDIX_FIGURE_KEYS
    assert registry.MAIN_TABLES == TABLE_KEYS
    assert registry.APPENDIX_TABLES == APPENDIX_TABLE_KEYS


def test_keys_are_unique_within_a_kind() -> None:
    for keys in (
        (*registry.MAIN_FIGURES, *registry.APPENDIX_FIGURES),
        (*registry.MAIN_TABLES, *registry.APPENDIX_TABLES),
    ):
        assert len(set(keys)) == len(keys)


def test_modules_resolve_from_the_key_alone() -> None:
    assert registry.module_name("figure", "finders") == (
        "tomography.site.figs.finders"
    )
    assert registry.module_name("table", "finders") == (
        "tomography.site.tables.finders"
    )


def test_a_module_that_exists_has_a_build_function() -> None:
    found = [
        (kind, key)
        for kind, keys in (
            ("figure", (*registry.MAIN_FIGURES, *registry.APPENDIX_FIGURES)),
            ("table", (*registry.MAIN_TABLES, *registry.APPENDIX_TABLES)),
        )
        for key in keys
        if (kind, key) not in registry.missing()
    ]
    assert found
    for kind, key in found:
        module = importlib.import_module(registry.module_name(kind, key))
        assert callable(module.build), (kind, key)


def test_the_modules_ported_from_the_first_draft_exist() -> None:
    ported = {
        ("figure", "method"),
        ("figure", "reading"),
        ("figure", "finders"),
        ("figure", "wording"),
        ("figure", "knowledge_reading"),
        ("figure", "knowledge_steering"),
        ("figure", "steering"),
        ("figure", "approximate"),
        ("figure", "truth"),
        ("figure", "a_layers"),
        ("figure", "a_finders_gpt2"),
        ("figure", "a_wording"),
        ("figure", "a_knowledge_matrix"),
        ("figure", "a_dose_types"),
        ("figure", "a_links"),
        ("table", "checkpoints"),
        ("table", "reading"),
        ("table", "finders"),
        ("table", "knowledge"),
        ("table", "approximate"),
        ("table", "steering"),
        ("table", "truth"),
        ("table", "base_post"),
        ("table", "reading_full"),
    }
    assert ported.isdisjoint(registry.missing())


def test_missing_lists_unbuilt_entries_with_their_module_path() -> None:
    assert ("table", "notation") in registry.missing()


@pytest.mark.xfail(
    strict=True,
    reason="the figure and table modules of WP1, WP2 and WP3 are not merged",
)
def test_every_registered_module_exists() -> None:
    assert registry.missing() == []
