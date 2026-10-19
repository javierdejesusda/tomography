import importlib
import importlib.util

from tomography.site import registry

FIGURE_KEYS = (
    "teaser",
    "method",
    "finders",
    "reading",
    "knowledge_reading",
    "knowledge_steering",
    "approximate",
    "truth",
    "steering",
    "base_post",
)
APPENDIX_FIGURE_KEYS = (
    "a_reading_curves",
    "a_layers",
    "a_finders_gpt2",
    "a_finders_qwen3",
    "wording",
    "a_wording",
    "a_truth_cities",
    "a_knowledge_matrix",
    "a_dose_runs",
    "a_dose_types",
    "a_links",
)
TABLE_KEYS = ("checkpoints", "reading")
APPENDIX_TABLE_KEYS = (
    "assets",
    "notation",
    "finders_all",
    "reading_full",
    "sizes",
    "prompts",
    "wordings",
    "truth",
    "truth_sets",
    "steering",
    "knowledge",
    "approximate",
    "base_post",
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
        ("figure", "a_reading_curves"),
        ("figure", "a_layers"),
        ("figure", "a_finders_gpt2"),
        ("figure", "a_wording"),
        ("figure", "a_truth_cities"),
        ("figure", "a_knowledge_matrix"),
        ("figure", "a_dose_types"),
        ("figure", "a_links"),
        ("table", "checkpoints"),
        ("table", "reading"),
        ("table", "knowledge"),
        ("table", "approximate"),
        ("table", "steering"),
        ("table", "truth"),
        ("table", "base_post"),
        ("table", "reading_full"),
    }
    assert ported.isdisjoint(registry.missing())


def test_every_registered_module_exists() -> None:
    assert registry.missing() == []


def test_the_cut_entries_have_no_module() -> None:
    for name in (
        "tomography.site.figs.a_reading_heat",
        "tomography.site.tables.finders",
    ):
        assert importlib.util.find_spec(name) is None
    assert "finders" not in registry.MAIN_TABLES
    assert "a_reading_heat" not in registry.APPENDIX_FIGURES


def test_the_registry_has_ten_main_figures_and_two_main_tables() -> None:
    assert len(registry.MAIN_FIGURES) == 10
    assert len(registry.MAIN_TABLES) == 2
