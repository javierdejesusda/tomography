import importlib
import importlib.util

from tomography.site import registry

FIGURE_KEYS = (
    "teaser",
    "method",
    "reading",
    "finders",
    "truth",
    "knowledge_reading",
    "steering",
    "base_post",
)
APPENDIX_FIGURE_KEYS = (
    "a_reading_curves",
    "wording",
    "a_truth_cities",
    "knowledge_steering",
    "a_knowledge_matrix",
    "a_knowledge_geometry",
    "a_knowledge_projections",
    "approximate",
)
TABLE_KEYS = ("checkpoints",)
APPENDIX_TABLE_KEYS = (
    "assets",
    "notation",
    "design",
    "reading",
    "finder_baselines",
    "steering_baselines",
    "detection_baselines",
    "approximate",
)
REMOVED = (
    ("figure", "a_layers"),
    ("figure", "a_finders_gpt2"),
    ("figure", "a_finders_qwen3"),
    ("figure", "a_wording"),
    ("figure", "a_dose_runs"),
    ("figure", "a_dose_types"),
    ("figure", "a_links"),
    ("table", "finders_all"),
    ("table", "reading_full"),
    ("table", "sizes"),
    ("table", "prompts"),
    ("table", "wordings"),
    ("table", "truth"),
    ("table", "truth_sets"),
    ("table", "steering"),
    ("table", "knowledge"),
    ("table", "knowledge_projections"),
    ("table", "base_post"),
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


def test_the_items_that_left_the_page_have_no_module() -> None:
    for kind, key in REMOVED:
        name = registry.module_name(kind, key)
        assert importlib.util.find_spec(name) is None, name


def test_the_main_text_has_eight_figures_and_one_table() -> None:
    assert len(registry.MAIN_FIGURES) == 8
    assert len(registry.MAIN_TABLES) == 1
    assert len(registry.APPENDIX_FIGURES) == 8
    assert len(registry.APPENDIX_TABLES) == 8
