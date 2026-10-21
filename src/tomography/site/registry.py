"""The list of figures and tables of the results paper and the site.

An entry is a key; the module that builds it is `figs.<key>` or
`tables.<key>` under `tomography.site`. A figure module defines
`build(manifest) -> FigureOutput` and a table module
`build(manifest) -> TableOutput`; both read only committed files through the
manifest and return the numbers and counts their caption quotes, so a test can
assert them.
"""

import dataclasses
import importlib
import importlib.util
from typing import Any

from matplotlib.figure import Figure

from tomography.site.manifest import Manifest
from tomography.site.stats import RoleCount

__all__ = [
    "APPENDIX_FIGURES",
    "APPENDIX_TABLES",
    "MAIN_FIGURES",
    "MAIN_TABLES",
    "FigureOutput",
    "RoleCount",
    "TableOutput",
    "build_figure",
    "build_table",
    "missing",
    "module_name",
]


@dataclasses.dataclass(frozen=True)
class FigureOutput:
    """A built figure with its caption parts and the numbers it quotes.

    Attributes:
        figure: The matplotlib figure, 5.5 in wide.
        claim: One sentence with the finding; every number in it is in
            `numbers`.
        detail: What is plotted, controls, and what each interval covers.
        numbers: Name to value for each number the claim quotes.
        csv: The exact numbers plotted, as CSV text.
        counts: Name to the per-role count the claim quotes.
        labels: Name to the checkpoint or other label a sentence names.
        full_size_print: Whether the graphic keeps its full size on paper and
            the caption may run onto the next page, for a figure whose
            caption is too tall to share a page with a full size graphic.
    """

    figure: Figure
    claim: str
    detail: str
    numbers: dict[str, float]
    csv: str
    counts: dict[str, dict[str, RoleCount]] = dataclasses.field(
        default_factory=dict
    )
    labels: dict[str, str] = dataclasses.field(default_factory=dict)
    full_size_print: bool = False


@dataclasses.dataclass(frozen=True)
class TableOutput:
    """A built table.

    Attributes:
        claim: One sentence saying what the table shows.
        detail: Columns, intervals, controls and sources.
        columns: Column headers.
        rows: Cell text, one list per row.
        align: `l` or `r` for each column.
        notes: Footer lines under the bottom rule.
        numbers: Name to value for each number the claim quotes.
        groups: Header spans over the columns after the first, as (label,
            number of columns, muted); empty when columns are ungrouped.
        row_groups: Index of the first row of each row group, which gets a
            rule above it; empty for none, and None to rule above every row
            with a label (the earlier behaviour).
        counts: Name to the per-role count the claim quotes.
        wrap_labels: Whether the row labels may wrap on paper, for a table
            that is too wide for the printed page otherwise.
        compact_print: Whether the table prints in a smaller type with
            tighter cells, for a table that overflows the printed page at
            the ordinary table size.
    """

    claim: str
    detail: str
    columns: list[str]
    rows: list[list[str]]
    align: str
    notes: list[str]
    numbers: dict[str, float]
    groups: tuple[tuple[str, int, bool], ...] = ()
    row_groups: tuple[int, ...] | None = None
    counts: dict[str, dict[str, RoleCount]] = dataclasses.field(
        default_factory=dict
    )
    wrap_labels: bool = False
    compact_print: bool = False


MAIN_FIGURES = (
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
APPENDIX_FIGURES = (
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
MAIN_TABLES = ("checkpoints", "reading")
APPENDIX_TABLES = (
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


def module_name(kind: str, key: str) -> str:
    """Names the module that builds an entry.

    Args:
        kind: `figure` or `table`.
        key: Registry key.

    Returns:
        The dotted module path.
    """
    package = "figs" if kind == "figure" else "tables"
    return f"tomography.site.{package}.{key}"


def missing() -> list[tuple[str, str]]:
    """Lists the registered entries whose module does not exist yet.

    Returns:
        (kind, key) pairs in registry order.
    """
    entries = [
        *(("figure", k) for k in (*MAIN_FIGURES, *APPENDIX_FIGURES)),
        *(("table", k) for k in (*MAIN_TABLES, *APPENDIX_TABLES)),
    ]
    return [
        (kind, key)
        for kind, key in entries
        if importlib.util.find_spec(module_name(kind, key)) is None
    ]


def build_figure(key: str, manifest: Manifest) -> FigureOutput:
    """Builds one figure from the committed results.

    Args:
        key: A figure key.
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    module: Any = importlib.import_module(module_name("figure", key))
    output: FigureOutput = module.build(manifest)
    return output


def build_table(key: str, manifest: Manifest) -> TableOutput:
    """Builds one table from the committed results.

    Args:
        key: A table key.
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    module: Any = importlib.import_module(module_name("table", key))
    output: TableOutput = module.build(manifest)
    return output
