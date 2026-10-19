"""The list of figures and tables of the results paper and the site.

Each entry names the module that builds it. A figure module defines
`build(manifest) -> FigureOutput` and a table module
`build(manifest) -> TableOutput`; both read only committed files through the
manifest and return the numbers their caption quotes, so a test can assert
them.
"""

import dataclasses
import importlib
from typing import Any

from matplotlib.figure import Figure

from tomography.site.manifest import Manifest


@dataclasses.dataclass(frozen=True)
class FigureOutput:
    """A built figure with its caption parts and the numbers it quotes.

    Attributes:
        figure: The matplotlib figure, 5.5 in wide.
        claim: One sentence with the finding; every number in it is in
            `numbers`.
        detail: What is plotted, baselines, and what each interval covers.
        numbers: Name to value for each number the claim quotes.
        csv: The exact numbers plotted, as CSV text.
    """

    figure: Figure
    claim: str
    detail: str
    numbers: dict[str, float]
    csv: str


@dataclasses.dataclass(frozen=True)
class TableOutput:
    """A built table.

    Attributes:
        claim: One sentence saying what the table shows.
        detail: Columns, intervals, baselines and sources.
        columns: Column headers.
        rows: Cell text, one list per row.
        align: `l` or `r` for each column.
        notes: Footer lines under the bottom rule.
        numbers: Name to value for each number the claim quotes.
        groups: Header spans over the columns after the first, as (label,
            number of columns, muted); empty when columns are ungrouped.
    """

    claim: str
    detail: str
    columns: list[str]
    rows: list[list[str]]
    align: str
    notes: list[str]
    numbers: dict[str, float]
    groups: tuple[tuple[str, int, bool], ...] = ()


@dataclasses.dataclass(frozen=True)
class Spec:
    """One figure or table of the paper.

    Attributes:
        key: Stable id, such as `fig3`, `table2` or `figA1`.
        number: Label shown in the caption.
        title: Short name.
        module: Dotted module that holds `build`.
        sources: Experiment ids the builder reads.
    """

    key: str
    number: str
    title: str
    module: str
    sources: tuple[str, ...]


def _fig(n: int, title: str, *sources: str) -> Spec:
    return Spec(
        f"fig{n}", str(n), title, f"tomography.site.figs.fig{n}", sources
    )


def _table(n: int, title: str, *sources: str) -> Spec:
    return Spec(
        f"table{n}",
        str(n),
        title,
        f"tomography.site.tables.table{n}",
        sources,
    )


def _appendix_table(n: int, title: str, *sources: str) -> Spec:
    return Spec(
        f"tableA{n}",
        f"A{n}",
        title,
        f"tomography.site.tables.table_a{n}",
        sources,
    )


def _appendix(n: int, title: str, *sources: str) -> Spec:
    return Spec(
        f"figA{n}", f"A{n}", title, f"tomography.site.figs.a{n}", sources
    )


FIGURES = (
    _fig(1, "Method schematic"),
    _fig(2, "Where and how well a concept can be read", "r1", "r2", "r3", "r7"),
    _fig(3, "The paper's PCA finder against supervised finders", "p4"),
    _fig(4, "Prompt wording and reader transfer", "p1"),
    _fig(
        5,
        "A knowledge direction against popularity and confidence",
        "p9b",
        "p9c",
    ),
    _fig(6, "Dose response of pushing along the knowledge direction", "p9d"),
    _fig(7, "Sentiment steering and fluency", "c1"),
    _fig(8, "Approximate answers and range readers", "p10a", "p10b"),
    _fig(
        9, "Where reading does not beat text or noise", "r5", "r6", "r7", "a1"
    ),
)

TABLES = (
    _table(1, "Models"),
    _table(2, "Reading accuracy by model", "r1", "r2", "r3", "r5", "r7", "p6"),
    _table(3, "Direction finders", "p4", "m1"),
    _table(4, "Knowledge directions", "p9a", "p9b", "p9c"),
    _table(5, "Approximate answers and links", "p10a", "p10b", "p10c"),
    _table(6, "Steering and fluency", "c1", "m1", "c2"),
    _table(7, "TruthfulQA and honesty", "r5", "r6", "a1"),
    _table(8, "Base against post-trained", "p8", "p3", "p2"),
)

APPENDIX = (
    _appendix(
        1, "Layer curves of every concept and model", "r1", "r2", "r3", "r7"
    ),
    _appendix(2, "All finders on all models", "p4"),
    _appendix(3, "Annotated transfer matrices", "p1"),
    _appendix(4, "P9b transfer matrix with text baselines flagged", "p9b"),
    _appendix(5, "Dose curves of the post-trained runs", "p9d"),
    _appendix(6, "P10c cosine and AUROC against the random band", "p10c"),
)

APPENDIX_TABLES = (
    _appendix_table(
        1,
        "Reading accuracy with intervals and selected layers",
        "r1",
        "r2",
        "r3",
        "r5",
        "r7",
    ),
)


def load(spec: Spec) -> Any:
    """Imports the module that builds a figure or table.

    Args:
        spec: The entry.

    Returns:
        The module, which defines `build`.
    """
    return importlib.import_module(spec.module)


def build_figure(spec: Spec, manifest: Manifest) -> FigureOutput:
    """Builds one figure from the committed results.

    Args:
        spec: A figure entry.
        manifest: The run resolver.

    Returns:
        The figure and its caption parts.
    """
    output: FigureOutput = load(spec).build(manifest)
    return output


def build_table(spec: Spec, manifest: Manifest) -> TableOutput:
    """Builds one table from the committed results.

    Args:
        spec: A table entry.
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    output: TableOutput = load(spec).build(manifest)
    return output
