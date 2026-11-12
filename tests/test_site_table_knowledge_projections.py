from pathlib import Path
import runpy
from typing import Any

import numpy as np
import pytest

from tomography.site import cells
from tomography.site import registry
from tomography.site import stats
from tomography.site import views
from tomography.site.manifest import Manifest
from tomography.viz import paper

_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
ROOT, data, models, npz, run = (
    _R["ROOT"],
    _R["data"],
    _R["models"],
    _R["npz"],
    _R["run"],
)
SPEC = "knowledge_projections"
DASHES = (chr(0x2014), chr(0x2013))
COLUMNS = [
    "Model",
    "Contrast",
    "Committed",
    "Recomputed",
    "Published",
    "Random",
    "Shuffled labels",
    "Words",
    "Characters",
    "Popularity",
]
TOLERANCE = 0.02


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table(SPEC, Manifest(ROOT))


def _table(model: str) -> dict[str, dict[str, Any]]:
    payload = data(run("p9b", model), "table.json")
    return {
        r[0]: dict(zip(payload["columns"], r, strict=True))
        for r in payload["rows"]
    }


def _ends(values: np.ndarray) -> list[str]:
    return [cells.num(float(values.min())), cells.num(float(values.max()))]


def _expected(model: str) -> list[list[str]]:
    """Recomputes the rows of a checkpoint from the files of its runs."""
    path = run("v1", model)
    stored = npz(path, "projections.npz")
    names = data(path, "projections.json")["types"]
    methods = data(path, "projections.json")["methods"]
    layer = data(path, "summary.json")["layer"]
    k = stored["layers"].tolist().index(layer)
    table = _table(model)
    rows = []
    for t, name in enumerate(names):
        row = table[name]
        rows.append(
            [
                paper.display_name(model) if t == 0 else "",
                name.replace("_", " "),
                cells.num(row["logistic"]),
                cells.num(
                    float(
                        stored["test"][
                            methods.index("logistic"), t, :, k
                        ].mean()
                    )
                ),
                cells.num(float(stored["published"][t, k])),
                cells.num(
                    float(
                        stored["test"][methods.index("random"), t, :, k].mean()
                    )
                ),
                cells.num(float(stored["shuffled"][t, :, k].mean())),
                cells.num(row["text_word"]),
                cells.num(row["text_char"]),
                cells.num(row["popularity"]),
            ]
        )
    return rows


def _expected_ends(model: str) -> list[tuple[str, ...]]:
    """Recomputes the lowest and highest draw of each control of a row."""
    path = run("v1", model)
    stored = npz(path, "projections.npz")
    names = data(path, "projections.json")["types"]
    methods = data(path, "projections.json")["methods"]
    k = stored["layers"].tolist().index(data(path, "summary.json")["layer"])
    return [
        (
            *_ends(stored["test"][methods.index("random"), t, :, k]),
            *_ends(stored["shuffled"][t, :, k]),
        )
        for t in range(len(names))
    ]


def test_columns_and_alignment(output: registry.TableOutput) -> None:
    assert output.columns == COLUMNS
    assert output.align == "ll" + "r" * 8
    assert output.groups == ()
    assert output.compact_print


def test_there_is_one_row_per_checkpoint_and_contrast_from_the_files(
    output: registry.TableOutput,
) -> None:
    want: list[list[str]] = []
    starts = []
    for model in models("v1"):
        starts.append(len(want))
        want += _expected(model)
    assert len(want) == 95
    assert output.rows == want
    assert output.row_groups == tuple(starts)
    assert output.numbers["rows"] == len(want)
    assert all("_" not in cell for row in output.rows for cell in row)
    assert cells.MISSING in {row[-1] for row in output.rows}


def test_the_page_shows_means_and_the_data_file_the_draw_ranges(
    output: registry.TableOutput,
) -> None:
    assert output.data_columns == (
        "Random: lowest draw",
        "Random: highest draw",
        "Shuffled labels: lowest draw",
        "Shuffled labels: highest draw",
    )
    want = [row for model in models("v1") for row in _expected_ends(model)]
    assert list(output.data_cells) == want
    assert not any("[" in cell for row in output.rows for cell in row)


def test_the_committed_value_is_the_one_the_knowledge_table_reads() -> None:
    for model in models("v1"):
        summary = data(run("v1", model), "summary.json")
        reading = data(run("p9b", model), "summary.json")
        assert summary["layer"] == reading["layer"]
        assert summary["surface_equal"]
        found = views.load(Manifest(ROOT), model)
        assert found is not None
        table = _table(model)
        assert [c.name for c in found.contrasts] == list(table)
        for contrast in found.contrasts:
            assert contrast.committed == table[contrast.name]["logistic"]


def test_a_run_that_follows_another_reading_run_is_refused() -> None:
    class Moved(Manifest):
        def run(self, exp: str, model: str, **select: str) -> Path | None:
            other = "gpt2-large" if exp == "p9b" else model
            return super().run(exp, other, **select)

    with pytest.raises(ValueError, match="follows"):
        views.load(Moved(ROOT), "gpt2-xl")


def test_a_checkpoint_without_a_run_has_no_views() -> None:
    assert "qwen3-4b" not in models("v1")
    assert views.load(Manifest(ROOT), "qwen3-4b") is None


def _misses() -> dict[str, list[tuple[str, str, float]]]:
    found: dict[str, list[tuple[str, str, float]]] = {}
    for model in models("v1"):
        check = data(run("v1", model), "summary.json")["reproduction"]
        found[model] = [
            (name, method, gap)
            for name, by_method in check["mean_diff_at_layer"].items()
            for method, gap in by_method.items()
            if gap > check["tolerance"]
        ]
        assert check["tolerance"] == TOLERANCE
        assert check["within_tolerance"] == (not found[model])
    return found


def test_the_reproduction_is_counted_per_role_and_its_misses_are_named(
    output: registry.TableOutput,
) -> None:
    misses = _misses()
    assert output.counts["within_tolerance"] == stats.per_role(
        {model: not found for model, found in misses.items()}
    )
    every = {m: m in misses for m in _R["MANIFEST"].checkpoints()}
    assert output.counts["with_run"] == stats.per_role(every)
    named = [
        (model, *miss) for model, found in misses.items() for miss in found
    ]
    assert named == [("qwen3-1.7b", "kuq_controversial", "random", named[0][3])]
    assert named[0][3] == pytest.approx(0.0354, abs=5e-5)
    assert output.notes == [
        "Outside the tolerance of 0.020: the random direction of kuq "
        "controversial on Qwen3-1.7B, whose mean over the five draws moves "
        "by 0.035."
    ]
    assert output.numbers["tolerance"] == TOLERANCE
    assert output.numbers["misses"] == 1


def test_the_largest_gaps_of_the_reader_are_quoted(
    output: registry.TableOutput,
) -> None:
    gaps = []
    for model in models("v1"):
        check = data(run("v1", model), "summary.json")["reproduction"]
        gaps += [g["logistic"] for g in check["mean_diff_at_layer"].values()]
    assert output.numbers["reader_gap_max"] == pytest.approx(max(gaps))
    assert max(gaps) < TOLERANCE
    assert f"{cells.num(max(gaps))}" in output.detail


def test_the_caption_says_what_each_column_is(
    output: registry.TableOutput,
) -> None:
    text = f"{output.claim} {output.detail}"
    for phrase in (
        "AUROC",
        "float32",
        "five draws",
        "mean over the five draws, and the data file adds the lowest and "
        "the highest draw",
        f"{stats.per_role_text(output.counts['within_tolerance'])} with "
        "stored views",
        "published reader",
        "labels shuffled",
        "controls",
        stats.per_role_text(output.counts["with_run"]),
        stats.per_role_text(output.counts["within_tolerance"]),
    ):
        assert phrase in text, phrase
    assert "in brackets" not in text
    assert not any(d in text for d in DASHES)
    assert "baseline" not in text.lower()
    assert "_" not in text


def test_the_caption_names_the_contrasts_the_text_alone_separates(
    output: registry.TableOutput,
) -> None:
    perfect: dict[str, None] = {}
    for model in models("v1"):
        table = _table(model)
        for name in data(run("v1", model), "projections.json")["types"]:
            row = table[name]
            if "1.000" in (
                cells.num(row["text_word"]),
                cells.num(row["text_char"]),
            ):
                perfect[name.replace("_", " ")] = None
    assert {
        "pre vs post cutoff",
        "real vs synthal",
        "unknowable detail",
    } <= set(perfect)
    assert (
        "The classifier on the words or the one on the characters reads "
        f"1.000 for {stats.join_and(list(perfect))} on at least one "
        "checkpoint each: there the text of the questions alone separates "
        "the two sides."
    ) in output.detail
