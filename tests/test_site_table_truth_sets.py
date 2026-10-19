from pathlib import Path
import re
import runpy

import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import paper

_HERE = Path(__file__).parent
_R = runpy.run_path(str(_HERE / "_results.py"))
_M = runpy.run_path(str(_HERE / "_site_math.py"))
ROOT, data, models, npz, run = (
    _R["ROOT"],
    _R["data"],
    _R["models"],
    _R["npz"],
    _R["run"],
)
fmt, groups, wilson = _M["fmt"], _M["groups"], _M["wilson"]
DASHES = (chr(0x2014), chr(0x2013))
SETS = (
    "cities",
    "neg_cities",
    "sp_en_trans",
    "neg_sp_en_trans",
    "larger_than",
    "smaller_than",
    "cities_cities_conj",
    "cities_cities_disj",
    "common_claim_true_false",
)
NAMES = [
    "cities",
    "negated cities",
    "Spanish to English",
    "negated Spanish to English",
    "larger than",
    "smaller than",
    "cities conjunction",
    "cities disjunction",
    "common claims",
]


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table("truth_sets", Manifest(ROOT))


def test_columns_are_the_nine_sets_in_display_names(
    output: registry.TableOutput,
) -> None:
    assert output.columns == ["Model", *NAMES]
    assert output.align == "l" + "r" * 9
    assert output.row_groups == groups(models("r7"))


def test_every_cell_is_reader_and_random_with_a_star_below_half(
    output: registry.TableOutput,
) -> None:
    stars = 0
    for row, model in zip(output.rows, models("r7"), strict=True):
        path = run("r7", model)
        summary = data(path, "summary.json")
        layer = summary["val_layer"]
        assert row[0] == paper.display_name(model)
        for col, key in enumerate(SETS, 1):
            block = summary["sets"][key]
            reader = block["accuracy_at_val_layer"]
            series = data(path, f"curves_{key}.json")["series"]
            mean = npz(path, f"curves_{key}.npz")["mean"]
            random = float(mean[series.index("random")][layer - 1])
            _, hi = wilson(reader, block["n_eval_pairs"])
            star = "*" if hi < 0.5 else ""
            stars += bool(star)
            assert row[col] == f"{fmt(reader)}{star} ({fmt(random)})", (
                model,
                key,
            )
    assert output.numbers["below_half"] == stars
    assert len(output.rows) * 9 == 90


def test_the_caption_explains_the_parentheses_and_the_star(
    output: registry.TableOutput,
) -> None:
    text = output.claim + " " + output.detail
    assert "random direction at that layer in parentheses" in text
    assert "Wilson interval lies below 0.5" in text
    assert "layer chosen on validation" in text
    assert not any(ch in text for ch in DASHES)


def test_no_underscore_or_run_code(output: registry.TableOutput) -> None:
    texts = [output.claim, output.detail, *output.columns]
    texts += [c for r in output.rows for c in r]
    assert not any("_" in t or "`" in t for t in texts)
    assert not any(re.search(r"\b[RPMCA][0-9]+[a-d]?\b", t) for t in texts)
