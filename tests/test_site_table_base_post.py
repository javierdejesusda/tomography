from pathlib import Path
import runpy

import pytest

from tomography.site import registry
from tomography.site.manifest import Manifest

_HERE = Path(__file__).parent
_R = runpy.run_path(str(_HERE / "_results.py"))
_M = runpy.run_path(str(_HERE / "_site_math.py"))
ROOT, data, models, run = _R["ROOT"], _R["data"], _R["models"], _R["run"]
fmt = _M["fmt"]
DASHES = (chr(0x2014), chr(0x2013))
MEASURES = {
    "r1": "sentiment (layer chosen on the scored pairs)",
    "r2": "utility",
    "r3": "morality",
    "r7": "truth (transfer sets)",
    "r5": "honesty (function template)",
}
SIZES = ("0.6B", "1.7B")


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table("base_post", Manifest(ROOT))


def _stored() -> dict[str, dict[str, dict]]:
    found: dict[str, dict[str, dict]] = {}
    for model in models("p8"):
        payload = data(run("p8", model), "table.json")
        for row in payload["rows"]:
            rec = dict(zip(payload["columns"], row, strict=True))
            found.setdefault(rec["tuned"], {})[rec["source"]] = rec
    return found


def test_columns_and_row_groups(output: registry.TableOutput) -> None:
    assert output.columns == [
        "Measure",
        "Size",
        "Base",
        "Post-trained",
        "Difference",
        "Interval",
        "Base depth",
        "Post-trained depth",
    ]
    assert output.align == "llrrrrrr"
    assert output.row_groups == (0, 5)


def test_rows_equal_the_stored_comparisons(
    output: registry.TableOutput,
) -> None:
    stored = _stored()
    assert len(output.rows) == 10
    expected = []
    for size in SIZES:
        tuned = f"qwen3-{size.lower()}"
        for source, name in MEASURES.items():
            row = stored[tuned][source]
            expected.append(
                [
                    name,
                    size,
                    fmt(row["base_accuracy"]),
                    fmt(row["tuned_accuracy"]),
                    fmt(row["delta"]),
                    f"[{fmt(row['lo'])}, {fmt(row['hi'])}]",
                    fmt(row["base_depth"]),
                    fmt(row["tuned_depth"]),
                ]
            )
    assert output.rows == expected


def test_caption_names_the_statistic(output: registry.TableOutput) -> None:
    text = output.claim + " " + output.detail
    assert "post-trained minus base" in text
    assert "percentile bootstrap interval over the 10 seeds" in text
    assert "own layer" in text
    assert "Figure" in text or "[[fig:base_post]]" in text
    assert not any(ch in text for ch in DASHES)
    assert "_" not in text.replace("[[fig:base_post]]", "")


def test_numbers_count_intervals_off_zero(
    output: registry.TableOutput,
) -> None:
    rows = _stored()
    low = sum(r["hi"] < 0 for sizes in rows.values() for r in sizes.values())
    assert output.numbers["below_zero"] == low
