"""Table 8: Base against post-trained.

Three blocks in one table: the accuracy difference between each post-trained
Qwen3 checkpoint and its base (P8), how many stimulus pairs the sentiment
reader needs (P3), and what a misleading sentence does to the sentiment
reader (P2).
"""

from typing import Any

from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _cells
from tomography.viz import paper

CLOSE = 0.02
COLUMNS = ["Block", "Model", "Measure", "Value"]
ALIGN = "lllr"
DETAIL = (
    "P8 rows give the accuracy of the post-trained checkpoint minus that of "
    "its base for each reading task, with the 95 percent bootstrap interval "
    "over the readers that the run stores, which resamples the readers and "
    "not the evaluation pairs. P3 rows give the sentiment reader accuracy at "
    "the smallest and the largest number of stimulus pairs, with the "
    "interval over the reader draws, and the smallest number of pairs whose "
    "accuracy is within 0.02 of the accuracy at the largest number. P2 rows "
    "give the sentiment accuracy at the last token for the original "
    "stimulus and when a misleading sentence is added, with the interval "
    "over the reader draws."
)


def _p8(manifest: Manifest) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for model in manifest.models("p8"):
        run = manifest.run("p8", model)
        assert run is not None
        found.extend(_cells.read_table(run / "table.json"))
    return found


def _p3(manifest: Manifest, model: str) -> list[dict[str, Any]]:
    run = manifest.run("p3", model)
    assert run is not None
    rows = [
        r
        for r in _cells.read_table(run / "table.json")
        if r["task"] == "sentiment" and r["target"] == "sentiment"
    ]
    return sorted(rows, key=lambda r: r["size"])


def _p2(manifest: Manifest, model: str) -> dict[str, dict[str, Any]]:
    run = manifest.run("p2", model)
    assert run is not None
    return {
        r["condition"]: r
        for r in _cells.read_table(run / "table.json")
        if r["task"] == "sentiment" and r["position"] == "last"
    }


def _close(rows: list[dict[str, Any]]) -> int:
    top = rows[-1]["accuracy"]
    return int(next(r["size"] for r in rows if r["accuracy"] >= top - CLOSE))


def _acc(row: dict[str, Any]) -> str:
    return _cells.interval(row["accuracy"], row["lo"], row["hi"])


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    pairs = _p8(manifest)
    rows: list[list[str]] = []
    for pair in pairs:
        rows.append(
            [
                "P8",
                f"{paper.display_name(pair['base'])} to "
                f"{paper.display_name(pair['tuned'])}",
                f"{pair['source']} {pair['kind']}, post-trained minus base",
                _cells.interval(pair["delta"], pair["lo"], pair["hi"]),
            ]
        )
    close: list[int] = []
    for model in manifest.models("p3"):
        table = _p3(manifest, model)
        name = paper.display_name(model)
        small, large = table[0], table[-1]
        close.append(_close(table))
        rows.append(
            [
                "P3",
                name,
                f"sentiment accuracy, {int(small['size'])} pairs",
                _acc(small),
            ]
        )
        rows.append(
            [
                "P3",
                name,
                f"sentiment accuracy, {int(large['size'])} pairs",
                _acc(large),
            ]
        )
        rows.append(
            [
                "P3",
                name,
                "smallest pairs within 0.02 of the largest",
                str(close[-1]),
            ]
        )
    p2_models = manifest.models("p2")
    lowers = 0
    for model in p2_models:
        found = _p2(manifest, model)
        name = paper.display_name(model)
        rows.append(
            [
                "P2",
                name,
                "sentiment accuracy, original",
                _acc(found["original"]),
            ]
        )
        rows.append(
            [
                "P2",
                name,
                "sentiment accuracy, misleading sentence",
                _acc(found["against"]),
            ]
        )
        lowers += found["against"]["accuracy"] < found["original"]["accuracy"]
    numbers = {
        "pairs": float(len(pairs)),
        "post_lower": float(sum(p["hi"] < 0 for p in pairs)),
        "post_higher": float(sum(p["lo"] > 0 for p in pairs)),
        "close_min": float(min(close)),
        "close_max": float(max(close)),
        "p2_models": float(len(p2_models)),
        "misleading_lowers": float(lowers),
    }
    claim = (
        "Post-training lowers the reading accuracy with an interval below "
        f"zero in {int(numbers['post_lower'])} of {len(pairs)} "
        f"base-to-post-trained comparisons and raises it in "
        f"{int(numbers['post_higher'])}; the sentiment reader comes within "
        f"0.02 of its largest-sample accuracy at {int(numbers['close_min'])} "
        f"to {int(numbers['close_max'])} pairs; and a misleading sentence "
        f"lowers its accuracy on {lowers} of {len(p2_models)} models."
    )
    notes = [
        "P8 ran for the Qwen3 0.6B and 1.7B pairs; the R5 row is the "
        "function-template honesty reader, the others use the concept "
        "template.",
    ]
    return TableOutput(
        claim,
        DETAIL,
        COLUMNS,
        rows,
        _cells.check_align(ALIGN, len(COLUMNS)),
        notes,
        numbers,
    )
