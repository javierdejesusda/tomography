"""Table A8: Stored base against post-trained comparisons.

The paired comparisons of the Qwen3 0.6B and 1.7B pairs as the repository
stores them: for each reading task the accuracy of the base and of the
post-trained checkpoint, each read at its own layer, and the difference with
its bootstrap interval over seeds.
"""

import json
from typing import Any

from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _cells

MEASURES = {
    "r2": "utility",
    "r3": "morality",
    "r7": "truth (transfer sets)",
    "r5": "honesty (function template)",
}
COLUMNS = [
    "Measure",
    "Size",
    "Base",
    "Post-trained",
    "Difference",
    "Interval",
    "Base depth",
    "Post-trained depth",
]


def _stored(manifest: Manifest) -> dict[str, dict[str, dict[str, Any]]]:
    found: dict[str, dict[str, dict[str, Any]]] = {}
    for model in manifest.models("p8"):
        run = manifest.run("p8", model)
        assert run is not None
        for row in _cells.read_table(run / "table.json"):
            found.setdefault(row["tuned"], {})[row["source"]] = row
    return found


def _chance(manifest: Manifest) -> float:
    """Reads the chance level the stored curves of every row agree on."""
    levels = set()
    for model in manifest.models("p8"):
        run = manifest.run("p8", model)
        assert run is not None
        for path in run.glob("curves_r*_qwen3-*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            levels.add(payload["data"]["chance"])
    assert len(levels) == 1, levels
    return float(levels.pop())


def _size(tuned: str) -> str:
    return tuned.split("-")[1].upper()


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the stored paired comparisons.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    stored = _stored(manifest)
    tuned_models = sorted(stored, key=lambda k: float(_size(k)[:-1]))
    rows: list[list[str]] = []
    row_groups: list[int] = []
    below = 0
    for tuned in tuned_models:
        row_groups.append(len(rows))
        for source, name in MEASURES.items():
            rec = stored[tuned][source]
            below += rec["hi"] < 0
            rows.append(
                [
                    name,
                    _size(tuned),
                    _cells.num(rec["base_accuracy"]),
                    _cells.num(rec["tuned_accuracy"]),
                    _cells.num(rec["delta"]),
                    f"[{_cells.num(rec['lo'])}, {_cells.num(rec['hi'])}]",
                    _cells.num(rec["base_depth"]),
                    _cells.num(rec["tuned_depth"]),
                ]
            )
    claim = "Stored comparisons of the 0.6B and 1.7B pairs."
    detail = (
        "The difference is post-trained minus base, taken per seed on the "
        "same stimuli, with a percentile bootstrap interval over the 10 "
        "seeds; each checkpoint is read at its own layer, given as relative "
        "depth. Each accuracy is read from pairs with two balanced classes, "
        "where chance is [[num:table:base_post.chance]]; no random-direction "
        "accuracy is stored for these comparisons, so none is shown. "
        "[[fig:base_post]] shows all three pairs with intervals for "
        "the test items."
    )
    return TableOutput(
        claim,
        detail,
        COLUMNS,
        rows,
        _cells.check_align("llrrrrrr", len(COLUMNS)),
        [],
        {
            "below_zero": float(below),
            "rows": float(len(rows)),
            "chance": _chance(manifest),
        },
        row_groups=tuple(row_groups),
    )
