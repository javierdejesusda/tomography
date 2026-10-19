"""Table 7: TruthfulQA and honesty.

One row per model: TruthfulQA MC1 of the validation-layer reader of R6 with
the standard log-probability row and the two text-only classifiers, and the
honesty transfer accuracy of R5 with the seed agreement of A1 at the same
layer.
"""

from typing import Any

from tomography.site import runfiles
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _cells
from tomography.viz import paper

TEXT = ("text_word", "text_char")
COLUMNS = [
    "Model",
    "Reader",
    "Standard",
    "Text words",
    "Text chars",
    "Honesty transfer",
    "Seed agreement",
]
ALIGN = "lrrrrrr"
DETAIL = (
    "Reader is the MC1 of the logistic reader at the layer chosen on "
    "validation, with a 95 percent bootstrap interval that resamples the 15 "
    "readers; the best-layer rule selects on test and is not used. Standard "
    "is the log-probability row and Text words and Text chars are the "
    "text-only classifiers, both with 95 percent Wilson intervals over the "
    "173 questions, so their widths are not comparable with the reader's. "
    "Honesty transfer is the accuracy of the R5 persona reader on plain true "
    "and false statements with no refit, with the bootstrap interval over "
    "its readers; it is exploratory, with 59 test pairs, and the transfer "
    "layer is chosen on 53 validation pairs. Seed agreement is the mean "
    "agreement of the reader directions across seeds at the layer of the "
    "run."
)


def _r6(
    manifest: Manifest, model: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    run = manifest.run("r6", model)
    assert run is not None
    rows = _cells.read_table(run / "table.json")
    reader = next(
        r
        for r in rows
        if r["method"] == "lat"
        and r["layer_rule"] == "validation-layer"
        and r["source"] == "truthfulqa"
    )
    standard = next(r for r in rows if r["method"] == "standard")
    text = {r["method"]: r for r in rows if r["method"] in TEXT}
    return reader, standard, text


def _cell(row: dict[str, Any]) -> str:
    return _cells.interval(row["mc1"], row["lo"], row["hi"])


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.

    Raises:
        ValueError: If two runs store different text-only rows.
    """
    models = manifest.models("r6")
    parts = {m: _r6(manifest, m) for m in models}
    text: dict[str, dict[str, Any]] = {}
    for _, _, found in parts.values():
        for name, row in found.items():
            if name in text and text[name]["mc1"] != row["mc1"]:
                raise ValueError(f"the {name} row differs between models")
            text[name] = row
    low = min(r["lo"] for r in text.values())
    high = max(r["hi"] for r in text.values())
    rows: list[list[str]] = []
    accuracy: list[float] = []
    agreement: list[float] = []
    for model in models:
        reader, standard, _ = parts[model]
        run5, run_a1 = manifest.run("r5", model), manifest.run("a1", model)
        honesty = agree = None
        if run5 is not None:
            summary = runfiles.summary(run5)
            lo, hi = summary["ci"]
            honesty = _cells.interval(summary["test_accuracy"], lo, hi)
            accuracy.append(float(summary["test_accuracy"]))
        if run_a1 is not None:
            value = float(runfiles.summary(run_a1)["agreement_at_layer"]["r5"])
            agree = _cells.num(value)
            agreement.append(value)
        rows.append(
            [
                paper.display_name(model),
                _cell(reader),
                _cell(standard),
                _cell(text["text_word"]),
                _cell(text["text_char"]),
                honesty or _cells.MISSING,
                agree or _cells.MISSING,
            ]
        )
    numbers = {
        "models": float(len(models)),
        "inside_text_interval": float(
            sum(low <= parts[m][0]["mc1"] <= high for m in models)
        ),
        "above_standard": float(
            sum(parts[m][0]["mc1"] > parts[m][1]["mc1"] for m in models)
        ),
        "honesty_min": min(accuracy),
        "honesty_max": max(accuracy),
        "agreement_min": min(agreement),
    }
    claim = (
        "On TruthfulQA the validation-layer reader reaches an MC1 inside the "
        "interval of the text-only classifiers on "
        f"{int(numbers['inside_text_interval'])} of {len(models)} models and "
        f"above the standard log-probability row on "
        f"{int(numbers['above_standard'])}; honesty transfer accuracy spans "
        f"{numbers['honesty_min']:.3f} to {numbers['honesty_max']:.3f} while "
        f"the seed agreement stays at {numbers['agreement_min']:.3f} or more."
    )
    notes = [
        "The text-only rows are model independent: they classify the "
        "questions from their words and characters. The run of GPT-2 small "
        "did not store them, so every row shows the value stored by the "
        "other runs, which agree with each other."
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
