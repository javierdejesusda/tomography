"""Table 4: Knowledge directions.

One row per model: the guess-controlled count of known questions (P9a), the
within-relation AUROC of the known-versus-unknown reader against popularity
and two text-only classifiers (P9b), and the gain in error-prediction AUROC
over the model confidence from adding the label-free direction or the probe
(P9c).
"""

from typing import Any

from tomography.site import store as store_module
from tomography.site.manifest import Manifest
from tomography.site.registry import TableOutput
from tomography.site.tables import _cells
from tomography.viz import paper

COLUMNS = [
    "Model",
    "Known questions",
    "Test known / unknown",
    "Reader",
    "Popularity",
    "Text words",
    "Text chars",
    "Confidence + lat",
    "Confidence + probe",
]
ALIGN = "lrrrrrrrr"
DETAIL = (
    "One row per model; each model is labeled on its own behavior, so the "
    "question sets differ between rows. Known questions are counted after "
    "the guess control (P9a). Reader, Popularity, Text words and Text chars "
    "are the within-relation AUROC of separating known from unknown PopQA "
    "questions on the test questions (P9b, `obscure_real`); the stored "
    "interval of the reader resamples the five reader draws, not the test "
    "questions, so none is shown. The last two columns are the change in "
    "AUROC for predicting wrong PopQA answers when the label-free direction "
    "(lat) or the supervised probe is added to the model confidence, with a "
    "95 percent bootstrap interval that resamples the test questions (P9c). "
    "A model without a known-versus-unknown contrast has no reader and no "
    "lat."
)


def _records(
    manifest: Manifest, exp: str, model: str, name: str
) -> list[dict[str, Any]]:
    run = manifest.run(exp, model)
    assert run is not None
    return _cells.read_table(run / name)


def _gain(hit: dict[str, Any] | None) -> str:
    if hit is None:
        return _cells.MISSING
    return _cells.interval(hit["diff_vs_confidence"], hit["lo"], hit["hi"])


def _field(row: dict[str, Any] | None, name: str) -> str:
    return _cells.num(row[name] if row else None)


def build(manifest: Manifest) -> TableOutput:
    """Builds the table from the committed results.

    Args:
        manifest: The run resolver.

    Returns:
        The table and its caption parts.
    """
    store = store_module.Store(manifest.root)
    rows: list[list[str]] = []
    readers: dict[str, dict[str, Any]] = {}
    lat: dict[str, dict[str, Any]] = {}
    probe: dict[str, dict[str, Any]] = {}
    models = manifest.models("p9a")
    for model in models:
        run = manifest.run("p9a", model)
        assert run is not None
        known = store.summary(run)["guess"]["n_known"]
        found = [
            r
            for r in _records(manifest, "p9b", model, "table.json")
            if r["type"] == "obscure_real"
        ]
        reader = found[0] if found else None
        if reader:
            readers[model] = reader
        increments = {
            r["features"]: r
            for r in _records(manifest, "p9c", model, "table_incremental.json")
            if r["dataset"] == "popqa"
        }
        if "confidence+lat" in increments:
            lat[model] = increments["confidence+lat"]
        if "confidence+probe" in increments:
            probe[model] = increments["confidence+probe"]

        pair = (
            _cells.count(reader["n_positive_test"], reader["n_negative_test"])
            if reader
            else _cells.MISSING
        )
        rows.append(
            [
                paper.display_name(model),
                f"{known:,}",
                pair,
                _field(reader, "logistic_within"),
                _field(reader, "popularity_within"),
                _field(reader, "text_word_within"),
                _field(reader, "text_char_within"),
                _gain(lat.get(model)),
                _gain(probe.get(model)),
            ]
        )
    numbers = {
        "models": float(len(models)),
        "with_contrast": float(len(readers)),
        "reader_above_popularity": float(
            sum(
                r["logistic_within"] > r["popularity_within"]
                for r in readers.values()
            )
        ),
        "reader_above_text": float(
            sum(
                r["logistic_within"]
                > max(r["text_word_within"], r["text_char_within"])
                for r in readers.values()
            )
        ),
        "models_with_lat": float(len(lat)),
        "lat_above_zero": float(sum(v["lo"] > 0 for v in lat.values())),
        "probe_above_zero": float(sum(v["lo"] > 0 for v in probe.values())),
    }
    claim = (
        f"Of the {len(models)} models, {len(readers)} have a "
        "known-versus-unknown contrast, and there the reader's "
        "within-relation AUROC exceeds popularity on "
        f"{int(numbers['reader_above_popularity'])} and both text baselines "
        f"on {int(numbers['reader_above_text'])}; adding the probe to the "
        "model confidence raises the error AUROC with an interval above zero "
        f"on {int(numbers['probe_above_zero'])} of {len(models)} models, "
        "against "
        f"{int(numbers['lat_above_zero'])} of {len(lat)} for the "
        "label-free direction."
    )
    notes = [
        "The text-only classifiers use only the words or characters of the "
        "question; popularity is the PopQA popularity of the subject."
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
