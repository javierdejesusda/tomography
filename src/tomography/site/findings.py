"""Headline findings and the experiment overview, built from result files.

Each function reads the committed runs through a `Store` and returns a
`Finding`: a claim, one or two figures, numbers computed from the files, and
the caveats that must stay next to the chart. A function returns None when
its experiments have no runs yet, so results that land later appear on the
next build without any edit here.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import math
import re

from tomography.site import panels as svg
from tomography.site.store import MODEL_ORDER
from tomography.site.store import Record
from tomography.site.store import Store
from tomography.site.store import is_post_trained

READER = "reader"
STYLE_CLASSES = ("s1", "s2", "s3", "s4", "s5", "s6")


@dataclasses.dataclass(frozen=True)
class Figure:
    """A group of panels drawn as one figure with one legend.

    Attributes:
        title: Short heading above the panels.
        panels: Dot or line panels.
        keys: Extra legend entries as (kind, label); kind is filled,
            hollow, band or dashed. Series entries come from the panels.
        note: One line saying how to read the figure.
    """

    title: str
    panels: tuple[svg.DotPanel | svg.LinePanel, ...]
    keys: tuple[tuple[str, str], ...]
    note: str = ""
    columns: int = 1


@dataclasses.dataclass(frozen=True)
class Finding:
    """One headline result.

    Attributes:
        key: Anchor id.
        title: Section title.
        claim: One sentence stating the result.
        figures: Charts for the claim.
        numbers: Sentences with numbers computed from the files.
        caveats: What must be read next to the chart.
        sources: Experiment ids behind the figures.
    """

    key: str
    title: str
    claim: str
    figures: tuple[Figure, ...]
    numbers: tuple[str, ...]
    caveats: tuple[str, ...]
    sources: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ExperimentInfo:
    """One row of the experiment table.

    Attributes:
        key: Experiment id.
        question: What the experiment asks.
        models: Number of models with a run.
        status: `done`, `partial` or `pending`.
    """

    key: str
    question: str
    models: int
    status: str


@dataclasses.dataclass(frozen=True)
class ModelInfo:
    """One row of the model table.

    Attributes:
        key: Checkpoint key.
        role: `pretrained`, `base` or `post-trained`.
        params: Parameter count as text.
        blocks: Number of transformer blocks, when known.
        experiments: Number of experiments with a run on the model.
    """

    key: str
    role: str
    params: str
    blocks: int | None
    experiments: int


@dataclasses.dataclass(frozen=True)
class Overview:
    """Tables for the models and experiments sections.

    Attributes:
        models: One row per checkpoint with results.
        experiments: One row per experiment id.
    """

    models: tuple[ModelInfo, ...]
    experiments: tuple[ExperimentInfo, ...]


def _family(model: str) -> str:
    return "gpt2" if model.startswith("gpt2") else "qwen3"


def _step_candidates() -> list[float]:
    return [m * 10.0**e for e in range(-3, 3) for m in (1, 2, 5)]


def _axis(
    values: Sequence[float],
) -> tuple[tuple[float, float], tuple[float, ...]]:
    lo, hi = min(values), max(values)
    pad = (hi - lo or 1.0) * 0.08
    lo, hi = lo - pad, hi + pad
    step = next(s for s in _step_candidates() if (hi - lo) / s <= 5)
    start = math.floor(lo / step) * step
    end = math.ceil(hi / step) * step
    count = round((end - start) / step)
    ticks = tuple(round(start + i * step, 6) for i in range(count + 1))
    return (round(start, 6), round(end, 6)), ticks


def _size_key(model: str) -> str:
    return model.removesuffix("-base")


def signed(value: float) -> str:
    """Formats a difference with a sign, without a negative zero.

    Args:
        value: The difference.

    Returns:
        The value with three decimals, or `0.000` when it rounds to zero.
    """
    text = f"{value:+.3f}"
    return "0.000" if text in ("+0.000", "-0.000") else text


def _mean(cell: Sequence[Record], key: str) -> float:
    return float(sum(r[key] for r in cell) / len(cell))


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _keys(extras: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    return tuple(extras)


MODEL_KEYS = (
    ("filled", "Base or GPT-2 checkpoint"),
    ("hollow", "Post-trained checkpoint"),
)


def _reader_point(
    exp: str, summary: Record
) -> tuple[float, tuple[float, float], int] | None:
    if exp == "r1":
        keys = ("best_accuracy", "ci", "best_layer")
        if not all(k in summary for k in keys):
            return None
        return summary[keys[0]], tuple(summary[keys[1]]), summary[keys[2]]
    if exp == "r7":
        test = summary.get("sets", {}).get("test")
        if test is None:
            return None
        return (
            test["accuracy_at_val_layer"],
            tuple(test["ci_at_val_layer"]),
            summary["val_layer"],
        )
    if "test_accuracy" not in summary:
        return None
    return summary["test_accuracy"], tuple(summary["ci"]), summary["best_layer"]


def scaling(store: Store) -> Finding | None:
    """Reading accuracy by concept and checkpoint, against random directions.

    Args:
        store: Result store.

    Returns:
        The finding, or None when no reader runs exist.
    """
    concepts = (
        ("r1", "Sentiment (R1)"),
        ("r2", "Utility (R2)"),
        ("r3", "Morality (R3)"),
        ("r7", "Truth (R7)"),
    )
    styles = {READER: svg.Style("s1", "circle", "Reader accuracy")}
    panels: list[svg.DotPanel] = []
    numbers: list[str] = []
    random_max = 0.0
    for exp, title in concepts:
        rows: list[svg.Row] = []
        for model in store.models(exp):
            run = store.latest(exp, model, "summary.json")
            point = _reader_point(exp, store.summary(run)) if run else None
            if run is None or point is None:
                continue
            acc, (lo, hi), layer = point
            ctrl = store.curve_at(run, "curves", "random", layer)
            tip = f"reader {acc:.3f} [{lo:.3f}, {hi:.3f}]"
            if ctrl is not None:
                tip += f"; random direction {ctrl[0]:.3f}"
                random_max = max(random_max, ctrl[0])
            rows.append(
                svg.Row(
                    model,
                    (
                        svg.Mark(
                            READER,
                            acc,
                            lo,
                            hi,
                            hollow=is_post_trained(model),
                            tip=tip,
                        ),
                    ),
                    band=(ctrl[1], ctrl[2]) if ctrl else None,
                    group=_family(model),
                )
            )
        if not rows:
            continue
        panels.append(
            svg.DotPanel(
                uid=f"scaling-{exp}",
                title=title,
                rows=tuple(rows),
                styles=styles,
                domain=(0.4, 1.0),
                ticks=(0.4, 0.6, 0.8, 1.0),
                refs=((0.5, "chance"),),
                band_label="random direction at the same layer",
            )
        )
        best = max(rows, key=lambda r: r.marks[0].value)
        worst = min(rows, key=lambda r: r.marks[0].value)
        numbers.append(
            f"{title}: from {worst.marks[0].value:.3f} on {worst.label} "
            f"to {best.marks[0].value:.3f} on {best.label}."
        )
    if not panels:
        return None
    numbers.append(
        f"A random direction reaches up to {random_max:.3f} at the layer the "
        "reader uses, so the baseline is not 0.5 on the larger models."
    )
    figure = Figure(
        "Held-out accuracy rises with model size on morality and truth, "
        "rises slowly on utility, and does not follow size on sentiment.",
        tuple(panels),
        _keys([*MODEL_KEYS, ("band", "Random direction (interval)")]),
        "Held-out accuracy of the logistic reader. Whiskers are the stored "
        "95% interval over the training draws. The grey band is the "
        "random-direction series at the same layer. Open markers are "
        "post-trained checkpoints.",
    )
    return Finding(
        "scaling",
        "Reading gets easier with size for morality and truth, not sentiment",
        "Held-out accuracy rises with model size on morality and truth, "
        "rises slowly on utility, and does not follow size on sentiment. "
        "Qwen3 checkpoints sit above GPT-2 on every concept.",
        (figure,),
        tuple(numbers),
        (
            "The random direction is not at 0.5 on Qwen3, so read each point "
            "against its grey band.",
            "Intervals cover the training draw only, not the evaluation "
            "pairs. Truth has few test pairs.",
            "Base and post-trained checkpoints are separate points and are "
            "never averaged. The post-trained 4B sentiment reader is the "
            "lowest of the Qwen3 checkpoints, so size does not order it.",
            "Sentiment has no test split: both its numbers are optimistic by "
            "the layer choice. Honesty transfer (R5) is exploratory and is "
            "not drawn.",
            "One run per cell. A reader that separates negated or compound "
            "statements is not shown to track truth rather than polarity.",
        ),
        ("r1", "r2", "r3", "r7"),
    )


def _within_rows(
    store: Store, exp: str, name: str, keep: Callable[[Record], bool]
) -> dict[str, list[Record]]:
    found: dict[str, list[Record]] = {}
    for model in store.models(exp):
        run = store.latest(exp, model, name)
        if run is None:
            continue
        found[model] = [r for r in store.rows(run, name) if keep(r)]
    return found


def knowledge(store: Store) -> Finding | None:
    """Known versus unknown questions: reader, baselines and error prediction.

    Args:
        store: Result store.

    Returns:
        The finding, or None when no P9 runs exist.
    """
    read_styles = {
        READER: svg.Style("s1", "circle", "Reader"),
        "popularity": svg.Style("s4", "diamond", "Subject popularity"),
        "text_word": svg.Style("s5", "square", "Words only"),
        "text_char": svg.Style("s6", "triangle", "Characters only"),
    }
    b_rows: list[svg.Row] = []
    margins: list[float] = []
    for model, rows in _within_rows(
        store, "p9b", "table.json", lambda r: r["type"] == "obscure_real"
    ).items():
        rec = rows[0] if rows else None
        if rec is None or not _is_number(rec.get("logistic_within")):
            continue
        post = is_post_trained(model)
        marks = [
            svg.Mark(
                READER,
                rec["logistic_within"],
                rec["lo_within"],
                rec["hi_within"],
                post,
            )
        ]
        for key in ("popularity", "text_word", "text_char"):
            value = rec.get(f"{key}_within")
            if isinstance(value, int | float):
                marks.append(svg.Mark(key, value, hollow=post))
        if _is_number(rec.get("popularity_within")):
            margins.append(rec["logistic_within"] - rec["popularity_within"])
        b_rows.append(svg.Row(model, tuple(marks), group=_family(model)))
    err_styles = {
        "answer_logprob": svg.Style("s2", "square", "Model confidence"),
        "probe": svg.Style("s3", "triangle", "Probe trained on correctness"),
        "lat": svg.Style("s1", "circle", "Reader, no labels"),
        "popularity": svg.Style("s4", "diamond", "Subject popularity"),
    }
    e_rows: list[svg.Row] = []
    below = 0
    compared = 0
    for model, rows in _within_rows(
        store, "p9c", "table.json", lambda r: r["dataset"] == "popqa"
    ).items():
        by = {r["predictor"]: r for r in rows}
        marks = [
            svg.Mark(
                key,
                by[key]["auroc_within"],
                by[key]["lo_within"],
                by[key]["hi_within"],
                is_post_trained(model),
            )
            for key in err_styles
            if key in by
        ]
        if not marks:
            continue
        if "lat" in by and "answer_logprob" in by:
            compared += 1
            below += (
                by["lat"]["auroc_within"] < by["answer_logprob"]["auroc_within"]
            )
        e_rows.append(svg.Row(model, tuple(marks), group=_family(model)))
    if not b_rows and not e_rows:
        return None
    domain, ticks = (0.4, 1.0), (0.4, 0.6, 0.8, 1.0)
    panels = []
    if b_rows:
        panels.append(
            svg.DotPanel(
                "knowledge-read",
                "Known against unknown",
                tuple(b_rows),
                read_styles,
                domain,
                ticks,
                stagger=True,
                refs=((0.5, "chance"),),
            )
        )
    if e_rows:
        panels.append(
            svg.DotPanel(
                "knowledge-errors",
                "Predicting wrong answers",
                tuple(e_rows),
                err_styles,
                domain,
                ticks,
                stagger=True,
                refs=((0.5, "chance"),),
            )
        )
    figures = [
        Figure(
            "A direction separates known from unknown questions above "
            "popularity, but does not predict errors better than the "
            "model's confidence.",
            tuple(panels),
            _keys(MODEL_KEYS[1:]),
            "Within-relation AUROC on PopQA. (a) Reader, subject popularity "
            "and two text-only baselines for known against unknown "
            "questions; the whisker on the reader is the interval over five "
            "reader draws. (b) Predictors of a wrong answer; whiskers are "
            "item-bootstrap 95% intervals. Open markers are post-trained "
            "checkpoints.",
            columns=len(panels),
        )
    ]
    numbers: list[str] = []
    if margins:
        numbers.append(
            "The reader is above subject popularity on "
            f"{sum(m > 0 for m in margins)}"
            f" of {len(margins)} checkpoints, by {min(margins):.3f} to "
            f"{max(margins):.3f}."
        )
    if e_rows:
        numbers.append(
            f"The label-free reader is below the model's own confidence on "
            f"{below} of {compared} checkpoints that have it."
        )
    for model in store.models("p9c"):
        run = store.latest("p9c", model, "table_incremental.json")
        if run is None:
            continue
        for rec in store.rows(run, "table_incremental.json"):
            if (
                rec["dataset"] == "popqa"
                and rec["features"] == "confidence+lat"
            ):
                numbers.append(
                    f"{model}: adding the reader to confidence changes pooled "
                    f"AUROC by {signed(rec['diff_vs_confidence'])} "
                    f"[{signed(rec['lo'])}, {signed(rec['hi'])}]."
                )
    return Finding(
        "knowledge",
        "A direction for known against unknown is real but predicts errors "
        "no better than confidence",
        "Within a relation, a direction fitted on known against unknown "
        "questions separates them better than subject popularity, but as an "
        "error predictor it does not beat the model's own answer "
        "probability.",
        tuple(figures),
        tuple(numbers),
        (
            "Each model is labelled on its own behavior, so question sets "
            "differ between models. Do not compare counts or AUROCs across "
            "rows as if the questions were shared.",
            "Popularity and wording are the only surface controls. A reader "
            "above both is not shown to read what the model knows.",
            "The known side of the test is small. The reader reads a layer "
            "chosen on its validation split.",
            "GPT-2 small has no known against unknown contrast. Text-only "
            "baselines reach near-perfect scores on cross-dataset contrasts, "
            "so those are never drawn as evidence of knowledge.",
        ),
        ("p9b", "p9c"),
    )


def steering_along_direction(store: Store) -> Finding | None:
    """Adding the known against unknown direction while the model answers.

    Args:
        store: Result store.

    Returns:
        The finding, or None when no P9d runs exist.
    """
    entries: dict[str, list[tuple[tuple[int, int, str], svg.Row, float]]] = {}
    for model in store.models("p9d"):
        runs = [
            r
            for r in store.runs("p9d", model)
            if (r / "summary.json").is_file()
        ]
        for run in runs:
            data = store.summary(run)
            head = data["headline"]
            template = data.get("template", "")
            label = f"{model} {template}" if len(runs) > 1 else model
            slope, reach = head["lat_slope"], head["max_abs_random_slope"]
            mark = svg.Mark(
                "slope",
                slope,
                head["lat_lo"],
                head["lat_hi"],
                hollow=not head["distinguishable"],
            )
            rank = (
                MODEL_ORDER.index(model) if model in MODEL_ORDER else 99,
                int(template != "fewshot"),
                template,
            )
            row = svg.Row(label, (mark,), (-reach, reach), _family(model))
            entries.setdefault(head["metric"], []).append((rank, row, reach))
    if not entries:
        return None
    styles = {"slope": svg.Style("s1", "circle", "Slope along the direction")}
    titles = {
        "entropy": "Entropy shift per unit coefficient (base checkpoints)",
        "abstention": "Abstention shift per unit coefficient (post-trained)",
    }
    panels: list[svg.DotPanel] = []
    for metric, title in titles.items():
        items = sorted(entries.get(metric, []), key=lambda e: e[0])
        if not items:
            continue
        rows = tuple(i[1] for i in items)
        values = [0.0]
        for row in rows:
            m = row.marks[0]
            values += [m.lo or 0.0, m.hi or 0.0, *(row.band or (0.0, 0.0))]
        domain, ticks = _axis(values)
        panels.append(
            svg.DotPanel(
                f"steer-{metric}",
                title,
                rows,
                styles,
                domain,
                ticks,
                refs=((0.0, "no effect"),),
                band_label="largest slope of three random directions",
                fmt="{:.2f}",
            )
        )
    runs_all = [e for items in entries.values() for e in items]
    clear = sum(not r[1].marks[0].hollow for r in runs_all)
    numbers = (
        f"The direction is distinguishable from three random directions in "
        f"{clear} of {len(runs_all)} runs.",
    )
    legend = _keys(
        [
            ("filled", "Distinguishable from random directions"),
            ("hollow", "Not distinguishable"),
            ("band", "Largest random-direction slope, both signs"),
        ],
    )
    figure = Figure(
        "Adding the known against unknown direction shifts entropy (base) "
        "or abstention (post-trained) more than random directions in most "
        "runs, and it also moves the known questions.",
        tuple(panels),
        legend,
        "Slope of the model's answer along the direction, per unit of "
        "coefficient. Each panel has its own axis: entropy and abstention "
        "are different quantities and are never put on one scale. Open "
        "markers are slopes that are not distinguishable from random "
        "directions.",
    )
    return Finding(
        "steering",
        "Pushing along the direction moves the answer, but not cleanly",
        "Adding the known against unknown direction while the model answers "
        "shifts its entropy (base) or abstention (post-trained) more than "
        "random directions in most runs, and also moves the known questions.",
        (figure,),
        numbers,
        (
            "Entropy and abstention differ, and each model's entropy has its "
            "own units, so slopes are not compared across rows.",
            "Distinguishable is judged against three random arms. In several "
            "runs a random direction moves the outcome as much.",
            "Known items move too, so this is not shown to be control over "
            "what the model knows. The abstention indicator is a phrase "
            "match with no human check.",
            "No perplexity or distinct-1 is stored for these runs, so the "
            "steered answers are never called fluent.",
        ),
        ("p9d",),
    )


FREE_CELL = {
    "condition": "free",
    "format": "fewshot",
    "shot_order": "range_last",
    "decoding": "greedy",
}


def _in_cell(rec: Record) -> bool:
    return all(rec.get(k) == v for k, v in FREE_CELL.items())


def _span(values: Sequence[float]) -> str:
    low, high = round(min(values)), round(max(values))
    return str(low) if low == high else f"{low} to {high}"


def _links_figure(store: Store) -> tuple[Figure, list[str]] | None:
    cos_rows: list[svg.Row] = []
    auc_rows: list[svg.Row] = []
    layers: set[int] = set()
    cos_out = auc_out = cells = 0
    for model in store.models("p10c"):
        run = store.latest("p10c", model, "summary.json")
        if run is None:
            continue
        summary = store.summary(run)
        layers.add(summary["layer"])
        post = is_post_trained(model)
        group = _family(model)
        cos_marks: list[svg.Mark] = []
        auc_marks: list[svg.Mark] = []
        cos_band = auc_band = None
        for name, link in summary["cosine"].items():
            cos_band = (link["band_lo"], link["band_hi"])
            cos_marks.append(svg.Mark("cosine", link["cosine"], hollow=post))
            cos_out += not cos_band[0] <= link["cosine"] <= cos_band[1]
            cells += 1
            known = summary["knowledge"][name]
            auc_band = (known["rand_lo"], known["rand_hi"])
            auc_marks.append(
                svg.Mark(
                    "auroc", known["auroc"], known["lo"], known["hi"], post
                )
            )
            auc_out += known["hi"] < auc_band[0] or known["lo"] > auc_band[1]
        cos_rows.append(svg.Row(model, tuple(cos_marks), cos_band, group))
        auc_rows.append(svg.Row(model, tuple(auc_marks), auc_band, group))
    if not cos_rows:
        return None
    layer = sorted(layers)[0] if len(layers) == 1 else None
    cos_values = [0.0]
    for row in cos_rows:
        cos_values += [m.value for m in row.marks] + list(row.band or ())
    domain, ticks = _axis(cos_values)
    where = f"layer {layer}" if layer is not None else "the selected layer"
    panels = (
        svg.DotPanel(
            "links-cosine",
            "Cosine with the P9 directions",
            tuple(cos_rows),
            {"cosine": svg.Style("s1", "circle", "One P9 type")},
            domain,
            ticks,
            refs=((0.0, "Orthogonal"),),
            band_label="random unit vectors (2.5th to 97.5th percentile)",
            fmt="{:.3f}",
            whiskers=False,
        ),
        svg.DotPanel(
            "links-auroc",
            "AUROC of the P9 readers",
            tuple(auc_rows),
            {"auroc": svg.Style("s1", "circle", "One P9 type")},
            (0.0, 1.0),
            (0.0, 0.5, 1.0),
            refs=((0.5, "Chance"),),
            band_label="random directions (95% interval)",
            whiskers=False,
        ),
    )
    numbers = [
        f"At {where}, {cos_out} of {cells} cosines between the range "
        "direction and a P9 direction lie outside the band of random "
        f"vectors, and {auc_out} of {cells} AUROC intervals lie outside "
        "the random-direction band."
    ]
    figure = Figure(
        "The range direction is nearly orthogonal to the P9 knowledge "
        "directions, and no P9 reader separates range from exact questions "
        "beyond its random band.",
        panels,
        _keys([("band", "Random directions")]),
        f"Each dot is one P9 type at {where}, the layer P10b selected. "
        "(a) Cosine between the range direction and the direction of the "
        "P9 reader. (b) AUROC of that reader on the 720 range and exact "
        "questions. Grey bands are the 95% range of the random-direction "
        "control. Open markers are post-trained checkpoints.",
        columns=2,
    )
    return figure, numbers


def approximate_answers(store: Store) -> Finding | None:
    """Range against exact quantities: behavior, readers and baselines.

    Args:
        store: Result store.

    Returns:
        The finding, or None when no P10 runs exist.
    """
    kinds = {
        "range": svg.Style("s1", "circle", "Questions with a sourced range"),
        "exact": svg.Style("s2", "square", "Questions with an exact value"),
        "people": svg.Style("s3", "triangle", "People heights"),
    }
    b_rows: list[svg.Row] = []
    people: list[tuple[float, float, float]] = []
    for model in store.models("p10a"):
        run = store.latest("p10a", model, "table.csv")
        if run is None:
            continue
        recs = [r for r in store.rows(run, "table.csv") if _in_cell(r)]

        def pick(
            family: str, kind: str, recs: list[Record] = recs
        ) -> Record | None:
            hits = [
                r for r in recs if r["family"] == family and r["kind"] == kind
            ]
            return hits[0] if hits else None

        post = is_post_trained(model)
        marks = []
        for key, family, kind in (
            ("range", "all", "range"),
            ("exact", "all", "exact"),
            ("people", "person_height", "exact"),
        ):
            rec = pick(family, kind)
            if rec is not None:
                marks.append(
                    svg.Mark(
                        key,
                        rec["range_rate"],
                        rec["range_lo"],
                        rec["range_hi"],
                        post,
                    )
                )
        if marks:
            b_rows.append(svg.Row(model, tuple(marks), group=_family(model)))
        diag = store.latest("p10a", model, "diagnostic.csv")
        if diag is not None and _family(model) == "qwen3":
            hit = [r for r in store.rows(diag, "diagnostic.csv") if _in_cell(r)]
            if hit:
                people.append(
                    (hit[0]["people"], hit[0]["n_people"], hit[0]["breeds"])
                )
    reader_styles = {
        READER: svg.Style("s1", "circle", "Reader"),
        "random": svg.Style("s0", "diamond", "Random direction"),
        "text_word": svg.Style("s5", "square", "Words only"),
    }
    r_rows: list[svg.Row] = []
    train, test = "height+wingspan", "duration"
    for model in store.models("p10b"):
        run = store.latest("p10b", model, "transfer.csv")
        if run is None:
            continue
        post = is_post_trained(model)
        marks = []
        for rec in store.rows(run, "transfer.csv"):
            if (
                rec["template"] == "concept"
                and rec["train_set"] == train
                and rec["test_family"] == test
                and rec["method"] in ("logistic", "random")
            ):
                key = READER if rec["method"] == "logistic" else "random"
                marks.append(
                    svg.Mark(key, rec["auroc"], rec["lo"], rec["hi"], post)
                )
        if (run / "baselines.csv").is_file():
            for rec in store.rows(run, "baselines.csv"):
                if (
                    rec["baseline"] == "text_word"
                    and rec["train_set"] == train
                    and rec["test_family"] == test
                ):
                    marks.append(
                        svg.Mark(
                            "text_word",
                            rec["auroc"],
                            rec["lo"],
                            rec["hi"],
                            post,
                        )
                    )
        if marks:
            r_rows.append(svg.Row(model, tuple(marks), group=_family(model)))
    read_low = min(
        0.4, min((m.value for r in r_rows for m in r.marks), default=0.4)
    )
    read_low = math.floor(read_low * 10) / 10
    read_domain = (read_low, 1.0)
    read_step = 0.2 if 1.0 - read_low > 0.5 else 0.1
    read_ticks = tuple(
        round(read_low + i * read_step, 6)
        for i in range(round((1.0 - read_low) / read_step) + 1)
    )
    link_figure = _links_figure(store)
    if not b_rows and not r_rows and link_figure is None:
        return None
    figures: list[Figure] = []
    if b_rows:
        figures.append(
            Figure(
                "Qwen3 checkpoints give a range on only part of the range "
                "questions, almost never on exact ones and never for people "
                "heights; GPT-2 sizes give ranges for exact questions too.",
                (
                    svg.DotPanel(
                        "approx-behavior",
                        "Share of answers that are a range",
                        tuple(b_rows),
                        kinds,
                        (0.0, 1.0),
                        (0.0, 0.25, 0.5, 0.75, 1.0),
                        stagger=True,
                        fmt="{:.3f}",
                    ),
                ),
                _keys(MODEL_KEYS[1:]),
                "Share of greedy answers that are a range, free few-shot "
                "prompt, range example last. Whiskers are Wilson 95% "
                "intervals. Open markers are post-trained checkpoints.",
            )
        )
    if r_rows:
        figures.append(
            Figure(
                "A reader separates range from exact questions on held-out "
                "durations, but so do a text-only baseline and a random "
                "direction.",
                (
                    svg.DotPanel(
                        "approx-reading",
                        "AUROC on held-out durations",
                        tuple(r_rows),
                        reader_styles,
                        read_domain,
                        read_ticks,
                        stagger=True,
                        refs=((0.5, "chance"),),
                    ),
                ),
                _keys(MODEL_KEYS[1:]),
                "AUROC on held-out duration questions, fitted on height and "
                "wingspan, concept template. Whiskers are item-bootstrap "
                "95% intervals.",
            )
        )
    if link_figure is not None:
        figures.append(link_figure[0])
    numbers = list(link_figure[1]) if link_figure else []
    if people:
        total = people[0][1]
        numbers.append(
            f"People heights, greedy free few-shot: a range on "
            f"{_span([p[0] for p in people])} of {total:.0f} answers on "
            f"every Qwen3 checkpoint, against {_span([p[2] for p in people])}"
            f" of {total:.0f} for dog breeds."
        )
    return Finding(
        "approximate",
        "Models hedge by topic, and readers do not beat simple text",
        "Qwen3 checkpoints give a range for part of the range questions and "
        "almost never for exact ones, and never for a person's height. A "
        "reader separates the two kinds of question, but so does a "
        "text-only baseline, and often a random direction too.",
        tuple(figures),
        tuple(numbers),
        (
            "The range side is living things and the exact side is "
            "artifacts, and the wording differs. People heights are a "
            "diagnostic, not a clean contrast.",
            "A reader claim needs the people diagnostic and the random "
            "band. An AUROC of 1.000 has a zero-width interval and is not "
            "certainty.",
            "GPT-2 sizes give ranges for exact questions too, so their "
            "hedging is not read as understanding. The parser has no human "
            "check, and it is one question set and one seed.",
            "The links to knowledge use the layer P10b selected, which is a "
            "layer-1 value, not a finding about where a link lives. The "
            "knowledge readers score no people, so the link cannot separate "
            "a range-valued answer from a living subject.",
        ),
        ("p10a", "p10b", "p10c"),
    )


def sentiment_steering(store: Store) -> Finding | None:
    """Sentiment steering with perplexity and repetition on the same axes.

    Args:
        store: Result store.

    Returns:
        The finding, or None when no C1 runs exist.
    """
    per_model: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for model in store.models("c1"):
        run = store.latest("c1", model, "table.json")
        if run is None:
            continue
        rows = store.rows(run, "table.json")
        none = [r for r in rows if r["arm"] == "none"]
        steered = [r for r in rows if r["arm"] == "linear"]
        if not none or not steered:
            continue
        base = none[0]
        series: dict[str, list[tuple[float, float]]] = {
            "score": [(0.0, base["score"])],
            "ratio": [(0.0, 1.0)],
            "distinct": [(0.0, base["distinct_1"])],
        }
        for coeff in sorted({r["coeff"] for r in steered}):
            cell = [r for r in steered if r["coeff"] == coeff]
            series["score"].append((coeff, _mean(cell, "score")))
            series["ratio"].append(
                (coeff, _mean(cell, "perplexity") / base["perplexity"])
            )
            series["distinct"].append((coeff, _mean(cell, "distinct_1")))
        for values in series.values():
            values.sort()
        per_model[model] = series
    if not per_model:
        return None
    figures: list[Figure] = []
    all_panels: list[svg.LinePanel] = []
    numbers: list[str] = []
    groups = (("qwen3", "Qwen3"), ("gpt2", "GPT-2"))
    for family, name in groups:
        models = [m for m in per_model if _family(m) == family]
        if not models:
            continue
        sizes = list(dict.fromkeys(_size_key(m) for m in models))

        def lines(
            key: str, models: list[str] = models, sizes: list[str] = sizes
        ) -> tuple[svg.LineSeries, ...]:
            return tuple(
                svg.LineSeries(
                    m,
                    m,
                    STYLE_CLASSES[sizes.index(_size_key(m))],
                    tuple(per_model[m][key]),
                    dashed=is_post_trained(m),
                )
                for i, m in enumerate(models)
            )

        ratios = [y for m in models for _, y in per_model[m]["ratio"]]
        top = next(
            (t for t in (2, 5, 10, 20, 50, 100, 200) if t >= max(ratios)),
            math.ceil(max(ratios)),
        )
        distincts = [y for m in models for _, y in per_model[m]["distinct"]]
        d_top = math.ceil(max(distincts) * 20) / 20
        coeffs = sorted({x for m in models for x, _ in per_model[m]["score"]})
        x_domain = (coeffs[0], coeffs[-1])
        x_ticks = tuple(coeffs)
        panels = (
            svg.LinePanel(
                f"c1-{family}-score",
                f"{name}: sentiment score",
                lines("score"),
                y_domain=(0.0, 1.0),
                y_ticks=(0.0, 0.5, 1.0),
                x_label="relative coefficient",
                fmt="{:.2f}",
                x_domain=x_domain,
                x_ticks=x_ticks,
            ),
            svg.LinePanel(
                f"c1-{family}-ppl",
                f"{name}: perplexity ratio",
                lines("ratio"),
                y_domain=(0.8, float(top)),
                y_ticks=tuple(
                    t for t in (1, 2, 5, 10, 20, 50, 100, 200) if t <= top
                ),
                y_log=True,
                x_label="relative coefficient",
                refs=((1.5, "limit 1.5x"),),
                fmt="{:.2f}",
                x_domain=x_domain,
                x_ticks=x_ticks,
            ),
            svg.LinePanel(
                f"c1-{family}-distinct",
                f"{name}: distinct-1",
                lines("distinct"),
                y_domain=(0.0, d_top),
                y_ticks=tuple(
                    round(i * 0.05, 2) for i in range(round(d_top / 0.05) + 1)
                ),
                x_label="relative coefficient",
                refs=((0.15, "limit 0.15"),),
                fmt="{:.3f}",
                x_domain=x_domain,
                x_ticks=x_ticks,
            ),
        )
        all_panels.extend(panels)
        for m in models:
            ok = [
                x
                for (x, r), (_, d) in zip(
                    per_model[m]["ratio"], per_model[m]["distinct"], strict=True
                )
                if r <= 1.5 and d >= 0.15 and x != 0.0
            ]
            span = f"{min(ok):+g} to {max(ok):+g}" if ok else "none"
            numbers.append(f"{m}: coefficients that stay fluent: {span}.")
    figures.append(
        Figure(
            "A small push moves the sentiment score while perplexity and "
            "repetition stay near normal; larger pushes do not.",
            tuple(all_panels),
            _keys([("dashed", "Post-trained checkpoint (dashed)")]),
            "Sentiment steering, mean over three steering readers. Each "
            "hue is a model size, lighter for smaller; solid lines are base "
            "or GPT-2 checkpoints and dashed lines are post-trained ones. "
            "Perplexity is relative to the unsteered text (log axis). The "
            "dotted limits, 1.5x and 0.15, are a descriptive rule of the "
            "write-up: text counts as fluent only when both hold.",
            columns=3,
        )
    )
    return Finding(
        "fluency",
        "A small push steers sentiment and stays fluent; bigger pushes score "
        "high on repetitive text",
        "Adding the sentiment direction at a small coefficient moves a "
        "sentiment judge the intended way while perplexity and repetition "
        "stay near normal. At larger coefficients the score keeps rising "
        "while the text turns repetitive.",
        tuple(figures),
        tuple(numbers),
        (
            "A score is shown only with perplexity and distinct-1 in the "
            "same figure. A high score at a coefficient that fails the "
            "limits is the score of repetitive text.",
            "The two judges are on different scales, so only arms scored by "
            "the same judge are compared.",
            "Base and post-trained checkpoints are separate lines. This "
            "experiment has no random-direction arm.",
            "Earlier steering runs on GPT-2 store no fluency measures and "
            "are not charted.",
        ),
        ("c1",),
    )


def template_sensitivity(store: Store) -> Finding | None:
    """Spread of one fixed reader across prompt wordings.

    Args:
        store: Result store.

    Returns:
        The finding, or None when no P1 runs exist.
    """
    styles = {
        "variant": svg.Style("s1", "circle", "Prompt variant"),
        "paper": svg.Style("s2", "diamond", "Paper template"),
    }
    rows: list[svg.Row] = []
    numbers: list[str] = []
    for model in store.models("p1"):
        run = store.latest("p1", model, "table.json")
        if run is None:
            continue
        cells = [
            r
            for r in store.rows(run, "table.json")
            if r["task"] == "sentiment" and r["variant"] != "no_trailing_space"
        ]
        paper = [r for r in cells if r["variant"] == "paper"]
        if not paper:
            continue
        post = is_post_trained(model)

        def mark(rec: Record, key: str, post: bool = post) -> svg.Mark:
            tip = (
                f"{rec['variant']} {rec['report']:.3f} "
                f"[{rec['lo']:.3f}, {rec['hi']:.3f}]"
            )
            return svg.Mark(key, rec["report"], rec["lo"], rec["hi"], post, tip)

        marks = [mark(r, "variant") for r in cells if r["variant"] != "paper"]
        marks.append(mark(paper[0], "paper"))
        randoms = [r["random"] for r in cells]
        rows.append(
            svg.Row(
                model,
                tuple(marks),
                (min(randoms), max(randoms)),
                _family(model),
            )
        )
        low = min(cells, key=lambda r: r["report"])
        high = max(cells, key=lambda r: r["report"])
        numbers.append(
            f"{model}: paper template {paper[0]['report']:.3f}, lowest "
            f"{low['variant']} {low['report']:.3f}, highest {high['variant']} "
            f"{high['report']:.3f}, span {high['report'] - low['report']:.3f}."
        )
    if not rows:
        return None
    panel = svg.DotPanel(
        "templates",
        "Sentiment accuracy of one reader across prompt wordings",
        tuple(rows),
        styles,
        (0.4, 1.0),
        (0.4, 0.6, 0.8, 1.0),
        refs=((0.5, "chance"),),
        band_label="random direction across the same variants",
        stagger=True,
        whiskers=False,
    )
    figure = Figure(
        "Changing only the wording of the prompt moves a fixed sentiment "
        "reader by a tenth to a fifth of accuracy, and the paper template is "
        "neither the best nor the worst.",
        (panel,),
        _keys([*MODEL_KEYS[1:], ("band", "Random direction (range)")]),
        "Accuracy of one sentiment reader for each prompt variant (blue) "
        "and the paper template (vermillion). The grey band is the "
        "range of a random direction across the same variants. Open "
        "markers are post-trained checkpoints.",
    )
    return Finding(
        "templates",
        "The reading depends on how the prompt is worded",
        "Changing only the wording of the prompt moves a fixed sentiment "
        "reader by a tenth to a fifth of accuracy, and the paper template "
        "is neither the best nor the worst.",
        (figure,),
        tuple(numbers),
        (
            "A span takes the extremes of many variants, so it exceeds the "
            "difference between two variants chosen in advance.",
            "Intervals cover the training draw only. Every cell is one run, "
            "and one layer is used for all variants.",
            "The table does not say which aspect of the prompt drives a "
            "difference.",
            "Base and post-trained checkpoints are not pooled.",
        ),
        ("p1",),
    )


def finder_comparison(store: Store) -> Finding | None:
    """Direction finders against a random direction on GPT-2 and Qwen3.

    Args:
        store: Result store.

    Returns:
        The finding, or None when no P4 runs exist.
    """
    styles = {
        "random": svg.Style("s0", "diamond", "Random direction"),
        "pca_diff": svg.Style(
            "s2", "square", "Paper finder (PCA of differences)"
        ),
        "logistic": svg.Style("s1", "circle", "Logistic regression"),
    }
    tasks = (
        ("sentiment", "Sentiment", "label_contrastive"),
        ("truth", "Truth", "label_contrastive"),
        ("utility", "Utility", "matched"),
    )
    runs = {m: store.latest("p4", m, "table.json") for m in store.models("p4")}
    panels: list[svg.DotPanel] = []
    numbers: list[str] = []
    for task, name, pairing in tasks:
        rows: list[svg.Row] = []
        wins = {"gpt2": 0, "qwen3": 0}
        totals = {"gpt2": 0, "qwen3": 0}
        for model, run in runs.items():
            if run is None:
                continue
            cells = [
                r
                for r in store.rows(run, "table.json")
                if r["task"] == task
                and r["pairing"] == pairing
                and r["preprocess"] == "center"
            ]
            by = {r["method"]: r for r in cells}
            marks = tuple(
                svg.Mark(
                    k,
                    by[k]["report"],
                    by[k]["lo"],
                    by[k]["hi"],
                    is_post_trained(model),
                )
                for k in styles
                if k in by
            )
            if marks:
                rows.append(svg.Row(model, marks, group=_family(model)))
            if "pca_diff" in by and "random" in by:
                totals[_family(model)] += 1
                wins[_family(model)] += (
                    by["pca_diff"]["report"] > by["random"]["report"]
                )
        if not rows:
            continue
        panels.append(
            svg.DotPanel(
                f"finders-{task}",
                f"{name} ({pairing.replace('_', '-')} pairs)",
                tuple(rows),
                styles,
                (0.4, 1.0),
                (0.4, 0.6, 0.8, 1.0),
                stagger=True,
                refs=((0.5, "chance"),),
            )
        )
        numbers.append(
            f"{name}: the paper finder is above the random direction on "
            f"{wins['gpt2']} of {totals['gpt2']} GPT-2 and "
            f"{wins['qwen3']} of {totals['qwen3']} Qwen3 checkpoints."
        )
    if not panels:
        return None
    figure = Figure(
        "On sentiment and truth the paper finder is not above a random "
        "direction on any GPT-2 size, while logistic regression is.",
        tuple(panels),
        _keys(MODEL_KEYS[1:]),
        "Test accuracy in the reference cell of each task, centered "
        "activations. Whiskers are 95% intervals over ten seeds. Open "
        "markers are post-trained checkpoints.",
    )
    return Finding(
        "finders",
        "The paper's PCA finder does not read sentiment or truth on GPT-2",
        "On sentiment and truth, the paper's PCA-of-differences finder is not "
        "above a random direction on any GPT-2 size, while logistic "
        "regression is clearly above it. On utility it is above the random "
        "direction but below logistic regression. Larger Qwen3 models close "
        "part of the gap.",
        (figure,),
        tuple(numbers),
        (
            "A random direction is chosen and oriented the same way as the "
            "others, so it is the baseline, not 0.5.",
            "Intervals are over ten seeds and each cell is one run. The "
            "result is about these data and pairings, not a claim that the "
            "paper's method is wrong.",
            "Reading rank is not steering rank: the finder that reads best "
            "does not always steer best.",
        ),
        ("p4",),
    )


def truthfulqa(store: Store) -> Finding | None:
    """TruthfulQA multiple choice against text-only baselines.

    Args:
        store: Result store.

    Returns:
        The finding, or None when no R6 runs exist.
    """
    styles = {
        READER: svg.Style("s1", "circle", "Reader at the validation layer"),
        "standard": svg.Style("s2", "diamond", "Standard log-probability"),
    }
    per_model: dict[str, list[Record]] = {}
    text: dict[str, Record] = {}
    for model in store.models("r6"):
        run = store.latest("r6", model, "table.json")
        if run is None:
            continue
        rows = store.rows(run, "table.json")
        per_model[model] = rows
        for rec in rows:
            if rec["method"] in ("text_word", "text_char"):
                text[str(rec["method"])] = rec
    if not per_model:
        return None
    band = None
    if text:
        band = (
            min(r["lo"] for r in text.values()),
            max(r["hi"] for r in text.values()),
        )
    rows_out: list[svg.Row] = []
    inside = 0
    for model, rows in per_model.items():
        reader = [
            r
            for r in rows
            if r["method"] == "lat"
            and r["layer_rule"] == "validation-layer"
            and r["source"] == "truthfulqa"
        ]
        standard = [r for r in rows if r["method"] == "standard"]
        marks = []
        post = is_post_trained(model)
        if reader:
            r = reader[0]
            marks.append(svg.Mark(READER, r["mc1"], r["lo"], r["hi"], post))
            if band and band[0] <= r["mc1"] <= band[1]:
                inside += 1
        if standard:
            marks.append(svg.Mark("standard", standard[0]["mc1"], hollow=post))
        if marks:
            rows_out.append(svg.Row(model, tuple(marks), band, _family(model)))
    panel = svg.DotPanel(
        "truthfulqa",
        "TruthfulQA MC1 accuracy on the held-out questions",
        tuple(rows_out),
        styles,
        (0.1, 0.9),
        (0.2, 0.4, 0.6, 0.8),
        band_label="range of the two text-only baselines (Wilson 95%)",
    )
    numbers = []
    if text:
        word, char = text.get("text_word"), text.get("text_char")
        if word and char:
            numbers.append(
                f"Text-only baselines: words {word['mc1']:.3f} "
                f"[{word['lo']:.3f}, {word['hi']:.3f}], characters "
                f"{char['mc1']:.3f} [{char['lo']:.3f}, {char['hi']:.3f}]."
            )
        numbers.append(
            f"The reader falls inside that band on {inside} of "
            f"{len(rows_out)} checkpoints."
        )
    figure = Figure(
        "A classifier that sees only the answer words scores about as well "
        "as the reader, which sits inside the text-only band on most "
        "checkpoints.",
        (panel,),
        _keys([*MODEL_KEYS[1:], ("band", "Text-only baselines")]),
        "TruthfulQA multiple-choice accuracy on held-out questions. Reader "
        "whiskers are intervals over 15 readers; the grey band is a "
        "Wilson interval over questions. They are different kinds of "
        "interval, so compare positions, not widths.",
    )
    return Finding(
        "truthfulqa",
        "On TruthfulQA the reader does not clearly beat text-only baselines",
        "A reader trained on TruthfulQA questions scores well, but a "
        "classifier that sees only the answer words scores about as well, "
        "and the reader sits inside its interval on most checkpoints.",
        (figure,),
        tuple(numbers),
        (
            "Only the validation-layer row is shown. The best-layer row "
            "picks its layer on the test questions and is a ceiling.",
            "The text baselines do not depend on the model, so they are "
            "read from any run that stores them.",
            "Honesty transfer is a related negative result: it is at or "
            "below chance on several models while the reader seeds agree "
            "closely.",
        ),
        ("r6",),
    )


QUESTIONS = {
    "r1": "Can a sentiment direction separate SST-2 phrases?",
    "r2": "Does a direction track pleasantness (ETHICS utility)?",
    "r3": "Does a direction track morality (ETHICS commonsense)?",
    "r4": "Can six emotions be read one against the rest?",
    "r5": "Does an honesty direction from personas transfer to statements?",
    "r6": "Does a truth reader beat text-only baselines on TruthfulQA?",
    "r7": "Does a truth reader hold on out-of-distribution fact sets?",
    "a1": "Do different seeds find the same direction?",
    "a2": "What tokens does a direction point at (logit lens)?",
    "a5": "How do directions relate to sparse autoencoder features?",
    "p1": "How much does the prompt template change the reading?",
    "p2": "Does token position or a misleading sentence change it?",
    "p3": "How many training pairs does a reader need?",
    "p4": "Which finder, pairing and preprocessing work?",
    "p5": "Do persona wording and instructions confound honesty?",
    "p6": "How does accuracy scale per decade of parameters?",
    "p7": "Does the reading template change steering?",
    "p8": "What changes between base and post-trained models?",
    "c1": "Does adding a sentiment direction steer fluent text?",
    "c2": "Does adding an emotion direction steer emotion?",
    "c4": "Can steering or LoRRA make a model more truthful?",
    "c5": "Can directions detect lies in a gallery on TriviaQA?",
    "m1": "How do six finders compare on read, steer and erase?",
    "lr1": "Can a LoRRA adapter be trained along a direction?",
    "p9a": "Which questions does a model know, from its behavior?",
    "p9b": "Is there a known against unknown direction?",
    "p9c": "Does it predict errors beyond the model's confidence?",
    "p9d": "What happens when the model is steered along it?",
    "p10a": "Does the model answer with ranges or points?",
    "p10b": "Can a reader tell range from exact questions?",
    "p10c": "Does the range direction link to knowledge and hedging?",
}

EXPECTED = {"p6": 1, "p8": 1, "p9d": 9}


def _size(count: float) -> str:
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    return f"{count / 1e6:.0f}M"


def _params(store: Store) -> dict[str, str]:
    out: dict[str, str] = {}
    run = store.latest("p6", "gpt2", "table.json")
    if run is not None:
        for rec in store.rows(run, "table.json"):
            if _is_number(rec.get("params")):
                out[str(rec["model"])] = _size(rec["params"])
    return out


def overview(store: Store) -> Overview:
    """Builds the model and experiment tables from the results tree.

    Args:
        store: Result store.

    Returns:
        Tables for the overview sections.
    """
    experiments = []
    seen: dict[str, int] = {}
    for key, question in QUESTIONS.items():
        models = store.models(key)
        for model in models:
            seen[model] = seen.get(model, 0) + 1
        expected = EXPECTED.get(key, len(MODEL_ORDER))
        status = (
            "pending"
            if not models
            else "done"
            if len(models) >= expected
            else "partial"
        )
        experiments.append(ExperimentInfo(key, question, len(models), status))
    params = _params(store)
    infos = []
    for model in sorted(
        seen, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99
    ):
        match = re.match(r"qwen3-([\d.]+b)", model)
        nominal = match.group(1).upper() if match else "unknown"
        blocks = None
        run = store.latest("r3", model, "curves.json")
        if run is not None:
            blocks = store.data(run, "curves.json").get("n_layers")
        role = (
            "pretrained"
            if _family(model) == "gpt2"
            else "post-trained"
            if is_post_trained(model)
            else "base"
        )
        infos.append(
            ModelInfo(
                model, role, params.get(model, nominal), blocks, seen[model]
            )
        )
    return Overview(tuple(infos), tuple(experiments))


BUILDERS: tuple[Callable[[Store], Finding | None], ...] = (
    scaling,
    knowledge,
    steering_along_direction,
    approximate_answers,
    sentiment_steering,
    template_sensitivity,
    finder_comparison,
    truthfulqa,
)


def build_findings(store: Store) -> list[Finding]:
    """Builds every headline finding that has data.

    Args:
        store: Result store.

    Returns:
        Findings in page order.
    """
    found = (build(store) for build in BUILDERS)
    return [f for f in found if f is not None]
