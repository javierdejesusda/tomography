import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tomography.site import findings
from tomography.site import panels as shapes
from tomography.site import store as store_module


def write_run(
    root: Path,
    exp: str,
    model: str,
    digest: str,
    files: dict[str, Any],
    started: str = "2026-10-16T00:00:00+00:00",
) -> Path:
    run = root / exp / model / digest
    run.mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps({"started_at": started}))
    for name, content in files.items():
        if name.endswith(".json"):
            payload = {"_meta": {"experiment": exp}, "data": content}
            (run / name).write_text(json.dumps(payload))
        elif name.endswith(".csv"):
            with (run / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(content[0]))
                writer.writeheader()
                writer.writerows(content)
        else:
            np.savez(run / name, **content)
    return run


def table(columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    return {"columns": columns, "rows": rows}


def curves(layers: list[int], random_mean: float, lo: float, hi: float) -> Any:
    n = len(layers)
    shape = (3, n)
    return {
        "curves.json": {"series": ["reader", "random", "shuffled"]},
        "curves.npz": {
            "x": np.array(layers),
            "mean": np.full(shape, random_mean),
            "lo": np.full(shape, lo),
            "hi": np.full(shape, hi),
        },
    }


def reader_runs(root: Path) -> None:
    for exp, model, layer, acc, ci, digest in [
        ("r1", "gpt2", 2, 0.77, [0.75, 0.79], "a1"),
        ("r1", "qwen3-0.6b-base", 3, 0.95, [0.94, 0.96], "a2"),
        ("r1", "qwen3-0.6b", 3, 0.86, [0.84, 0.88], "a3"),
    ]:
        files = {
            "summary.json": {
                "best_layer": layer,
                "best_accuracy": acc,
                "ci": ci,
            },
            **curves([1, 2, 3], 0.5, 0.48, 0.55),
        }
        write_run(root, exp, model, digest, files)
    for exp in ("r2", "r3"):
        files = {
            "summary.json": {
                "best_layer": 2,
                "test_accuracy": 0.66,
                "ci": [0.64, 0.68],
            },
            **curves([1, 2, 3], 0.51, 0.49, 0.53),
        }
        write_run(root, exp, "gpt2", "b1", files)
    files = {
        "summary.json": {
            "val_layer": 2,
            "sets": {
                "test": {
                    "accuracy_at_val_layer": 0.61,
                    "ci_at_val_layer": [0.58, 0.64],
                }
            },
        },
        **curves([1, 2, 3], 0.52, 0.5, 0.56),
    }
    write_run(root, "r7", "gpt2", "c1", files)


def test_store_orders_models_and_picks_the_latest_run(tmp_path: Path) -> None:
    write_run(
        tmp_path, "r1", "qwen3-0.6b", "old", {"summary.json": {}}, "2026-01-01"
    )
    write_run(
        tmp_path,
        "r1",
        "qwen3-0.6b",
        "new",
        {"summary.json": {"v": 2}},
        "2026-02-01",
    )
    write_run(tmp_path, "r1", "gpt2-xl", "x", {"summary.json": {}})
    write_run(tmp_path, "r1", "gpt2", "y", {"summary.json": {}})
    store = store_module.Store(tmp_path)
    assert store.models("r1") == ["gpt2", "gpt2-xl", "qwen3-0.6b"]
    run = store.latest("r1", "qwen3-0.6b", "summary.json")
    assert run is not None
    assert run.name == "new"
    assert store.summary(run) == {"v": 2}
    assert store.latest("r1", "gpt2", "table.json") is None
    assert store.models("missing") == []


def test_store_reads_tables_in_every_stored_shape(tmp_path: Path) -> None:
    cols = table(["a", "b"], [[1, 2], [3, None]])
    run = write_run(
        tmp_path,
        "e",
        "gpt2",
        "h",
        {
            "table.json": cols,
            "list.json": [{"a": 1}],
            "rows.csv": [{"a": "1.5", "b": ""}, {"a": "x", "b": "2"}],
        },
    )
    store = store_module.Store(tmp_path)
    assert store.rows(run, "table.json") == [
        {"a": 1, "b": 2},
        {"a": 3, "b": None},
    ]
    assert store.rows(run, "list.json") == [{"a": 1}]
    assert store.rows(run, "rows.csv") == [
        {"a": 1.5, "b": None},
        {"a": "x", "b": 2.0},
    ]


def test_store_reads_a_curve_series_at_a_layer(tmp_path: Path) -> None:
    run = write_run(
        tmp_path, "r3", "gpt2", "h", curves([1, 2, 3], 0.5, 0.4, 0.6)
    )
    store = store_module.Store(tmp_path)
    assert store.curve_at(run, "curves", "random", 2) == (0.5, 0.4, 0.6)
    assert store.curve_at(run, "curves", "random", 99) is None


def test_scaling_pairs_each_reader_with_its_random_band(tmp_path: Path) -> None:
    reader_runs(tmp_path)
    finding = findings.scaling(store_module.Store(tmp_path))
    assert finding is not None
    sentiment = finding.figures[0].panels[0]
    assert isinstance(sentiment, shapes.DotPanel)
    assert [r.label for r in sentiment.rows] == [
        "gpt2",
        "qwen3-0.6b-base",
        "qwen3-0.6b",
    ]
    gpt2, base, post = sentiment.rows
    assert gpt2.marks[0].value == 0.77
    assert (gpt2.marks[0].lo, gpt2.marks[0].hi) == (0.75, 0.79)
    assert gpt2.band == (0.48, 0.55)
    assert [
        m.hollow for m in (gpt2.marks[0], base.marks[0], post.marks[0])
    ] == [
        False,
        False,
        True,
    ]
    titles = [p.title for p in finding.figures[0].panels]
    assert any("Truth" in t for t in titles)
    assert any("random" in line.lower() for line in finding.numbers)


def test_scaling_is_absent_without_reader_runs(tmp_path: Path) -> None:
    assert findings.scaling(store_module.Store(tmp_path)) is None


def test_knowledge_compares_reader_to_popularity_and_text(
    tmp_path: Path,
) -> None:
    cols_b = [
        "type",
        "logistic_within",
        "lo_within",
        "hi_within",
        "popularity_within",
        "text_word_within",
        "text_char_within",
    ]
    for model, reader in [("gpt2", None), ("gpt2-xl", 0.9), ("qwen3-4b", 0.93)]:
        rows = [
            ["obscure_real", reader, 0.88, 0.92, 0.8, 0.5, 0.55],
            ["other", 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ]
        write_run(
            tmp_path, "p9b", model, "b", {"table.json": table(cols_b, rows)}
        )
    cols_c = ["dataset", "predictor", "auroc_within", "lo_within", "hi_within"]
    for model in ("gpt2-xl", "qwen3-4b"):
        rows = [
            ["popqa", "answer_logprob", 0.86, 0.84, 0.88],
            ["popqa", "lat", 0.77, 0.75, 0.79],
            ["popqa", "probe", 0.81, 0.79, 0.83],
            ["popqa", "popularity", 0.68, 0.66, 0.7],
            ["popqa", "template", 0.5, 0.5, 0.5],
            ["triviaqa", "lat", 0.1, 0.1, 0.1],
        ]
        write_run(
            tmp_path, "p9c", model, "c", {"table.json": table(cols_c, rows)}
        )
    inc = table(
        ["dataset", "features", "diff_vs_confidence", "lo", "hi"],
        [["popqa", "confidence+lat", 0.011, -0.006, 0.032]],
    )
    write_run(
        tmp_path, "p9c", "gpt2-large", "c", {"table_incremental.json": inc}
    )
    finding = findings.knowledge(store_module.Store(tmp_path))
    assert finding is not None
    read_panel, err_panel = finding.figures[0].panels
    assert len(finding.figures) == 1
    assert [r.label for r in read_panel.rows] == ["gpt2-xl", "qwen3-4b"]
    assert {m.series for m in read_panel.rows[0].marks} == {
        "reader",
        "popularity",
        "text_word",
        "text_char",
    }
    series = {m.series for m in err_panel.rows[0].marks}
    assert series == {"answer_logprob", "probe", "lat", "popularity"}
    assert any("+0.011" in line for line in finding.numbers)


def test_steering_forest_separates_entropy_from_abstention(
    tmp_path: Path,
) -> None:
    def headline(metric: str, slope: float, dist: bool) -> dict[str, Any]:
        return {
            "template": "fewshot",
            "headline": {
                "metric": metric,
                "lat_slope": slope,
                "lat_lo": slope - 0.1,
                "lat_hi": slope + 0.1,
                "max_abs_random_slope": 0.4,
                "distinguishable": dist,
                "selective": True,
            },
        }

    write_run(
        tmp_path,
        "p9d",
        "qwen3-1.7b-base",
        "a",
        {"summary.json": headline("entropy", -5.0, True)},
    )
    write_run(
        tmp_path,
        "p9d",
        "qwen3-1.7b",
        "b",
        {"summary.json": headline("abstention", 0.04, False)},
    )
    write_run(
        tmp_path,
        "p9d",
        "qwen3-1.7b",
        "c",
        {
            "summary.json": {
                **headline("abstention", -1.1, True),
                "template": "concept",
            }
        },
    )
    finding = findings.steering_along_direction(store_module.Store(tmp_path))
    assert finding is not None
    panels = finding.figures[0].panels
    assert [p.title.split(" ")[0] for p in panels] == ["Entropy", "Abstention"]
    entropy, abstention = panels
    assert entropy.rows[0].band == (-0.4, 0.4)
    assert entropy.rows[0].marks[0].hollow is False
    labels = [r.label for r in abstention.rows]
    assert labels == ["qwen3-1.7b fewshot", "qwen3-1.7b concept"]
    assert abstention.rows[0].marks[0].hollow is True


def test_approximate_answers_reports_people_and_the_random_control(
    tmp_path: Path,
) -> None:
    cols = ["family", "kind", "condition", "format", "shot_order", "decoding"]
    base = {
        "condition": "free",
        "format": "fewshot",
        "shot_order": "range_last",
        "decoding": "greedy",
    }

    def row(family: str, kind: str, rate: float) -> dict[str, Any]:
        return {
            "family": family,
            "kind": kind,
            **base,
            "range_rate": rate,
            "range_lo": rate - 0.05,
            "range_hi": rate + 0.05,
        }

    table_csv = [
        row("all", "range", 0.5),
        row("all", "exact", 0.0),
        row("person_height", "exact", 0.0),
        {**row("all", "range", 0.9), "decoding": "sampled"},
    ]
    assert cols[0] in table_csv[0]
    diag = [
        {
            **base,
            "n_people": 120,
            "people": 0,
            "n_breeds": 120,
            "breeds": 100,
        }
    ]
    write_run(
        tmp_path,
        "p10a",
        "qwen3-4b",
        "a",
        {"table.csv": table_csv, "diagnostic.csv": diag},
    )
    transfer = [
        {
            "template": "concept",
            "method": method,
            "train_set": "height+wingspan",
            "test_family": "duration",
            "auroc": value,
            "lo": value - 0.01,
            "hi": min(value + 0.01, 1.0),
        }
        for method, value in [
            ("logistic", 0.99),
            ("random", 0.95),
            ("pca_diff", 0.5),
        ]
    ]
    baselines = [
        {
            "baseline": "text_word",
            "train_set": "height+wingspan",
            "test_family": "duration",
            "auroc": 0.93,
            "lo": 0.83,
            "hi": 1.0,
        }
    ]
    write_run(
        tmp_path,
        "p10b",
        "qwen3-4b",
        "b",
        {"transfer.csv": transfer, "baselines.csv": baselines},
    )
    finding = findings.approximate_answers(store_module.Store(tmp_path))
    assert finding is not None
    behavior = finding.figures[0].panels[0]
    assert {m.series for m in behavior.rows[0].marks} == {
        "range",
        "exact",
        "people",
    }
    assert behavior.rows[0].marks[0].value == 0.5
    reading = finding.figures[1].panels[0]
    values = {m.series: m.value for m in reading.rows[0].marks}
    assert values == {"reader": 0.99, "random": 0.95, "text_word": 0.93}
    assert any("0 of 120" in line for line in finding.numbers)


def test_sentiment_steering_pairs_score_with_fluency(tmp_path: Path) -> None:
    cols = ["arm", "coeff", "seed", "score", "perplexity", "distinct_1"]
    rows = [["none", 0.0, None, 0.6, 50.0, 0.3]]
    for coeff, score, ppl, dist in [
        (-0.1, 0.3, 60.0, 0.28),
        (0.1, 0.8, 100.0, 0.2),
    ]:
        for seed, wobble in ((0, 0.0), (1, 0.1)):
            rows.append(["linear", coeff, seed, score + wobble, ppl, dist])
    write_run(tmp_path, "c1", "gpt2", "a", {"table.json": table(cols, rows)})
    finding = findings.sentiment_steering(store_module.Store(tmp_path))
    assert finding is not None
    score, ppl, distinct = finding.figures[0].panels[:3]
    assert finding.figures[0].columns == 3
    assert isinstance(score, shapes.LinePanel)
    (line,) = score.series
    assert [x for x, _ in line.points] == [-0.1, 0.0, 0.1]
    assert [y for _, y in line.points] == pytest.approx([0.35, 0.6, 0.85])
    (ratio,) = ppl.series
    assert ratio.points[1] == (0.0, 1.0)
    assert ratio.points[2][1] == pytest.approx(2.0)
    assert ppl.y_log
    assert (1.5, "limit 1.5x") in ppl.refs
    assert any(v == 0.15 for v, _ in distinct.refs)


def test_template_sensitivity_drops_the_variant_missing_on_post_models(
    tmp_path: Path,
) -> None:
    def rows(with_space: bool) -> list[dict[str, Any]]:
        out = []
        for variant, report in [
            ("paper", 0.8),
            ("bare", 0.6),
            ("no_trailing_space", 0.9),
        ]:
            if variant == "no_trailing_space" and not with_space:
                continue
            out.append(
                {
                    "task": "sentiment",
                    "variant": variant,
                    "report": report,
                    "lo": report - 0.02,
                    "hi": report + 0.02,
                    "random": 0.5,
                }
            )
        out.append(
            {
                "task": "truth",
                "variant": "paper",
                "report": 0.7,
                "lo": 0.6,
                "hi": 0.8,
                "random": 0.5,
            }
        )
        return out

    write_run(tmp_path, "p1", "gpt2", "a", {"table.json": rows(True)})
    write_run(tmp_path, "p1", "qwen3-0.6b", "b", {"table.json": rows(False)})
    finding = findings.template_sensitivity(store_module.Store(tmp_path))
    assert finding is not None
    (panel,) = finding.figures[0].panels
    for row in panel.rows:
        assert [m.value for m in row.marks if m.series == "variant"] == [0.6]
        assert [m.value for m in row.marks if m.series == "paper"] == [0.8]
        assert row.band == (0.5, 0.5)
    assert any("0.200" in line for line in finding.numbers)


def test_finder_comparison_uses_the_reference_cell(tmp_path: Path) -> None:
    def cell(
        task: str, method: str, pairing: str, pre: str, report: float
    ) -> dict[str, Any]:
        return {
            "task": task,
            "method": method,
            "pairing": pairing,
            "preprocess": pre,
            "report": report,
            "lo": report - 0.01,
            "hi": report + 0.01,
        }

    rows = []
    for task, pairing in [
        ("sentiment", "label_contrastive"),
        ("truth", "label_contrastive"),
        ("utility", "matched"),
    ]:
        for method, value in [
            ("random", 0.5),
            ("pca_diff", 0.51),
            ("logistic", 0.8),
        ]:
            rows.append(cell(task, method, pairing, "center", value))
            rows.append(cell(task, method, pairing, "unit", 0.99))
            rows.append(cell(task, method, "other", "center", 0.98))
    write_run(tmp_path, "p4", "gpt2", "a", {"table.json": rows})
    finding = findings.finder_comparison(store_module.Store(tmp_path))
    assert finding is not None
    panels = finding.figures[0].panels
    assert [p.title.split(" ")[0] for p in panels] == [
        "Sentiment",
        "Truth",
        "Utility",
    ]
    values = {m.series: m.value for m in panels[0].rows[0].marks}
    assert values == {"random": 0.5, "pca_diff": 0.51, "logistic": 0.8}


def test_truthfulqa_draws_the_text_baseline_band(tmp_path: Path) -> None:
    cols = ["method", "layer_rule", "source", "mc1", "lo", "hi"]

    def rows(with_text: bool) -> list[list[Any]]:
        out: list[list[Any]] = [
            ["standard", None, None, 0.3, 0.25, 0.39],
            ["lat", "validation-layer", "truthfulqa", 0.7, 0.69, 0.71],
            ["lat", "best-layer", "truthfulqa", 0.9, 0.9, 0.9],
            ["lat", "validation-layer", "arc", 0.1, 0.1, 0.1],
        ]
        if with_text:
            out += [
                ["text_word", None, None, 0.676, 0.603, 0.742],
                ["text_char", None, None, 0.671, 0.597, 0.736],
            ]
        return out

    write_run(
        tmp_path, "r6", "gpt2", "a", {"table.json": table(cols, rows(False))}
    )
    write_run(
        tmp_path, "r6", "gpt2-xl", "b", {"table.json": table(cols, rows(True))}
    )
    finding = findings.truthfulqa(store_module.Store(tmp_path))
    assert finding is not None
    (panel,) = finding.figures[0].panels
    for row in panel.rows:
        assert row.band == (0.597, 0.742)
        values = {m.series: m.value for m in row.marks}
        assert values == {"reader": 0.7, "standard": 0.3}
    assert any("0.676" in line for line in finding.numbers)


def test_overview_counts_experiments_and_flags_pending_ones(
    tmp_path: Path,
) -> None:
    reader_runs(tmp_path)
    info = findings.overview(store_module.Store(tmp_path))
    by_id = {e.key: e for e in info.experiments}
    assert by_id["r1"].models == 3
    assert by_id["r1"].status == "partial"
    assert by_id["p10c"].models == 0
    assert by_id["p10c"].status == "pending"
    assert [m.key for m in info.models][:1] == ["gpt2"]
    assert info.models[0].blocks == 3 or info.models[0].blocks is None


def test_signed_numbers_never_print_negative_zero() -> None:
    assert findings.signed(-0.0000001) == "0.000"
    assert findings.signed(0.0114) == "+0.011"
    assert findings.signed(-0.0114) == "-0.011"


def test_finder_comparison_counts_wins_per_family(tmp_path: Path) -> None:
    def rows(pca: float) -> list[dict[str, Any]]:
        return [
            {
                "task": "sentiment",
                "method": method,
                "pairing": "label_contrastive",
                "preprocess": "center",
                "report": value,
                "lo": value,
                "hi": value,
            }
            for method, value in [("random", 0.5), ("pca_diff", pca)]
        ]

    write_run(tmp_path, "p4", "gpt2", "a", {"table.json": rows(0.45)})
    write_run(tmp_path, "p4", "qwen3-1.7b-base", "b", {"table.json": rows(0.9)})
    finding = findings.finder_comparison(store_module.Store(tmp_path))
    assert finding is not None
    assert "0 of 1 GPT-2" in finding.numbers[0]
    assert "1 of 1 Qwen3" in finding.numbers[0]


def test_links_to_knowledge_use_the_layer_p10b_selected(tmp_path: Path) -> None:
    def run(model: str, cosine: float, auroc: float) -> None:
        summary = {
            "layer": 1,
            "cosine": {
                "obscure_real": {
                    "contrast": "knowledge",
                    "cosine": cosine,
                    "band_lo": -0.05,
                    "band_hi": 0.06,
                }
            },
        }
        rows = [
            {
                "p9_type": "obscure_real",
                "contrast": "knowledge",
                "layer": layer,
                "items": "all",
                "test_family": "pooled",
                "n_range": 360,
                "n_exact": 360,
                "auroc": auroc if layer == 1 else 0.99,
                "lo": auroc - 0.03,
                "hi": auroc + 0.03,
                "rand_lo": 0.1,
                "rand_hi": 0.9,
            }
            for layer in (1, 2)
        ]
        write_run(
            tmp_path,
            "p10c",
            model,
            "a",
            {"summary.json": summary, "knowledge.csv": rows},
        )

    run("gpt2", 0.1, 0.45)
    run("qwen3-4b", 0.01, 0.95)
    finding = findings.approximate_answers(store_module.Store(tmp_path))
    assert finding is not None
    assert len(finding.figures) == 1
    cosine, auroc = finding.figures[0].panels
    assert [r.label for r in cosine.rows] == ["gpt2", "qwen3-4b"]
    assert cosine.rows[0].marks[0].value == 0.1
    assert cosine.rows[0].band == (-0.05, 0.06)
    assert auroc.rows[0].marks[0].value == 0.45
    assert auroc.rows[0].band == (0.1, 0.9)
    assert any("1 of 2" in line for line in finding.numbers)
    assert not any("yet" in c for c in finding.caveats)
