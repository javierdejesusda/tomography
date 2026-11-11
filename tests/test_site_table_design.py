import importlib
import inspect
import json
from pathlib import Path
import re
import runpy

import pytest

from tomography.site import registry
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.tables import design

_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
ROOT, data, models, run = _R["ROOT"], _R["data"], _R["models"], _R["run"]
SPEC = "design"
DASHES = (chr(0x2014), chr(0x2013))
ASPECTS = [
    "Study",
    "Question",
    "Stimuli",
    "Labels",
    "Splits",
    "Controls",
    "Published baselines",
    "Metric",
    "Checkpoints",
]
STUDIES = [
    "Reading concepts",
    "Finders",
    "Known and unknown questions",
    "Approximate answers",
    "Truth",
    "Steering",
    "Post-training",
]
CODE = re.compile(r"\"([a-z]+[0-9]+[a-d]?)\"")
RUN_CODE = re.compile(r"\b[RPMCA][0-9]+[a-d]?\b", re.IGNORECASE)
NAMES = ("GPT-2", "SST-2", "Qwen3", "MC1", "WikiText-103", "distinct-1")
PLACEHOLDER = re.compile(r"\{[a-z_.]+\}")


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table(SPEC, Manifest(ROOT))


def _shown() -> set[str]:
    """Collects the experiments that the other figures and tables read."""
    stored = {path.name for path in ROOT.iterdir() if path.is_dir()}
    found: set[str] = set()
    for package, keys in (
        ("figs", (*registry.MAIN_FIGURES, *registry.APPENDIX_FIGURES)),
        ("tables", (*registry.MAIN_TABLES, *registry.APPENDIX_TABLES)),
    ):
        for key in keys:
            if (package, key) == ("tables", SPEC):
                continue
            module = importlib.import_module(f"tomography.site.{package}.{key}")
            found |= set(CODE.findall(inspect.getsource(module))) & stored
    return found


def _blocks(output: registry.TableOutput) -> list[dict[str, str]]:
    """Reads the rows back as one mapping of aspect to text per experiment."""
    size = len(ASPECTS)
    blocks = []
    for start in range(0, len(output.rows), size):
        rows = output.rows[start : start + size]
        assert [row[1] for row in rows] == ASPECTS
        assert all(row[0] == "" for row in rows[1:])
        blocks.append({"Experiment": rows[0][0], **{r[1]: r[2] for r in rows}})
    return blocks


def test_every_experiment_is_a_block_of_the_same_aspects(
    output: registry.TableOutput,
) -> None:
    assert output.columns == ["Experiment", "Aspect", "Design"]
    assert output.align == "lll"
    blocks = _blocks(output)
    assert len(blocks) == len(design.CARDS) == 25
    assert [b["Experiment"] for b in blocks] == [c.name for c in design.CARDS]
    assert len({b["Experiment"] for b in blocks}) == len(blocks)
    assert output.row_groups == tuple(range(0, len(output.rows), len(ASPECTS)))
    assert output.numbers["experiments"] == len(blocks)
    assert output.numbers["studies"] == len(STUDIES)


def test_every_shown_experiment_has_a_block() -> None:
    shown = _shown()
    assert len(shown) >= 25
    covered = [code for card in design.CARDS for code in card.codes]
    assert len(covered) == len(set(covered))
    assert set(covered) == shown


def test_the_blocks_follow_the_studies_in_order(
    output: registry.TableOutput,
) -> None:
    order = [block["Study"].rstrip(".") for block in _blocks(output)]
    assert list(dict.fromkeys(order)) == STUDIES
    for study in STUDIES:
        places = [i for i, name in enumerate(order) if name == study]
        assert places == list(range(places[0], places[-1] + 1)), study


def test_counts_in_the_cells_are_read_from_the_runs(
    output: registry.TableOutput,
) -> None:
    for card in design.CARDS:
        for text in (
            card.question,
            card.stimuli,
            card.labels,
            card.splits,
            card.controls,
            card.baselines,
            card.metric,
        ):
            bare = PLACEHOLDER.sub(" ", text)
            for name in (*NAMES, "95 percent"):
                bare = bare.replace(name, " ")
            assert re.findall(r"\d+", bare) == [], text
    blocks = {b["Experiment"]: b for b in _blocks(output)}
    for model in models("r2"):
        sizes = data(run("r2", model), "summary.json")["n_eval_pairs"]
        assert (sizes["val"], sizes["test"]) == (3473, 4808)
    assert "3,473 validation pairs" in blocks["Utility"]["Splits"]
    assert "4,808 test pairs" in blocks["Utility"]["Splits"]
    assert output.numbers["utility_n_eval_pairs_val"] == 3473
    stored = json.loads(
        (run("r6", "gpt2") / "config.json").read_text(encoding="utf-8")
    )
    assert stored["n_seeds"] == output.numbers["truthfulqa_n_seeds"] == 15
    assert "15 readers" in blocks["TruthfulQA"]["Splits"]
    recipe = data(run("p9a", "qwen3-1.7b"), "summary.json")["recipe"]
    assert recipe["known_min_correct"] == 3
    assert output.numbers["labels_recipe_known_min_correct"] == 3
    assert not any(
        PLACEHOLDER.search(cell) for row in output.rows for cell in row
    )


def test_a_count_that_differs_between_checkpoints_is_refused() -> None:
    assert design.agreed("n_pairs", {"gpt2": 64, "gpt2-xl": 64}) == 64
    with pytest.raises(ValueError, match="n_pairs"):
        design.agreed("n_pairs", {"gpt2": 64, "gpt2-xl": 32})
    with pytest.raises(ValueError, match="n_pairs"):
        design.agreed("n_pairs", {"gpt2": None, "gpt2-xl": 32})


def test_checkpoints_are_counted_per_role(
    output: registry.TableOutput,
) -> None:
    blocks = {b["Experiment"]: b for b in _blocks(output)}
    every = _R["MANIFEST"].checkpoints()
    for card in design.CARDS:
        cell = blocks[card.name]["Checkpoints"]
        if card.key in ("text_only", "post_training"):
            continue
        (code,) = card.codes
        want = stats.per_role({m: m in models(code) for m in every})
        assert output.counts[card.key] == want, card.name
        assert cell.startswith(stats.per_role_text(want)), card.name
    assert blocks["Text-only classifiers"]["Checkpoints"].startswith("None")
    pairs = data(run("p8", "qwen3-1.7b"), "summary.json")["pairs"]
    paired = {model for pair in pairs for model in pair}
    want = stats.per_role({m: m in paired for m in every})
    assert output.counts["post_training"] == want
    assert blocks["Base against post-trained"]["Checkpoints"].startswith(
        stats.per_role_text(want)
    )
    steering = output.counts["knowledge_steering"]
    assert (steering["gpt2"].hits, steering["gpt2"].total) == (3, 4)


def test_controls_and_published_baselines_are_kept_apart(
    output: registry.TableOutput,
) -> None:
    for block in _blocks(output):
        assert "baseline" not in block["Controls"].lower(), block["Experiment"]
        for kind in ("random", "shuffled", "popularity"):
            assert kind not in block["Published baselines"].lower(), block
    blocks = {b["Experiment"]: b for b in _blocks(output)}
    assert "random direction" in blocks["Utility"]["Controls"]
    assert blocks["Utility"]["Published baselines"].startswith("None")
    assert "CCS" in blocks["Published finders"]["Published baselines"]
    assert (
        "P(True)" in blocks["Predicting wrong answers"]["Published baselines"]
    )
    assert (
        "semantic entropy"
        in blocks["Published error detectors"]["Published baselines"]
    )


def test_the_text_keeps_the_rules_of_the_page(
    output: registry.TableOutput,
) -> None:
    texts = [output.claim, output.detail, *(c for r in output.rows for c in r)]
    assert not any("_" in t for t in texts)
    assert not any(ch in t for t in texts for ch in DASHES)
    for text in texts:
        assert RUN_CODE.search(text.replace("MC1", "")) is None, text
        bare = text.lower().replace("first principal component", "")
        assert "first" not in bare, text
    for row in output.rows:
        assert row[2].endswith((".", "?")), row
    assert "per role" in output.detail
    assert "same on every checkpoint" in output.detail
