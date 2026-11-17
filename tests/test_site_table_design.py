import importlib
import inspect
import json
from pathlib import Path
import re
import runpy

import pytest

from tomography.site import params
from tomography.site import registry
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.site.tables import design

_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
ROOT, data, models, run = _R["ROOT"], _R["data"], _R["models"], _R["run"]
SPEC = "design"
DASHES = (chr(0x2014), chr(0x2013))
COLUMNS = [
    "Experiment",
    "Stimuli and labels",
    "Splits",
    "Controls",
    "Published baselines",
    "Checkpoints",
]
STUDIES = [
    "Reading",
    "Finders",
    "Truth",
    "Knowledge",
    "Steering",
    "Post-training",
]
EXPERIMENTS = [
    "Sentiment",
    "Utility",
    "Morality",
    "Truth",
    "Honesty transfer",
    "Prompt wording",
    "Finder comparison",
    "Published finders",
    "TruthfulQA",
    "Labels",
    "Knowledge reading",
    "Error prediction",
    "Published detectors",
    "Knowledge steering",
    "Activation views",
    "Approximate answers",
    "Range reader",
    "Sentiment steering",
    "Utility test",
    "Published steering methods",
    "Base against post-trained",
]
CODE = re.compile(r"\"([a-z]+[0-9]+[a-d]?)\"")
RUN_CODE = re.compile(r"\b[RPMCA][0-9]+[a-d]?\b", re.IGNORECASE)
TOKEN = re.compile(r"\[\[(.+?)\]\]")
NAMES = (
    "GPT-2",
    "SST-2",
    "Qwen3",
    "WikiText-103",
    "NQ-open",
)
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


def _cells(output: registry.TableOutput) -> dict[str, dict[str, str]]:
    """Reads the rows back as experiment to column to cell."""
    return {
        row[0]: dict(zip(COLUMNS[1:], row[1:], strict=True))
        for row in output.rows
    }


def _config(code: str, model: str) -> dict:
    return json.loads(
        (run(code, model) / "config.json").read_text(encoding="utf-8")
    )


def test_every_experiment_is_one_row_under_its_study(
    output: registry.TableOutput,
) -> None:
    assert output.columns == COLUMNS
    assert output.align == "llllll"
    assert [row[0] for row in output.rows] == EXPERIMENTS
    assert [row.name for row in design.ROWS] == EXPERIMENTS
    assert all(len(row) == len(COLUMNS) for row in output.rows)
    assert [text for _, text in output.headings] == STUDIES
    order = [row.study for row in design.ROWS]
    assert list(dict.fromkeys(order)) == STUDIES
    assert [place for place, _ in output.headings] == [
        order.index(study) for study in STUDIES
    ]
    for study in STUDIES:
        places = [i for i, name in enumerate(order) if name == study]
        assert places == list(range(places[0], places[-1] + 1)), study
    assert output.row_groups == ()
    assert not output.span_labels
    assert output.compact_print
    assert output.data_columns == ("Study",)
    assert output.data_cells == tuple((study,) for study in order)
    assert output.numbers["experiments"] == len(EXPERIMENTS)
    assert output.numbers["studies"] == len(STUDIES)
    assert output.numbers["rows"] == len(output.rows)


def test_the_cells_stay_within_the_word_budget(
    output: registry.TableOutput,
) -> None:
    words = sum(len(cell.split()) for row in output.rows for cell in row)
    assert words <= 760, words
    caption = f"{output.claim} {output.detail}"
    assert len(caption.split()) <= 170, len(caption.split())


def test_every_shown_experiment_has_a_row() -> None:
    shown = _shown()
    covered = list(design.RUNS.values())
    assert len(covered) == len(set(covered))
    assert {row.key for row in design.ROWS} | {"training_size"} == set(
        design.RUNS
    )
    assert shown - set(covered) == {"tb1"}
    assert set(covered) - shown <= {"r1", "p9a"}
    controls = " ".join(row.controls for row in design.ROWS)
    assert "the words alone" in controls


def test_counts_in_the_cells_are_read_from_the_runs(
    output: registry.TableOutput,
) -> None:
    reported = params.REPORT_PAIRS["sentiment"]
    for text in (
        design.CLAIM,
        design.DETAIL,
        *(t for row in design.ROWS for t in row.texts()),
    ):
        bare = TOKEN.sub(" ", PLACEHOLDER.sub(" ", text))
        for name in (*NAMES, f"({reported} pairs)"):
            bare = bare.replace(name, " ")
        assert re.findall(r"\d+", bare) == [], text
    cells = _cells(output)
    for model in models("r2"):
        sizes = data(run("r2", model), "summary.json")["n_eval_pairs"]
        assert (sizes["val"], sizes["test"]) == (3473, 4808)
    assert cells["Utility"]["Splits"] == (
        "3,473 validation and 4,808 test pairs."
    )
    assert output.numbers["utility_n_eval_pairs_val"] == 3473
    sizes = data(run("r3", "gpt2"), "summary.json")["n_eval_pairs"]
    assert cells["Morality"]["Splits"] == (
        f"{sizes['val']} validation and {sizes['test']} test pairs."
    )
    truth = data(run("r7", "gpt2"), "summary.json")["sets"]["test"]
    assert truth["n_eval_pairs"] == 59
    assert output.numbers["truth_sets_test_n_eval_pairs"] == 59
    assert cells["Truth"]["Splits"] == "59 test pairs."
    honesty = data(run("r5", "gpt2"), "summary.json")["n_eval_pairs"]
    assert cells["Honesty transfer"]["Splits"] == (
        f"{honesty['transfer']['val']} validation and "
        f"{honesty['transfer']['test']} test pairs."
    )
    main = _config("r1", "gpt2")
    sizes_run = _config("p3", "gpt2")
    assert (main["n_eval_pairs"], sizes_run["n_seeds"]) == (256, 20)
    assert cells["Sentiment"]["Splits"] == (
        "Main run: the 256 validation pairs choose the layer and give the "
        "score (optimistic). Reported run: 20 draws; half of the validation "
        f"sentences select, the other half ({reported} pairs) reports."
    )
    assert output.numbers["training_size_n_seeds"] == 20
    stored = _config("r6", "gpt2")
    assert stored["n_seeds"] == output.numbers["truthfulqa_n_seeds"] == 15
    summary = data(run("r6", "gpt2"), "summary.json")
    assert cells["TruthfulQA"]["Splits"] == (
        f"15 readers; {summary['n_train']} train, {summary['n_val']} "
        f"validation, {summary['n_test']} test questions."
    )
    assert cells["TruthfulQA"]["Stimuli and labels"].startswith(
        f"{int(output.numbers['truthfulqa_n_questions'])} multiple-choice"
    )
    recipe = data(run("p9a", "qwen3-1.7b"), "summary.json")["recipe"]
    assert recipe["known_min_correct"] == 3
    assert output.numbers["labels_recipe_known_min_correct"] == 3
    samples = int(output.numbers["labels_n_samples"])
    assert (
        f"at least 3 of {samples} sampled answers are correct"
        in (cells["Labels"]["Stimuli and labels"])
    )
    reading = _config("p9b", "qwen3-1.7b")
    assert cells["Knowledge reading"]["Splits"] == (
        f"{reading['n_seeds']} draws of {reading['n_pairs']} training pairs."
    )
    assert output.numbers["knowledge_reading_n_seeds"] == reading["n_seeds"]
    random = _config("p9d", "qwen3-1.7b-base")["n_random"]
    assert cells["Knowledge steering"]["Controls"] == (
        f"{random} random directions of the same norm."
    )
    boot = int(output.numbers["post_training_n_boot"])
    assert (
        f"bootstrap of {boot:,} resamples"
        in (cells["Base against post-trained"]["Splits"])
    )
    assert not any(
        PLACEHOLDER.search(cell) for row in output.rows for cell in row
    )
    assert not PLACEHOLDER.search(output.detail)


def test_a_count_that_differs_between_checkpoints_is_refused() -> None:
    assert design.agreed("n_pairs", {"gpt2": 64, "gpt2-xl": 64}) == 64
    with pytest.raises(ValueError, match="n_pairs"):
        design.agreed("n_pairs", {"gpt2": 64, "gpt2-xl": 32})
    with pytest.raises(ValueError, match="n_pairs"):
        design.agreed("n_pairs", {"gpt2": None, "gpt2-xl": 32})


def test_a_row_with_standard_splits_has_the_draws_the_caption_states(
    output: registry.TableOutput,
) -> None:
    readers = int(output.numbers["utility_n_seeds"])
    pairs = int(output.numbers["utility_n_pairs"])
    assert (
        f"Standard splits are {readers} readers, each on its own draw of "
        f"{pairs} training pairs" in output.detail
    )
    config = _config("r2", "gpt2")
    assert (config["n_seeds"], config["n_pairs"]) == (readers, pairs)
    standard = [row for row in design.ROWS if row.draws == "utility"]
    assert {row.name for row in standard} >= {
        "Prompt wording",
        "Published finders",
        "Truth",
    }
    for row in design.ROWS:
        if row.splits == design.STANDARD:
            assert row.draws == "utility", row.name
    for row in standard:
        assert output.numbers[f"{row.key}_n_seeds"] == readers, row.name
        assert output.numbers[f"{row.key}_n_pairs"] == pairs, row.name
    (errors,) = [r for r in design.ROWS if r.key == "error_prediction"]
    assert errors.draws == "knowledge_reading"
    assert errors.splits == design.ABOVE
    for name in ("n_seeds", "n_pairs"):
        assert (
            output.numbers[f"error_prediction_{name}"]
            == output.numbers[f"knowledge_reading_{name}"]
        )


def test_checkpoints_are_counted_per_role(
    output: registry.TableOutput,
) -> None:
    cells = _cells(output)
    every = _R["MANIFEST"].checkpoints()
    for row in design.ROWS:
        cell = cells[row.name]["Checkpoints"]
        if row.key == "post_training":
            continue
        want = stats.per_role(
            {m: m in models(design.RUNS[row.key]) for m in every}
        )
        assert output.counts[row.key] == want, row.name
        assert cell.startswith(stats.per_role_compact(want)), row.name
        assert cell.endswith("."), row.name
    assert set(output.counts) == set(design.RUNS)
    pairs = data(run("p8", "qwen3-1.7b"), "summary.json")["pairs"]
    paired = {model for pair in pairs for model in pair}
    want = stats.per_role({m: m in paired for m in every})
    assert output.counts["post_training"] == want
    assert cells["Base against post-trained"]["Checkpoints"] == (
        f"{stats.per_role_compact(want)}."
    )
    steering = output.counts["knowledge_steering"]
    assert (steering["gpt2"].hits, steering["gpt2"].total) == (3, 4)
    assert cells["Knowledge steering"]["Checkpoints"] == "3/4, 3/3, 3/3."
    assert cells["Utility"]["Checkpoints"] == "All."
    full = ", ".join(
        f"{c.total}/{c.total}"
        for c in (
            stats.per_role(dict.fromkeys(every, True))[r] for r in stats.ROLES
        )
    )
    assert output.detail.endswith(
        "Checkpoints are counted per role in the order GPT-2, Qwen3 base, "
        f"Qwen3 post-trained; All stands for {full}."
    )


def test_controls_and_published_baselines_are_kept_apart(
    output: registry.TableOutput,
) -> None:
    cells = _cells(output)
    for name, row in cells.items():
        assert "baseline" not in row["Controls"].lower(), name
        for kind in ("random", "shuffled", "popularity"):
            assert kind not in row["Published baselines"].lower(), name
    assert cells["Utility"]["Controls"] == "Standard."
    assert cells["Prompt wording"]["Controls"] == (
        "Random direction, shuffled labels."
    )
    assert (
        "contrast-consistent search"
        in (cells["Published finders"]["Published baselines"])
    )
    assert cells["Error prediction"]["Published baselines"] == (
        "The model's confidence."
    )
    assert (
        "semantic entropy"
        in (cells["Published detectors"]["Published baselines"])
    )
    assert "P(True)" in cells["Published detectors"]["Published baselines"]
    steering = cells["Published steering methods"]
    assert steering["Published baselines"] == (
        "The five methods of [[table:steering_baselines]]."
    )
    assert steering["Controls"] == "Unsteered model, random unit direction."
    assert "20 tuning prompts" in steering["Splits"]
    assert "40 held-out prompts" in steering["Splits"]
    assert (
        "on post-trained checkpoints, how closely the topic relates to the "
        "instruction." in steering["Stimuli and labels"]
    )
    assert "follows the instruction" not in steering["Stimuli and labels"]
    assert cells["Sentiment steering"]["Controls"].startswith(
        "No random direction:"
    )


def test_the_text_keeps_the_rules_of_the_page(
    output: registry.TableOutput,
) -> None:
    texts = [output.claim, output.detail, *(c for r in output.rows for c in r)]
    for text in texts:
        bare = TOKEN.sub("", text)
        assert "_" not in bare, text
        assert not any(ch in bare for ch in DASHES), text
        assert "`" not in bare, text
        assert RUN_CODE.search(bare) is None, text
        assert "first" not in bare.lower(), text
    for row in output.rows:
        for cell in row[1:]:
            assert cell.endswith("."), row
    assert "per role" in output.detail
    assert "same on every checkpoint" in output.detail


def test_the_caption_defines_its_terms(output: registry.TableOutput) -> None:
    assert output.claim == "The design of each experiment the main text uses."
    for words in (
        "a Splits cell gives the sizes and what differs.",
        "Standard controls are a random direction oriented on the training "
        "labels, the reader scored against shuffled labels, and the words "
        "alone.",
        "A control uses no direction fitted for the concept; a baseline is a "
        "published method run on the same items.",
        "Select pairs choose the layer, and report pairs, which that choice "
        "does not use, give the score.",
        "Post hoc marks a choice made after results were seen.",
    ):
        assert words in output.detail, words
    assert "Zou" not in output.detail
    assert "Inside a study" not in output.detail


def test_the_post_hoc_marks_stay_in_their_cells(
    output: registry.TableOutput,
) -> None:
    cells = _cells(output)
    assert cells["Sentiment"]["Published baselines"] == (
        "None; the reader is itself a logistic-regression probe, made the "
        "default finder after pilot runs (post hoc)."
    )
    assert (
        "The PopQA part of the pool was widened from four to twelve "
        "relations after the guess filter of earlier labelling runs left "
        "too few known questions (post hoc)."
    ) in cells["Labels"]["Stimuli and labels"]
    marked = [
        name
        for name, row in cells.items()
        if any("(post hoc)" in cell for cell in row.values())
    ]
    assert marked == ["Sentiment", "Labels"]
    for name in ("Utility", "Morality", "Truth", "Honesty transfer"):
        assert cells[name]["Published baselines"] == "As above.", name
    assert "(optimistic)" in cells["Sentiment"]["Splits"]


def test_gpt2_small_is_scoped_where_it_has_no_known_questions(
    output: registry.TableOutput,
) -> None:
    cells = _cells(output)
    assert cells["Knowledge reading"]["Checkpoints"] == (
        "All; too few known questions on GPT-2 small."
    )
    assert cells["Error prediction"]["Checkpoints"] == (
        "All; no direction on GPT-2 small."
    )
    assert cells["Published detectors"]["Checkpoints"].endswith(
        "; no direction on GPT-2 small."
    )
    assert cells["Published steering methods"]["Checkpoints"].endswith(
        "; GPT-2 XL is steered as a base checkpoint."
    )


def test_the_rows_state_what_their_runs_do(
    output: registry.TableOutput,
) -> None:
    cells = _cells(output)
    for name in ("Knowledge reading", "Range reader"):
        baselines = cells[name]["Published baselines"]
        assert baselines in (
            "The finder of the paper, mean difference.",
            "As knowledge reading.",
        ), name
    utility = cells["Utility test"]
    assert "among five finders" in utility["Published baselines"]
    vectors = _config("m1", "gpt2")["n_steer_seeds"]
    assert f"{vectors} steering vectors per finder" in utility["Controls"]
    labels = cells["Labels"]
    assert labels["Controls"] == "None."
    assert labels["Published baselines"] == "None."
    assert "or no gold answer exists" in labels["Stimuli and labels"]
    views = cells["Activation views"]
    assert views["Stimuli and labels"] == (
        "The questions of knowledge reading, at every layer it read."
    )
    assert "labels permuted inside each group" in views["Controls"]
    people = int(output.numbers["approximate_reading_n_people"])
    assert f"{people} people" in cells["Range reader"]["Controls"]
    assert "held-out family" in cells["Range reader"]["Splits"]
    finders = cells["Finder comparison"]
    assert "the random direction included" in finders["Splits"]
    assert "report pairs" in finders["Splits"]


def test_the_numbers_keep_the_names_the_text_of_the_page_reads(
    output: registry.TableOutput,
) -> None:
    for name in (
        "sentiment_min_words",
        "sentiment_n_eval_pairs",
        "sentiment_n_seeds",
        "sentiment_n_pairs",
        "utility_n_seeds",
        "utility_n_pairs",
        "utility_n_eval_pairs_test",
        "morality_n_eval_pairs_test",
        "training_size_n_seeds",
        "wording_n_seeds",
        "pairing_n_pairs",
        "published_finders_ccs_restarts",
        "published_finders_n_seeds",
        "truth_n_seeds",
        "truthfulqa_n_seeds",
        "truthfulqa_n_train",
        "truthfulqa_n_val",
        "truthfulqa_n_test",
        "honesty_n_eval_pairs_transfer_val",
        "honesty_n_eval_pairs_transfer_test",
        "labels_min_known",
        "labels_n_shots",
        "labels_n_samples",
        "labels_recipe_known_min_correct",
        "knowledge_reading_n_seeds",
        "knowledge_reading_n_pairs",
        "error_prediction_n_seeds",
        "knowledge_steering_n_random",
        "approximate_answers_n_per_side",
        "approximate_answers_n_samples",
        "approximate_reading_n_seeds",
        "approximate_reading_n_people",
        "sentiment_steering_n_prompts",
        "sentiment_steering_num_samples",
        "sentiment_steering_n_neutral",
        "sentiment_steering_n_steer_seeds",
        "finders_steering_n_prompts",
        "finders_steering_n_steer_seeds",
        "steering_baselines_max_new_tokens",
        "steering_baselines_n_neutral",
        "steering_baselines_n_pairs",
        "steering_baselines_n_tune",
        "steering_baselines_n_held_out",
        "post_training_n_boot",
    ):
        assert name in output.numbers, name
    for gone in ("token_position", "agreement", "approximate_links"):
        assert not any(name.startswith(gone) for name in output.numbers), gone
        assert gone not in output.counts
    assert "text_only" not in output.counts
    assert (
        output.numbers["sentiment_steering_n_neutral"]
        == (_config("c1", "gpt2")["n_neutral"])
    )
    assert output.numbers["labels_n_shots"] == 15
