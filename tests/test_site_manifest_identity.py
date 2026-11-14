from pathlib import Path
import re

from tomography.site import manifest as manifest_module

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FINDINGS = ROOT / "docs" / "findings.md"
HASH = re.compile(r"[0-9a-f]{12}\b")
EXPERIMENT_ID = re.compile(r"[A-Z]+[0-9]+[a-z]?")
RUN_READ = re.compile(EXPERIMENT_ID.pattern + " run read")
# The sentiment-only P1 runs (superseded by the all-task runs), the M2
# shuffled-only runs (which only supplement a full run) and the earlier runs of
# the experiments that take their latest run stay committed, but no selection
# of `Manifest.run` picks them.
UNSELECTED = frozenset(
    {
        "P1",
        "P1 (sentiment)",
        "M2 shuffled-only run",
        *(f"{exp} earlier run" for exp in ("C2", "C4", "C5", "M1", "P7")),
    }
)
# M3 stage 2 reads these cell sets; the 1.7B pair ran it one task per run.
M3_STAGE_2 = "stimulus,final,upper"
# A row label left out of this map names the default run of its experiment, so
# its check calls `Manifest.run` with no selection and pins that default.
SELECTIONS: dict[str, dict[str, str]] = {
    "P1 (all tasks, pca_diff)": {"method": "pca_diff"},
    "P9d concept": {"template": "concept"},
    "R8 CommonsenseQA": {"task": "csqa"},
    "R8 ARC-Easy": {"task": "arc_easy"},
    "R8 ARC-Challenge": {"task": "arc_challenge"},
    "R8 RACE": {"task": "race"},
    "R8 TruthfulQA MC1": {"task": "tqa"},
    "M3 stage 2": {"cell_sets": M3_STAGE_2},
    **{
        f"M3 stage 2 {task}": {"cell_sets": M3_STAGE_2, "task": task}
        for task in ("sentiment", "truth", "truth_polarity", "utility")
    },
    "M3 28 layers": {"task": "truth"},
    "M3 28 layers, wide C grid": {
        "task": "truth",
        "logistic_c": "0.001,0.01,0.1,1,10,100,1000",
    },
    "M3 stage 1, wide C grid": {"logistic_c": "0.001,0.01,0.1,1,10,100,1000"},
    "M3 flat, wide C grid": {
        "logistic_c": "0.001,0.01,0.1,1,10,100,1000",
        "poolings": "flat",
    },
    "M3 MLP head, wide C grid": {
        "logistic_c": "0.001,0.01,0.1,1,10,100,1000",
        "cell_head": "mlp",
    },
}


def _identity_section() -> list[str]:
    lines = FINDINGS.read_text(encoding="utf-8").splitlines()
    start = lines.index("## Run identity")
    end = next(
        i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")
    )
    return lines[start:end]


def named_runs() -> list[tuple[str, str, str]]:
    """Reads every run the Run identity tables name, one per table cell.

    A table headed "Experiment" has a row per experiment and a column per
    checkpoint. A table headed "Checkpoint" is transposed: a row per
    checkpoint, and the columns headed by an experiment id, or by "<id> run
    read" (a run of that experiment which the run of the row read), name
    runs; its other columns (batch size, commit, times) are not read.

    Returns:
        (label, model key, hash) for every cell that starts with a hash,
        such as "94faa23c130d (batch size 4)". The label is the row label, or
        the column header in a transposed table.

    Raises:
        ValueError: If a table has another first header, so that its runs
            would go unread.
    """
    found: list[tuple[str, str, str]] = []
    header: list[str] = []
    for line in _identity_section():
        if not line.startswith("|"):
            header = []
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not header:
            header = cells
            if header[0] not in ("Experiment", "Checkpoint"):
                raise ValueError(f"unread Run identity table: {line}")
            continue
        if set(cells[0]) <= {"-", " "}:
            continue
        for column, cell in zip(header[1:], cells[1:], strict=True):
            if header[0] == "Experiment":
                label, model = cells[0], column
            elif EXPERIMENT_ID.fullmatch(column) or RUN_READ.fullmatch(column):
                label, model = column, cells[0]
            else:
                continue
            if run := HASH.match(cell):
                found.append((label, model, run.group()))
    return found


def runs_read() -> list[tuple[str, str, str]]:
    """The cells of the "<id> run read" columns, as `named_runs` gives them."""
    return [entry for entry in named_runs() if RUN_READ.fullmatch(entry[0])]


def unselected() -> list[tuple[str, str, str]]:
    """The cells of the rows in `UNSELECTED`, as `named_runs` gives them."""
    return [entry for entry in named_runs() if entry[0] in UNSELECTED]


def reported() -> list[tuple[str, str, str]]:
    """The cells that name the run a selection of `Manifest.run` picks.

    Returns:
        The cells of `named_runs`, except those of the rows in `UNSELECTED`
        and of the run read columns.
    """
    return [
        (label, model, run)
        for label, model, run in named_runs()
        if label not in UNSELECTED and not RUN_READ.fullmatch(label)
    ]


def test_the_reported_hashes_are_parsed_from_the_run_identity_tables() -> None:
    table = reported()
    assert ("R1", "gpt2", "bd4438c46cef") in table
    assert ("C5", "gpt2-xl", "0ad95833261d") in table


def test_a_cell_is_read_by_its_leading_hash_past_a_note() -> None:
    assert ("P10a", "qwen3-4b-base", "94faa23c130d") in reported()


def test_a_checkpoint_table_is_read_by_its_experiment_columns() -> None:
    table = reported()
    assert ("P10a", "gpt2", "5af932e9547d") in table
    assert ("V1", "qwen3-1.7b", "419275ac7e0d") in table
    assert ("B2", "gpt2-large", "43f761636126") in table
    labels = {label.split()[0] for label, _, _ in table}
    assert all(EXPERIMENT_ID.fullmatch(label) for label in labels)


def test_the_m2_shuffled_only_runs_are_not_reported_runs() -> None:
    m2 = [(m, run) for label, m, run in reported() if label.startswith("M2")]
    assert m2 == [
        ("qwen3-0.6b-base", "dfd0cd00a2b7"),
        ("qwen3-0.6b", "4f11f2a29935"),
        ("qwen3-1.7b-base", "84a62d2075b6"),
        ("qwen3-1.7b", "f39cce3f3a97"),
    ]


def test_each_row_names_the_run_its_selection_picks() -> None:
    found = manifest_module.Manifest(RESULTS)
    problems = []
    named_default = set()
    for label, model, named in reported():
        exp = label.split()[0].lower()
        if label not in SELECTIONS:
            named_default.add((exp, model))
        chosen = found.run(exp, model, **SELECTIONS.get(label, {}))
        got = chosen.name if chosen else None
        if got != named:
            problems.append(f"{label}/{model}: selected {got}, named {named}")
    for exp in found.experiments():
        for model in found.models(exp):
            unnamed = (exp, model) not in named_default
            if unnamed and len(found.runs(exp, model)) > 1:
                problems.append(f"{exp}/{model}: no Run identity entry")
    assert problems == []


def test_every_selection_names_a_row() -> None:
    assert set(SELECTIONS) <= {label for label, _, _ in reported()}


def test_an_unselected_row_names_a_committed_run_no_selection_picks() -> None:
    found = manifest_module.Manifest(RESULTS)
    problems = []
    for label, model, named in unselected():
        exp = label.split()[0].lower()
        committed = [run.path.name for run in found.runs(exp, model)]
        chosen = found.run(exp, model)
        if named not in committed or (chosen and chosen.name == named):
            problems.append(f"{label}/{model}: {named}")
    assert unselected()
    assert problems == []


def test_a_run_read_column_names_a_committed_run() -> None:
    read = runs_read()
    assert ("P9a run read", "gpt2", "5461200f995c") in read
    assert ("P9c run read", "qwen3-1.7b", "7c142334e5fe") in read
    missing = [
        f"{label}/{model}: {run}"
        for label, model, run in read
        if not (RESULTS / label.split()[0].lower() / model / run).is_dir()
    ]
    assert missing == []


def test_every_committed_run_is_named_in_a_row() -> None:
    found = manifest_module.Manifest(RESULTS)
    named = {
        (label.split()[0].lower(), model, run)
        for label, model, run in named_runs()
    }
    unnamed = [
        f"{exp}/{model}/{run.path.name}"
        for exp in found.experiments()
        for model in found.models(exp)
        for run in found.runs(exp, model)
        if (exp, model, run.path.name) not in named
    ]
    assert unnamed == []
