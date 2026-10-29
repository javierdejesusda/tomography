from pathlib import Path
import re

from tomography.site import manifest as manifest_module

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FINDINGS = ROOT / "docs" / "findings.md"
HASH = re.compile(r"^[0-9a-f]{12}$")
# The sentiment-only P1 runs (superseded by the all-task runs) and the M2
# shuffled-only runs (which only supplement a full run) stay committed, but no
# selection of `Manifest.run` picks them.
UNSELECTED = frozenset({"P1", "P1 (sentiment)", "M2 shuffled-only run"})
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
}


def _identity_section() -> list[str]:
    lines = FINDINGS.read_text(encoding="utf-8").splitlines()
    start = lines.index("## Run identity")
    end = next(
        i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")
    )
    return lines[start:end]


def reported() -> list[tuple[str, str, str]]:
    """Reads the runs the Run identity tables name, one per table cell.

    Returns:
        (row label, model key, hash) for every hash cell, except the cells of
        the rows in `UNSELECTED`.
    """
    found: list[tuple[str, str, str]] = []
    models: list[str] = []
    for line in _identity_section():
        if not line.startswith("|"):
            models = []
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "Experiment":
            models = cells[1:]
            continue
        label = cells[0]
        if not models or set(label) <= {"-", " "} or label in UNSELECTED:
            continue
        for model, cell in zip(models, cells[1:], strict=False):
            if HASH.match(cell):
                found.append((label, model, cell))
    return found


def test_the_reported_hashes_are_parsed_from_the_run_identity_tables() -> None:
    table = reported()
    assert ("R1", "gpt2", "bd4438c46cef") in table
    assert ("C5", "gpt2-xl", "0ad95833261d") in table


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
        select = SELECTIONS.get(label, {})
        if not select:
            named_default.add((exp, model))
        chosen = found.run(exp, model, **select)
        if chosen is None or chosen.name != named:
            got = chosen.name if chosen else None
            problems.append(f"{label}/{model}: selected {got}, named {named}")
    for exp in found.experiments():
        for model in found.models(exp):
            several = len(found.runs(exp, model)) > 1
            if several and (exp, model) not in named_default:
                problems.append(f"{exp}/{model}: no Run identity entry")
    assert problems == []
