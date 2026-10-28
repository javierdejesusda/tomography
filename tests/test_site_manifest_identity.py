from pathlib import Path
import re

from tomography.site import manifest as manifest_module

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FINDINGS = ROOT / "docs" / "findings.md"
HASH = re.compile(r"^[0-9a-f]{12}$")
SKIPPED = ("pca_diff", "sentiment", "concept", "shuffled-only")


def _identity_section() -> list[str]:
    lines = FINDINGS.read_text(encoding="utf-8").splitlines()
    start = lines.index("## Run identity")
    end = next(
        i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")
    )
    return lines[start:end]


def reported() -> dict[tuple[str, str], set[str]]:
    """Reads the hashes the Run identity tables name per experiment and model.

    Returns:
        Map from (experiment id, model key) to the reported hashes. Rows of
        the P1 pca_diff and sentiment-only runs, the P9d concept template and
        the M2 shuffled-only runs are not the default run and are left out.
    """
    found: dict[tuple[str, str], set[str]] = {}
    models: list[str] = []
    for line in _identity_section():
        if not line.startswith("|"):
            models = []
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "Experiment":
            models = cells[1:]
            continue
        if not models or set(cells[0]) <= {"-", " "}:
            continue
        label = cells[0].lower()
        exp = label.split()[0]
        if any(word in label for word in SKIPPED):
            continue
        for model, cell in zip(models, cells[1:], strict=False):
            if HASH.match(cell):
                found.setdefault((exp, model), set()).add(cell)
    return found


def _p1_all_task_cells(
    table: dict[tuple[str, str], set[str]],
) -> dict[tuple[str, str], set[str]]:
    """Keeps the all-task logistic cells of P1 where a follow-up names them."""
    follow_up: dict[tuple[str, str], set[str]] = {}
    models: list[str] = []
    for line in _identity_section():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "Experiment":
            models = cells[1:]
        elif (
            cells[0].startswith("P1 (all tasks") and "pca_diff" not in cells[0]
        ):
            for model, cell in zip(models, cells[1:], strict=False):
                if HASH.match(cell):
                    follow_up[("p1", model)] = {cell}
    return {**table, **follow_up}


def test_the_reported_hashes_are_parsed_from_the_run_identity_tables() -> None:
    table = reported()
    assert table[("r1", "gpt2")] == {"bd4438c46cef"}
    assert table[("c5", "gpt2-xl")] == {"0ad95833261d"}


def test_the_m2_shuffled_only_runs_are_not_reported_runs() -> None:
    table = reported()
    assert table[("m2", "qwen3-0.6b-base")] == {"dfd0cd00a2b7"}
    assert table[("m2", "qwen3-1.7b")] == {"f39cce3f3a97"}


def test_the_selected_run_is_the_one_run_identity_names() -> None:
    table = _p1_all_task_cells(reported())
    found = manifest_module.Manifest(RESULTS)
    problems = []
    for exp in found.experiments():
        for model in found.models(exp):
            hashes = [p.name for p in (RESULTS / exp / model).iterdir()]
            if len(hashes) < 2:
                continue
            chosen = found.run(exp, model)
            named = table.get((exp, model))
            if named is None:
                problems.append(f"{exp}/{model}: no Run identity entry")
            elif chosen is None or chosen.name not in named:
                got = chosen.name if chosen else None
                problems.append(f"{exp}/{model}: selected {got}, named {named}")
    assert problems == []
