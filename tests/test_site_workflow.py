from pathlib import Path
import re

WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pages.yml"
)


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_workflow_deploys_on_main_for_results_and_site_changes() -> None:
    text = _text()
    assert "branches: [main]" in text
    assert "workflow_dispatch:" in text
    for path in (
        "results/**",
        "src/tomography/site/**",
        ".github/workflows/pages.yml",
    ):
        assert f'"{path}"' in text


def test_workflow_uses_minimal_permissions_and_a_concurrency_group() -> None:
    text = _text()
    top = text.split("jobs:", 1)[0]
    assert "contents: read" in top
    deploy = text.split("deploy:", 1)[1]
    assert "pages: write" in deploy
    assert "id-token: write" in deploy
    assert "concurrency:" in text
    assert "cancel-in-progress: false" in text


def test_workflow_pins_every_action_and_uses_the_pages_actions() -> None:
    text = _text()
    uses = re.findall(r"uses:\s*(\S+)", text)
    assert uses
    assert all(re.search(r"@v\d+(\.\d+)*$", u) for u in uses), uses
    joined = " ".join(uses)
    assert "actions/upload-pages-artifact@" in joined
    assert "actions/deploy-pages@" in joined
    assert "actions/checkout@" in joined


def test_workflow_builds_the_site_with_the_package_command() -> None:
    text = _text()
    assert "python -m tomography.site build --out _site" in text
    assert "path: _site" in text
    assert "environment:" in text
    assert "github-pages" in text


def test_workflow_checks_out_the_history_for_the_masthead_commit() -> None:
    text = _text()
    assert "fetch-depth: 0" in text


def test_workflow_rebuilds_when_the_references_or_the_viz_rules_change() -> (
    None
):
    text = _text()
    for path in ("docs/references.bib", "src/tomography/viz/**"):
        assert f'"{path}"' in text


def test_workflow_installs_the_cpu_torch_group_not_the_cuda_one() -> None:
    lines = [ln for ln in _text().splitlines() if "uv sync" in ln]
    assert lines
    for line in lines:
        assert "--group cpu" in line
        assert "--group cu130" not in line


def test_workflow_prints_the_pdf_between_the_build_and_the_upload() -> None:
    text = _text()
    command = "python -m tomography.site pdf --site _site"
    assert command in text
    assert (
        text.index("tomography.site build")
        < text.index(command)
        < text.index("upload-pages-artifact")
    )


def test_workflow_checks_pull_requests_but_deploys_only_from_main() -> None:
    text = _text()
    on_block = text.split("permissions:", 1)[0]
    assert "pull_request:" in on_block
    assert on_block.count('"src/tomography/site/**"') == 2
    deploy = text.split("  deploy:", 1)[1]
    assert "github.event_name != 'pull_request'" in deploy.split("steps:")[0]
