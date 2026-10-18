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
