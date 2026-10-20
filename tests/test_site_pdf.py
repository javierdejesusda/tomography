from pathlib import Path
import subprocess
from urllib import parse
from urllib import request

import pytest

from tomography.site import __main__ as cli
from tomography.site import pdf

PAGE = (
    '<html><body><img src="a.png" loading="lazy" alt="a">'
    '<img src="b.png" loading="lazy" alt="b"></body></html>'
)


class FakeBrowser:
    """Stands in for `subprocess.run` and records what it was asked to do."""

    def __init__(self, output: bytes | None = b"%PDF-1.7\nbody") -> None:
        self.output = output
        self.commands: list[list[str]] = []
        self.kwargs: dict[str, object] = {}
        self.printed = ""

    def __call__(
        self, command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        self.commands.append(command)
        self.kwargs = kwargs
        uri = command[-1]
        copy = Path(request.url2pathname(parse.urlparse(uri).path))
        self.printed = copy.read_text("utf-8")
        target = next(a for a in command if a.startswith("--print-to-pdf="))
        if self.output is not None:
            Path(target.removeprefix("--print-to-pdf=")).write_bytes(
                self.output
            )
        return subprocess.CompletedProcess(command, 0, b"", b"")


@pytest.fixture
def site(tmp_path: Path) -> Path:
    (tmp_path / "index.html").write_text(PAGE, encoding="utf-8")
    return tmp_path


def test_the_pdf_has_a_fixed_name() -> None:
    assert pdf.NAME == "tomography-results.pdf"


def test_the_browser_prints_the_copy_headless_without_header_or_footer(
    site: Path,
) -> None:
    fake = FakeBrowser()
    out = pdf.print_page(site, "chrome-bin", run=fake)
    assert out == site / pdf.NAME
    (command,) = fake.commands
    assert command[0] == "chrome-bin"
    assert "--headless" in command
    assert "--no-pdf-header-footer" in command
    assert f"--print-to-pdf={(site / pdf.NAME).resolve()}" in command
    assert any(a.startswith("--virtual-time-budget=") for a in command)
    assert command[-1] == (site / pdf.PRINT_NAME).resolve().as_uri()


def test_the_browser_output_reaches_the_log(site: Path) -> None:
    fake = FakeBrowser()
    pdf.print_page(site, "chrome-bin", run=fake)
    assert "capture_output" not in fake.kwargs
    assert fake.kwargs["check"] is True


def test_the_output_path_is_absolute_for_a_relative_site(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(site.parent)
    fake = FakeBrowser()
    out = pdf.print_page(Path(site.name), "chrome-bin", run=fake)
    flag = next(a for a in fake.commands[0] if a.startswith("--print-to-pdf="))
    assert Path(flag.removeprefix("--print-to-pdf=")).is_absolute()
    assert out.read_bytes().startswith(b"%PDF")


def test_the_print_copy_loads_every_image_and_the_page_is_untouched(
    site: Path,
) -> None:
    fake = FakeBrowser()
    pdf.print_page(site, "chrome-bin", run=fake)
    assert 'loading="lazy"' not in fake.printed
    assert fake.printed.count("<img ") == 2
    assert (site / "index.html").read_text("utf-8") == PAGE


def test_the_print_copy_is_removed_after_success(site: Path) -> None:
    pdf.print_page(site, "chrome-bin", run=FakeBrowser())
    assert not (site / pdf.PRINT_NAME).exists()
    assert (site / pdf.NAME).read_bytes().startswith(b"%PDF")


def test_the_print_copy_is_removed_when_the_browser_fails(site: Path) -> None:
    def failing(command: list[str], **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        pdf.print_page(site, "chrome-bin", run=failing)
    assert not (site / pdf.PRINT_NAME).exists()


def test_a_missing_pdf_raises_and_removes_the_copy(site: Path) -> None:
    with pytest.raises(RuntimeError, match="no PDF"):
        pdf.print_page(site, "chrome-bin", run=FakeBrowser(output=None))
    assert not (site / pdf.PRINT_NAME).exists()


def test_a_file_that_is_not_a_pdf_raises(site: Path) -> None:
    with pytest.raises(RuntimeError, match="not a PDF"):
        pdf.print_page(site, "chrome-bin", run=FakeBrowser(output=b"<html>"))


def test_find_browser_takes_the_first_candidate_on_the_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = {"chromium": "/usr/bin/chromium", "msedge": "/usr/bin/msedge"}
    monkeypatch.setattr(pdf.shutil, "which", found.get)
    assert pdf.find_browser() == "/usr/bin/chromium"


def test_find_browser_names_the_candidates_when_none_is_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pdf.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError) as error:
        pdf.find_browser()
    for name in pdf.BROWSERS:
        assert name in str(error.value)


def test_command_line_pdf_prints_the_site_and_reports_the_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = []

    def fake(site: Path, browser: str) -> Path:
        calls.append((site, browser))
        out = site / pdf.NAME
        out.write_bytes(b"%PDF-" + b"x" * 2043)
        return out

    monkeypatch.setattr(pdf, "print_page", fake)
    code = cli.main(["pdf", "--site", str(tmp_path), "--browser", "chrome-bin"])
    assert code == 0
    assert calls == [(tmp_path, "chrome-bin")]
    assert capsys.readouterr().out == f"wrote {tmp_path / pdf.NAME} (2 KiB)\n"


def test_command_line_pdf_defaults_to_the_browser_on_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake(site: Path, browser: str) -> Path:
        calls.append((site, browser))
        out = site / pdf.NAME
        out.write_bytes(b"%PDF-")
        return out

    monkeypatch.setattr(pdf, "find_browser", lambda: "found-bin")
    monkeypatch.setattr(pdf, "print_page", fake)
    assert cli.main(["pdf", "--site", str(tmp_path)]) == 0
    assert calls == [(tmp_path, "found-bin")]
