import importlib
from pathlib import Path
import re

import pytest

from tomography import judges
from tomography import model
from tomography.site import assets
from tomography.site import references
from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.viz import paper

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BIB = ROOT / "docs" / "references.bib"
DASHES = (chr(0x2014), chr(0x2013))

DATASET_MODULES = {
    "sst2": "sst2",
    "ethics": "ethics",
    "repe_facts": "facts",
    "geometry_of_truth": "geometry_of_truth",
    "commonclaim": "geometry_of_truth",
    "truthfulqa": "truthfulqa",
    "arc": "arc",
    "popqa": "popqa",
    "triviaqa": "triviaqa",
    "kuq": "kuq",
    "falseqa": "falseqa",
    "synthal": "synthal",
    "nq_open": "nq_open",
    "wikitext103": "wikitext",
}
EXPECTED_ORDER = (
    "gpt2",
    "qwen3",
    "siebert",
    "distilbert",
    "sst2",
    "ethics",
    "repe_facts",
    "geometry_of_truth",
    "commonclaim",
    "truthfulqa",
    "arc",
    "popqa",
    "triviaqa",
    "kuq",
    "falseqa",
    "synthal",
    "nq_open",
    "wikidata",
    "wikitext103",
)


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table("assets", Manifest(RESULTS))


def _rows() -> dict[str, assets.Asset]:
    return {a.key: a for a in assets.ASSETS}


def test_the_table_lists_every_asset_once_in_order(
    output: registry.TableOutput,
) -> None:
    assert tuple(a.key for a in assets.ASSETS) == EXPECTED_ORDER
    assert output.columns == [
        "Asset",
        "Used for",
        "Citation",
        "Hub ID or source",
        "Revision",
        "License as stated",
    ]
    names = []
    for a in assets.ASSETS:
        if a.key in ("gpt2", "qwen3"):
            names += [
                paper.display_name(k)
                for k in assets.CHECKPOINTS
                if k.startswith(a.key)
            ]
        else:
            names.append(a.name)
    assert [r[0] for r in output.rows] == names
    assert output.align == "llllll"


def test_every_dataset_revision_and_license_equals_the_code_pin() -> None:
    rows = _rows()
    for key, module in DATASET_MODULES.items():
        pin = importlib.import_module(f"tomography.datasets.{module}").SOURCE
        row = rows[key]
        assert row.license.lower() == pin.license.lower(), key
        if pin.kind == "hub":
            assert row.location == pin.location, key
            assert row.revision == pin.revision[:12], key
        else:
            sha = re.search(r"/([0-9a-f]{40})/", pin.location)
            if sha:
                found = re.match(
                    r"https://raw\.githubusercontent\.com/([^/]+/[^/]+)/",
                    pin.location,
                )
                assert found is not None
                assert row.location == f"github.com/{found.group(1)}", key
                assert row.revision == sha.group(1)[:12], key
            else:
                digest = next(iter(pin.sha256.values()))
                assert row.location == pin.location[8:] + pin.files[0], key
                assert row.revision == f"SHA-256 {digest[:12]}", key


def test_models_and_judges_equal_the_code_pins() -> None:
    rows = _rows()
    for key, spec in model.MODELS.items():
        assert assets.CHECKPOINTS[key].repo_id == spec.repo_id
        assert assets.CHECKPOINTS[key].revision == spec.revision
        assert assets.CHECKPOINTS[key].license == spec.license
    gpt2 = [s for k, s in model.MODELS.items() if k.startswith("gpt2")]
    for spec in gpt2:
        assert spec.repo_id.split("/")[1] in rows["gpt2"].location
    primary = judges.JUDGES["sentiment-primary"]
    secondary = judges.JUDGES["sentiment-secondary"]
    assert rows["siebert"].location == primary.repo_id
    assert rows["siebert"].revision == primary.revision[:12]
    assert rows["siebert"].license.lower() == "none declared"
    assert rows["distilbert"].location == secondary.repo_id
    assert rows["distilbert"].revision == secondary.revision[:12]
    assert rows["distilbert"].license.lower() == secondary.license.lower()


def test_every_cite_key_resolves_in_the_bibliography() -> None:
    bib = references.parse(BIB.read_text("utf-8"))
    for asset in assets.ASSETS:
        assert asset.cite, asset.key
        for key in asset.cite:
            assert key in bib, (asset.key, key)


def test_the_citation_cell_holds_page_tokens(
    output: registry.TableOutput,
) -> None:
    rows = {r[0]: r for r in output.rows}
    nq = rows["NQ-open"][2]
    assert nq == "[[cite:kwiatkowski2019natural,lee2019latent]]"


def test_the_caption_says_what_the_table_holds(
    output: registry.TableOutput,
) -> None:
    text = output.claim + " " + output.detail
    assert "pinned revisions" in text
    assert "licenses as stated" in text
    assert "not redistributed" in text
    assert not any(ch in text for ch in DASHES)
    assert "_" not in text


def test_the_table_declares_no_row_groups(output: registry.TableOutput) -> None:
    assert output.row_groups == ()


def test_no_cell_has_a_dash_character_or_an_internal_code(
    output: registry.TableOutput,
) -> None:
    cells = [c for row in output.rows for c in row]
    assert not any(ch in c for c in cells for ch in DASHES)
    assert not any(re.search(r"\b[RPMCA][0-9]+[a-d]?\b", c) for c in cells)


def test_every_checkpoint_revision_is_listed(
    output: registry.TableOutput,
) -> None:
    rows = {r[0]: r for r in output.rows}
    for key, pin in assets.CHECKPOINTS.items():
        row = rows[paper.display_name(key)]
        assert row[3] == pin.repo_id, key
        assert row[4] == pin.revision[:12], key
        assert row[5].lower() == pin.license.lower(), key
    assert len(output.rows) == len(assets.ASSETS) - 2 + len(assets.CHECKPOINTS)


def test_a_license_that_says_to_cite_carries_the_citation_marker(
    output: registry.TableOutput,
) -> None:
    rows = {r[0]: r for r in output.rows}
    assert rows["SST-2"][5] == (
        "Unknown (cite Socher et al., 2013 [[cite:socher2013recursive]])"
    )
    for row in output.rows:
        if "(cite " in row[5]:
            assert "[[cite:" in row[5], row[0]
