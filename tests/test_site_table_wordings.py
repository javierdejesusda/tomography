from pathlib import Path
import re
import runpy

import pytest

from tomography.experiments import variants
from tomography.site import registry
from tomography.site.manifest import Manifest
from tomography.templates import ChatFormat

_HERE = Path(__file__).parent
_R = runpy.run_path(str(_HERE / "_results.py"))
ROOT, data, models, run = _R["ROOT"], _R["data"], _R["models"], _R["run"]
DASHES = (chr(0x2014), chr(0x2013))
DROPPED = "no_trailing_space"
KINDS = {
    "paper": "none, the wording of the paper",
    "no_template": "template removed, chat tags kept",
    "bare": "template and chat tags removed",
    "no_chat_tags": "chat tags removed, template kept",
    "neutral_frame": "frame that does not name the concept",
    "no_trailing_question": "closing question removed",
    "paraphrase": "paraphrase of the instruction",
    "scaffold": "closing sentence unrelated to the concept",
    "concept_swap": "concept word replaced by an unrelated one",
}


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table("wordings", Manifest(ROOT))


def _names() -> list[str]:
    labels = data(run("p1", "gpt2"), "transfer_sentiment.json")["rows"]
    return [n for n in labels if n != DROPPED]


def _templates() -> dict[str, variants.Variant]:
    made, _ = variants.template_variants(
        variants.READING_TASKS["sentiment"], ChatFormat("plain")
    )
    return {v.name: v for v in made}


def test_columns_and_no_row_groups(output: registry.TableOutput) -> None:
    assert output.columns == [
        "No.",
        "Wording",
        "Kind of change",
        "Template",
    ]
    assert output.align == "rlll"
    assert output.row_groups == ()


def test_rows_follow_the_figure_order_with_escaped_templates(
    output: registry.TableOutput,
) -> None:
    names, made = _names(), _templates()
    assert len(output.rows) == len(names) == 16
    for i, (row, name) in enumerate(zip(output.rows, names, strict=True), 1):
        template = made[name].template
        assert row[0] == str(i)
        assert row[2] == KINDS[made[name].kind]
        assert row[3] == template.user.replace("\n", "\n"), name
    assert output.rows[0][1] == "paper"
    assert output.rows[1][1] == "no template"
    assert [r[1] for r in output.rows if "paraphrase" in r[1]] == [
        "paraphrase 1",
        "paraphrase 2",
        "paraphrase 3",
    ]


def test_every_checkpoint_scores_the_same_wordings() -> None:
    expected = _names()
    for model in models("p1"):
        labels = data(run("p1", model), "transfer_sentiment.json")["rows"]
        assert [n for n in labels if n != DROPPED] == expected, model


def test_the_dropped_wording_is_left_out_and_said(
    output: registry.TableOutput,
) -> None:
    assert not any("trailing space" in r[1] for r in output.rows)
    text = output.claim + " " + output.detail
    assert "Figures 5 and A4" in text or "[[fig:wording]]" in text
    assert not any(ch in text for ch in DASHES)


def test_no_underscore_or_run_code(output: registry.TableOutput) -> None:
    cells = [c for r in output.rows for c in r[:3]]
    cells += [output.claim, output.detail, *output.columns]
    flat = [re.sub(r"\[\[.*?\]\]", "", c) for c in cells]
    assert not any("_" in c or "`" in c for c in flat)
