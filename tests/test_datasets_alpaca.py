from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tomography.datasets import alpaca
from tomography.datasets.alpaca import SOURCE
from tomography.datasets.alpaca import load_alpaca
from tomography.datasets.alpaca import parse_alpaca
from tomography.datasets.records import assign_splits

FILE = "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"


def _row(instruction: str, output: str, input_text: str = "") -> dict[str, str]:
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "text": f"{instruction} {output}",
    }


ROWS = [
    _row("Give three tips.", "1. Eat well.\n2. Sleep.\n3. Move. "),
    _row("Summarize the text.", "A summary.", input_text="Some long text."),
    _row("Name a color.", "Blue."),
]


def test_source_pins_the_hub_revision() -> None:
    assert SOURCE.name == "alpaca"
    assert SOURCE.kind == "hub"
    assert SOURCE.location == "tatsu-lab/alpaca"
    assert SOURCE.revision == "dce01c9b08f87459cf36a430d809084718273017"
    assert SOURCE.files == (FILE,)


def test_parse_keeps_only_the_instructions_without_an_input() -> None:
    examples = parse_alpaca(ROWS)

    assert [e.meta["instruction"] for e in examples] == [
        "Give three tips.",
        "Name a color.",
    ]


def test_parse_treats_a_blank_input_as_empty() -> None:
    (example,) = parse_alpaca([_row("Say hi.", "Hi.", input_text="  \n")])

    assert example.text == "Hi."


def test_parse_uses_the_response_as_text_and_the_prompt_as_meta() -> None:
    first = parse_alpaca(ROWS)[0]

    assert first.text == "1. Eat well.\n2. Sleep.\n3. Move."
    assert dict(first.meta) == {"instruction": "Give three tips."}
    assert first.label == -1


def test_parse_numbers_uids_by_the_row_of_the_source_file() -> None:
    examples = parse_alpaca(ROWS)

    assert [e.uid for e in examples] == ["alpaca-0", "alpaca-2"]


def test_parse_drops_a_row_without_a_response() -> None:
    rows = [_row("Say nothing.", ""), _row("Say nothing.", " \n"), *ROWS]

    examples = parse_alpaca(rows)

    assert [e.uid for e in examples] == ["alpaca-2", "alpaca-4"]


def test_parse_assigns_splits_by_uid() -> None:
    rows = [_row(f"Instruction {i}.", f"Response {i}.") for i in range(300)]

    examples = parse_alpaca(rows)

    expected = assign_splits([e.uid for e in examples], dataset="alpaca")
    assert [e.split for e in examples] == expected
    assert {"train", "val", "test"} == set(expected)


def test_parse_gives_every_example_its_own_meta() -> None:
    first, second = parse_alpaca(ROWS)

    assert first.meta is not second.meta


@pytest.fixture
def served(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[dict[str, object]]:
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(ROWS), path)
    calls: list[dict[str, object]] = []

    def fake_fetch(source: object, **kwargs: object) -> Path:
        calls.append({"source": source, **kwargs})
        return path

    monkeypatch.setattr(alpaca, "fetch", fake_fetch)
    return calls


def test_load_reads_the_parquet(
    served: list[dict[str, object]], tmp_path: Path
) -> None:
    examples = load_alpaca(cache=tmp_path, offline=True)

    assert served == [{"source": SOURCE, "cache": tmp_path, "offline": True}]
    assert examples == parse_alpaca(ROWS)


@pytest.mark.network
def test_load_reads_the_pinned_revision() -> None:
    examples = load_alpaca()

    assert len(examples) == 31_311
    assert len({e.uid for e in examples}) == len(examples)
    assert all(e.text and e.meta["instruction"] for e in examples)
