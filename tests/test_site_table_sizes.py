from pathlib import Path
import re
import runpy

import pytest

from tomography.site import registry
from tomography.site import runfiles
from tomography.site.manifest import Manifest
from tomography.viz import paper

_HERE = Path(__file__).parent
_R = runpy.run_path(str(_HERE / "_results.py"))
_M = runpy.run_path(str(_HERE / "_site_math.py"))
ROOT, data, models, npz, run = (
    _R["ROOT"],
    _R["data"],
    _R["models"],
    _R["npz"],
    _R["run"],
)
fmt, groups = _M["fmt"], _M["groups"]
DASHES = (chr(0x2014), chr(0x2013))
SIZES = (2, 8, 32, 64, 128, 512)


@pytest.fixture(scope="module")
def output() -> registry.TableOutput:
    return registry.build_table("sizes", Manifest(ROOT))


def test_columns_and_groups(output: registry.TableOutput) -> None:
    assert output.columns == ["Model", *(str(s) for s in SIZES)]
    assert output.align == "l" + "r" * 6
    assert output.row_groups == groups(models("p3"))
    assert len(output.rows) == 10


def test_every_cell_is_reader_and_random_on_the_report_split(
    output: registry.TableOutput,
) -> None:
    for row, model in zip(output.rows, models("p3"), strict=True):
        path = run("p3", model)
        meta = data(path, "sizes_sentiment.json")
        values = npz(path, "sizes_sentiment.npz")
        layer = meta["layer"] - 1
        split = meta["splits"].index("report")
        reader = meta["series"].index("reader")
        random = meta["series"].index("random")
        assert row[0] == paper.display_name(model)
        for col, size in enumerate(SIZES, 1):
            i = meta["sizes"].index(size)
            acc = values["accuracy"]
            r = acc[i, split, reader, :, layer].mean()
            q = acc[i, split, random, :, layer].mean()
            assert row[col] == f"{fmt(r)} ({fmt(q)})", (model, size)


def test_caption_states_draws_and_pairs_and_drops_the_tautology(
    output: registry.TableOutput,
) -> None:
    text = output.claim + " " + output.detail
    assert "mean over 20 draws of training pairs on 214 report pairs" in text
    assert "random direction at the same layer in parentheses" in text
    assert "within 0.02" not in text
    assert not any(ch in text for ch in DASHES)
    assert output.numbers["draws"] == 20
    assert output.numbers["report_pairs"] == 214


def test_no_underscore_or_run_code(output: registry.TableOutput) -> None:
    texts = [output.claim, output.detail, *output.columns]
    texts += [c for r in output.rows for c in r]
    assert not any("_" in t or "`" in t for t in texts)
    assert not any(re.search(r"\b[RPMCA][0-9]+[a-d]?\b", t) for t in texts)


def test_the_layer_is_looked_up_in_the_layers_array(
    output: registry.TableOutput, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_data, real_arrays = runfiles.data, runfiles.arrays

    def shifted_data(path: Path, name: str):  # type: ignore[no-untyped-def]
        out = real_data(path, name)
        if name == "sizes_sentiment.json":
            out = {**out, "layer": out["layer"] + 100}
        return out

    def shifted_arrays(path: Path, name: str):  # type: ignore[no-untyped-def]
        out = real_arrays(path, name)
        if name == "sizes_sentiment.npz":
            out = {**out, "layers": out["layers"] + 100}
        return out

    monkeypatch.setattr(runfiles, "data", shifted_data)
    monkeypatch.setattr(runfiles, "arrays", shifted_arrays)
    shifted = registry.build_table("sizes", Manifest(ROOT))
    assert shifted.rows == output.rows
