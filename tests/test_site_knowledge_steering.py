import csv
import io
from pathlib import Path
import runpy
from typing import Any

import numpy as np
import pytest

from tomography.site import registry
from tomography.site import stats
from tomography.site.manifest import Manifest
from tomography.viz import checks
from tomography.viz import paper

RESULTS = Path(__file__).resolve().parents[1] / "results"
_R = runpy.run_path(str(Path(__file__).with_name("_results.py")))
_T = runpy.run_path(str(Path(__file__).with_name("_titles.py")))
_L = runpy.run_path(str(Path(__file__).with_name("_legends.py")))
SPEC = "knowledge_steering"
HEIGHT = 3.9
EXEMPLARS = (
    ("gpt2-xl", "fewshot"),
    ("qwen3-4b-base", "fewshot"),
    ("qwen3-4b", "fewshot"),
)


@pytest.fixture(scope="module")
def output() -> registry.FigureOutput:
    return registry.build_figure(SPEC, Manifest(RESULTS))


def _rows(output: registry.FigureOutput) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(output.csv)))


def _runs() -> list[tuple[str, str]]:
    found = []
    for model in _R["models"]("p9d"):
        templates = sorted(
            {
                str(r.config["template"])
                for r in Manifest(RESULTS).runs("p9d", model)
            },
            key=("fewshot", "concept").index,
        )
        found.extend((model, t) for t in templates)
    return found


def _path(model: str, template: str) -> Path:
    path: Path = _R["run"]("p9d", model, template=template)
    return path


def _headline(model: str, template: str) -> dict[str, Any]:
    head: dict[str, Any] = _R["data"](_path(model, template), "summary.json")[
        "headline"
    ]
    return head


def _table(model: str, template: str, name: str) -> list[dict[str, Any]]:
    payload = _R["data"](_path(model, template), name)
    return [
        dict(zip(payload["columns"], r, strict=True)) for r in payload["rows"]
    ]


def _label(model: str, template: str) -> str:
    return f"{model} {template}"


def _curves(model: str, template: str) -> dict[str, np.ndarray]:
    path = _path(model, template)
    stem = (
        "curves_obscure_real_entropy"
        if _headline(model, template)["metric"] == "entropy"
        else "curves_obscure_real"
    )
    names = _R["data"](path, f"{stem}.json")["series"]
    arrays = _R["npz"](path, f"{stem}.npz")
    return {
        "x": arrays["x"],
        **{
            f"{kind}:{name}": arrays[kind][i]
            for i, name in enumerate(names)
            for kind in ("mean", "lo", "hi")
        },
    }


def _shift(curves: dict[str, np.ndarray], series: str) -> float:
    mean = curves[f"mean:{series}"]
    origin = float(mean[int(np.argmin(np.abs(curves["x"])))])
    return float(np.max(np.abs(mean - origin)))


def _role_counts(
    flags: dict[tuple[str, str], bool],
) -> dict[str, stats.RoleCount]:
    counts = {r: [0, 0] for r in stats.ROLES}
    for (model, _), flag in flags.items():
        cell = counts[stats.role(model)]
        cell[0] += int(flag)
        cell[1] += 1
    return {r: stats.RoleCount(h, t) for r, (h, t) in counts.items()}


def _runs_text(counts: dict[str, stats.RoleCount]) -> str:
    names = {"gpt2": "GPT-2", "base": "Qwen3 base", "post": "post-trained"}
    parts = [
        f"{counts[r].hits} of {counts[r].total} {names[r]} runs"
        for r in stats.ROLES
        if counts[r].total
    ]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def test_there_are_twelve_runs_in_two_units() -> None:
    runs = _runs()
    assert len(runs) == 12
    metrics = [_headline(*r)["metric"] for r in runs]
    assert metrics.count("entropy") == 6
    assert metrics.count("abstention") == 6


def test_forests_hold_each_unit_on_its_own_axis(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for panel, metric in (("a", "entropy"), ("b", "abstention")):
        keys = {
            r["run"] for r in rows if r["panel"] == panel and r["series"] != ""
        }
        want = {
            _label(*run)
            for run in _runs()
            if _headline(*run)["metric"] == metric
        }
        assert keys == want


def test_each_row_has_direction_known_and_the_three_random_arms(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for run in _runs():
        head = _headline(*run)
        panel = "a" if head["metric"] == "entropy" else "b"
        dose = _table(*run, "table_dose.json")
        got = {
            r["series"]: r
            for r in rows
            if r["panel"] == panel and r["run"] == _label(*run)
        }
        assert set(got) == {
            "direction_unknown",
            "direction_known",
            "random0",
            "random1",
            "random2",
        }
        unknown = got["direction_unknown"]
        assert float(unknown["value"]) == pytest.approx(head["lat_slope"])
        assert float(unknown["lo"]) == pytest.approx(head["lat_lo"])
        assert float(unknown["hi"]) == pytest.approx(head["lat_hi"])
        known = next(
            r
            for r in dose
            if r["type"] == head["group"]
            and r["metric"] == head["metric"]
            and r["side"] == "known"
            and r["arm"] == "lat0"
            and r["vs"] == ""
        )
        assert float(got["direction_known"]["value"]) == pytest.approx(
            known["slope"]
        )
        for arm in ("random0", "random1", "random2"):
            want = next(
                r
                for r in dose
                if r["type"] == head["group"]
                and r["metric"] == head["metric"]
                and r["side"] == "unknown"
                and r["arm"] == arm
                and r["vs"] == ""
            )
            assert float(got[arm]["value"]) == pytest.approx(want["slope"])
            assert float(got[arm]["lo"]) == pytest.approx(want["lo"])
            assert float(got[arm]["hi"]) == pytest.approx(want["hi"])


def test_the_distinguishable_flag_is_the_per_arm_interval_criterion() -> None:
    for run in _runs():
        head = _headline(*run)
        assert head["distinguishable"] == all(
            v[1] > 0 for v in head["abs_gaps"].values()
        )


def test_dose_panels_draw_the_stored_curves_for_three_runs(
    output: registry.FigureOutput,
) -> None:
    rows = _rows(output)
    for letter, run in zip("cde", EXEMPLARS, strict=True):
        curves = _curves(*run)
        for series in ("lat: known", "lat: unknown", "random: unknown"):
            got = [
                r
                for r in rows
                if r["panel"] == letter
                and r["run"] == _label(*run)
                and r["series"] == series
            ]
            assert [float(r["x"]) for r in got] == list(curves["x"])
            assert [float(r["value"]) for r in got] == pytest.approx(
                list(curves[f"mean:{series}"])
            )
            assert [float(r["lo"]) for r in got] == pytest.approx(
                list(curves[f"lo:{series}"])
            )


def test_counts_are_per_role_over_runs(output: registry.FigureOutput) -> None:
    runs = _runs()
    for name, key in (
        ("distinguishable", "distinguishable"),
        ("selective", "selective"),
    ):
        want = _role_counts({r: bool(_headline(*r)[key]) for r in runs})
        for role in stats.ROLES:
            got = output.counts[name][role]
            assert (got.hits, got.total) == (want[role].hits, want[role].total)
            assert output.numbers[f"{name}_{role}"] == want[role].hits
            assert f"n_{role}" not in output.numbers
    assert output.counts["distinguishable"]["post"].total == 6
    assert output.counts["distinguishable"]["gpt2"].total == 3


def test_known_at_least_unknown_runs_come_from_the_curves(
    output: registry.FigureOutput,
) -> None:
    held = {
        run
        for run in _runs()
        if _shift(_curves(*run), "lat: known")
        >= _shift(_curves(*run), "lat: unknown")
    }
    assert {m for m, _ in held} == {
        "gpt2-large",
        "gpt2-xl",
        "qwen3-0.6b-base",
        "qwen3-1.7b-base",
    }
    assert "known_ge_unknown" not in output.numbers
    for role in stats.ROLES:
        assert f"known_ge_unknown_{role}" not in output.numbers
    counts = output.counts["known_ge_unknown"]
    for role in stats.ROLES:
        assert counts[role].hits == sum(stats.role(m) == role for m, _ in held)
        assert (
            counts[role].total == output.counts["distinguishable"][role].total
        )
    for model, template in held:
        plain = _headline(model, template)["metric"] == "entropy"
        name = paper.display_name(model if plain else f"{model} {template}")
        assert name in output.detail


def test_caption_is_the_template_filled_from_the_files(
    output: registry.FigureOutput,
) -> None:
    runs = _runs()
    dist = _role_counts(
        {r: bool(_headline(*r)["distinguishable"]) for r in runs}
    )
    sel = _role_counts({r: bool(_headline(*r)["selective"]) for r in runs})
    assert output.claim == (
        "Adding the knowledge direction moves unknown questions more than "
        "random directions do on GPT-2 and base checkpoints, and not "
        "consistently on post-trained ones."
    )
    assert (
        "The direction moves unknown questions more than each of the 3 "
        f"random directions on {_runs_text(dist)}, and more than known "
        f"questions on {_runs_text(sel)}, by the gap between the absolute "
        "mean slopes over the coefficient grid, whatever the sign; with 3 "
        "random directions a direction passes by chance with probability at "
        "most 1/4."
    ) in output.detail
    assert (
        "The outcome is the entropy shift for GPT-2 and base checkpoints and "
        "the abstention rate for post-trained ones."
    ) in output.detail
    assert "linear slope" not in output.detail
    assert all(dist[r].hits == dist[r].total for r in ("gpt2", "base"))
    assert dist["post"].hits < dist["post"].total
    assert "Figure A6" in output.detail
    assert "Figure A7" in output.detail
    assert "resample the questions" in output.detail


def test_size_guards_and_layout(output: registry.FigureOutput) -> None:
    fig = output.figure
    assert tuple(fig.get_size_inches()) == (paper.FULL_WIDTH, HEIGHT)
    assert checks.guard_problems(fig) == []
    assert checks.problems(fig) == []


def test_five_panel_letters_and_no_dashes(
    output: registry.FigureOutput,
) -> None:
    checks.draw(output.figure)
    letters = [
        t.get_text()
        for ax in output.figure.axes
        for t in ax.texts
        if t.get_text().startswith(r"$\mathbf")
    ]
    assert sorted(letters) == [rf"$\mathbf{{({c})}}$" for c in "abcde"]
    text = f"{output.claim} {output.detail} {output.csv}"
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text


def test_svg_and_pdf_bytes_are_the_same_for_the_same_input() -> None:
    first = registry.build_figure(SPEC, Manifest(RESULTS))
    second = registry.build_figure(SPEC, Manifest(RESULTS))
    with paper.paper_style():
        assert paper.svg_bytes(first.figure) == paper.svg_bytes(second.figure)
        assert paper.pdf_bytes(first.figure) == paper.pdf_bytes(second.figure)


def test_caption_describes_the_grey_band_as_the_mean_of_the_random_arms(
    output: registry.FigureOutput,
) -> None:
    assert (
        "the mean over the 3 random directions on unknown questions with its "
        "95 percent interval that resamples the questions (grey band)"
    ) in output.detail


def test_letters_do_not_touch_their_titles(
    output: registry.FigureOutput,
) -> None:
    gaps = _T["letter_title_gaps"](output.figure)
    assert len(gaps) == 3
    assert min(gaps) >= 2.0


def test_no_two_legend_keys_draw_the_same_symbol(
    output: registry.FigureOutput,
) -> None:
    assert _L["marker_clashes"](output.figure) == []
