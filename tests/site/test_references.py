from pathlib import Path

import pytest

from tomography.site import references

BIB = Path(__file__).resolve().parents[2] / "docs" / "references.bib"

SAMPLE = """\
@misc{zou2025x,
      title={Representation Engineering: A Top-Down Approach},
      author={Andy Zou and Long Phan and Dan Hendrycks},
      year={2025},
      eprint={2310.01405},
      url={https://arxiv.org/abs/2310.01405},
}

@inproceedings{li2023inference,
  title = {Inference-Time Intervention: Eliciting Truthful Answers},
  author = {Li, Kenneth and Patel, Oam and Vi{\\'e}gas, Fernanda},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2023},
  url = {https://example.org/li},
}

@article{solo2024one,
  title = {A Solo Paper},
  author = {Solo, Han},
  journal = {Journal of Tests},
  year = {2024},
}
"""


def test_entries_are_found_by_key_prefix() -> None:
    found = references.parse(SAMPLE)
    assert set(found) == {"zou2025x", "li2023inference", "solo2024one"}
    assert references.find(found, "zou2025").title.startswith("Representation")


def test_authors_are_written_surname_first_with_et_al() -> None:
    found = references.parse(SAMPLE)
    assert references.format_authors(found["zou2025x"]) == "A. Zou et al."
    assert references.format_authors(found["li2023inference"]) == "K. Li et al."
    assert references.format_authors(found["solo2024one"]) == "H. Solo"


def test_latex_accents_are_decoded() -> None:
    found = references.parse(SAMPLE)
    assert "Viegas" not in references.format_authors(found["li2023inference"])
    assert references.plain(r"Vi{\'e}gas") == "Viegas"


def test_a_reference_line_has_authors_year_title_and_venue() -> None:
    entry = references.parse(SAMPLE)["li2023inference"]
    line = references.format_line(entry)
    assert line.startswith("K. Li et al. (2023). Inference-Time Intervention")
    assert "Advances in Neural Information Processing Systems" in line
    assert entry.url == "https://example.org/li"


def test_the_repository_bibliography_has_the_entries_the_page_cites() -> None:
    found = references.parse(BIB.read_text("utf-8"))
    for prefix in references.CITED:
        assert references.find(found, prefix) is not None, prefix


def test_a_missing_key_raises() -> None:
    found = references.parse(SAMPLE)
    with pytest.raises(KeyError):
        references.find(found, "nobody1999")


def test_the_raw_entry_text_is_kept_for_the_cite_block() -> None:
    found = references.parse(SAMPLE)
    assert found["zou2025x"].raw.startswith("@misc{zou2025x,")
    assert found["zou2025x"].raw.rstrip().endswith("}")
