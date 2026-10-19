from pathlib import Path

import pytest

from tomography.site import references

BIB = Path(__file__).resolve().parents[1] / "docs" / "references.bib"
DASHES = (chr(0x2014), chr(0x2013))

SAMPLE = """\
@misc{zou2025x,
      title={Representation Engineering: A Top-Down Approach},
      author={Andy Zou and Long Phan and Dan Hendrycks},
      year={2023},
      eprint={2310.01405},
      url={https://arxiv.org/abs/2310.01405},
}

@inproceedings{li2023inference,
  title = {Inference-Time Intervention: Eliciting Truthful Answers},
  author = {Li, Kenneth and Patel, Oam and Vi{\\'e}gas, Fernanda},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {36},
  pages = {41451--41530},
  doi = {10.52202/075280-1797},
  year = {2023},
  url = {https://example.org/li},
}

@article{solo2024one,
  title = {A Solo Paper},
  author = {Solo, Han},
  journal = {Journal of Tests},
  volume = {7},
  number = {2},
  pages = {1--9},
  year = {2024},
}

@article{vran2014wiki,
  title = {Wikidata},
  author = {Vrande{\\v{c}}i{\\'c}, Denny and Kr{\\"o}tzsch, Markus},
  journal = {Communications of the ACM},
  year = {2014},
}
"""


def test_entries_are_found_by_exact_key() -> None:
    found = references.parse(SAMPLE)
    assert references.find(found, "zou2025x").title.startswith("Representation")
    with pytest.raises(KeyError):
        references.find(found, "zou2025")


def test_authors_are_written_in_full_up_to_ten() -> None:
    found = references.parse(SAMPLE)
    assert references.format_authors(found["zou2025x"]) == (
        "A. Zou, L. Phan and D. Hendrycks"
    )
    assert references.format_authors(found["solo2024one"]) == "H. Solo"


def test_more_than_ten_authors_are_cut_to_ten_and_et_al() -> None:
    names = " and ".join(f"Name{i}, Given" for i in range(12))
    entry = references.parse(f"@misc{{many,\n author = {{{names}}},\n}}")[
        "many"
    ]
    text = references.format_authors(entry)
    assert text.endswith("et al.")
    assert "Name9" in text
    assert "Name10" not in text


def test_latex_accents_become_unicode() -> None:
    found = references.parse(SAMPLE)
    assert references.format_authors(found["li2023inference"]).startswith(
        "K. Li, O. Patel and F. Viégas"
    )
    line = references.format_authors(found["vran2014wiki"])
    assert "Vrandečić" in line
    assert "Krötzsch" in line


def test_a_line_has_volume_pages_and_year() -> None:
    entry = references.parse(SAMPLE)["li2023inference"]
    line = references.format_line(entry)
    assert line.startswith("K. Li, O. Patel and F. Viégas (2023).")
    assert "Advances in Neural Information Processing Systems" in line
    assert "36" in line
    assert "41451-41530" in line
    article = references.format_line(references.parse(SAMPLE)["solo2024one"])
    assert "Journal of Tests 7(2), 1-9." in article


def test_the_link_text_is_the_doi_or_the_arxiv_id_never_a_url() -> None:
    found = references.parse(SAMPLE)
    text, url = references.link(found["li2023inference"])
    assert (text, url) == (
        "doi:10.52202/075280-1797",
        "https://doi.org/10.52202/075280-1797",
    )
    text, url = references.link(found["zou2025x"])
    assert (text, url) == (
        "arXiv:2310.01405",
        "https://arxiv.org/abs/2310.01405",
    )
    assert references.link(found["solo2024one"]) == ("", "")


def test_the_raw_entry_text_is_kept_for_the_cite_block() -> None:
    found = references.parse(SAMPLE)
    assert found["zou2025x"].raw.startswith("@misc{zou2025x,")


def test_the_zou_entry_is_dated_2023_with_the_arxiv_version_noted() -> None:
    bib = references.parse(BIB.read_text("utf-8"))
    entry = bib[references.PAPER_KEY]
    assert entry.year == "2023"
    assert "AI Transparency" in entry.title
    assert "v4" in entry.fields["note"]


def test_the_new_entries_exist_with_their_identifiers() -> None:
    bib = references.parse(BIB.read_text("utf-8"))
    assert bib["li2023inference"].fields["pages"] == "41451--41530"
    assert bib["hanley1982meaning"].fields["doi"] == (
        "10.1148/radiology.143.1.7063747"
    )
    assert bib["kwiatkowski2019natural"].fields["doi"] == (
        "10.1162/tacl_a_00276"
    )
    assert bib["lee2019latent"].fields["doi"] == "10.18653/v1/P19-1612"
    assert bib["clark2018think"].fields["eprint"] == "1803.05457"
    assert bib["vrandecic2014wikidata"].fields["doi"] == "10.1145/2629489"


def test_no_bibliography_text_has_a_dash_character() -> None:
    text = BIB.read_text("utf-8")
    assert not any(ch in text for ch in DASHES)
