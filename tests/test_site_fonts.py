import hashlib
from pathlib import Path
import re

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src/tomography/site/static"
FONTS = STATIC / "fonts"

SHA256 = {
    "lmroman10-regular.woff2": (
        "36a79526d9269e08ba4dca8cb307ade7b6ac7155dd0cfa4a54b8c95983be4e34"
    ),
    "lmroman10-bold.woff2": (
        "9810f12fb1e8b755c478b640d2ee6c2682a4addaaca7574c1c1294203e550ffb"
    ),
    "lmroman10-italic.woff2": (
        "43b9bb6b35636141db0a7008e7b249c305b04bad22f0797f38192282defa5de1"
    ),
    "lmroman10-bolditalic.woff2": (
        "8cc9f9386522e6bf758cd97be111a2905d9c1c79757d774a1e51003ef8f5ac20"
    ),
    "lmmono10-regular.woff2": (
        "654a1ade8aa4d3494cedb82da87eb8976cf4d42774d74f5d4296667862b76c48"
    ),
    "latinmodern-math-ell.woff2": (
        "5e82ac0d28f8588e8cd49e15747c82e1889597f7c471323c231662e10ec39c7c"
    ),
    "GUST-FONT-LICENSE.txt": (
        "49ea6cb9257bbee0a3979c48a774cd221550ac1c20c95549efe45fc99cc18050"
    ),
}


@pytest.mark.parametrize("name", sorted(SHA256))
def test_each_font_file_has_the_recorded_sha256(name: str) -> None:
    digest = hashlib.sha256((FONTS / name).read_bytes()).hexdigest()
    assert digest == SHA256[name]


def test_no_unrecorded_file_sits_in_the_fonts_directory() -> None:
    assert {p.name for p in FONTS.iterdir()} == set(SHA256)


@pytest.mark.parametrize("name", [n for n in SHA256 if n.endswith(".woff2")])
def test_font_files_are_woff2(name: str) -> None:
    assert (FONTS / name).read_bytes()[:4] == b"wOF2"


def test_the_gust_license_text_ships_with_the_fonts() -> None:
    text = (FONTS / "GUST-FONT-LICENSE.txt").read_text("utf-8")
    assert "GUST Font License" in text


def test_every_font_the_stylesheet_names_exists() -> None:
    css = (STATIC / "site.css").read_text("utf-8")
    urls = re.findall(r'url\("(fonts/[^"]+)"\)', css)
    assert len(urls) == 6
    for url in urls:
        assert (STATIC / url).is_file(), url


def test_the_layer_symbol_comes_from_a_latin_modern_math_subset() -> None:
    css = (STATIC / "site.css").read_text("utf-8")
    face = css[css.index('font-family: "LM Math ell"') :].split("}", 1)[0]
    assert 'url("fonts/latinmodern-math-ell.woff2")' in face
    assert "unicode-range: U+2113;" in face
    assert "font-weight: 100 900;" in face
    assert 'font-family: "LM Math ell", "LM Roman", "Latin Modern Roman"' in css
