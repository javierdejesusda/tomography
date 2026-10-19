import ast
from pathlib import Path
import re

import pytest

SITE = Path(__file__).resolve().parents[1] / "src" / "tomography" / "site"
LITERAL = re.compile(r"\b(Figures?|Tables?)\s+(A\d+|\d+)\b")


def _strings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.ClassDef) and (
            node.body and isinstance(node.body[0], ast.Expr)
        ):
            first = node.body[0].value
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                skip.add(id(first))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in skip
    ]


@pytest.mark.parametrize("kind", ["figs", "tables"])
def test_no_caption_writes_a_literal_figure_or_table_number(kind: str) -> None:
    found = []
    for path in sorted((SITE / kind).glob("*.py")):
        for text in _strings(path):
            found.extend(
                f"{path.name}: {m.group(0)}" for m in LITERAL.finditer(text)
            )
    assert found == []
