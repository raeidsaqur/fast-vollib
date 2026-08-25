"""SymPy is a test dependency, not part of the production package."""

from __future__ import annotations

import ast
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src" / "fast_vollib"


def _direct_sympy_imports(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.partition(".")[0] == "sympy" for alias in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.module.partition(".")[0] == "sympy":
                lines.append(node.lineno)
    return lines


def test_production_sources_do_not_import_sympy() -> None:
    violations = {
        str(path.relative_to(PROJECT_ROOT)): lines
        for path in SOURCE_ROOT.rglob("*.py")
        if (lines := _direct_sympy_imports(path))
    }
    assert violations == {}


def test_sympy_is_not_a_published_runtime_requirement() -> None:
    requirements = metadata.requires("fast-vollib") or []
    sympy_requirements = [
        requirement for requirement in requirements if requirement.lower().startswith("sympy")
    ]
    assert sympy_requirements == []
