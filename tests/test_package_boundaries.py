"""Which package may depend on which, and that the source still parses as 3.10.

Two rules that are cheap to state, expensive to discover late, and impossible to
see in a diff.

**Dependency direction.**  ``tests/simulation/test_import_isolation.py`` already
checks one edge at runtime, by scanning ``sys.modules`` in a subprocess.  That
catches an edge only if the offending module happens to be imported by the probe.
This file reads the source instead, so an edge is caught the moment it is
written, in any module, whether or not anything imports it yet.

The direction matters because the layers mean different things.  A contract holds
terms; it cannot depend on a curve, a process, or an engine without acquiring a
valuation opinion.  A process holds dynamics; it cannot depend on a contract
without becoming a pricer.  Neither can be recovered once the cycle exists,
because by then something legitimately needs it.

**Python 3.10.**  ``pyproject.toml`` declares ``requires-python = ">=3.10"`` while
``ruff.toml`` sets ``target-version = "py311"``, so ruff will not object to
3.11-only syntax and nothing else in the repository looks.  The CI matrix does
include 3.10, but it reports the breakage as several hundred collection errors on
one job, long after the commit that caused it.  ``ast.parse(...,
feature_version=(3, 10))`` says which file and which line, here, in a second.

It is a *syntax* check and claims nothing more: a 3.11-only standard-library call
compiles fine and is not caught.  ``enum.StrEnum`` is the live example, which is
why :mod:`fast_vollib.instruments.enums` spells its members ``class X(str, Enum)``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "fast_vollib"

#: The floor version ``pyproject.toml`` promises.
MINIMUM_PYTHON = (3, 10)

#: For each package, the sibling packages it must never import.
#:
#: Stated as what is *forbidden* rather than what is allowed, because the
#: allowed set includes every private helper module and would turn an ordinary
#: refactor into a test edit. The forbidden set is the architecture.
FORBIDDEN_IMPORTS: dict[str, frozenset[str]] = {
    # A contract holds terms. It cannot reach a curve, a process, an engine, or
    # a surface without holding a valuation opinion as well.
    "instruments": frozenset({"rates", "processes", "pricing", "simulation", "surface"}),
    # Dynamics know nothing about contracts or engines. ``rates`` is absent
    # here and governed separately below: a process may reach it, but only
    # from inside a function body.
    "processes": frozenset({"instruments", "pricing", "simulation", "surface"}),
    # Pricers are called by engines and surfaces, never the reverse.
    "pricing": frozenset({"simulation", "surface", "diagnostics"}),
    # Curves sit at the bottom. They are reached into by pricing and
    # simulation and reach back into nothing but the array API, which
    # RATES_ALLOWED below states positively.
    "rates": frozenset(
        {"instruments", "processes", "pricing", "simulation", "surface", "diagnostics"}
    ),
}

#: The only first-party modules a curve may depend on.
#:
#: Stated as an allow-list rather than a forbid-list, uniquely for this package,
#: because here the claim is absolute: ``fast_vollib.rates`` is the bottom of
#: the graph. A forbid-list would have to be extended every time a package is
#: added, and would silently stop being true in between.
RATES_ALLOWED = frozenset({"fast_vollib._array_api"})

#: Packages that may be imported by a process only from inside a function body.
#:
#: :meth:`fast_vollib.processes.CIRShortRate.discount_curve` hands back a curve,
#: which is a one-way convenience. Doing it at module level would make importing
#: any process import the whole rates package and invite the reverse edge later.
#:
#: :mod:`fast_vollib.pricing.bcc97` reaches the CIR kernel for the same reason
#: and under the same rule: a caller who wanted Black-76 should not load a
#: term-structure package. ``tests/rates/test_import_isolation.py`` checks the
#: consequence at runtime; this checks the cause, in any module, whether or not
#: anything imports it yet.
FUNCTION_LOCAL_ONLY: dict[str, frozenset[str]] = {
    "processes": frozenset({"rates"}),
    "pricing": frozenset({"rates"}),
}


def _package_of(path: Path) -> str:
    """The dotted package a module lives in, e.g. ``fast_vollib.processes``."""
    return path.parent.relative_to(SOURCE_ROOT.parent).as_posix().replace("/", ".")


def _absolute(package: str, node: ast.ImportFrom) -> set[str]:
    """The absolute module names one ``from ... import ...`` refers to.

    Relative imports are resolved against the containing package, which the
    SymPy boundary check in ``tests/symbolic`` does not do -- and every
    first-party import in this codebase is relative, so an unresolved check
    would pass by never matching anything.
    """
    if not node.level:
        if node.module and node.module.split(".")[0] == "fast_vollib":
            return {node.module}
        return set()
    parts = package.split(".")
    base = ".".join(parts[: len(parts) - (node.level - 1)])
    if node.module:
        return {f"{base}.{node.module}"}
    # ``from . import a, b`` names submodules of ``base``.
    return {f"{base}.{alias.name}" for alias in node.names}


def _imports(path: Path, *, module_level_only: bool) -> set[str]:
    """Absolute ``fast_vollib.*`` modules ``path`` imports.

    With ``module_level_only`` the walk stops at any function or class body, so
    a deliberately deferred import is not reported as a module-level edge.
    """
    package = _package_of(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import):
                names.update(
                    alias.name for alias in child.names if alias.name.split(".")[0] == "fast_vollib"
                )
            elif isinstance(child, ast.ImportFrom):
                names.update(_absolute(package, child))
            elif module_level_only and isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            else:
                visit(child)

    visit(tree)
    return names


def _top_level(name: str) -> str:
    """``fast_vollib.processes.cir`` -> ``processes``; a root module -> itself."""
    tail = name.removeprefix("fast_vollib.")
    return tail.split(".")[0] if tail != name else name


def _modules_of(package: str) -> list[Path]:
    directory = SOURCE_ROOT / package
    return sorted(directory.rglob("*.py")) if directory.is_dir() else []


def _violations(package: str, forbidden: frozenset[str], *, module_level_only: bool) -> dict:
    found = {}
    for path in _modules_of(package):
        offending = sorted(
            name
            for name in _imports(path, module_level_only=module_level_only)
            if _top_level(name) in forbidden
        )
        if offending:
            found[str(path.relative_to(PROJECT_ROOT))] = offending
    return found


# --- the resolver itself, because a broken one passes everything ---------------


def test_relative_imports_resolve_to_absolute_names() -> None:
    """A boundary check that resolves nothing would report no violations."""
    heston = SOURCE_ROOT / "processes" / "heston.py"
    resolved = _imports(heston, module_level_only=True)
    # ``from .._array_api import ...`` (level 2) and ``from .gbm import ...`` (level 1).
    assert "fast_vollib._array_api" in resolved, resolved
    assert "fast_vollib.processes.gbm" in resolved, resolved
    assert all(name.startswith("fast_vollib") for name in resolved), resolved


def test_the_forbidden_sets_name_packages_that_exist() -> None:
    """Guards a rule silently disarmed by a rename or a typo."""
    for package, forbidden in {**FORBIDDEN_IMPORTS, **FUNCTION_LOCAL_ONLY}.items():
        assert _modules_of(package), package
        for name in forbidden:
            assert (SOURCE_ROOT / name).is_dir(), (package, name)


# --- the rules -----------------------------------------------------------------


@pytest.mark.parametrize("package", sorted(FORBIDDEN_IMPORTS), ids=str)
def test_a_package_does_not_import_the_layers_above_it(package: str) -> None:
    assert _violations(package, FORBIDDEN_IMPORTS[package], module_level_only=False) == {}


@pytest.mark.parametrize("package", sorted(FUNCTION_LOCAL_ONLY), ids=str)
def test_a_deferred_dependency_is_not_taken_at_module_level(package: str) -> None:
    assert _violations(package, FUNCTION_LOCAL_ONLY[package], module_level_only=True) == {}


def test_rates_depends_on_nothing_but_the_array_api() -> None:
    """The positive form of the rule: an allow-list, because it is absolute."""
    violations = {}
    for path in _modules_of("rates"):
        offending = sorted(
            name
            for name in _imports(path, module_level_only=False)
            # A curve module may of course import its own siblings.
            if not name.startswith("fast_vollib.rates") and name not in RATES_ALLOWED
        )
        if offending:
            violations[str(path.relative_to(PROJECT_ROOT))] = offending
    assert violations == {}


def test_the_rates_allow_list_is_not_vacuous() -> None:
    """A resolver returning nothing would make the rule above pass trivially."""
    kernel = SOURCE_ROOT / "rates" / "cir.py"
    resolved = _imports(kernel, module_level_only=False)
    assert "fast_vollib._array_api" in resolved, resolved
    assert any(name.startswith("fast_vollib.rates") for name in resolved), resolved


# --- the language floor --------------------------------------------------------


def test_every_source_file_parses_under_the_minimum_python() -> None:
    """``match``, ``except*``, and PEP 695 generics would all fail here."""
    files = sorted(SOURCE_ROOT.rglob("*.py"))
    assert len(files) > 50, "the scan found suspiciously few files"
    failures = []
    for path in files:
        try:
            ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
                feature_version=MINIMUM_PYTHON,
            )
        except SyntaxError as error:
            failures.append(f"{path.relative_to(PROJECT_ROOT)}:{error.lineno}: {error.msg}")
    assert failures == []


def test_the_declared_floor_matches_the_packaging_metadata() -> None:
    """The check is worthless if it drifts from what the project promises."""
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10"' in text
    assert MINIMUM_PYTHON == (3, 10)
