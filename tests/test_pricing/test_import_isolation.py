"""Pricing a Fourier integral must not import a numerical backend.

Same contract as the instruments and simulation packages, for the same reason: a
process that only wants a Heston price should not pay for several hundred
megabytes of CUDA runtime.  The package docstring already promises it; nothing
checked it until now, and the promise is about to be tested harder -- the fixed
income and BCC97 work adds modules here that reach into
:mod:`fast_vollib.instruments` and :mod:`fast_vollib.rates`, either of which
could bring a backend in through a side door.

Checked in a fresh interpreter, because ``sys.modules`` is process-global and
the suite has already imported torch by the time this runs.

SciPy is deliberately *not* in ``HEAVY``.  It is a hard dependency of the
project rather than an optional backend.  The package still keeps it out of its
module bodies -- deferred into
:func:`~fast_vollib.pricing.heston._gauss_legendre` -- and that is checked here
by reading the source rather than by watching ``sys.modules``: a bare ``import
fast_vollib`` already pulls SciPy in through the backend modules, so a runtime
probe would report ``True`` no matter what this package did.
"""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import textwrap

HEAVY = ("torch", "jax", "jaxlib", "numba", "triton")

PREAMBLE = f"""
import sys
sys.path.insert(0, {str(Path(__file__).resolve().parents[2] / "src")!r})
HEAVY = {HEAVY!r}
"""


def run_isolated(body: str) -> str:
    script = PREAMBLE + textwrap.dedent(body)
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, (
        f"subprocess failed:\n--- stdout ---\n{completed.stdout}\n"
        f"--- stderr ---\n{completed.stderr}"
    )
    return completed.stdout.strip()


def test_importing_the_pricing_package_adds_no_heavy_module() -> None:
    output = run_isolated(
        """
        import importlib, pkgutil
        import fast_vollib
        before = {name for name in HEAVY if name in sys.modules}

        import fast_vollib.pricing as pricing
        for info in pkgutil.iter_modules(pricing.__path__):
            importlib.import_module(f"{pricing.__name__}.{info.name}")

        after = {name for name in HEAVY if name in sys.modules}
        print(",".join(sorted(after - before)))
        """
    )
    assert output == "", f"pricing imported: {output}"


def test_a_price_is_computable_with_every_backend_uninstallable() -> None:
    """Not just importable: the quadrature has to run without them too."""
    output = run_isolated(
        """
        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in HEAVY:
                    raise ImportError(f"blocked: {name}")
                return None

        sys.meta_path.insert(0, Blocker())

        from fast_vollib.pricing import heston_price

        price = float(
            heston_price(
                forward=100.0, strike=100.0, maturity=1.0,
                v0=0.04, kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.7,
            )
        )
        assert 0.0 < price < 100.0, price

        leaked = sorted(name for name in HEAVY if name in sys.modules)
        print(",".join(leaked))
        """
    )
    assert output == "", f"heavy modules present despite the blocker: {output}"


def test_importing_pricing_does_not_pull_in_an_engine_or_a_surface() -> None:
    """The dependency runs one way; a cycle here would be a latent import error."""
    output = run_isolated(
        """
        import importlib, pkgutil
        import fast_vollib.pricing as pricing
        for info in pkgutil.iter_modules(pricing.__path__):
            importlib.import_module(f"{pricing.__name__}.{info.name}")
        pulled = sorted(
            name for name in sys.modules
            if name.startswith(("fast_vollib.simulation", "fast_vollib.surface",
                                "fast_vollib.diagnostics"))
        )
        print(",".join(pulled))
        """
    )
    assert output == "", f"pricing pulled in: {output}"


def test_no_pricing_module_imports_scipy_at_module_level() -> None:
    """SciPy is reached from inside a function, so importing costs nothing.

    Read from the source, not from ``sys.modules``: a bare ``import
    fast_vollib`` already pulls SciPy in through the backend modules, so a
    runtime probe would pass whatever this package did.
    """
    package = Path(__file__).resolve().parents[2] / "src" / "fast_vollib" / "pricing"
    offenders: dict[str, list[int]] = {}
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        lines: list[int] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                lines += [
                    node.lineno for alias in node.names if alias.name.split(".")[0] == "scipy"
                ]
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] == "scipy":
                    lines.append(node.lineno)
        if lines:
            offenders[path.name] = lines
    assert offenders == {}
