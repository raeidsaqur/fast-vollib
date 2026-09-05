"""Describing or applying a curve must not import a numerical backend.

Same contract as the instruments, simulation, and pricing packages, and for the
same reason: a service that discounts a cashflow should not pay for several
hundred megabytes of CUDA runtime.

This package has a second, stricter rule of its own.  It sits at the bottom of
the dependency graph -- ``pricing`` and ``simulation`` reach into it, and it is
supposed to reach back into nothing but the array API.  A cycle here would show
up as an import error that depends on which module a caller happened to import
first, which is the worst kind to debug, so it is checked directly.

``tests/test_package_boundaries.py`` states the same rule by reading the source;
this states it by running the import in a fresh interpreter.  The two catch
different mistakes: a source scan sees an edge nobody has exercised yet, and a
runtime scan sees an edge introduced through a re-export that no ``import``
statement in this package spells out.
"""

from __future__ import annotations

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


def test_importing_the_rates_package_adds_no_heavy_module() -> None:
    output = run_isolated(
        """
        import importlib, pkgutil
        import fast_vollib
        before = {name for name in HEAVY if name in sys.modules}

        import fast_vollib.rates as rates
        for info in pkgutil.iter_modules(rates.__path__):
            importlib.import_module(f"{rates.__name__}.{info.name}")

        after = {name for name in HEAVY if name in sys.modules}
        print(",".join(sorted(after - before)))
        """
    )
    assert output == "", f"rates imported: {output}"


def test_a_bond_is_priceable_with_every_backend_uninstallable() -> None:
    """Not just importable: the whole fixed-income path has to run without them."""
    output = run_isolated(
        """
        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in HEAVY:
                    raise ImportError(f"blocked: {name}")
                return None

        sys.meta_path.insert(0, Blocker())

        from fast_vollib.instruments import ZeroCouponBond
        from fast_vollib.pricing import present_value
        from fast_vollib.rates import CIRDiscountCurve, FlatDiscountCurve

        bond = ZeroCouponBond(maturity=2.0, face_value=100.0)
        flat = present_value(bond, discount_curve=FlatDiscountCurve(rate=0.03))
        assert abs(flat - 94.17645335842487) < 1e-12, flat

        cir = present_value(
            bond,
            discount_curve=CIRDiscountCurve(
                kappa=0.3, theta=0.04, volatility=0.1, initial_rate=0.04
            ),
        )
        assert 0.0 < cir < 100.0, cir

        leaked = sorted(name for name in HEAVY if name in sys.modules)
        print(",".join(leaked))
        """
    )
    assert output == "", f"heavy modules present despite the blocker: {output}"


def test_rates_does_not_import_any_other_fast_vollib_package() -> None:
    """The dependency runs one way. Only the array API is permitted below it."""
    output = run_isolated(
        """
        import importlib, pkgutil
        forbidden = (
            "fast_vollib.instruments", "fast_vollib.simulation", "fast_vollib.pricing",
            "fast_vollib.surface", "fast_vollib.processes", "fast_vollib.diagnostics",
            "fast_vollib.backends", "fast_vollib.jackel",
        )
        # The delta over the parent package, not the absolute set: importing
        # any submodule runs fast_vollib/__init__.py first, which loads the
        # functional API and its backends. That is the parent's cost, and
        # attributing it to this package would make the rule untestable.
        import fast_vollib
        before = {n for n in sys.modules if n.startswith(forbidden)}

        import fast_vollib.rates as rates
        for info in pkgutil.iter_modules(rates.__path__):
            importlib.import_module(f"{rates.__name__}.{info.name}")

        after = {n for n in sys.modules if n.startswith(forbidden)}
        print(",".join(sorted(after - before)))
        """
    )
    assert output == "", f"rates pulled in: {output}"


def test_importing_pricing_does_not_eagerly_import_rates() -> None:
    """``present_value`` reaches for a shipped curve only when it is handed one.

    A caller discounting against their own duck-typed curve should not load a
    package they never mention.
    """
    output = run_isolated(
        """
        import fast_vollib.pricing as pricing
        eager = "fast_vollib.rates" in sys.modules

        from fast_vollib.instruments import ZeroCouponBond

        class OwnCurve:
            def discount_factor(self, maturity):
                return 1.0 / (1.0 + 0.04 * maturity)

        value = pricing.present_value(
            ZeroCouponBond(maturity=1.0, face_value=100.0), discount_curve=OwnCurve()
        )
        assert abs(value - 100.0 / 1.04) < 1e-12, value
        after_own_curve = "fast_vollib.rates" in sys.modules
        print(f"{eager},{after_own_curve}")
        """
    )
    assert output == "False,False"
