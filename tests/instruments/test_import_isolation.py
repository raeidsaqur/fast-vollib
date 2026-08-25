"""Describing an instrument must not import a numerical backend.

The instruments package is usable wherever a contract needs to be represented
-- a scheduler deciding what to price, a service validating a submitted record,
a process with no GPU and no torch installed.  Importing several hundred
megabytes of CUDA runtime to construct an option would make that impossible,
so the package's imports are checked in a subprocess rather than asserted in a
docstring.

Each test runs in a fresh interpreter: ``sys.modules`` is process-global, and
by the time this suite runs the test session itself has imported torch.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

HEAVY = ("torch", "jax", "jaxlib", "numba", "triton")

PREAMBLE = f"""
import sys
sys.path.insert(0, {str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src")!r})
HEAVY = {HEAVY!r}
"""


def run_isolated(body: str) -> str:
    """Execute ``body`` in a fresh interpreter and return its stdout."""
    script = PREAMBLE + textwrap.dedent(body)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"subprocess failed:\n--- stdout ---\n{completed.stdout}\n"
        f"--- stderr ---\n{completed.stderr}"
    )
    return completed.stdout.strip()


def test_importing_instruments_adds_no_heavy_module() -> None:
    """Nothing under ``fast_vollib.instruments`` pulls in a numerical backend.

    Measured as a delta. The parent package eagerly imports its optional
    differentiable-IV wrappers when torch is installed, which is pre-existing
    behaviour of the functional API and outside this package's control; what
    this package controls is that it adds nothing of its own.
    """
    output = run_isolated(
        """
        import importlib, pkgutil
        import fast_vollib
        before = {name for name in HEAVY if name in sys.modules}

        import fast_vollib.instruments as instruments
        for info in pkgutil.iter_modules(instruments.__path__):
            importlib.import_module(f"{instruments.__name__}.{info.name}")

        after = {name for name in HEAVY if name in sys.modules}
        print(",".join(sorted(after - before)))
        """
    )
    assert output == "", f"instruments imported: {output}"


def test_instruments_is_usable_with_every_backend_uninstallable() -> None:
    """The package works in a process where torch, jax, and numba do not exist.

    A finder that refuses the heavy modules simulates the minimal install. If
    any import were unconditional rather than lazy, this raises.
    """
    output = run_isolated(
        """
        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in HEAVY:
                    raise ImportError(f"blocked: {name}")
                return None

        sys.meta_path.insert(0, Blocker())

        import numpy as np
        from fast_vollib.instruments import (
            EuropeanOption, capabilities, instrument_from_json, instrument_to_json,
            instrument_types, payoff,
        )

        option = EuropeanOption(
            underlier="SPX", option_type="c", strike=5000.0, maturity=0.75,
        )
        assert instrument_from_json(instrument_to_json(option)) == option
        assert float(payoff(option, np.array([5100.0]))[0]) == 100.0
        assert set(instrument_types()) >= {"european_option", "forward"}

        # With no backend installed, no differentiable route is advertised.
        assert capabilities(EuropeanOption).native_autodiff == frozenset()

        leaked = sorted(name for name in HEAVY if name in sys.modules)
        print(",".join(leaked))
        """
    )
    assert output == "", f"heavy modules present despite blocker: {output}"


def test_bare_import_of_fast_vollib_does_not_import_instruments() -> None:
    """The lazy export keeps the functional API's import cost unchanged."""
    output = run_isolated(
        """
        import fast_vollib
        eager = "fast_vollib.instruments" in sys.modules
        _ = fast_vollib.instruments.EuropeanOption
        lazy_then_present = "fast_vollib.instruments" in sys.modules
        print(f"{eager},{lazy_then_present}")
        """
    )
    assert output == "False,True"


def test_capabilities_does_not_import_the_backend_it_reports_on() -> None:
    """Availability is answered with ``find_spec``, which never executes a module.

    The finder installed here returns a valid spec for torch and jax whose
    loader raises if anything actually imports them. A capability set that
    still reports both backends therefore proves the question was answered from
    the module *spec*, not by paying to import a numerical stack.
    """
    output = run_isolated(
        """
        from fast_vollib.instruments import EuropeanOption, capabilities

        import importlib.util

        class ExplodingLoader:
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                raise AssertionError(f"{module.__name__} was actually imported")

        class SpecOnlyFinder:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("torch", "jax"):
                    return importlib.util.spec_from_loader(name, ExplodingLoader())
                return None

        for name in list(sys.modules):
            if name.split(".")[0] in HEAVY:
                del sys.modules[name]
        sys.meta_path.insert(0, SpecOnlyFinder())

        caps = capabilities(EuropeanOption)
        backends = sorted({backend for _m, _s, backend in caps.native_autodiff})
        executed = sorted(name for name in HEAVY if name in sys.modules)
        print(f"{','.join(backends)}|{','.join(executed)}")
        """
    )
    backends, executed = output.split("|")
    assert executed == "", f"capabilities executed: {executed}"
    assert backends == "jax,torch"
