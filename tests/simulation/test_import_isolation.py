"""Describing or driving a simulation must not import a numerical backend.

Same contract as the instruments package, for the same reason: a service that
validates a scenario request, or a scheduler deciding what to simulate, should
not pay for several hundred megabytes of CUDA runtime.  Checked in a fresh
interpreter, because ``sys.modules`` is process-global and this session has
already imported torch by the time the suite runs.
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


def test_importing_the_new_packages_adds_no_heavy_module() -> None:
    output = run_isolated(
        """
        import importlib, pkgutil
        import fast_vollib
        before = {name for name in HEAVY if name in sys.modules}

        import fast_vollib.processes as processes
        import fast_vollib.simulation as simulation
        import fast_vollib._random_api  # noqa: F401
        for package in (processes, simulation):
            for info in pkgutil.iter_modules(package.__path__):
                importlib.import_module(f"{package.__name__}.{info.name}")

        after = {name for name in HEAVY if name in sys.modules}
        print(",".join(sorted(after - before)))
        """
    )
    assert output == "", f"simulation imported: {output}"


def test_simulation_runs_with_every_optional_backend_uninstallable() -> None:
    """A NumPy-only process must be able to simulate and evaluate a payoff."""
    output = run_isolated(
        """
        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in HEAVY:
                    raise ImportError(f"blocked: {name}")
                return None

        sys.meta_path.insert(0, Blocker())

        import numpy as np
        from fast_vollib.instruments import EuropeanOption
        from fast_vollib.processes import GBM
        from fast_vollib.simulation import simulate

        scenario = simulate(
            "ACME",
            GBM.risk_neutral(rate=0.0, volatility=0.2),
            initial_state=100.0,
            time_grid=np.array([0.0, 0.5, 1.0]),
            n_paths=64,
            rng=0,
        )
        assert scenario.n_paths == 64
        assert float(scenario.spot[0, 0]) == 100.0
        option = EuropeanOption(
            underlier="ACME", option_type="call", strike=100.0, maturity=1.0,
        )
        assert scenario.payoff(option).shape == (64,)

        leaked = sorted(name for name in HEAVY if name in sys.modules)
        print(",".join(leaked))
        """
    )
    assert output == "", f"heavy modules present despite blocker: {output}"


def test_bare_import_of_fast_vollib_does_not_import_the_new_packages() -> None:
    output = run_isolated(
        """
        import fast_vollib
        eager = [
            name for name in ("processes", "simulation")
            if f"fast_vollib.{name}" in sys.modules
        ]
        _ = fast_vollib.simulation.simulate
        _ = fast_vollib.processes.GBM
        lazy = [
            name for name in ("processes", "simulation")
            if f"fast_vollib.{name}" in sys.modules
        ]
        print(f"{','.join(eager)}|{','.join(lazy)}")
        """
    )
    assert output == "|processes,simulation"


def test_processes_does_not_import_simulation_or_instruments() -> None:
    """The dependency runs one way; a cycle here would be a latent import error."""
    output = run_isolated(
        """
        import importlib, pkgutil
        import fast_vollib.processes as processes
        for info in pkgutil.iter_modules(processes.__path__):
            importlib.import_module(f"{processes.__name__}.{info.name}")
        pulled = sorted(
            name for name in sys.modules
            if name.startswith(("fast_vollib.simulation", "fast_vollib.instruments"))
        )
        print(",".join(pulled))
        """
    )
    assert output == "", f"processes pulled in: {output}"
