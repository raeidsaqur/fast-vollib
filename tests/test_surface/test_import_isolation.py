"""Describing or fitting a surface must not import a numerical backend.

The surface layer is what a benchmark runner and a web API import to list
algorithms and load results.  Both are processes that may never evaluate a
tensor, and charging them a torch import -- seconds of startup, hundreds of
megabytes, a CUDA context -- for the privilege of reading a capability document
would be a real cost paid for nothing.

Each test runs in a fresh interpreter: ``sys.modules`` is process-global, and by
the time this suite runs the session itself has imported torch.  Every check is
a *delta* against a baseline taken after ``import fast_vollib``, so it measures
what this package pulls in rather than what the environment happens to have.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

#: Modules whose import is the cost this suite exists to prevent.
HEAVY = ("torch", "jax", "jaxlib", "numba", "triton", "matplotlib")

_PROBE = """
import json, sys
HEAVY = {heavy!r}


def heavy():
    return sorted({{name.split(".")[0] for name in sys.modules if name.split(".")[0] in HEAVY}})


import fast_vollib
baseline = heavy()
{body}
print(json.dumps({{"baseline": baseline, "final": heavy(), "value": value}}))
"""


def _probe(body: str) -> dict:
    """Run ``body`` in a fresh interpreter and return its recorded module deltas."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(heavy=HEAVY, body=body)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


# --- the import itself --------------------------------------------------------


def test_importing_the_surface_package_pulls_in_no_backend() -> None:
    probe = _probe("import fast_vollib.surface\nvalue = None\n")
    assert probe["final"] == probe["baseline"]


def test_listing_algorithms_pulls_in_no_backend() -> None:
    probe = _probe(
        "from fast_vollib.surface import list_algorithms\n"
        "value = [entry.spec.public_id for entry in list_algorithms()]\n"
    )
    assert probe["final"] == probe["baseline"]
    assert "flat" in probe["value"]


def test_rendering_the_capability_document_pulls_in_no_backend() -> None:
    probe = _probe(
        "from fast_vollib.surface import capabilities_document\n"
        "value = capabilities_document()['schema']\n"
    )
    assert probe["final"] == probe["baseline"]
    assert probe["value"] == "fast-vollib-surface-capabilities-v1"


def test_fitting_and_materializing_a_surface_pulls_in_no_backend() -> None:
    probe = _probe(
        "import numpy as np\n"
        "from fast_vollib.surface import (\n"
        "    SurfaceGridSpec, SurfaceObservations, materialize_surface,\n"
        ")\n"
        "from fast_vollib.surface.fitting import FlatVolatilityCalibrator\n"
        "observations = SurfaceObservations(k=[-0.1, 0.0, 0.1], T=[1.0] * 3, iv=[0.2] * 3)\n"
        "fitted = FlatVolatilityCalibrator().fit(observations)\n"
        "grid = SurfaceGridSpec(k=np.linspace(-0.2, 0.2, 5), T=[0.5, 1.0])\n"
        "value = bool(materialize_surface(fitted, grid).validate().passed)\n"
    )
    assert probe["final"] == probe["baseline"]
    assert probe["value"] is True


def test_the_capability_document_does_not_import_the_dependency_it_reports_on() -> None:
    # Availability is decided with importlib.util.find_spec, which walks the
    # finders and stops. Executing the module to find out whether it exists
    # would defeat the purpose of asking.
    probe = _probe(
        "from fast_vollib.surface.capabilities import _module_installed\n"
        "value = [_module_installed(name) for name in ('torch', 'jax')]\n"
    )
    assert probe["final"] == probe["baseline"]
