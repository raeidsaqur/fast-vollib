"""Importing diagnostics must not drag in a heavy or graphical dependency.

The diagnostics layer runs inside overnight batch jobs on headless numerics
hosts.  Importing it must cost no more than importing ``fast_vollib`` itself
already does, so the check is relative: whatever heavy modules bare
``fast_vollib`` pulls in are the baseline, and ``fast_vollib.diagnostics`` may
not add to them.  Each probe runs in its own subprocess so nothing this test
session already imported can mask the answer.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
WATCHED = ("torch", "jax", "jaxlib", "numba", "triton", "matplotlib")

_PROBE = """
import json, sys
WATCHED = {watched!r}

def heavy():
    return sorted({{name.split(".")[0] for name in sys.modules if name.split(".")[0] in WATCHED}})

import fast_vollib
baseline = heavy()
import fast_vollib.diagnostics as diagnostics
after_import = heavy()
{extra}
print(json.dumps({{"baseline": baseline, "after_import": after_import, "final": heavy()}}))
"""


def _probe(extra: str = "") -> dict:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(watched=WATCHED, extra=extra)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_importing_diagnostics_adds_no_heavy_module():
    observed = _probe()
    assert observed["after_import"] == observed["baseline"]


def test_matplotlib_is_never_imported_by_the_package_import():
    observed = _probe()
    assert "matplotlib" not in observed["after_import"]


def test_using_the_numerics_api_stays_matplotlib_free():
    extra = """
import numpy as np
quotes = diagnostics.SurfaceQuotes(k=[-0.1, 0.0, 0.1], T=[0.5, 0.5, 0.5], iv=[0.2, 0.2, 0.2])
sample = diagnostics.diagnose_fit(np.array([0.21, 0.20, 0.19]), quotes)
report = diagnostics.DiagnosticReport.from_records(
    [diagnostics.DiagnosticRecord(model="m", split="s", sample_id=1, diagnostics=sample)]
)
from fast_vollib.diagnostics import serialization
serialization.loads(serialization.dumps(report))
"""
    observed = _probe(extra)
    assert observed["final"] == observed["baseline"]
    assert "matplotlib" not in observed["final"]


def test_the_plotting_helpers_are_reachable_but_lazy():
    extra = """
resolved = callable(diagnostics.plot_density)
assert resolved
"""
    observed = _probe(extra)
    # Touching a figure helper resolves the plots module but must not import
    # matplotlib: that happens on the first *call*, with an actionable message.
    assert "matplotlib" not in observed["final"]


def test_an_unknown_attribute_still_raises_attribute_error():
    from fast_vollib import diagnostics

    with pytest.raises(AttributeError, match="has no attribute 'plot_nothing'"):
        diagnostics.plot_nothing
