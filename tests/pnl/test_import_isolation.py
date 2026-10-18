"""Incremental import cost and lazy root attribute parity in fresh interpreters."""

import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "package",
    ["observations", "portfolio", "pnl", "pnl.plots", "forecasting", "forecasting.adapters"],
)
def test_no_new_optional_runtime_imports(package):
    code = """
import fast_vollib, importlib, json, sys
before=set(sys.modules)
importlib.import_module('fast_vollib.'+sys.argv[1])
added=set(sys.modules)-before
print(json.dumps(sorted(m for m in added if m.split('.')[0] in ('torch','jax','numba','triton','matplotlib'))))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, package], text=True, capture_output=True, check=True
    )
    assert json.loads(result.stdout) == []


def test_root_exports_are_lazy_consistent_and_real():
    code = """
import fast_vollib as fv, importlib, json, sys
names=('observations','portfolio','pnl','forecasting','integrations')
assert not any('fast_vollib.'+n in sys.modules for n in names)
for name in names:
    assert name in fv.__all__ and name in dir(fv)
    assert getattr(fv,name) is importlib.import_module('fast_vollib.'+name)
assert fv.integrations is importlib.import_module('fast_vollib.integrations')
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=True
    )
    assert result.stdout.strip() == "ok"
