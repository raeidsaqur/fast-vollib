"""Test-session bootstrap: source path, and the required-backend contract.

Optional backends are normally *skipped* when absent, which is right for a
developer running the suite on a laptop and wrong for CI. A skipped test
reports green, so a job that was supposed to exercise the torch route can pass
while proving nothing about it.

``FV_REQUIRE_BACKENDS`` closes that gap. Set it to a comma-separated list of
backend names and the session refuses to start if one of them is missing, and
turns any per-test skip attributed to one of them into a failure. Each CI job
requests exactly the backend it installed.

    FV_REQUIRE_BACKENDS=torch,jax uv run pytest tests/ -q
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
UPSTREAM = ROOT.parent / "py_vollib_vectorized"

for path in (SRC, UPSTREAM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


#: Backends a test may be skipped for, and the module whose presence defines
#: "installed". ``find_spec`` answers that without importing the module.
BACKEND_MODULES = {"torch": "torch", "jax": "jax", "numba": "numba"}

_REQUIRE_ENV = "FV_REQUIRE_BACKENDS"

#: Session-scoped record of what the caller required, read by the report hook.
_REQUIRED_KEY = pytest.StashKey[tuple]()


def required_backends() -> tuple[str, ...]:
    """Backend names the caller declared this session must actually exercise."""
    raw = os.environ.get(_REQUIRE_ENV, "")
    return tuple(name.strip().lower() for name in raw.split(",") if name.strip())


def _installed(backend: str) -> bool:
    try:
        return importlib.util.find_spec(BACKEND_MODULES[backend]) is not None
    except (ImportError, ValueError):  # pragma: no cover - malformed installation
        return False


def pytest_configure(config: pytest.Config) -> None:
    """Fail the session up front when a requested backend is not installed."""
    requested = required_backends()
    if not requested:
        return
    unknown = sorted(set(requested) - set(BACKEND_MODULES))
    if unknown:
        raise pytest.UsageError(
            f"{_REQUIRE_ENV} names unknown backend(s) {', '.join(unknown)}. "
            f"Known backends: {', '.join(sorted(BACKEND_MODULES))}."
        )
    missing = [name for name in requested if not _installed(name)]
    if missing:
        raise pytest.UsageError(
            f"{_REQUIRE_ENV}={os.environ[_REQUIRE_ENV]!r} requires "
            f"{', '.join(missing)}, which {'is' if len(missing) == 1 else 'are'} not "
            f"installed. Install the matching extra, or unset {_REQUIRE_ENV} to let "
            f"the affected tests skip."
        )
    config.stash[_REQUIRED_KEY] = requested


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Turn "backend not installed" skips into failures when it was required.

    The session-level check above already proves the module is importable, so a
    remaining skip that names a required backend means the test opted out for a
    reason the requester did not accept -- exactly what this environment
    variable exists to surface.
    """
    outcome = yield
    report = outcome.get_result()
    if report.outcome != "skipped":
        return
    requested = item.config.stash.get(_REQUIRED_KEY, ())
    if not requested:
        return
    reason = _skip_reason(report)
    named = [name for name in requested if name in reason.lower()]
    if not named:
        return
    report.outcome = "failed"
    report.longrepr = (
        f"{_REQUIRE_ENV} requires {', '.join(named)}, but this test was skipped: {reason}"
    )


def _skip_reason(report: pytest.TestReport) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr)
