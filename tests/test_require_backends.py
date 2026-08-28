"""``FV_REQUIRE_BACKENDS`` turns an optional-backend skip into a failure.

The mechanism lives in the session ``conftest.py``, so it is exercised the only
way that proves it: by running a nested pytest session whose conftest is this
repository's, against a test that skips for a backend reason.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

CONFTEST = (Path(__file__).resolve().parent / "conftest.py").read_text(encoding="utf-8")

SKIPPING_TEST = """
    import pytest

    def test_skips_for_a_backend():
        pytest.skip("torch not installed")

    def test_skips_for_an_unrelated_reason():
        pytest.skip("no CUDA device")

    def test_passes():
        assert True
"""


#: The nested session maps "torch" onto a module that is always importable, so
#: these tests exercise the report hook on any host -- including a CI job that
#: installed no optional backend at all. The uninstalled-backend branch has its
#: own test below, which redirects the map the other way.
INSTALLED_BACKEND_MAP = 'BACKEND_MODULES = {"torch": "numpy", "jax": "jax", "numba": "numba"}'
SHIPPED_BACKEND_MAP = 'BACKEND_MODULES = {"torch": "torch", "jax": "jax", "numba": "numba"}'


@pytest.fixture
def session(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> pytest.Pytester:
    monkeypatch.delenv("FV_REQUIRE_BACKENDS", raising=False)
    pytester.makeconftest(CONFTEST.replace(SHIPPED_BACKEND_MAP, INSTALLED_BACKEND_MAP))
    pytester.makepyfile(test_backend_skips=SKIPPING_TEST)
    return pytester


def test_unset_leaves_skips_alone(session: pytest.Pytester) -> None:
    result = session.runpytest_subprocess("-q")
    result.assert_outcomes(passed=1, skipped=2)


def test_a_required_backend_turns_its_skip_into_a_failure(
    session: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the skip that names the required backend is converted."""
    monkeypatch.setenv("FV_REQUIRE_BACKENDS", "torch")
    result = session.runpytest_subprocess("-q")
    result.assert_outcomes(passed=1, skipped=1, failed=1)
    result.stdout.fnmatch_lines(["*FV_REQUIRE_BACKENDS requires torch*"])


def test_an_unknown_backend_name_stops_the_session(
    session: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo is a usage error, and the message lists the names that work."""
    monkeypatch.setenv("FV_REQUIRE_BACKENDS", "definitely-not-a-backend")
    result = session.runpytest_subprocess("-q")
    assert result.ret != 0
    result.stderr.fnmatch_lines(["*unknown backend*"])


def test_a_known_but_uninstalled_backend_stops_the_session(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: absence must abort, not produce a run of green skips.

    The backend map is redirected at a module that does not exist, which is the
    only way to exercise the uninstalled branch on a host where torch and jax
    are both present -- and a host without them could not run this suite's
    backend tests at all.
    """
    monkeypatch.delenv("FV_REQUIRE_BACKENDS", raising=False)
    pytester.makeconftest(
        CONFTEST.replace(
            SHIPPED_BACKEND_MAP,
            'BACKEND_MODULES = {"torch": "fv_absent_backend", "jax": "jax", "numba": "numba"}',
        )
    )
    pytester.makepyfile(test_backend_skips=SKIPPING_TEST)
    monkeypatch.setenv("FV_REQUIRE_BACKENDS", "torch")
    result = pytester.runpytest_subprocess("-q")
    assert result.ret != 0
    result.stderr.fnmatch_lines(["*requires torch, which is not installed*"])


def test_the_installed_backends_of_this_host_satisfy_a_request() -> None:
    """Whatever this host has, requesting it must be accepted rather than error."""
    import importlib.util

    from conftest import BACKEND_MODULES, _installed

    for backend, module in BACKEND_MODULES.items():
        assert _installed(backend) is (importlib.util.find_spec(module) is not None)


# --- the type-checker baseline -------------------------------------------------


def test_the_mypy_baseline_is_well_formed() -> None:
    """A corrupted baseline would silently accept anything, or reject everything.

    Running mypy here would cost more than the whole rest of this file, so the
    check is structural: every recorded line must look like a diagnostic the
    checker will actually compare against.
    """
    import re

    baseline = Path(__file__).resolve().parents[1] / "scripts" / "mypy_baseline.txt"
    lines = [line for line in baseline.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "an empty baseline would accept any new diagnostic"
    pattern = re.compile(r"^src/fast_vollib/[\w/.]+\.py:\d+: (error|note): .+$")
    for line in lines:
        assert pattern.match(line), line
    assert lines == sorted(lines), "the baseline is stored sorted so diffs stay readable"
