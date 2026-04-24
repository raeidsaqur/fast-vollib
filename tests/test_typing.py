"""Opt-in typing harness.

Verifies two things:

1. A default ``import fast_vollib`` does not drag ``jaxtyping`` or
   ``beartype`` into ``sys.modules`` — the hot paths have zero cost from
   the typing enhancement unless the user opts in.
2. When the user *does* opt in via :func:`fast_vollib._typing.enable_runtime_checks`
   the public API still returns correct values on a smoke batch.

The second test is skipped if the ``[typecheck]`` extra is not installed.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest


def _subprocess_env() -> dict[str, str]:
    """Build an env that lets a child python `import fast_vollib` from source."""
    env = os.environ.copy()
    # Propagate sys.path so the child finds fast_vollib even if it's not
    # installed in the child interpreter's site-packages.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [existing])
    return env


def test_default_import_does_not_load_jaxtyping() -> None:
    """In a fresh subprocess, importing fast_vollib must not import jaxtyping/beartype."""
    code = textwrap.dedent(
        """
        import sys
        import fast_vollib  # noqa: F401

        bad = [m for m in ("jaxtyping", "beartype") if m in sys.modules]
        if bad:
            raise SystemExit(f"unexpected modules loaded: {bad}")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _have_typecheck_extra() -> bool:
    try:
        import beartype  # noqa: F401
        import jaxtyping  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _have_typecheck_extra(),
    reason="install with [typecheck] extra to run runtime-check smoke",
)
def test_runtime_checks_smoke() -> None:
    """With runtime checks enabled, the public API still works on valid input."""
    code = textwrap.dedent(
        """
        import numpy as np

        from fast_vollib._typing import enable_runtime_checks
        enable_runtime_checks()

        import fast_vollib  # imported after the hook

        S = np.array([100.0, 100.0])
        K = np.array([100.0, 110.0])
        t = np.array([0.5, 0.5])
        r = np.array([0.05, 0.05])
        sigma = np.array([0.2, 0.2])
        flag = np.array(["c", "p"])

        out = fast_vollib.fast_black_scholes(
            flag, S, K, t, r, sigma, return_as="numpy", backend="numpy"
        )
        assert out.shape == (2,)
        assert np.all(np.isfinite(out))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_typing_aliases_are_lazy() -> None:
    """The _typing module itself must not import jaxtyping at runtime."""
    # Clear any cached import then re-import to confirm no transitive load.
    for name in list(sys.modules):
        if name.startswith("fast_vollib"):
            del sys.modules[name]
    sys.modules.pop("jaxtyping", None)
    sys.modules.pop("beartype", None)

    importlib.import_module("fast_vollib._typing")

    assert "jaxtyping" not in sys.modules
    assert "beartype" not in sys.modules


def test_public_api_unaffected_by_annotations() -> None:
    """Signatures gained annotations but the runtime behavior is unchanged."""
    import fast_vollib

    S = np.array([100.0])
    K = np.array([100.0])
    t = np.array([0.25])
    r = np.array([0.05])
    sigma = np.array([0.2])
    flag = np.array(["c"])

    price = fast_vollib.fast_black_scholes(
        flag, S, K, t, r, sigma, return_as="numpy", backend="numpy"
    )
    assert price.shape == (1,)
    assert np.isfinite(price[0])
