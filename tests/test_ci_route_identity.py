# Author: Raeid Saqur
# Date: 2026-11-19
"""Route identity for backend-pinned test sessions.

The CI torch and JAX jobs set ``FAST_VOLLIB_BACKEND``. These tests assert that default calls
(``backend="auto"``) in such a session really run on that backend, so a job cannot pass while
exercising a different route. Without the variable they are skipped.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from fast_vollib import backends, fast_implied_volatility
from fast_vollib.config import get_backend
from fast_vollib.models import fast_black_scholes

PINNED = os.environ.get("FAST_VOLLIB_BACKEND", "")

pytestmark = pytest.mark.skipif(
    PINNED in ("", "auto"), reason="FAST_VOLLIB_BACKEND is not set to a backend"
)


def test_auto_resolves_to_the_pinned_backend() -> None:
    assert get_backend("auto") == PINNED


@pytest.mark.parametrize("solver", [None, "jackel"])
def test_default_calls_run_on_the_pinned_backend(monkeypatch: pytest.MonkeyPatch, solver) -> None:
    import importlib

    if solver == "jackel" and PINNED == "numba":
        pytest.skip("the Jäckel solver has no numba implementation")
    calls: list[str] = []
    module = (
        backends.get_module(PINNED)
        if solver is None
        else importlib.import_module(f"fast_vollib.jackel.{PINNED}_backend")
    )
    original = module.implied_volatility

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(PINNED)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "implied_volatility", spy)
    price = fast_black_scholes("c", 100.0, 100.0, 1.0, 0.02, 0.2, return_as="numpy")
    iv = fast_implied_volatility(
        price, 100.0, 100.0, 1.0, 0.02, "c", solver=solver, return_as="numpy"
    )
    assert calls == [PINNED]
    np.testing.assert_allclose(iv, 0.2, rtol=1e-8)
