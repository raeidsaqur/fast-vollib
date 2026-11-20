# Author: Raeid Saqur
# Date: 2026-11-20
"""The Jäckel refusal messages (outside domain, not converged) suggest solver="halley"."""

from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pytest

from fast_vollib import fast_implied_volatility

HINT = 'solver="halley"'
BACKENDS = ["numpy"] + [
    b for b, m in (("torch", "torch"), ("jax", "jax")) if importlib.util.find_spec(m)
]


def _warnings(**kw) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fast_implied_volatility(return_as="numpy", **kw)
    return np.asarray(out, dtype=float), [
        str(w.message) for w in caught if str(w.message).startswith("Implied volatility")
    ]


# beta = 1e-310 / sqrt(100 * 200) < DBL_MIN: outside the Jäckel domain, inside Halley's
OUTSIDE = dict(
    price=np.array([1e-310, 10.0]), S=100.0, K=np.array([200.0, 100.0]), t=1.0, r=0.0, flag="c"
)
ZERO = dict(price=np.array([0.0, 10.0]), S=100.0, K=100.0, t=1.0, r=0.0, flag="c")


@pytest.mark.parametrize("backend", BACKENDS)
def test_jackel_refusal_names_halley(backend: str) -> None:
    iv, msgs = _warnings(solver="jackel", backend=backend, **OUTSIDE)
    assert np.isnan(iv[0]) and np.isfinite(iv[1])
    assert len(msgs) == 1 and msgs[0].startswith("Implied volatility is undefined for 1 of 2 rows")
    assert HINT in msgs[0]
    iv_h, msgs_h = _warnings(solver="halley", backend=backend, **OUTSIDE)
    assert np.isfinite(iv_h).all() and not msgs_h


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_hint_when_halley_cannot_help(backend: str) -> None:
    _, msgs = _warnings(solver="jackel", backend=backend, **ZERO)
    assert len(msgs) == 1 and HINT not in msgs[0]
    _, msgs_h = _warnings(solver="halley", backend=backend, **ZERO)
    assert len(msgs_h) == 1 and HINT not in msgs_h[0]
