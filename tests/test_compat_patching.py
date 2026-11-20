# Author: Raeid Saqur
# Date: 2026-11-19
"""py_vollib / py_vollib_vectorized patching, each case in a fresh interpreter.

Patching mutates the upstream packages for the life of a process, so every case runs in its own
subprocess against the real installed signatures: positional and keyword calls, ``q`` honoured
for Black-Scholes-Merton on every backend, idempotent patching, upstream signatures preserved
and fast_vollib's own functions left untouched.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

import fast_vollib

_BACKENDS = """
from fast_vollib.backends import available_backends
BACKENDS = available_backends()
"""


def _run(code: str) -> None:
    src = os.path.dirname(os.path.dirname(os.path.abspath(fast_vollib.__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _BACKENDS + textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "CASE-OK" in proc.stdout, proc.stdout[-3000:]


def test_patch_py_vollib_matches_upstream_signatures_and_values() -> None:
    pytest.importorskip("py_vollib")
    _run(
        """
        import inspect, math, warnings
        warnings.simplefilter("ignore")
        import py_vollib.black, py_vollib.black_scholes, py_vollib.black_scholes_merton
        import py_vollib.black.implied_volatility as b_iv
        import py_vollib.black_scholes.implied_volatility as bs_iv
        import py_vollib.black_scholes_merton.implied_volatility as bsm_iv
        import py_vollib.black.greeks.numerical as b_num
        import py_vollib.black_scholes.greeks.numerical as bs_num
        import py_vollib.black_scholes_merton.greeks.numerical as bsm_num
        import py_vollib.black.greeks.analytical as b_an
        import py_vollib.black_scholes.greeks.analytical as bs_an
        import py_vollib.black_scholes_merton.greeks.analytical as bsm_an

        S, K, t, r, q, sig = 100.0, 95.0, 0.75, 0.04, 0.025, 0.27
        # references from the unpatched upstream functions
        p_b = py_vollib.black.black("c", S, K, t, r, sig)
        p_bs = py_vollib.black_scholes.black_scholes("p", S, K, t, r, sig)
        p_bsm = py_vollib.black_scholes_merton.black_scholes_merton("c", S, K, t, r, sig, q)
        greeks = ("delta", "gamma", "rho", "theta", "vega")
        ref_g = {
            ("b", g): getattr(b_an, g)("c", S, K, t, r, sig) for g in greeks
        } | {("bs", g): getattr(bs_an, g)("p", S, K, t, r, sig) for g in greeks} | {
            ("bsm", g): getattr(bsm_an, g)("c", S, K, t, r, sig, q) for g in greeks
        }
        targets = [
            (py_vollib.black, "black"), (py_vollib.black_scholes, "black_scholes"),
            (py_vollib.black_scholes_merton, "black_scholes_merton"),
            (b_iv, "implied_volatility"), (bs_iv, "implied_volatility"),
            (bsm_iv, "implied_volatility"),
        ] + [(m, g) for m in (b_num, bs_num, bsm_num) for g in greeks]
        sigs = {(id(m), n): str(inspect.signature(getattr(m, n))) for m, n in targets}

        from fast_vollib import fast_implied_volatility, fast_black
        from fast_vollib.compat import patch_py_vollib
        patch_py_vollib()
        first = {(id(m), n): getattr(m, n) for m, n in targets}
        patch_py_vollib()  # idempotent
        assert all(getattr(m, n) is first[(id(m), n)] for m, n in targets)
        # upstream signatures preserved; fast_vollib's own functions untouched
        assert all(str(inspect.signature(getattr(m, n))) == sigs[(id(m), n)] for m, n in targets)
        assert not hasattr(fast_implied_volatility, "__wrapped__")
        assert not hasattr(fast_black, "__wrapped__")
        inspect.signature(fast_implied_volatility)

        close = lambda a, b, tol: math.isclose(a, b, rel_tol=tol, abs_tol=1e-14)
        # pricing, positional and keyword
        v = py_vollib.black.black("c", S, K, t, r, sig)
        assert isinstance(v, float) and close(v, p_b, 1e-12)
        v = py_vollib.black_scholes.black_scholes(flag="p", S=S, K=K, t=t, r=r, sigma=sig)
        assert close(v, p_bs, 1e-12)
        assert close(py_vollib.black_scholes_merton.black_scholes_merton("c", S, K, t, r, sig, q),
                     p_bsm, 1e-12)
        # implied volatility, positional (upstream order) and keyword, every backend
        for be in BACKENDS:
            assert close(b_iv.implied_volatility(p_b, S, K, r, t, "c", backend=be), sig, 1e-8)
            assert close(bs_iv.implied_volatility(p_bs, S, K, t, r, "p", backend=be), sig, 1e-8)
            got = bsm_iv.implied_volatility(p_bsm, S, K, t, r, q, "c", backend=be)
            assert close(got, sig, 1e-8), (be, got)
            got = bsm_iv.implied_volatility(price=p_bsm, S=S, K=K, t=t, r=r, q=q, flag="c",
                                            backend=be)
            assert close(got, sig, 1e-8), (be, got)
        # greeks: model routing (Black-76, BSM q) against py_vollib's analytical values
        for g in greeks:
            assert close(getattr(b_num, g)("c", S, K, t, r, sig), ref_g[("b", g)], 1e-9), g
            assert close(getattr(bs_num, g)("p", S, K, t, r, sig), ref_g[("bs", g)], 1e-9), g
            assert close(getattr(bsm_num, g)("c", S, K, t, r, sig, q), ref_g[("bsm", g)], 1e-9), g
        print("CASE-OK")
        """
    )


def test_patch_py_vollib_vectorized_positional_q_on_every_backend() -> None:
    pytest.importorskip("py_vollib_vectorized")
    _run(
        """
        import warnings
        warnings.simplefilter("ignore")
        import numpy as np
        import py_vollib_vectorized as pvv
        from fast_vollib import fast_black_scholes_merton
        from fast_vollib.compat import patch_py_vollib_vectorized

        S = np.array([100.0, 100.0]); K = np.array([95.0, 110.0]); t = np.array([0.5, 1.5])
        r = np.array([0.03, 0.03]); q = np.array([0.02, 0.04]); sig = np.array([0.2, 0.35])
        flag = ["c", "p"]
        price = fast_black_scholes_merton(flag, S, K, t, r, sig, q, return_as="numpy",
                                          backend="numpy")
        patch_py_vollib_vectorized()
        patch_py_vollib_vectorized()  # idempotent
        for be in BACKENDS:
            iv = pvv.vectorized_implied_volatility(price, S, K, t, r, flag, q,
                                                   model="black_scholes_merton",
                                                   return_as="numpy", backend=be)
            assert np.allclose(iv, sig, rtol=1e-8), (be, iv)
        print("CASE-OK")
        """
    )
