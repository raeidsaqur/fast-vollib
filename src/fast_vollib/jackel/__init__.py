"""
fast-vollib Jäckel module — machine-precision implied volatility.

Stand-alone implementation of Peter Jäckel's "Let's Be Rational" (2016)
algorithm.  All code lives here; the main `backends/` module continues to
use its original Halley×8 + bisection solver.

Public API
----------
jackel_iv_black(price, F, K, T, is_call) -> sigma
    Undiscounted Black-76 IV to machine precision (~2e-11 relative error).

normalised_black_call(x, s) -> b
    Normalised call price b(x,s) = exp(x/2)·Φ(x/s+s/2) − exp(−x/2)·Φ(x/s−s/2).

numpy_backend.implied_volatility(model, price, s, k, t, r, flag, ...) -> sigma
    Full-model IV (Black-76, BSM, BSM-Merton) using Jäckel's solver.

torch_backend.implied_volatility(...)  — stub (I-5): same as numpy for now
jax_backend.implied_volatility(...)    — stub (I-6): same as numpy for now
"""

from .jackel_iv import (
    jackel_iv_black,
    jackel_iv_normalized,
    normalised_black_call,
    normalised_vega,
)

__all__ = [
    "jackel_iv_black",
    "jackel_iv_normalized",
    "normalised_black_call",
    "normalised_vega",
]
