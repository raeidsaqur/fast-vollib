from __future__ import annotations

import numpy as np

from ..types import ModelLiteral

# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def is_available() -> bool:
    try:
        import jax  # noqa: F401
    except ImportError:
        return False
    return True


def _ensure_x64() -> None:
    """Enable JAX double precision.  Must be called before any JAX computation."""
    import jax
    jax.config.update("jax_enable_x64", True)


def to_native(values: np.ndarray):
    import jax.numpy as jnp
    return jnp.asarray(values)


def from_native(values) -> np.ndarray:
    return np.asarray(values)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SQRT2 = 2.0 ** 0.5
_SQRT2PI = (2.0 * 3.141592653589793) ** 0.5


# ---------------------------------------------------------------------------
# Normal distribution helpers (erfc-based for accurate extreme tails)
# ---------------------------------------------------------------------------

def _normal_cdf(x):
    """Accurate normal CDF using erfc; matches scipy.special.ndtr for |x|>8."""
    import jax.scipy.special as jss
    return 0.5 * jss.erfc(-x / _SQRT2)


def _normal_pdf(x):
    import jax.numpy as jnp
    return jnp.exp(-0.5 * x * x) / _SQRT2PI


# ---------------------------------------------------------------------------
# Core pricing (all ops on JAX arrays; @jax.jit applied at call sites)
# ---------------------------------------------------------------------------

def _d1_d2(s, k, t, r, sigma, q):
    import jax.numpy as jnp
    sqrt_t = jnp.sqrt(jnp.maximum(t, 1e-32))
    vol_term = jnp.maximum(sigma * sqrt_t, 1e-32)
    d1 = (jnp.log(jnp.maximum(s, 1e-32) / jnp.maximum(k, 1e-32))
          + (r - q + 0.5 * sigma ** 2) * t) / vol_term
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def _bsm_price_j(is_call, s, k, t, r, sigma, q):
    import jax.numpy as jnp
    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    discounted_spot = s * jnp.exp(-q * t)
    discounted_strike = k * jnp.exp(-r * t)
    call = discounted_spot * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
    put = discounted_strike * _normal_cdf(-d2) - discounted_spot * _normal_cdf(-d1)
    return jnp.where(is_call, call, put)


def _vega_raw_j(s, k, t, r, sigma, q):
    import jax.numpy as jnp
    d1, _ = _d1_d2(s, k, t, r, sigma, q)
    return s * jnp.exp(-q * t) * _normal_pdf(d1) * jnp.sqrt(jnp.maximum(t, 1e-32))


def _flag_to_bool(flag: np.ndarray):
    """Convert flag array ('c'/'p') to a boolean JAX array (True = call)."""
    import jax.numpy as jnp
    return jnp.array(flag == "c")


def price_black(flag: np.ndarray, f: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    _ensure_x64()
    import jax, jax.numpy as jnp
    is_call = _flag_to_bool(flag)
    ft, kt, tt, rt, st = (jnp.asarray(x, dtype=jnp.float64) for x in (f, k, t, r, sigma))
    qt = jnp.zeros_like(rt)
    out = jax.jit(_bsm_price_j)(is_call, ft, kt, tt, rt, st, qt)
    return np.asarray(out)


def price_black_scholes(flag: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    _ensure_x64()
    import jax, jax.numpy as jnp
    is_call = _flag_to_bool(flag)
    st, kt, tt, rt, sigt = (jnp.asarray(x, dtype=jnp.float64) for x in (s, k, t, r, sigma))
    qt = jnp.zeros_like(rt)
    out = jax.jit(_bsm_price_j)(is_call, st, kt, tt, rt, sigt, qt)
    return np.asarray(out)


def price_black_scholes_merton(flag: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray, q: np.ndarray) -> np.ndarray:
    _ensure_x64()
    import jax, jax.numpy as jnp
    is_call = _flag_to_bool(flag)
    st, kt, tt, rt, sigt, qt = (jnp.asarray(x, dtype=jnp.float64) for x in (s, k, t, r, sigma, q))
    out = jax.jit(_bsm_price_j)(is_call, st, kt, tt, rt, sigt, qt)
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def greeks(model: ModelLiteral, flag: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray, q: np.ndarray | None = None) -> dict[str, np.ndarray]:
    _ensure_x64()
    import jax, jax.numpy as jnp

    is_call = _flag_to_bool(flag)
    st, kt, tt, rt, sigt = (jnp.asarray(x, dtype=jnp.float64) for x in (s, k, t, r, sigma))
    qv = jnp.zeros_like(rt) if q is None else jnp.asarray(q, dtype=jnp.float64)

    @jax.jit
    def _greeks_jit(is_call, st, kt, tt, rt, sigt, qv):
        d1, d2 = _d1_d2(st, kt, tt, rt, sigt, qv)
        carry = jnp.exp(-qv * tt)
        disc = jnp.exp(-rt * tt)
        pdf = _normal_pdf(d1)
        sqrt_t = jnp.sqrt(jnp.maximum(tt, 1e-32))
        safe_s = jnp.maximum(st, 1e-32)
        safe_sig = jnp.maximum(sigt, 1e-32)

        delta = jnp.where(is_call,
                          carry * _normal_cdf(d1),
                          carry * (_normal_cdf(d1) - 1.0))
        gamma = carry * pdf / (safe_s * safe_sig * sqrt_t)
        vega = st * carry * pdf * sqrt_t * 0.01
        theta_call = (-(st * carry * pdf * sigt) / (2.0 * sqrt_t)
                      - rt * kt * disc * _normal_cdf(d2)
                      + qv * st * carry * _normal_cdf(d1)) / 365.0
        theta_put = (-(st * carry * pdf * sigt) / (2.0 * sqrt_t)
                     + rt * kt * disc * _normal_cdf(-d2)
                     - qv * st * carry * _normal_cdf(-d1)) / 365.0
        rho_call = kt * tt * disc * _normal_cdf(d2) * 0.01
        rho_put = -kt * tt * disc * _normal_cdf(-d2) * 0.01
        rho = jnp.where(is_call, rho_call, rho_put)
        theta = jnp.where(is_call, theta_call, theta_put)

        if model == "black":
            delta = disc * jnp.where(is_call, _normal_cdf(d1), _normal_cdf(d1) - 1.0)
            gamma = disc * pdf / (safe_s * safe_sig * sqrt_t)

        return delta, gamma, theta, rho, vega

    delta, gamma, theta, rho, vega = _greeks_jit(is_call, st, kt, tt, rt, sigt, qv)
    return {
        "delta": np.asarray(delta),
        "gamma": np.asarray(gamma),
        "theta": np.asarray(theta),
        "rho": np.asarray(rho),
        "vega": np.asarray(vega),
    }


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

_HALLEY_ITERS = 8
_BISECT_ITERS = 50
_IV_LO = 1e-8
_IV_HI = 10.0


def implied_volatility(model: ModelLiteral, price: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, flag: np.ndarray, q: np.ndarray | None = None, on_error: str = "warn") -> np.ndarray:
    _ensure_x64()
    import jax, jax.numpy as jnp
    from ..utils.validation import handle_error

    is_call = _flag_to_bool(flag)
    pt = jnp.asarray(price, dtype=jnp.float64)
    st, kt, tt, rt = (jnp.asarray(x, dtype=jnp.float64) for x in (s, k, t, r))
    qv = jnp.zeros_like(rt) if q is None else jnp.asarray(q, dtype=jnp.float64)

    valid = tt > 0

    # Brenner-Subrahmanyam initial guess with floor=0.30
    sqrt_t = jnp.sqrt(jnp.maximum(tt, 1e-8))
    sigma = jnp.clip(pt / jnp.maximum(st * sqrt_t, 1e-12) * _SQRT2PI, 0.30, 5.0)

    def _price_vega_d1d2(sig):
        d1, d2 = _d1_d2(st, kt, tt, rt, sig, qv)
        ds = st * jnp.exp(-qv * tt)
        dk = kt * jnp.exp(-rt * tt)
        sqrt_tt = jnp.sqrt(jnp.maximum(tt, 1e-32))
        call = ds * _normal_cdf(d1) - dk * _normal_cdf(d2)
        put = dk * _normal_cdf(-d2) - ds * _normal_cdf(-d1)
        px = jnp.where(is_call, call, put)
        vega = ds * _normal_pdf(d1) * sqrt_tt
        return px, vega, d1, d2

    # Halley's method (3rd order; 8 iters ≡ ~12 Newton iters in accuracy)
    @jax.jit
    def _halley_step(sigma):
        px, vega, d1, d2 = _price_vega_d1d2(sigma)
        residual = px - pt
        safe_vega = jnp.where(vega > 1e-14, vega, jnp.full_like(vega, jnp.inf))
        newton_step = residual / safe_vega
        safe_sigma = jnp.where(sigma > 1e-8, sigma, jnp.full_like(sigma, jnp.inf))
        halley_denom = 1.0 - newton_step * d1 * d2 / safe_sigma
        halley_denom = jnp.where(jnp.abs(halley_denom) > 0.05, halley_denom, jnp.sign(halley_denom + 1e-15) * 0.05)
        return jnp.clip(sigma - newton_step / halley_denom, _IV_LO, _IV_HI)

    for _ in range(_HALLEY_ITERS):
        sigma = _halley_step(sigma)

    def _price(sig):
        return _bsm_price_j(is_call, st, kt, tt, rt, sig, qv)

    px_final = _price(sigma)
    underflow_stuck = (px_final == 0.0) & (pt > 0.0)
    not_converged = (jnp.abs(px_final - pt) > 1e-6) | underflow_stuck

    if not_converged.any():
        sigma_lo = jnp.where(not_converged, jnp.full_like(sigma, _IV_LO), sigma)
        sigma_hi = jnp.where(not_converged, jnp.full_like(sigma, _IV_HI), sigma)

        @jax.jit
        def _bisect_step(sigma_lo, sigma_hi):
            sigma_mid = 0.5 * (sigma_lo + sigma_hi)
            px_mid = _price(sigma_mid)
            mid_res = px_mid - pt
            new_lo = jnp.where(not_converged & (mid_res < 0), sigma_mid, sigma_lo)
            new_hi = jnp.where(not_converged & (mid_res >= 0), sigma_mid, sigma_hi)
            return new_lo, new_hi

        for _ in range(_BISECT_ITERS):
            sigma_lo, sigma_hi = _bisect_step(sigma_lo, sigma_hi)
        sigma = jnp.where(not_converged, 0.5 * (sigma_lo + sigma_hi), sigma)

    result = jnp.where(valid, sigma, 0.0)
    return np.asarray(result)
