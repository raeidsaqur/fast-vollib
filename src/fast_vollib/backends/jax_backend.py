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

_SQRT2 = 2.0**0.5
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
    d1 = (
        jnp.log(jnp.maximum(s, 1e-32) / jnp.maximum(k, 1e-32)) + (r - q + 0.5 * sigma**2) * t
    ) / vol_term
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


def price_black(
    flag: np.ndarray, f: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    _ensure_x64()
    import jax
    import jax.numpy as jnp

    is_call = _flag_to_bool(flag)
    ft, kt, tt, rt, st = (jnp.asarray(x, dtype=jnp.float64) for x in (f, k, t, r, sigma))
    qt = rt  # Black-76: q=r so carry=disc and d1 = [ln(F/K)+0.5σ²T]/(σ√T)
    out = jax.jit(_bsm_price_j)(is_call, ft, kt, tt, rt, st, qt)
    return np.asarray(out)


def price_black_scholes(
    flag: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    _ensure_x64()
    import jax
    import jax.numpy as jnp

    is_call = _flag_to_bool(flag)
    st, kt, tt, rt, sigt = (jnp.asarray(x, dtype=jnp.float64) for x in (s, k, t, r, sigma))
    qt = jnp.zeros_like(rt)
    out = jax.jit(_bsm_price_j)(is_call, st, kt, tt, rt, sigt, qt)
    return np.asarray(out)


def price_black_scholes_merton(
    flag: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    _ensure_x64()
    import jax
    import jax.numpy as jnp

    is_call = _flag_to_bool(flag)
    st, kt, tt, rt, sigt, qt = (jnp.asarray(x, dtype=jnp.float64) for x in (s, k, t, r, sigma, q))
    out = jax.jit(_bsm_price_j)(is_call, st, kt, tt, rt, sigt, qt)
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

# Module-level JIT'd greeks core, keyed by model string (same pattern as torch backend)
_jit_greeks_cores: "dict[str, object]" = {}


def _get_jit_greeks_core(model: str):
    if model not in _jit_greeks_cores:
        import jax

        _is_black = model == "black"

        @jax.jit
        def _greeks_core(is_call, st, kt, tt, rt, sigt, qv):
            import jax.numpy as jnp

            d1, d2 = _d1_d2(st, kt, tt, rt, sigt, qv)
            carry = jnp.exp(-qv * tt)
            disc = jnp.exp(-rt * tt)
            pdf = _normal_pdf(d1)
            sqrt_t = jnp.sqrt(jnp.maximum(tt, 1e-32))
            safe_s = jnp.maximum(st, 1e-32)
            safe_sig = jnp.maximum(sigt, 1e-32)

            cdf_d1 = _normal_cdf(d1)
            cdf_d2 = _normal_cdf(d2)
            cdf_nd1 = 1.0 - cdf_d1
            cdf_nd2 = 1.0 - cdf_d2

            if _is_black:
                delta = disc * jnp.where(is_call, cdf_d1, cdf_d1 - 1.0)
                gamma = disc * pdf / (safe_s * safe_sig * sqrt_t)
            else:
                delta = jnp.where(is_call, carry * cdf_d1, carry * (cdf_d1 - 1.0))
                gamma = carry * pdf / (safe_s * safe_sig * sqrt_t)

            vega = st * carry * pdf * sqrt_t * 0.01
            theta_call = (
                -(st * carry * pdf * sigt) / (2.0 * sqrt_t)
                - rt * kt * disc * cdf_d2
                + qv * st * carry * cdf_d1
            ) / 365.0
            theta_put = (
                -(st * carry * pdf * sigt) / (2.0 * sqrt_t)
                + rt * kt * disc * cdf_nd2
                - qv * st * carry * cdf_nd1
            ) / 365.0
            rho_call = kt * tt * disc * cdf_d2 * 0.01
            rho_put = -kt * tt * disc * cdf_nd2 * 0.01
            rho = jnp.where(is_call, rho_call, rho_put)
            theta = jnp.where(is_call, theta_call, theta_put)
            return delta, gamma, theta, rho, vega

        _jit_greeks_cores[model] = _greeks_core
    return _jit_greeks_cores[model]


def greeks(
    model: ModelLiteral,
    flag: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    _ensure_x64()
    import jax.numpy as jnp

    is_call = _flag_to_bool(flag)
    st, kt, tt, rt, sigt = (jnp.asarray(x, dtype=jnp.float64) for x in (s, k, t, r, sigma))
    if model == "black" and q is None:
        qv = rt  # Black-76: q=r so carry=disc and d1 = [ln(F/K)+0.5σ²T]/(σ√T)
    else:
        qv = jnp.zeros_like(rt) if q is None else jnp.asarray(q, dtype=jnp.float64)

    delta, gamma, theta, rho, vega = _get_jit_greeks_core(model)(is_call, st, kt, tt, rt, sigt, qv)
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
_BISECT_ITERS = 30  # 10/(2^30)≈9e-9, sufficient accuracy
_IV_LO = 1e-8
_IV_HI = 10.0


# Module-level JIT'd functions — defined once so JAX caches properly.
# Closures over JIT'd functions cause retracing on every call when the
# closed-over concrete arrays change; explicit arguments avoid this.


def _jax_halley_step(sigma, pt, ds, dk, sqrt_tt, st, kt, tt, rt, qv, is_call):
    """Single Halley step; meant to be called via a module-level @jax.jit wrapper."""
    import jax.numpy as jnp

    d1, d2 = _d1_d2(st, kt, tt, rt, sigma, qv)
    cdf_d1 = _normal_cdf(d1)
    cdf_d2 = _normal_cdf(d2)
    call = ds * cdf_d1 - dk * cdf_d2
    put = dk * (1.0 - cdf_d2) - ds * (1.0 - cdf_d1)
    px = jnp.where(is_call, call, put)
    vega = ds * _normal_pdf(d1) * sqrt_tt
    residual = px - pt
    safe_vega = jnp.where(vega > 1e-14, vega, jnp.full_like(vega, jnp.inf))
    newton_step = residual / safe_vega
    safe_sigma = jnp.where(sigma > 1e-8, sigma, jnp.full_like(sigma, jnp.inf))
    halley_denom = 1.0 - newton_step * d1 * d2 / safe_sigma
    halley_denom = jnp.where(
        jnp.abs(halley_denom) > 0.05, halley_denom, jnp.sign(halley_denom + 1e-15) * 0.05
    )
    return jnp.clip(sigma - newton_step / halley_denom, _IV_LO, _IV_HI)


def _jax_bisect_step(
    sigma_lo, sigma_hi, pt, ds, dk, sqrt_tt, st, kt, tt, rt, qv, is_call, not_converged
):
    """Single bisection step; meant to be called via a module-level @jax.jit wrapper."""
    import jax.numpy as jnp

    sigma_mid = 0.5 * (sigma_lo + sigma_hi)
    d1, d2 = _d1_d2(st, kt, tt, rt, sigma_mid, qv)
    cdf_d1 = _normal_cdf(d1)
    cdf_d2 = _normal_cdf(d2)
    call = ds * cdf_d1 - dk * cdf_d2
    put = dk * (1.0 - cdf_d2) - ds * (1.0 - cdf_d1)
    px_mid = jnp.where(is_call, call, put)
    mid_res = px_mid - pt
    new_lo = jnp.where(not_converged & (mid_res < 0), sigma_mid, sigma_lo)
    new_hi = jnp.where(not_converged & (mid_res >= 0), sigma_mid, sigma_hi)
    return new_lo, new_hi


_jit_halley: "object | None" = None
_jit_halley_all: "object | None" = None
_jit_bisect: "object | None" = None
_jit_iv_pre: "object | None" = None
_jit_iv_post: "object | None" = None


def _get_jit_halley():
    global _jit_halley
    if _jit_halley is None:
        import jax

        _jit_halley = jax.jit(_jax_halley_step)
    return _jit_halley


def _get_jit_halley_all():
    """Fused 8-iteration Halley loop via lax.fori_loop — one XLA kernel."""
    global _jit_halley_all
    if _jit_halley_all is None:
        import jax

        @jax.jit
        def _halley_all(sigma, pt, ds, dk, sqrt_tt, st, kt, tt, rt, qv, is_call):
            def body(_, sig):
                return _jax_halley_step(sig, pt, ds, dk, sqrt_tt, st, kt, tt, rt, qv, is_call)

            return jax.lax.fori_loop(0, _HALLEY_ITERS, body, sigma)

        _jit_halley_all = _halley_all
    return _jit_halley_all


def _get_jit_bisect():
    global _jit_bisect
    if _jit_bisect is None:
        import jax

        _jit_bisect = jax.jit(_jax_bisect_step)
    return _jit_bisect


def _get_jit_iv_pre():
    """JIT'd pre-computation: ds, dk, sqrt_tt, below_intrinsic, sigma_init."""
    global _jit_iv_pre
    if _jit_iv_pre is None:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def _iv_pre(pt, st, kt, tt, rt, qv, is_call):
            ds = st * jnp.exp(-qv * tt)
            dk = kt * jnp.exp(-rt * tt)
            sqrt_tt = jnp.sqrt(jnp.maximum(tt, 1e-32))
            intrinsic = jnp.where(is_call, jnp.maximum(ds - dk, 0.0), jnp.maximum(dk - ds, 0.0))
            below_intrinsic = pt < intrinsic - 1e-10
            sqrt_t_init = jnp.sqrt(jnp.maximum(tt, 1e-8))
            sigma_init = jnp.clip(pt / jnp.maximum(st * sqrt_t_init, 1e-12) * _SQRT2PI, 0.30, 5.0)
            return ds, dk, sqrt_tt, below_intrinsic, sigma_init

        _jit_iv_pre = _iv_pre
    return _jit_iv_pre


def _get_jit_iv_post():
    """JIT'd post-computation: convergence check and result masking."""
    global _jit_iv_post
    if _jit_iv_post is None:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def _iv_post(
            sigma, pt, ds, dk, sqrt_tt, st, kt, tt, rt, qv, is_call, valid, below_intrinsic
        ):
            d1, d2 = _d1_d2(st, kt, tt, rt, sigma, qv)
            cdf_d1 = _normal_cdf(d1)
            cdf_d2 = _normal_cdf(d2)
            call = ds * cdf_d1 - dk * cdf_d2
            put = dk * (1.0 - cdf_d2) - ds * (1.0 - cdf_d1)
            px_final = jnp.where(is_call, call, put)
            underflow_stuck = (px_final == 0.0) & (pt > 0.0)
            not_converged = (jnp.abs(px_final - pt) > 1e-6) | underflow_stuck
            result = jnp.where(valid, sigma, 0.0)
            result = jnp.where(below_intrinsic, jnp.zeros_like(result), result)
            return result, not_converged

        _jit_iv_post = _iv_post
    return _jit_iv_post


def implied_volatility(
    model: ModelLiteral,
    price: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    flag: np.ndarray,
    q: np.ndarray | None = None,
    on_error: str = "warn",
) -> np.ndarray:
    _ensure_x64()
    import jax
    import jax.numpy as jnp

    is_call = _flag_to_bool(flag)
    pt = jnp.asarray(price, dtype=jnp.float64)
    st, kt, tt, rt = (jnp.asarray(x, dtype=jnp.float64) for x in (s, k, t, r))
    if model == "black" and q is None:
        qv = rt  # Black-76: q=r so carry=disc and d1 = [ln(F/K)+0.5σ²T]/(σ√T)
    else:
        qv = jnp.zeros_like(rt) if q is None else jnp.asarray(q, dtype=jnp.float64)

    valid = tt > 0

    # JIT'd pre-computation: ds, dk, sqrt_tt, below_intrinsic, sigma_init
    ds, dk, sqrt_tt, below_intrinsic, sigma = _get_jit_iv_pre()(pt, st, kt, tt, rt, qv, is_call)

    # Halley's method — fused 8-iteration loop via module-level lax.fori_loop JIT
    sigma = _get_jit_halley_all()(sigma, pt, ds, dk, sqrt_tt, st, kt, tt, rt, qv, is_call)

    # JIT'd convergence check and result masking
    result, not_converged = _get_jit_iv_post()(
        sigma, pt, ds, dk, sqrt_tt, st, kt, tt, rt, qv, is_call, valid, below_intrinsic
    )

    if not_converged.any():
        sigma_lo = jnp.where(not_converged, jnp.full_like(sigma, _IV_LO), sigma)
        sigma_hi = jnp.where(not_converged, jnp.full_like(sigma, _IV_HI), sigma)
        bisect = _get_jit_bisect()
        for _ in range(_BISECT_ITERS):
            sigma_lo, sigma_hi = bisect(
                sigma_lo, sigma_hi, pt, ds, dk, sqrt_tt, st, kt, tt, rt, qv, is_call, not_converged
            )
        sigma = jnp.where(not_converged, 0.5 * (sigma_lo + sigma_hi), sigma)
        result = jnp.where(valid, sigma, 0.0)
        result = jnp.where(below_intrinsic, jnp.zeros_like(result), result)

    return np.asarray(result)
