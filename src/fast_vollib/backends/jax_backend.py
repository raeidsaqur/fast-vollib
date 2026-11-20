from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np

from ..types import ModelLiteral, OnErrorLiteral
from . import _iv_common as _ivc

if TYPE_CHECKING:
    from .._typing import FlagArray, Float1D, OptionalFloat1D  # noqa: F401

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
    flag: FlagArray, f: Float1D, k: Float1D, t: Float1D, r: Float1D, sigma: Float1D
) -> Float1D:
    _ensure_x64()
    import jax
    import jax.numpy as jnp

    is_call = _flag_to_bool(flag)
    ft, kt, tt, rt, st = (jnp.asarray(x, dtype=jnp.float64) for x in (f, k, t, r, sigma))
    qt = rt  # Black-76: q=r so carry=disc and d1 = [ln(F/K)+0.5σ²T]/(σ√T)
    out = jax.jit(_bsm_price_j)(is_call, ft, kt, tt, rt, st, qt)
    return np.asarray(out)


def price_black_scholes(
    flag: FlagArray, s: Float1D, k: Float1D, t: Float1D, r: Float1D, sigma: Float1D
) -> Float1D:
    _ensure_x64()
    import jax
    import jax.numpy as jnp

    is_call = _flag_to_bool(flag)
    st, kt, tt, rt, sigt = (jnp.asarray(x, dtype=jnp.float64) for x in (s, k, t, r, sigma))
    qt = jnp.zeros_like(rt)
    out = jax.jit(_bsm_price_j)(is_call, st, kt, tt, rt, sigt, qt)
    return np.asarray(out)


def price_black_scholes_merton(
    flag: FlagArray,
    s: Float1D,
    k: Float1D,
    t: Float1D,
    r: Float1D,
    sigma: Float1D,
    q: Float1D,
) -> Float1D:
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
    flag: FlagArray,
    s: Float1D,
    k: Float1D,
    t: Float1D,
    r: Float1D,
    sigma: Float1D,
    q: OptionalFloat1D = None,
) -> dict[str, Float1D]:
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
# Implied volatility — safeguarded Halley on the out-of-the-money price
# (problem statement, constants and row policy: backends/_iv_common.py).
# One jitted program per model: classification, MAIN_ITERS Halley steps on every
# row (lax.fori_loop), then a masked bracket phase (lax.while_loop) that runs only
# while some row has not converged.
# ---------------------------------------------------------------------------


def _iv_log_objective_j(a, s, l_fo, l_ko, upper, tiny, use_tail: bool):
    """(ln w, |w'|/w, d1, d2), w = c(s) or u(s) = Fo - c(s); see numpy_backend._iv_log_objective.

    ``use_tail`` is static: the log_ndtr evaluation is traced in only when requested.
    """
    import jax.numpy as jnp
    import jax.scipy.special as jss

    d1 = a / s + 0.5 * s
    d2 = d1 - s
    fo = jnp.exp(l_fo)
    ko = jnp.exp(l_ko)
    n1 = _normal_cdf(jnp.where(upper, -d1, d1))
    n2 = _normal_cdf(d2)
    w = jnp.where(upper, fo * n1 + ko * n2, fo * n1 - ko * n2)
    # w <= 0: c underflowed (sub-normal cancellation) -> ln c = -inf, the root lies above s
    lw = jnp.where(w > 0, jnp.log(jnp.where(w > 0, w, 1.0)), -jnp.inf)

    def tail(lw):
        la = l_fo + jss.log_ndtr(jnp.where(upper, -d1, d1))
        lb = l_ko + jss.log_ndtr(d2)
        lo_branch = jnp.where(
            lb < la, la + jnp.log1p(-jnp.exp(jnp.minimum(lb - la, 0.0))), -jnp.inf
        )
        return jnp.where(tiny, jnp.where(upper, jnp.logaddexp(la, lb), lo_branch), lw)

    lw = tail(lw) if use_tail else lw
    dlw = jnp.exp(l_fo - 0.5 * d1 * d1 - _ivc.LOG_SQRT2PI - lw)
    return lw, dlw, d1, d2


def _iv_step_j(state, consts, it):
    """One safeguarded Halley step on the active rows; see numpy_backend._iv_step."""
    import jax.numpy as jnp

    s, lo, hi, conv, active = state
    a, l_target, l_fo, l_ko, upper, tiny, use_tail = consts
    lw, dlw, d1, d2 = _iv_log_objective_j(a, s, l_fo, l_ko, upper, tiny, use_tail)
    sgn = jnp.where(upper, -1.0, 1.0)
    g = sgn * (lw - l_target)
    below = g < 0
    lo_n = jnp.where(below, s, lo)
    hi_n = jnp.where(below, hi, s)
    delta = g / dlw
    den = 1.0 - 0.5 * delta * (d1 * d2 / s - sgn * dlw)
    den = jnp.where(jnp.abs(den) < 0.5, jnp.where(den < 0, -0.5, 0.5), den)
    step = delta / den
    cand = s - step
    small = jnp.abs(step) <= _ivc.ACCEPT_RTOL * s
    cand_ok = jnp.isfinite(cand)
    ok = cand_ok & (((cand > lo_n) & (cand < hi_n)) | small)
    fb = jnp.where(
        jnp.isinf(hi_n), 4.0 * s, jnp.where(lo_n > 0, jnp.sqrt(jnp.abs(lo_n * hi_n)), 0.25 * s)
    )
    s_n = jnp.minimum(jnp.where(ok, cand, fb), _ivc.S_MAX)
    bounded = jnp.isfinite(hi_n) & (lo_n > 0)
    tight = (hi_n <= lo_n * (1.0 + _ivc.BRACKET_RTOL)) | (jnp.nextafter(lo_n, jnp.inf) >= hi_n)
    done = (small & cand_ok) | (bounded & tight & (it >= _ivc.MAIN_ITERS - 1))
    s = jnp.where(active, s_n, s)
    lo = jnp.where(active, lo_n, lo)
    hi = jnp.where(active, hi_n, hi)
    conv = conv | (active & done)
    return s, lo, hi, conv, active & ~done


_jit_iv: "dict[str, object]" = {}


def _get_jit_iv(model: str):
    if model not in _jit_iv:
        import jax
        import jax.numpy as jnp

        @functools.partial(jax.jit, static_argnames=("use_tail", "precise"))
        def _iv(pt, st, kt, tt, rt, qv, is_call, use_tail=False, precise=False):
            rows = _ivc.classify(jnp, model, pt, st, kt, tt, rt, qv, is_call)
            need_precise = jnp.any(rows["near"])
            if precise:  # rows within the error bound of the double-precision bounds
                exact = _ivc.classify(jnp, model, pt, st, kt, tt, rt, qv, is_call, precise=True)
                near = rows["near"]
                rows = {key: jnp.where(near, exact[key], val) for key, val in rows.items()}
            interior = rows["interior"]
            one = jnp.ones_like(pt)
            fo = jnp.where(interior, rows["Fo"], one)
            ko = jnp.where(interior, rows["Ko"], one)
            tv = jnp.where(interior, rows["tv"], one)
            gap = jnp.where(interior, rows["gap"], one)
            a = jnp.where(interior, rows["a"], 0.0)
            sqrt_t = jnp.sqrt(jnp.where(interior, tt, one))
            upper = gap < tv
            l_target = jnp.log(jnp.where(upper, gap, tv))
            l_fo = jnp.log(fo)
            l_ko = jnp.log(ko)
            tiny = (l_target - l_fo) < _ivc.TINY_LOG
            consts = (a, l_target, l_fo, l_ko, upper, tiny, use_tail)
            s0 = (
                jnp.clip(tv / (fo * sqrt_t) * _ivc.SQRT2PI, _ivc.SEED_SIGMA_LO, _ivc.SEED_SIGMA_HI)
                * sqrt_t
            )
            state = (s0, jnp.zeros_like(pt), jnp.full_like(pt, jnp.inf), interior & False, interior)

            def body(i, st_):
                return _iv_step_j(st_, consts, i)

            state = jax.lax.fori_loop(0, _ivc.MAIN_ITERS, body, state)

            def cond(c):
                i, st_ = c
                return (i < _ivc.MAIN_ITERS + _ivc.MAX_ITERS) & jnp.any(st_[4])

            def wbody(c):
                i, st_ = c
                return i + 1, _iv_step_j(st_, consts, i)

            _, state = jax.lax.while_loop(cond, wbody, (_ivc.MAIN_ITERS, state))
            s, _, _, conv, _ = state
            sigma = jnp.where(interior & conv, s / sqrt_t, jnp.nan)
            sigma = jnp.where(rows["zero_vol"], 0.0, sigma)
            code = jnp.where(interior & ~conv, _ivc.NOT_CONVERGED, rows["code"])
            # Row-code counts on the device: the host receives N_CODES integers, not the codes.
            counts = jnp.bincount(code.astype(jnp.int32).reshape(-1), length=_ivc.N_CODES)
            return sigma, counts, jnp.any(tiny & interior), need_precise

        _jit_iv[model] = _iv
    return _jit_iv[model]


def _rescale_subnormal_quotes(price, s, k):
    """Scale (price, S, K) by an exact power of two where the quote is below 2**-960.

    XLA's CPU runtime flushes sub-normal float64 values to zero, which would turn a tiny but
    valid quote into a zero price. Implied volatility is invariant under a common scaling of
    price, underlying and strike, so those rows are lifted into the normal range on the host.
    """
    p = np.asarray(price, dtype=np.float64)
    tiny = (p > 0) & (p < 2.0**-960)
    if not tiny.any():
        return price, s, k
    s_a, k_a = np.broadcast_arrays(np.asarray(s, np.float64), np.asarray(k, np.float64))
    need = -960 - np.frexp(p)[1] + 1
    room = 1000 - np.frexp(np.maximum(s_a, k_a))[1]
    e = np.where(tiny, np.clip(np.minimum(need, room), 0, None), 0)
    return np.ldexp(p, e), np.ldexp(s_a, e), np.ldexp(k_a, e)


def implied_volatility(
    model: ModelLiteral,
    price: Float1D,
    s: Float1D,
    k: Float1D,
    t: Float1D,
    r: Float1D,
    flag: FlagArray,
    q: OptionalFloat1D = None,
    on_error: OnErrorLiteral = "warn",
) -> Float1D:
    _ensure_x64()
    import jax
    import jax.numpy as jnp

    shape, price, s, k, t, r, flag, q = _ivc.flatten_inputs(price, s, k, t, r, flag, q)
    is_call = _flag_to_bool(flag)
    if jax.default_backend() == "cpu":
        price, s, k = _rescale_subnormal_quotes(price, s, k)
    pt, st, kt, tt, rt = (jnp.asarray(x, dtype=jnp.float64) for x in (price, s, k, t, r))
    if q is not None and model == "black_scholes_merton":
        qv = jnp.asarray(q, dtype=jnp.float64)
    else:
        qv = jnp.zeros_like(rt)
    # The deep-tail (log_ndtr) evaluation is compiled in only when a row needs it, so the common
    # program has no data-dependent branch inside the Halley loop; the program reports the need.
    # Likewise the double-double bounds: compiled in only when a row lies within the error bound
    # of the double-precision ones (both flags are reported by the program itself).
    fn = _get_jit_iv(model)
    sigma, counts, need_tail, need_precise = fn(pt, st, kt, tt, rt, qv, is_call)
    if bool(need_tail) or bool(need_precise):
        sigma, counts, _, _ = fn(
            pt, st, kt, tt, rt, qv, is_call, use_tail=bool(need_tail), precise=bool(need_precise)
        )
    freq = np.asarray(counts)
    _ivc.report_row_errors(
        {c: int(n) for c, n in enumerate(freq) if c != _ivc.OK and n}, int(pt.size), on_error
    )
    return np.asarray(sigma).reshape(shape)
