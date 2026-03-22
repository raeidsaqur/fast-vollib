from __future__ import annotations

import numpy as np

from ..types import ModelLiteral

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def is_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(arr: np.ndarray, device):
    import torch
    return torch.as_tensor(arr, dtype=torch.float64, device=device)


def to_native(values: np.ndarray):
    import torch
    return torch.as_tensor(values, dtype=torch.float64, device=_device())


def from_native(values) -> np.ndarray:
    import torch
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------

_SQRT2 = 2.0 ** 0.5
_SQRT2PI = (2.0 * 3.141592653589793) ** 0.5


def _normal_cdf(x):
    """Accurate normal CDF for all x, including extreme tails.

    torch.distributions.Normal.cdf uses erf which loses precision for |x|>~8
    (the tail value underflows to 0.0 or 1.0).  erfc preserves ~16 significant
    digits down to the float64 underflow floor (~5e-324).
    """
    import torch
    return 0.5 * torch.special.erfc(-x / _SQRT2)


def _normal_pdf(x):
    import torch
    return torch.exp(-0.5 * x * x) / _SQRT2PI


# ---------------------------------------------------------------------------
# Core pricing (all ops on torch tensors)
# ---------------------------------------------------------------------------

def _d1_d2(s, k, t, r, sigma, q):
    import torch
    sqrt_t = torch.sqrt(torch.clamp(t, min=1e-32))
    vol_term = torch.clamp(sigma * sqrt_t, min=1e-32)
    d1 = (torch.log(torch.clamp(s, min=1e-32) / torch.clamp(k, min=1e-32))
          + (r - q + 0.5 * sigma ** 2) * t) / vol_term
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def _bsm_price_t(flag, s, k, t, r, sigma, q):
    import torch
    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    discounted_spot = s * torch.exp(-q * t)
    discounted_strike = k * torch.exp(-r * t)
    call = discounted_spot * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
    put = discounted_strike * _normal_cdf(-d2) - discounted_spot * _normal_cdf(-d1)
    is_call = (flag == "c")
    if isinstance(is_call, np.ndarray):
        import torch
        is_call = torch.as_tensor(is_call, device=s.device)
    return torch.where(is_call, call, put)


def _vega_raw_t(s, k, t, r, sigma, q):
    import torch
    d1, _ = _d1_d2(s, k, t, r, sigma, q)
    return s * torch.exp(-q * t) * _normal_pdf(d1) * torch.sqrt(torch.clamp(t, min=1e-32))


def price_black(flag: np.ndarray, f: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    import torch
    dev = _device()
    ft = _to_tensor(f, dev)
    kt = _to_tensor(k, dev)
    tt = _to_tensor(t, dev)
    rt = _to_tensor(r, dev)
    st = _to_tensor(sigma, dev)
    qt = torch.zeros_like(rt)
    out = _bsm_price_t(flag, ft, kt, tt, rt, st, qt)
    return out.cpu().numpy()


def price_black_scholes(flag: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    import torch
    dev = _device()
    st = _to_tensor(s, dev)
    kt = _to_tensor(k, dev)
    tt = _to_tensor(t, dev)
    rt = _to_tensor(r, dev)
    sigt = _to_tensor(sigma, dev)
    qt = torch.zeros_like(rt)
    out = _bsm_price_t(flag, st, kt, tt, rt, sigt, qt)
    return out.cpu().numpy()


def price_black_scholes_merton(flag: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray, q: np.ndarray) -> np.ndarray:
    dev = _device()
    st = _to_tensor(s, dev)
    kt = _to_tensor(k, dev)
    tt = _to_tensor(t, dev)
    rt = _to_tensor(r, dev)
    sigt = _to_tensor(sigma, dev)
    qt = _to_tensor(q, dev)
    out = _bsm_price_t(flag, st, kt, tt, rt, sigt, qt)
    return out.cpu().numpy()


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def greeks(model: ModelLiteral, flag: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, sigma: np.ndarray, q: np.ndarray | None = None) -> dict[str, np.ndarray]:
    import torch
    dev = _device()
    st = _to_tensor(s, dev)
    kt = _to_tensor(k, dev)
    tt = _to_tensor(t, dev)
    rt = _to_tensor(r, dev)
    sigt = _to_tensor(sigma, dev)
    qv = torch.zeros_like(rt) if q is None else _to_tensor(q, dev)

    d1, d2 = _d1_d2(st, kt, tt, rt, sigt, qv)
    carry = torch.exp(-qv * tt)
    disc = torch.exp(-rt * tt)
    pdf = _normal_pdf(d1)
    sqrt_t = torch.sqrt(torch.clamp(tt, min=1e-32))
    safe_s = torch.clamp(st, min=1e-32)
    safe_sig = torch.clamp(sigt, min=1e-32)

    is_call = torch.as_tensor(flag == "c", device=dev)

    delta = torch.where(is_call,
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
    rho = torch.where(is_call, rho_call, rho_put)
    theta = torch.where(is_call, theta_call, theta_put)

    if model == "black":
        delta = disc * torch.where(is_call, _normal_cdf(d1), _normal_cdf(d1) - 1.0)
        gamma = disc * pdf / (safe_s * safe_sig * sqrt_t)

    return {name: arr.cpu().numpy() for name, arr in {
        "delta": delta, "gamma": gamma, "theta": theta, "rho": rho, "vega": vega,
    }.items()}


# ---------------------------------------------------------------------------
# Implied volatility (vectorized Newton-Raphson on GPU tensors)
# ---------------------------------------------------------------------------

_HALLEY_ITERS = 8
_BISECT_ITERS = 50
_IV_LO = 1e-8
_IV_HI = 10.0


def _price_vega_d1d2_t(model, flag, s, k, t, r, sigma, q):
    """Return (price, raw_vega, d1, d2) in a single pass."""
    import torch
    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    discounted_spot = s * torch.exp(-q * t)
    discounted_strike = k * torch.exp(-r * t)
    sqrt_t = torch.sqrt(torch.clamp(t, min=1e-32))
    call = discounted_spot * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
    put = discounted_strike * _normal_cdf(-d2) - discounted_spot * _normal_cdf(-d1)
    is_call = torch.as_tensor(flag == "c", device=s.device)
    price = torch.where(is_call, call, put)
    vega = discounted_spot * _normal_pdf(d1) * sqrt_t
    return price, vega, d1, d2


def _price_for_model_t(model, flag, s, k, t, r, sigma, q):
    import torch
    qt = q if q is not None else torch.zeros_like(r)
    px, _, _, _ = _price_vega_d1d2_t(model, flag, s, k, t, r, sigma, qt)
    return px


def _initial_guess_t(price, s, t):
    import torch
    sqrt_t = torch.sqrt(torch.clamp(t, min=1e-8))
    approx = price / torch.clamp(s * sqrt_t, min=1e-12) * _SQRT2PI
    return torch.clamp(approx, 0.30, 5.0)


def implied_volatility(model: ModelLiteral, price: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: np.ndarray, flag: np.ndarray, q: np.ndarray | None = None, on_error: str = "warn") -> np.ndarray:
    import torch
    from ..utils.validation import handle_error

    dev = _device()
    pt = _to_tensor(price, dev)
    st = _to_tensor(s, dev)
    kt = _to_tensor(k, dev)
    tt = _to_tensor(t, dev)
    rt = _to_tensor(r, dev)
    qv = torch.zeros_like(rt) if q is None else _to_tensor(q, dev)

    valid = tt > 0
    sigma = _initial_guess_t(pt, st, tt)

    # Halley's method (3rd order)
    for _ in range(_HALLEY_ITERS):
        px, vega, d1, d2 = _price_vega_d1d2_t(model, flag, st, kt, tt, rt, sigma, qv)
        residual = px - pt
        safe_vega = torch.where(vega > 1e-14, vega, torch.full_like(vega, float("inf")))
        newton_step = residual / safe_vega
        safe_sigma = torch.where(sigma > 1e-8, sigma, torch.full_like(sigma, float("inf")))
        halley_denom = 1.0 - newton_step * d1 * d2 / safe_sigma
        halley_denom = torch.where(halley_denom.abs() > 0.05, halley_denom, torch.sign(halley_denom + 1e-15) * 0.05)
        sigma = torch.clamp(sigma - newton_step / halley_denom, _IV_LO, _IV_HI)

    px_final = _price_for_model_t(model, flag, st, kt, tt, rt, sigma, qv)
    underflow_stuck = (px_final == 0.0) & (pt > 0.0)
    not_converged = (torch.abs(px_final - pt) > 1e-6) | underflow_stuck

    # Bisection fallback
    if not_converged.any():
        sigma_lo = torch.where(not_converged, torch.full_like(sigma, _IV_LO), sigma)
        sigma_hi = torch.where(not_converged, torch.full_like(sigma, _IV_HI), sigma)
        for _ in range(_BISECT_ITERS):
            sigma_mid = 0.5 * (sigma_lo + sigma_hi)
            px_mid = _price_for_model_t(model, flag, st, kt, tt, rt, sigma_mid, qv)
            mid_res = px_mid - pt
            sigma_lo = torch.where(not_converged & (mid_res < 0), sigma_mid, sigma_lo)
            sigma_hi = torch.where(not_converged & (mid_res >= 0), sigma_mid, sigma_hi)
        sigma = torch.where(not_converged, 0.5 * (sigma_lo + sigma_hi), sigma)

    result = torch.where(valid, sigma, torch.zeros_like(sigma))
    return result.cpu().numpy()
