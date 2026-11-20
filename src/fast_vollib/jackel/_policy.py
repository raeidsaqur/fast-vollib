# Author: Raeid Saqur
# Date: 2026-11-19
"""Row policy shared by the Jäckel full-model wrappers (NumPy, PyTorch, JAX).

The wrappers use the same classification and ``on_error`` signal as the Halley backends
(``backends._iv_common``): bounds in double-double precision; T = 0 and quotes exactly at
intrinsic return 0.0; zero prices, prices below intrinsic or at/above the upper bound and invalid
inputs return NaN with one aggregated warning/error per call. Interior rows are handed to the
solver as an out-of-the-money call on the discounted forward/strike pair (Fo <= Ko) at the time
value; Black prices are homogeneous in (price, F, K), so this is the same problem in normalised
form, per row, with no host fallback.

Domain: the Jäckel solver works on the normalised price beta = tv / sqrt(Fo Ko). Rows with
beta below the smallest normal double are refused (NaN with the signal): there beta itself is
sub-normal and the solver's lower-branch objective has lost its precision. The Halley solver
accepts those rows.

Verification: every solved row is repriced on its better-conditioned side, the time value
c(s) = Fo N(d1) - Ko N(d2) or the distance to the upper bound u(s) = Fo N(-d1) + Ko N(d2)
(no cancellation), and must reproduce the quote's value to within the quote's own rounding
(half an ulp of the price) plus VERIFY_RTOL relative. Otherwise the
row is reported as not converged (NaN with the signal) rather than returned as a wrong finite
volatility. This happens close to the upper bound, where the solver's normalised price leaves
too little of b_max - beta, and for near-at-the-money rows with tiny beta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import ndtr

from ..backends import _iv_common as _ivc

#: Agreement required between the repriced value and the quote's value (see module doc): half an
#: ulp of the quoted price (the quote's own rounding interval) plus VERIFY_RTOL of the target.
#: Wider bands (one or two ulps) were measured to let wrong volatilities through and are not used.
VERIFY_RTOL = 1e-8


@dataclass
class Prepared:
    shape: tuple
    result: np.ndarray  # NaN, 0.0 on zero-volatility rows
    code: np.ndarray  # int8 row codes
    solve: np.ndarray  # flat indices of the rows to solve
    beta: np.ndarray  # normalised out-of-the-money price, per solved row
    x: np.ndarray  # -|ln(F/K)|, per solved row
    price: np.ndarray
    tv: np.ndarray
    gap: np.ndarray
    fo: np.ndarray
    ko: np.ndarray
    t: np.ndarray


def prepare(model: str, price: Any, s: Any, k: Any, t: Any, r: Any, flag: Any, q: Any) -> Prepared:
    shape, price, s, k, t, r, flag, q = _ivc.flatten_inputs(price, s, k, t, r, flag, q)
    qv = q if (q is not None and model == "black_scholes_merton") else np.zeros_like(r)
    rows = _ivc.classify_np(model, price, s, k, t, r, qv, flag == "c")
    code = rows["code"]
    result = np.full(price.shape, np.nan)
    result[rows["zero_vol"]] = 0.0
    idx = np.flatnonzero(rows["interior"])
    tv, gap, fo, ko, a = (rows[key][idx] for key in ("tv", "gap", "Fo", "Ko", "a"))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        beta = tv / (np.sqrt(fo) * np.sqrt(ko))
    outside = ~(beta >= _ivc.JAECKEL_BETA_MIN)
    code[idx[outside]] = _ivc.OUTSIDE_DOMAIN
    keep = ~outside
    return Prepared(
        shape,
        result,
        code,
        idx[keep],
        beta[keep],
        a[keep],
        price[idx][keep],
        tv[keep],
        gap[keep],
        fo[keep],
        ko[keep],
        t[idx][keep],
    )


def _reprices(prep: Prepared, sigma: np.ndarray) -> np.ndarray:
    """True where sigma reproduces the quote on its better-conditioned side."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        s = sigma * np.sqrt(prep.t)
        d1 = prep.x / s + 0.5 * s
        d2 = d1 - s
        upper = prep.gap < prep.tv
        model = np.where(
            upper,
            prep.fo * ndtr(-d1) + prep.ko * ndtr(d2),
            prep.fo * ndtr(d1) - prep.ko * ndtr(d2),
        )
        target = np.where(upper, prep.gap, prep.tv)
        band = 0.5 * np.spacing(prep.price) + VERIFY_RTOL * target
        return np.abs(model - target) <= band


def finish(prep: Prepared, sigma: np.ndarray, on_error: str) -> np.ndarray:
    """Store the solved volatilities; non-finite or non-positive ones are 'not converged'."""
    sigma = np.asarray(sigma, dtype=np.float64)
    good = np.isfinite(sigma) & (sigma > 0.0)
    good &= _reprices(prep, np.where(good, sigma, 1.0))
    prep.result[prep.solve] = np.where(good, sigma, np.nan)
    prep.code[prep.solve[~good]] = _ivc.NOT_CONVERGED
    _ivc.report_row_errors(_ivc.count_codes(prep.code), prep.code.size, on_error)
    return prep.result.reshape(prep.shape)


def solve_native(
    xp: Any,
    ndtr: Any,
    solver: Any,
    model: str,
    price: Any,
    s: Any,
    k: Any,
    t: Any,
    r: Any,
    q: Any,
    is_call: Any,
    zero_vol_value: float = 0.0,
) -> tuple[Any, Any, Any]:
    """The same row policy on native torch / JAX arrays (traceable; used by the autograd wrappers).

    ``solver(tv, fo, ko, t)`` returns annualised volatilities for an out-of-the-money call on
    (fo, ko) at time value tv. Returns ``(sigma, invalid, code)``: sigma is NaN on refused rows
    and ``zero_vol_value`` on zero-volatility rows (T = 0, quote exactly at intrinsic); ``invalid`` marks rows without a well-defined solution
    (for the implicit-function backward); ``code`` holds the row codes.
    """
    rows = _ivc.classify(xp, model, price, s, k, t, r, q, is_call)
    exact = _ivc.classify(xp, model, price, s, k, t, r, q, is_call, precise=True)
    near = rows["near"]
    rows = {key: xp.where(near, exact[key], value) for key, value in rows.items()}
    interior = rows["interior"]
    one = xp.ones_like(price)
    tv = xp.where(interior, rows["tv"], one)
    gap = xp.where(interior, rows["gap"], one)
    fo = xp.where(interior, rows["Fo"], one)
    ko = xp.where(interior, rows["Ko"], one)
    a = xp.where(interior, rows["a"], -one)
    tt = xp.where(interior, t, one)
    beta = tv / (xp.sqrt(fo) * xp.sqrt(ko))
    outside = interior & ~(beta >= _ivc.JAECKEL_BETA_MIN)
    solve = interior & ~outside
    sigma = solver(tv, fo, ko, tt)
    finite = xp.isfinite(sigma) & (sigma > 0.0)
    sig_safe = xp.where(finite, sigma, one)
    sv = sig_safe * xp.sqrt(tt)
    d1 = a / sv + 0.5 * sv
    d2 = d1 - sv
    upper = gap < tv
    model_value = xp.where(upper, fo * ndtr(-d1) + ko * ndtr(d2), fo * ndtr(d1) - ko * ndtr(d2))
    target = xp.where(upper, gap, tv)
    spacing = xp.nextafter(price, xp.full_like(price, float("inf"))) - price
    band = 0.5 * spacing + VERIFY_RTOL * target
    good = solve & finite & (xp.abs(model_value - target) <= band)
    code = rows["code"]
    code = xp.where(outside, _ivc.OUTSIDE_DOMAIN, code)
    code = xp.where(solve & ~good, _ivc.NOT_CONVERGED, code)
    out = xp.where(good, sigma, xp.full_like(price, float("nan")))
    out = xp.where(rows["zero_vol"], xp.full_like(price, zero_vol_value), out)
    return out, ~good, code


def report_codes(code: Any, on_error: str) -> None:
    """Aggregated on_error signal from a (host-convertible) array of row codes."""
    values = np.asarray(code)
    _ivc.report_row_errors(_ivc.count_codes(values), values.size, on_error)
