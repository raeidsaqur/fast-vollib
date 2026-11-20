# Author: Raeid Saqur
# Date: 2026-11-19
"""Independent high-precision Black-76 evaluator and bracketed implied-volatility inverse.

This is the adjudication reference for the implied-volatility validation suites. It
shares no code with ``fast_vollib``: prices are evaluated with ``mpmath`` from the
*exact* binary values of the inputs, and the inverse is a sign-checked bisection on
``log(sigma)`` whose precision is raised until the bracket and the classification
stop changing.

Model. All three public models reduce to a discounted Black-76 price::

    P = D * [ F * Phi(d1) - K * Phi(d2) ]          (call)
    P = D * [ K * Phi(-d2) - F * Phi(-d1) ]        (put)
    d1 = ln(F/K)/v + v/2,  d2 = d1 - v,  v = sigma * sqrt(T)

with ``D = exp(-r T)`` and, for Black-Scholes(-Merton), ``F = S exp((r - q) T)``.

For a quote ``P`` the script reports

* a **class**: ``invalid_input``, ``expiry`` (T = 0), ``below_lower_bound``,
  ``lower_bound`` (P equals the no-arbitrage lower bound exactly, sigma = 0),
  ``interior``, ``upper_bound`` (sigma = inf), ``above_upper_bound``, or
  ``unresolved`` (no stable bracket within the precision budget);
* for interior quotes, the **point inverse** of the exact quote (bracket endpoints and
  the float64 value of its midpoint) and the working precision that stabilised it;
* the **rounding interval**: the exact set of real prices that round to the quote in
  the tested dtype (round-half-even midpoints to the adjacent representable values),
  intersected with the admissible price range, and its image in sigma. An endpoint
  that touches the lower or upper price bound maps to sigma = 0 or sigma = inf.

The point inverse tests a solver; the interval measures the implied-volatility
ambiguity caused by representing the price in floating point.

Usage::

    python scripts/reference_inverse.py point --F 100 --K 90.48374180359595 --T 1 \
        --price 9.516258... --flag c
    python scripts/reference_inverse.py csv rows.csv out.csv [--dtype float64] [--jobs N]

The CSV mode reads columns ``flag``, ``K_hex``, ``T_hex``, ``price_hex``, either ``F_hex``
(forward form) or ``S_hex`` with optional ``q_hex`` (spot form), and optionally ``r_hex``
(default 0) and ``sigma_gen_hex`` (to compare the generating quote with the independent
evaluator rounded to the tested dtype). Hex strings use
``float.hex`` format.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass, field
import json
import math
import sys
from typing import Any

import mpmath
import numpy as np

SCHEMA = "fv-iv-reference/1"
# The reference values are tied to this exact mpmath release, which the dev lock pins
# (it enters the dev group through sympy). Refuse to run on anything else.
MPMATH_VERSION = "1.3.0"
if mpmath.__version__ != MPMATH_VERSION:  # pragma: no cover - environment guard
    raise ImportError(
        f"reference_inverse requires mpmath=={MPMATH_VERSION}; found {mpmath.__version__}"
    )
BASE_DPS = 40
MAX_DPS = 1600
TARGET_REL_WIDTH = mpmath.mpf(2) ** -96  # relative bracket width on sigma
SIGMA_SEARCH_MIN = mpmath.mpf(10) ** -300
SIGMA_SEARCH_MAX = mpmath.mpf(10) ** 6


# ---------------------------------------------------------------------------
# Exact inputs
# ---------------------------------------------------------------------------


def exact(value: float) -> mpmath.mpf:
    """The exact real value of a binary floating-point number (no rounding)."""
    if not math.isfinite(value):
        raise ValueError("non-finite input")
    m, e = math.frexp(value)
    return mpmath.ldexp(mpmath.mpf(int(m * 2**53)), e - 53)


@dataclass(frozen=True)
class Contract:
    """Exact binary inputs of one contract.

    Forward form (Black-76): give ``F``. Spot form (Black-Scholes / Black-Scholes-
    Merton): give ``spot`` (and ``q``); the forward ``S * exp((r - q) T)`` is then
    evaluated in high precision from the exact inputs and ``F`` is ignored.
    """

    F: float
    K: float
    T: float
    is_call: bool
    r: float = 0.0
    spot: float | None = None
    q: float = 0.0

    def underlying(self) -> float:
        return self.F if self.spot is None else self.spot

    def mp_inputs(self) -> tuple[mpmath.mpf, mpmath.mpf, mpmath.mpf, mpmath.mpf]:
        K, T, r = exact(self.K), exact(self.T), exact(self.r)
        if self.spot is None:
            F = exact(self.F)
        else:
            F = exact(self.spot) * mpmath.exp((r - exact(self.q)) * T)
        D = mpmath.exp(-r * T)
        return F, K, T, D


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def _phi_cdf(z: mpmath.mpf) -> mpmath.mpf:
    return mpmath.erfc(-z / mpmath.sqrt(2)) / 2


def black_price(c: Contract, sigma: mpmath.mpf) -> mpmath.mpf:
    """Discounted Black-76 price at the current mpmath working precision."""
    F, K, T, D = c.mp_inputs()
    v = sigma * mpmath.sqrt(T)
    if v == 0:
        intrinsic = F - K if c.is_call else K - F
        return D * max(intrinsic, mpmath.mpf(0))
    if mpmath.isinf(v):
        return D * (F if c.is_call else K)
    d1 = mpmath.log(F / K) / v + v / 2
    d2 = d1 - v
    if c.is_call:
        return D * (F * _phi_cdf(d1) - K * _phi_cdf(d2))
    return D * (K * _phi_cdf(-d2) - F * _phi_cdf(-d1))


def price_bounds(c: Contract) -> tuple[mpmath.mpf, mpmath.mpf]:
    F, K, _, D = c.mp_inputs()
    if c.is_call:
        return D * max(F - K, mpmath.mpf(0)), D * F
    return D * max(K - F, mpmath.mpf(0)), D * K


# ---------------------------------------------------------------------------
# Inverse
# ---------------------------------------------------------------------------


@dataclass
class Inverse:
    status: str  # "ok" | "unresolved"
    lo: mpmath.mpf | None = None
    hi: mpmath.mpf | None = None
    dps: int = 0


def _bisect(c: Contract, target: mpmath.mpf, lo: mpmath.mpf, hi: mpmath.mpf) -> Inverse:
    """Sign-checked bisection on log(sigma) at the current precision."""
    f_lo = black_price(c, lo) - target
    f_hi = black_price(c, hi) - target
    while f_lo >= 0:
        if lo <= SIGMA_SEARCH_MIN:
            return Inverse("unresolved")
        lo /= 16
        f_lo = black_price(c, lo) - target
    while f_hi <= 0:
        if hi >= SIGMA_SEARCH_MAX:
            return Inverse("unresolved")
        hi *= 4
        f_hi = black_price(c, hi) - target
    while hi / lo - 1 > TARGET_REL_WIDTH:
        mid = mpmath.sqrt(lo * hi)
        f_mid = black_price(c, mid) - target
        if f_mid < 0:
            lo = mid
        elif f_mid > 0:
            hi = mid
        else:
            return Inverse("ok", mid, mid, mpmath.mp.dps)
    return Inverse("ok", lo, hi, mpmath.mp.dps)


def _start_dps(target: mpmath.mpf, lower: mpmath.mpf) -> int:
    """Extra digits for the cancellation between the quote and its lower bound."""
    if lower <= 0 or target <= lower:
        return BASE_DPS
    ratio = mpmath.log10(target / (target - lower))
    return BASE_DPS + int(mpmath.ceil(ratio))


def invert(c: Contract, target: mpmath.mpf) -> Inverse:
    """Bracketed inverse, with precision doubled until the float64 midpoint is stable."""
    with mpmath.workdps(BASE_DPS + 40):
        lower, _ = price_bounds(c)
        dps = _start_dps(target, lower)
    previous: Inverse | None = None
    lo, hi = mpmath.mpf("0.01"), mpmath.mpf(1)
    while dps <= MAX_DPS:
        with mpmath.workdps(dps):
            if previous is not None and previous.status == "ok":
                widen = mpmath.mpf(2) ** -60
                lo = previous.lo * (1 - widen)
                hi = previous.hi * (1 + widen)
            current = _bisect(c, mpmath.mpf(target), lo, hi)
        if previous is not None and current.status == previous.status == "ok":
            if _f64_mid(current) == _f64_mid(previous):
                current.dps = dps
                return current
        if previous is not None and current.status == previous.status == "unresolved":
            return current
        previous = current
        dps *= 2
    return Inverse("unresolved", dps=dps)


def _f64_mid(inv: Inverse) -> float:
    assert inv.lo is not None and inv.hi is not None
    with mpmath.workdps(60):
        return nearest(mpmath.sqrt(inv.lo * inv.hi))


# ---------------------------------------------------------------------------
# Rounding interval
# ---------------------------------------------------------------------------


def rounding_interval(quote: float, dtype: str) -> tuple[mpmath.mpf, mpmath.mpf, bool]:
    """Exact real interval of prices that round to ``quote`` in ``dtype``.

    Returns (lower, upper, closed). Under round-half-even, both midpoints belong to
    the interval exactly when the quote's significand is even.
    """
    np_t: Any = {"float64": np.float64, "float32": np.float32}[dtype]
    q = np_t(quote)
    below = np.nextafter(q, np_t(-np.inf))
    above = np.nextafter(q, np_t(np.inf))
    with mpmath.workdps(BASE_DPS + 40):  # midpoints need one bit more than the dtype
        eq, eb, ea = exact(float(q)), exact(float(below)), exact(float(above))
        lo = (eb + eq) / 2
        hi = (eq + ea) / 2
    bits = q.view(np.uint64 if dtype == "float64" else np.uint32)
    closed = int(bits) % 2 == 0
    return lo, hi, closed


# ---------------------------------------------------------------------------
# Row adjudication
# ---------------------------------------------------------------------------


@dataclass
class Result:
    schema: str = SCHEMA
    cls: str = ""
    ref_f64: float | None = None
    ref_lo_hex: str | None = None
    ref_hi_hex: str | None = None
    ref_dps: int = 0
    interval_dtype: str = "float64"
    interval_closed: bool | None = None
    sigma_int_lo: float | None = None  # 0.0 means the interval reaches the lower price bound
    sigma_int_hi: float | None = None  # inf means the interval reaches the upper price bound
    sigma_int_lo_status: str = ""
    sigma_int_hi_status: str = ""
    interval_empty: bool | None = None
    tv_over_quote_ulp: float | None = None
    beta: float | None = None  # normalised price P / (D sqrt(F K))
    h_quote: float | None = None  # max distance from ref to the quote's sigma interval endpoints
    h_forward: float | None = None  # sigma shift from one rounding of a derived forward
    h_discount: float | None = None  # sigma shift from one rounding of the discount factor
    generator_quote_matches: bool | None = None
    generator_quote_ulps: float | None = None
    notes: list[str] = field(default_factory=list)


def nearest(v: mpmath.mpf, dtype: str = "float64") -> float:
    """Correctly rounded (to nearest, ties to even) conversion, including subnormals."""
    np_t: Any = {"float64": np.float64, "float32": np.float32}[dtype]
    with mpmath.workdps(max(mpmath.mp.dps, 40)):
        guess = np_t(float(v))
        cands = [np.nextafter(guess, np_t(-np.inf)), guess, np.nextafter(guess, np_t(np.inf))]
        cands = [c for c in cands if np.isfinite(c)]
        errs = [abs(exact(float(c)) - v) for c in cands]
        best = min(errs)
        ties = [c for c, e in zip(cands, errs) if e == best]
        if len(ties) > 1:
            uint = np.uint64 if dtype == "float64" else np.uint32
            ties = [c for c in ties if int(np_t(c).view(uint)) % 2 == 0] or ties
        return float(ties[0])


def _down(v: mpmath.mpf) -> float:
    """float64 at or just below v (outward rounding of a lower endpoint)."""
    f = nearest(v)
    return f if exact(f) <= v else float(np.nextafter(f, -np.inf))


def _up(v: mpmath.mpf) -> float:
    """float64 at or just above v (outward rounding of an upper endpoint)."""
    f = nearest(v)
    return f if exact(f) >= v else float(np.nextafter(f, np.inf))


def adjudicate(
    c: Contract, quote: float, dtype: str = "float64", sigma_gen: float | None = None
) -> Result:
    res = Result(interval_dtype=dtype)
    und = c.underlying()
    vals = (und, c.K, c.T, c.r, c.q, quote)
    if not all(math.isfinite(v) for v in vals) or und <= 0 or c.K <= 0 or c.T < 0 or quote < 0:
        res.cls = "invalid_input"
        return res
    if c.T == 0:
        res.cls = "expiry"
        return res
    q = exact(float(np.float32(quote)) if dtype == "float32" else quote)
    with mpmath.workdps(BASE_DPS + 40):
        lower, upper = price_bounds(c)
        F, K, _, D = c.mp_inputs()
        res.beta = nearest(q / (D * mpmath.sqrt(F * K)))
        np_t = np.float32 if dtype == "float32" else np.float64
        spacing = abs(exact(float(np.spacing(np_t(quote))))) if quote != 0 else None
        if spacing:
            res.tv_over_quote_ulp = float((q - lower) / spacing)
        if sigma_gen is not None:
            gen = exact(sigma_gen)
            with mpmath.workdps(BASE_DPS + 200):
                p_gen = black_price(c, gen)
            p_gen_dt = nearest(p_gen, dtype)
            res.generator_quote_matches = p_gen_dt == float(quote)
            res.generator_quote_ulps = (
                float((exact(float(quote)) - exact(p_gen_dt)) / spacing) if spacing else None
            )
    if q < lower:
        res.cls = "below_lower_bound"
    elif q == lower:
        res.cls = "lower_bound"
    elif q > upper:
        res.cls = "above_upper_bound"
    elif q == upper:
        res.cls = "upper_bound"
    else:
        inv = invert(c, q)
        if inv.status != "ok":
            res.cls = "unresolved"
        else:
            res.cls = "interior"
            assert inv.lo is not None and inv.hi is not None
            with mpmath.workdps(inv.dps):
                res.ref_f64 = nearest(mpmath.sqrt(inv.lo * inv.hi))
                res.ref_lo_hex = _down(inv.lo).hex()
                res.ref_hi_hex = _up(inv.hi).hex()
            res.ref_dps = inv.dps

    # Rounding interval in price, intersected with the admissible range.
    p_lo, p_hi, closed = rounding_interval(quote, dtype)
    res.interval_closed = closed
    if p_hi <= lower or p_lo >= upper or (not closed and (p_hi == lower or p_lo == upper)):
        res.interval_empty = True
        return res
    res.interval_empty = False
    if p_lo <= lower:
        res.sigma_int_lo, res.sigma_int_lo_status = 0.0, "touches_lower_bound"
    else:
        inv_lo = invert(c, p_lo)
        if inv_lo.status == "ok" and inv_lo.lo is not None:
            res.sigma_int_lo, res.sigma_int_lo_status = _down(inv_lo.lo), "ok"
        else:
            res.sigma_int_lo_status = "unresolved"
    if p_hi >= upper:
        res.sigma_int_hi, res.sigma_int_hi_status = math.inf, "touches_upper_bound"
    else:
        inv_hi = invert(c, p_hi)
        if inv_hi.status == "ok" and inv_hi.hi is not None:
            res.sigma_int_hi, res.sigma_int_hi_status = _up(inv_hi.hi), "ok"
        else:
            res.sigma_int_hi_status = "unresolved"
    if res.ref_f64 is not None:
        _sensitivities(c, quote, res)
    return res


def _sensitivities(c: Contract, quote: float, res: Result) -> None:
    """Half-widths in sigma attributable to unavoidable input roundings.

    ``h_quote`` comes from the quote's rounding interval. ``h_forward`` is the sigma
    shift caused by one correctly rounded evaluation of a forward that must be derived
    (spot form: F = S exp((r - q) T)); it is zero when F is itself a given input.
    ``h_discount`` is the shift caused by one rounding of D = exp(-r T); zero when r = 0.
    """
    ref = res.ref_f64
    assert ref is not None
    lo, hi = res.sigma_int_lo, res.sigma_int_hi
    finite = [abs(ref - v) for v in (lo, hi) if v is not None and math.isfinite(v) and v > 0]
    res.h_quote = max(finite) if finite else None
    with mpmath.workdps(BASE_DPS + 20):
        F, K, T, D = c.mp_inputs()
        sigma = exact(ref)
        v = sigma * mpmath.sqrt(T)
        d1 = mpmath.log(F / K) / v + v / 2
        vega = D * F * mpmath.npdf(d1) * mpmath.sqrt(T)
        if vega == 0:
            return
        u = mpmath.mpf(2) ** -53
        dp_df = D * (_phi_cdf(d1) if c.is_call else -_phi_cdf(-d1))
        res.h_forward = nearest(abs(dp_df) * F * u / vega) if c.spot is not None else 0.0
        res.h_discount = nearest(exact(quote) * u / vega) if c.r != 0 else 0.0


# ---------------------------------------------------------------------------
# Jäckel-zone diagnostics (independent of the library implementation)
# ---------------------------------------------------------------------------


def jaeckel_zone(F: float, K: float, quote_undiscounted: float, is_call: bool) -> str:
    """Which of the four "Let's Be Rational" initial-guess zones the quote falls in.

    Uses the normalised out-of-the-money price beta at x = -|ln(F/K)| and the zone
    boundaries b_l < b_c < b_h of Jäckel (2015), evaluated here in mpmath.
    """
    with mpmath.workdps(80):
        Fm, Km = exact(F), exact(K)
        x = mpmath.log(Fm / Km)
        root = mpmath.sqrt(Fm * Km)
        beta = exact(quote_undiscounted) / root
        intrinsic = (Fm - Km if is_call else Km - Fm) / root
        if intrinsic > 0:
            beta -= intrinsic  # in-the-money: reduce to the out-of-the-money price
        xn = -abs(x)
        if xn == 0:
            return "atm"
        b_max = mpmath.exp(xn / 2)

        def b(s: mpmath.mpf) -> mpmath.mpf:
            return mpmath.exp(xn / 2) * _phi_cdf(xn / s + s / 2) - mpmath.exp(-xn / 2) * _phi_cdf(
                xn / s - s / 2
            )

        def vega(s: mpmath.mpf) -> mpmath.mpf:
            return mpmath.npdf(xn / s + s / 2) * mpmath.exp(xn / 2)

        s_c = mpmath.sqrt(2 * abs(xn))
        b_c, v_c = b(s_c), vega(s_c)
        s_l = s_c - b_c / v_c
        b_l = b(s_l) if s_l > 0 else mpmath.mpf(0)
        s_h = s_c + (b_max - b_c) / v_c
        b_h = b(s_h)
        if beta < b_l:
            return "z1_lower"
        if beta < b_c:
            return "z2_lower_middle"
        if beta <= b_h:
            return "z3_upper_middle"
        return "z4_upper"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _row(args: tuple[dict[str, str], str]) -> dict[str, Any]:
    row, dtype = args
    fh = float.fromhex
    spot = fh(row["S_hex"]) if row.get("S_hex") else None
    c = Contract(
        F=fh(row["F_hex"]) if row.get("F_hex") else math.nan,
        K=fh(row["K_hex"]),
        T=fh(row["T_hex"]),
        is_call=row["flag"] == "c",
        r=fh(row["r_hex"]) if row.get("r_hex") else 0.0,
        spot=spot,
        q=fh(row["q_hex"]) if row.get("q_hex") else 0.0,
    )
    sigma_gen = fh(row["sigma_gen_hex"]) if row.get("sigma_gen_hex") else None
    try:
        res = adjudicate(c, fh(row["price_hex"]), dtype=dtype, sigma_gen=sigma_gen)
        out = asdict(res)
    except Exception as exc:  # recorded as a class, never swallowed silently
        out = asdict(Result(cls="reference_exception", notes=[f"{type(exc).__name__}: {exc}"]))
    out["notes"] = ";".join(out["notes"])
    for key in ("ref_f64", "sigma_int_lo", "sigma_int_hi", "h_quote", "h_forward", "h_discount"):
        if out[key] is not None:
            out[key + "_hex"] = float(out[key]).hex()
    return {**{k: row[k] for k in row if k in ("flag", "i", "id")}, **out}


def _iter_csv(path: str) -> Iterable[dict[str, str]]:
    with open(path, newline="") as fh:
        yield from csv.DictReader(fh)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("point")
    for name in ("F", "K", "T", "price"):
        p.add_argument(f"--{name}", type=float, required=True)
    p.add_argument("--r", type=float, default=0.0)
    p.add_argument("--flag", choices=("c", "p"), required=True)
    p.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    p.add_argument("--sigma-gen", type=float, default=None)
    c = sub.add_parser("csv")
    c.add_argument("inp")
    c.add_argument("out")
    c.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    c.add_argument("--jobs", type=int, default=1)
    ns = ap.parse_args(argv)
    if ns.cmd == "point":
        con = Contract(F=ns.F, K=ns.K, T=ns.T, is_call=ns.flag == "c", r=ns.r)
        res = asdict(adjudicate(con, ns.price, dtype=ns.dtype, sigma_gen=ns.sigma_gen))
        res["mpmath"] = mpmath.__version__
        sys.stdout.write(json.dumps(res, default=str, indent=1) + "\n")
        return 0
    rows = list(_iter_csv(ns.inp))
    work = [(row, ns.dtype) for row in rows]
    if ns.jobs > 1:
        with ProcessPoolExecutor(ns.jobs) as ex:
            results = list(ex.map(_row, work, chunksize=16))
    else:
        results = [_row(w) for w in work]
    keys: list[str] = []
    for r in results:
        keys.extend(k for k in r if k not in keys)
    with open(ns.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(results)
    sys.stdout.write(json.dumps({"rows": len(results), "mpmath": mpmath.__version__}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
