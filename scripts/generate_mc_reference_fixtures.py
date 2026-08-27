"""Freeze high-precision reference values for the Monte Carlo oracles.

The oracles in ``tests/simulation/test_monte_carlo.py`` are independent
closed-form implementations written beside the tests. That protects against the
library being wrong, and not against the *oracle* being wrong: an edit could
change a formula and its test together and nothing would notice.

So the reference values are also frozen here, computed a second time through a
different route -- ``mpmath`` at fifty decimal digits, with its own error
function and exponential rather than ``scipy.special.ndtr`` and ``math.exp`` --
and checked in. A test compares the two. Agreement to double precision between
two implementations at different precisions, written from the same published
formula but not sharing a line of code, is a meaningful check on both.

No external pricing library is involved. QuantLib and its neighbours are not
dependencies of this project at any tier, and a fixture that could not be
regenerated from what is checked in would not be reproducible.

Run after changing a convention or adding a case:

    uv run python scripts/generate_mc_reference_fixtures.py

The output is deterministic, so a stale fixture shows up as a diff.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import mpmath as mp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "monte_carlo"

#: Fifty digits: far enough above float64 that the fixture's own rounding is
#: irrelevant to a comparison at 1e-12.
mp.mp.dps = 50

#: Every case shares this market and process, under an explicitly risk-neutral
#: GBM. The engine never makes a process risk-neutral; these values are only
#: meaningful against one that is.
MARKET = {
    "spot": "100.0",
    "rate": "0.02",
    "dividend_yield": "0.0",
    "volatility": "0.25",
    "maturity": "1.0",
    "measure": "risk-neutral GBM: drift = rate - dividend_yield",
    "log_drift": "-0.01125",
    "log_drift_note": (
        "rate - volatility**2 / 2, deliberately non-zero: at volatility 0.2 it "
        "vanishes, which removes the drift term from the geometric-Asian and "
        "variance formulas and makes an at-the-money digital call and put equal."
    ),
}


def _norm_cdf(x: mp.mpf) -> mp.mpf:
    """The standard normal CDF from mpmath's error function.

    Deliberately not ``scipy.special.ndtr``: an oracle that shares an
    implementation with the thing it checks is not an oracle.
    """
    return (1 + mp.erf(x / mp.sqrt(2))) / 2


def _black_scholes(option_type: str, spot, strike, rate, volatility, maturity):
    spread = volatility * mp.sqrt(maturity)
    d1 = (mp.log(spot / strike) + (rate + volatility**2 / 2) * maturity) / spread
    d2 = d1 - spread
    discounted_strike = strike * mp.exp(-rate * maturity)
    if option_type == "call":
        return spot * _norm_cdf(d1) - discounted_strike * _norm_cdf(d2)
    return discounted_strike * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def _cash_or_nothing(option_type, spot, strike, rate, volatility, maturity, cash_amount):
    spread = volatility * mp.sqrt(maturity)
    d2 = (mp.log(spot / strike) + (rate - volatility**2 / 2) * maturity) / spread
    sign = 1 if option_type == "call" else -1
    return cash_amount * mp.exp(-rate * maturity) * _norm_cdf(sign * d2)


def _discrete_geometric_asian(option_type, spot, strike, rate, volatility, maturity, n_steps):
    """The geometric average of lognormal fixings is itself lognormal.

    Fixings at ``t_1 .. t_n`` exclude the valuation date, matching the
    contract. ``log G`` is normal with mean
    ``log S0 + (r - sigma^2/2) * mean(t_i)`` and variance
    ``sigma^2 * sum_ij min(t_i, t_j) / n^2``.
    """
    times = [maturity * mp.mpf(i) / n_steps for i in range(1, n_steps + 1)]
    mean_time = sum(times) / n_steps
    covariance = sum(min(a, b) for a in times for b in times)
    mean_log = mp.log(spot) + (rate - volatility**2 / 2) * mean_time
    variance = volatility**2 * covariance / n_steps**2
    spread = mp.sqrt(variance)
    d2 = (mean_log - mp.log(strike)) / spread
    d1 = d2 + spread
    forward = mp.exp(mean_log + variance / 2)
    discount = mp.exp(-rate * maturity)
    if option_type == "call":
        return discount * (forward * _norm_cdf(d1) - strike * _norm_cdf(d2))
    return discount * (strike * _norm_cdf(-d2) - forward * _norm_cdf(-d1))


def _expected_realized_variance(rate, volatility, maturity, n_steps):
    """``E[RV] = sigma^2 + a^2 * sum(dt_i^2) / T`` with ``a = mu - sigma^2/2``.

    The expectation for the schedule actually monitored, not its continuous
    limit: each step's log return is ``a dt + sigma sqrt(dt) Z``, whose second
    moment is ``a^2 dt^2 + sigma^2 dt``.
    """
    step = maturity / n_steps
    adjusted = rate - volatility**2 / 2
    return volatility**2 + adjusted**2 * (n_steps * step**2) / maturity


def _forward(spot, strike, rate, maturity):
    return mp.exp(-rate * maturity) * (spot * mp.exp(rate * maturity) - strike)


def build_cases() -> list[dict[str, object]]:
    spot = mp.mpf(MARKET["spot"])
    rate = mp.mpf(MARKET["rate"])
    volatility = mp.mpf(MARKET["volatility"])
    maturity = mp.mpf(MARKET["maturity"])
    cases: list[dict[str, object]] = []

    for option_type in ("call", "put"):
        for strike in ("90.0", "100.0", "110.0"):
            cases.append(
                {
                    "instrument": "european_option",
                    "option_type": option_type,
                    "strike": strike,
                    "value": _black_scholes(
                        option_type, spot, mp.mpf(strike), rate, volatility, maturity
                    ),
                }
            )
        for strike in ("95.0", "100.0", "105.0"):
            cases.append(
                {
                    "instrument": "binary_option",
                    "option_type": option_type,
                    "strike": strike,
                    "cash_amount": "1.0",
                    "value": _cash_or_nothing(
                        option_type, spot, mp.mpf(strike), rate, volatility, maturity, mp.mpf(1)
                    ),
                }
            )
        for n_steps in (4, 8, 32):
            cases.append(
                {
                    "instrument": "asian_option",
                    "averaging_method": "geometric",
                    "strike_convention": "fixed",
                    "option_type": option_type,
                    "strike": "100.0",
                    "n_steps": n_steps,
                    "fixings": "S_1..S_n on an even grid; the valuation date is excluded",
                    "value": _discrete_geometric_asian(
                        option_type, spot, mp.mpf(100), rate, volatility, maturity, n_steps
                    ),
                }
            )

    for delivery_price in ("90.0", "100.0", "110.0"):
        cases.append(
            {
                "instrument": "forward",
                "delivery_price": delivery_price,
                "value": _forward(spot, mp.mpf(delivery_price), rate, maturity),
            }
        )

    for n_steps in (4, 16, 64):
        expectation = _expected_realized_variance(rate, volatility, maturity, n_steps)
        cases.append(
            {
                "instrument": "variance_swap",
                "strike_variance": "0.0",
                "n_steps": n_steps,
                "convention": "sum(log(S_i/S_{i-1})**2) / T; no mean adjustment, no 252",
                "value": mp.exp(-rate * maturity) * expectation,
            }
        )

    for case in cases:
        case["value"] = mp.nstr(case["value"], 30, strip_zeros=False)
    return cases


def render() -> str:
    cases = build_cases()
    body = {
        "description": (
            "Discounted reference values for the Monte Carlo oracles, computed with "
            "mpmath at 50 decimal digits from published closed forms. Not produced by "
            "any pricing library, and regenerable from the checked-in generator."
        ),
        "generator": "scripts/generate_mc_reference_fixtures.py",
        "precision_digits": mp.mp.dps,
        "library": f"mpmath {mp.__version__}",
        "market": MARKET,
        "conventions": {
            "discounting": "exp(-rate * maturity)",
            "asian_fixings": "S_1..S_n, evenly spaced, valuation date excluded",
            "binary": "cash-or-nothing, strict inequality at the strike",
            "variance": "sum of squared log returns divided by the year fraction",
        },
        "cases": cases,
    }
    text = json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    body["content_sha256"] = digest
    return json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False) + "\n"


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / "gbm_closed_forms.json"
    text = render()
    previous = path.read_text(encoding="utf-8") if path.exists() else None
    if previous == text:
        sys.stdout.write(f"{path} is up to date.\n")
        return 0
    path.write_text(text, encoding="utf-8")
    sys.stdout.write(f"Wrote {path}.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
