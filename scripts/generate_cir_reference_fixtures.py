"""Freeze fifty-digit CIR bond prices, computed from the paper, not from the library.

An oracle, in the sense ``generate_mc_reference_fixtures.py`` is one and
``generate_heston_reference_fixtures.py`` deliberately is not: every value here
is computed in ``mpmath`` at fifty significant digits **directly from Cox,
Ingersoll and Ross (1985) equation 23 as published**, with no rearrangement --
no ``delta``, no ``log1p``, no factoring of ``exp(gamma*tau)``.  At fifty digits
the literal form is perfectly well conditioned, so it needs none of the repairs
float64 does, and that is exactly what makes it an independent check on
:func:`fast_vollib.rates.cir_discount_factor`, which shares not one line with it.

The parameter sets are chosen to make the check adversarial rather than
comfortable:

*The volatility sweep* runs from 0.5 down to 1e-10 at the book parameters. This
is where the naive float64 forms fail: computing ``gamma - kappa`` by
subtraction loses every significant digit as ``sigma -> 0``, and the resulting
price is wrong in the third decimal place by ``sigma = 1e-8``. The shipped form
never forms that difference.

*The regime table* spans ``gamma*tau`` from 0.3 to about 3600, which is well
past the ``exp(gamma*tau)`` overflow at 709 that the published form hits
directly, and includes 200- and 500-year maturities where ``A(tau)`` underflows
to zero long before ``log A`` stops being an ordinary number.

*The stress rows* cover overflow and cancellation explicitly:
``volatility = 1e-10`` and ``gamma*tau = 2500``.

*The deterministic rows* have ``volatility`` exactly zero, where
``2*kappa*theta/sigma**2`` is not a number and the kernel takes its one special
branch. The reference uses the closed-form integral of the noiseless mean-
reverting path, which is a different formula rather than a limit of the same
one, so the branch is checked against mathematics instead of against itself.

Run after changing the kernel or adding a case:

    uv run python scripts/generate_cir_reference_fixtures.py

The output is deterministic, so a stale fixture shows up as a diff.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import mpmath as mp
from reference_fixtures import write_or_check

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "rates"

#: Fifty digits: far enough above float64 that the reference's own rounding is
#: irrelevant to a comparison at 1e-15.
mp.mp.dps = 50

#: Representative CIR parameters for independent numerical checks.
BOOK = {"kappa": "0.3", "theta": "0.04", "initial_rate": "0.04", "maturity": "1.0"}

#: Where the naive float64 forms lose their digits.
VOLATILITY_SWEEP = (
    "0.5", "0.1", "0.01", "0.001", "0.0001",
    "0.00001", "0.000001", "0.0000001", "0.00000001", "0.0000000001",
)  # fmt: skip

#: (kappa, theta, volatility, initial_rate, maturity, why this row is here)
REGIMES = (
    ("0.3", "0.04", "0.1", "0.04", "1.0", "the reference one-year price"),
    ("0.3", "0.04", "0.1", "0.04", "10.0", "ordinary parameters, ten years"),
    ("2.0", "0.03", "0.5", "0.05", "5.0", "fast reversion, high vol-of-rate"),
    ("0.5", "0.02", "0.02", "0.02", "30.0", "small vol-of-rate over thirty years"),
    ("1.5", "0.05", "0.4", "0.01", "0.25", "short maturity, rate far below theta"),
    ("3.0", "0.04", "2.0", "0.04", "200.0", "gamma*tau ~ 825; the published form overflows"),
    ("1.0", "0.04", "5.0", "0.04", "500.0", "gamma*tau ~ 3571; log A ~ -100"),
    ("0.05", "0.04", "0.9", "0.04", "100.0", "very slow reversion"),
    ("0.3", "0.04", "0.000001", "0.09", "50.0", "tiny vol-of-rate, long maturity"),
    ("5.0", "0.04", "0.00000001", "0.04", "500.0", "gamma*tau = 2500 exactly"),
    ("0.3", "0.04", "0.0000000001", "0.04", "1.0", "volatility = 1e-10"),
    ("0.3", "0.04", "0.0000000001", "0.09", "50.0", "volatility = 1e-10, long maturity"),
)

#: volatility exactly zero: the kernel's one special branch.
DETERMINISTIC = (
    ("0.8", "0.03", "0.05", "0.5"),
    ("0.3", "0.04", "0.04", "1.0"),
    ("2.0", "0.02", "0.10", "30.0"),
    ("0.05", "0.06", "0.01", "100.0"),
)


def literal_bond_price(kappa, theta, volatility, initial_rate, maturity):
    """CIR (1985) eq. 23, transcribed as published and evaluated at 50 digits.

    Deliberately *not* rearranged. ``exp(gamma*tau)`` is formed directly, and
    ``gamma - kappa`` appears nowhere because the published form does not
    contain it -- the whole point is that this shares no algebra with the
    float64 kernel it checks. mpmath carries enough digits that none of the
    conditioning problems of float64 apply.
    """
    k, th, sg, r, t = (mp.mpf(x) for x in (kappa, theta, volatility, initial_rate, maturity))
    if t == 0:
        return mp.mpf(1)
    gamma = mp.sqrt(k * k + 2 * sg * sg)
    growth = mp.e ** (gamma * t)
    denominator = (gamma + k) * (growth - 1) + 2 * gamma
    a = (2 * gamma * mp.e ** ((k + gamma) * t / 2) / denominator) ** (2 * k * th / (sg * sg))
    b = 2 * (growth - 1) / denominator
    return a * mp.e ** (-b * r)


def deterministic_bond_price(kappa, theta, initial_rate, maturity):
    """``exp(-int_0^tau r_t dt)`` for the noiseless mean-reverting rate.

    At ``volatility == 0`` the state is ``r_t = theta + (r_0 - theta) e^{-kappa t}``
    exactly, whose integral is elementary. A separate formula, not a limit of
    the one above -- which is what makes it a check on the kernel's special
    branch rather than a restatement of it.
    """
    k, th, r, t = (mp.mpf(x) for x in (kappa, theta, initial_rate, maturity))
    if t == 0:
        return mp.mpf(1)
    return mp.e ** (-(th * t + (r - th) * (1 - mp.e ** (-k * t)) / k))


def _case(kappa, theta, volatility, initial_rate, maturity, note, price):
    gamma = mp.sqrt(mp.mpf(kappa) ** 2 + 2 * mp.mpf(volatility) ** 2)
    return {
        "kappa": kappa,
        "theta": theta,
        "volatility": volatility,
        "initial_rate": initial_rate,
        "maturity": maturity,
        "note": note,
        "gamma_times_maturity": mp.nstr(gamma * mp.mpf(maturity), 8, strip_zeros=False),
        "discount_factor": mp.nstr(price, 30, strip_zeros=False),
        "log_discount_factor": mp.nstr(mp.log(price), 30, strip_zeros=False),
    }


def build_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for volatility in VOLATILITY_SWEEP:
        price = literal_bond_price(
            BOOK["kappa"], BOOK["theta"], volatility, BOOK["initial_rate"], BOOK["maturity"]
        )
        cases.append(
            _case(
                BOOK["kappa"],
                BOOK["theta"],
                volatility,
                BOOK["initial_rate"],
                BOOK["maturity"],
                f"volatility sweep at the book parameters, sigma = {volatility}",
                price,
            )
        )
    for kappa, theta, volatility, initial_rate, maturity, note in REGIMES:
        price = literal_bond_price(kappa, theta, volatility, initial_rate, maturity)
        cases.append(_case(kappa, theta, volatility, initial_rate, maturity, note, price))
    for kappa, theta, initial_rate, maturity in DETERMINISTIC:
        price = deterministic_bond_price(kappa, theta, initial_rate, maturity)
        cases.append(
            _case(
                kappa,
                theta,
                "0.0",
                initial_rate,
                maturity,
                "volatility exactly zero: the deterministic mean-reverting branch",
                price,
            )
        )
    return cases


def render() -> str:
    body: dict[str, object] = {
        "description": (
            "Zero-coupon bond prices under the risk-neutral CIR (1985) short rate, "
            "computed with mpmath at 50 decimal digits from equation 23 as published "
            "-- unrearranged, and sharing no algebra with the float64 kernel it "
            "checks. Not produced by any pricing library, and regenerable from the "
            "checked-in generator."
        ),
        "generator": "scripts/generate_cir_reference_fixtures.py",
        "precision_digits": mp.mp.dps,
        "library": f"mpmath {mp.__version__}",
        "reference": (
            "Cox, J. C., Ingersoll, J. E., Ross, S. A. (1985). A Theory of the Term "
            "Structure of Interest Rates. Econometrica 53(2), 385-407, equation 23."
        ),
        "parameterization": (
            "Risk-neutral: dr = kappa (theta - r) dt + volatility sqrt(r) dW. The "
            "market price of rate risk is absorbed into kappa and theta."
        ),
        "conventions": {
            "discounting": "P(0, maturity) = A(maturity) exp(-B(maturity) * initial_rate)",
            "deterministic_branch": (
                "volatility == 0 uses exp(-(theta*T + (r0 - theta)(1 - exp(-kappa*T))/kappa)), "
                "the integral of the noiseless mean-reverting path -- a separate formula "
                "rather than a limit of equation 23, whose exponent is undefined there."
            ),
            "maturity_zero": "P(0, 0) = 1 exactly.",
        },
        "cases": build_cases(),
    }
    text = json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False)
    body["content_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False) + "\n"


def main() -> int:
    write_or_check(FIXTURE_DIR / "cir_discount_factors.json", render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
