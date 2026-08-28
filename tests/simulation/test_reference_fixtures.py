"""The oracles are themselves checked against a frozen high-precision reference.

An independent closed form written beside a test protects against the library
being wrong. It does not protect against the *oracle* being wrong: an edit could
change a formula and the test that uses it together, and nothing would notice.

So the same values are frozen in ``tests/fixtures/monte_carlo``, computed a
second time through a different route — ``mpmath`` at fifty digits, with its
own error function, rather than ``scipy.special.ndtr`` — and compared here at
double precision. Two implementations at different precisions, written from the
same published formula but sharing no code, agreeing to 1e-12 is a meaningful
check on both.

The Monte Carlo estimates are then checked against the frozen values as well,
so the whole chain is pinned: fixture, oracle, engine.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from fast_vollib.instruments import (
    AsianOption,
    BinaryOption,
    EuropeanOption,
    Forward,
    VanillaMarketInputs,
    VarianceSwap,
    price_instrument,
)
from fast_vollib.processes import GBM
from fast_vollib.simulation import MonteCarloEngine

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "monte_carlo" / "gbm_closed_forms.json"
)
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
MARKET = FIXTURE["market"]

SPOT = float(MARKET["spot"])
RATE = float(MARKET["rate"])
VOLATILITY = float(MARKET["volatility"])
MATURITY = float(MARKET["maturity"])

#: Passed explicitly to every oracle below. They default to the constants in
#: ``test_monte_carlo``, and a silent divergence between the two files would
#: compare an oracle at one market against a reference at another -- which would
#: look like a numerical disagreement and be a bookkeeping one.
MARKET_KWARGS = {"spot": SPOT, "rate": RATE, "volatility": VOLATILITY}

#: Agreement required between two independent closed forms. Both compute the
#: same quantity, so anything beyond accumulated float64 rounding is a real
#: disagreement rather than noise.
CLOSED_FORM_TOLERANCE = 1e-12


def cases(instrument: str) -> list[dict[str, Any]]:
    return [case for case in FIXTURE["cases"] if case["instrument"] == instrument]


def ident(case: dict[str, Any]) -> str:
    parts = [str(case.get(key)) for key in ("option_type", "strike", "n_steps") if key in case]
    return "-".join(parts) or str(case.get("delivery_price") or case.get("strike_variance"))


# --- the fixture itself --------------------------------------------------------


def test_the_fixture_records_its_provenance() -> None:
    """A reference nobody can reproduce is not a reference."""
    assert FIXTURE["generator"] == "scripts/generate_mc_reference_fixtures.py"
    assert FIXTURE["precision_digits"] >= 30
    assert FIXTURE["library"].startswith("mpmath")
    assert "content_sha256" in FIXTURE
    assert set(MARKET) >= {"spot", "rate", "volatility", "maturity", "measure"}
    assert "risk-neutral" in MARKET["measure"]


def test_the_fixture_market_is_not_degenerate() -> None:
    """``r - sigma^2 / 2`` must not vanish, or several oracles lose their power.

    At that point the geometric-Asian and variance formulas drop their drift
    term entirely and an at-the-money digital call and put are worth the same,
    so neither a sign error in the drift correction nor a call/put swap would be
    visible.
    """
    log_drift = RATE - 0.5 * VOLATILITY**2
    assert abs(log_drift) > 1e-3, log_drift
    assert float(MARKET["log_drift"]) == pytest.approx(log_drift, abs=1e-12)


def test_the_oracles_and_the_fixture_agree_on_the_market() -> None:
    """The two files carry their own constants; they must not drift apart."""
    from .test_monte_carlo import (
        MATURITY as ORACLE_MATURITY,
        RATE as ORACLE_RATE,
        SPOT as ORACLE_SPOT,
        VOLATILITY as ORACLE_VOLATILITY,
    )

    assert (ORACLE_SPOT, ORACLE_RATE, ORACLE_VOLATILITY, ORACLE_MATURITY) == (
        SPOT,
        RATE,
        VOLATILITY,
        MATURITY,
    )


def test_a_digital_call_and_put_are_not_the_same_number() -> None:
    """Guards the binary oracle against the degeneracy above."""
    calls = {
        c["strike"]: float(c["value"]) for c in cases("binary_option") if c["option_type"] == "call"
    }
    puts = {
        c["strike"]: float(c["value"]) for c in cases("binary_option") if c["option_type"] == "put"
    }
    for strike, call_value in calls.items():
        assert call_value != pytest.approx(puts[strike], abs=1e-9), strike


def test_the_fixture_states_the_conventions_it_was_computed_under() -> None:
    conventions = FIXTURE["conventions"]
    assert "valuation date excluded" in conventions["asian_fixings"]
    assert "strict inequality" in conventions["binary"]
    assert "divided by the year fraction" in conventions["variance"]


def test_the_fixture_regenerates_byte_for_byte() -> None:
    """A stale reference is worse than none: it pins the wrong answer."""
    import sys

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from generate_mc_reference_fixtures import render

    assert FIXTURE_PATH.read_text(encoding="utf-8") == render()


def test_every_case_carries_a_finite_value() -> None:
    assert FIXTURE["cases"]
    for case in FIXTURE["cases"]:
        assert math.isfinite(float(case["value"])), case


# --- the analytic kernel against the reference ---------------------------------


@pytest.mark.parametrize("case", cases("european_option"), ids=ident)
def test_the_library_kernel_matches_the_high_precision_reference(case: dict[str, Any]) -> None:
    """The shipped Black-Scholes kernel, against fifty digits of mpmath."""
    option = EuropeanOption(
        underlier="ACME",
        option_type=case["option_type"],
        strike=float(case["strike"]),
        maturity=MATURITY,
    )
    market = VanillaMarketInputs(underlying=SPOT, rate=RATE, volatility=VOLATILITY)
    got = float(price_instrument(option, market, model="black_scholes")[0])
    assert got == pytest.approx(float(case["value"]), abs=CLOSED_FORM_TOLERANCE)


# --- the in-test oracles against the reference ---------------------------------


@pytest.mark.parametrize("case", cases("binary_option"), ids=ident)
def test_the_binary_oracle_matches_the_reference(case: dict[str, Any]) -> None:
    from .test_monte_carlo import cash_or_nothing

    digital = BinaryOption(
        underlier="ACME",
        option_type=case["option_type"],
        strike=float(case["strike"]),
        maturity=MATURITY,
        cash_amount=float(case["cash_amount"]),
    )
    assert cash_or_nothing(digital, **MARKET_KWARGS) == pytest.approx(
        float(case["value"]), abs=CLOSED_FORM_TOLERANCE
    )


@pytest.mark.parametrize("case", cases("asian_option"), ids=ident)
def test_the_geometric_asian_oracle_matches_the_reference(case: dict[str, Any]) -> None:
    from .test_monte_carlo import discrete_geometric_asian

    contract = AsianOption(
        underlier="ACME",
        option_type=case["option_type"],
        strike=float(case["strike"]),
        averaging_method="geometric",
        strike_convention="fixed",
        maturity=MATURITY,
    )
    got = discrete_geometric_asian(contract, int(case["n_steps"]), **MARKET_KWARGS)
    assert got == pytest.approx(float(case["value"]), abs=CLOSED_FORM_TOLERANCE)


@pytest.mark.parametrize("case", cases("variance_swap"), ids=ident)
def test_the_variance_oracle_matches_the_reference(case: dict[str, Any]) -> None:
    from .test_monte_carlo import expected_realized_variance

    process = GBM.risk_neutral(rate=RATE, volatility=VOLATILITY)
    discounted = math.exp(-RATE * MATURITY) * expected_realized_variance(
        process, int(case["n_steps"]), maturity=MATURITY
    )
    assert discounted == pytest.approx(float(case["value"]), abs=CLOSED_FORM_TOLERANCE)


# --- the engine against the reference ------------------------------------------


def price(instrument: Any, *, n_steps: int, n_paths: int = 100_000, rng: int = 20260825) -> Any:
    return MonteCarloEngine().price(
        instrument,
        VanillaMarketInputs(underlying=SPOT, rate=RATE),
        process=GBM.risk_neutral(rate=RATE, volatility=VOLATILITY),
        n_paths=n_paths,
        n_steps=n_steps,
        rng=rng,
    )


def within_band(result: Any, reference: float) -> bool:
    """Five standard errors, plus a floor for double-precision arithmetic."""
    return abs(result.price - reference) <= 5.0 * result.stderr + 1e-12


@pytest.mark.parametrize("case", cases("forward"), ids=ident)
def test_forward_monte_carlo_matches_the_reference(case: dict[str, Any]) -> None:
    contract = Forward(
        underlier="ACME", delivery_price=float(case["delivery_price"]), maturity=MATURITY
    )
    assert within_band(price(contract, n_steps=4), float(case["value"]))


@pytest.mark.parametrize("case", cases("european_option"), ids=ident)
def test_european_monte_carlo_matches_the_reference(case: dict[str, Any]) -> None:
    contract = EuropeanOption(
        underlier="ACME",
        option_type=case["option_type"],
        strike=float(case["strike"]),
        maturity=MATURITY,
    )
    assert within_band(price(contract, n_steps=1), float(case["value"]))


@pytest.mark.parametrize("case", cases("binary_option"), ids=ident)
def test_binary_monte_carlo_matches_the_reference(case: dict[str, Any]) -> None:
    contract = BinaryOption(
        underlier="ACME",
        option_type=case["option_type"],
        strike=float(case["strike"]),
        maturity=MATURITY,
        cash_amount=float(case["cash_amount"]),
    )
    assert within_band(price(contract, n_steps=1), float(case["value"]))


@pytest.mark.parametrize("case", cases("asian_option"), ids=ident)
def test_geometric_asian_monte_carlo_matches_the_reference(case: dict[str, Any]) -> None:
    contract = AsianOption(
        underlier="ACME",
        option_type=case["option_type"],
        strike=float(case["strike"]),
        averaging_method="geometric",
        strike_convention="fixed",
        maturity=MATURITY,
    )
    assert within_band(price(contract, n_steps=int(case["n_steps"])), float(case["value"]))


@pytest.mark.parametrize("case", cases("variance_swap"), ids=ident)
def test_variance_swap_monte_carlo_matches_the_reference(case: dict[str, Any]) -> None:
    contract = VarianceSwap(
        underlier="ACME", strike_variance=float(case["strike_variance"]), maturity=MATURITY
    )
    assert within_band(price(contract, n_steps=int(case["n_steps"])), float(case["value"]))
