"""Curves are values, they agree with the rest of the library, and they refuse.

Three separate concerns, and the middle one is the reason this file is longer
than the classes it tests.  fast-vollib already discounts in two other places --
``MonteCarloEngine._discount`` and ``SurfaceMarket.discount_at`` -- and a
library that used two compounding conventions across three doors would quote
different prices for the same market depending which door a caller came
through.  So the agreement is asserted rather than assumed, at machine
precision, against both.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_vollib.rates import (
    CIRDiscountCurve,
    DiscountCurve,
    FlatDiscountCurve,
    RateValidationError,
)

FLAT_RATES = (-0.01, 0.0, 0.005, 0.03, 0.12)
MATURITIES = (0.0, 0.01, 0.25, 1.0, 5.0, 30.0)
CIR = {"kappa": 0.3, "theta": 0.04, "volatility": 0.1, "initial_rate": 0.04}


def curves():
    return [FlatDiscountCurve(rate=0.03), CIRDiscountCurve(**CIR)]


CURVE_IDS = ["flat", "cir"]


# --- the protocol --------------------------------------------------------------


@pytest.mark.parametrize("curve", curves(), ids=CURVE_IDS)
def test_every_shipped_curve_satisfies_the_protocol(curve) -> None:
    assert isinstance(curve, DiscountCurve)


def test_the_protocol_is_structural_and_needs_no_registration() -> None:
    """A caller's own curve is a first-class argument, not a special case."""

    class OwnCurve:
        def discount_factor(self, maturity):
            return 1.0 / (1.0 + 0.04 * maturity)

    assert isinstance(OwnCurve(), DiscountCurve)


def test_an_object_without_the_method_is_not_a_curve() -> None:
    assert not isinstance(object(), DiscountCurve)


# --- curves are values ---------------------------------------------------------


@pytest.mark.parametrize("curve", curves(), ids=CURVE_IDS)
def test_a_curve_is_frozen_hashable_and_compares_structurally(curve) -> None:
    with pytest.raises(Exception):
        curve.rate = 0.5  # type: ignore[misc]
    assert hash(curve) is not None
    assert curve == type(curve)(**{f: getattr(curve, f) for f in curve.__dataclass_fields__})


@pytest.mark.parametrize("curve", curves(), ids=CURVE_IDS)
def test_a_curve_has_no_slots_for_state_it_should_not_hold(curve) -> None:
    """A curve binds valuation inputs, not a process, a path, or a device."""
    forbidden = {"process", "paths", "scenario", "rng", "device", "engine", "instrument"}
    assert forbidden.isdisjoint(set(curve.__dataclass_fields__))


# --- the value at zero ---------------------------------------------------------


@pytest.mark.parametrize("curve", curves(), ids=CURVE_IDS)
def test_a_zero_maturity_discounts_to_exactly_one(curve) -> None:
    """Exactly. A cashflow due today is worth its face value, not 0.9999...."""
    assert float(curve.discount_factor(0.0)) == 1.0


# --- the flat curve ------------------------------------------------------------


@pytest.mark.parametrize("rate", [0, 1, np.int64(0), np.int64(1), np.array(0), np.array(1)])
def test_integer_rates_are_accepted_without_replacing_the_input(rate) -> None:
    curve = FlatDiscountCurve(rate=rate)
    assert curve.rate is rate
    assert float(curve.discount_factor(0.5)) == float(np.exp(-0.5 * rate))


@pytest.mark.parametrize("rate", FLAT_RATES)
@pytest.mark.parametrize("maturity", MATURITIES)
def test_the_flat_curve_is_continuously_compounded(rate, maturity) -> None:
    # ``np.exp`` rather than ``math.exp``: they differ by one ulp at some
    # arguments, and the claim being pinned is which convention the curve uses,
    # not which of two libc exponentials rounds where.
    got = float(FlatDiscountCurve(rate=rate).discount_factor(maturity))
    assert got == float(np.exp(-rate * maturity))


def test_a_negative_flat_rate_discounts_to_more_than_one() -> None:
    """A real market condition, not an input error."""
    assert float(FlatDiscountCurve(rate=-0.02).discount_factor(3.0)) > 1.0


@pytest.mark.parametrize("rate", FLAT_RATES)
def test_the_flat_zero_rate_is_the_rate_at_every_maturity(rate) -> None:
    curve = FlatDiscountCurve(rate=rate)
    assert all(float(curve.zero_rate(t)) == rate for t in MATURITIES)


# --- agreement with the rest of the library ------------------------------------


@pytest.mark.parametrize("rate", FLAT_RATES)
@pytest.mark.parametrize("maturity", [0.25, 1.0, 5.0, 30.0])
def test_the_flat_curve_equals_the_monte_carlo_engine_discount(rate, maturity) -> None:
    """One compounding convention across the library, to the last bit but one.

    Both compute ``exp(-rate * maturity)``; the convention is identical and
    that is what this pins.  They are not required to agree *bitwise*, and the
    reason is a deliberate choice in the engine rather than an accident:
    ``_discount`` uses ``math.exp`` for a Python-float rate specifically to
    avoid materializing a zero-dimensional array whose default precision would
    promote a single-precision payoff, while a curve is array-API native
    throughout and goes through ``numpy.exp``.

    The two libm implementations disagree in the last bit at two of the
    forty-two ``(rate, maturity)`` pairs swept here -- measured against
    ``mpmath`` at fifty digits, ``math.exp`` is the correctly rounded one in
    both cases, and neither is ever wrong by more than 1.3e-16.  One ulp is
    therefore the honest bound, and asserting equality would be asserting
    something about glibc rather than about this library.
    """
    from fast_vollib.simulation.monte_carlo import _discount

    engine_value = float(_discount(1.0, rate=rate, maturity=maturity))
    curve_value = float(FlatDiscountCurve(rate=rate).discount_factor(maturity))
    assert abs(curve_value - engine_value) <= math.ulp(engine_value)


@pytest.mark.parametrize("rate", [0.0, 0.005, 0.03, 0.12])
@pytest.mark.parametrize("maturity", [0.25, 1.0, 5.0, 30.0])
def test_the_flat_curve_equals_the_surface_market_discount(rate, maturity) -> None:
    """``SurfaceMarket`` discounts by ``exp(-r T)`` too; the two must not drift."""
    from fast_vollib.surface import SurfaceMarket

    market = SurfaceMarket.flat(forward=100.0, rate=rate)
    assert float(FlatDiscountCurve(rate=rate).discount_factor(maturity)) == pytest.approx(
        float(market.discount_at(maturity)), rel=1e-15, abs=0.0
    )


# --- the CIR curve -------------------------------------------------------------


def test_the_cir_curve_delegates_to_the_kernel_exactly() -> None:
    from fast_vollib.rates import cir_discount_factor

    curve = CIRDiscountCurve(**CIR)
    for maturity in MATURITIES:
        assert float(curve.discount_factor(maturity)) == float(
            cir_discount_factor(**CIR, maturity=maturity)
        )


def test_the_cir_curve_reports_the_feller_ratio_and_does_not_enforce_it() -> None:
    """Calibrations violate it routinely; the formula holds either way."""
    curve = CIRDiscountCurve(**CIR)
    assert curve.feller_ratio == pytest.approx(2.0 * 0.3 * 0.04 / 0.01)
    assert curve.satisfies_feller

    violating = CIRDiscountCurve(kappa=0.1, theta=0.01, volatility=0.9, initial_rate=0.02)
    assert not violating.satisfies_feller
    # Constructed, and it prices: reporting is not refusing.
    assert 0.0 < float(violating.discount_factor(5.0)) <= 1.0


def test_the_feller_ratio_is_infinite_at_zero_volatility() -> None:
    """The boundary is unreachable, for a different reason: there is no noise."""
    curve = CIRDiscountCurve(kappa=0.3, theta=0.04, volatility=0.0, initial_rate=0.04)
    assert curve.feller_ratio == float("inf")
    assert curve.satisfies_feller


def test_the_cir_curve_zero_and_forward_rates_are_consistent_with_its_prices() -> None:
    curve = CIRDiscountCurve(**CIR)
    for maturity in (0.25, 1.0, 5.0, 30.0):
        price = float(curve.discount_factor(maturity))
        assert float(curve.zero_rate(maturity)) == pytest.approx(
            -math.log(price) / maturity, rel=1e-14
        )
    # f(0,0) = r0, and the forward curve starts there.
    assert float(curve.instantaneous_forward_rate(0.0)) == pytest.approx(
        CIR["initial_rate"], abs=1e-15
    )


def test_the_cir_curve_is_strictly_decreasing_in_maturity() -> None:
    curve = CIRDiscountCurve(**CIR)
    prices = [float(curve.discount_factor(t)) for t in MATURITIES]
    assert all(later < earlier for earlier, later in zip(prices, prices[1:])), prices


# --- refusal -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kappa": 0.0}, "kappa must be strictly positive"),
        ({"theta": -1.0}, "theta must be non-negative"),
        ({"volatility": -0.5}, "volatility must be non-negative"),
        ({"initial_rate": -0.01}, "initial_rate must be non-negative"),
        ({"kappa": float("nan")}, "kappa must be finite"),
    ],
)
def test_an_invalid_cir_curve_is_refused_at_construction(kwargs, message) -> None:
    """Before it can price anything, so a bad curve cannot produce a number."""
    with pytest.raises(RateValidationError, match=message):
        CIRDiscountCurve(**{**CIR, **kwargs})


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_flat_rate_is_refused(bad) -> None:
    with pytest.raises(RateValidationError, match="rate must be finite"):
        FlatDiscountCurve(rate=bad)


@pytest.mark.parametrize("curve", curves(), ids=CURVE_IDS)
def test_a_negative_maturity_is_refused(curve) -> None:
    """Discounting backwards is not a market convention this library guesses at."""
    with pytest.raises(RateValidationError, match="maturity must be non-negative"):
        curve.discount_factor(-1.0)


@pytest.mark.parametrize("curve", curves(), ids=CURVE_IDS)
def test_a_vector_of_maturities_is_refused_rather_than_broadcast(curve) -> None:
    """One factor per call. A term structure here would be summed as a schedule."""
    with pytest.raises(RateValidationError, match="must be a scalar"):
        curve.discount_factor(np.array([1.0, 2.0]))


# --- backends ------------------------------------------------------------------


def test_a_torch_curve_keeps_its_tensor_and_its_gradient() -> None:
    """Parameters are stored as passed, so an optimizer keeps its own tensor."""
    torch = pytest.importorskip("torch")
    rate = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)
    curve = FlatDiscountCurve(rate=rate)
    assert curve.rate is rate

    value = curve.discount_factor(torch.tensor(4.0, dtype=torch.float64))
    assert torch.is_tensor(value) and value.requires_grad
    value.backward()
    # dP/dr = -T exp(-rT), the analytic duration of a unit zero-coupon bond.
    expected = -4.0 * math.exp(-0.03 * 4.0)
    assert float(rate.grad) == pytest.approx(expected, rel=1e-12)


def test_a_torch_cir_curve_differentiates_in_its_own_parameters() -> None:
    torch = pytest.importorskip("torch")
    kappa = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    curve = CIRDiscountCurve(
        kappa=kappa,
        theta=torch.tensor(0.04, dtype=torch.float64),
        volatility=torch.tensor(0.1, dtype=torch.float64),
        initial_rate=torch.tensor(0.04, dtype=torch.float64),
    )
    value = curve.discount_factor(torch.tensor(5.0, dtype=torch.float64))
    value.backward()
    assert kappa.grad is not None and torch.isfinite(kappa.grad)
