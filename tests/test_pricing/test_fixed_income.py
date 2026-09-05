"""Present value: hand-computable, curve-agnostic, and differentiable.

The arithmetic is one weighted sum, so most of what is worth testing is not the
sum.  It is that the curve is never inferred, that a curve nobody registered
works, that a curve returning nonsense is refused before the nonsense is added
to anything, and that the result still carries a gradient when the curve holds a
tensor -- which is the property that distinguishes this routine from the Fourier
pricers next to it in the same package.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_vollib.instruments import (
    EuropeanOption,
    UnsupportedInstrumentError,
    ZeroCouponBond,
)
from fast_vollib.pricing import present_value
from fast_vollib.rates import CIRDiscountCurve, FlatDiscountCurve

CIR = {"kappa": 0.3, "theta": 0.04, "volatility": 0.1, "initial_rate": 0.04}


def curves():
    return [FlatDiscountCurve(rate=0.03), CIRDiscountCurve(**CIR)]


CURVE_IDS = ["flat", "cir"]


# --- the value -----------------------------------------------------------------


@pytest.mark.parametrize("rate", [0.0, 0.01, 0.05, -0.005])
@pytest.mark.parametrize("maturity", [0.0, 0.5, 2.0, 30.0])
def test_a_zero_coupon_bond_is_its_face_times_the_discount_factor(rate, maturity) -> None:
    """Hand-computable, which is the point of testing the flat curve first."""
    bond = ZeroCouponBond(maturity=maturity, face_value=250.0)
    got = present_value(bond, discount_curve=FlatDiscountCurve(rate=rate))
    assert got == pytest.approx(250.0 * math.exp(-rate * maturity), rel=1e-14)


@pytest.mark.parametrize("curve", curves(), ids=CURVE_IDS)
def test_a_payment_due_now_is_worth_its_face_value_under_any_curve(curve) -> None:
    """``P(0, 0) == 1`` exactly, so this is an equality rather than an approximation."""
    assert (
        present_value(ZeroCouponBond(maturity=0.0, face_value=77.5), discount_curve=curve) == 77.5
    )


@pytest.mark.parametrize("curve", curves(), ids=CURVE_IDS)
def test_the_face_value_scales_the_present_value_exactly(curve) -> None:
    unit = present_value(ZeroCouponBond(maturity=3.0), discount_curve=curve)
    for face in (0.5, 10.0, 1e6):
        scaled = present_value(ZeroCouponBond(maturity=3.0, face_value=face), discount_curve=curve)
        assert scaled == face * unit


@pytest.mark.parametrize("curve", curves(), ids=CURVE_IDS)
def test_the_present_value_equals_the_curve_read_directly(curve) -> None:
    """No hidden convention between the curve and the sum."""
    for maturity in (0.25, 1.0, 7.0):
        bond = ZeroCouponBond(maturity=maturity, face_value=100.0)
        assert present_value(bond, discount_curve=curve) == 100.0 * float(
            curve.discount_factor(maturity)
        )


def test_a_longer_bond_is_worth_less_under_a_positive_curve() -> None:
    values = [
        present_value(ZeroCouponBond(maturity=t), discount_curve=CIRDiscountCurve(**CIR))
        for t in (0.5, 1.0, 5.0, 30.0)
    ]
    assert all(later < earlier for earlier, later in zip(values, values[1:])), values


def test_the_result_is_a_python_float_by_default() -> None:
    """Following ``price_instrument``'s convention: host output, formatted."""
    got = present_value(ZeroCouponBond(maturity=1.0), discount_curve=FlatDiscountCurve(rate=0.03))
    assert type(got) is float


# --- the curve is an argument, never an inference ------------------------------


def test_a_duck_typed_curve_works_with_no_registration() -> None:
    """The protocol is structural; a caller's own firm curve is first class."""

    class SimpleCompounding:
        def discount_factor(self, maturity):
            return 1.0 / (1.0 + 0.04 * maturity)

    bond = ZeroCouponBond(maturity=5.0, face_value=100.0)
    assert present_value(bond, discount_curve=SimpleCompounding()) == pytest.approx(
        100.0 / 1.2, rel=1e-14
    )


def test_a_currency_on_the_contract_does_not_select_a_curve() -> None:
    """Descriptive metadata never picks a valuation input."""
    curve = FlatDiscountCurve(rate=0.03)
    usd = ZeroCouponBond(maturity=2.0, currency="USD")
    eur = ZeroCouponBond(maturity=2.0, currency="EUR")
    none = ZeroCouponBond(maturity=2.0)
    assert (
        present_value(usd, discount_curve=curve)
        == present_value(eur, discount_curve=curve)
        == present_value(none, discount_curve=curve)
    )


def test_the_discount_curve_argument_is_keyword_only() -> None:
    """So a call site always names the curve that produced the number."""
    with pytest.raises(TypeError):
        present_value(ZeroCouponBond(maturity=1.0), FlatDiscountCurve(rate=0.03))  # type: ignore[misc]


# --- refusal -------------------------------------------------------------------


def test_an_instrument_that_is_not_fixed_income_is_refused() -> None:
    option = EuropeanOption(underlier="ACME", option_type="call", strike=100.0, maturity=1.0)
    with pytest.raises(UnsupportedInstrumentError, match="is not one"):
        present_value(option, discount_curve=FlatDiscountCurve(rate=0.03))  # type: ignore[arg-type]


def test_an_object_that_is_not_a_curve_is_refused_by_name() -> None:
    with pytest.raises(UnsupportedInstrumentError, match="discount_factor"):
        present_value(ZeroCouponBond(maturity=1.0), discount_curve=object())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_discount_factor_is_refused_before_it_is_summed(bad) -> None:
    """A NaN added into a total is a silent wrong answer, not a loud one."""

    class Broken:
        def discount_factor(self, maturity):
            return bad

    with pytest.raises(ValueError, match="non-finite factor"):
        present_value(ZeroCouponBond(maturity=1.0), discount_curve=Broken())


def test_a_term_structure_returned_by_a_curve_is_refused() -> None:
    """One factor per maturity; an array here would be summed as several bonds."""

    class Vector:
        def discount_factor(self, maturity):
            return np.array([0.9, 0.8, 0.7])

    with pytest.raises(ValueError, match="one factor per maturity"):
        present_value(ZeroCouponBond(maturity=1.0), discount_curve=Vector())


def test_a_users_curve_may_return_a_non_positive_factor() -> None:
    """Positivity is a theorem about the shipped curves, not a rule imposed on all.

    A caller modelling a defaulted or written-down claim is entitled to a zero
    factor, and this routine is not the place to overrule their market.
    """

    class Written:
        def discount_factor(self, maturity):
            return 0.0

    assert present_value(ZeroCouponBond(maturity=1.0), discount_curve=Written()) == 0.0


# --- gradients -----------------------------------------------------------------


def test_a_torch_curve_gives_a_present_value_that_carries_a_gradient() -> None:
    """The property that separates this routine from the Fourier pricers.

    ``dPV/d rate = -T * face * exp(-rT)`` -- the analytic (negative) dollar
    duration of a zero-coupon bond, which is a closed form, so the gradient is
    checked against the number rather than merely for existence.
    """
    torch = pytest.importorskip("torch")
    rate = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)
    bond = ZeroCouponBond(maturity=4.0, face_value=1000.0)

    value = present_value(bond, discount_curve=FlatDiscountCurve(rate=rate), return_native=True)
    assert torch.is_tensor(value) and value.requires_grad

    value.backward()
    expected = -4.0 * 1000.0 * math.exp(-0.03 * 4.0)
    assert float(rate.grad) == pytest.approx(expected, rel=1e-12)


def test_a_torch_cir_curve_differentiates_the_present_value_in_kappa() -> None:
    torch = pytest.importorskip("torch")
    kappa = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    curve = CIRDiscountCurve(
        kappa=kappa,
        theta=torch.tensor(0.04, dtype=torch.float64),
        volatility=torch.tensor(0.1, dtype=torch.float64),
        initial_rate=torch.tensor(0.04, dtype=torch.float64),
    )
    value = present_value(
        ZeroCouponBond(maturity=10.0, face_value=100.0), discount_curve=curve, return_native=True
    )
    value.backward()
    assert kappa.grad is not None and torch.isfinite(kappa.grad)


def test_return_native_false_detaches_and_formats_a_torch_value() -> None:
    """The default is a host number, following ``price_instrument``."""
    torch = pytest.importorskip("torch")
    rate = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)
    got = present_value(
        ZeroCouponBond(maturity=2.0, face_value=100.0),
        discount_curve=FlatDiscountCurve(rate=rate),
    )
    assert type(got) is float
    assert got == pytest.approx(100.0 * math.exp(-0.06), rel=1e-14)


def test_a_jax_curve_is_traceable_and_differentiable() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    bond = ZeroCouponBond(maturity=4.0, face_value=1000.0)

    def value(rate):
        return present_value(bond, discount_curve=FlatDiscountCurve(rate=rate), return_native=True)

    assert float(jax.jit(value)(jnp.asarray(0.03))) == pytest.approx(
        1000.0 * math.exp(-0.12), rel=1e-5
    )
    gradient = float(jax.grad(value)(jnp.asarray(0.03)))
    assert gradient == pytest.approx(-4.0 * 1000.0 * math.exp(-0.12), rel=1e-4)
