r"""``bcc97_price``: two normalizations, five reductions, and one Monte Carlo.

The model adds exactly one factor to Bates, so the tests are arranged around
what that factor has to do and what it must *not* do.

**Two normalizations, two identity sets.**  The discounted transform is not a
characteristic function -- ``Phi(0) = P(0,T)`` and ``Phi(-i) = S_0 e^{-qT}`` --
while the T-forward-normalized one is, with ``phi(0) = phi(-i) = 1``.  Both sets
are asserted, and the two forms are checked against each other, which is what
makes the normalization itself evidence rather than an assumption.

**The reductions are chained, and each link is bitwise where the arithmetic
allows it.**  A deterministic flat rate gives back ``bates_price`` bit for bit;
no jumps on top of that gives ``heston_price`` bit for bit; a flat diffusion
instead of jumps gives Merton's series; both together give Black-Scholes.  The
Heston and Black-Scholes references are the library's own kernels, reached
through the public API, and Merton's series is written from the paper in
``test_bates.py`` and imported here so there is one copy of it.

**And the transform is checked against the sampler.**  ``BCC97`` simulated with
``PathwiseShortRateDiscounting`` uses *the same rate path* in the drift and in
the discount factor -- the property its own tests pin pathwise -- so the two
routes price the same contract and must agree inside a stated budget.

The reductions retain the shared ``vol_of_vol = 1e-4`` test parameter. It is
not a numerical floor: ``test_bates.py`` separately checks stable convergence
of the rationalized Heston transform down to ``vol_of_vol = 1e-12``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_vollib import fast_black
from fast_vollib.pricing import bates_price, bcc97_price, heston_price
from fast_vollib.pricing._fourier import half_line
from fast_vollib.pricing.bates import bates_characteristic_function
from fast_vollib.pricing.bcc97 import (
    BCC97_QUADRATURE_NODES,
    bcc97_characteristic_function,
    bcc97_discounted_transform,
    bcc97_forward_measure,
)
from fast_vollib.processes import BCC97, CIRShortRate, HestonVariance, LognormalJumps
from fast_vollib.rates import cir_discount_factor
from fast_vollib.simulation.discounting import PathwiseShortRateDiscounting

from .test_bates import LIMIT_VOL_OF_VOL, flat_diffusion, merton_forward_call

SPOT = 100.0
STRIKES = np.array([60.0, 80.0, 100.0, 130.0, 180.0])
MATURITIES = (0.02, 0.25, 1.0, 10.0)

HESTON_SETS = {
    "moderate": {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "vol_of_vol": 0.3, "rho": -0.7},
    "heavy": {"v0": 0.09, "kappa": 0.5, "theta": 0.06, "vol_of_vol": 0.9, "rho": -0.9},
    "fast": {"v0": 0.01, "kappa": 5.0, "theta": 0.02, "vol_of_vol": 0.2, "rho": 0.4},
}
JUMP_SETS = {
    "none": {},
    "mild": {"jump_intensity": 0.5, "mean_log_jump": -0.05, "jump_volatility": 0.2},
    "wild": {"jump_intensity": 5.0, "mean_log_jump": -0.02, "jump_volatility": 0.45},
}
RATE_SETS = {
    "calm": {
        "rate_kappa": 0.3,
        "rate_theta": 0.04,
        "rate_volatility": 0.1,
        "initial_rate": 0.03,
    },
    "volatile": {
        "rate_kappa": 0.15,
        "rate_theta": 0.08,
        "rate_volatility": 0.6,
        "initial_rate": 0.05,
    },
}


#: ``bates_characteristic_function`` has no defaults for the jump parameters --
#: ``bates_price`` supplies them -- so the "none" set is spelled out when the
#: transform is called directly.
def explicit_jumps(jumps: dict[str, float]) -> dict[str, float]:
    return {
        "jump_intensity": 0.0,
        "mean_log_jump": 0.0,
        "jump_volatility": 0.0,
        **jumps,
    }


#: A rate model with no randomness at all, held at one level: the configuration
#: whose price must be Bates's, bit for bit.
def flat_rate(rate: float) -> dict[str, float]:
    return {
        "rate_kappa": 0.3,
        "rate_theta": rate,
        "rate_volatility": 0.0,
        "initial_rate": rate,
    }


#: Deterministic but *not* flat: zero vol-of-rate with the short rate away from
#: its long-run level, so the curve slopes. Still an exact reduction, and a
#: different one -- it exercises the branch with a non-trivial ``B``.
def sloping_rate(initial: float, long_run: float) -> dict[str, float]:
    return {
        "rate_kappa": 0.6,
        "rate_theta": long_run,
        "rate_volatility": 0.0,
        "initial_rate": initial,
    }


# --- the two identity sets -------------------------------------------------------


@pytest.mark.parametrize("heston", list(HESTON_SETS.values()), ids=list(HESTON_SETS))
@pytest.mark.parametrize("jumps", list(JUMP_SETS.values()), ids=list(JUMP_SETS))
@pytest.mark.parametrize("rates", list(RATE_SETS.values()), ids=list(RATE_SETS))
@pytest.mark.parametrize("maturity", MATURITIES)
def test_the_normalized_transform_is_one_at_zero_and_at_minus_i(
    heston, jumps, rates, maturity
) -> None:
    """``phi(0) = 1`` because it is a characteristic function; ``phi(-i) = 1``
    because the T-forward is a martingale. The second is what the jump
    compensator and the rate normalization together buy."""
    phi = bcc97_characteristic_function(
        np.array([0.0 + 0.0j, -1.0j]), maturity=maturity, **heston, **jumps, **rates
    )
    assert abs(complex(phi[0]) - 1.0) < 1e-12
    assert abs(complex(phi[1]) - 1.0) < 1e-11


@pytest.mark.parametrize("heston", list(HESTON_SETS.values()), ids=list(HESTON_SETS))
@pytest.mark.parametrize("jumps", list(JUMP_SETS.values()), ids=list(JUMP_SETS))
@pytest.mark.parametrize("rates", list(RATE_SETS.values()), ids=list(RATE_SETS))
@pytest.mark.parametrize("maturity", MATURITIES)
def test_the_discounted_transform_prices_the_bond_and_the_forward(
    heston, jumps, rates, maturity
) -> None:
    """``Phi(0) = P(0,T)`` and ``Phi(-i) = S_0 e^{-qT}``.

    Neither identity survives normalization -- ``phi`` is one at both arguments
    by construction -- so this is the check that the *unnormalized* product is
    right, and therefore that the normalization divided by the correct things.
    """
    dividend_yield = 0.015
    discount, forward = bcc97_forward_measure(
        spot=SPOT, maturity=maturity, dividend_yield=dividend_yield, **rates
    )
    phi = bcc97_discounted_transform(
        np.array([0.0 + 0.0j, -1.0j]),
        spot=SPOT,
        maturity=maturity,
        dividend_yield=dividend_yield,
        **heston,
        **jumps,
        **rates,
    )
    assert abs(complex(phi[0]) - discount) < 1e-14 * discount
    assert abs(complex(phi[1]) - SPOT * math.exp(-dividend_yield * maturity)) < 1e-11 * SPOT


@pytest.mark.parametrize("rates", list(RATE_SETS.values()), ids=list(RATE_SETS))
@pytest.mark.parametrize("maturity", MATURITIES)
def test_the_two_forms_are_the_same_transform(rates, maturity) -> None:
    r"""``phi^T(u) = Phi(u) F^{-iu} / P``, the relation the collapsed form was
    derived from.

    :func:`bcc97_characteristic_function` evaluates the *algebraically*
    collapsed version -- the spot power and the forward power having cancelled
    on paper -- so this checks the cancellation against the uncollapsed product
    rather than trusting the algebra.
    """
    heston = HESTON_SETS["moderate"]
    jumps = JUMP_SETS["mild"]
    dividend_yield = 0.015
    discount, forward = bcc97_forward_measure(
        spot=SPOT, maturity=maturity, dividend_yield=dividend_yield, **rates
    )
    u = np.array([0.3, 1.0, 4.0, 12.0]) - 0.5j
    normalized = bcc97_characteristic_function(u, maturity=maturity, **heston, **jumps, **rates)
    discounted = bcc97_discounted_transform(
        u,
        spot=SPOT,
        maturity=maturity,
        dividend_yield=dividend_yield,
        **heston,
        **jumps,
        **rates,
    )
    rebuilt = discounted * np.power(forward, -1j * u) / discount
    np.testing.assert_allclose(normalized, rebuilt, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("rates", list(RATE_SETS.values()), ids=list(RATE_SETS))
@pytest.mark.parametrize("maturity", MATURITIES)
def test_the_discount_factor_is_the_cir_bond_bitwise(rates, maturity) -> None:
    """The pricer's own bond and the library's bond are the same number.

    A pricer that discounted with anything else would still produce a smile;
    it would just be a smile for a different curve.
    """
    discount, _forward = bcc97_forward_measure(spot=SPOT, maturity=maturity, **rates)
    reference = cir_discount_factor(
        kappa=rates["rate_kappa"],
        theta=rates["rate_theta"],
        volatility=rates["rate_volatility"],
        initial_rate=rates["initial_rate"],
        maturity=maturity,
    )
    assert float(discount).hex() == float(reference).hex()


# --- the reductions ---------------------------------------------------------------


@pytest.mark.parametrize("heston", list(HESTON_SETS.values()), ids=list(HESTON_SETS))
@pytest.mark.parametrize("jumps", list(JUMP_SETS.values()), ids=list(JUMP_SETS))
@pytest.mark.parametrize("maturity", MATURITIES)
def test_a_deterministic_rate_leaves_the_transform_bitwise_bates(heston, jumps, maturity) -> None:
    r"""The mechanism the price reduction rests on, checked directly.

    Design section 9.1 asks for equality "at transform and price level", and the
    two say different things: a price could agree because a quadrature happened
    to cancel an error, while the transform agreeing bitwise at every node --
    including the shifted arguments the inversions actually evaluate, ``u - i/2``
    and ``u - i`` -- means the rate factor really is ``exp(0) = 1`` and the
    multiplication really is a no-op.

    Checked on the node set Lewis would use, not on a tidy grid, so the
    arguments are the ones the pricer passes.
    """
    rates = flat_rate(0.03)
    u, _weights = half_line(64, 1.0 / math.sqrt(max(heston["v0"], heston["theta"]) * maturity))
    for shift in (0.0, -0.5j, -1.0j):
        argument = u + shift
        got = bcc97_characteristic_function(argument, maturity=maturity, **heston, **jumps, **rates)
        expected = bates_characteristic_function(
            argument, maturity=maturity, **heston, **explicit_jumps(jumps)
        )
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("heston", list(HESTON_SETS.values()), ids=list(HESTON_SETS))
@pytest.mark.parametrize("jumps", list(JUMP_SETS.values()), ids=list(JUMP_SETS))
@pytest.mark.parametrize("maturity", MATURITIES)
def test_a_sloping_deterministic_rate_also_leaves_the_transform_alone(
    heston, jumps, maturity
) -> None:
    """Deterministic, not flat: the rate factor is one because the coefficients
    are linear in ``s``, which does not depend on the curve being level."""
    rates = sloping_rate(initial=0.01, long_run=0.06)
    u, _weights = half_line(64, 1.0 / math.sqrt(max(heston["v0"], heston["theta"]) * maturity))
    got = bcc97_characteristic_function(u - 0.5j, maturity=maturity, **heston, **jumps, **rates)
    expected = bates_characteristic_function(
        u - 0.5j, maturity=maturity, **heston, **explicit_jumps(jumps)
    )
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("rates", list(RATE_SETS.values()), ids=list(RATE_SETS))
def test_a_stochastic_rate_does_move_the_transform(rates) -> None:
    """Otherwise the two tests above would be satisfied by a factor of one."""
    u = np.array([0.5, 2.0, 6.0]) - 0.5j
    heston, jumps = HESTON_SETS["moderate"], JUMP_SETS["mild"]
    got = bcc97_characteristic_function(u, maturity=5.0, **heston, **jumps, **rates)
    reference = bates_characteristic_function(u, maturity=5.0, **heston, **explicit_jumps(jumps))
    assert not np.allclose(got, reference, rtol=1e-6)


@pytest.mark.parametrize("heston", list(HESTON_SETS.values()), ids=list(HESTON_SETS))
@pytest.mark.parametrize("jumps", list(JUMP_SETS.values()), ids=list(JUMP_SETS))
@pytest.mark.parametrize("maturity", MATURITIES)
@pytest.mark.parametrize("formulation", ["lewis", "gatheral"])
@pytest.mark.parametrize("is_call", [True, False])
def test_a_deterministic_flat_rate_is_bitwise_bates(
    heston, jumps, maturity, formulation, is_call
) -> None:
    r"""``rate_volatility = 0`` with ``rate_theta = initial_rate`` is Bates.

    Bitwise, and that requires the rate factor to be *exactly* one rather than
    one to fifteen digits: the factor is
    ``exp(log Psi(1-iu) - (1-iu) log Psi(1))``, formed by differencing the
    affine coefficients, and in the deterministic branch those are exactly
    linear in ``s`` so each difference is a value minus its own bits.

    Priced at the forward and discount *this model* reports, because that is
    what makes the comparison about the transform. Whether ``P`` also equals
    ``exp(-rT)`` to the bit is a separate question, and a separate test.
    """
    rate = 0.03
    dividend_yield = 0.01
    rates = flat_rate(rate)
    discount, forward = bcc97_forward_measure(
        spot=SPOT, maturity=maturity, dividend_yield=dividend_yield, **rates
    )
    got = bcc97_price(
        spot=SPOT,
        strike=STRIKES,
        maturity=maturity,
        dividend_yield=dividend_yield,
        is_call=is_call,
        formulation=formulation,
        **heston,
        **jumps,
        **rates,
    )
    expected = bates_price(
        forward=forward,
        strike=STRIKES,
        maturity=maturity,
        discount=discount,
        is_call=is_call,
        formulation=formulation,
        **heston,
        **jumps,
    )
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("maturity", MATURITIES)
def test_a_flat_deterministic_curve_is_the_flat_discount_factor(maturity) -> None:
    """``Psi(1) = e^{-rT}`` and ``F = S_0 e^{(r-q)T}``, to the last few bits.

    Not asserted bitwise: the deterministic branch reaches ``e^{-rT}`` through
    ``theta*tau + (r_0 - theta) B``, which is a different arithmetic route from
    ``math.exp(-r*T)`` even though the two agree here at every case tried.
    """
    rate, dividend_yield = 0.03, 0.01
    discount, forward = bcc97_forward_measure(
        spot=SPOT, maturity=maturity, dividend_yield=dividend_yield, **flat_rate(rate)
    )
    assert discount == pytest.approx(math.exp(-rate * maturity), rel=1e-15)
    assert forward == pytest.approx(SPOT * math.exp((rate - dividend_yield) * maturity), rel=1e-14)


@pytest.mark.parametrize("maturity", MATURITIES)
@pytest.mark.parametrize("formulation", ["lewis", "gatheral"])
def test_a_sloping_deterministic_curve_is_still_bitwise_bates(maturity, formulation) -> None:
    """The reduction is to *deterministic* rates, not to flat ones.

    With ``rate_volatility = 0`` and the short rate away from its long-run
    level the curve slopes, ``B`` is not proportional to the maturity, and the
    bond is not ``e^{-r_0 T}`` -- and the rate factor is still exactly one,
    because linearity in ``s`` is what it depends on and that is untouched.
    """
    rates = sloping_rate(initial=0.01, long_run=0.06)
    discount, forward = bcc97_forward_measure(spot=SPOT, maturity=maturity, **rates)
    assert discount != pytest.approx(math.exp(-0.01 * maturity), rel=1e-6) or maturity < 0.05
    got = bcc97_price(
        spot=SPOT,
        strike=STRIKES,
        maturity=maturity,
        formulation=formulation,
        **HESTON_SETS["moderate"],
        **JUMP_SETS["mild"],
        **rates,
    )
    expected = bates_price(
        forward=forward,
        strike=STRIKES,
        maturity=maturity,
        discount=discount,
        formulation=formulation,
        **HESTON_SETS["moderate"],
        **JUMP_SETS["mild"],
    )
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("heston", list(HESTON_SETS.values()), ids=list(HESTON_SETS))
@pytest.mark.parametrize("maturity", MATURITIES)
def test_no_jumps_and_a_deterministic_rate_is_bitwise_heston(heston, maturity) -> None:
    """The second link of the chain, against the library's own Heston pricer."""
    rate = 0.03
    rates = flat_rate(rate)
    discount, forward = bcc97_forward_measure(spot=SPOT, maturity=maturity, **rates)
    got = bcc97_price(spot=SPOT, strike=STRIKES, maturity=maturity, **heston, **rates)
    expected = heston_price(
        forward=forward, strike=STRIKES, maturity=maturity, discount=discount, **heston
    )
    np.testing.assert_array_equal(got, expected)


def test_zero_intensity_jumps_are_the_no_jump_defaults_bitwise() -> None:
    """BCC97 with the intensity switched off is SVSI, and it is the same call.

    The jump parameters default to zero, so this pins that passing them
    explicitly changes nothing -- a compensator that were applied twice, or a
    quadrature scale that read the mean jump size at zero intensity, would show
    up here.
    """
    rates = RATE_SETS["volatile"]
    heston = HESTON_SETS["heavy"]
    svsi = bcc97_price(spot=SPOT, strike=STRIKES, maturity=2.0, **heston, **rates)
    explicit = bcc97_price(
        spot=SPOT,
        strike=STRIKES,
        maturity=2.0,
        jump_intensity=0.0,
        mean_log_jump=-0.3,
        jump_volatility=0.4,
        **heston,
        **rates,
    )
    np.testing.assert_array_equal(svsi, explicit)


@pytest.mark.parametrize("jumps", [JUMP_SETS["mild"], JUMP_SETS["wild"]], ids=["mild", "wild"])
@pytest.mark.parametrize("maturity", [0.25, 1.0, 3.0])
@pytest.mark.parametrize("sigma", [0.15, 0.35])
def test_a_flat_diffusion_and_a_deterministic_rate_reduce_to_merton(jumps, maturity, sigma) -> None:
    r"""Merton (1976), as a Poisson mixture of Black-76 calls in forward space.

    The reference is the series written from the paper in ``test_bates.py`` and
    imported rather than copied, evaluated with the library's Black-76 kernel --
    which the Fourier route never touches. The tolerance is set by
    ``vol_of_vol = 1e-4`` rather than zero; see the module docstring.
    """
    rate = 0.04
    rates = flat_rate(rate)
    discount, forward = bcc97_forward_measure(spot=SPOT, maturity=maturity, **rates)
    got = bcc97_price(
        spot=SPOT,
        strike=STRIKES,
        maturity=maturity,
        **flat_diffusion(sigma),
        **jumps,
        **rates,
    )
    expected = discount * merton_forward_call(forward, STRIKES, maturity, sigma, **jumps)
    np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("maturity", [0.25, 1.0, 3.0])
@pytest.mark.parametrize("sigma", [0.15, 0.35])
def test_no_jumps_and_a_flat_diffusion_reduce_to_black_scholes(maturity, sigma) -> None:
    """The bottom of the lattice, against the library's own Black-76 kernel."""
    rate = 0.04
    rates = flat_rate(rate)
    discount, forward = bcc97_forward_measure(spot=SPOT, maturity=maturity, **rates)
    got = bcc97_price(
        spot=SPOT, strike=STRIKES, maturity=maturity, **flat_diffusion(sigma), **rates
    )
    expected = discount * np.asarray(
        fast_black("c", forward, STRIKES, maturity, 0.0, sigma, return_as="numpy"), dtype=float
    )
    np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)


def test_the_shared_reduction_parameter_is_unchanged() -> None:
    """Preserve the reduction-test parameter, not a model conditioning floor."""
    assert LIMIT_VOL_OF_VOL == 1e-4


# --- the quadrature ----------------------------------------------------------------


@pytest.mark.parametrize("heston", list(HESTON_SETS.values()), ids=list(HESTON_SETS))
@pytest.mark.parametrize("jumps", list(JUMP_SETS.values()), ids=list(JUMP_SETS))
@pytest.mark.parametrize("rates", list(RATE_SETS.values()), ids=list(RATE_SETS))
@pytest.mark.parametrize("maturity", MATURITIES)
def test_the_shipped_node_count_is_converged(heston, jumps, rates, maturity) -> None:
    """Re-measured for stochastic rates rather than inherited from Bates.

    Against ten times as many nodes the worst case over this whole grid is
    3.5e-7 on a forward of order 100 -- 3.5e-9 of the forward, and the same
    order Bates measured, so the CIR factor does not change what the quadrature
    has to resolve. The bound below carries a factor of three of headroom.
    """
    shipped = bcc97_price(spot=SPOT, strike=STRIKES, maturity=maturity, **heston, **jumps, **rates)
    refined = bcc97_price(
        spot=SPOT,
        strike=STRIKES,
        maturity=maturity,
        n_nodes=10 * BCC97_QUADRATURE_NODES,
        **heston,
        **jumps,
        **rates,
    )
    assert float(np.max(np.abs(shipped - refined))) < 1e-6


@pytest.mark.parametrize("heston", list(HESTON_SETS.values()), ids=list(HESTON_SETS))
@pytest.mark.parametrize("jumps", list(JUMP_SETS.values()), ids=list(JUMP_SETS))
@pytest.mark.parametrize("rates", list(RATE_SETS.values()), ids=list(RATE_SETS))
@pytest.mark.parametrize("maturity", MATURITIES)
def test_the_two_formulations_agree(heston, jumps, rates, maturity) -> None:
    """Different integrals, differently-behaved integrands, one answer.

    At twice the shipped node count, because Gatheral's integrand has a
    removable singularity at the origin and converges about an order slower --
    measured at 2.6e-5 with 768 nodes and 4.8e-8 with 1536. The node count is
    raised rather than the tolerance loosened: the disagreement is Gatheral's
    quadrature error, and a bound wide enough to hide it would also hide a real
    difference between the two routes.
    """
    common = dict(spot=SPOT, strike=STRIKES, maturity=maturity, n_nodes=2 * BCC97_QUADRATURE_NODES)
    lewis = bcc97_price(formulation="lewis", **common, **heston, **jumps, **rates)
    gatheral = bcc97_price(formulation="gatheral", **common, **heston, **jumps, **rates)
    np.testing.assert_allclose(lewis, gatheral, rtol=1e-6, atol=1e-6)


# --- behaviour ----------------------------------------------------------------------


@pytest.mark.parametrize("rates", list(RATE_SETS.values()), ids=list(RATE_SETS))
def test_put_call_parity_holds_on_the_model_s_own_forward(rates) -> None:
    """``c - p = P (F - K)`` -- with ``P`` and ``F`` from the rate model, which
    is the only pair the price is consistent with."""
    maturity = 1.5
    discount, forward = bcc97_forward_measure(
        spot=SPOT, maturity=maturity, dividend_yield=0.02, **rates
    )
    common = dict(
        spot=SPOT,
        strike=STRIKES,
        maturity=maturity,
        dividend_yield=0.02,
        **HESTON_SETS["moderate"],
        **JUMP_SETS["mild"],
        **rates,
    )
    call = bcc97_price(is_call=True, **common)
    put = bcc97_price(is_call=False, **common)
    np.testing.assert_allclose(call - put, discount * (forward - STRIKES), rtol=1e-12, atol=1e-12)


def test_a_volatile_rate_changes_the_price() -> None:
    """Otherwise the whole factor could be one and every test above would pass.

    Same bond, different rate volatility: ``rate_theta`` and ``initial_rate``
    are held so the *level* is comparable, and only the randomness differs.
    """
    common = dict(
        spot=SPOT,
        strike=STRIKES,
        maturity=5.0,
        rate_kappa=0.3,
        rate_theta=0.04,
        initial_rate=0.04,
        **HESTON_SETS["moderate"],
    )
    still = bcc97_price(rate_volatility=0.0, **common)
    moving = bcc97_price(rate_volatility=0.5, **common)
    assert not np.allclose(still, moving, rtol=1e-6)


def test_a_vector_of_maturities_gets_one_forward_each() -> None:
    """The forward depends on the maturity here, so a surface is not one bond."""
    maturities = np.array([0.5, 0.5, 2.0, 2.0])
    strikes = np.array([90.0, 110.0, 90.0, 110.0])
    rates = RATE_SETS["calm"]
    together = bcc97_price(
        spot=SPOT, strike=strikes, maturity=maturities, **HESTON_SETS["moderate"], **rates
    )
    one_at_a_time = np.array(
        [
            float(
                bcc97_price(
                    spot=SPOT,
                    strike=strike,
                    maturity=maturity,
                    **HESTON_SETS["moderate"],
                    **rates,
                )
            )
            for strike, maturity in zip(strikes, maturities)
        ]
    )
    # Close rather than bitwise, and the reason is the shared quadrature shell
    # rather than anything here: ``fourier_price`` batches every strike at one
    # maturity into a single matrix product, so a two-strike batch and two
    # one-strike calls take different BLAS paths and differ in the last bit.
    # The claim under test is that each maturity got its own bond and forward,
    # which a shared one would break at the second decimal place.
    np.testing.assert_allclose(together, one_at_a_time, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("rates", list(RATE_SETS.values()), ids=list(RATE_SETS))
def test_prices_are_monotone_in_strike_and_bounded(rates) -> None:
    discount, forward = bcc97_forward_measure(spot=SPOT, maturity=1.0, **rates)
    calls = bcc97_price(
        spot=SPOT,
        strike=STRIKES,
        maturity=1.0,
        **HESTON_SETS["moderate"],
        **JUMP_SETS["mild"],
        **rates,
    )
    assert np.all(np.diff(calls) < 0.0)
    assert np.all(calls > np.maximum(discount * (forward - STRIKES), 0.0) - 1e-12)
    assert np.all(calls < discount * forward + 1e-12)


def test_a_non_positive_spot_or_maturity_is_refused() -> None:
    common = dict(strike=STRIKES, **HESTON_SETS["moderate"], **RATE_SETS["calm"])
    with pytest.raises(ValueError, match="spot"):
        bcc97_price(spot=-1.0, maturity=1.0, **common)
    with pytest.raises(ValueError, match="maturity"):
        bcc97_price(spot=SPOT, maturity=0.0, **common)


# --- against the sampler ------------------------------------------------------------


def test_the_fourier_price_agrees_with_a_pathwise_monte_carlo() -> None:
    r"""The capstone: two routes, one model, the same rate path in both roles.

    The simulation discounts with
    :class:`~fast_vollib.simulation.PathwiseShortRateDiscounting`, whose
    trapezoid is the quadrature ``BCC97``'s drift uses -- so the rate cancels
    pathwise between the drift and the discount factor, and what is left to
    compare is the transform against the sampler.

    The budget is four standard errors plus a small allowance for the scheme's
    discretization at 64 steps. Measured at 400,000 paths the largest deviation
    over these strikes was 0.4 standard errors, so the allowance is headroom
    rather than a fitted constant.
    """
    heston = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "vol_of_vol": 0.3, "rho": -0.7}
    jumps = {"jump_intensity": 0.5, "mean_log_jump": -0.05, "jump_volatility": 0.2}
    rates = {
        "rate_kappa": 0.5,
        "rate_theta": 0.05,
        "rate_volatility": 0.2,
        "initial_rate": 0.03,
    }
    dividend_yield = 0.01
    maturity = 1.0
    strikes = np.array([80.0, 100.0, 120.0])

    exact = bcc97_price(
        spot=SPOT,
        strike=strikes,
        maturity=maturity,
        dividend_yield=dividend_yield,
        **heston,
        **jumps,
        **rates,
    )

    process = BCC97(
        variance=HestonVariance(
            kappa=heston["kappa"],
            theta=heston["theta"],
            vol_of_vol=heston["vol_of_vol"],
            rho=heston["rho"],
        ),
        jumps=LognormalJumps(**jumps),
        rates=CIRShortRate(
            kappa=rates["rate_kappa"],
            theta=rates["rate_theta"],
            volatility=rates["rate_volatility"],
        ),
        dividend_yield=dividend_yield,
    )
    grid = np.linspace(0.0, maturity, 65)
    paths = process.sample(
        initial_state={
            "spot": SPOT,
            "variance": heston["v0"],
            "short_rate": rates["initial_rate"],
        },
        time_grid=grid,
        n_paths=200_000,
        rng=2026,
    )
    factors = PathwiseShortRateDiscounting().discount_factors(
        states=paths, time_grid=grid, state_names=BCC97.state_names
    )

    for index, strike in enumerate(strikes):
        sample = factors * np.maximum(paths[:, -1, 0] - strike, 0.0)
        stderr = float(sample.std(ddof=1) / np.sqrt(len(sample)))
        assert abs(float(sample.mean()) - float(exact[index])) < 4.0 * stderr + 0.02
