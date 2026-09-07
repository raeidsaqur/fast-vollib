r"""``bates_price``: the identities, the reductions, and one documented limit.

The reductions carry most of the weight, because a jump model has no published
reference table anyone agrees on.  Three of them are checked against something
this module shares no code with:

* **Heston** -- the library's own ``heston_price``, and the agreement is
  *bitwise*, not approximate, because at zero intensity the transform, the
  quadrature scale, and therefore every node are identical.
* **Merton (1976)** -- the jump series written out here from the paper,
  evaluated with the library's Black-76 kernel, which the Fourier route never
  touches.
* **Black-Scholes** -- the same kernel with no jumps at all.

The fourth check is internal but independent: ``'lewis'`` and ``'gatheral'`` are
different integrals with differently-behaved integrands, and they must agree.

Small positive vol-of-vol is checked against the constant-variance limit;
the characteristic function retains the stochastic model using a rationalized
expression in its poorly conditioned parameter region.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_vollib import fast_black
from fast_vollib.pricing import bates_price, heston_price
from fast_vollib.pricing.bates import (
    BATES_QUADRATURE_NODES,
    bates_characteristic_function,
)

FORWARD = 100.0
STRIKES = np.array([60.0, 80.0, 100.0, 130.0, 180.0])
MATURITIES = (0.02, 0.05, 0.25, 1.0, 3.0, 10.0, 30.0)

HESTON_SETS = {
    "moderate": {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "vol_of_vol": 0.3, "rho": -0.7},
    "heavy": {"v0": 0.09, "kappa": 0.5, "theta": 0.06, "vol_of_vol": 0.9, "rho": -0.9},
    "fast": {"v0": 0.01, "kappa": 5.0, "theta": 0.02, "vol_of_vol": 0.2, "rho": 0.4},
}
JUMP_SETS = {
    "mild": {"jump_intensity": 0.5, "mean_log_jump": -0.05, "jump_volatility": 0.2},
    "crash": {"jump_intensity": 0.3, "mean_log_jump": -0.40, "jump_volatility": 0.05},
    "wild": {"jump_intensity": 5.0, "mean_log_jump": -0.02, "jump_volatility": 0.45},
}

#: Shared reduction-test parameter, not a lower bound on supported vol-of-vol.
LIMIT_VOL_OF_VOL = 1e-4


def flat_diffusion(sigma: float) -> dict[str, float]:
    """Heston parameters standing in for a constant volatility ``sigma``."""
    return {
        "v0": sigma * sigma,
        "kappa": 1.0,
        "theta": sigma * sigma,
        "vol_of_vol": LIMIT_VOL_OF_VOL,
        "rho": 0.0,
    }


def merton_forward_call(
    forward: float,
    strike: np.ndarray,
    maturity: float,
    sigma: float,
    jump_intensity: float,
    mean_log_jump: float,
    jump_volatility: float,
    n_terms: int = 200,
) -> np.ndarray:
    r"""Merton (1976) as a Poisson mixture of Black-76 calls, written from the paper.

    Conditional on ``k`` jumps in ``[0, T]``, the log forward return is normal:
    the diffusion contributes variance ``sigma^2 T`` and the jumps contribute
    ``k delta^2`` plus a mean shift ``k m``, while the compensator removes
    ``lambda mu_J T`` from the drift. So conditionally the spot is log-normal
    with

    ``F_k = F exp(-lambda mu_J T) (1 + mu_J)^k``   and
    ``sigma_k = sqrt(sigma^2 + k delta^2 / T)``,

    and the price is the Poisson-weighted sum of Black-76 calls on those. The
    weights sum the forwards back to ``F`` exactly -- ``sum_k P(k) F_k = F`` --
    which is the martingale property and is asserted below.

    This shares no code with the Fourier route: the per-term price is the
    library's own Black-76 kernel, reached through the public API.
    """
    mu_jump = math.expm1(mean_log_jump + 0.5 * jump_volatility * jump_volatility)
    total = np.zeros(np.shape(strike), dtype=float)
    weight = math.exp(-jump_intensity * maturity)
    for k in range(n_terms):
        if k:
            weight *= jump_intensity * maturity / k
        forward_k = forward * math.exp(-jump_intensity * mu_jump * maturity) * (1.0 + mu_jump) ** k
        sigma_k = math.sqrt(sigma * sigma + k * jump_volatility**2 / maturity)
        total += weight * np.asarray(
            fast_black("c", forward_k, strike, maturity, 0.0, sigma_k, return_as="numpy"),
            dtype=float,
        )
        if k > jump_intensity * maturity + 60 and weight < 1e-300:
            break
    return total


# --- the two identities --------------------------------------------------------


@pytest.mark.parametrize("heston", list(HESTON_SETS.values()), ids=list(HESTON_SETS))
@pytest.mark.parametrize("jumps", list(JUMP_SETS.values()), ids=list(JUMP_SETS))
@pytest.mark.parametrize("maturity", [0.05, 1.0, 10.0])
def test_the_transform_is_normalized_at_zero_and_at_minus_i(heston, jumps, maturity) -> None:
    """``phi(0) = 1`` because it is a characteristic function; ``phi(-i) = 1``
    because the forward is a martingale.

    The second is what the compensator buys. Without it the jumps would add
    ``exp(lambda mu_J T)`` of drift and every price would be wrong in a way no
    individual term looks wrong -- which is why it is asserted rather than
    assumed.
    """
    parameters = {"maturity": maturity, **heston, **jumps}
    assert abs(bates_characteristic_function(0.0 + 0.0j, **parameters) - 1.0) < 1e-14
    assert abs(bates_characteristic_function(-1.0j, **parameters) - 1.0) < 1e-11


def test_dropping_the_compensator_would_break_the_martingale_identity() -> None:
    """So the identity above is load-bearing rather than trivially satisfied."""
    jumps = JUMP_SETS["crash"]
    mu_jump = math.expm1(jumps["mean_log_jump"] + 0.5 * jumps["jump_volatility"] ** 2)
    uncompensated = math.exp(jumps["jump_intensity"] * mu_jump * 1.0)
    assert abs(uncompensated - 1.0) > 0.03


# --- reduction: Heston, bitwise ------------------------------------------------


@pytest.mark.parametrize("heston", list(HESTON_SETS.values()), ids=list(HESTON_SETS))
@pytest.mark.parametrize("formulation", ["lewis", "gatheral"])
@pytest.mark.parametrize("maturity", MATURITIES)
def test_zero_jump_intensity_is_the_heston_price_bitwise(heston, formulation, maturity) -> None:
    """Not "agrees to 1e-12": the same number.

    At zero intensity the compensator is zero, the jump factor is ``exp(0)``,
    and the quadrature scale's jump term is exactly zero -- so the nodes, the
    integrand, and the summation order are all identical to Heston's. Anything
    less than equality would mean one of those three had drifted.
    """
    common = {
        "forward": FORWARD,
        "strike": STRIKES,
        "maturity": maturity,
        "formulation": formulation,
        **heston,
    }
    np.testing.assert_array_equal(bates_price(**common), heston_price(**common))


def test_the_defaults_are_the_no_jump_case() -> None:
    """So ``bates_price`` with Heston parameters alone is Heston."""
    common = {"forward": FORWARD, "strike": STRIKES, "maturity": 1.0, **HESTON_SETS["moderate"]}
    np.testing.assert_array_equal(
        bates_price(**common),
        bates_price(**common, jump_intensity=0.0, mean_log_jump=0.0, jump_volatility=0.0),
    )


def test_the_bitwise_reduction_survives_puts_and_a_discount_factor() -> None:
    common = {
        "forward": FORWARD,
        "strike": STRIKES,
        "maturity": 1.0,
        "is_call": False,
        "discount": 0.97,
        **HESTON_SETS["heavy"],
    }
    np.testing.assert_array_equal(bates_price(**common), heston_price(**common))


# --- reduction: Merton ---------------------------------------------------------


@pytest.mark.parametrize("jumps", list(JUMP_SETS.values()), ids=list(JUMP_SETS))
@pytest.mark.parametrize("maturity", [0.05, 0.25, 1.0, 3.0])
def test_a_flat_diffusion_reduces_to_mertons_own_series(jumps, maturity) -> None:
    """Against the paper's series, evaluated with a kernel this route never uses."""
    sigma = 0.2
    reference = merton_forward_call(FORWARD, STRIKES, maturity, sigma, **jumps)
    produced = bates_price(
        forward=FORWARD, strike=STRIKES, maturity=maturity, **flat_diffusion(sigma), **jumps
    )
    np.testing.assert_allclose(produced, reference, atol=2e-6, rtol=0.0)


def test_the_merton_series_sums_its_forwards_back_to_the_forward() -> None:
    """The martingale property of the reference, so a broken reference is caught.

    Without this, a reference with the compensator dropped would disagree with
    the pricer and the test above would read as a failure of the pricer.
    """
    jumps = JUMP_SETS["crash"]
    maturity = 1.0
    mu_jump = math.expm1(jumps["mean_log_jump"] + 0.5 * jumps["jump_volatility"] ** 2)
    weight = math.exp(-jumps["jump_intensity"] * maturity)
    total = 0.0
    for k in range(200):
        if k:
            weight *= jumps["jump_intensity"] * maturity / k
        total += (
            weight
            * FORWARD
            * math.exp(-jumps["jump_intensity"] * mu_jump * maturity)
            * (1.0 + mu_jump) ** k
        )
    assert total == pytest.approx(FORWARD, rel=1e-12)


# --- reduction: Black-Scholes --------------------------------------------------


@pytest.mark.parametrize("sigma", [0.1, 0.2, 0.45])
@pytest.mark.parametrize("maturity", [0.05, 1.0, 3.0])
def test_no_jumps_and_a_flat_diffusion_reduce_to_black_scholes(sigma, maturity) -> None:
    reference = np.asarray(
        fast_black("c", FORWARD, STRIKES, maturity, 0.0, sigma, return_as="numpy"), dtype=float
    )
    produced = bates_price(
        forward=FORWARD, strike=STRIKES, maturity=maturity, **flat_diffusion(sigma)
    )
    np.testing.assert_allclose(produced, reference, atol=2e-6, rtol=0.0)


# --- the two formulations agree ------------------------------------------------


@pytest.mark.parametrize("heston", list(HESTON_SETS.values()), ids=list(HESTON_SETS))
@pytest.mark.parametrize("jumps", list(JUMP_SETS.values()), ids=list(JUMP_SETS))
@pytest.mark.parametrize("maturity", MATURITIES)
def test_lewis_and_gatheral_agree(heston, jumps, maturity) -> None:
    """Different integrals with differently-behaved integrands; the agreement is
    what this module offers in place of a published reference table.

    Taken at **1536 nodes rather than the default 768**, and that is a
    measurement rather than a concession. The Gatheral integrand has a removable
    singularity at the origin and is the harder of the two to resolve: at 768 it
    is accurate to 4.1e-7 of the forward on jump-heavy parameters where Lewis is
    already at 3.2e-9, and at 1536 it reaches 4.1e-10. Both routes converge to
    the *same* value -- 1e-11 apart at 7680 nodes -- so the gap at the default
    count is Gatheral's quadrature error, not a disagreement about the model.
    Loosening the tolerance instead would have hidden which of the two was
    wrong.
    """
    common = {
        "forward": FORWARD,
        "strike": STRIKES,
        "maturity": maturity,
        "n_nodes": 2 * BATES_QUADRATURE_NODES,
        **heston,
        **jumps,
    }
    lewis = bates_price(**common, formulation="lewis")
    gatheral = bates_price(**common, formulation="gatheral")
    np.testing.assert_allclose(lewis, gatheral, atol=1e-7, rtol=0.0)


def test_the_two_routes_converge_to_one_another() -> None:
    """The claim the tolerance above rests on: the gap is quadrature, not model.

    At ten times the default node count the two formulations agree to about
    1e-11 on the hardest case swept, which is the float64 floor for a price of
    this size.
    """
    common = {
        "forward": FORWARD,
        "strike": STRIKES,
        "maturity": 0.25,
        "n_nodes": 10 * BATES_QUADRATURE_NODES,
        **HESTON_SETS["heavy"],
        **JUMP_SETS["wild"],
    }
    np.testing.assert_allclose(
        bates_price(**common, formulation="lewis"),
        bates_price(**common, formulation="gatheral"),
        atol=1e-9,
        rtol=0.0,
    )


def test_the_gatheral_route_is_the_weaker_one_at_the_default_node_count() -> None:
    """Recorded so the default is understood as a choice about Lewis.

    If this ever stops holding -- because the Gatheral integrand was improved,
    or because the default count rose -- the docstring on
    ``BATES_QUADRATURE_NODES`` is out of date and says so here.
    """
    common = {
        "forward": FORWARD,
        "strike": STRIKES,
        "maturity": 0.02,
        **HESTON_SETS["fast"],
        **JUMP_SETS["wild"],
    }

    def error(formulation: str) -> float:
        coarse = bates_price(**common, formulation=formulation)
        fine = bates_price(**common, formulation=formulation, n_nodes=7680)
        return float(np.max(np.abs(coarse - fine)))

    assert error("lewis") < 1e-6
    assert error("gatheral") > 10.0 * error("lewis")


# --- convergence ---------------------------------------------------------------


@pytest.mark.parametrize("heston", list(HESTON_SETS.values()), ids=list(HESTON_SETS))
@pytest.mark.parametrize("jumps", list(JUMP_SETS.values()), ids=list(JUMP_SETS))
def test_the_default_node_count_has_converged(heston, jumps) -> None:
    """Measured against ten times the default node count over the parameter grid."""
    for maturity in MATURITIES:
        common = {
            "forward": FORWARD,
            "strike": STRIKES,
            "maturity": maturity,
            **heston,
            **jumps,
        }
        coarse = bates_price(**common, n_nodes=BATES_QUADRATURE_NODES)
        fine = bates_price(**common, n_nodes=10 * BATES_QUADRATURE_NODES)
        np.testing.assert_allclose(coarse, fine, atol=1e-6, rtol=0.0)


def test_the_quadrature_scale_widens_with_the_jumps() -> None:
    """The reason the scale was re-derived rather than inherited.

    A jump-heavy short-dated integrand is wider than its diffusion alone
    implies, and a node set built for the diffusion resolves it badly. With the
    jump term removed, the ``wild`` set at a two-week maturity is visibly wrong
    at the default node count.
    """
    from fast_vollib.pricing.bates import _quadrature_scale

    diffusion_only = _quadrature_scale(0.02, 0.04, 0.04, 0.0, 0.0, 0.0)
    with_jumps = _quadrature_scale(0.02, 0.04, 0.04, 5.0, -0.02, 0.45)
    assert with_jumps < diffusion_only


def test_the_quadrature_scale_is_heston_s_own_when_there_are_no_jumps() -> None:
    """Bit for bit, which is what makes the Heston reduction bitwise."""
    from fast_vollib.pricing.bates import _quadrature_scale
    from fast_vollib.pricing.heston import _quadrature_scale as heston_scale

    for maturity in MATURITIES:
        for name, parameters in HESTON_SETS.items():
            assert _quadrature_scale(
                maturity, parameters["v0"], parameters["theta"], 0.0, 0.0, 0.0
            ) == heston_scale(maturity, parameters), (name, maturity)


# --- the small-volatility limit -------------------------------------------------


def test_the_heston_transform_is_stable_as_vol_of_vol_approaches_zero() -> None:
    """The rationalized transform converges to the constant-variance price."""
    reference = np.asarray(
        fast_black("c", FORWARD, STRIKES, 1.0, 0.0, 0.2, return_as="numpy"), dtype=float
    )

    def error(vol_of_vol: float) -> float:
        priced = bates_price(
            forward=FORWARD,
            strike=STRIKES,
            maturity=1.0,
            v0=0.04,
            kappa=1.0,
            theta=0.04,
            vol_of_vol=vol_of_vol,
            rho=0.0,
        )
        return float(np.max(np.abs(priced - reference)))

    assert error(LIMIT_VOL_OF_VOL) < 1e-6
    assert error(1e-8) < 1e-8
    assert error(1e-12) < 1e-8


# --- ordinary behaviour --------------------------------------------------------


@pytest.mark.parametrize("jumps", list(JUMP_SETS.values()), ids=list(JUMP_SETS))
def test_jumps_raise_the_price_of_an_out_of_the_money_option(jumps) -> None:
    """Compensated, so the change is the distribution's shape rather than drift."""
    common = {"forward": FORWARD, "maturity": 1.0, **HESTON_SETS["moderate"]}
    wing = np.array([60.0, 180.0])
    with_jumps = bates_price(**common, strike=wing, **jumps)
    without = bates_price(**common, strike=wing)
    assert np.all(with_jumps >= without - 1e-12)
    assert np.any(with_jumps > without * 1.001)


def test_put_call_parity_is_exact_on_the_forward() -> None:
    common = {
        "forward": FORWARD,
        "strike": STRIKES,
        "maturity": 1.0,
        "discount": 0.97,
        **HESTON_SETS["moderate"],
        **JUMP_SETS["mild"],
    }
    calls = bates_price(**common, is_call=True)
    puts = bates_price(**common, is_call=False)
    np.testing.assert_allclose(calls - puts, 0.97 * (FORWARD - STRIKES), rtol=1e-13)


def test_prices_stay_inside_their_no_arbitrage_bounds() -> None:
    for jumps in JUMP_SETS.values():
        for maturity in MATURITIES:
            priced = bates_price(
                forward=FORWARD,
                strike=STRIKES,
                maturity=maturity,
                **HESTON_SETS["heavy"],
                **jumps,
            )
            assert np.all(priced >= np.maximum(FORWARD - STRIKES, 0.0) - 1e-8)
            assert np.all(priced <= FORWARD + 1e-8)


def test_it_broadcasts_forward_strike_and_maturity_together() -> None:
    strikes = STRIKES[:, None]
    maturities = np.array(MATURITIES)[None, :]
    priced = bates_price(
        forward=FORWARD,
        strike=strikes,
        maturity=maturities,
        **HESTON_SETS["moderate"],
        **JUMP_SETS["mild"],
    )
    assert priced.shape == (len(STRIKES), len(MATURITIES))
    for i, strike in enumerate(STRIKES):
        for j, maturity in enumerate(MATURITIES):
            single = bates_price(
                forward=FORWARD,
                strike=strike,
                maturity=maturity,
                **HESTON_SETS["moderate"],
                **JUMP_SETS["mild"],
            )
            assert priced[i, j] == pytest.approx(float(single), rel=1e-14)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"formulation": "carr_madan"}, "formulation must be one of"),
        ({"n_nodes": 4}, "n_nodes must be at least 8"),
        ({"strike": -1.0}, "strictly positive"),
        ({"forward": 0.0}, "strictly positive"),
        ({"maturity": 0.0}, "maturity must be strictly positive"),
    ],
)
def test_an_invalid_request_is_refused(kwargs, message) -> None:
    common = {
        "forward": FORWARD,
        "strike": 100.0,
        "maturity": 1.0,
        **HESTON_SETS["moderate"],
        **JUMP_SETS["mild"],
    }
    common.update(kwargs)
    with pytest.raises(ValueError, match=message):
        bates_price(**common)
