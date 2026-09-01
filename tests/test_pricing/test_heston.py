"""Heston Fourier pricing: exact identities, two formulations, and a stated accuracy."""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib.models import fast_black
from fast_vollib.pricing import (
    DEFAULT_QUADRATURE_NODES,
    heston_call_price,
    heston_characteristic_function,
    heston_price,
)

#: A moderate parameter set (Feller ratio 1.78) and a heavy-tailed one
#: (Feller 0.07), so every accuracy claim is made where the integrand is hard
#: rather than only where it is easy.
MODERATE = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "vol_of_vol": 0.3, "rho": -0.7}
HEAVY = {"v0": 0.09, "kappa": 0.5, "theta": 0.06, "vol_of_vol": 0.9, "rho": -0.9}
FAST = {"v0": 0.01, "kappa": 5.0, "theta": 0.02, "vol_of_vol": 0.2, "rho": 0.4}

FORWARD = 100.0
STRIKES = np.array([60.0, 80.0, 100.0, 130.0, 180.0])
MATURITIES = (0.02, 0.05, 0.25, 1.0, 3.0, 10.0, 30.0)

#: Agreement required between the two Fourier formulations at the default node
#: count.  Measured, not chosen: the worst case over the swept parameter sets,
#: maturities, and strikes is 4e-7 of the forward, at vol-of-vol 0.9 against a
#: Feller ratio of 0.07, and every ordinary case is a thousand times better.
FORMULATION_TOL = 1e-6

#: An identity that holds exactly in exact arithmetic, evaluated in float64
#: through a handful of complex exponentials and logarithms.
EXACT_TOL = 1e-12


# --- the characteristic function ----------------------------------------------


@pytest.mark.parametrize("parameters", [MODERATE, HEAVY, FAST], ids=["moderate", "heavy", "fast"])
@pytest.mark.parametrize("maturity", [0.05, 1.0, 30.0], ids=["short", "one-year", "long"])
def test_the_characteristic_function_is_one_at_the_origin(parameters, maturity) -> None:
    value = heston_characteristic_function(0.0, maturity=maturity, **parameters)
    assert abs(complex(value) - 1.0) < EXACT_TOL


@pytest.mark.parametrize("parameters", [MODERATE, HEAVY, FAST], ids=["moderate", "heavy", "fast"])
@pytest.mark.parametrize("maturity", [0.05, 1.0, 30.0], ids=["short", "one-year", "long"])
def test_the_forward_is_a_martingale(parameters, maturity) -> None:
    """``phi(-i) = E[F_T]/F_0 = 1``.

    The single most sensitive check on the exponent's algebra: a sign error
    anywhere in ``C`` or ``D`` moves this away from one, and it is the identity
    that a plausible-looking but wrong branch choice fails first.
    """
    value = heston_characteristic_function(-1j, maturity=maturity, **parameters)
    assert abs(complex(value) - 1.0) < EXACT_TOL


@pytest.mark.parametrize("parameters", [MODERATE, HEAVY], ids=["moderate", "heavy"])
def test_the_characteristic_function_is_conjugate_symmetric(parameters) -> None:
    u = np.linspace(0.05, 20.0, 41)
    positive = heston_characteristic_function(u, maturity=1.5, **parameters)
    negative = heston_characteristic_function(-u, maturity=1.5, **parameters)
    np.testing.assert_allclose(negative, np.conj(positive), rtol=0, atol=0)


def test_the_first_moment_matches_the_exact_integrated_variance() -> None:
    """``-i phi'(0) = E[Y_T] = -E[int v]/2``, with the integral in closed form.

    ``E[int_0^T v_s ds] = theta T + (v0 - theta)(1 - e^{-kappa T})/kappa``.  The
    derivative is taken by a central difference with ``h = 1e-4``: the O(h^2)
    truncation is 1e-8 and the cancellation floor is 1e-12/h = 1e-8, so 1e-7 is
    the honest band and it is four orders tighter than any algebra error.
    """
    maturity = 2.0
    parameters = {"v0": 0.05, "kappa": 1.5, "theta": 0.03, "vol_of_vol": 0.4, "rho": -0.6}
    integrated = (
        parameters["theta"] * maturity
        + (parameters["v0"] - parameters["theta"])
        * (1.0 - np.exp(-parameters["kappa"] * maturity))
        / parameters["kappa"]
    )
    h = 1e-4
    up = heston_characteristic_function(h, maturity=maturity, **parameters)
    down = heston_characteristic_function(-h, maturity=maturity, **parameters)
    derivative = (complex(up) - complex(down)) / (2 * h)
    assert abs((-1j * derivative).real - (-0.5 * integrated)) < 1e-7


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"maturity": 0.0}, "maturity must be strictly positive"),
        ({"vol_of_vol": 0.0}, "vol_of_vol must be strictly positive"),
    ],
)
def test_degenerate_characteristic_function_inputs_are_refused(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        heston_characteristic_function(1.0, **{"maturity": 1.0, **MODERATE, **kwargs})


# --- the price ----------------------------------------------------------------


def test_the_two_formulations_agree_across_the_swept_range() -> None:
    worst = 0.0
    for parameters in (MODERATE, HEAVY, FAST):
        for maturity in MATURITIES:
            lewis = heston_price(forward=FORWARD, strike=STRIKES, maturity=maturity, **parameters)
            gatheral = heston_price(
                forward=FORWARD,
                strike=STRIKES,
                maturity=maturity,
                formulation="gatheral",
                **parameters,
            )
            worst = max(worst, float(np.max(np.abs(lewis - gatheral))))
    assert worst < FORMULATION_TOL, worst


def test_the_quadrature_converges_in_the_node_count() -> None:
    """Gauss-Legendre on the mapped half-line converges geometrically.

    Measured against a 4096-node reference at the hardest parameter set, so the
    claim in :data:`DEFAULT_QUADRATURE_NODES` is a fact about this code rather
    than an aspiration.
    """
    reference = heston_price(forward=FORWARD, strike=STRIKES, maturity=1.0, n_nodes=4096, **HEAVY)
    errors = []
    for n_nodes in (128, 256, 512, DEFAULT_QUADRATURE_NODES):
        priced = heston_price(
            forward=FORWARD, strike=STRIKES, maturity=1.0, n_nodes=n_nodes, **HEAVY
        )
        errors.append(float(np.max(np.abs(priced - reference))))
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1e-8


def test_the_small_vol_of_vol_zero_correlation_limit_is_black() -> None:
    """At ``rho = 0`` the price approaches Black's at second order in vol-of-vol.

    With zero correlation the leading correction to the Black price is O(xi^2),
    so halving the vol-of-vol should quarter the error.  A first-order
    discrepancy would mean the drift or the correlation term is wrong, and no
    tolerance would hide it -- which is why the *rate* is asserted, not a band.
    """
    maturity = 1.5
    errors = []
    for vol_of_vol in (4e-3, 2e-3, 1e-3):
        parameters = {
            "v0": 0.05,
            "kappa": 1.5,
            "theta": 0.03,
            "vol_of_vol": vol_of_vol,
            "rho": 0.0,
        }
        integrated = 0.03 * maturity + (0.05 - 0.03) * (1 - np.exp(-1.5 * maturity)) / 1.5
        black = fast_black(
            "c",
            np.full(STRIKES.size, FORWARD),
            STRIKES,
            np.full(STRIKES.size, maturity),
            np.zeros(STRIKES.size),
            np.full(STRIKES.size, np.sqrt(integrated / maturity)),
            return_as="numpy",
            backend="numpy",
        )
        priced = heston_price(forward=FORWARD, strike=STRIKES, maturity=maturity, **parameters)
        errors.append(float(np.max(np.abs(priced - black))))
    ratios = [errors[index] / errors[index + 1] for index in range(len(errors) - 1)]
    assert all(3.0 < ratio < 5.0 for ratio in ratios), (errors, ratios)


@pytest.mark.parametrize("parameters", [MODERATE, HEAVY], ids=["moderate", "heavy"])
def test_put_call_parity_holds_exactly(parameters) -> None:
    # Parity is how the put is produced, so this asserts the identity is applied
    # to the undiscounted forward price and that the discount is a clean factor.
    for discount in (1.0, 0.93):
        call = heston_price(
            forward=FORWARD,
            strike=STRIKES,
            maturity=1.25,
            discount=discount,
            **parameters,
        )
        put = heston_price(
            forward=FORWARD,
            strike=STRIKES,
            maturity=1.25,
            is_call=False,
            discount=discount,
            **parameters,
        )
        np.testing.assert_allclose(call - put, discount * (FORWARD - STRIKES), rtol=0, atol=0)


@pytest.mark.parametrize("parameters", [MODERATE, HEAVY, FAST], ids=["moderate", "heavy", "fast"])
def test_call_prices_obey_the_no_arbitrage_bounds(parameters) -> None:
    strikes = np.linspace(40.0, 250.0, 60)
    for maturity in (0.05, 1.0, 10.0):
        priced = heston_price(forward=FORWARD, strike=strikes, maturity=maturity, **parameters)
        intrinsic = np.maximum(FORWARD - strikes, 0.0)
        assert bool(np.all(priced >= intrinsic - 1e-9))
        assert bool(np.all(priced <= FORWARD + 1e-9))
        # Monotone decreasing in strike, and the slope never steeper than -1.
        slope = np.diff(priced) / np.diff(strikes)
        assert bool(np.all(slope <= 1e-9))
        assert bool(np.all(slope >= -1.0 - 1e-9))


def test_the_price_is_continuous_and_increasing_in_maturity() -> None:
    """The little-trap branch is what makes this true past a year.

    The textbook characteristic function is algebraically identical and wraps the
    complex logarithm at long maturities, which shows up here as a jump in an
    otherwise smooth curve.  A dense sweep to thirty years catches that.
    """
    maturities = np.linspace(0.02, 30.0, 300)
    priced = np.array(
        [
            float(heston_price(forward=FORWARD, strike=100.0, maturity=T, **HEAVY))
            for T in maturities
        ]
    )
    assert bool(np.all(np.diff(priced) > 0.0))
    # Smooth as well as increasing: an at-the-money price grows like the square
    # root of maturity, so its increments on a uniform grid fall monotonically.
    # A wrapped logarithm shows up as one increment larger than its predecessor,
    # which this catches and a band on the largest increment would not -- the
    # genuine first increment is thirty times the median.
    steps = np.diff(priced)
    assert bool(np.all(np.diff(steps) < 0.0))


def test_pricing_is_bitwise_deterministic() -> None:
    first = heston_price(forward=FORWARD, strike=STRIKES, maturity=1.0, **MODERATE)
    second = heston_price(forward=FORWARD, strike=STRIKES, maturity=1.0, **MODERATE)
    np.testing.assert_array_equal(first, second)


def test_a_whole_smile_costs_one_quadrature_per_maturity() -> None:
    # Vectorised over strikes at a fixed maturity: the answer must not depend on
    # whether the strikes arrived together or one at a time.
    together = heston_price(forward=FORWARD, strike=STRIKES, maturity=0.75, **MODERATE)
    apart = np.array(
        [
            float(heston_price(forward=FORWARD, strike=strike, maturity=0.75, **MODERATE))
            for strike in STRIKES
        ]
    )
    # Not bitwise, and the tolerance is ABSOLUTE for a reason. The quadrature is
    # a matrix-vector product over 768 nodes whose BLAS reduction order depends
    # on the shape it was called with, and the price is F minus that sum, so the
    # two orders differ by roughly sqrt(768) * eps * F -- an error on the scale
    # of the forward, not of the price. A relative tolerance would therefore be
    # met at the money and violated by six orders of magnitude on a 8e-6 wing
    # price, which is exactly the accuracy characteristic
    # PRICE_NOISE_PER_UNIT_FORWARD documents and the Heston surface's vega guard
    # exists to respect. Measured worst case here: 1.7e-13 at F = 100.
    np.testing.assert_allclose(together, apart, rtol=0.0, atol=1e-14 * FORWARD)


def test_broadcasting_matches_elementwise_evaluation() -> None:
    strikes = np.array([[80.0, 100.0], [120.0, 140.0]])
    maturities = np.array([[0.5], [2.0]])
    grid = heston_price(forward=FORWARD, strike=strikes, maturity=maturities, **MODERATE)
    assert grid.shape == (2, 2)
    for row in range(2):
        for column in range(2):
            single = heston_price(
                forward=FORWARD,
                strike=strikes[row, column],
                maturity=maturities[row, 0],
                **MODERATE,
            )
            # Absolute, scaled by the forward, for the reason given above.
            np.testing.assert_allclose(grid[row, column], single, rtol=0.0, atol=1e-14 * FORWARD)


def test_the_call_helper_matches_the_general_entry_point() -> None:
    np.testing.assert_array_equal(
        heston_call_price(forward=FORWARD, strike=STRIKES, maturity=1.0, **MODERATE),
        heston_price(forward=FORWARD, strike=STRIKES, maturity=1.0, **MODERATE),
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"formulation": "carr-madan"}, "formulation must be one of"),
        ({"n_nodes": 4}, "n_nodes must be at least 8"),
        ({"strike": -1.0}, "strictly positive"),
        ({"maturity": 0.0}, "maturity must be strictly positive"),
    ],
)
def test_invalid_pricing_inputs_are_refused(kwargs, message) -> None:
    base = {"forward": FORWARD, "strike": 100.0, "maturity": 1.0, **MODERATE}
    with pytest.raises(ValueError, match=message):
        heston_price(**{**base, **kwargs})


# --- against the process ------------------------------------------------------


def test_the_fourier_price_agrees_with_monte_carlo_from_the_process() -> None:
    """The two are independent routes to the same number.

    The Fourier price inverts the characteristic function; the Monte Carlo price
    samples the SDE.  They share the parameters and nothing else, so agreement
    within the sampling error is real evidence.  The band is 4 standard errors of
    the Monte Carlo mean, computed from the sample itself -- not a fixed
    tolerance -- so it tightens automatically if the path count is ever raised.
    """
    from fast_vollib.processes import Heston

    parameters = MODERATE
    maturity = 1.0
    n_paths = 200_000
    process = Heston(
        kappa=parameters["kappa"],
        theta=parameters["theta"],
        vol_of_vol=parameters["vol_of_vol"],
        rho=parameters["rho"],
        drift=0.0,
    )
    paths = process.sample(
        initial_state={"spot": FORWARD, "variance": parameters["v0"]},
        time_grid=np.linspace(0.0, maturity, 65),
        n_paths=n_paths,
        rng=20260831,
    )
    terminal = paths[:, -1, 0]
    strikes = np.array([85.0, 100.0, 115.0])
    payoffs = np.maximum(terminal[:, None] - strikes[None, :], 0.0)
    monte_carlo = payoffs.mean(axis=0)
    stderr = payoffs.std(axis=0, ddof=1) / np.sqrt(n_paths)
    fourier = heston_price(forward=FORWARD, strike=strikes, maturity=maturity, **parameters)
    np.testing.assert_array_less(np.abs(monte_carlo - fourier), 4.0 * stderr)
