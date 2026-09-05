"""Properties of the CIR kernel that no reference table can express.

``test_reference_fixtures.py`` pins the kernel to fifty digits at twenty-six
points.  A table cannot say that the affine coefficients solve the differential
equation they are supposed to solve, that the deterministic branch is the limit
of the general one rather than a different model, or that the whole thing
differentiates on torch.  Those are here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_vollib.rates import (
    RateValidationError,
    cir_affine_coefficients,
    cir_discount_factor,
    cir_instantaneous_forward_rate,
    cir_zero_rate,
)

BOOK = {"kappa": 0.3, "theta": 0.04, "volatility": 0.1, "initial_rate": 0.04}
ROUGH = {"kappa": 0.5, "theta": 0.06, "volatility": 0.9, "initial_rate": 0.03}
FAST = {"kappa": 3.0, "theta": 0.02, "volatility": 0.4, "initial_rate": 0.05}
SETS = [BOOK, ROUGH, FAST]
IDS = ["book", "rough", "fast"]

MATURITIES = (0.0, 1e-6, 0.01, 0.25, 1.0, 5.0, 30.0, 100.0)

#: Step for the central differences below. Their two error terms -- truncation,
#: of order ``h**2``, and roundoff, of order ``eps / h`` on a quantity of order
#: one -- balance near ``eps ** (1/3) ~ 6e-6``; ``1e-5`` puts both near 2e-11.
DERIVATIVE_STEP = 1e-5

#: Absolute noise floor left by that difference quotient. Not a fitted
#: tolerance: it is what float64 leaves, and it matters because ``dB/dtau``
#: decays to about 1e-7 at long maturities, where a purely relative bound would
#: be asking the quotient for more digits than it has.
DERIVATIVE_FLOOR = 1e-9


# --- the limits, which are exact rather than approached ------------------------


@pytest.mark.parametrize("parameters", SETS, ids=IDS)
def test_a_zero_maturity_is_worth_exactly_one(parameters) -> None:
    """Not "to within a tolerance": a payment due now is worth its face value.

    Worth stating as its own test because the kernel has no branch for it. Every
    term of the exponent vanishes on its own at ``tau = 0``, and this is what
    says so.
    """
    assert float(cir_discount_factor(**parameters, maturity=0.0)) == 1.0


@pytest.mark.parametrize("parameters", SETS, ids=IDS)
def test_the_affine_coefficients_vanish_at_zero_maturity(parameters) -> None:
    log_a, b = cir_affine_coefficients(
        kappa=parameters["kappa"],
        theta=parameters["theta"],
        volatility=parameters["volatility"],
        maturity=0.0,
    )
    assert float(log_a) == 0.0
    assert float(b) == 0.0


@pytest.mark.parametrize("parameters", SETS, ids=IDS)
def test_the_zero_rate_at_zero_maturity_is_the_short_rate(parameters) -> None:
    assert float(cir_zero_rate(**parameters, maturity=0.0)) == parameters["initial_rate"]


@pytest.mark.parametrize("parameters", SETS, ids=IDS)
def test_the_instantaneous_forward_at_zero_maturity_is_the_short_rate(parameters) -> None:
    """``f(0, 0) = r_0`` is the definition of the short rate."""
    got = float(cir_instantaneous_forward_rate(**parameters, maturity=0.0))
    assert got == pytest.approx(parameters["initial_rate"], abs=1e-15)


# --- the vanishing-volatility limit --------------------------------------------


@pytest.mark.parametrize("parameters", SETS, ids=IDS)
@pytest.mark.parametrize("maturity", [0.25, 1.0, 30.0])
def test_the_deterministic_branch_is_the_limit_of_the_general_one(parameters, maturity) -> None:
    """The kernel's one branch has to be continuous with what it branches from.

    A branch that returned a different model at exactly zero would be a
    discontinuity disguised as a special case. The general form is evaluated at
    ``volatility = 1e-10``, which is small enough that the limit is reached to
    within a few ulp and large enough that the general code path runs.
    """
    shared = {k: v for k, v in parameters.items() if k != "volatility"}
    deterministic = float(cir_discount_factor(**shared, volatility=0.0, maturity=maturity))
    approached = float(cir_discount_factor(**shared, volatility=1e-10, maturity=maturity))
    assert approached == pytest.approx(deterministic, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("maturity", [0.5, 5.0, 30.0])
def test_the_deterministic_branch_is_the_closed_form_integral(maturity) -> None:
    """Checked against the elementary integral, not against the general form."""
    kappa, theta, r0 = 0.8, 0.03, 0.05
    expected = math.exp(
        -(theta * maturity + (r0 - theta) * (1.0 - math.exp(-kappa * maturity)) / kappa)
    )
    got = float(
        cir_discount_factor(
            kappa=kappa, theta=theta, volatility=0.0, initial_rate=r0, maturity=maturity
        )
    )
    assert got == pytest.approx(expected, rel=1e-15, abs=0.0)


def test_a_vanishing_volatility_does_not_lose_digits_as_it_shrinks() -> None:
    """The whole reason the delta form exists.

    The naive rearrangement forms ``gamma - kappa`` by subtraction and is wrong
    in the third decimal place by ``sigma = 1e-8``. The shipped form converges
    monotonically to the deterministic limit instead.
    """
    shared = {"kappa": 0.3, "theta": 0.04, "initial_rate": 0.04, "maturity": 1.0}
    limit = float(cir_discount_factor(**shared, volatility=0.0))
    errors = [
        abs(float(cir_discount_factor(**shared, volatility=sigma)) - limit)
        for sigma in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
    ]
    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] < 1e-14


# --- the Riccati system the coefficients solve ---------------------------------


@pytest.mark.parametrize("parameters", SETS, ids=IDS)
@pytest.mark.parametrize("maturity", [0.3, 1.7, 12.0])
def test_the_coefficients_solve_the_riccati_system(parameters, maturity) -> None:
    r"""The model-independent oracle.

    ``A`` and ``B`` are defined as the solution of

        dB/dtau   = 1 - kappa B - (sigma^2 / 2) B^2,   B(0) = 0
        dlogA/dtau = -kappa theta B,                   logA(0) = 0

    so differentiating what the kernel returns and substituting is a check that
    does not go through the closed form at all. A transcription error anywhere
    in the algebra breaks it.
    """
    kappa, theta, sigma = parameters["kappa"], parameters["theta"], parameters["volatility"]
    step = DERIVATIVE_STEP

    def coefficients(tau):
        log_a, b = cir_affine_coefficients(kappa=kappa, theta=theta, volatility=sigma, maturity=tau)
        return float(log_a), float(b)

    log_a_plus, b_plus = coefficients(maturity + step)
    log_a_minus, b_minus = coefficients(maturity - step)
    _log_a, b = coefficients(maturity)

    d_b = (b_plus - b_minus) / (2.0 * step)
    d_log_a = (log_a_plus - log_a_minus) / (2.0 * step)

    assert d_b == pytest.approx(
        1.0 - kappa * b - 0.5 * sigma * sigma * b * b, rel=1e-6, abs=DERIVATIVE_FLOOR
    )
    assert d_log_a == pytest.approx(-kappa * theta * b, rel=1e-6, abs=DERIVATIVE_FLOOR)


@pytest.mark.parametrize("parameters", SETS, ids=IDS)
@pytest.mark.parametrize("maturity", [0.3, 1.7, 12.0])
def test_the_instantaneous_forward_is_the_derivative_of_the_log_price(parameters, maturity) -> None:
    """``f = -d log P / dT``, computed analytically, checked by differencing."""
    step = DERIVATIVE_STEP
    up = math.log(float(cir_discount_factor(**parameters, maturity=maturity + step)))
    down = math.log(float(cir_discount_factor(**parameters, maturity=maturity - step)))
    numerical = -(up - down) / (2.0 * step)
    assert float(cir_instantaneous_forward_rate(**parameters, maturity=maturity)) == pytest.approx(
        numerical, rel=1e-6, abs=DERIVATIVE_FLOOR
    )


# --- shape of the curve --------------------------------------------------------


@pytest.mark.parametrize("parameters", SETS, ids=IDS)
def test_the_price_is_an_admissible_discount_factor_everywhere(parameters) -> None:
    for maturity in MATURITIES:
        price = float(cir_discount_factor(**parameters, maturity=maturity))
        assert 0.0 < price <= 1.0, (parameters, maturity, price)


@pytest.mark.parametrize("parameters", SETS, ids=IDS)
def test_the_price_decreases_strictly_in_maturity(parameters) -> None:
    """A non-negative short rate cannot make a longer bond worth more."""
    prices = [float(cir_discount_factor(**parameters, maturity=t)) for t in MATURITIES]
    assert all(later < earlier for earlier, later in zip(prices, prices[1:])), prices


def test_a_higher_initial_rate_gives_a_lower_price() -> None:
    """``B(tau) > 0``, so the price is strictly decreasing in ``r_0``."""
    shared = {"kappa": 0.3, "theta": 0.04, "volatility": 0.1, "maturity": 5.0}
    prices = [float(cir_discount_factor(**shared, initial_rate=r)) for r in (0.0, 0.02, 0.05, 0.2)]
    assert all(later < earlier for earlier, later in zip(prices, prices[1:])), prices


def test_the_price_is_bitwise_deterministic() -> None:
    first = cir_discount_factor(**BOOK, maturity=7.5)
    second = cir_discount_factor(**BOOK, maturity=7.5)
    assert float(first) == float(second)


# --- validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kappa": 0.0}, "kappa must be strictly positive"),
        ({"kappa": -1.0}, "kappa must be strictly positive"),
        ({"theta": -0.01}, "theta must be non-negative"),
        ({"volatility": -0.1}, "volatility must be non-negative"),
        ({"initial_rate": -0.01}, "initial_rate must be non-negative"),
        ({"maturity": -1.0}, "maturity must be non-negative"),
        ({"kappa": float("nan")}, "kappa must be finite"),
        ({"maturity": float("inf")}, "maturity must be finite"),
        ({"kappa": True}, "kappa must be a real number, not a bool"),
        ({"theta": 1 + 2j}, "theta must be a real number, not complex"),
        ({"volatility": "0.1"}, "volatility must be a real number or a scalar array"),
    ],
)
def test_invalid_parameters_are_refused(kwargs, message) -> None:
    call = {**BOOK, "maturity": 1.0, **kwargs}
    with pytest.raises(RateValidationError, match=message):
        cir_discount_factor(**call)


def test_a_term_structure_of_parameters_is_refused_rather_than_broadcast() -> None:
    """One value per name. A vector kappa is a different model, not a batch."""
    with pytest.raises(RateValidationError, match="must be a scalar"):
        cir_discount_factor(**{**BOOK, "kappa": np.array([0.3, 0.5])}, maturity=1.0)


# --- backends ------------------------------------------------------------------


def test_numpy_scalar_inputs_are_accepted() -> None:
    got = cir_discount_factor(
        kappa=np.float64(0.3),
        theta=np.float64(0.04),
        volatility=np.float64(0.1),
        initial_rate=np.float64(0.04),
        maturity=np.float64(1.0),
    )
    assert float(got) == pytest.approx(0.9608408119927458, rel=1e-15)


def test_the_kernel_carries_a_torch_gradient_back_to_its_parameters() -> None:
    """The claim that the kernel is array-API native, tested rather than stated.

    A bond price that differentiates with respect to ``kappa`` is what makes
    this usable inside a calibration loss; the Fourier pricers in
    :mod:`fast_vollib.pricing` cannot do it.
    """
    torch = pytest.importorskip("torch")
    kappa = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    initial_rate = torch.tensor(0.04, dtype=torch.float64, requires_grad=True)
    price = cir_discount_factor(
        kappa=kappa,
        theta=torch.tensor(0.04, dtype=torch.float64),
        volatility=torch.tensor(0.1, dtype=torch.float64),
        initial_rate=initial_rate,
        maturity=torch.tensor(5.0, dtype=torch.float64),
    )
    assert torch.is_tensor(price)
    assert price.requires_grad
    price.backward()
    assert kappa.grad is not None and torch.isfinite(kappa.grad)
    # dP/dr0 = -B P, and B > 0, so a higher short rate lowers the price.
    assert float(initial_rate.grad) < 0.0
    _log_a, b = cir_affine_coefficients(kappa=0.3, theta=0.04, volatility=0.1, maturity=5.0)
    assert float(initial_rate.grad) == pytest.approx(-float(b) * float(price.detach()), rel=1e-10)


def test_the_kernel_matches_across_numpy_and_torch() -> None:
    torch = pytest.importorskip("torch")
    host = float(cir_discount_factor(**BOOK, maturity=3.0))
    native = float(
        cir_discount_factor(
            **{k: torch.tensor(v, dtype=torch.float64) for k, v in BOOK.items()},
            maturity=torch.tensor(3.0, dtype=torch.float64),
        )
    )
    assert native == pytest.approx(host, rel=1e-14)


def test_the_kernel_runs_under_jax_jit_and_grad() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def price(kappa):
        return cir_discount_factor(
            kappa=kappa,
            theta=jnp.float64(0.04) if jax.config.read("jax_enable_x64") else 0.04,
            volatility=0.1,
            initial_rate=0.04,
            maturity=5.0,
        )

    value = jax.jit(price)(jnp.asarray(0.3))
    assert float(value) == pytest.approx(
        float(cir_discount_factor(**{**BOOK, "maturity": 5.0})), rel=1e-5
    )
    gradient = jax.grad(price)(jnp.asarray(0.3))
    assert math.isfinite(float(gradient))
