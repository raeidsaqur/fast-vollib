r"""``Psi(s) = E[exp(-s * int_0^tau r_u du)]``, checked five independent ways.

The transform is the bond formula with an ``s`` threaded through it, and that
is exactly what makes it dangerous: *every* wrong transcription still prices a
bond correctly, because at ``s = 1`` the missing factor is one.  So the bond
check is here (it must hold, bitwise) but it is not the evidence.  The evidence
is four things that are blind to ``s = 1``:

``rescaling``
    For real ``s > 0``, ``s * r_t`` is itself a CIR process, with parameters
    ``(kappa, s*theta, volatility*sqrt(s))`` started at ``s*r_0`` -- so
    ``Psi(s)`` is a *bond price under the library's existing, frozen kernel*
    at different arguments.  No new formula is involved on the reference side.

``the Riccati system``
    ``dB/dtau = s - kappa*B - (volatility**2/2)*B**2`` and
    ``d(log A)/dtau = -kappa*theta*B``, which the closed form must satisfy at
    complex ``s`` as well as real.  Finite differences of the shipped function
    against its own derivative is the model-independent oracle: it knows
    nothing about how the coefficients were derived, and a wrong branch shows
    up as a residual that does not shrink.

``the small-tau expansion``
    ``log A_s = -kappa*theta*s*tau**2/2 + O(tau**3)``, which holds only because
    two ``O(tau)`` terms cancel -- and they cancel only with the ``s`` in the
    first term.  ``test_the_expansion_detects_a_dropped_s`` computes the wrong
    variant here and shows it fails, so the check is not vacuous.

``the deterministic limit``
    At ``volatility == 0`` the integral is deterministic and the exponent is
    the bond exponent times ``s`` exactly.

Continuity in maturity is the branch gate, and the module claims something
stronger than "measured continuous": that ``1 + x_s`` is confined to a disk
excluding the origin, so the principal logarithm *cannot* wind.  Both the
conclusion and the geometric premise are asserted below.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

from fast_vollib.rates import (
    RateValidationError,
    cir_affine_coefficients,
    cir_discount_factor,
    cir_integrated_rate_coefficients,
    cir_integrated_rate_transform,
)

#: Curves spanning the shapes that matter: Feller-satisfying and violating, a
#: near-deterministic vol-of-rate, a fast and a slow mean reversion.
CURVES = [
    {"kappa": 0.3, "theta": 0.04, "volatility": 0.1},
    {"kappa": 2.0, "theta": 0.05, "volatility": 0.6},  # Feller violated
    {"kappa": 0.05, "theta": 0.02, "volatility": 0.02},
    {"kappa": 1.5, "theta": 0.08, "volatility": 1e-6},  # near-deterministic
]
MATURITIES = [0.0, 0.01, 0.25, 1.0, 5.0, 30.0]
INITIAL_RATES = [0.0, 0.01, 0.04, 0.15]

#: The arguments the library's own inversions evaluate at: ``1 - iu`` for the
#: discounted transform, ``1/2 - iu`` for Lewis, ``-iu`` for Gatheral's first
#: integral. All have non-negative real part, which is what keeps
#: ``kappa**2 + 2*volatility**2*s`` off the branch cut.
COMPLEX_ARGUMENTS = [
    1.0 + 0.0j,
    1.0 - 0.75j,
    1.0 - 12.0j,
    0.5 - 3.0j,
    0.0 - 8.0j,
    0.25 + 4.0j,
]


def ids(prefix, values):
    return [f"{prefix}{index}" for index, _ in enumerate(values)]


# --- (a) the bond, bitwise -----------------------------------------------------


@pytest.mark.parametrize("curve", CURVES, ids=ids("curve", CURVES))
@pytest.mark.parametrize("maturity", MATURITIES)
@pytest.mark.parametrize("initial_rate", INITIAL_RATES)
def test_psi_at_one_is_the_bond_price_bitwise(curve, maturity, initial_rate) -> None:
    """Not "close to": equal. The two paths must reduce term for term.

    A tolerance here would hide precisely the rearrangement error this check
    exists to catch, because any plausible rearrangement agrees to 1e-15.
    """
    psi = cir_integrated_rate_transform(1.0, maturity=maturity, initial_rate=initial_rate, **curve)
    bond = cir_discount_factor(maturity=maturity, initial_rate=initial_rate, **curve)
    assert float(psi).hex() == float(bond).hex()


@pytest.mark.parametrize("curve", CURVES, ids=ids("curve", CURVES))
@pytest.mark.parametrize("maturity", MATURITIES)
def test_the_coefficients_at_one_are_the_affine_coefficients_bitwise(curve, maturity) -> None:
    log_a, b = cir_integrated_rate_coefficients(1.0, maturity=maturity, **curve)
    log_a_ref, b_ref = cir_affine_coefficients(maturity=maturity, **curve)
    assert float(log_a).hex() == float(log_a_ref).hex()
    assert float(b).hex() == float(b_ref).hex()


@pytest.mark.parametrize("curve", CURVES, ids=ids("curve", CURVES))
@pytest.mark.parametrize("maturity", MATURITIES)
@pytest.mark.parametrize("initial_rate", INITIAL_RATES)
def test_psi_at_zero_is_exactly_one(curve, maturity, initial_rate) -> None:
    """``gamma_0 = kappa``, ``delta_0 = 0``, so every term vanishes on its own.

    Asserted as an exact equality because it falls out of the algebra rather
    than being special-cased: a branch here would be a second code path with
    nothing to keep it in step.
    """
    psi = cir_integrated_rate_transform(0.0, maturity=maturity, initial_rate=initial_rate, **curve)
    assert float(psi) == 1.0


# --- the rescaling identity: real s against the frozen kernel -------------------


@pytest.mark.parametrize("curve", CURVES, ids=ids("curve", CURVES))
@pytest.mark.parametrize("maturity", MATURITIES)
@pytest.mark.parametrize("s", [0.25, 0.5, 2.0, 7.0])
def test_a_real_argument_is_a_bond_on_the_rescaled_process(curve, maturity, s) -> None:
    r"""``s * r_t`` is CIR with ``(kappa, s*theta, volatility*sqrt(s))``.

    Because :math:`d(sr) = \kappa(s\theta - sr)dt + \sigma\sqrt{s}\sqrt{sr}\,dW`,
    so :math:`E[\exp(-s\int r)] = E[\exp(-\int sr)]` is a bond price on that
    process.  The reference side is the library's existing kernel at different
    arguments, which shares no line with the new code path.
    """
    initial_rate = 0.04
    psi = float(
        cir_integrated_rate_transform(s, maturity=maturity, initial_rate=initial_rate, **curve)
    )
    rescaled = float(
        cir_discount_factor(
            kappa=curve["kappa"],
            theta=s * curve["theta"],
            volatility=curve["volatility"] * math.sqrt(s),
            initial_rate=s * initial_rate,
            maturity=maturity,
        )
    )
    assert psi == pytest.approx(rescaled, rel=1e-13, abs=1e-300)


# --- (b) the small-tau expansion ------------------------------------------------


@pytest.mark.parametrize("curve", CURVES, ids=ids("curve", CURVES))
@pytest.mark.parametrize("s", [1.0, 2.5, 1.0 - 3.0j, 0.5 - 1.0j])
def test_log_a_matches_its_small_tau_expansion(curve, s) -> None:
    r"""``log A_s -> -kappa*theta*s*tau**2/2``, at second order in ``tau``.

    The two :math:`O(\tau)` terms cancel only when the first carries the
    :math:`s`, so the relative error of the quadratic must vanish as
    :math:`\tau \to 0`.  Checked as a *decreasing* sequence rather than at one
    point: a wrong constant would pass a single loose bound.
    """
    errors = []
    for tau in (1e-2, 1e-3, 1e-4):
        log_a, _ = cir_integrated_rate_coefficients(s, maturity=tau, **curve)
        leading = -curve["kappa"] * curve["theta"] * s * tau * tau / 2.0
        errors.append(abs(complex(log_a) - leading) / abs(leading))
    # Each tenfold refinement should cut the relative error by about ten.
    assert errors[0] < 5e-2
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < 0.2 * coarse


@pytest.mark.parametrize("curve", CURVES, ids=ids("curve", CURVES))
def test_the_expansion_detects_a_dropped_s(curve) -> None:
    """The check above is only worth having if the wrong form fails it.

    Reproduced here is a missing transform-argument factor: the simplified first
    term of the bond formula, which lacks the ``s``.  It prices a bond exactly
    and is wrong for every other argument, so this is the shape of the bug the
    expansion exists to catch.
    """
    s = 2.5
    kappa, theta, sigma = curve["kappa"], curve["theta"], curve["volatility"]
    tau = 1e-3
    right = complex(cir_integrated_rate_coefficients(s, maturity=tau, **curve)[0])

    # The wrong variant differs from the right one by exactly the ``s`` that was
    # dropped from the first term, so it is built by algebra from the shipped
    # value rather than by transcribing the formula a second time -- which keeps
    # the *logarithm* identical and isolates the one difference being tested.
    gamma = cmath.sqrt(kappa * kappa + 2.0 * sigma * sigma * s)
    wrong = right + 2.0 * kappa * theta * tau * (s - 1.0) / (gamma + kappa)

    leading = -kappa * theta * s * tau * tau / 2.0
    assert abs(wrong - leading) / abs(leading) > 1.0
    assert abs(right - leading) / abs(leading) < 1e-2


# --- (c) the Riccati system, at real and complex s ------------------------------


@pytest.mark.parametrize("curve", CURVES[:3], ids=ids("curve", CURVES[:3]))
@pytest.mark.parametrize("s", COMPLEX_ARGUMENTS, ids=ids("s", COMPLEX_ARGUMENTS))
@pytest.mark.parametrize("tau", [0.05, 0.5, 2.0, 9.0])
def test_the_coefficients_solve_the_riccati_system(curve, s, tau) -> None:
    r"""Central differences against the ODE the affine form is defined by.

    .. math::

        \frac{dB}{d\tau} = s - \kappa B - \tfrac{1}{2}\sigma^2 B^2,
        \qquad
        \frac{d\log A}{d\tau} = -\kappa\theta B .

    This is the oracle that is independent of the derivation: it is the
    *definition* of the coefficients, and a wrong branch of the square root or
    the logarithm produces a residual that does not shrink with the step.
    """
    kappa, theta, sigma = curve["kappa"], curve["theta"], curve["volatility"]
    h = 1e-5 * max(tau, 1.0)

    def coefficients(t):
        log_a, b = cir_integrated_rate_coefficients(s, maturity=t, **curve)
        return complex(log_a), complex(b)

    log_a, b = coefficients(tau)
    log_a_up, b_up = coefficients(tau + h)
    log_a_down, b_down = coefficients(tau - h)

    db = (b_up - b_down) / (2.0 * h)
    dlog_a = (log_a_up - log_a_down) / (2.0 * h)

    expected_db = s - kappa * b - 0.5 * sigma * sigma * b * b
    expected_dlog_a = -kappa * theta * b

    assert abs(db - expected_db) < 1e-6 * max(1.0, abs(expected_db))
    assert abs(dlog_a - expected_dlog_a) < 1e-6 * max(1.0, abs(expected_dlog_a))


@pytest.mark.parametrize("curve", CURVES[:3], ids=ids("curve", CURVES[:3]))
@pytest.mark.parametrize("s", COMPLEX_ARGUMENTS[1:], ids=ids("s", COMPLEX_ARGUMENTS[1:]))
def test_the_closed_form_agrees_with_integrating_the_ode(curve, s) -> None:
    """A fourth-order integration of the Riccati system from ``tau = 0``.

    Where the finite-difference check is local -- it would not notice a branch
    error that was consistent across a small step -- this one accumulates from
    the initial condition ``B(0) = 0``, so it pins the *global* solution.
    """
    kappa, theta, sigma = curve["kappa"], curve["theta"], curve["volatility"]
    horizon = 5.0
    n_steps = 20000
    h = horizon / n_steps

    def derivative(state):
        b, _ = state
        return (s - kappa * b - 0.5 * sigma * sigma * b * b, -kappa * theta * b)

    def step(state):
        k1 = derivative(state)
        k2 = derivative(tuple(v + 0.5 * h * d for v, d in zip(state, k1)))
        k3 = derivative(tuple(v + 0.5 * h * d for v, d in zip(state, k2)))
        k4 = derivative(tuple(v + h * d for v, d in zip(state, k3)))
        return tuple(
            v + h * (a + 2.0 * bb + 2.0 * c + d) / 6.0
            for v, a, bb, c, d in zip(state, k1, k2, k3, k4)
        )

    state = (0.0 + 0.0j, 0.0 + 0.0j)
    for _ in range(n_steps):
        state = step(state)

    log_a, b = cir_integrated_rate_coefficients(s, maturity=horizon, **curve)
    assert abs(complex(b) - state[0]) < 1e-8 * max(1.0, abs(state[0]))
    assert abs(complex(log_a) - state[1]) < 1e-8 * max(1.0, abs(state[1]))


# --- (d) the deterministic limit ------------------------------------------------


@pytest.mark.parametrize("maturity", MATURITIES)
@pytest.mark.parametrize("s", [1.0, 3.0, 1.0 - 2.0j])
def test_zero_volatility_is_the_deterministic_exponent_times_s(maturity, s) -> None:
    r"""``Psi(s) = exp(-s * (theta*tau + (r_0 - theta)(1 - e^{-kappa tau})/kappa))``.

    The rate is then an ordinary solution of an ODE, so the integral is a
    number rather than a random variable and the transform is its exponential.
    Every deterministic-rate BCC97 configuration is this reduction.
    """
    kappa, theta, initial_rate = 0.8, 0.03, 0.05
    psi = cir_integrated_rate_transform(
        s,
        kappa=kappa,
        theta=theta,
        volatility=0.0,
        initial_rate=initial_rate,
        maturity=maturity,
    )
    integral = (
        theta * maturity + (initial_rate - theta) * (1.0 - math.exp(-kappa * maturity)) / kappa
    )
    assert complex(psi) == pytest.approx(cmath.exp(-s * integral), rel=1e-14, abs=1e-300)


def test_a_vanishing_volatility_approaches_the_deterministic_branch() -> None:
    """The branch is a limit the general form actually reaches, not a patch.

    ``2*kappa*theta/volatility**2`` diverges as the vol-of-rate vanishes, so
    this is the place a naive rearrangement loses every digit; the module's
    ``log1p`` form is what keeps the two sides converging.
    """
    s = 1.0 - 2.0j
    shared = dict(kappa=0.8, theta=0.03, initial_rate=0.05, maturity=3.0)
    exact = complex(cir_integrated_rate_transform(s, volatility=0.0, **shared))
    for volatility, bound in ((1e-4, 1e-8), (1e-6, 1e-12), (1e-8, 1e-15)):
        near = complex(cir_integrated_rate_transform(s, volatility=volatility, **shared))
        assert abs(near - exact) < bound * abs(exact)


# --- (e) continuity in maturity, and why no branch tracking is needed -----------


@pytest.mark.parametrize("curve", CURVES[:3], ids=ids("curve", CURVES[:3]))
@pytest.mark.parametrize("s", COMPLEX_ARGUMENTS, ids=ids("s", COMPLEX_ARGUMENTS))
def test_the_transform_is_continuous_in_maturity(curve, s) -> None:
    """A branch jump is a discontinuity, and this is what looks for one.

    ``log A`` is swept over a fine grid to fifty years -- long enough that a
    strongly oscillating argument would have wound several times had it been
    going to -- and each step's change is bounded by the derivative the Riccati
    system gives, ``|d log A / d tau| = kappa*theta*|B|``.

    The headroom factor covers the supremum of ``|B|`` *inside* a step, which
    the two endpoints only bracket; ``1.5`` is generous for a grid this fine
    and the measured worst case is 1.00.  What makes the check meaningful is
    the second assertion: the smallest possible branch jump is
    ``2*pi*(2*kappa*theta/volatility**2)``, the coefficient standing in front
    of the logarithm, and the tolerance is asserted to be a small fraction of
    it -- so a wound branch could not hide inside the headroom.
    """
    kappa, theta, sigma = curve["kappa"], curve["theta"], curve["volatility"]
    taus = np.linspace(0.0, 50.0, 2001)
    pairs = [cir_integrated_rate_coefficients(s, maturity=t, **curve) for t in taus]
    values = [complex(log_a) for log_a, _ in pairs]
    bs = [complex(b) for _, b in pairs]
    step = taus[1] - taus[0]

    smallest_branch_jump = 2.0 * math.pi * (2.0 * kappa * theta / (sigma * sigma))
    for index in range(len(taus) - 1):
        jump = abs(values[index + 1] - values[index])
        bound = 1.5 * kappa * theta * max(abs(bs[index]), abs(bs[index + 1])) * step
        assert jump <= bound + 1e-12
        assert bound < 0.05 * smallest_branch_jump


@pytest.mark.parametrize("curve", CURVES, ids=ids("curve", CURVES))
@pytest.mark.parametrize("s", COMPLEX_ARGUMENTS, ids=ids("s", COMPLEX_ARGUMENTS))
def test_one_plus_x_stays_in_a_disk_that_excludes_the_origin(curve, s) -> None:
    r"""The geometric premise behind "no branch tracking is needed".

    ``1 + x_s = ((gamma+kappa) + (gamma-kappa) z) / (2 gamma)`` with
    ``|z| <= 1``, and ``|gamma+kappa| > |gamma-kappa|`` because
    ``|gamma+kappa|**2 - |gamma-kappa|**2 = 4 kappa Re(gamma) > 0``.  So the
    whole ``tau``-path lies in an open disk that misses the origin, its
    argument spans less than ``pi``, and the principal logarithm is the
    continuous one.  Asserted here so the docstring's claim is checked rather
    than believed.
    """
    kappa, sigma = curve["kappa"], curve["volatility"]
    gamma = cmath.sqrt(kappa * kappa + 2.0 * sigma * sigma * s)
    assert gamma.real > 0.0
    assert abs(gamma + kappa) > abs(gamma - kappa)

    centre = (gamma + kappa) / (2.0 * gamma)
    radius = abs(gamma - kappa) / abs(2.0 * gamma)
    assert radius < abs(centre)

    for tau in np.linspace(0.0, 80.0, 801):
        z = cmath.exp(-gamma * tau)
        one_plus_x = ((gamma + kappa) + (gamma - kappa) * z) / (2.0 * gamma)
        assert abs(one_plus_x - centre) <= radius + 1e-12
        # The consequence: never on the cut, so log1p never jumps.
        assert abs(cmath.phase(one_plus_x)) < math.pi - 1e-9


# --- the domain refusal ---------------------------------------------------------


def test_an_argument_outside_the_representation_domain_is_refused() -> None:
    """The implemented square-root branch excludes a non-positive discriminant."""
    kappa, sigma = 0.5, 0.4
    critical = -(kappa**2) / (2.0 * sigma**2)
    with pytest.raises(RateValidationError, match="representation-domain"):
        cir_integrated_rate_transform(
            critical - 0.1,
            kappa=kappa,
            theta=0.04,
            volatility=sigma,
            initial_rate=0.03,
            maturity=1.0,
        )
    # Just inside, it is an ordinary number.
    value = cir_integrated_rate_transform(
        critical + 0.1,
        kappa=kappa,
        theta=0.04,
        volatility=sigma,
        initial_rate=0.03,
        maturity=1.0,
    )
    assert math.isfinite(float(value))


def test_a_complex_argument_on_the_cut_is_refused_but_a_nearby_one_is_not() -> None:
    """The cut is reached only by a real ``s``; an imaginary part steps off it."""
    kappa, sigma = 0.5, 0.4
    critical = -(kappa**2) / (2.0 * sigma**2)
    shared = dict(kappa=kappa, theta=0.04, volatility=sigma, initial_rate=0.03, maturity=1.0)
    with pytest.raises(RateValidationError):
        cir_integrated_rate_transform(complex(critical - 1.0, 0.0), **shared)
    value = cir_integrated_rate_transform(complex(critical - 1.0, 1e-6), **shared)
    assert np.isfinite(complex(value))


def test_an_array_argument_is_refused_if_any_entry_is_singular() -> None:
    """One bad node in a quadrature must not be averaged in silently."""
    kappa, sigma = 0.5, 0.4
    critical = -(kappa**2) / (2.0 * sigma**2)
    with pytest.raises(RateValidationError):
        cir_integrated_rate_transform(
            np.array([1.0, 0.5, critical - 3.0]),
            kappa=kappa,
            theta=0.04,
            volatility=sigma,
            initial_rate=0.03,
            maturity=1.0,
        )


# --- shape and backends ---------------------------------------------------------


def test_an_array_of_arguments_is_evaluated_elementwise() -> None:
    """A quadrature over ``u`` is one call, which is why ``s`` is not a scalar."""
    curve = {"kappa": 0.3, "theta": 0.04, "volatility": 0.1}
    u = np.linspace(0.0, 20.0, 64)
    s = 1.0 - 1j * u
    values = cir_integrated_rate_transform(s, initial_rate=0.04, maturity=2.0, **curve)
    assert values.shape == u.shape
    one_at_a_time = np.array(
        [
            cir_integrated_rate_transform(
                np.complex128(each), initial_rate=0.04, maturity=2.0, **curve
            )
            for each in s
        ]
    )
    # Close, not bitwise, and the reason is NumPy rather than this module:
    # its complex-multiply *array* loop and its complex-multiply *scalar* path
    # are different code and disagree in the last bit (measured: 30 of 64
    # random products under NumPy 2.2.6). ``den = (gamma+kappa) + delta*z`` is
    # where that first shows up. The claim under test is that ``s`` broadcasts
    # elementwise, so the bound is a few ulp of the value.
    np.testing.assert_allclose(values, one_at_a_time, rtol=8.0 * 2.0**-53, atol=0.0)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_a_real_argument_keeps_the_caller_s_backend(backend) -> None:
    """The kernel is array-API native and the new entry point does not break that.

    Real ``s`` only: a complex quadrature is host-side float64 by construction,
    which is what :mod:`fast_vollib.pricing` already documents.
    """
    module = pytest.importorskip(backend)
    if backend == "torch":
        kappa = module.tensor(0.3, dtype=module.float64, requires_grad=True)
    else:
        module = pytest.importorskip("jax.numpy")
        kappa = module.asarray(0.3, dtype=module.float64)

    psi = cir_integrated_rate_transform(
        2.0, kappa=kappa, theta=0.04, volatility=0.1, initial_rate=0.04, maturity=1.5
    )
    reference = cir_integrated_rate_transform(
        2.0, kappa=0.3, theta=0.04, volatility=0.1, initial_rate=0.04, maturity=1.5
    )
    detach = getattr(psi, "detach", None)
    assert float(detach() if detach else psi) == pytest.approx(float(reference), rel=1e-14)
    if backend == "torch":
        psi.backward()
        assert kappa.grad is not None and float(kappa.grad) != 0.0


# --- the complex logarithm's own conditioning -----------------------------------


def test_a_small_volatility_stays_accurate_at_a_complex_argument() -> None:
    r"""The ``log1p`` form has to survive being complex, and by default it does not.

    NumPy, torch and JAX all evaluate a complex ``log1p`` as ``log(1 + z)``,
    which is precisely the cancellation the real :func:`math.log1p` exists to
    avoid.  Multiplied by the ``2*kappa*theta/volatility**2`` standing in front
    of it, that lost precision is amplified without bound as the vol-of-rate
    falls -- at ``volatility = 1e-6`` the transform came back with a relative
    error of 4e3 before :func:`fast_vollib.rates.cir._log1p` was written.

    The measurement here is the same one that found it: the small-``tau``
    expansion, which is exact to :math:`O(\tau^3)` and therefore reports the
    numerical error directly once ``tau`` is small.
    """
    s = 1.0 - 3.0j
    curve = {"kappa": 1.5, "theta": 0.08, "volatility": 1e-6}
    for tau in (1e-2, 1e-3, 1e-4):
        log_a, _ = cir_integrated_rate_coefficients(s, maturity=tau, **curve)
        leading = -curve["kappa"] * curve["theta"] * s * tau * tau / 2.0
        # First-order convergence in tau, with no numerical floor in the way.
        assert abs(complex(log_a) - leading) / abs(leading) < 0.6 * tau


def test_the_kahan_decomposition_beats_the_backend_log1p() -> None:
    """The helper's claim, measured against fifty digits.

    Recorded as a test rather than a comment because it is the justification
    for not simply calling ``xp.log1p``: if a future backend fixed its complex
    ``log1p``, this would still pass, and if the helper regressed to the naive
    form it would not.
    """
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 50

    from fast_vollib._array_api import get_namespace
    from fast_vollib.rates.cir import _log1p

    xp = get_namespace(np.zeros(1))
    for z in (1e-6 + 2e-6j, -1e-8 + 3e-9j, 1e-12 - 1e-12j, 0.3 - 0.4j):
        reference = complex(mpmath.log(mpmath.mpc(1) + mpmath.mpc(z.real, z.imag)))
        accurate = complex(_log1p(xp, np.complex128(z)))
        assert abs(accurate - reference) <= 4.0 * 2.0**-53 * abs(reference)

    # And the real branch is untouched: still the backend's own log1p, bit for bit.
    for x in (1e-12, -1e-8, 0.25, -0.5):
        assert float(_log1p(xp, np.float64(x))) == float(np.log1p(np.float64(x)))


def test_the_refusal_uses_the_package_s_own_error_type() -> None:
    """One exception type per package, and it is also a ``ValueError``.

    ``except ValueError`` written before ``fast_vollib.rates`` existed keeps
    working; ``except RateError`` catches the whole layer. A bare ``ValueError``
    here would have been the only refusal in the module outside that hierarchy.
    """
    from fast_vollib.rates import RateError

    assert issubclass(RateValidationError, RateError)
    assert issubclass(RateValidationError, ValueError)
    with pytest.raises(RateError):
        cir_integrated_rate_transform(
            -100.0, kappa=0.5, theta=0.04, volatility=0.4, initial_rate=0.03, maturity=1.0
        )
