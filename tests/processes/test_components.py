"""The three component axes: validation, the closed unions, and one compensator.

Most of what components do is checked through ``Bates``, which is where they are
used.  What is checked here is what they promise on their own -- in particular
that ``mean_relative_jump`` is computed once, accurately, and in the caller's
namespace, because the sampler and the characteristic function both read it and
a disagreement between them would show up as a mispriced option rather than as
a failing assertion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_vollib._simulation_errors import SimulationValidationError
from fast_vollib.processes import (
    CIRShortRate,
    ConstantShortRate,
    ConstantVariance,
    HestonVariance,
    LognormalJumps,
    NoJumps,
)
from fast_vollib.processes.components import (
    JUMP_TYPES,
    SHORT_RATE_TYPES,
    VARIANCE_TYPES,
    validate_component,
)

HESTON = {"kappa": 2.0, "theta": 0.04, "vol_of_vol": 0.3, "rho": -0.7}
JUMPS = {"jump_intensity": 0.5, "mean_log_jump": -0.05, "jump_volatility": 0.2}


# --- the markers hold no level -------------------------------------------------


@pytest.mark.parametrize("marker", [ConstantVariance(), NoJumps(), ConstantShortRate()])
def test_a_constant_marker_stores_nothing(marker) -> None:
    """An API able to hold two disagreeing values for one quantity eventually will."""
    assert dict(marker.params()) == {}
    assert not marker.__dataclass_fields__
    assert marker == type(marker)()
    assert hash(marker) == hash(type(marker)())


def test_no_jumps_reports_exactly_zero_rather_than_nearly_zero() -> None:
    """The compensator is subtracted from a drift and added to an exponent, so
    a merely-small value would move every price in the last digits."""
    assert NoJumps().mean_relative_jump == 0.0
    assert NoJumps().drift_compensator == 0.0
    assert type(NoJumps().drift_compensator) is float


# --- the compensator, written once ---------------------------------------------


@pytest.mark.parametrize("mean", [-0.3, -0.05, 0.0, 0.02])
@pytest.mark.parametrize("volatility", [0.0, 0.05, 0.4])
def test_the_mean_relative_jump_matches_a_fifty_digit_reference(mean, volatility) -> None:
    """``E[J] = exp(m + delta^2/2) - 1`` from the log-normal moment formula.

    Checked against ``mpmath`` at fifty digits rather than against
    ``math.exp(x) - 1`` in double precision, because the naive form is the
    *less* accurate of the two: at ``x = 0.05`` it is already wrong in the last
    two bits, so using it as the reference would fail a correct implementation.
    """
    import mpmath

    with mpmath.workdps(50):
        exponent = mean + 0.5 * volatility**2
        reference = float(mpmath.e ** mpmath.mpf(exponent) - 1)

    component = LognormalJumps(jump_intensity=1.0, mean_log_jump=mean, jump_volatility=volatility)
    assert float(component.mean_relative_jump) == pytest.approx(reference, rel=1e-15, abs=1e-300)


def test_the_mean_relative_jump_survives_an_exponent_the_naive_form_cannot() -> None:
    """``exp(x) - 1`` loses its leading digits for small ``x``; ``expm1`` does not.

    At ``x = 1e-12`` -- a mean log jump of a tenth of a basis point, which is
    ordinary near a calibration boundary -- the naive form is wrong in the
    fifth digit. At ``1e-17`` it returns exactly zero, and the drift correction
    disappears entirely.
    """
    small = LognormalJumps(jump_intensity=1.0, mean_log_jump=1e-12, jump_volatility=0.0)
    assert float(small.mean_relative_jump) == pytest.approx(1e-12, rel=1e-15)
    assert abs((math.exp(1e-12) - 1.0) / 1e-12 - 1.0) > 1e-5

    vanishing = LognormalJumps(jump_intensity=1.0, mean_log_jump=1e-17, jump_volatility=0.0)
    assert float(vanishing.mean_relative_jump) == pytest.approx(1e-17, rel=1e-15)
    assert math.exp(1e-17) - 1.0 == 0.0


def test_the_compensator_is_the_intensity_times_the_mean_jump() -> None:
    component = LognormalJumps(**JUMPS)
    assert float(component.drift_compensator) == pytest.approx(
        float(JUMPS["jump_intensity"]) * float(component.mean_relative_jump), rel=1e-15
    )


def test_a_zero_intensity_compensates_nothing() -> None:
    component = LognormalJumps(**{**JUMPS, "jump_intensity": 0.0})
    assert float(component.drift_compensator) == 0.0


def test_the_compensator_stays_native_and_differentiable() -> None:
    """The Fourier transform reads the same property, so it must carry a graph."""
    torch = pytest.importorskip("torch")
    mean = torch.tensor(-0.05, dtype=torch.float64, requires_grad=True)
    component = LognormalJumps(
        jump_intensity=torch.tensor(0.5, dtype=torch.float64),
        mean_log_jump=mean,
        jump_volatility=torch.tensor(0.2, dtype=torch.float64),
    )
    component.drift_compensator.backward()
    # d/dm [lambda (e^{m + d^2/2} - 1)] = lambda e^{m + d^2/2}
    assert float(mean.grad) == pytest.approx(0.5 * math.exp(-0.05 + 0.02), rel=1e-12)


# --- validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kappa", 0.0, "variance.kappa must be strictly positive"),
        ("theta", -1.0, "variance.theta must be strictly positive"),
        ("vol_of_vol", 0.0, "variance.vol_of_vol must be strictly positive"),
        ("rho", 1.0, "strictly inside"),
        ("rho", -1.0, "strictly inside"),
        ("kappa", float("nan"), "finite"),
    ],
)
def test_an_invalid_variance_component_is_refused(field, value, message) -> None:
    with pytest.raises(SimulationValidationError, match=message):
        HestonVariance(**{**HESTON, field: value})


def test_the_variance_component_reports_feller_and_does_not_enforce_it() -> None:
    assert HestonVariance(**HESTON).feller_ratio == pytest.approx(1.7777777, rel=1e-6)
    assert HestonVariance(**HESTON).satisfies_feller
    violating = HestonVariance(kappa=0.5, theta=0.06, vol_of_vol=0.9, rho=-0.9)
    assert not violating.satisfies_feller
    assert violating.params()["kappa"] == 0.5  # constructed, and usable


def test_the_variance_component_carries_no_drift() -> None:
    """A drift belongs to the spot; one here could contradict the facade's."""
    assert "drift" not in HestonVariance(**HESTON).__dataclass_fields__


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("jump_intensity", -1.0, "jumps.jump_intensity must be non-negative"),
        ("jump_volatility", -0.1, "jumps.jump_volatility must be non-negative"),
        ("mean_log_jump", float("inf"), "finite"),
        ("jump_intensity", float("nan"), "finite"),
    ],
)
def test_an_invalid_jump_component_is_refused(field, value, message) -> None:
    with pytest.raises(SimulationValidationError, match=message):
        LognormalJumps(**{**JUMPS, field: value})


@pytest.mark.parametrize("admissible_zero", ["jump_intensity", "jump_volatility"])
def test_the_two_zeros_that_are_models_rather_than_errors(admissible_zero) -> None:
    """Zero intensity is the reduction; zero jump volatility is a fixed jump size."""
    assert LognormalJumps(**{**JUMPS, admissible_zero: 0.0}) is not None


def test_a_parameter_is_stored_as_the_object_that_was_passed() -> None:
    intensity = np.float64(0.5)
    assert (
        LognormalJumps(**{**JUMPS, "jump_intensity": intensity}).params()["jump_intensity"]
        is intensity
    )


# --- the closed unions ---------------------------------------------------------


def test_the_unions_list_exactly_the_shipped_components() -> None:
    assert VARIANCE_TYPES == (HestonVariance, ConstantVariance)
    assert JUMP_TYPES == (LognormalJumps, NoJumps)
    assert SHORT_RATE_TYPES == (CIRShortRate, ConstantShortRate)


@pytest.mark.parametrize(
    ("admissible", "member"),
    [
        (VARIANCE_TYPES, ConstantVariance()),
        (JUMP_TYPES, NoJumps()),
        (SHORT_RATE_TYPES, ConstantShortRate()),
        (SHORT_RATE_TYPES, CIRShortRate(kappa=0.3, theta=0.04, volatility=0.1)),
    ],
)
def test_a_member_of_a_union_is_accepted_and_returned_unchanged(admissible, member) -> None:
    assert validate_component(member, field="x", admissible=admissible) is member


def test_a_duck_typed_look_alike_is_refused_naming_the_admissible_types() -> None:
    """A sampler reads a component's fields by name, so resemblance is not enough."""

    class Impostor:
        kappa = theta = vol_of_vol = rho = 0.1

    with pytest.raises(SimulationValidationError, match="HestonVariance, ConstantVariance"):
        validate_component(Impostor(), field="variance", admissible=VARIANCE_TYPES)


def test_a_component_from_the_wrong_axis_is_refused() -> None:
    with pytest.raises(SimulationValidationError, match="LognormalJumps, NoJumps"):
        validate_component(ConstantVariance(), field="jumps", admissible=JUMP_TYPES)


def test_the_cir_process_is_a_short_rate_component_and_the_same_object() -> None:
    """A rate model is exactly the thing that can be simulated alone or driven
    inside a larger model, so there is no separate component wrapper for it."""
    process = CIRShortRate(kappa=0.3, theta=0.04, volatility=0.1)
    assert validate_component(process, field="rates", admissible=SHORT_RATE_TYPES) is process
    assert process.state_names == ("short_rate",)
