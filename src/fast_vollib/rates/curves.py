r"""Concrete discount curves: a flat one, and one that is a CIR term structure.

Both are frozen, slotted dataclasses that bind *valuation inputs* and nothing
else.  A curve holds no instrument, no simulated state, no engine, and no
process -- :class:`CIRDiscountCurve` binds four numbers and calls the kernel in
:mod:`fast_vollib.rates.cir`; it does not own a
:class:`~fast_vollib.processes.CIRShortRate` and cannot be sampled.  That
separation is what lets the same curve discount a bond, a swap leg, and a
simulated payoff without any of them learning about the others.

Parameters are stored exactly as passed.  A ``requires_grad`` torch tensor stays
that tensor, so a present value built on the curve is differentiable in
``rate``, or in ``kappa``, without the curve making a detached copy on the way
in.

Both curves are continuously compounded.  That is the convention
:meth:`fast_vollib.simulation.MonteCarloEngine._discount` and
:meth:`fast_vollib.surface.SurfaceMarket.discount_at` already use, and a library
that used two compounding conventions in three places would be reporting
different prices for the same market depending on which door a caller came
through.

Examples
--------
>>> from fast_vollib.rates import CIRDiscountCurve, FlatDiscountCurve
>>> flat = FlatDiscountCurve(rate=0.03)
>>> float(round(flat.discount_factor(2.0), 12))
0.941764533584
>>> cir = CIRDiscountCurve(kappa=0.3, theta=0.04, volatility=0.1, initial_rate=0.04)
>>> float(round(cir.discount_factor(1.0), 12))
0.960840811993
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._array_api import concrete_float, get_namespace
from ._validate import ensure_scalar_parameter
from .cir import cir_discount_factor, cir_instantaneous_forward_rate, cir_zero_rate
from .errors import RateValidationError

__all__ = [
    "EXTRAPOLATIONS",
    "CIRDiscountCurve",
    "FlatDiscountCurve",
    "InterpolatedDiscountCurve",
]

#: What a curve does past its last pillar.  The vocabulary is
#: :data:`fast_vollib.surface.EXTRAPOLATIONS` verbatim, because the two are the
#: same decision about the same kind of object and two spellings of it would
#: eventually mean two different things.
EXTRAPOLATIONS = ("error", "flat")


@dataclass(frozen=True, slots=True)
class FlatDiscountCurve:
    r"""One continuously compounded rate at every maturity: ``P = exp(-rate * t)``.

    Parameters
    ----------
    rate : float or scalar array
        The continuously compounded zero rate, finite. **Negative rates are
        accepted**, and then ``P > 1``: that is a real market condition, not an
        input error, and refusing it would make the curve unable to represent
        the front end of several government curves.

    Notes
    -----
    Continuously compounded, matching
    :meth:`fast_vollib.simulation.MonteCarloEngine._discount` and
    :meth:`fast_vollib.surface.SurfaceMarket.discount_at`; a test compares all
    three rather than leaving the agreement to be assumed.

    Examples
    --------
    >>> from fast_vollib.rates import FlatDiscountCurve
    >>> curve = FlatDiscountCurve(rate=0.05)
    >>> float(curve.discount_factor(0.0))
    1.0
    >>> float(round(curve.discount_factor(1.0), 12))
    0.951229424501

    A negative rate discounts to more than one, which is the point:

    >>> float(FlatDiscountCurve(rate=-0.01).discount_factor(1.0)) > 1.0
    True
    """

    rate: Any

    def __post_init__(self) -> None:
        ensure_scalar_parameter(self.rate, field="rate")

    def discount_factor(self, maturity: Any) -> Any:
        """``exp(-rate * maturity)``; exactly ``1`` at ``maturity == 0``."""
        ensure_scalar_parameter(maturity, field="maturity", non_negative=True)
        xp = get_namespace(self.rate, maturity)
        return xp.exp(-self.rate * maturity)

    def zero_rate(self, maturity: Any) -> Any:
        """The zero rate, which is :attr:`rate` at every maturity."""
        ensure_scalar_parameter(maturity, field="maturity", non_negative=True)
        return self.rate


@dataclass(frozen=True, slots=True)
class CIRDiscountCurve:
    r"""The CIR (1985) term structure, bound to one set of risk-neutral parameters.

    Parameters
    ----------
    kappa : float or scalar array
        Mean-reversion speed, strictly positive.
    theta : float or scalar array
        Long-run short rate, non-negative.
    volatility : float or scalar array
        Vol-of-rate :math:`\sigma`, non-negative. Exactly zero gives the
        deterministic mean-reverting limit, which is exact rather than
        approached.
    initial_rate : float or scalar array
        The short rate now, non-negative. The model cannot produce a negative
        rate at any later time either, which is a property of the square-root
        diffusion and a reason to reach for something else on a negative-rate
        curve.

    Notes
    -----
    The parameters are risk-neutral; see :mod:`fast_vollib.rates.cir` for the
    mapping from Bakshi, Cao and Chen's :math:`\theta_R - \kappa_R R` drift.

    This curve binds parameters and nothing else. It holds no process, no path,
    and no random state, so the same object can discount a cashflow and describe
    a model without either meaning leaking into the other.

    Examples
    --------
    >>> from fast_vollib.rates import CIRDiscountCurve
    >>> curve = CIRDiscountCurve(kappa=0.3, theta=0.04, volatility=0.1, initial_rate=0.04)
    >>> float(curve.discount_factor(0.0))
    1.0
    >>> float(round(curve.zero_rate(10.0), 12))
    0.038876035206

    The Feller ratio is reported and never enforced:

    >>> float(round(curve.feller_ratio, 6))
    2.4
    >>> curve.satisfies_feller
    True
    """

    kappa: Any
    theta: Any
    volatility: Any
    initial_rate: Any

    def __post_init__(self) -> None:
        ensure_scalar_parameter(self.kappa, field="kappa", positive=True)
        ensure_scalar_parameter(self.theta, field="theta", non_negative=True)
        ensure_scalar_parameter(self.volatility, field="volatility", non_negative=True)
        ensure_scalar_parameter(self.initial_rate, field="initial_rate", non_negative=True)

    @property
    def feller_ratio(self) -> float:
        r""":math:`2\kappa\theta/\sigma^2`. Above one the rate never reaches zero.

        ``nan`` when a parameter is a tracer with no readable value, and
        ``inf`` at ``volatility == 0``, where the rate is deterministic and the
        boundary is unreachable for a different reason.
        """
        kappa = concrete_float(self.kappa)
        theta = concrete_float(self.theta)
        sigma = concrete_float(self.volatility)
        if kappa is None or theta is None or sigma is None:  # pragma: no cover - traced
            return float("nan")
        if sigma == 0.0:
            return float("inf")
        return 2.0 * kappa * theta / (sigma * sigma)

    @property
    def satisfies_feller(self) -> bool:
        """Whether :attr:`feller_ratio` exceeds one."""
        return bool(self.feller_ratio > 1.0)

    def discount_factor(self, maturity: Any) -> Any:
        """``P(0, maturity)``; exactly ``1`` at ``maturity == 0``."""
        return cir_discount_factor(
            kappa=self.kappa,
            theta=self.theta,
            volatility=self.volatility,
            initial_rate=self.initial_rate,
            maturity=maturity,
        )

    def zero_rate(self, maturity: Any) -> Any:
        """``-log P(0, maturity) / maturity``; :attr:`initial_rate` at zero."""
        return cir_zero_rate(
            kappa=self.kappa,
            theta=self.theta,
            volatility=self.volatility,
            initial_rate=self.initial_rate,
            maturity=maturity,
        )

    def instantaneous_forward_rate(self, maturity: Any) -> Any:
        """``f(0, maturity)``, from the Riccati system rather than by differencing."""
        return cir_instantaneous_forward_rate(
            kappa=self.kappa,
            theta=self.theta,
            volatility=self.volatility,
            initial_rate=self.initial_rate,
            maturity=maturity,
        )


@dataclass(frozen=True, slots=True)
class InterpolatedDiscountCurve:
    r"""An observed term structure, log-linear in :math:`P` between its pillars.

    A *container*, not a bootstrapper.  It holds discount factors somebody else
    produced -- a stripped curve, a vendor file, a calibration's output -- and
    answers between them by one stated rule.  It does not build a curve from
    deposits and swaps, and it does not know where its numbers came from.

    Parameters
    ----------
    maturities : sequence or array
        Pillar times in years, strictly increasing and strictly positive.
    discount_factors : sequence or array
        One factor per pillar, finite and strictly positive. **Not required to
        be decreasing**: a negative-rate market curve legitimately has
        :math:`P > 1`, and refusing that would make the class unable to hold
        several real government curves.
    extrapolation : {'error', 'flat'}, default 'error'
        What happens past the last pillar. ``'flat'`` holds the last pillar's
        *zero rate*, so :math:`P` keeps decaying; holding the last *factor*
        instead would make the curve flat in price, which is a forward rate of
        zero and is not what "flat" means about a curve.

    Notes
    -----
    **The rule.** Log-linear in :math:`P` is linear in :math:`r(T)\,T`, so
    :math:`\log P` is interpolated linearly and exponentiated.  Written as a
    convex combination, :math:`(1-w)\log P_i + w \log P_{i+1}`, which returns
    the pillar's own log-factor *exactly* at either end of a segment -- the
    algebraically equal :math:`\log P_i + w(\log P_{i+1} - \log P_i)` does not.

    **The origin is not a pillar.**  :math:`P(0,0) = 1` is a fact about
    discount factors rather than an observation, so it anchors the short end and
    the region below the first pillar is interpolation against it, not
    extrapolation.  That is what makes ``discount_factor(0.0)`` exactly ``1.0``
    while ``discount_factor(1e-9)`` is a number rather than an error -- the
    alternative, special-casing zero and refusing everything just above it, is
    not a rule anyone could rely on.  In rate terms the short end is flat at the
    first pillar's zero rate, which is the usual market convention.

    **Exactness.**  The interpolation returns a pillar's ``log`` factor to the
    bit, so the factor itself round-trips through ``exp(log(P))`` -- identical
    for about 99.8% of realistic factors and never more than one ulp away.
    ``P(0,0)`` is exactly ``1.0``, because ``exp(0.0)`` is.

    **Against** :class:`fast_vollib.surface.SurfaceMarket`.  That class
    interpolates the zero *rate* linearly in :math:`T` and discounts by
    :math:`e^{-r(T)T}`; this one is linear in :math:`r(T)\,T`.  They agree at
    pillars and **differ between them**, by more the further apart the pillars
    and the more curved the rate.  Neither is more correct; they are different
    conventions, and this one is canonical for fixed income. Nothing here
    changes ``SurfaceMarket``.

    Examples
    --------
    >>> import numpy as np
    >>> from fast_vollib.rates import InterpolatedDiscountCurve
    >>> curve = InterpolatedDiscountCurve(
    ...     maturities=np.array([1.0, 2.0, 5.0]),
    ...     discount_factors=np.array([0.97, 0.94, 0.85]),
    ... )
    >>> float(curve.discount_factor(0.0))
    1.0
    >>> float(curve.discount_factor(2.0))
    0.94
    >>> round(float(curve.zero_rate(1.0)), 12)
    0.030459207485

    Past the last pillar the default refuses rather than inventing a level:

    >>> from fast_vollib.rates import RateValidationError
    >>> try:
    ...     curve.discount_factor(7.0)
    ... except RateValidationError as error:
    ...     print(str(error).split(".")[0])
    Maturity 7

    >>> flat = InterpolatedDiscountCurve(
    ...     maturities=np.array([1.0, 2.0, 5.0]),
    ...     discount_factors=np.array([0.97, 0.94, 0.85]),
    ...     extrapolation="flat",
    ... )
    >>> round(float(flat.discount_factor(7.0)), 10)
    0.7965013129
    """

    maturities: Any
    discount_factors: Any
    extrapolation: str = "error"

    def __post_init__(self) -> None:
        if self.extrapolation not in EXTRAPOLATIONS:
            raise RateValidationError(
                f"extrapolation must be one of {EXTRAPOLATIONS}; got "
                f"{self.extrapolation!r}. A curve does not guess what to do past its "
                f"own data."
            )
        n_times = _pillar_count(self.maturities, field="maturities")
        n_factors = _pillar_count(self.discount_factors, field="discount_factors")
        if n_times == 0:
            raise RateValidationError(
                "maturities must not be empty: a curve with no pillars answers nothing."
            )
        if n_factors != n_times:
            raise RateValidationError(
                f"discount_factors has {n_factors} entries but maturities has "
                f"{n_times}. One factor per pillar."
            )
        # Shape is knowable while tracing and has been checked. Values are not,
        # so the rest runs only when the numbers can be read -- the same split
        # ``ensure_scalar_parameter`` makes, and the reason a curve built from
        # traced pillars inside ``jax.jit`` is constructible at all.
        times = _readable_pillars(self.maturities)
        factors = _readable_pillars(self.discount_factors)
        if times is None or factors is None:
            return
        if not bool(np.all(np.isfinite(times))) or not bool(np.all(times > 0.0)):
            raise RateValidationError(
                f"maturities must be finite and strictly positive; got {times.tolist()}. "
                f"P(0, 0) = 1 anchors the short end and is not supplied as a pillar."
            )
        if times.size > 1 and not bool(np.all(np.diff(times) > 0.0)):
            raise RateValidationError(
                f"maturities must be strictly increasing; got {times.tolist()}."
            )
        if not bool(np.all(np.isfinite(factors))) or not bool(np.all(factors > 0.0)):
            raise RateValidationError(
                f"discount_factors must be finite and strictly positive; got "
                f"{factors.tolist()}. A non-positive factor has no logarithm, and this "
                f"curve is linear in one."
            )

    @property
    def n_pillars(self) -> int:
        """How many observed pillars the curve holds."""
        return _pillar_count(self.maturities, field="maturities")

    def discount_factor(self, maturity: Any) -> Any:
        """``P(0, maturity)``, in the namespace the pillars and query select."""
        maturity = ensure_scalar_parameter(maturity, field="maturity", non_negative=True)
        self._check_range(maturity)
        xp = get_namespace(self.maturities, self.discount_factors, maturity)
        return xp.exp(self._log_factor(xp, maturity))

    def zero_rate(self, maturity: Any) -> Any:
        """``-log P / maturity``; at zero maturity the first pillar's own rate.

        The limit is not a convention chosen here: the short end is flat by
        construction, so :math:`r(0)` *is* the first pillar's zero rate.
        """
        maturity = ensure_scalar_parameter(maturity, field="maturity", non_negative=True)
        self._check_range(maturity)
        xp = get_namespace(self.maturities, self.discount_factors, maturity)
        # ``-log P / T`` is 0/0 at the origin. The limit is not a convention
        # chosen for convenience: the short end is flat by construction, so
        # ``r(0)`` *is* the first pillar's zero rate. Written with ``where``
        # rather than a concrete branch so a traced query at zero gets the
        # limit too, instead of a NaN the caller cannot see coming.
        times = xp.asarray(self.maturities)
        first = -xp.log(xp.asarray(self.discount_factors)[0]) / times[0]
        at_origin = maturity == 0.0
        divisor = xp.where(at_origin, xp.asarray(1.0, like=times), maturity)
        return xp.where(at_origin, first, -self._log_factor(xp, maturity) / divisor)

    # --- the kernel ------------------------------------------------------------

    def _log_factor(self, xp: Any, maturity: Any) -> Any:
        """``log P`` at one maturity, anchored at the origin.

        Every operation is array-API native, so a torch or JAX pillar vector
        carries a gradient through to the factor -- which is what a bucketed
        sensitivity of a bond to its own curve is.
        """
        times = xp.asarray(self.maturities)
        logs = xp.log(xp.asarray(self.discount_factors))
        zero = xp.zeros((1,), like=times)
        times = xp.concatenate((zero, times))
        logs = xp.concatenate((zero * logs[:1], logs))

        count = times.shape[0]
        index = xp.searchsorted(times, maturity, side="right") - 1
        index = xp.clip(index, 0, count - 2)
        left = xp.take(times, index)
        right = xp.take(times, index + 1)
        weight = (maturity - left) / (right - left)
        # The convex form, not ``l0 + w * (l1 - l0)``: at ``w == 1`` this is
        # exactly ``l1`` and at ``w == 0`` exactly ``l0``, which is what makes
        # the curve return a pillar's own value rather than a rounding of it.
        interior = (1.0 - weight) * xp.take(logs, index) + weight * xp.take(logs, index + 1)

        last_time = times[count - 1]
        if self.extrapolation == "error":
            # The bound was already enforced above for a readable maturity. A
            # traced one has no value to check, so rather than silently
            # extending the last segment -- a plausible number from a curve
            # that was told to refuse -- the answer past the end is NaN.
            #
            # A *constant* NaN, not one built from the pillars. ``where`` takes
            # the gradient of the branch it did not select and multiplies it by
            # zero, and zero times NaN is NaN: routing the pillars through a
            # NaN would make every in-range gradient NaN too.
            outside = xp.zeros((), like=interior) + float("nan")
        else:
            outside = logs[count - 1] * (maturity / last_time)
        return xp.where(maturity <= last_time, interior, outside)

    def _check_range(self, maturity: Any) -> None:
        """Refuse a maturity past the last pillar when ``extrapolation='error'``.

        A value check, so it runs when the number is readable and is skipped
        under a trace -- the same rule every other curve parameter follows. The
        traced case is not left to guess: see :meth:`_log_factor`.
        """
        if self.extrapolation != "error":
            return
        value = concrete_float(maturity)
        pillars = _readable_pillars(self.maturities)
        if value is None or pillars is None:
            return
        last = float(pillars[-1])
        if value > last:
            raise RateValidationError(
                f"Maturity {value!r} is past the curve's last pillar {last!r} and "
                f"extrapolation='error'. Pass extrapolation='flat' to hold the last "
                f"pillar's zero rate, or supply a pillar that covers it."
            )


def _pillar_count(values: Any, *, field: str) -> int:
    """How many pillars ``values`` holds, without reading any of them.

    Rank and length are static properties: a JAX tracer knows its own shape.
    So this check always runs, and the ones that need a *number* do not.
    """
    if isinstance(values, (str, bytes)):
        raise RateValidationError(f"{field} must be a sequence of numbers, not a string.")
    shape = getattr(values, "shape", None)
    if shape is None:
        try:
            return len(values)
        except TypeError:
            raise RateValidationError(
                f"{field} must be a sequence of numbers; got {type(values).__name__}."
            ) from None
    if len(shape) != 1:
        raise RateValidationError(
            f"{field} must be one-dimensional; got shape {tuple(shape)}. A curve holds "
            f"one term structure, not a surface."
        )
    return int(shape[0])


def _readable_pillars(values: Any) -> np.ndarray | None:
    """The pillar vector as float64 NumPy, or ``None`` when it cannot be read.

    ``None`` means "traced", and every caller treats that as "skip the value
    check" rather than as an error. Nothing numerical goes through here: the
    factors a caller gets back are computed in their own namespace.
    """
    if hasattr(values, "detach"):
        # ``.cpu()`` as well: a CUDA tensor is a perfectly readable value, and
        # letting its conversion fail below would silently skip the finite and
        # positive checks on a curve that could have been checked.
        values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
    try:
        array = np.asarray(values, dtype=np.float64)
    except Exception:
        return None
    return array.reshape(-1)
