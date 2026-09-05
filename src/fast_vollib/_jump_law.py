r"""One moment of the log-normal jump law, so that two callers cannot disagree.

:math:`\ln(1 + J) \sim N(m, \delta^2)`, so

.. math:: \mu_J := E[J] = e^{m + \delta^2/2} - 1,

and that single number appears in two places that must agree exactly:

* the **sampler** subtracts :math:`\lambda \mu_J` from the drift, so that the
  discounted spot is a martingale despite the jumps;
* the **characteristic function** carries :math:`-i u \lambda \mu_J T` in its
  exponent, for the same reason.

Written twice they would eventually differ -- in a sign, or in the last bits --
and nothing would fail until a Fourier price disagreed with a Monte Carlo one,
which is a slow and confusing way to find a typo.  So it is written once, here,
in a module that belongs to neither :mod:`fast_vollib.processes` nor
:mod:`fast_vollib.pricing`, because the fact belongs to neither: it is a
property of the distribution.

``expm1`` rather than ``exp(x) - 1``
-----------------------------------
The exponent is small in every realistic calibration, and that is exactly where
the naive form fails.  At :math:`m + \delta^2/2 = 10^{-12}` -- a mean log jump
near zero -- ``exp(x) - 1`` is already wrong in the fifth
significant digit, because the leading ``1`` consumes the mantissa before the
subtraction can give it back.  At :math:`10^{-17}` it returns exactly zero and
the compensation disappears.  ``expm1`` is correctly rounded throughout.

References
----------
Merton, R. C. (1976). Option pricing when underlying stock returns are
discontinuous. *Journal of Financial Economics* 3(1-2), 125-144.
"""

from __future__ import annotations

import math
from typing import Any

from ._array_api import concrete_float

__all__ = ["mean_relative_jump"]


def mean_relative_jump(mean_log_jump: Any, jump_volatility: Any) -> Any:
    """``E[J] = exp(mean_log_jump + jump_volatility^2 / 2) - 1``.

    Evaluated in the caller's own namespace, so a ``requires_grad`` tensor or a
    JAX tracer passes through and the compensator a price depends on stays
    differentiable in the jump parameters.

    Parameters
    ----------
    mean_log_jump, jump_volatility : float or array
        The mean and standard deviation of ``ln(1 + J)``.

    Returns
    -------
    The mean relative jump size, ``mu_J`` in Bakshi, Cao and Chen (1997).

    Examples
    --------
    >>> from fast_vollib._jump_law import mean_relative_jump
    >>> float(round(mean_relative_jump(-0.05, 0.2), 10))
    -0.0295544665

    A vanishing exponent keeps its digits, which ``exp(x) - 1`` does not:

    >>> import math
    >>> float(mean_relative_jump(1e-17, 0.0))
    1e-17
    >>> math.exp(1e-17) - 1.0
    0.0
    """
    exponent = mean_log_jump + 0.5 * jump_volatility * jump_volatility
    concrete = concrete_float(exponent)
    if concrete is not None and not hasattr(exponent, "shape"):
        return math.expm1(concrete)
    from ._array_api import get_namespace

    return get_namespace(exponent).expm1(exponent)
