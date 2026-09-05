"""The Cox-Ingersoll-Ross short rate as a simulable process.

:mod:`fast_vollib.rates` already prices a zero-coupon bond under this model in
closed form, and where a closed form exists this library says so rather than
simulating.  What the closed form cannot do is carry a *path*, and a path is
what BCC97 needs: the same realized short rate has to drive the spot's drift
and the discount integral at once, so the two are correlated and no
``P(0,T)``-times-expected-payoff shortcut is available.

The dynamics under the risk-neutral measure are

.. math:: dr_t = \\kappa(\\theta - r_t)\\,dt + \\sigma\\sqrt{r_t}\\,dW_t

which is the same square-root diffusion Heston's variance follows, so the two
share :mod:`fast_vollib.processes._square_root` and differ only in what else
the Brownian motion drives -- here, nothing.

``kappa`` and ``theta`` are risk-neutral, matching
:class:`fast_vollib.rates.CIRDiscountCurve`; the mapping from Bakshi, Cao and
Chen's ``theta_R - kappa_R R`` parameterization is documented there and holds
unchanged.

Why ``volatility`` must be strictly positive
--------------------------------------------
The curve accepts ``volatility=0`` and evaluates the deterministic limit
analytically.  The process refuses it, for the same reason
:class:`~fast_vollib.processes.Heston` refuses ``vol_of_vol=0``: at zero
volatility there is nothing to simulate.  Every scheme here would still return
numbers -- the quadratic-exponential branch divides by a floored ``psi`` and
the exact transition has infinite degrees of freedom -- and those numbers would
be arrived at through a guard rather than through the mathematics.  The limit
is available and exact: ``process.discount_curve(initial_rate=r0)`` with
``volatility=0`` prices it in closed form.

References
----------
Cox, J. C., Ingersoll, J. E., Ross, S. A. (1985). A Theory of the Term
Structure of Interest Rates. *Econometrica* 53(2), 385-407.

Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.
Springer, section 3.4.

Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic
volatility model. *Journal of Computational Finance* 11(3), 1-42.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from .._array_api import concrete_float, get_namespace
from .._random_api import (
    gamma as _gamma_draws,
    poisson as _poisson_draws,
    random_stream,
    resolve_device,
    resolve_dtype,
    resolve_namespace,
    split,
    standard_normal,
)
from .._simulation_errors import SimulationValidationError, UnsupportedProcessError
from ._square_root import full_truncation_step, quadratic_exponential_step
from .gbm import _validate_parameter

__all__ = ["CIR_SCHEMES", "CIRShortRate"]

#: The transition schemes, in the order of decreasing generality.
#:
#: The first two are biased and available on every backend.  The third is exact
#: at the grid points and needs a gamma sampler bound to the caller's
#: generator, which torch does not publish; see :func:`fast_vollib._random_api.gamma`.
CIR_SCHEMES = ("quadratic_exponential", "full_truncation_euler", "exact_transition")

#: The scheme that draws laws torch cannot reproduce from a supplied generator.
_EXACT = "exact_transition"


@dataclass(frozen=True, slots=True)
class CIRShortRate:
    """A mean-reverting, non-negative short rate.

    Parameters
    ----------
    kappa : float or scalar array
        Mean-reversion speed, strictly positive.
    theta : float or scalar array
        Long-run mean, strictly positive.  Zero is excluded because it makes
        the process degenerate at the origin: ``r = 0`` is then absorbing, and
        the exact transition's degrees of freedom vanish with it.
    volatility : float or scalar array
        Vol-of-rate, strictly positive; see the module note.

    Notes
    -----
    The Feller condition :math:`2\\kappa\\theta > \\sigma^2` keeps the rate
    strictly positive.  It is reported by :attr:`feller_ratio` and **not**
    enforced, exactly as in :class:`fast_vollib.rates.CIRDiscountCurve` and
    :class:`~fast_vollib.processes.Heston`: calibrations violate it routinely
    and all three schemes stay well defined when it fails.

    Parameters are stored exactly as passed, so a torch tensor with
    ``requires_grad=True`` stays that tensor.

    Examples
    --------
    >>> from fast_vollib.processes import CIRShortRate
    >>> process = CIRShortRate(kappa=0.3, theta=0.04, volatility=0.1)
    >>> process.state_names
    ('short_rate',)
    >>> float(round(process.feller_ratio, 6))
    2.4
    >>> float(process.discount_curve(initial_rate=0.04).discount_factor(0.0))
    1.0
    """

    kappa: Any
    theta: Any
    volatility: Any

    #: The single state variable this process evolves.
    state_names: ClassVar[tuple[str, ...]] = ("short_rate",)

    def __post_init__(self) -> None:
        _validate_parameter(self.kappa, field="kappa", positive=True)
        _validate_parameter(self.theta, field="theta", positive=True)
        _validate_parameter(self.volatility, field="volatility", positive=True)

    def params(self) -> Mapping[str, Any]:
        """The three parameters, as the original objects."""
        return MappingProxyType(
            {"kappa": self.kappa, "theta": self.theta, "volatility": self.volatility}
        )

    @property
    def feller_ratio(self) -> float:
        """``2 kappa theta / sigma^2``.  Above one the rate never reaches zero."""
        kappa = concrete_float(self.kappa)
        theta = concrete_float(self.theta)
        sigma = concrete_float(self.volatility)
        if kappa is None or theta is None or sigma is None:  # pragma: no cover - traced
            return float("nan")
        return 2.0 * kappa * theta / (sigma * sigma)

    @property
    def satisfies_feller(self) -> bool:
        """Whether :attr:`feller_ratio` exceeds one."""
        return bool(self.feller_ratio > 1.0)

    def discount_curve(self, *, initial_rate: Any) -> Any:
        """The analytic curve this process implies, given a starting rate.

        A one-way convenience, and one-way in both senses.  The import is local
        so that importing a process does not import the whole rates package,
        and the direction is not reversible: a curve is a valuation input that
        several processes could have produced, so it does not hand back one.

        The initial rate is an argument rather than a field because it is
        *state*, not a parameter -- the same dynamics started from a different
        rate are the same process.
        """
        from ..rates import CIRDiscountCurve

        return CIRDiscountCurve(
            kappa=self.kappa,
            theta=self.theta,
            volatility=self.volatility,
            initial_rate=initial_rate,
        )

    def sample(
        self,
        *,
        initial_state: Mapping[str, Any],
        time_grid: Any,
        n_paths: int,
        rng: Any,
        antithetic: bool = False,
        scheme: str = "quadratic_exponential",
    ) -> Any:
        """Paths shaped ``(n_paths, n_times, 1)``.

        Parameters
        ----------
        initial_state : Mapping
            Must contain ``"short_rate"``, non-negative.
        time_grid : array-like
            Increasing times starting at zero.  Column zero of every path is
            the initial state exactly.
        n_paths : int
        rng : object
            A generator, PRNG key, or integer seed, per
            :mod:`fast_vollib._random_api`.
        antithetic : bool, default False
            Draw half the normals and mirror them.  Refused with
            ``exact_transition`` when the degrees of freedom are at most one,
            where the transition contains no normal to mirror; see
            :meth:`_exact_step`.
        scheme : str, default 'quadratic_exponential'
            One of :data:`CIR_SCHEMES`.

        Returns
        -------
        array
            In the namespace, device, and dtype the inputs selected.  The
            trailing axis has length one, matching ``state_names``, so a
            caller indexes the rate the same way whatever the process.

        Raises
        ------
        SimulationValidationError
            On an unknown scheme, a missing or invalid initial state.
        UnsupportedProcessError
            For ``exact_transition`` on torch, or with parameters this library
            declines to branch on; raised before any draw is taken.
        """
        if scheme not in CIR_SCHEMES:
            raise SimulationValidationError(f"scheme must be one of {CIR_SCHEMES}; got {scheme!r}.")
        rate = _validate_parameter(
            _require(initial_state, "short_rate"),
            field="initial_state['short_rate']",
            non_negative=True,
        )
        inputs = {
            "initial_state['short_rate']": rate,
            "kappa": self.kappa,
            "theta": self.theta,
            "volatility": self.volatility,
            "time_grid": time_grid,
            "rng": rng,
        }
        namespace = resolve_namespace(inputs)

        degrees_of_freedom = self._precheck(
            namespace, time_grid=time_grid, antithetic=antithetic, scheme=scheme
        )

        device = resolve_device(namespace, inputs)
        dtype = resolve_dtype(namespace, inputs)
        stream = random_stream(rng, namespace=namespace, device=device, dtype=dtype)

        xp = get_namespace(time_grid, rate, self.kappa, self.theta, self.volatility)
        steps = time_grid[1:] - time_grid[:-1]
        n_steps = len(steps)

        rates = self._sample_columns(
            xp,
            stream,
            rate=rate,
            steps=steps,
            n_steps=n_steps,
            n_paths=int(n_paths),
            antithetic=antithetic,
            scheme=scheme,
            degrees_of_freedom=degrees_of_freedom,
        )
        return xp.stack([xp.stack(rates, axis=1)], axis=2)

    # --- the two halves, so a driven configuration reuses them ------------------
    #
    # ``BCC97`` needs a rate path inside a larger sampler rather than a scenario
    # of its own, and it needs the refusals to fire before *its* stream is built
    # too. Splitting ``sample`` here rather than reimplementing either half is
    # what makes "the short rate under BCC97 is this process" a fact about the
    # code and not a claim about two files.

    def _precheck(
        self, namespace: str, *, time_grid: Any, antithetic: bool, scheme: str
    ) -> float | None:
        """Settle everything the exact transition cannot do, before any draw.

        Returns the degrees of freedom when the exact transition was asked for,
        because computing them requires concrete parameters and the sampler
        needs them too; ``None`` otherwise.

        Called before the RNG is built, so a refusal never costs a draw and
        never leaves a caller's generator advanced by a call that failed.
        """
        if scheme != _EXACT:
            return None
        self._reject_exact_on(namespace)
        degrees_of_freedom = self._exact_degrees_of_freedom()
        if antithetic and degrees_of_freedom <= 1.0:
            raise UnsupportedProcessError(
                f"Antithetic sampling is refused for 'exact_transition' at "
                f"{degrees_of_freedom} degrees of freedom. Below one degree of freedom "
                f"the exact transition is a Poisson mixture of chi-squares and contains "
                f"no normal to mirror, so the two halves of every pair would be "
                f"identical: the run would cost twice the draws and reduce no variance. "
                f"Ask for n_paths // 2 paths without antithetic, which produces exactly "
                f"the first half of this request."
            )
        self._reject_traced_grid(time_grid)
        return degrees_of_freedom

    def _sample_columns(
        self,
        xp: Any,
        stream: Any,
        *,
        rate: Any,
        steps: Any,
        n_steps: int,
        n_paths: int,
        antithetic: bool,
        scheme: str,
        degrees_of_freedom: float | None,
    ) -> list[Any]:
        """The rate path as ``n_steps + 1`` columns of shape ``(n_paths,)``.

        Columns rather than a stacked array because the caller that needs this
        rather than :meth:`sample` is stepping a spot alongside it and reads one
        time at a time.
        """
        if scheme == _EXACT:
            assert degrees_of_freedom is not None  # settled by ``_precheck``
            return self._sample_exact(
                xp,
                stream,
                rate=rate,
                steps=steps,
                n_steps=n_steps,
                n_paths=n_paths,
                antithetic=antithetic,
                degrees_of_freedom=degrees_of_freedom,
            )
        return self._sample_discretized(
            xp,
            stream,
            rate=rate,
            steps=steps,
            n_steps=n_steps,
            n_paths=n_paths,
            antithetic=antithetic,
            scheme=scheme,
        )

    # --- schemes ---------------------------------------------------------------

    def _sample_discretized(
        self,
        xp: Any,
        stream: Any,
        *,
        rate: Any,
        steps: Any,
        n_steps: int,
        n_paths: int,
        antithetic: bool,
        scheme: str,
    ) -> list[Any]:
        """One normal per step, drawn as a single block from the caller's stream.

        A single block is the whole point on JAX: a sampler that draws once uses
        the key it was handed, and only a sampler that draws more than once
        splits.  Splitting here would change nothing about the mathematics and
        everything about which numbers come out.
        """
        normals = standard_normal(stream, (n_paths, n_steps), antithetic=antithetic)
        r = xp.zeros((n_paths,), like=normals) + xp.asarray(rate, like=normals)
        kappa = xp.asarray(self.kappa, like=normals)
        theta = xp.asarray(self.theta, like=normals)
        sigma = xp.asarray(self.volatility, like=normals)

        path = [r]
        for index in range(n_steps):
            dt = xp.asarray(steps[index], like=normals)
            z = normals[:, index]
            if scheme == "quadratic_exponential":
                r = quadratic_exponential_step(
                    xp, r, kappa=kappa, theta=theta, xi=sigma, dt=dt, z=z
                )
            else:
                r, _plus, _root = full_truncation_step(
                    xp, r, kappa=kappa, theta=theta, xi=sigma, dt=dt, z=z
                )
            path.append(r)
        return path

    def _sample_exact(
        self,
        xp: Any,
        stream: Any,
        *,
        rate: Any,
        steps: Any,
        n_steps: int,
        n_paths: int,
        antithetic: bool,
        degrees_of_freedom: float,
    ) -> list[Any]:
        """Glasserman (2004) section 3.4, in whichever of its two forms applies.

        The transition law of the CIR state over a step of length
        :math:`\\Delta` is exactly

        .. math:: r_{t+\\Delta} = c \\cdot \\chi'^2_d(\\lambda),
            \\quad c = \\frac{\\sigma^2(1 - e^{-\\kappa\\Delta})}{4\\kappa},
            \\quad d = \\frac{4\\kappa\\theta}{\\sigma^2},
            \\quad \\lambda = \\frac{r_t e^{-\\kappa\\Delta}}{c}

        a non-central chi-square with :math:`d` degrees of freedom.  "Exact"
        means exact *at the grid points* and says nothing whatever about the
        path between them, which matters here because what this process is for
        is an integral over that path.

        The two constructions differ in how much can be drawn in advance, and
        that is what decides the JAX key layout:

        ``d > 1``
            :math:`(Z + \\sqrt\\lambda)^2 + \\chi^2_{d-1}`.  Neither :math:`Z`
            nor the chi-square depends on the state, so both are whole blocks
            drawn up front -- two blocks, two split keys.
        ``d <= 1``
            :math:`\\chi^2_{d + 2N}` with :math:`N \\sim \\text{Poisson}
            (\\lambda/2)`.  Here :math:`\\lambda` depends on the state the step
            starts from, so the draws are unavoidably sequential: two blocks
            per step, and ``2 * n_steps`` split keys.
        """
        kappa = concrete_float(self.kappa)
        sigma = concrete_float(self.volatility)
        assert kappa is not None and sigma is not None  # checked before any draw

        path: list[Any] = []
        d = degrees_of_freedom

        if n_steps == 0:
            # A one-point grid has no transition to draw. The other two schemes
            # return the initial state for it, and returning something else --
            # or failing inside ``split`` -- would make the scheme change what
            # the grid means.
            template = standard_normal(stream, (n_paths, 0))
            return [xp.zeros((n_paths,), like=template) + xp.asarray(rate, like=template)]

        if d > 1.0:
            normal_stream, gamma_stream = split(stream, 2)
            normals = standard_normal(normal_stream, (n_paths, n_steps), antithetic=antithetic)
            chi_rest = 2.0 * _gamma_draws(
                gamma_stream, (n_paths, n_steps), 0.5 * (d - 1.0), antithetic=antithetic
            )
            r = xp.zeros((n_paths,), like=normals) + xp.asarray(rate, like=normals)
            path.append(r)
            for index in range(n_steps):
                c, decay = self._exact_constants(kappa, sigma, steps[index])
                c_arr = xp.asarray(c, like=normals)
                lam = r * (decay / c)
                root = xp.sqrt(xp.maximum(lam, xp.asarray(0.0, like=lam)))
                r = c_arr * ((normals[:, index] + root) ** 2 + chi_rest[:, index])
                path.append(r)
            return path

        streams = split(stream, 2 * n_steps)

        # The first step is peeled rather than folded into the loop, and the
        # reason is the key rule rather than tidiness. Every later step forms
        # its Poisson rate from the state the step starts from, so it needs an
        # array to have been drawn already; the first step does not, because
        # every path starts at the same rate and ``lambda_0`` is therefore a
        # scalar. Drawing a throwaway array to serve as a template would mean
        # drawing from ``streams[0]`` twice.
        c, decay = self._exact_constants(kappa, sigma, steps[0])
        counts = _poisson_draws(streams[0], (n_paths,), 0.5 * rate * (decay / c))
        r = xp.zeros((n_paths,), like=counts) + xp.asarray(rate, like=counts)
        path.append(r)
        r = (
            xp.asarray(c, like=counts)
            * 2.0
            * _gamma_draws(streams[1], (n_paths,), 0.5 * d + counts)
        )
        path.append(r)

        for index in range(1, n_steps):
            c, decay = self._exact_constants(kappa, sigma, steps[index])
            counts = _poisson_draws(streams[2 * index], (n_paths,), 0.5 * r * (decay / c))
            r = (
                xp.asarray(c, like=counts)
                * 2.0
                * _gamma_draws(streams[2 * index + 1], (n_paths,), 0.5 * d + counts)
            )
            path.append(r)
        return path

    @staticmethod
    def _exact_constants(kappa: float, sigma: float, dt: Any) -> tuple[float, float]:
        """``(c, e^{-kappa dt})`` for one step, as host floats.

        Host floats because they are built from parameters this scheme has
        already required to be concrete, and because reading them once per step
        keeps the per-path arithmetic in the backend.
        """
        step = float(dt)
        decay = math.exp(-kappa * step)
        return sigma * sigma * (1.0 - decay) / (4.0 * kappa), decay

    # --- refusals --------------------------------------------------------------

    def _reject_exact_on(self, namespace: str) -> None:
        if namespace == "torch":
            raise UnsupportedProcessError(
                "'exact_transition' needs a gamma sampler bound to the generator it was "
                "given, and torch publishes none: torch.distributions.Gamma.sample takes "
                "no generator and would draw from the global torch stream, making a "
                "seeded run irreproducible. Use 'quadratic_exponential' or "
                "'full_truncation_euler' on torch, both of which need only normals, or "
                "run the exact transition on NumPy or JAX."
            )

    def _reject_traced_grid(self, time_grid: Any) -> None:
        """The step lengths are read as host floats, so the grid must be readable.

        Consistent with the parameter rule above rather than an extra
        restriction: the scheme already needs concrete parameters to choose a
        construction, and it needs concrete step lengths to form ``c`` and the
        decay. Checking here turns what would otherwise surface as a raw
        ``ConcretizationTypeError`` from inside the loop into the same named
        refusal, raised before any draw.
        """
        try:
            last = concrete_float(time_grid[-1])
        except (TypeError, IndexError):  # pragma: no cover - caught downstream
            return
        if last is None:
            raise UnsupportedProcessError(
                "'exact_transition' reads each step length to form the transition's "
                "scale and decay, which cannot be done for a traced time grid. Trace "
                "'quadratic_exponential' or 'full_truncation_euler' instead -- both "
                "take the grid as an array -- or supply a concrete grid."
            )

    def _exact_degrees_of_freedom(self) -> float:
        """``4 kappa theta / sigma^2``, required to be a readable number.

        The two constructions in Glasserman section 3.4 are genuinely different
        formulas and the choice between them is not a rounding detail, so it
        cannot be deferred to a value that may not exist yet.  Under a JAX trace
        of the parameters there is no such value, and rather than pick a branch
        and record the wrong one into the jaxpr this refuses and names the two
        schemes that trace without a branch.
        """
        kappa = concrete_float(self.kappa)
        theta = concrete_float(self.theta)
        sigma = concrete_float(self.volatility)
        if kappa is None or theta is None or sigma is None:
            raise UnsupportedProcessError(
                "'exact_transition' chooses between two constructions by the degrees of "
                "freedom 4*kappa*theta/volatility**2, which cannot be read from traced "
                "parameters. Trace 'quadratic_exponential' or 'full_truncation_euler' "
                "instead -- neither branches on a parameter -- or supply concrete "
                "parameters."
            )
        return 4.0 * kappa * theta / (sigma * sigma)


def _require(initial_state: Mapping[str, Any], name: str) -> Any:
    try:
        return initial_state[name]
    except (KeyError, TypeError):
        raise SimulationValidationError(
            f"CIRShortRate evolves a 'short_rate' state, which the initial state must "
            f"supply; {name!r} is missing from "
            f"{sorted(initial_state) if hasattr(initial_state, 'keys') else '?'}."
        ) from None
