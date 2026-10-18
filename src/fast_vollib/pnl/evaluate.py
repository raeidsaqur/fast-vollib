"""Prepare immutable terminal coefficients once; evaluate through native payoff dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from .._array_api import concrete_bool, get_namespace
from ..instruments import EuropeanOption, SettlementType, payoff
from ..instruments.payoffs import _ScenarioInput
from ..portfolio import ListedContract, PositionSnapshot, year_fraction
from ..portfolio._validate import money
from ..portfolio.contracts import require_standard
from ..portfolio.errors import IncompleteBasisError
from .errors import PnLAssumptionError, PnLError, PnLHorizonError

if TYPE_CHECKING:
    from .results import PnLProfile


@dataclass(frozen=True, slots=True, kw_only=True)
class CoefficientConversion:
    label: str
    exact: Fraction
    converted: float
    absolute_error: Fraction

    def __post_init__(self) -> None:
        if (
            not isinstance(self.exact, Fraction)
            or not isinstance(self.absolute_error, Fraction)
            or not np.isfinite(self.converted)
        ):
            raise PnLError("coefficient conversion requires exact rational provenance")
        if self.absolute_error != abs(self.exact - Fraction(self.converted)):
            raise PnLError("coefficient conversion error disagrees with its exact input")


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedTerminal:
    snapshot: PositionSnapshot
    terminal_assumption: str
    future_costs: Decimal
    dtype: str
    convention: str | None
    units: tuple[EuropeanOption, ...]
    sizes: tuple[float, ...]
    cash: float
    costs: float
    conversion_errors: tuple[CoefficientConversion, ...]
    caveat: str | None

    def conversion_error_bound(self, spot: Any) -> Any:
        """Exact coefficient-only error bound at each supplied finite host spot.

        This excludes arithmetic rounding in payoff evaluation and summation.
        There is no finite global bound asserted over an unbounded spot domain.
        """
        values = np.asarray(spot)
        if not np.isfinite(values).all() or (values < 0).any():
            raise PnLError("conversion bound needs finite nonnegative spots")
        errors = {x.label: x.absolute_error for x in self.conversion_errors}
        bounds = []
        for value in values.flat:
            s = Fraction(float(value))
            bound = errors["cash"] + errors["future_costs"]
            for i, leg in enumerate(self.snapshot.position.legs):
                k = Fraction(leg.contract.strike)
                intrinsic = (
                    max(s - k, Fraction(0))
                    if leg.contract.option_type.value == "call"
                    else max(k - s, Fraction(0))
                )
                bound += (
                    errors[f"size:{i}"] * intrinsic
                    + abs(Fraction(self.sizes[i])) * errors[f"strike:{i}"]
                )
            bounds.append(np.nextafter(float(bound), np.inf) if bound else 0.0)
        return np.asarray(bounds).reshape(values.shape)


def _unit_contract(contract: ListedContract, *, maturity: float, strike: float) -> EuropeanOption:
    """Intrinsic-shape projection; the source descriptor retains exercise terms."""
    return EuropeanOption(
        underlier=contract.underlier,
        option_type=contract.option_type,
        strike=strike,
        maturity=maturity,
        notional=1,
    )


def prepare_terminal(
    snapshot: PositionSnapshot,
    *,
    terminal_assumption: Any,
    future_costs: Any = Decimal("0"),
    dtype: str = "float64",
    convention: str | None = None,
) -> PreparedTerminal:
    """Prepare standard intrinsic-equivalent terminal P&L with explicit assumptions."""
    if snapshot.cumulative_cash is None:
        raise IncompleteBasisError("terminal trade P&L requires complete entry cash")
    if dtype not in ("float32", "float64"):
        raise PnLError("terminal dtype must be float32 or float64")
    costs = money(future_costs, nonnegative=True)
    if terminal_assumption not in (
        "standard_intrinsic_equivalent",
        "hold_to_expiry_no_early_exercise",
    ):
        raise PnLAssumptionError("an explicit supported terminal assumption is required")
    legs = snapshot.position.legs
    if len(snapshot.position.maturities) > 1:
        raise PnLHorizonError("terminal P&L requires a common live expiry")
    if convention is not None:
        year_fraction(snapshot.valuation_time, snapshot.valuation_time, convention=convention)
    caveats = (
        {
            "Hold-to-expiry intrinsic-equivalent illustration; early exercise, assignment, physical delivery, funding, ex-dividend exercise and pin risk are not modelled."
        }
        if terminal_assumption == "hold_to_expiry_no_early_exercise"
        else set()
    )
    for leg in legs:
        c = leg.contract
        require_standard(c)
        if c.exercise == "american" or c.settlement is SettlementType.PHYSICAL:
            if terminal_assumption != "hold_to_expiry_no_early_exercise":
                raise PnLAssumptionError(
                    "American/physical terms require hold_to_expiry_no_early_exercise"
                )
            caveats.add(
                "Hold-to-expiry intrinsic-equivalent illustration; early exercise, assignment, physical delivery, funding, ex-dividend exercise and pin risk are not modelled."
            )
        if c.expiry_instant is not None:
            if snapshot.valuation_time > c.expiry_instant:
                raise PnLHorizonError(
                    "valuation is after expiry; reconcile lifecycle evidence first"
                )
        elif snapshot.valuation_time.date() >= c.expiry_date:
            raise PnLHorizonError(
                "expiry-day or later valuation requires an explicitly resolved instant"
            )
    conversions = []

    def convert(label: Any, value: Any) -> Any:
        with np.errstate(over="ignore", under="ignore"):
            out = float(np.asarray(str(value), dtype=dtype))
        if not np.isfinite(out):
            raise PnLError("nonfinite coefficient conversion")
        conversions.append(
            CoefficientConversion(
                label=label,
                exact=Fraction(value),
                converted=out,
                absolute_error=abs(Fraction(value) - Fraction(out)),
            )
        )
        return out

    units, sizes = [], []
    for i, leg in enumerate(legs):
        strike = convert(f"strike:{i}", leg.contract.strike)
        if strike <= 0:
            raise PnLError("strike underflows the prepared dtype")
        units.append(_unit_contract(leg.contract, maturity=0, strike=strike))
        sizes.append(convert(f"size:{i}", leg.signed_size))
    cash = convert("cash", snapshot.cumulative_cash)
    future = convert("future_costs", costs)
    return PreparedTerminal(
        snapshot=snapshot,
        terminal_assumption=terminal_assumption,
        future_costs=costs,
        dtype=dtype,
        convention=convention,
        units=tuple(units),
        sizes=tuple(sizes),
        cash=cash,
        costs=future,
        conversion_errors=tuple(conversions),
        caveat=" ".join(sorted(caveats)) or None,
    )


def _checked_spot(prepared: Any, spot: Any) -> Any:
    dtype = getattr(spot, "dtype", None)
    if str(dtype).removeprefix("torch.") != prepared.dtype:
        raise PnLError(f"spot must have prepared floating dtype {prepared.dtype}")
    ns = get_namespace(spot)
    valid = ns.isfinite(spot) & (spot >= 0)
    if concrete_bool(ns.all(valid)) is False:
        raise PnLError("terminal spots must be finite and nonnegative")
    return ns, valid


def _evaluate(prepared: Any, spot: Any, scenario: Any = None) -> Any:
    ns, valid = _checked_spot(prepared, spot)
    # Zero slope stays connected even for an empty inventory.
    total = spot * 0.0 + ns.scalar(prepared.cash, like=spot) - ns.scalar(prepared.costs, like=spot)
    for i, (leg, unit, size) in enumerate(
        zip(prepared.snapshot.position.legs, prepared.units, prepared.sizes)
    ):
        if scenario is None:
            value = payoff(unit, spot)
        else:
            c = leg.contract
            if c.expiry_instant is None or prepared.convention is None:
                raise PnLHorizonError(
                    "Scenario evaluation requires resolved expiries and an explicit day count"
                )
            maturity = year_fraction(
                prepared.snapshot.valuation_time, c.expiry_instant, convention=prepared.convention
            )
            value = scenario.payoff(_unit_contract(c, maturity=maturity, strike=unit.strike))
        total = total + ns.scalar(size, like=spot) * value
    return ns.where(valid, total, ns.scalar(float("nan"), like=spot))


def terminal_pnl(prepared: PreparedTerminal, spot: Any) -> Any:
    """Native shape/dtype/device and spot gradients, using existing payoff dispatch.

    Concrete invalid values raise. JAX traced invalid entries yield NaN.
    At strikes, the backend's clip subgradient applies (and may differ by backend).
    """
    if isinstance(spot, _ScenarioInput):
        from ..simulation import Scenario

        if not isinstance(spot, Scenario):
            raise PnLError("expected a validated Scenario")
        return _evaluate(prepared, spot.terminal("spot"), scenario=spot)
    return _evaluate(prepared, spot)


def expiry_profile(
    snapshot: PositionSnapshot,
    spot: Any,
    *,
    terminal_assumption: Any,
    future_costs: Any = Decimal("0"),
    convention: str | None = None,
) -> PnLProfile:
    """Owned float64 host report; exact metrics cover S >= 0 independently of spot."""
    from ..observations.records import digest
    from ..portfolio.serialization import snapshot_to_dict
    from .metrics import analytic_metrics
    from .results import PnLProfile, ResultMeta

    grid = np.array(spot, dtype=np.float64, copy=True)
    if grid.ndim != 1 or grid.size == 0:
        raise PnLError("profile spot must be a nonempty one-dimensional grid")
    prepared = prepare_terminal(
        snapshot,
        terminal_assumption=terminal_assumption,
        future_costs=future_costs,
        convention=convention,
    )
    total = terminal_pnl(prepared, grid)[None, :]
    columns = [size * payoff(unit, grid) for size, unit in zip(prepared.sizes, prepared.units)]
    leg_values = np.stack(columns, axis=-1)[None, :, :] if columns else np.empty((1, grid.size, 0))
    horizon = None
    expiries = {leg.contract.expiry_instant for leg in snapshot.position.legs}
    policies = tuple(
        sorted(
            {
                leg.contract.expiry_policy
                for leg in snapshot.position.legs
                if leg.contract.expiry_policy
            }
        )
    )
    if (
        snapshot.position.legs
        and None not in expiries
        and len(expiries) == 1
        and convention is not None
    ):
        expiry = next(iter(expiries))
        assert expiry is not None
        horizon = year_fraction(snapshot.valuation_time, expiry, convention=convention)
    if len(expiries - {None}) > 1:
        raise PnLHorizonError("live legs disagree on resolved expiry instant")
    meta = ResultMeta(
        origin=snapshot.origin,
        basis="expiry_intrinsic",
        valuation_time=snapshot.valuation_time,
        knowledge_cutoff=snapshot.knowledge_cutoff,
        knowledge_label=snapshot.knowledge_label,
        world_id=snapshot.world_id,
        observations_digest=snapshot.observations_digest,
        currency=snapshot.currency,
        caveat=prepared.caveat,
        terminal_assumption=terminal_assumption,
        accounting_policy=snapshot.accounting_policy,
        time_policy=convention,
        expiry_policies=policies,
        input_digest=digest(
            {
                "snapshot": snapshot_to_dict(snapshot),
                "spot": grid.tolist(),
                "terminal_assumption": terminal_assumption,
                "future_costs": str(prepared.future_costs),
                "convention": convention,
            }
        ),
    )
    return PnLProfile(
        horizons=(horizon,),
        target_date=snapshot.position.maturities[0] if snapshot.position.legs else None,
        spot=grid,
        leg_values=leg_values,
        total=total,
        valid=np.ones(total.shape, dtype=bool),
        leg_valid=np.ones(leg_values.shape, dtype=bool),
        cumulative_cash=snapshot.cumulative_cash,
        paid_fees=snapshot.paid_fees,
        future_costs=prepared.future_costs,
        leg_ids=tuple(x.leg_id for x in snapshot.position.legs),
        meta=meta,
        metrics=analytic_metrics(prepared),
        strikes=tuple(x.contract.strike for x in snapshot.position.legs),
        contracts=tuple(x.contract for x in snapshot.position.legs),
        signed_sizes=tuple(x.signed_size for x in snapshot.position.legs),
        conversion_errors=prepared.conversion_errors,
        conversion_bound=prepared.conversion_error_bound(grid),
    )
