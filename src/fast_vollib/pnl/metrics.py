"""Exact rational piecewise-linear analysis over all nonnegative terminal spots."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any

from ..instruments import OptionType
from .errors import PnLError


@dataclass(frozen=True, slots=True, kw_only=True)
class PayoffMetrics:
    breakevens: tuple[Fraction, ...]
    zero_intervals: tuple[tuple[Fraction, Fraction | None], ...]
    minimum_pnl: Fraction | Decimal
    maximum_pnl: Fraction | Decimal
    minimum_regions: tuple[tuple[Fraction, Fraction | None], ...]
    maximum_regions: tuple[tuple[Fraction, Fraction | None], ...]
    tail_slope: Fraction

    def __post_init__(self) -> None:
        roots = tuple(self.breakevens)
        if (
            any(not isinstance(x, Fraction) or x < 0 for x in roots)
            or tuple(sorted(set(roots))) != roots
        ):
            raise PnLError("breakevens must be unique sorted nonnegative rationals")
        object.__setattr__(self, "breakevens", roots)
        for field in ("zero_intervals", "minimum_regions", "maximum_regions"):
            regions = tuple(tuple(x) for x in getattr(self, field))
            for pair in regions:
                if len(pair) != 2:
                    raise PnLError("metric intervals require two endpoints")
                a, b = pair
                if (
                    not isinstance(a, Fraction)
                    or a < 0
                    or (b is not None and (not isinstance(b, Fraction) or b < a))
                ):
                    raise PnLError("invalid metric interval")
            object.__setattr__(self, field, regions)
        for field in ("minimum_pnl", "maximum_pnl"):
            value = getattr(self, field)
            if not isinstance(value, Fraction) and not (
                isinstance(value, Decimal) and value.is_infinite()
            ):
                raise PnLError("extrema require exact rationals or signed unbounded values")
        if self.minimum_pnl > self.maximum_pnl or not isinstance(self.tail_slope, Fraction):
            raise PnLError("invalid extrema or tail slope")

    @property
    def maximum_loss(self) -> Any:
        return max(0, -self.minimum_pnl)

    @property
    def maximum_profit(self) -> Any:
        return max(0, self.maximum_pnl)


def _value_terms(contracts: Any, sizes: Any, cash: Fraction, spot: Fraction) -> Fraction:
    value = cash
    for contract, size in zip(contracts, sizes):
        strike = Fraction(contract.strike)
        intrinsic = (
            max(spot - strike, Fraction(0))
            if contract.option_type is OptionType.CALL
            else max(strike - spot, Fraction(0))
        )
        value += Fraction(size) * intrinsic
    return value


def analytic_metrics(prepared: Any) -> Any:
    """Exact terminal metrics, sharing the same payoff terms as terminal laws."""
    legs = prepared.snapshot.position.legs
    cash = Fraction(prepared.snapshot.cumulative_cash) - Fraction(prepared.future_costs)
    return _analyze_terms(tuple(x.contract for x in legs), tuple(x.signed_size for x in legs), cash)


def _analyze_terms(contracts: Any, sizes: Any, cash: Fraction) -> Any:
    """Use Decimal inputs as exact rationals; the plotting grid is never consulted."""
    knots = sorted({Fraction(0), *(Fraction(c.strike) for c in contracts)})

    def at(spot: Any) -> Any:
        return _value_terms(contracts, sizes, cash, spot)

    values = [at(k) for k in knots]
    tail = sum(
        (
            Fraction(size)
            for contract, size in zip(contracts, sizes)
            if contract.option_type is OptionType.CALL
        ),
        Fraction(0),
    )
    roots: set[Fraction] = set()
    zeros: list[tuple[Fraction, Fraction | None]] = []
    for a, b, va, vb in zip(knots, knots[1:], values, values[1:]):
        if va == vb == 0:
            zeros.append((a, b))
        elif va == 0:
            roots.add(a)
        elif vb == 0:
            roots.add(b)
        elif va * vb < 0:
            roots.add(a - va * (b - a) / (vb - va))
    if tail == 0 and values[-1] == 0:
        zeros.append((knots[-1], None))
    elif tail != 0:
        root = knots[-1] - values[-1] / tail
        if root >= knots[-1]:
            roots.add(root)
    elif values[-1] == 0:
        roots.add(knots[-1])

    def merge(regions: Any) -> Any:
        result: list[tuple[Fraction, Fraction | None]] = []
        for a, b in sorted(regions, key=lambda x: x[0]):
            if result and result[-1][1] == a:
                result[-1] = (result[-1][0], b)
            else:
                result.append((a, b))
        return tuple(result)

    zeros = merge(zeros)
    isolated_roots = tuple(
        sorted(r for r in roots if not any(a <= r and (b is None or r <= b) for a, b in zeros))
    )
    minimum = Decimal("-Infinity") if tail < 0 else min(values)
    maximum = Decimal("Infinity") if tail > 0 else max(values)

    def regions(extreme: Any) -> Any:
        if isinstance(extreme, Decimal) and not extreme.is_finite():
            return ()
        flats: list[tuple[Fraction, Fraction | None]] = [
            (a, b)
            for a, b, va, vb in zip(knots, knots[1:], values, values[1:])
            if va == vb == extreme
        ]
        if tail == 0 and values[-1] == extreme:
            flats.append((knots[-1], None))
        points = [
            (k, k)
            for k, v in zip(knots, values)
            if v == extreme and not any(a <= k and (b is None or k <= b) for a, b in flats)
        ]
        return merge(flats + points)

    return PayoffMetrics(
        breakevens=isolated_roots,
        zero_intervals=tuple(zeros),
        minimum_pnl=minimum,
        maximum_pnl=maximum,
        minimum_regions=regions(minimum),
        maximum_regions=regions(maximum),
        tail_slope=tail,
    )
