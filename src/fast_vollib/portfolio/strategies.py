"""Explicit role-based hypothetical entries; no symbol parsing or inferred fills."""

from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal, localcontext
from typing import Any

from ..instruments import OptionType
from ._validate import LEDGER_CONTEXT, allocate, money, quantity as check_quantity
from .contracts import ListedContract
from .errors import PortfolioError
from .position import OptionLeg, Position


@dataclass(frozen=True, slots=True, kw_only=True)
class IronCondorPrices:
    put_long: Decimal
    put_short: Decimal
    call_short: Decimal
    call_long: Decimal

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, money(getattr(self, field.name), nonnegative=True))


@dataclass(frozen=True, slots=True, kw_only=True)
class VerticalPrices:
    long: Decimal
    short: Decimal

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, money(getattr(self, field.name), nonnegative=True))


@dataclass(frozen=True, slots=True, kw_only=True)
class ButterflyPrices:
    lower: Decimal
    middle: Decimal
    upper: Decimal

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, money(getattr(self, field.name), nonnegative=True))


@dataclass(frozen=True, slots=True, kw_only=True)
class StraddlePrices:
    put: Decimal
    call: Decimal

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, money(getattr(self, field.name), nonnegative=True))


StranglePrices = StraddlePrices
CalendarPrices = VerticalPrices
DiagonalPrices = VerticalPrices
IronButterflyPrices = IronCondorPrices


def _build(
    roles: Any, prices: Any, *, quantity: int, fees: Decimal, common_expiry: Any = True
) -> Position:
    check_quantity(quantity, positive=True)
    contracts = [c for _, c, _ in roles]
    if len({c.key for c in contracts}) != len(contracts):
        raise PortfolioError("strategy requires distinct contract keys")
    if common_expiry and len({c.expiry_date for c in contracts}) != 1:
        raise PortfolioError("strategy requires a common expiry")
    portions = allocate(fees, [abs(n) * quantity for _, _, n in roles])
    with localcontext(LEDGER_CONTEXT):
        return Position(
            legs=tuple(
                OptionLeg(
                    contract=c,
                    quantity=n * quantity,
                    allocated_entry=money(getattr(prices, role), nonnegative=True),
                    allocated_fees=fee,
                    leg_id=role,
                )
                for (role, c, n), fee in zip(roles, portions)
            )
        )


def _kind(contract: ListedContract, kind: OptionType) -> None:
    if contract.option_type is not kind:
        raise PortfolioError("contract option type disagrees with strategy role")


def iron_condor(
    *,
    put_long: ListedContract,
    put_short: ListedContract,
    call_short: ListedContract,
    call_long: ListedContract,
    quantity: int,
    prices: Any,
    fees: Decimal,
) -> Position:
    for c in (put_long, put_short):
        _kind(c, OptionType.PUT)
    for c in (call_short, call_long):
        _kind(c, OptionType.CALL)
    if not put_long.strike < put_short.strike < call_short.strike < call_long.strike:
        raise PortfolioError("iron condor requires ordered strikes")
    return _build(
        (
            ("put_long", put_long, 1),
            ("put_short", put_short, -1),
            ("call_short", call_short, -1),
            ("call_long", call_long, 1),
        ),
        prices,
        quantity=quantity,
        fees=fees,
    )


def iron_butterfly(
    *,
    put_long: ListedContract,
    put_short: ListedContract,
    call_short: ListedContract,
    call_long: ListedContract,
    quantity: int,
    prices: Any,
    fees: Decimal,
) -> Position:
    for c in (put_long, put_short):
        _kind(c, OptionType.PUT)
    for c in (call_short, call_long):
        _kind(c, OptionType.CALL)
    if not put_long.strike < put_short.strike == call_short.strike < call_long.strike:
        raise PortfolioError("iron butterfly requires equal short strikes inside its wings")
    return _build(
        (
            ("put_long", put_long, 1),
            ("put_short", put_short, -1),
            ("call_short", call_short, -1),
            ("call_long", call_long, 1),
        ),
        prices,
        quantity=quantity,
        fees=fees,
    )


def butterfly(
    *,
    lower: ListedContract,
    middle: ListedContract,
    upper: ListedContract,
    quantity: int,
    prices: Any,
    fees: Decimal,
) -> Position:
    if (
        len({c.option_type for c in (lower, middle, upper)}) != 1
        or not lower.strike < middle.strike < upper.strike
    ):
        raise PortfolioError("butterfly requires one option type and ordered strikes")
    return _build(
        (("lower", lower, 1), ("middle", middle, -2), ("upper", upper, 1)),
        prices,
        quantity=quantity,
        fees=fees,
    )


def vertical_spread(
    *, long: ListedContract, short: ListedContract, quantity: int, prices: Any, fees: Decimal
) -> Position:
    if long.option_type is not short.option_type or long.strike == short.strike:
        raise PortfolioError("vertical requires matching type and distinct strikes")
    return _build((("long", long, 1), ("short", short, -1)), prices, quantity=quantity, fees=fees)


def straddle(
    *, put: ListedContract, call: ListedContract, quantity: int, prices: Any, fees: Decimal
) -> Position:
    _kind(put, OptionType.PUT)
    _kind(call, OptionType.CALL)
    if put.strike != call.strike:
        raise PortfolioError("straddle requires equal strikes")
    return _build((("put", put, 1), ("call", call, 1)), prices, quantity=quantity, fees=fees)


def strangle(
    *, put: ListedContract, call: ListedContract, quantity: int, prices: Any, fees: Decimal
) -> Position:
    _kind(put, OptionType.PUT)
    _kind(call, OptionType.CALL)
    if put.strike >= call.strike:
        raise PortfolioError("strangle requires put strike below call strike")
    return _build((("put", put, 1), ("call", call, 1)), prices, quantity=quantity, fees=fees)


def calendar_spread(
    *, long: ListedContract, short: ListedContract, quantity: int, prices: Any, fees: Decimal
) -> Position:
    if (
        long.option_type is not short.option_type
        or long.strike != short.strike
        or long.expiry_date <= short.expiry_date
    ):
        raise PortfolioError("calendar requires matching strike/type and later long expiry")
    return _build(
        (("long", long, 1), ("short", short, -1)),
        prices,
        quantity=quantity,
        fees=fees,
        common_expiry=False,
    )


def diagonal_spread(
    *, long: ListedContract, short: ListedContract, quantity: int, prices: Any, fees: Decimal
) -> Position:
    if (
        long.option_type is not short.option_type
        or long.strike == short.strike
        or long.expiry_date <= short.expiry_date
    ):
        raise PortfolioError(
            "diagonal requires distinct strikes, matching type and later long expiry"
        )
    return _build(
        (("long", long, 1), ("short", short, -1)),
        prices,
        quantity=quantity,
        fees=fees,
        common_expiry=False,
    )
