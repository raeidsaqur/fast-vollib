"""Execution records and explicit user-selected trade groupings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping

from ..observations import ObservationStamp
from ..observations.records import text
from ._validate import money, quantity
from .contracts import ListedContract
from .errors import PortfolioError


@dataclass(frozen=True, slots=True, kw_only=True)
class FeeRecord:
    kind: str
    amount: Decimal
    currency: str

    def __post_init__(self) -> None:
        if self.kind not in ("commission", "exchange", "regulatory", "other"):
            raise PortfolioError("unknown fee kind")
        object.__setattr__(self, "amount", money(self.amount, nonnegative=True))
        if (
            not isinstance(self.currency, str)
            or len(self.currency) != 3
            or not self.currency.isupper()
        ):
            raise PortfolioError("fee currency must be an uppercase currency code")


@dataclass(frozen=True, slots=True, kw_only=True)
class Execution:
    contract_key: str
    quantity: int
    premium: Decimal
    execution_id: str
    stamp: ObservationStamp
    source_sequence: int | None = None
    fees: tuple[FeeRecord, ...] = ()
    order_id: str | None = None

    def __post_init__(self) -> None:
        text(self.contract_key, "contract_key")
        text(self.execution_id, "execution_id")
        if self.order_id is not None:
            text(self.order_id, "order_id")
        quantity(self.quantity)
        object.__setattr__(self, "premium", money(self.premium, nonnegative=True))
        if not isinstance(self.stamp, ObservationStamp) or not isinstance(
            self.stamp.event_time, datetime
        ):
            raise PortfolioError("execution requires an instant event stamp")
        if self.source_sequence is not None and (
            type(self.source_sequence) is not int or self.source_sequence < 0
        ):
            raise PortfolioError("source_sequence must be a nonnegative integer")
        fees = tuple(self.fees)
        if any(not isinstance(f, FeeRecord) for f in fees):
            raise PortfolioError("fees must be FeeRecord values")
        object.__setattr__(self, "fees", fees)


@dataclass(frozen=True, slots=True, kw_only=True)
class Trade:
    label: str
    currency: str
    executions: tuple[Execution, ...]
    contracts: Mapping[str, ListedContract]
    completeness: str = "complete"
    inventory_record_id: str | None = None

    def __post_init__(self) -> None:
        text(self.label, "label")
        text(self.currency, "currency")
        if len(self.currency) != 3 or not self.currency.isalpha() or not self.currency.isupper():
            raise PortfolioError("trade currency must be an uppercase three-letter code")
        contracts = dict(self.contracts)
        fills = tuple(self.executions)
        if self.completeness not in ("complete", "positions_only", "partial"):
            raise PortfolioError("unknown trade completeness")
        if len({x.execution_id for x in fills}) != len(fills):
            raise PortfolioError("trade groups unique execution identities, not deliveries")
        if any(
            k != c.key or c.currency != self.currency or c.stamp.world_id != "actual"
            for k, c in contracts.items()
        ):
            raise PortfolioError("trade contracts must agree on key, currency and actual world")
        if len({c.underlier for c in contracts.values()}) > 1:
            raise PortfolioError("trade must have one underlier")
        if any(x.contract_key not in contracts or x.stamp.world_id != "actual" for x in fills):
            raise PortfolioError("trade execution references unknown contract or world")
        if self.completeness == "positions_only" and (fills or not self.inventory_record_id):
            raise PortfolioError("positions-only trade needs inventory_record_id and no fills")
        object.__setattr__(self, "executions", fills)
        object.__setattr__(self, "contracts", MappingProxyType(contracts))
