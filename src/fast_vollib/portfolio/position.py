"""Owned positions and separate hypothetical/executed snapshot factories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, localcontext
from typing import Any

from ..observations import ObservationSnapshot
from ..observations.records import digest, text, utc
from ._validate import LEDGER_CONTEXT, money, quantity
from .contracts import ListedContract
from .errors import IncompleteBasisError, PortfolioError


@dataclass(frozen=True, slots=True, kw_only=True)
class OptionLeg:
    contract: ListedContract
    quantity: int
    allocated_entry: Decimal | None
    allocated_fees: Decimal
    leg_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.contract, ListedContract):
            raise PortfolioError("leg needs a listed contract")
        quantity(self.quantity)
        text(self.leg_id, "leg_id")
        if self.allocated_entry is not None:
            object.__setattr__(
                self, "allocated_entry", money(self.allocated_entry, nonnegative=True)
            )
        object.__setattr__(self, "allocated_fees", money(self.allocated_fees, nonnegative=True))

    @property
    def signed_size(self) -> Any:
        with localcontext(LEDGER_CONTEXT):
            return self.quantity * self.contract.multiplier


@dataclass(frozen=True, slots=True, kw_only=True)
class Position:
    legs: tuple[OptionLeg, ...]

    def __post_init__(self) -> None:
        legs = tuple(self.legs)
        if any(not isinstance(x, OptionLeg) for x in legs):
            raise PortfolioError("position requires OptionLeg records")
        if len({x.leg_id for x in legs}) != len(legs):
            raise PortfolioError("leg IDs must be unique")
        if (
            len(
                {
                    (x.contract.underlier, x.contract.currency, x.contract.stamp.world_id)
                    for x in legs
                }
            )
            > 1
        ):
            raise PortfolioError("position must share underlier, currency and world")
        terms: dict[str, ListedContract] = {}
        for leg in legs:
            if leg.contract.key in terms and terms[leg.contract.key] != leg.contract:
                raise PortfolioError("same contract key has conflicting terms")
            terms[leg.contract.key] = leg.contract
        object.__setattr__(self, "legs", legs)

    @classmethod
    def from_legs(cls, legs: Any) -> Any:
        return cls(legs=tuple(legs))

    @property
    def underlier(self) -> Any:
        return self.legs[0].contract.underlier if self.legs else None

    @property
    def maturities(self) -> Any:
        return tuple(sorted({x.contract.expiry_date for x in self.legs}))

    @property
    def is_single_expiry(self) -> Any:
        return len(self.maturities) == 1


@dataclass(frozen=True, slots=True, kw_only=True)
class PositionSnapshot:
    position: Position
    lots: tuple
    cumulative_cash: Decimal | None
    realized: Decimal | None
    paid_fees: Decimal
    valuation_time: datetime
    knowledge_cutoff: datetime
    knowledge_label: str
    observations_digest: str
    world_id: str
    origin: str
    accounting_policy: str
    completeness: str
    currency: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "valuation_time", utc(self.valuation_time))
        object.__setattr__(self, "knowledge_cutoff", utc(self.knowledge_cutoff))
        for name in (
            "knowledge_label",
            "observations_digest",
            "world_id",
            "origin",
            "accounting_policy",
            "completeness",
            "currency",
        ):
            text(getattr(self, name), name)
        object.__setattr__(self, "lots", tuple(self.lots))
        from .lots import OpenLot

        if not isinstance(self.position, Position) or any(
            not isinstance(x, OpenLot) for x in self.lots
        ):
            raise PortfolioError("snapshot requires owned position and OpenLot records")
        for field in ("cumulative_cash", "realized", "paid_fees"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, money(value, nonnegative=field == "paid_fees"))
        if self.completeness not in ("complete", "positions_only", "partial"):
            raise PortfolioError("unknown completeness")
        if self.completeness == "complete" and (
            self.cumulative_cash is None or self.realized is None
        ):
            raise PortfolioError("complete snapshot requires cash and realized P&L")
        if self.origin == "positions_only" and self.completeness != "positions_only":
            raise PortfolioError("inventory-only origin requires unavailable entry basis")
        if self.origin == "executed":
            if len(self.lots) != len(self.position.legs):
                raise PortfolioError("executed inventory must correspond to its lots")
            for lot, leg in zip(self.lots, self.position.legs):
                if (lot.contract_key, lot.quantity, lot.premium, lot.opening_fees) != (
                    leg.contract.key,
                    leg.quantity,
                    leg.allocated_entry,
                    leg.allocated_fees,
                ):
                    raise PortfolioError("snapshot lots and legs disagree")
            if self.completeness == "complete":
                assert self.cumulative_cash is not None
            with localcontext(LEDGER_CONTEXT):
                if (
                    self.completeness == "complete"
                    and self.cumulative_cash is not None
                    and self.realized
                    != self.cumulative_cash + sum((x.basis for x in self.lots), Decimal(0))
                ):
                    raise PortfolioError("snapshot cash/basis identity is inconsistent")
        expected = (
            "as_known" if self.knowledge_cutoff <= self.valuation_time else "latest_corrected"
        )
        if self.knowledge_label != expected:
            raise PortfolioError("knowledge label disagrees with valuation/cutoff")
        if self.origin not in ("executed", "hypothetical", "positions_only"):
            raise PortfolioError("unknown position origin")
        if self.origin == "hypothetical" and self.world_id == "actual":
            raise PortfolioError("hypothetical position requires a non-actual world")
        if self.origin == "executed" and self.world_id != "actual":
            raise PortfolioError("executed snapshot requires actual evidence")
        if any(
            x.contract.stamp.world_id != self.world_id or x.contract.currency != self.currency
            for x in self.position.legs
        ):
            raise PortfolioError("snapshot world/currency conflicts with its legs")
        if self.completeness != "complete" and (
            self.cumulative_cash is not None or self.realized is not None
        ):
            raise PortfolioError("incomplete basis must remain unavailable")

    @property
    def open_basis(self) -> Any:
        if self.cumulative_cash is None or self.realized is None:
            return None
        with localcontext(LEDGER_CONTEXT):
            return self.realized - self.cumulative_cash

    def accounting(self, inventory_value: Any) -> Any:
        """Return (realized, unrealized, total) before any estimated future cost."""
        if self.open_basis is None:
            raise IncompleteBasisError("entry basis is unavailable")
        with localcontext(LEDGER_CONTEXT):
            inventory_value = money(inventory_value)
            return (
                self.realized,
                inventory_value - self.open_basis,
                self.cumulative_cash + inventory_value,
            )


def _label(valuation: Any, cutoff: Any) -> Any:
    return "as_known" if cutoff <= valuation else "latest_corrected"


def hypothetical_snapshot(
    position: Position, *, valuation_time: datetime, knowledge_cutoff: datetime, world_id: str
) -> PositionSnapshot:
    from .serialization import position_to_dict

    if world_id == "actual" or any(x.contract.stamp.world_id != world_id for x in position.legs):
        raise PortfolioError("hypothetical snapshot requires one explicit non-actual world")
    valuation, cutoff = utc(valuation_time), utc(knowledge_cutoff)
    with localcontext(LEDGER_CONTEXT):
        fees, cash = Decimal(0), Decimal(0)
        for leg in position.legs:
            if leg.allocated_entry is None:
                raise IncompleteBasisError("hypothetical entry premium must be explicit")
            if leg.contract.stamp.available_time > cutoff:
                raise PortfolioError("contract is unavailable at the cutoff")
            cash -= leg.signed_size * leg.allocated_entry + leg.allocated_fees
            fees += leg.allocated_fees
    currency = position.legs[0].contract.currency if position.legs else None
    if currency is None:
        raise PortfolioError("an empty hypothetical entry has no declared currency")
    return PositionSnapshot(
        position=position,
        lots=(),
        cumulative_cash=cash,
        realized=Decimal(0),
        paid_fees=fees,
        valuation_time=valuation,
        knowledge_cutoff=cutoff,
        knowledge_label=_label(valuation, cutoff),
        observations_digest=digest(position_to_dict(position)),
        world_id=world_id,
        origin="hypothetical",
        accounting_policy="nominal_entry/1:decimal50",
        completeness="complete",
        currency=currency,
    )


def position_snapshot(
    trade: Any, *, observations: Any, valuation_time: Any, policy: Any, tie_policy: Any = None
) -> Any:
    """Resolve records first, then retain fills through the economic valuation time."""
    from .lots import match_lots
    from .serialization import contract_from_dict, execution_from_dict, position_from_dict

    if not isinstance(observations, ObservationSnapshot) or observations.world_id != "actual":
        raise PortfolioError("executed snapshots require an actual observation snapshot")
    valuation = utc(valuation_time)
    cutoff = observations.knowledge_cutoff
    contracts, executions = {}, {}
    inventory = None
    for record in observations.selected_records:
        if record.kind == "listed_contract":
            if record.payload_schema != "portfolio.contract/1":
                raise PortfolioError("unsupported contract observation codec")
            contract = contract_from_dict(dict(record.payload))
            if record.stamp != contract.stamp or record.entities != (contract.key,):
                raise PortfolioError("contract record lineage disagrees with payload")
            if contract.key in contracts:
                raise PortfolioError("multiple unlinked contract versions")
            contracts[contract.key] = contract
        elif record.kind == "execution":
            if record.payload_schema != "portfolio.execution/1":
                raise PortfolioError("unsupported execution observation codec")
            fill = execution_from_dict(dict(record.payload))
            if record.stamp != fill.stamp or record.entities != (
                fill.execution_id,
                fill.contract_key,
            ):
                raise PortfolioError("execution record lineage disagrees with payload")
            if fill.execution_id in executions:
                raise PortfolioError("multiple unlinked execution versions")
            executions[fill.execution_id] = fill
        elif record.kind == "position" and record.record_id == trade.inventory_record_id:
            if record.payload_schema != "portfolio.position/1":
                raise PortfolioError("unsupported position observation codec")
            inventory = position_from_dict(dict(record.payload))
            if any(x.contract.stamp.available_time > cutoff for x in inventory.legs):
                raise PortfolioError("inventory embeds unavailable contract terms")
    if trade.completeness == "positions_only":
        if inventory is None:
            raise IncompleteBasisError(
                "positions-only inventory requires selected position evidence"
            )
        return PositionSnapshot(
            position=inventory,
            lots=(),
            cumulative_cash=None,
            realized=None,
            paid_fees=Decimal(0),
            valuation_time=valuation,
            knowledge_cutoff=cutoff,
            knowledge_label=_label(valuation, cutoff),
            observations_digest=observations.digest,
            world_id="actual",
            origin="positions_only",
            accounting_policy="inventory_only/1",
            completeness="positions_only",
            currency=trade.currency,
        )
    selected = []
    for anchor in trade.executions:
        candidate = executions.get(anchor.execution_id)
        if candidate is None:
            if anchor.stamp.available_time <= cutoff and trade.completeness == "complete":
                raise IncompleteBasisError("missing referenced execution evidence")
            continue
        fill = candidate
        if fill.contract_key != anchor.contract_key:
            raise PortfolioError("execution changes economic identity")
        event_time = fill.stamp.event_time
        if not isinstance(event_time, datetime):
            raise PortfolioError("execution event must be an instant")
        if event_time > valuation:
            continue
        if fill.contract_key not in contracts:
            raise IncompleteBasisError("missing referenced contract evidence")
        selected.append(fill)
    resolved = {k: c for k, c in contracts.items() if k in trade.contracts}
    if any(
        c.currency != trade.currency or c.underlier != trade.contracts[k].underlier
        for k, c in resolved.items()
    ):
        raise PortfolioError("selected contract terms change trade currency/underlier")
    ledger = match_lots(selected, contracts=resolved, policy=policy, tie_policy=tie_policy)
    legs = tuple(
        OptionLeg(
            contract=resolved[lot.contract_key],
            quantity=lot.quantity,
            allocated_entry=lot.premium,
            allocated_fees=lot.opening_fees,
            leg_id=f"lot:{lot.execution_id}",
        )
        for lot in ledger.lots
    )
    complete = trade.completeness == "complete"
    return PositionSnapshot(
        position=Position(legs=legs),
        lots=ledger.lots,
        cumulative_cash=ledger.cumulative_cash if complete else None,
        realized=ledger.realized if complete else None,
        paid_fees=ledger.paid_fees,
        valuation_time=valuation,
        knowledge_cutoff=cutoff,
        knowledge_label=_label(valuation, cutoff),
        observations_digest=observations.digest,
        world_id="actual",
        origin="executed",
        accounting_policy=ledger.policy,
        completeness=trade.completeness,
        currency=trade.currency,
    )
