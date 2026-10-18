"""Deterministic FIFO lots with residual-preserving fees and nominal cash."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from typing import Any

from ..observations.records import digest, text
from ._validate import LEDGER_CONTEXT, allocate, money, quantity
from .errors import PortfolioError


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenLot:
    contract_key: str
    execution_id: str
    quantity: int
    premium: Decimal
    opening_fees: Decimal
    basis: Decimal

    def __post_init__(self) -> None:
        text(self.contract_key, "contract_key")
        text(self.execution_id, "execution_id")
        quantity(self.quantity)
        for field in ("premium", "opening_fees", "basis"):
            object.__setattr__(
                self, field, money(getattr(self, field), nonnegative=field != "basis")
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class LotLedger:
    lots: tuple[OpenLot, ...]
    cumulative_cash: Decimal
    realized: Decimal
    paid_fees: Decimal
    policy: str


def match_lots(
    selected_executions: Any, *, contracts: Any, policy: Any, tie_policy: Any = None
) -> Any:
    """FIFO at Decimal precision 50; ties require source sequence or record-ID policy."""
    from .serialization import execution_to_dict

    if policy != "fifo" or tie_policy not in (None, "execution_id"):
        raise PortfolioError("unsupported lot/tie policy")
    unique: dict[str, Any] = {}
    for fill in selected_executions:
        prior = unique.get(fill.execution_id)
        if prior is not None and digest(execution_to_dict(prior)) != digest(
            execution_to_dict(fill)
        ):
            raise PortfolioError("conflicting execution delivery; resolve observations first")
        unique[fill.execution_id] = fill
    fills = list(unique.values())
    for fill in fills:
        contract = contracts.get(fill.contract_key)
        if contract is None or fill.stamp.world_id != contract.stamp.world_id:
            raise PortfolioError("execution lacks compatible contract terms")
        if any(f.currency != contract.currency for f in fill.fees):
            raise PortfolioError("fee currency disagrees with contract")
    groups: dict[tuple, list[Any]] = {}
    for fill in fills:
        groups.setdefault((fill.contract_key, fill.stamp.event_time), []).append(fill)
    for group in groups.values():
        if len(group) > 1 and tie_policy is None:
            if (
                any(f.source_sequence is None for f in group)
                or len({f.source_sequence for f in group}) != len(group)
                or len({f.stamp.source for f in group}) != 1
            ):
                raise PortfolioError(
                    "equal execution times require unique source_sequence or tie_policy='execution_id'"
                )
    fills.sort(
        key=lambda f: (
            f.stamp.event_time,
            f.execution_id if tie_policy else (f.source_sequence or 0),
            f.execution_id,
        )
    )
    lots: list[Any] = []
    cash, paid = Decimal(0), Decimal(0)
    with localcontext(LEDGER_CONTEXT):
        for fill in fills:
            multiplier = contracts[fill.contract_key].multiplier
            fee = sum((f.amount for f in fill.fees), Decimal(0))
            cash -= fill.quantity * multiplier * fill.premium + fee
            paid += fee
            remaining = abs(fill.quantity)
            sign = 1 if fill.quantity > 0 else -1
            matches = []
            for i, lot in enumerate(lots):
                if remaining and lot.contract_key == fill.contract_key and lot.quantity * sign < 0:
                    size = min(remaining, abs(lot.quantity))
                    matches.append((i, size))
                    remaining -= size
            portions = [size for _, size in matches] + ([remaining] if remaining else [])
            fee_parts = allocate(fee, portions)
            # Closing fees are already in cash. Opening fees remain in signed basis.
            for i, size in matches:
                lot = lots[i]
                residual = abs(lot.quantity) - size
                if residual:
                    released_fee, retained_fee = allocate(lot.opening_fees, (size, residual))
                    basis = (
                        1 if lot.quantity > 0 else -1
                    ) * residual * multiplier * lot.premium + retained_fee
                    lots[i] = replace(
                        lot,
                        quantity=(1 if lot.quantity > 0 else -1) * residual,
                        opening_fees=retained_fee,
                        basis=basis,
                    )
                else:
                    lots[i] = None
            lots = [lot for lot in lots if lot is not None]
            if remaining:
                opening_fee = fee_parts[-1]
                lots.append(
                    OpenLot(
                        contract_key=fill.contract_key,
                        execution_id=fill.execution_id,
                        quantity=sign * remaining,
                        premium=fill.premium,
                        opening_fees=opening_fee,
                        basis=sign * remaining * multiplier * fill.premium + opening_fee,
                    )
                )
        basis = sum((lot.basis for lot in lots), Decimal(0))
        return LotLedger(
            lots=tuple(lots),
            cumulative_cash=cash,
            realized=cash + basis,
            paid_fees=paid,
            policy=f"fifo/1:decimal50:tie={tie_policy or 'source_sequence'}",
        )
