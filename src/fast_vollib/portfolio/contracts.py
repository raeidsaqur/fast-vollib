"""Listed option descriptors preserve source exercise and delivery terms."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from ..instruments import InstrumentRef, OptionType, SettlementType
from ..observations import ObservationStamp
from ..observations.records import text, utc
from ._validate import money
from .errors import PortfolioError, UnsupportedContractError


@dataclass(frozen=True, slots=True, kw_only=True)
class ListedContract:
    key: str
    underlier: InstrumentRef
    option_type: OptionType
    strike: Decimal
    expiry_date: date
    exercise: str
    settlement: SettlementType
    multiplier: Decimal
    deliverable: str
    currency: str
    stamp: ObservationStamp
    expiry_instant: datetime | None = None
    expiry_policy: str | None = None
    source_symbol_id: str | None = None

    def __post_init__(self) -> None:
        text(self.key, "key")
        if self.source_symbol_id is not None:
            text(self.source_symbol_id, "source_symbol_id")
        if not isinstance(self.underlier, InstrumentRef):
            raise PortfolioError("underlier must be an InstrumentRef")
        if not isinstance(self.option_type, OptionType) or not isinstance(
            self.settlement, SettlementType
        ):
            raise PortfolioError("option_type and settlement require their explicit enums")
        if self.exercise not in ("american", "european"):
            raise PortfolioError("unknown source exercise designation")
        for name in ("strike", "multiplier"):
            value = money(getattr(self, name), nonnegative=True)
            if value <= 0:
                raise PortfolioError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if type(self.expiry_date) is not date:
            raise PortfolioError("expiry_date must be a legal date")
        text(self.deliverable, "deliverable")
        if (
            not isinstance(self.currency, str)
            or len(self.currency) != 3
            or not self.currency.isalpha()
            or not self.currency.isupper()
        ):
            raise PortfolioError("currency must be an uppercase three-letter code")
        if self.underlier.currency not in (None, self.currency):
            raise PortfolioError("underlier and premium currency disagree")
        if not isinstance(self.stamp, ObservationStamp):
            raise PortfolioError("contract requires an observation stamp")
        if (self.expiry_instant is None) != (self.expiry_policy is None):
            raise PortfolioError("resolved expiry requires its explicit policy identifier")
        if self.expiry_instant is not None:
            object.__setattr__(self, "expiry_instant", utc(self.expiry_instant))
            text(self.expiry_policy, "expiry_policy")


def require_standard(contract: Any) -> Any:
    """Accept only the explicitly documented standard delivery descriptors."""
    physical = (
        contract.settlement is SettlementType.PHYSICAL
        and contract.deliverable == f"100 {contract.underlier.identifier}"
    )
    cash = (
        contract.settlement is SettlementType.CASH
        and contract.deliverable == "standard_cash_equivalent:100"
    )
    if contract.multiplier != Decimal(100) or not (physical or cash):
        raise UnsupportedContractError(
            "only standard 100-share delivery or explicit standard_cash_equivalent:100 terms are supported"
        )
