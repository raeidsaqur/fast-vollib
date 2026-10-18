"""Listed options, explicit strategies and evidence-derived Decimal accounting."""

from __future__ import annotations

from .contracts import ListedContract
from .daycount import resolve_expiry_instant, year_fraction
from .errors import IncompleteBasisError, PortfolioError, UnsupportedContractError
from .lots import LotLedger, OpenLot, match_lots
from .position import (
    OptionLeg,
    Position,
    PositionSnapshot,
    hypothetical_snapshot,
    position_snapshot,
)
from .strategies import (
    ButterflyPrices,
    CalendarPrices,
    DiagonalPrices,
    IronButterflyPrices,
    IronCondorPrices,
    StraddlePrices,
    StranglePrices,
    VerticalPrices,
    butterfly,
    calendar_spread,
    diagonal_spread,
    iron_butterfly,
    iron_condor,
    straddle,
    strangle,
    vertical_spread,
)
from .trades import Execution, FeeRecord, Trade

__all__ = [
    "ListedContract",
    "Execution",
    "FeeRecord",
    "Trade",
    "OptionLeg",
    "Position",
    "PositionSnapshot",
    "LotLedger",
    "OpenLot",
    "match_lots",
    "position_snapshot",
    "hypothetical_snapshot",
    "resolve_expiry_instant",
    "year_fraction",
    "PortfolioError",
    "IncompleteBasisError",
    "UnsupportedContractError",
    "IronCondorPrices",
    "IronButterflyPrices",
    "ButterflyPrices",
    "VerticalPrices",
    "StraddlePrices",
    "StranglePrices",
    "CalendarPrices",
    "DiagonalPrices",
    "iron_condor",
    "iron_butterfly",
    "butterfly",
    "vertical_spread",
    "straddle",
    "strangle",
    "calendar_spread",
    "diagonal_spread",
]
