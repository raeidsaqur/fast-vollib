"""Named saved-payload variants; units and fee semantics are never guessed."""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class EndpointMapping:
    version: str
    quote_envelope: str
    closed_pnl_key: str
    open_pnl_key: str
    iv_unit: str
    execution_cost_basis: str = "positive_premium_times_quantity"
    activity_quantity_basis: str = "signed_contract_quantity"
    quote_source_time: str = "unknown"


MAPPINGS = MappingProxyType(
    {
        f"questrade.saved/1:{shape}-{units}": EndpointMapping(
            version=f"questrade.saved/1:{shape}-{units}",
            quote_envelope="quotes" if shape == "properties" else "optionQuotes",
            closed_pnl_key="closedPnL" if shape == "properties" else "closedPnl",
            open_pnl_key="openPnL" if shape == "properties" else "openPnl",
            iv_unit=units,
        )
        for shape in ("properties", "examples")
        for units in ("percent", "decimal")
    }
)
SIDE_SIGN = MappingProxyType(
    {"Buy": 1, "Sell": -1, "Short": -1, "Cov": 1, "BTO": 1, "STC": -1, "STO": -1, "BTC": 1}
)
FEE_FIELDS = (
    "commission",
    "orderPlacementCommission",
    "executionFee",
    "secFee",
    "canadianExecutionFee",
)


def get_mapping(version: str) -> EndpointMapping:
    try:
        return MAPPINGS[version]
    except (KeyError, TypeError) as exc:
        raise ValueError("unsupported saved Questrade mapping variant") from exc


ROW_FIELDS = MappingProxyType(
    {
        "symbols": frozenset(
            "symbol symbolId prevDayClosePrice highPrice52 lowPrice52 averageVol3Months averageVol20Days outstandingShares eps pe dividend yield exDate marketCap optionType optionDurationType optionRoot optionContractDeliverables optionExerciseType listingExchange description securityType optionExpiryDate dividendDate optionStrikePrice isQuotable hasOptions currency minTicks industrySector industryGroup industrySubGroup tradeUnit isTradable".split()
        ),
        "executions": frozenset(
            "symbol symbolId quantity side price id orderId orderChainId exchangeExecId timestamp notes venue totalCost orderPlacementCommission commission executionFee secFee canadianExecutionFee parentId".split()
        ),
        "quotes": frozenset(
            "underlying underlyingId symbol symbolId bidPrice bidSize askPrice askSize lastTradePriceTrHrs lastTradePrice lastTradeSize lastTradeTick lastTradeTime volume openPrice highPrice highPricehighPrice lowPrice volatility delta gamma theta vega rho openInterest delay isHalted VWAP".split()
        ),
        "positions": frozenset(
            "symbol symbolId openQuantity closedQuantity currentMarketValue currentPrice averageEntryPrice closedPnL openPnL closedPnl openPnl totalCost isRealTime isUnderReorg".split()
        ),
        "activities": frozenset(
            "tradeDate transactionDate settlementDate action symbol symbolId description currency quantity price grossAmount commission netAmount type".split()
        ),
    }
)
