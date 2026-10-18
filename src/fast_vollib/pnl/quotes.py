"""Observed option marks with explicit freshness, delay, size and side policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from decimal import Decimal, localcontext
from enum import IntFlag
import json
from types import MappingProxyType
from typing import Any

import numpy as np

from ..observations import ObservationRecord, ObservationSnapshot, ObservationStamp
from ..observations.records import digest, text, utc
from ..portfolio import PositionSnapshot
from ..portfolio._validate import LEDGER_CONTEXT, money
from ..portfolio.contracts import require_standard
from .errors import PnLError, PnLHorizonError


class QuoteFlags(IntFlag):
    OK = 0
    MISSING = 1
    ONE_SIDED = 2
    CROSSED = 4
    STALE = 8
    HALTED = 16
    ZERO_SIZE = 32
    DELAYED = 64


def _delay(value: int | None) -> int | None:
    if value is not None and (type(value) is not int or value < 0):
        raise PnLError("delay must be nonnegative integer seconds or explicitly unknown")
    return value


def _size(value: int | None) -> int | None:
    if value is not None and (type(value) is not int or value < 0):
        raise PnLError("quote size must be a nonnegative integer or missing")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class OptionQuote:
    contract_key: str
    bid: Decimal | None
    ask: Decimal | None
    bid_size: int | None
    ask_size: int | None
    last: Decimal | None
    last_trade_time: datetime | None
    flags: QuoteFlags
    stamp: ObservationStamp
    quote_time_known: bool = True
    delay_s: int | None = None
    reported_estimates: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        from ..observations.records import freeze

        text(self.contract_key, "contract_key")
        if not isinstance(self.stamp, ObservationStamp) or not isinstance(
            self.stamp.event_time, datetime
        ):
            raise PnLError("option quote needs an instant ObservationStamp")
        if not isinstance(self.flags, QuoteFlags) or int(self.flags) & ~127:
            raise PnLError("unknown quote flags")
        if type(self.quote_time_known) is not bool:
            raise PnLError("quote_time_known must be boolean")
        if not self.quote_time_known and not self.stamp.timing_note:
            raise PnLError("unknown source quote time requires an explicit timing note")
        for name in ("bid", "ask", "last"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, money(value, nonnegative=True))
        for name in ("bid_size", "ask_size"):
            _size(getattr(self, name))
        _delay(self.delay_s)
        if self.last_trade_time is not None:
            object.__setattr__(self, "last_trade_time", utc(self.last_trade_time))
        object.__setattr__(self, "reported_estimates", freeze(dict(self.reported_estimates or {})))


@dataclass(frozen=True, slots=True, kw_only=True)
class QuotePolicy:
    name: str
    mark_rule: str
    max_quote_age_s: float
    max_underlier_age_s: float
    permitted_delay_s: int
    allow_zero_size: bool
    allow_unknown_size: bool = False
    allow_unknown_quote_time: bool = False
    allow_unknown_delay: bool = False

    def __post_init__(self) -> None:
        text(self.name, "quote policy name")
        if self.mark_rule not in ("midpoint", "close_estimate"):
            raise PnLError("mark_rule must be midpoint or close_estimate")
        for name in ("max_quote_age_s", "max_underlier_age_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise PnLError("maximum ages must be finite nonnegative seconds")
        if _delay(self.permitted_delay_s) is None:
            raise PnLError("permitted delay must be explicit")
        for name in (
            "allow_zero_size",
            "allow_unknown_size",
            "allow_unknown_quote_time",
            "allow_unknown_delay",
        ):
            if type(getattr(self, name)) is not bool:
                raise PnLError("quote permissions must be explicit booleans")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "pnl.quote_policy/1", **asdict(self)}


@dataclass(frozen=True, slots=True, kw_only=True)
class QuoteSnapshot:
    quotes: Mapping[str, OptionQuote]
    underlying: Decimal | None
    stamp: ObservationStamp
    delay_s: int | None
    valuation_time: datetime
    knowledge_cutoff: datetime
    underlying_time_known: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.stamp, ObservationStamp) or not isinstance(
            self.stamp.event_time, datetime
        ):
            raise PnLError("snapshot stamp describes the underlying quote instant")
        quotes = dict(self.quotes)
        if any(
            not isinstance(q, OptionQuote) or key != q.contract_key for key, q in quotes.items()
        ):
            raise PnLError("quote mapping keys must identify typed option quotes")
        if any(q.stamp.world_id != self.stamp.world_id for q in quotes.values()):
            raise PnLError("quote worlds cannot be mixed")
        object.__setattr__(self, "quotes", MappingProxyType(quotes))
        if self.underlying is not None:
            object.__setattr__(self, "underlying", money(self.underlying, nonnegative=True))
        for name in ("valuation_time", "knowledge_cutoff"):
            object.__setattr__(self, name, utc(getattr(self, name)))
        _delay(self.delay_s)
        if type(self.underlying_time_known) is not bool:
            raise PnLError("underlying_time_known must be boolean")
        if not self.underlying_time_known and not self.stamp.timing_note:
            raise PnLError("unknown underlying quote time requires a timing note")

    @property
    def digest(self) -> str:
        from ..observations.records import unplain
        from ..observations.serialization import stamp_to_dict

        return digest(
            {
                "quotes": {key: unplain(quote_to_dict(q)) for key, q in self.quotes.items()},
                "underlying": self.underlying,
                "stamp": unplain(stamp_to_dict(self.stamp)),
                "delay_s": self.delay_s,
                "valuation_time": self.valuation_time,
                "knowledge_cutoff": self.knowledge_cutoff,
                "underlying_time_known": self.underlying_time_known,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LegQuoteStatus:
    leg_id: str
    contract_key: str
    status: str
    flags: QuoteFlags
    reasons: tuple[str, ...]
    age_s: float | None
    delay_s: int | None
    bid: Decimal | None
    ask: Decimal | None
    bid_size: int | None
    ask_size: int | None
    mark: Decimal | None
    signed_value: Decimal | None
    quote_time: datetime | None
    available_time: datetime | None
    source: str | None

    def __post_init__(self) -> None:
        text(self.leg_id, "leg_id")
        text(self.contract_key, "contract_key")
        if self.source is not None:
            text(self.source, "quote source")
        if self.status not in ("available", "incomplete") or not isinstance(self.flags, QuoteFlags):
            raise PnLError("invalid per-leg quote status")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if any(not isinstance(x, str) or not x for x in self.reasons):
            raise PnLError("quote reasons must be immutable text")
        if self.age_s is not None and (not np.isfinite(self.age_s) or self.age_s < 0):
            raise PnLError("quote age must be nonnegative or unknown")
        for name in ("mark", "signed_value", "bid", "ask"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, money(value, nonnegative=name != "signed_value"))
        for name in ("quote_time", "available_time"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, utc(getattr(self, name)))
        for name in ("bid_size", "ask_size"):
            _size(getattr(self, name))
        _delay(self.delay_s)
        if (self.mark is None) != (self.signed_value is None) or (self.status == "available") != (
            self.mark is not None
        ):
            raise PnLError("quote availability must agree with mark/value")


def quote_to_dict(quote: OptionQuote) -> dict[str, Any]:
    from ..observations.records import plain
    from ..observations.serialization import stamp_to_dict

    return {
        "schema": "pnl.option_quote/1",
        **{
            f.name: stamp_to_dict(quote.stamp)
            if f.name == "stamp"
            else int(quote.flags)
            if f.name == "flags"
            else plain(getattr(quote, f.name))
            for f in fields(quote)
        },
    }


def quote_from_dict(value: Any) -> OptionQuote:
    from ..observations.records import unplain
    from ..observations.serialization import stamp_from_dict, strict_fields

    strict_fields(value, ["schema", *(f.name for f in fields(OptionQuote))])
    if value["schema"] != "pnl.option_quote/1" or type(value["flags"]) is not int:
        raise PnLError("unknown option quote schema/flags")
    data = {k: unplain(v) for k, v in value.items() if k != "schema"}
    data["stamp"] = stamp_from_dict(value["stamp"])
    data["flags"] = QuoteFlags(data["flags"])
    return OptionQuote(**data)


def quote_record(quote: OptionQuote, *, record_id: str) -> ObservationRecord:
    from ..observations.records import unplain

    return ObservationRecord(
        record_id=record_id,
        kind="option_quote",
        entities=(quote.contract_key,),
        payload=unplain(quote_to_dict(quote)),
        payload_schema="pnl.option_quote/1",
        stamp=quote.stamp,
    )


def quotes_as_of(
    observations: ObservationSnapshot,
    *,
    valuation_time: datetime,
    underlying: Decimal | None,
    stamp: ObservationStamp,
    delay_s: int | None,
    underlying_time_known: bool = True,
) -> QuoteSnapshot:
    """Select the newest admissible quote event per contract after revision resolution."""
    from ..observations.records import plain

    valuation_time = utc(valuation_time)
    quotes: dict[str, OptionQuote] = {}
    for record in observations.selected_records:
        if record.kind != "option_quote":
            continue
        if record.payload_schema != "pnl.option_quote/1":
            raise PnLError("unknown option quote observation schema")
        quote = quote_from_dict(plain(record.payload))
        if record.stamp != quote.stamp or record.entities != (quote.contract_key,):
            raise PnLError("quote evidence and typed payload disagree")
        if _event(quote.stamp) > valuation_time:
            continue
        old = quotes.get(quote.contract_key)
        key = (_event(quote.stamp), quote.stamp.available_time)
        if old is None or key > (_event(old.stamp), old.stamp.available_time):
            quotes[quote.contract_key] = quote
        elif key == (_event(old.stamp), old.stamp.available_time) and quote != old:
            raise PnLError("same-time quotes conflict without a revision/tie policy")
    if stamp.world_id != observations.world_id:
        raise PnLError("underlying quote world disagrees with selected observations")
    return QuoteSnapshot(
        quotes=quotes,
        underlying=underlying,
        stamp=stamp,
        delay_s=delay_s,
        valuation_time=valuation_time,
        knowledge_cutoff=observations.knowledge_cutoff,
        underlying_time_known=underlying_time_known,
    )


def _event(stamp: ObservationStamp) -> datetime:
    if not isinstance(stamp.event_time, datetime):
        raise PnLError("quote event time must be an instant")
    return stamp.event_time


def _timing(
    stamp: ObservationStamp,
    known: bool,
    *,
    valuation: datetime,
    cutoff: datetime,
    maximum: float,
    allow_unknown: bool,
) -> tuple[float | None, list[str]]:
    reasons = []
    if stamp.available_time > cutoff:
        reasons.append("not_available_at_cutoff")
    if not known:
        if not allow_unknown:
            reasons.append("unknown_source_time")
        return None, reasons
    age = (valuation - _event(stamp)).total_seconds()
    if age < 0:
        reasons.append("future_quote")
        return None, reasons
    if age > maximum:
        reasons.append("stale")
    return age, reasons


def _delay_reasons(delay: int | None, flags: QuoteFlags, policy: QuotePolicy) -> list[str]:
    if delay is None:
        return [] if policy.allow_unknown_delay else ["unknown_delay"]
    if flags & QuoteFlags.DELAYED and delay == 0:
        return ["unquantified_delay"]
    if delay > policy.permitted_delay_s or (
        flags & QuoteFlags.DELAYED and policy.permitted_delay_s == 0
    ):
        return ["delay_not_permitted"]
    return []


def mark_position(
    snapshot: PositionSnapshot,
    quotes: QuoteSnapshot,
    *,
    policy: QuotePolicy,
    diagnostic: bool = False,
    future_costs: Decimal = Decimal("0"),
) -> Any:
    """Signed inventory quote marks; side estimates are not executable spread fills."""
    from ..portfolio.serialization import snapshot_to_dict
    from .results import PnLProfile, ResultMeta

    if (
        not isinstance(snapshot, PositionSnapshot)
        or not isinstance(quotes, QuoteSnapshot)
        or not isinstance(policy, QuotePolicy)
    ):
        raise PnLError("marking requires typed position, quote snapshot and explicit policy")
    if type(diagnostic) is not bool:
        raise PnLError("diagnostic must be boolean")
    if (snapshot.valuation_time, snapshot.knowledge_cutoff, snapshot.world_id) != (
        quotes.valuation_time,
        quotes.knowledge_cutoff,
        quotes.stamp.world_id,
    ):
        raise PnLError("quote snapshot valuation time, knowledge cutoff and world must agree")
    costs = money(future_costs, nonnegative=True)
    _, under_reasons = _timing(
        quotes.stamp,
        quotes.underlying_time_known,
        valuation=snapshot.valuation_time,
        cutoff=snapshot.knowledge_cutoff,
        maximum=policy.max_underlier_age_s,
        allow_unknown=policy.allow_unknown_quote_time,
    )
    if quotes.underlying is None:
        under_reasons.append("missing_underlying")
    under_reasons += _delay_reasons(quotes.delay_s, QuoteFlags.OK, policy)
    if not snapshot.position.legs:
        under_reasons = []
    details = []
    with localcontext(LEDGER_CONTEXT):
        for leg in snapshot.position.legs:
            require_standard(leg.contract)
            c = leg.contract
            if (c.expiry_instant is not None and snapshot.valuation_time > c.expiry_instant) or (
                c.expiry_instant is None and snapshot.valuation_time.date() >= c.expiry_date
            ):
                raise PnLHorizonError(
                    "quote marking cannot reconstruct an expired inventory lifecycle"
                )
            quote = quotes.quotes.get(leg.contract.key)
            flags = QuoteFlags.MISSING if quote is None else quote.flags
            reasons = []
            age = None
            delay = quotes.delay_s if quote is None or quote.delay_s is None else quote.delay_s
            value: Decimal | None = None
            if quote is None:
                reasons.append("missing_quote")
            else:
                age, reasons = _timing(
                    quote.stamp,
                    quote.quote_time_known,
                    valuation=snapshot.valuation_time,
                    cutoff=snapshot.knowledge_cutoff,
                    maximum=policy.max_quote_age_s,
                    allow_unknown=policy.allow_unknown_quote_time,
                )
                if quote.bid is None and quote.ask is None:
                    flags |= QuoteFlags.MISSING
                if (quote.bid is None) != (quote.ask is None):
                    flags |= QuoteFlags.ONE_SIDED
                if quote.bid is not None and quote.ask is not None and quote.bid > quote.ask:
                    flags |= QuoteFlags.CROSSED
                if "stale" in reasons:
                    flags |= QuoteFlags.STALE
                if delay is not None and delay > 0:
                    flags |= QuoteFlags.DELAYED
                for flag, name in (
                    (QuoteFlags.MISSING, "missing_quote"),
                    (QuoteFlags.CROSSED, "crossed"),
                    (QuoteFlags.STALE, "stale"),
                    (QuoteFlags.HALTED, "halted"),
                ):
                    if flags & flag:
                        reasons.append(name)
                if flags & QuoteFlags.ONE_SIDED and policy.mark_rule == "midpoint":
                    reasons.append("one_sided_midpoint")
                sides = (
                    ("bid", "ask")
                    if policy.mark_rule == "midpoint"
                    else (("bid",) if leg.quantity > 0 else ("ask",))
                )
                for side in sides:
                    if getattr(quote, side) is None:
                        reasons.append("missing_" + side)
                    size = getattr(quote, side + "_size")
                    if size == 0:
                        flags |= QuoteFlags.ZERO_SIZE
                    if size is None and not policy.allow_unknown_size:
                        reasons.append("unknown_" + side + "_size")
                if flags & QuoteFlags.ZERO_SIZE and not policy.allow_zero_size:
                    reasons.append("zero_size")
                reasons += _delay_reasons(delay, flags, policy)
                if not reasons:
                    if policy.mark_rule == "midpoint":
                        assert quote.bid is not None and quote.ask is not None
                        value = (quote.bid + quote.ask) / 2
                    else:
                        value = getattr(quote, sides[0])
            details.append(
                LegQuoteStatus(
                    leg_id=leg.leg_id,
                    contract_key=leg.contract.key,
                    status="available" if value is not None else "incomplete",
                    flags=flags,
                    reasons=tuple(dict.fromkeys(reasons)),
                    age_s=age,
                    delay_s=delay,
                    bid=None if quote is None else quote.bid,
                    ask=None if quote is None else quote.ask,
                    bid_size=None if quote is None else quote.bid_size,
                    ask_size=None if quote is None else quote.ask_size,
                    mark=value,
                    signed_value=None if value is None else leg.signed_size * value,
                    quote_time=None
                    if quote is None or not quote.quote_time_known
                    else _event(quote.stamp),
                    available_time=None if quote is None else quote.stamp.available_time,
                    source=None if quote is None else quote.stamp.source,
                )
            )
        inventory_complete = all(d.signed_value is not None for d in details)
        inventory = (
            sum((d.signed_value for d in details if d.signed_value is not None), Decimal(0))
            if inventory_complete
            else None
        )
        aggregate_available = (
            not under_reasons and inventory is not None and snapshot.cumulative_cash is not None
        )
        total = None
        if aggregate_available:
            assert snapshot.cumulative_cash is not None and inventory is not None
            total = snapshot.cumulative_cash + inventory - costs
        unrealized = (
            inventory - (snapshot.realized - snapshot.cumulative_cash)
            if inventory is not None
            and snapshot.realized is not None
            and snapshot.cumulative_cash is not None
            else None
        )
    if not diagnostic and (under_reasons or not inventory_complete):
        raise PnLError(
            "incomplete quote mark: "
            + ", ".join(under_reasons + [f"{d.leg_id}:{r}" for d in details for r in d.reasons])
        )
    caveat = (
        "Midpoint indication; not an executable fill."
        if policy.mark_rule == "midpoint"
        else "Sum of per-leg close estimates (sell longs at bid, buy shorts at ask); not an executable multi-leg fill."
    )
    if any(
        d.flags & (QuoteFlags.DELAYED | QuoteFlags.ZERO_SIZE)
        or d.age_s is None
        or d.bid_size is None
        or d.ask_size is None
        for d in details
    ):
        caveat += (
            " Indicative quote permissions and unknown fields remain labelled in per-leg status."
        )
    if under_reasons:
        caveat += " Underlying unavailable: " + ", ".join(under_reasons) + "."
    meta = ResultMeta(
        origin=snapshot.origin,
        basis="quote_mark",
        valuation_time=snapshot.valuation_time,
        knowledge_cutoff=snapshot.knowledge_cutoff,
        knowledge_label=snapshot.knowledge_label,
        world_id=snapshot.world_id,
        observations_digest=snapshot.observations_digest,
        currency=snapshot.currency,
        caveat=caveat,
        terminal_assumption="not_applicable:quote_mark",
        accounting_policy=snapshot.accounting_policy,
        input_digest=digest(
            {
                "snapshot": snapshot_to_dict(snapshot),
                "quotes": quotes.digest,
                "policy": policy.to_dict(),
                "future_costs": costs,
            }
        ),
        quote_policy=json.dumps(policy.to_dict(), sort_keys=True),
        validation_status="observed_quote_policy",
        schema_version="pnl.result/2",
    )
    values = np.array(
        [float(d.signed_value) if d.signed_value is not None else np.nan for d in details]
    ).reshape(1, 1, -1)
    valid = np.array([d.signed_value is not None for d in details], dtype=bool).reshape(1, 1, -1)
    return PnLProfile(
        horizons=(0.0,),
        target_date=None,
        spot=[float(quotes.underlying) if quotes.underlying is not None else np.nan],
        leg_values=values,
        total=[[float(total) if total is not None else np.nan]],
        valid=np.array([[aggregate_available]], dtype=bool),
        leg_valid=valid,
        cumulative_cash=snapshot.cumulative_cash,
        paid_fees=snapshot.paid_fees,
        future_costs=costs,
        leg_ids=tuple(x.leg_id for x in snapshot.position.legs),
        meta=meta,
        metrics=None,
        strikes=tuple(x.contract.strike for x in snapshot.position.legs),
        contracts=tuple(x.contract for x in snapshot.position.legs),
        signed_sizes=tuple(x.signed_size for x in snapshot.position.legs),
        conversion_errors=(),
        conversion_bound=[np.nan],
        quote_details=tuple(details),
        realized=snapshot.realized,
        unrealized=unrealized,
    )
