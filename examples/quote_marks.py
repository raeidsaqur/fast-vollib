"""Synthetic quote indications alongside an intrinsic-equivalent expiry curve."""

from decimal import Decimal as D
from pathlib import Path
import runpy

from fast_vollib.pnl import (
    OptionQuote,
    QuoteFlags,
    QuotePolicy,
    QuoteSnapshot,
    mark_position,
    plot_pnl,
)

setup = runpy.run_path(str(Path(__file__).with_name("terminal_pnl.py")))
snapshot = setup["snap"]
# Explicit synthetic bid/ask inputs; these are not observed broker quotes.
quotes = {
    leg.contract.key: OptionQuote(
        contract_key=leg.contract.key,
        bid=D("1.00"),
        ask=D("1.10"),
        bid_size=10,
        ask_size=10,
        last=None,
        last_trade_time=None,
        flags=QuoteFlags.OK,
        stamp=leg.contract.stamp,
    )
    for leg in snapshot.position.legs
}
book = QuoteSnapshot(
    quotes=quotes,
    underlying=D("600"),
    stamp=snapshot.position.legs[0].contract.stamp,
    delay_s=0,
    valuation_time=snapshot.valuation_time,
    knowledge_cutoff=snapshot.knowledge_cutoff,
)
policy = QuotePolicy(
    name="synthetic_current_midpoints",
    mark_rule="midpoint",
    max_quote_age_s=60.0,
    max_underlier_age_s=60.0,
    permitted_delay_s=0,
    allow_zero_size=False,
)
mark = mark_position(snapshot, book, policy=policy, future_costs=D("4"))
figure = plot_pnl([setup["profile"]], mark=mark)
assert mark.meta.basis == "quote_mark"
assert mark.meta.caveat and mark.quote_details
