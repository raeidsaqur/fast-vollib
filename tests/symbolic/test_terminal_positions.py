"""Independent symbolic piecewise oracle for a symmetric call butterfly."""

from decimal import Decimal as D
from fractions import Fraction

import numpy as np
import sympy as sp

from fast_vollib.instruments import InstrumentRef, OptionType, SettlementType
from fast_vollib.observations import ObservationStamp
from fast_vollib.pnl import expiry_profile
from fast_vollib.portfolio import ButterflyPrices, ListedContract, butterfly, hypothetical_snapshot


def test_symbolic_call_butterfly_piecewise_identity():
    from datetime import date, datetime, timezone

    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stamp = ObservationStamp(
        event_time=origin,
        available_time=origin,
        recorded_at=origin,
        source="synthetic",
        world_id="hypothetical:symbolic",
    )

    def contract(k):
        return ListedContract(
            key=f"call:{k}",
            underlier=InstrumentRef(identifier="ACME", currency="USD"),
            option_type=OptionType.CALL,
            strike=D(k),
            expiry_date=date(2026, 6, 19),
            exercise="european",
            settlement=SettlementType.CASH,
            multiplier=D(100),
            deliverable="standard_cash_equivalent:100",
            currency="USD",
            stamp=stamp,
        )

    position = butterfly(
        lower=contract(90),
        middle=contract(100),
        upper=contract(110),
        quantity=1,
        prices=ButterflyPrices(lower=D(12), middle=D(6), upper=D(2)),
        fees=D(4),
    )
    snapshot = hypothetical_snapshot(
        position, valuation_time=origin, knowledge_cutoff=origin, world_id=stamp.world_id
    )
    s = sp.Symbol("s", real=True, nonnegative=True)
    oracle = 100 * (sp.Max(s - 90, 0) - 2 * sp.Max(s - 100, 0) + sp.Max(s - 110, 0)) - 204
    profile = expiry_profile(
        snapshot,
        np.array([1.0, 92.04, 100.0, 107.96, 300.0]),
        terminal_assumption="standard_intrinsic_equivalent",
    )
    for x, y in zip(profile.spot, profile.total[0]):
        assert abs(float(oracle.subs(s, sp.Rational(str(x)))) - y) < 1e-10
    left = sp.solve(100 * (s - 90) - 204, s)[0]
    right = sp.solve(100 * (110 - s) - 204, s)[0]
    assert profile.breakevens == (
        Fraction(int(sp.numer(left)), int(sp.denom(left))),
        Fraction(int(sp.numer(right)), int(sp.denom(right))),
    )
    tail_distance = sp.Symbol("tail_distance", positive=True)
    tail_expression = sp.simplify(oracle.subs(s, 110 + tail_distance))
    assert sp.limit(tail_expression, tail_distance, sp.oo) == profile.minimum_pnl == -204
    assert oracle.subs(s, 100) == profile.maximum_pnl == 796
