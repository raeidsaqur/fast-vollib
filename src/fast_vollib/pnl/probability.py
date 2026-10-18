"""Analytic terminal profit probabilities under an explicitly declared lognormal law."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from decimal import Decimal, DecimalException, localcontext
from fractions import Fraction
import json
from typing import Any

import numpy as np

from .assumptions import finite
from .errors import PnLError, PnLHorizonError
from .results import PnLProfile, ResultMeta


@dataclass(frozen=True, slots=True, kw_only=True)
class LognormalTerminal:
    spot: float
    drift: float
    sigma: float
    T: float
    measure: str
    convention: str

    def __post_init__(self) -> None:
        for name in ("spot", "drift", "sigma", "T"):
            object.__setattr__(
                self,
                name,
                finite(
                    getattr(self, name),
                    name,
                    nonnegative=name in ("sigma", "T"),
                    positive=name == "spot",
                ),
            )
        if self.measure not in ("physical", "risk_neutral") or self.convention not in (
            "act_365",
            "act_365_25",
            "act_360",
        ):
            raise PnLError(
                "terminal law requires a financial measure and supported time convention"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class TerminalProfitProbability:
    probability: float
    law: LognormalTerminal
    profitable_intervals: tuple[tuple[Fraction, Fraction | None], ...]
    meta: ResultMeta
    title: str = "Under the specified terminal law"

    def __post_init__(self) -> None:
        object.__setattr__(self, "probability", finite(self.probability, "probability"))
        if (
            not np.isfinite(self.probability)
            or not 0 <= self.probability <= 1
            or not isinstance(self.law, LognormalTerminal)
        ):
            raise PnLError("invalid terminal-law probability")
        object.__setattr__(
            self, "profitable_intervals", tuple(tuple(x) for x in self.profitable_intervals)
        )
        previous: Fraction | None = Fraction(0)
        for interval in self.profitable_intervals:
            if len(interval) != 2:
                raise PnLError("profit intervals require exact rational endpoints")
            a, b = interval
            if (
                not isinstance(a, Fraction)
                or a < 0
                or previous is None
                or a < previous
                or (b is not None and (not isinstance(b, Fraction) or b <= a))
            ):
                raise PnLError("profit intervals must be ordered, disjoint and exact")
            previous = b
        if (
            self.meta.basis != "expiry_intrinsic"
            or self.meta.measure != self.law.measure
            or self.meta.time_policy != self.law.convention
            or self.title != "Under the specified terminal law"
        ):
            raise PnLError("terminal probability must retain its law, payoff basis and title")

    def to_json(self) -> str:
        from .serialization import _encode

        return json.dumps(
            {"schema": "pnl.terminal_probability/1", "result": _encode(self)},
            sort_keys=True,
            allow_nan=False,
        )


def probability_of_profit(profile: PnLProfile, law: LognormalTerminal) -> TerminalProfitProbability:
    from scipy.special import ndtr

    from ..observations.records import digest
    from ..simulation.scenario import horizon_tolerance
    from .metrics import _analyze_terms, _value_terms

    if (
        not isinstance(profile, PnLProfile)
        or profile.meta.basis != "expiry_intrinsic"
        or not isinstance(law, LognormalTerminal)
    ):
        raise PnLError(
            "profit probabilities require an expiry payoff and specified terminal law; proxies are refused"
        )
    if profile.cumulative_cash is None or len(profile.horizons) != 1 or profile.horizons[0] is None:
        raise PnLHorizonError("terminal law requires a resolved payoff horizon")
    if profile.meta.time_policy != law.convention or abs(
        profile.horizons[0] - law.T
    ) > horizon_tolerance(law.T):
        raise PnLHorizonError("terminal law and payoff horizon/day count disagree")
    cash = Fraction(profile.cumulative_cash) - Fraction(profile.future_costs)
    metrics = _analyze_terms(profile.contracts, profile.signed_sizes, cash)
    knots = {Fraction(0), *(Fraction(c.strike) for c in profile.contracts), *metrics.breakevens}
    for a, b in metrics.zero_intervals:
        knots.add(a)
        if b is not None:
            knots.add(b)
    ordered = sorted(knots)
    intervals = []
    for a, b in zip(ordered, [*ordered[1:], None]):
        at = a + 1 if b is None else (a + b) / 2
        if _value_terms(profile.contracts, profile.signed_sizes, cash, at) > 0:
            intervals.append((a, b))
    if law.T == 0 or law.sigma == 0:
        try:
            with localcontext() as ctx:
                ctx.prec = 100
                atom = (
                    Decimal(str(law.spot))
                    if law.T == 0
                    else Decimal(str(law.spot))
                    * (Decimal(str(law.drift)) * Decimal(str(law.T))).exp()
                )
                if not atom.is_finite() or atom <= 0:
                    raise PnLError("terminal atom is outside the supported numeric domain")
        except DecimalException as exc:
            raise PnLError("terminal atom is outside the supported numeric domain") from exc
        probability = float(
            _value_terms(profile.contracts, profile.signed_sizes, cash, Fraction(atom)) > 0
        )
    else:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            location = np.log(law.spot) + (law.drift - 0.5 * np.float64(law.sigma) ** 2) * law.T
            scale = law.sigma * np.sqrt(law.T)
        if not np.isfinite(location) or not np.isfinite(scale) or scale <= 0:
            raise PnLError("terminal law is outside the supported numeric domain")
        probability = 0.0
        for a, b in intervals:
            left = -np.inf if a == 0 else (np.log(float(a)) - location) / scale
            right = np.inf if b is None else (np.log(float(b)) - location) / scale
            probability += float(
                ndtr(-left) - ndtr(-right) if left > 0 else ndtr(right) - ndtr(left)
            )
        probability = min(1.0, max(0.0, probability))
    meta = replace(
        profile.meta,
        measure=law.measure,
        validation_status="specified_terminal_law",
        input_digest=digest({"payoff_input_digest": profile.meta.input_digest, "law": asdict(law)}),
    )
    return TerminalProfitProbability(
        probability=probability, law=law, profitable_intervals=tuple(intervals), meta=meta
    )


def probability_from_dict(value: Any) -> TerminalProfitProbability:
    from .serialization import _decode, _fields

    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "result"}
        or value["schema"] != "pnl.terminal_probability/1"
    ):
        raise PnLError("unknown terminal probability schema")
    data = _decode(value["result"])
    _fields(data, TerminalProfitProbability)
    _fields(data["law"], LognormalTerminal)
    _fields(data["meta"], ResultMeta)
    data["meta"] = ResultMeta(**data["meta"])
    data["law"] = LognormalTerminal(**data["law"])
    return TerminalProfitProbability(**data)
