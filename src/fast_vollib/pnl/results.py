"""Owned immutable host reports; native evaluation lives in terminal_pnl."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np

from ..observations.records import utc
from .errors import PnLError
from .metrics import PayoffMetrics


def _owned(value: Any, dtype: Any) -> Any:
    array = np.array(value, dtype=dtype, copy=True)
    # Immutable bytes prevent callers from re-enabling NumPy's WRITEABLE flag.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResultMeta:
    origin: str
    basis: str
    valuation_time: datetime
    knowledge_cutoff: datetime
    knowledge_label: str
    world_id: str
    observations_digest: str
    currency: str
    caveat: str | None
    terminal_assumption: str
    accounting_policy: str
    input_digest: str
    time_policy: str | None = None
    expiry_policies: tuple[str, ...] = ()
    measure: str | None = None
    model: str | None = None
    vol_source: str | None = None
    spot_movement: str | None = None
    time_movement: str | None = None
    validation_status: str = "analytic_terminal"
    schema_version: str = "pnl.result/1"
    quote_policy: str | None = None

    def __post_init__(self) -> None:
        utc(self.valuation_time)
        utc(self.knowledge_cutoff)
        object.__setattr__(self, "expiry_policies", tuple(self.expiry_policies))
        for name in (
            "origin",
            "basis",
            "knowledge_label",
            "world_id",
            "observations_digest",
            "currency",
            "terminal_assumption",
            "accounting_policy",
            "input_digest",
            "validation_status",
            "schema_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise PnLError("metadata identifiers must be nonempty strings")
        for name in (
            "caveat",
            "time_policy",
            "measure",
            "model",
            "vol_source",
            "spot_movement",
            "time_movement",
            "quote_policy",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise PnLError("optional metadata must be text or missing")
        if any(not isinstance(value, str) for value in self.expiry_policies):
            raise PnLError("expiry policies must be immutable text identifiers")
        if (self.schema_version, self.basis) not in (
            ("pnl.result/1", "expiry_intrinsic"),
            ("pnl.result/2", "quote_mark"),
            ("pnl.result/2", "model"),
            ("pnl.result/2", "european_proxy"),
        ):
            raise PnLError("unsupported result schema/basis")
        if self.basis == "quote_mark" and (
            not self.quote_policy
            or self.terminal_assumption != "not_applicable:quote_mark"
            or not self.caveat
        ):
            raise PnLError("quote marks require their explicit policy and indication caveat")
        if self.origin not in ("hypothetical", "executed", "positions_only"):
            raise PnLError("unknown result origin")
        if (self.origin == "hypothetical") == (self.world_id == "actual"):
            raise PnLError("result origin/world disagree")
        if self.basis in ("model", "european_proxy") and (
            self.terminal_assumption != "not_applicable:" + self.basis
            or self.model not in ("black", "black_scholes", "black_scholes_merton")
            or self.time_policy not in ("act_365", "act_365_25", "act_360")
        ):
            raise PnLError("model results require their explicit model, day count and basis")
        if self.basis == "expiry_intrinsic" and self.terminal_assumption not in (
            "standard_intrinsic_equivalent",
            "hold_to_expiry_no_early_exercise",
        ):
            raise PnLError("unknown terminal assumption")
        if self.terminal_assumption == "hold_to_expiry_no_early_exercise" and not self.caveat:
            raise PnLError("hold-to-expiry results must retain their caveat")
        label = "as_known" if self.knowledge_cutoff <= self.valuation_time else "latest_corrected"
        if self.knowledge_label != label:
            raise PnLError("knowledge label disagrees with result cutoff")


@dataclass(frozen=True, slots=True, kw_only=True)
class PnLProfile:
    horizons: tuple[float | None, ...]
    target_date: date | None
    spot: Any
    leg_values: Any
    total: Any
    valid: Any
    leg_valid: Any
    cumulative_cash: Decimal | None
    paid_fees: Decimal
    future_costs: Decimal
    leg_ids: tuple[str, ...]
    meta: ResultMeta
    metrics: PayoffMetrics | None
    strikes: tuple[Decimal, ...]
    contracts: tuple
    signed_sizes: tuple[Decimal, ...]
    conversion_errors: tuple
    conversion_bound: Any
    quote_details: tuple = ()
    realized: Decimal | None = None
    unrealized: Decimal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.meta, ResultMeta):
            raise PnLError("report requires typed metadata")
        terminal = self.meta.basis == "expiry_intrinsic"
        for field in (
            "horizons",
            "leg_ids",
            "strikes",
            "contracts",
            "signed_sizes",
            "conversion_errors",
            "quote_details",
        ):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        for field in ("spot", "leg_values", "total", "conversion_bound"):
            object.__setattr__(self, field, _owned(getattr(self, field), np.float64))
        for field in ("valid", "leg_valid"):
            if np.asarray(getattr(self, field)).dtype != np.dtype(bool):
                raise PnLError("validity masks must be boolean")
            object.__setattr__(self, field, _owned(getattr(self, field), bool))
        n, h, n_legs = self.spot.size, len(self.horizons), len(self.leg_ids)
        if (
            self.spot.shape != (n,)
            or n == 0
            or h < 1
            or (self.meta.basis in ("expiry_intrinsic", "quote_mark") and h != 1)
        ):
            raise PnLError("expiry profile requires one horizon and a nonempty spot vector")
        if (
            self.total.shape != (h, n)
            or self.valid.shape != (h, n)
            or self.leg_values.shape != (h, n, n_legs)
            or self.leg_valid.shape != (h, n, n_legs)
        ):
            raise PnLError("inconsistent report array shapes")
        if (
            len(set(self.leg_ids)) != n_legs
            or len(self.strikes) != n_legs
            or self.conversion_bound.shape != (n,)
        ):
            raise PnLError("inconsistent report leg identities/bounds")
        if (
            (self.spot < 0).any()
            or np.isinf(self.spot).any()
            or ((terminal or (self.valid.any() and n_legs)) and not np.isfinite(self.spot).all())
        ):
            raise PnLError("invalid report spots")
        if (
            not np.isfinite(self.total[self.valid]).all()
            or not np.isfinite(self.leg_values[self.leg_valid]).all()
        ):
            raise PnLError("valid report entries must be finite")
        if self.cumulative_cash is None and (terminal or self.valid.any()):
            raise PnLError("available total requires complete entry basis")
        if self.meta.basis in ("model", "european_proxy"):
            model_horizons = np.asarray(self.horizons, dtype=float)
            if (
                not np.isfinite(self.spot).all()
                or not np.isfinite(model_horizons).all()
                or (np.diff(model_horizons) <= 0).any()
            ):
                raise PnLError(
                    "model reports require finite spots and increasing resolved horizons"
                )
            aggregate = self.leg_valid.all(axis=-1) & (self.cumulative_cash is not None)
            if (
                not np.array_equal(self.valid, aggregate)
                or not np.isnan(self.total[~self.valid]).all()
                or not np.isnan(self.leg_values[~self.leg_valid]).all()
            ):
                raise PnLError("model report masks and unavailable cells disagree")
        from ..portfolio import ListedContract
        from ..portfolio._validate import money
        from ..portfolio.contracts import require_standard

        for field in ("cumulative_cash", "paid_fees", "future_costs", "realized", "unrealized"):
            if getattr(self, field) is None:
                continue
            object.__setattr__(
                self,
                field,
                money(getattr(self, field), nonnegative=field in ("paid_fees", "future_costs")),
            )
        if len(self.contracts) != n_legs or len(self.signed_sizes) != n_legs:
            raise PnLError("report contract terms disagree with legs")
        from .proxy import EuropeanProxyProfile

        if isinstance(self, EuropeanProxyProfile) != (self.meta.basis == "european_proxy"):
            raise PnLError("European proxy requires its separate result type")
        for contract, strike, size in zip(self.contracts, self.strikes, self.signed_sizes):
            if (
                not isinstance(contract, ListedContract)
                or contract.strike != strike
                or contract.stamp.world_id != self.meta.world_id
                or contract.currency != self.meta.currency
            ):
                raise PnLError("invalid report contract terms")
            money(size)
            if not size:
                raise PnLError("report signed size must be nonzero")
            require_standard(contract)
            if self.meta.basis == "model" and contract.exercise != "european":
                raise PnLError("standard model profiles refuse American source contracts")
            if terminal and (
                contract.exercise == "american" or contract.settlement.value == "physical"
            ):
                if (
                    self.meta.terminal_assumption != "hold_to_expiry_no_early_exercise"
                    or not self.meta.caveat
                ):
                    raise PnLError("report must retain American/physical terminal caveat")
        if any(
            horizon is not None and (not np.isfinite(horizon) or horizon < 0)
            for horizon in self.horizons
        ):
            raise PnLError("report horizons must be finite nonnegative elapsed years")
        if terminal and (
            not np.isfinite(self.conversion_bound).all() or (self.conversion_bound < 0).any()
        ):
            raise PnLError("invalid coefficient conversion bounds")
        if (terminal and not isinstance(self.metrics, PayoffMetrics)) or (
            not terminal and self.metrics is not None
        ):
            raise PnLError("exact payoff metrics apply only to terminal profiles")
        if self.meta.basis == "quote_mark":
            import json

            from .quotes import LegQuoteStatus, QuotePolicy

            assert self.meta.quote_policy is not None
            policy = json.loads(self.meta.quote_policy)
            if policy.pop("schema", None) != "pnl.quote_policy/1":
                raise PnLError("unknown quote policy schema")
            quote_policy = QuotePolicy(**policy)
            if n != 1 or len(self.quote_details) != n_legs:
                raise PnLError("quote marks require one point and per-leg status")
            for i, detail in enumerate(self.quote_details):
                if not isinstance(detail, LegQuoteStatus) or (
                    detail.leg_id,
                    detail.contract_key,
                ) != (self.leg_ids[i], self.contracts[i].key):
                    raise PnLError("per-leg quote identities disagree")
                if bool(self.leg_valid[0, 0, i]) != (detail.signed_value is not None):
                    raise PnLError("per-leg quote availability disagrees")
                if detail.signed_value is not None and self.leg_values[0, 0, i] != float(
                    detail.signed_value
                ):
                    raise PnLError("per-leg quote value disagrees")
                if detail.mark is not None:
                    from decimal import localcontext

                    from ..portfolio._validate import LEDGER_CONTEXT
                    from .quotes import QuoteFlags

                    with localcontext(LEDGER_CONTEXT):
                        if detail.signed_value != self.signed_sizes[i] * detail.mark:
                            raise PnLError("quote mark and signed inventory value disagree")
                        if quote_policy.mark_rule == "midpoint":
                            if (
                                detail.bid is None
                                or detail.ask is None
                                or detail.mark != (detail.bid + detail.ask) / 2
                            ):
                                raise PnLError("midpoint mark disagrees with both sides")
                        elif detail.mark != (
                            detail.bid if self.signed_sizes[i] > 0 else detail.ask
                        ):
                            raise PnLError("close mark disagrees with required side")
                    if (
                        detail.flags
                        & (
                            QuoteFlags.MISSING
                            | QuoteFlags.CROSSED
                            | QuoteFlags.STALE
                            | QuoteFlags.HALTED
                        )
                        or detail.reasons
                    ):
                        raise PnLError("unavailable quote cannot carry an available mark")
        from .evaluate import CoefficientConversion

        if any(not isinstance(value, CoefficientConversion) for value in self.conversion_errors):
            raise PnLError("report coefficient receipts must be immutable typed records")
        if self.cumulative_cash is None:
            return
        expected = (
            float(self.cumulative_cash) + self.leg_values.sum(axis=-1) - float(self.future_costs)
        )
        scale = (
            abs(float(self.cumulative_cash))
            + np.abs(self.leg_values).sum(axis=-1)
            + abs(float(self.future_costs))
            + 1
        )
        if np.any(
            np.abs(self.total - expected)[self.valid]
            > (64 * np.finfo(float).eps * scale)[self.valid]
        ):
            raise PnLError("report violates cash + signed inventory - future costs")

    @property
    def breakevens(self) -> Any:
        return self.metrics.breakevens if self.metrics is not None else None

    @property
    def minimum_pnl(self) -> Any:
        return self.metrics.minimum_pnl if self.metrics is not None else None

    @property
    def maximum_pnl(self) -> Any:
        return self.metrics.maximum_pnl if self.metrics is not None else None

    def to_frame(self) -> Any:
        """Fresh rows; paid fees explain cash and are not subtracted twice."""
        import pandas as pd

        h, n = len(self.horizons), self.spot.size
        rows = {
            "spot": np.tile(self.spot, h),
            "total": self.total.reshape(-1).copy(),
            "valid": self.valid.reshape(-1).copy(),
            "horizon": np.repeat(np.array(self.horizons, dtype=object), n),
            "basis": self.meta.basis,
            "model": self.meta.model,
            "vol_source": self.meta.vol_source,
            "spot_movement": self.meta.spot_movement,
            "time_movement": self.meta.time_movement,
            "time_policy": self.meta.time_policy,
            "currency": self.meta.currency,
            "caveat": self.meta.caveat,
            "quote_policy": self.meta.quote_policy,
            "valuation_time": self.meta.valuation_time,
            "knowledge_cutoff": self.meta.knowledge_cutoff,
            "knowledge_label": self.meta.knowledge_label,
            "origin": self.meta.origin,
            "realized": None if self.realized is None else str(self.realized),
            "unrealized": None if self.unrealized is None else str(self.unrealized),
        }
        for i, leg_id in enumerate(self.leg_ids):
            rows[f"leg:{leg_id}"] = self.leg_values[:, :, i].reshape(-1).copy()
            rows[f"leg_valid:{leg_id}"] = self.leg_valid[:, :, i].reshape(-1).copy()
            if self.quote_details:
                detail = self.quote_details[i]
                rows[f"status:{leg_id}"] = detail.status
                rows[f"reasons:{leg_id}"] = ",".join(detail.reasons)
                rows[f"flags:{leg_id}"] = str(detail.flags)
                rows[f"age_s:{leg_id}"] = detail.age_s
                rows[f"delay_s:{leg_id}"] = detail.delay_s
        return pd.DataFrame(rows)

    def to_dict(self) -> Any:
        from .serialization import profile_to_dict

        return profile_to_dict(self)

    def to_json(self) -> Any:
        from .serialization import dumps

        return dumps(self)
