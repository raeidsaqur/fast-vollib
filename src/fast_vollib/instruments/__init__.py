"""Nominal layer for fast-vollib: assets, contracts, and how they are priced.

fast-vollib's core is functional -- ``fast_black_scholes(flag, S, K, t, r,
sigma)`` and friends -- and that stays the canonical, fastest API.  What it has
no vocabulary for is *what* is being priced: nothing in it **is** an option, an
underlier, or a forward.  This package adds that layer without moving any math
into it.

The package is organized into five layers:

1. **Contracts** -- :class:`Asset`, :class:`Forward`, :class:`Future`,
   :class:`EuropeanOption`, and the digital and path-dependent contracts:
   frozen, slotted, keyword-only dataclasses holding contract terms and
   :class:`InstrumentRef` underliers.  Backend-free and JSON-serializable.
2. **Batches** -- :class:`EuropeanOptionBatch`: a homogeneous columnar
   container that is the unit of execution.  One batch is one vectorized kernel
   call, never a Python loop over contracts.
3. **Market inputs** -- :class:`VanillaMarketInputs`, passed alongside a
   contract and never stored on it.
4. **Payoffs and adapters** -- :func:`payoff` evaluates in the caller's own
   array namespace and preserves the autograd tape;
   :func:`price_instrument`, :func:`greeks_instrument`, and
   :func:`implied_volatility_instrument` are thin wrappers over the existing
   kernels, guarded by :func:`capabilities` and typed errors.
5. **Discovery** -- :func:`instrument_types` is a read-only registry of what
   exists and what can be done with it.

Two rules govern the whole package.  *Nothing is inferred*: the pricing model
and the IV solver are always explicit arguments, and an asset class may
validate a choice but never make one.  *Nothing falls back*: a request that
cannot be served exactly -- unknown model, unavailable solver, missing market
input, a differentiable route that is not installed -- raises one of the errors
in :mod:`fast_vollib.instruments.errors` instead of quietly answering a
different question.

Importing this package pulls in neither torch, jax, numba, nor triton.

Examples
--------
>>> import numpy as np
>>> from fast_vollib.instruments import (
...     Asset, AssetClass, EuropeanOption, VanillaMarketInputs, price_instrument,
... )
>>> acme = Asset(identifier="ACME", asset_class=AssetClass.EQUITY, currency="USD")
>>> option = EuropeanOption(
...     underlier=acme.ref(), option_type="call", strike=100.0, maturity=1.0,
... )
>>> market = VanillaMarketInputs(underlying=100.0, rate=0.02, volatility=0.2)
>>> price = price_instrument(option, market, model="black_scholes")
>>> float(np.round(price[0], 6))
8.916037
"""

from __future__ import annotations

from .base import Asset, Derivative, Instrument, InstrumentRef
from .batch import (
    EuropeanOptionBatch,
    forward_price,
    log_moneyness,
    moneyness,
    time_to_maturity,
)
from .capabilities import CapabilitySet, capabilities
from .enums import (
    AssetClass,
    AveragingMethod,
    BarrierType,
    ExerciseStyle,
    InstrumentKind,
    IVSolver,
    OptionType,
    PayoffRequirement,
    PricingModel,
    SettlementType,
    StrikeConvention,
)
from .errors import (
    InstrumentError,
    InstrumentValidationError,
    MissingMarketInputError,
    SerializationError,
    UnsupportedInstrumentError,
    UnsupportedModelError,
    UnsupportedSolverError,
)
from .exotics import (
    AsianOption,
    BarrierOption,
    BinaryOption,
    LookbackOption,
    VarianceSwap,
)
from .forwards import Forward, Future
from .market import VanillaMarketInputs
from .options import EuropeanOption
from .payoffs import payoff, payoff_requirement
from .pricing import (
    greeks_instrument,
    implied_volatility_instrument,
    price_instrument,
)
from .registry import InstrumentTypeInfo, instrument_type, instrument_types
from .serialization import (
    instrument_from_dict,
    instrument_from_json,
    instrument_to_dict,
    instrument_to_json,
)

__all__ = [
    "Asset",
    "AsianOption",
    "AveragingMethod",
    "BarrierOption",
    "BarrierType",
    "BinaryOption",
    "LookbackOption",
    "VarianceSwap",
    "StrikeConvention",
    "price_instrument",
    "implied_volatility_instrument",
    "greeks_instrument",
    "time_to_maturity",
    "moneyness",
    "log_moneyness",
    "forward_price",
    "EuropeanOptionBatch",
    "VanillaMarketInputs",
    "payoff_requirement",
    "payoff",
    "instrument_types",
    "instrument_type",
    "instrument_to_json",
    "instrument_to_dict",
    "instrument_from_json",
    "instrument_from_dict",
    "capabilities",
    "InstrumentTypeInfo",
    "CapabilitySet",
    "Future",
    "Forward",
    "EuropeanOption",
    "AssetClass",
    "Derivative",
    "ExerciseStyle",
    "IVSolver",
    "Instrument",
    "InstrumentError",
    "InstrumentKind",
    "InstrumentRef",
    "InstrumentValidationError",
    "MissingMarketInputError",
    "OptionType",
    "PayoffRequirement",
    "PricingModel",
    "SerializationError",
    "SettlementType",
    "UnsupportedInstrumentError",
    "UnsupportedModelError",
    "UnsupportedSolverError",
]
