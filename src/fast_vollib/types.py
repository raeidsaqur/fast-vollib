from typing import Literal

BackendLiteral = Literal["auto", "numpy", "torch", "jax", "numba"]
ModelLiteral = Literal["black", "black_scholes", "black_scholes_merton"]
OnErrorLiteral = Literal["raise", "warn", "ignore"]
ReturnAsLiteral = Literal["dataframe", "series", "numpy", "dict", "json"]

# Instrument vocabularies. These mirror the enums in
# ``fast_vollib.instruments.enums`` value-for-value (a test pins the two
# together) so that callers can annotate with either surface: the Literal for
# plain-string call sites, the Enum for exhaustive matching.
InstrumentKindLiteral = Literal[
    "asset",
    "forward",
    "future",
    "european_option",
    "binary_option",
    "asian_option",
    "barrier_option",
    "lookback_option",
    "variance_swap",
    "zero_coupon_bond",
    "fixed_rate_bond",
]
ExerciseLiteral = Literal["european"]
IVSolverLiteral = Literal["halley", "jackel"]
