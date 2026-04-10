from typing import Literal

BackendLiteral = Literal["auto", "numpy", "torch", "jax", "numba"]
ModelLiteral = Literal["black", "black_scholes", "black_scholes_merton"]
OnErrorLiteral = Literal["raise", "warn", "ignore"]
ReturnAsLiteral = Literal["dataframe", "series", "numpy", "dict", "json"]
