from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _iv_dispatch, backends
from .types import BackendLiteral, IVSolverLiteral, ModelLiteral, OnErrorLiteral, ReturnAsLiteral
from .utils.broadcast import maybe_format_data_and_broadcast, preprocess_flags
from .utils.formatting import format_named_output
from .utils.validation import ensure_on_error, validate_data

if TYPE_CHECKING:
    from ._typing import ArrayLike, FlagLike, OptionalArrayLike  # noqa: F401
    from .instruments.enums import IVSolver


def fast_implied_volatility(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    flag: FlagLike,
    q: OptionalArrayLike = None,
    *,
    on_error: OnErrorLiteral = "warn",
    model: ModelLiteral = "black_scholes",
    return_as: ReturnAsLiteral = "dataframe",
    dtype=np.float64,
    backend: BackendLiteral = "auto",
    return_native: bool = False,
    solver: IVSolver | IVSolverLiteral | None = None,
    **kwargs,
):
    """Implied volatility (Black-Scholes, Black-Scholes-Merton or Black-76).

    ``solver`` selects the inversion: ``None`` (default) or ``"halley"`` for the Halley solver,
    ``"jackel"`` for Jäckel's "Let's Be Rational" (NumPy, PyTorch and JAX backends). An unknown
    solver, or one the resolved backend cannot serve, raises ``UnsupportedSolverError``; the
    solver is never substituted. Other unrecognised keyword arguments are accepted and ignored
    for py_vollib_vectorized compatibility.
    """
    del kwargs
    ensure_on_error(on_error)
    resolved_solver = _iv_dispatch.resolve_solver(solver)
    flag = preprocess_flags(flag)
    if model == "black_scholes_merton":
        if q is None:
            raise ValueError(
                "Must pass a `q` to black scholes merton model (annualized continuous dividend yield)."
            )
        price, S, K, t, r, q, flag = maybe_format_data_and_broadcast(
            price, S, K, t, r, q, flag, dtype=dtype
        )
        validate_data(price, S, K, t, r, q)
    else:
        price, S, K, t, r, flag = maybe_format_data_and_broadcast(
            price, S, K, t, r, flag, dtype=dtype
        )
        validate_data(price, S, K, t, r)
    values, backend_name = _iv_dispatch.solve(
        model,
        price,
        S,
        K,
        t,
        r,
        flag,
        q if model == "black_scholes_merton" else None,
        solver=resolved_solver,
        backend=backend,
        on_error=on_error,
    )
    if return_native and backend_name in {"torch", "jax"}:
        return backends.get_module(backend_name).to_native(values)
    if return_as == "numpy":
        return values
    return format_named_output(values, "IV", return_as)


def fast_implied_volatility_black(
    price: ArrayLike,
    F: ArrayLike,
    K: ArrayLike,
    r: ArrayLike,
    t: ArrayLike,
    flag: FlagLike,
    *,
    on_error: OnErrorLiteral = "warn",
    return_as: ReturnAsLiteral = "dataframe",
    dtype=np.float64,
    backend: BackendLiteral = "auto",
    return_native: bool = False,
    solver: IVSolver | IVSolverLiteral | None = None,
    **kwargs,
):
    """Black-76 implied volatility; ``solver`` as in :func:`fast_implied_volatility`."""
    del kwargs
    return fast_implied_volatility(
        price,
        F,
        K,
        t,
        r,
        flag,
        on_error=on_error,
        model="black",
        return_as=return_as,
        dtype=dtype,
        backend=backend,
        return_native=return_native,
        solver=solver,
    )
