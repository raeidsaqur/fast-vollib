# Author: Raeid Saqur
# Date: 2026-11-19
"""The one implied-volatility dispatcher below the public surfaces.

The functional (``fast_implied_volatility[_black]``), DataFrame (``price_dataframe``) and typed
(``implied_volatility_instrument``) surfaces all resolve a (solver, backend) pair here and call
the backend module directly; none of them calls another public surface. A pair that cannot be
served raises :class:`~fast_vollib.instruments.errors.UnsupportedSolverError`; nothing is ever
substituted.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from . import backends
from .config import get_backend

if TYPE_CHECKING:
    from .instruments.enums import IVSolver
    from .types import BackendLiteral, IVSolverLiteral

#: Backends the Jäckel solver has a full-model wrapper for. Numba is not one of them; the direct
#: Triton Jäckel kernel is not a public route.
JACKEL_BACKENDS = ("numpy", "torch", "jax")


def resolve_solver(
    solver: IVSolver | IVSolverLiteral | None, *, default: str = "halley"
) -> IVSolver:
    """The solver enum for an explicit request; ``None`` means ``default``."""
    from .instruments.enums import IVSolver
    from .instruments.errors import UnsupportedSolverError

    if solver is None:
        return IVSolver(default)
    if isinstance(solver, IVSolver):
        return solver
    if isinstance(solver, str):
        try:
            return IVSolver(solver)
        except ValueError:
            pass
    valid = ", ".join(repr(member.value) for member in IVSolver)
    raise UnsupportedSolverError(
        f"Unknown implied-volatility solver {solver!r}. Available solvers: {valid}."
    )


def jackel_backend_module(backend_name: str, *, requested: BackendLiteral | str) -> Any:
    """The Jäckel full-model wrapper for a backend, or a typed refusal (never a substitute)."""
    from .instruments.errors import UnsupportedSolverError

    if backend_name in JACKEL_BACKENDS:
        return importlib.import_module(f"{__package__}.jackel.{backend_name}_backend")
    available = ", ".join(JACKEL_BACKENDS)
    detail = (
        f"Request solver='halley' for the {backend_name!r} backend, or a backend the "
        f"Jäckel solver supports; the solver is never substituted silently."
        if requested != "auto"
        else (
            f"backend='auto' resolved to {backend_name!r} here. Pass an explicit "
            f"backend the Jäckel solver supports, or solver='halley'."
        )
    )
    raise UnsupportedSolverError(
        f"The Jäckel solver has no {backend_name!r} implementation "
        f"(available: {available}). {detail}"
    )


def solve(
    model: str,
    price: Any,
    s: Any,
    k: Any,
    t: Any,
    r: Any,
    flag: Any,
    q: Any,
    *,
    solver: IVSolver,
    backend: BackendLiteral,
    on_error: str,
) -> tuple[Any, str]:
    """Invert preprocessed (broadcast, validated) arrays; returns (values, backend_name)."""
    from .instruments.enums import IVSolver

    backend_name = get_backend(backend)
    if solver is IVSolver.HALLEY:
        module = backends.get_module(backend_name)
    else:
        module = jackel_backend_module(backend_name, requested=backend)
    values = module.implied_volatility(model, price, s, k, t, r, flag, q=q, on_error=on_error)
    return values, backend_name
