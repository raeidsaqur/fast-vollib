# Author: Raeid Saqur
# Date: 2026-11-19
"""Implied-volatility domain sweep under the frozen protocol (revision 3).

Default: the canonical grid at stride 4 (21 x 21 x {C, P}, 882 rows), every class, every route.
``-m slow``: the full canonical grid (13,122 rows) and the held-out sample (2,000 rows).
"""

from __future__ import annotations

from _cases import available_routes, load_grid, load_heldout, run_route, score
import pytest

ROUTES = available_routes()

#: (solver, backend, device) -> keys allowed to fail, with the reason. Pre-existing, recorded.
KNOWN = {
    ("jackel", "torch", "cpu"): {
        "grid:c:364": "at-the-money (x = 0), relative error 1.07e-12 vs 1.0e-12; also in v0.2.4",
        "grid:p:364": "at-the-money (x = 0), relative error 1.07e-12 vs 1.0e-12; also in v0.2.4",
    },
}


def _device(backend: str) -> str:
    if backend == "torch":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if backend == "jax":
        import jax

        return jax.default_backend()
    return "cpu"


def _check(df, solver: str, backend: str) -> None:
    result = run_route(df, solver, backend)
    ok = score(df, result, solver)
    failed = set(df["key"][~ok])
    known = set(KNOWN.get((solver, backend, _device(backend)), {}))
    assert failed <= known, sorted(failed - known)[:20]
    assert result.lib_warnings_per_call <= 1


@pytest.mark.parametrize(("solver", "backend"), ROUTES, ids=[f"{s}-{b}" for s, b in ROUTES])
def test_coarse_grid(solver: str, backend: str) -> None:
    _check(load_grid(stride=4), solver, backend)


@pytest.mark.slow
@pytest.mark.parametrize(("solver", "backend"), ROUTES, ids=[f"{s}-{b}" for s, b in ROUTES])
def test_canonical_grid(solver: str, backend: str) -> None:
    _check(load_grid(stride=1), solver, backend)


@pytest.mark.slow
@pytest.mark.parametrize(("solver", "backend"), ROUTES, ids=[f"{s}-{b}" for s, b in ROUTES])
def test_heldout_sample(solver: str, backend: str) -> None:
    _check(load_heldout(), solver, backend)


def test_coarse_grid_covers_every_scoring_class() -> None:
    classes = set(load_grid(stride=4)["scoring"])
    # invalid inputs are covered by the adjudicated fixtures
    assert {"strict", "open_at_zero", "zero_quote", "inconsistent"} <= classes
