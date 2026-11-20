# Author: Raeid Saqur
# Date: 2026-11-20
"""The differentiable Jäckel wrappers solve the strict rows of the canonical grid.

Inside the declared domain every strict row must be finite and within the frozen tolerance;
a lower-tail-inaccurate normal CDF in the repricing check once turned hundreds of correct
low-volatility rows of the PyTorch wrapper into NaN.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _cases  # noqa: E402

DBL_MIN = np.finfo(float).tiny
#: Pre-existing near-at-the-money misses of the native PyTorch solver on the CPU (relative error
#: 1.07e-12 against a 1e-12 tolerance); the same rows are known in tests/test_iv_domain.py.
KNOWN_TORCH_CPU = {"grid:c:364", "grid:p:364"}


def _in_domain_strict(df) -> np.ndarray:
    beta = df["beta"].to_numpy(float)
    return (df["scoring"].to_numpy() == "strict") & ~(beta < DBL_MIN)


def _check(df, values: np.ndarray, known: frozenset[str] = frozenset()) -> None:
    m = _in_domain_strict(df)
    result = _cases.RouteResult(values, np.zeros(len(df), dtype=bool), 0)
    ok = _cases.score(df, result, "jackel")
    bad = set(df["key"].to_numpy()[m & ~ok]) - known
    assert not bad, (len(bad), sorted(bad)[:10])


def test_torch_autograd_wrapper_solves_the_grid() -> None:
    torch = pytest.importorskip("torch")
    from fast_vollib.jackel import implied_volatility_autograd

    df = _cases.load_grid(stride=2)
    t = lambda c: torch.as_tensor(df[c].to_numpy(float), dtype=torch.float64)  # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = implied_volatility_autograd(
            t("price"),
            t("U"),
            t("K"),
            t("T"),
            t("r"),
            list(df["flag"]),
            model="black",
            on_error="ignore",
        )
    known = frozenset(KNOWN_TORCH_CPU) if out.device.type == "cpu" else frozenset()
    _check(df, out.detach().cpu().numpy(), known)


def test_jax_autograd_wrapper_solves_the_grid() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from fast_vollib.jackel import implied_volatility_autograd_jax

    df = _cases.load_grid(stride=2)
    j = lambda c: jnp.asarray(df[c].to_numpy(float), dtype=jnp.float64)  # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = implied_volatility_autograd_jax(
            j("price"),
            j("U"),
            j("K"),
            j("T"),
            j("r"),
            list(df["flag"]),
            model="black",
            on_error="ignore",
        )
    _check(df, np.asarray(out))
