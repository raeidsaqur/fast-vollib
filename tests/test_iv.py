from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fast_vollib.backends import available_backends
from fast_vollib.implied_volatility import fast_implied_volatility
from fast_vollib.models import fast_black_scholes


def _load_fixture() -> pd.DataFrame:
    path = Path(__file__).parent / "fixtures" / "tests_data_py_vollib.json"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return pd.DataFrame(data["data"], index=data["index"], columns=data["columns"])


def test_implied_volatility_recovers_sigma() -> None:
    df = _load_fixture()
    flags = np.full(len(df), "c")
    prices = fast_black_scholes(
        flags, df["S"], df["K"], df["t"], df["R"], df["v"], return_as="numpy"
    )
    ivs = fast_implied_volatility(
        prices, df["S"], df["K"], df["t"], df["R"], flags, return_as="numpy"
    )
    # Skip cases where the option price underflows to ~0 in float64 (deep OTM, tiny T):
    # sigma is undetermined from a zero price, so IVs are NaN — not testable.
    solvable = np.abs(prices) > 1e-8
    assert np.allclose(ivs[solvable], df["v"].to_numpy()[solvable], atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_implied_volatility_raises_below_intrinsic_for_supported_backends(backend: str) -> None:
    if backend != "numpy" and backend not in available_backends():
        pytest.skip(f"{backend} not installed")

    with pytest.raises(ValueError, match="below intrinsic"):
        fast_implied_volatility(
            np.array([0.01]),
            np.array([100.0]),
            np.array([50.0]),
            np.array([1.0]),
            np.array([0.0]),
            np.array(["c"]),
            backend=backend,
            on_error="raise",
            return_as="numpy",
        )
