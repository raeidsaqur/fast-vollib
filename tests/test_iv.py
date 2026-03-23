from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fastiv.implied_volatility import vectorized_implied_volatility
from fastiv.models import vectorized_black_scholes


def _load_fixture() -> pd.DataFrame:
    path = Path(__file__).parent / "fixtures" / "copied_from_py_vollib_vectorized.json"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return pd.DataFrame(data["data"], index=data["index"], columns=data["columns"])


def test_implied_volatility_recovers_sigma() -> None:
    df = _load_fixture()
    flags = np.full(len(df), "c")
    prices = vectorized_black_scholes(flags, df["S"], df["K"], df["t"], df["R"], df["v"], return_as="numpy")
    ivs = vectorized_implied_volatility(prices, df["S"], df["K"], df["t"], df["R"], flags, return_as="numpy")
    # Skip cases where the option price underflows to ~0 in float64 (deep OTM, tiny T):
    # sigma is undetermined from a zero price, so IVs are NaN — not testable.
    solvable = np.abs(prices) > 1e-8
    assert np.allclose(ivs[solvable], df["v"].to_numpy()[solvable], atol=1e-6)
