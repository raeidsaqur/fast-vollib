from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

from fast_vollib.models import fast_black_scholes, fast_black_scholes_merton


def _load_fixture() -> pd.DataFrame:
    path = Path(__file__).parent / "fixtures" / "tests_data_py_vollib.json"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return pd.DataFrame(data["data"], index=data["index"], columns=data["columns"])


def test_black_scholes_matches_fixture_prices() -> None:
    df = _load_fixture()
    flags = np.full(len(df), "c")
    prices = fast_black_scholes(
        flags,
        df["S"],
        df["K"],
        df["t"],
        df["R"],
        df["v"],
        return_as="numpy",
    )
    assert_allclose(prices, df["bs_call"].to_numpy(), atol=1e-6)


def test_black_scholes_merton_zero_dividend_matches_fixture() -> None:
    df = _load_fixture()
    flags = np.full(len(df), "p")
    q = np.zeros(len(df))
    prices = fast_black_scholes_merton(
        flags,
        df["S"],
        df["K"],
        df["t"],
        df["R"],
        df["v"],
        q,
        return_as="numpy",
    )
    assert_allclose(prices, df["bs_put"].to_numpy(), atol=1e-6)
