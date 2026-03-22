from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fastiv.api import price_dataframe


def _load_fixture() -> pd.DataFrame:
    path = Path(__file__).parent / "fixtures" / "copied_from_py_vollib_vectorized.json"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    df = pd.DataFrame(data["data"], index=data["index"], columns=data["columns"])
    df["flag"] = "p"
    df["q"] = 0.0
    return df


def test_price_dataframe_adds_expected_columns() -> None:
    df = _load_fixture()
    result = price_dataframe(
        df,
        flag_col="flag",
        underlying_price_col="S",
        strike_col="K",
        annualized_tte_col="t",
        riskfree_rate_col="R",
        sigma_col="v",
        dividend_col="q",
        model="black_scholes_merton",
        inplace=False,
    )
    for column in ["Price", "delta", "gamma", "theta", "rho", "vega"]:
        assert column in result.columns
