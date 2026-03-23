from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from fast_vollib.api import get_all_greeks


def _load_fixture() -> pd.DataFrame:
    path = Path(__file__).parent / "fixtures" / "copied_from_py_vollib_vectorized.json"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return pd.DataFrame(data["data"], index=data["index"], columns=data["columns"])


def test_delta_gamma_vega_match_fixture_for_calls() -> None:
    df = _load_fixture()
    flags = np.full(len(df), "c")
    greeks = get_all_greeks(flags, df["S"], df["K"], df["t"], df["R"], df["v"], return_as="dict")
    assert_allclose(greeks["delta"], df["CD"].to_numpy(), atol=1e-4)
    assert_allclose(greeks["gamma"], df["CG"].to_numpy(), atol=1e-4)
    assert_allclose(greeks["vega"], df["CV"].to_numpy() * 0.01, atol=1e-3)
