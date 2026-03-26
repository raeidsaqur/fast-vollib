from __future__ import annotations

import numpy as np

from fast_vollib.api import get_all_greeks
from fast_vollib.implied_volatility import fast_implied_volatility


def test_iv_broadcasting() -> None:
    ivs = fast_implied_volatility(
        price=0.10,
        S=np.repeat([30.0], 2),
        K=30.0,
        t=10 / 365.0,
        r=0.05,
        flag=np.repeat(["c"], 2),
        return_as="numpy",
    )
    assert ivs[0] == ivs[1]


def test_greeks_broadcasting() -> None:
    greeks = get_all_greeks(
        sigma=0.10,
        S=np.repeat([30.0], 2),
        K=30.0,
        t=10 / 365.0,
        r=0.05,
        flag=np.repeat(["c"], 2),
        return_as="dict",
    )
    assert greeks["delta"][0] == greeks["delta"][1]
