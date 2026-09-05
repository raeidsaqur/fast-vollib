"""Stored numerical references with bounded cross-platform rounding.

See scripts/reference_fixtures.py for tolerances and strict comparison mode.
Repeated local renders must remain byte-identical; stored Linux references
need not share the host's libm or SciPy rounding. These regression fixtures
are not independent pricing oracles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from reference_fixtures import assert_reference_array, assert_reference_text

from fast_vollib.pricing import bcc97_price

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "pricing" / "bcc97_prices.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES = FIXTURE["cases"]
STRIKES = np.array(FIXTURE["strikes"], dtype=np.float64)


def ident(case: dict[str, Any]) -> str:
    side = "call" if case["is_call"] else "put"
    return (
        f"{case['heston']}-{case['jumps']}-{case['rates']}-"
        f"{case['formulation']}-T{case['maturity']}-{side}"
    )


def test_the_fixture_records_its_provenance() -> None:
    assert FIXTURE["generator"] == "scripts/generate_bcc97_reference_fixtures.py"
    assert "content_sha256" in FIXTURE
    assert CASES


def test_the_fixture_covers_the_grid_it_claims_to() -> None:
    assert {case["formulation"] for case in CASES} == {"lewis", "gatheral"}
    assert {case["is_call"] for case in CASES} == {True, False}
    assert {case["rates"] for case in CASES} == set(FIXTURE["rate_parameters"])
    assert FIXTURE["dividend_yield"] != 0.0, "a zero yield would not pin the forward"
    flat = FIXTURE["rate_parameters"]["flat"]
    assert flat["rate_volatility"] == 0.0
    assert flat["rate_theta"] == flat["initial_rate"], "the Bates reduction needs a flat curve"


def test_the_fixture_is_portable_and_regeneration_is_deterministic() -> None:
    import sys

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from generate_bcc97_reference_fixtures import render_prices

    rendered = render_prices()
    assert rendered == render_prices()
    assert_reference_text(rendered, FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=ident)
def test_the_pricer_reproduces_its_recorded_prices(case: dict[str, Any]) -> None:
    priced = np.asarray(
        bcc97_price(
            spot=FIXTURE["spot"],
            strike=STRIKES,
            maturity=case["maturity"],
            is_call=case["is_call"],
            dividend_yield=FIXTURE["dividend_yield"],
            formulation=case["formulation"],
            **FIXTURE["heston_parameters"][case["heston"]],
            **FIXTURE["jump_parameters"][case["jumps"]],
            **FIXTURE["rate_parameters"][case["rates"]],
        )
    )
    assert list(priced.shape) == case["shape"]
    expected = np.array([float.fromhex(x) for x in case["prices_hex"]]).reshape(priced.shape)
    assert_reference_array(priced, expected, pricing=True)
