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

from fast_vollib.pricing import FORMULATIONS, heston_price

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "pricing" / "heston_prices.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

FORWARD = float(FIXTURE["forward"])
STRIKES = np.array(FIXTURE["strikes"], dtype=np.float64)
DISCOUNT = float(FIXTURE["discount"])


def decode(case: dict[str, Any]) -> np.ndarray:
    flat = np.array([float.fromhex(text) for text in case["prices_hex"]], dtype=np.float64)
    return flat.reshape(tuple(case["shape"]))


def ident(case: dict[str, Any]) -> str:
    right = "call" if case["is_call"] else "put"
    return f"{case['parameters']}-{case['formulation']}-T{case['maturity']}-{right}"


def reprice(case: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        heston_price(
            forward=FORWARD,
            strike=STRIKES,
            maturity=case["maturity"],
            is_call=case["is_call"],
            discount=DISCOUNT,
            formulation=case["formulation"],
            **FIXTURE["parameter_sets"][case["parameters"]],
        )
    )


# --- the fixture itself --------------------------------------------------------


def test_the_fixture_records_its_provenance() -> None:
    assert FIXTURE["generator"] == "scripts/generate_heston_reference_fixtures.py"
    assert "float.hex()" in FIXTURE["encoding"]
    assert "content_sha256" in FIXTURE
    assert FIXTURE["cases"]


def test_the_fixture_covers_both_formulations_and_both_rights() -> None:
    """Freezing only calls would leave the put-call parity branch unpinned."""
    covered = {(case["formulation"], bool(case["is_call"])) for case in FIXTURE["cases"]}
    assert covered == {(f, r) for f in FORMULATIONS for r in (True, False)}


def test_the_fixture_discount_is_not_one() -> None:
    """A discount factor dropped on the floor must change the frozen answer."""
    assert DISCOUNT != 1.0


def test_the_fixture_spans_the_swept_grid() -> None:
    """The freeze covers the range the accuracy claims are made over."""
    assert set(FIXTURE["parameter_sets"]) == {"moderate", "heavy", "fast"}
    assert min(FIXTURE["maturities"]) <= 0.02
    assert max(FIXTURE["maturities"]) >= 30.0
    assert min(FIXTURE["strikes"]) < FORWARD < max(FIXTURE["strikes"])


def test_the_fixture_is_portable_and_regeneration_is_deterministic() -> None:
    import sys

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from generate_heston_reference_fixtures import render_prices

    rendered = render_prices()
    assert rendered == render_prices()
    assert_reference_text(rendered, FIXTURE_PATH.read_text(encoding="utf-8"))


def test_the_hex_encoding_round_trips_exactly() -> None:
    for case in FIXTURE["cases"]:
        for text in case["prices_hex"]:
            assert float.fromhex(text).hex() == text


# --- the pricer against the freeze ---------------------------------------------


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=ident)
def test_the_price_is_bitwise_unchanged(case: dict[str, Any]) -> None:
    """Compare with the portable numerical reference bound."""
    expected = decode(case)
    produced = reprice(case)
    assert produced.dtype == np.float64
    assert produced.shape == expected.shape
    assert_reference_array(produced, expected, pricing=True)


def test_the_frozen_calls_and_puts_satisfy_parity() -> None:
    """Guards against a fixture regenerated from a pricer with a broken parity.

    ``c - p = discount * (F - K)`` exactly, because the parity is applied to the
    undiscounted forward price before the discount factor multiplies through.
    """
    by_key = {ident(case): decode(case) for case in FIXTURE["cases"]}
    for name in FIXTURE["parameter_sets"]:
        for formulation in FORMULATIONS:
            for maturity in FIXTURE["maturities"]:
                calls = by_key[f"{name}-{formulation}-T{maturity}-call"]
                puts = by_key[f"{name}-{formulation}-T{maturity}-put"]
                np.testing.assert_allclose(
                    calls - puts, DISCOUNT * (FORWARD - STRIKES), rtol=0.0, atol=1e-12
                )
