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

import pytest
from reference_fixtures import assert_reference_array, assert_reference_text

from fast_vollib.instruments import VanillaMarketInputs
from fast_vollib.processes import GBM
from fast_vollib.simulation import MonteCarloEngine

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "simulation" / "engine_prices.json"
)
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES = FIXTURE["cases"]


def _generator():
    """The generator module, which also owns the instrument definitions."""
    import sys

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import generate_engine_reference_fixtures as module

    return module


def ident(case: dict[str, Any]) -> str:
    anti = "anti" if case["antithetic"] else "plain"
    return f"{case['process']}-{anti}-{case['instrument']}"


# --- the fixture itself --------------------------------------------------------


def test_the_fixture_records_its_provenance() -> None:
    assert FIXTURE["generator"] == "scripts/generate_engine_reference_fixtures.py"
    assert FIXTURE["namespace"] == "numpy"
    assert FIXTURE["dtype"] == "float64"
    assert "content_sha256" in FIXTURE
    assert CASES


def test_the_fixture_covers_every_type_the_engine_supports() -> None:
    """A freeze that missed a dispatch arm would licence changing it."""
    recorded = {case["type"] for case in CASES}
    supported = {t.__name__ for t in MonteCarloEngine.SUPPORTED_TYPES}
    assert supported <= recorded, supported - recorded


def test_the_fixture_covers_both_estimators() -> None:
    """Pair-averaging is a separate branch of ``_estimate``."""
    assert {case["antithetic"] for case in CASES} == {False, True}
    for case in CASES:
        expected = case["n_paths"] // 2 if case["antithetic"] else case["n_paths"]
        assert case["effective_samples"] == expected, case


def test_the_fixture_is_portable_and_regeneration_is_deterministic() -> None:
    """A stale reference pins the wrong answer."""
    rendered = _generator().render()
    assert rendered == _generator().render()
    assert_reference_text(rendered, FIXTURE_PATH.read_text(encoding="utf-8"))


# --- the engine against the record ---------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=ident)
def test_a_default_engine_call_reproduces_its_recorded_price(case: dict[str, Any]) -> None:
    module = _generator()
    market = VanillaMarketInputs(
        underlying=FIXTURE["market"]["underlying"], rate=FIXTURE["market"]["rate"]
    )
    process = GBM(**FIXTURE["process_parameters"][case["process"]])
    instrument = module.instruments()[case["instrument"]]

    result = MonteCarloEngine(antithetic=case["antithetic"]).price(
        instrument,
        market,
        process=process,
        n_paths=FIXTURE["n_paths"],
        n_steps=FIXTURE["n_steps"],
        rng=FIXTURE["seed"],
    )

    assert_reference_array(result.price, float.fromhex(case["price_hex"]))
    assert_reference_array(result.stderr, float.fromhex(case["stderr_hex"]))
    assert result.n_paths == case["n_paths"]
    assert result.effective_samples == case["effective_samples"]


def test_strict_comparison_rejects_a_last_bit_change(monkeypatch) -> None:
    """Portable mode permits one ulp; strict mode must reject it."""
    import math

    recorded = float.fromhex(CASES[0]["price_hex"])
    changed = math.nextafter(recorded, math.inf)
    assert float.fromhex(float(recorded).hex()) == recorded
    monkeypatch.setenv("FV_STRICT_REFERENCE_FIXTURES", "0")
    assert_reference_array(changed, recorded)
    monkeypatch.setenv("FV_STRICT_REFERENCE_FIXTURES", "1")
    assert_reference_array(recorded, recorded)
    with pytest.raises(AssertionError):
        assert_reference_array(changed, recorded)
