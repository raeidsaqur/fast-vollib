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

from fast_vollib._simulation_errors import UnsupportedProcessError
from fast_vollib.processes import CIR_SCHEMES, CIRShortRate

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "processes" / "cir_paths.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES = FIXTURE["cases"]

GRID = np.array(FIXTURE["time_grid"], dtype=np.float64)


def ident(case: dict[str, Any]) -> str:
    anti = "anti" if case["antithetic"] else "plain"
    return f"{case['parameters']}-{case['scheme']}-{anti}"


def produced(case: dict[str, Any]) -> Any:
    process = CIRShortRate(**FIXTURE["parameter_sets"][case["parameters"]])
    return process.sample(
        initial_state=FIXTURE["initial_state"],
        time_grid=GRID,
        n_paths=FIXTURE["n_paths"],
        rng=FIXTURE["seed"],
        antithetic=case["antithetic"],
        scheme=case["scheme"],
    )


# --- the fixture itself --------------------------------------------------------


def test_the_fixture_records_its_provenance() -> None:
    assert FIXTURE["generator"] == "scripts/generate_cir_path_fixtures.py"
    assert FIXTURE["namespace"] == "numpy"
    assert FIXTURE["dtype"] == "float64"
    assert "content_sha256" in FIXTURE
    assert CASES


def test_the_fixture_covers_every_scheme_and_both_exact_constructions() -> None:
    """A freeze that missed a branch would licence changing it.

    The exact transition uses one formula above one degree of freedom and a
    different one below, with different draw layouts, so a fixture holding only
    one side would leave the other unpinned.
    """
    assert {case["scheme"] for case in CASES} == set(CIR_SCHEMES)
    degrees = [case["degrees_of_freedom"] for case in CASES]
    assert max(degrees) > 1.0 and min(degrees) <= 1.0, degrees
    assert {case["antithetic"] for case in CASES} == {False, True}


def test_the_fixture_is_portable_and_regeneration_is_deterministic() -> None:
    """A stale reference pins the wrong answer."""
    import sys

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from generate_cir_path_fixtures import render

    rendered = render()
    assert rendered == render()
    assert_reference_text(rendered, FIXTURE_PATH.read_text(encoding="utf-8"))


# --- the sampler against the record --------------------------------------------


@pytest.mark.parametrize("case", [c for c in CASES if not c["refused"]], ids=ident)
def test_the_sampler_reproduces_its_recorded_paths(case: dict[str, Any]) -> None:
    paths = np.asarray(produced(case))
    assert list(paths.shape) == case["shape"]
    expected = np.array([float.fromhex(x) for x in case["paths_hex"]]).reshape(paths.shape)
    assert_reference_array(paths, expected)


@pytest.mark.parametrize("case", [c for c in CASES if c["refused"]], ids=ident)
def test_a_recorded_refusal_is_still_a_refusal(case: dict[str, Any]) -> None:
    """A refusal is behaviour too, so removing one shows up here as well.

    Below one degree of freedom the exact transition contains no normal, so an
    antithetic pair would be two identical paths.
    """
    with pytest.raises(UnsupportedProcessError):
        produced(case)


def test_strict_comparison_rejects_a_last_bit_change(monkeypatch) -> None:
    """Exercise the fixture comparator, not merely float inequality."""
    import math

    case = next(c for c in CASES if not c["refused"])
    value = float.fromhex(case["paths_hex"][-1])
    changed = math.nextafter(value, math.inf)
    assert float.fromhex(float(value).hex()) == value
    monkeypatch.setenv("FV_STRICT_REFERENCE_FIXTURES", "0")
    assert_reference_array(changed, value)
    monkeypatch.setenv("FV_STRICT_REFERENCE_FIXTURES", "1")
    assert_reference_array(value, value)
    with pytest.raises(AssertionError):
        assert_reference_array(changed, value)
