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

from fast_vollib.processes import (
    SCHEMES,
    Bates,
    ConstantVariance,
    HestonVariance,
    LognormalJumps,
    NoJumps,
)

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "processes" / "bates_paths.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES = FIXTURE["cases"]
GRID = np.array(FIXTURE["time_grid"], dtype=np.float64)


def ident(case: dict[str, Any]) -> str:
    anti = "anti" if case["antithetic"] else "plain"
    return f"{case['variance']}-{case['jumps']}-{case['scheme']}-{anti}"


def process_for(case: dict[str, Any]) -> Bates:
    variance_params = FIXTURE["variance_components"][case["variance"]]
    jump_params = FIXTURE["jump_components"][case["jumps"]]
    return Bates(
        variance=ConstantVariance()
        if variance_params is None
        else HestonVariance(**variance_params),
        jumps=NoJumps() if jump_params is None else LognormalJumps(**jump_params),
        drift=FIXTURE["drift"],
    )


def test_the_fixture_records_its_provenance() -> None:
    assert FIXTURE["generator"] == "scripts/generate_bates_reference_fixtures.py"
    assert FIXTURE["namespace"] == "numpy"
    assert FIXTURE["dtype"] == "float64"
    assert "content_sha256" in FIXTURE
    assert CASES


def test_the_fixture_covers_every_corner_of_the_lattice() -> None:
    """A freeze that missed a configuration would licence changing it."""
    assert {case["variance"] for case in CASES} == {"heston", "constant"}
    assert {case["jumps"] for case in CASES} == {"lognormal", "none"}
    assert {case["scheme"] for case in CASES} == set(SCHEMES)
    assert {case["antithetic"] for case in CASES} == {False, True}
    assert len(CASES) == 2 * 2 * len(SCHEMES) * 2


def test_the_fixture_is_portable_and_regeneration_is_deterministic() -> None:
    import sys

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from generate_bates_reference_fixtures import render_paths

    rendered = render_paths()
    assert rendered == render_paths()
    assert_reference_text(rendered, FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=ident)
def test_the_sampler_reproduces_its_recorded_paths(case: dict[str, Any]) -> None:
    paths = np.asarray(
        process_for(case).sample(
            initial_state=FIXTURE["initial_state"],
            time_grid=GRID,
            n_paths=FIXTURE["n_paths"],
            rng=FIXTURE["seed"],
            antithetic=case["antithetic"],
            scheme=case["scheme"],
        )
    )
    assert list(paths.shape) == case["shape"]
    expected = np.array([float.fromhex(x) for x in case["paths_hex"]]).reshape(paths.shape)
    assert_reference_array(paths, expected)
