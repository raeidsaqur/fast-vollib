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

from fast_vollib.processes import SCHEMES, Heston

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "processes" / "heston_paths.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

TIME_GRID = np.array(FIXTURE["time_grid"], dtype=np.float64)
INITIAL_STATE = FIXTURE["initial_state"]
N_PATHS = int(FIXTURE["n_paths"])
SEED = int(FIXTURE["seed"])


def decode(case: dict[str, Any]) -> np.ndarray:
    """The frozen array, reconstructed exactly from its hex literals."""
    flat = np.array([float.fromhex(text) for text in case["paths_hex"]], dtype=np.float64)
    return flat.reshape(tuple(case["shape"]))


def ident(case: dict[str, Any]) -> str:
    return (
        f"{case['parameters']}-{case['scheme']}-{'antithetic' if case['antithetic'] else 'plain'}"
    )


def resample(case: dict[str, Any]) -> np.ndarray:
    process = Heston(**FIXTURE["parameter_sets"][case["parameters"]])
    return np.asarray(
        process.sample(
            initial_state=INITIAL_STATE,
            time_grid=TIME_GRID,
            n_paths=N_PATHS,
            rng=SEED,
            antithetic=bool(case["antithetic"]),
            scheme=case["scheme"],
        )
    )


# --- the fixture itself --------------------------------------------------------


def test_the_fixture_records_its_provenance() -> None:
    """A frozen value whose origin is unrecorded cannot be audited."""
    assert FIXTURE["generator"] == "scripts/generate_heston_reference_fixtures.py"
    assert FIXTURE["namespace"] == "numpy"
    assert FIXTURE["dtype"] == "float64"
    assert "float.hex()" in FIXTURE["encoding"]
    assert "content_sha256" in FIXTURE
    assert FIXTURE["cases"]


def test_the_fixture_covers_both_schemes_antithetic_and_plain() -> None:
    """A freeze that skipped a branch would licence changing that branch."""
    covered = {(case["scheme"], bool(case["antithetic"])) for case in FIXTURE["cases"]}
    assert covered == {(scheme, flag) for scheme in SCHEMES for flag in (False, True)}


def test_the_fixture_covers_a_feller_violating_set_and_a_non_zero_drift() -> None:
    """The two places a sampler change is most likely to show up first."""
    sets = FIXTURE["parameter_sets"]
    ratios = {
        name: 2.0 * p["kappa"] * p["theta"] / p["vol_of_vol"] ** 2 for name, p in sets.items()
    }
    assert any(ratio > 1.0 for ratio in ratios.values()), ratios
    assert any(ratio < 1.0 for ratio in ratios.values()), ratios
    assert any(p["drift"] != 0.0 for p in sets.values())


def test_the_fixture_is_portable_and_regeneration_is_deterministic() -> None:
    """A stale freeze pins an output the library no longer produces."""
    import sys

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from generate_heston_reference_fixtures import render_paths

    rendered = render_paths()
    assert rendered == render_paths()
    assert_reference_text(rendered, FIXTURE_PATH.read_text(encoding="utf-8"))


def test_the_hex_encoding_round_trips_exactly() -> None:
    """The premise of the whole file: nothing is lost writing a float down."""
    for case in FIXTURE["cases"]:
        for text in case["paths_hex"]:
            assert float.fromhex(text).hex() == text


# --- the sampler against the freeze --------------------------------------------


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=ident)
def test_the_seeded_sample_is_bitwise_unchanged(case: dict[str, Any]) -> None:
    """Compare with the portable numerical reference bound."""
    expected = decode(case)
    produced = resample(case)
    assert produced.dtype == np.float64
    assert produced.shape == expected.shape
    assert_reference_array(produced, expected)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=ident)
def test_the_frozen_paths_start_at_the_initial_state_exactly(case: dict[str, Any]) -> None:
    """Catches a fixture regenerated from a broken sampler."""
    expected = decode(case)
    np.testing.assert_array_equal(expected[:, 0, 0], np.full(N_PATHS, INITIAL_STATE["spot"]))
    np.testing.assert_array_equal(expected[:, 0, 1], np.full(N_PATHS, INITIAL_STATE["variance"]))


def test_the_frozen_antithetic_paths_are_not_the_frozen_plain_ones() -> None:
    """Otherwise the antithetic cases would freeze nothing extra."""
    by_key = {ident(case): decode(case) for case in FIXTURE["cases"]}
    for scheme in SCHEMES:
        plain = by_key[f"calm-{scheme}-plain"]
        antithetic = by_key[f"calm-{scheme}-antithetic"]
        assert not np.array_equal(plain, antithetic)
