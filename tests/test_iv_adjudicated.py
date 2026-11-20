# Author: Raeid Saqur
# Date: 2026-11-19
"""Adjudicated implied-volatility fixtures on every available route (protocol revision 3).

The fixtures (``tests/fixtures/iv_adjudication/fixtures.json``) hold the three sub-normal Jäckel
calls, the Halley low-volatility example, boundary neighbours and matched controls, each with an
independent high-precision reference. Every route must pass every fixture under the frozen
protocol, except the listed known failures, which must fail only as NaN with the on_error signal.
"""

from __future__ import annotations

from _cases import available_routes, load_fixtures, run_route, score
import numpy as np
import pytest

ROUTES = available_routes()

#: (solver, fixture key) -> reason. Each must be NaN with the signal, never a finite value.
KNOWN = {
    ("jackel", "fixtures:B-upper-1ulp"): (
        "one ulp below the upper bound (sigma = 16.5): the Jäckel solver's normalised price leaves "
        "too little of b_max - beta, so the repricing check refuses the row; open item"
    ),
}


@pytest.fixture(scope="module")
def fixtures():
    return load_fixtures()


@pytest.mark.parametrize(("solver", "backend"), ROUTES, ids=[f"{s}-{b}" for s, b in ROUTES])
def test_adjudicated_fixtures(fixtures, solver: str, backend: str) -> None:
    result = run_route(fixtures, solver, backend)
    ok = score(fixtures, result, solver)
    failed = set(fixtures["key"][~ok])
    known = {key for (s, key) in KNOWN if s == solver}
    assert failed <= known, sorted(failed - known)
    for key in failed:
        i = int(np.flatnonzero(fixtures["key"] == key)[0])
        assert np.isnan(result.values[i]), (key, result.values[i])
        # Not silent in the batch: the call's aggregated warning counts every NaN row.
        assert result.batch_accounted is not None and result.batch_accounted[i], key
        if not result.signalled[i]:
            # The signal is attributed by re-running the row alone. On a knife-edge row a device
            # library may evaluate the one-row batch slightly differently and solve it; then the
            # lone result must itself pass the protocol (it is not a silent wrong value).
            alone = fixtures.iloc[[i]].reset_index(drop=True)
            lone = run_route(alone, solver, backend)
            assert score(alone, lone, solver).all(), (key, lone.values)
    assert result.lib_warnings_per_call <= 1  # one aggregated signal per call


@pytest.mark.parametrize("backend", [b for s, b in ROUTES if s == "halley"])
def test_halley_low_volatility_example(fixtures, backend: str) -> None:
    """F=100, K=100 e^-0.1, T=1, sigma_gen=0.0182: v0.2.4 returned 0.0176204 without a signal."""
    rows = fixtures[fixtures["key"].isin(["fixtures:H1", "fixtures:H1x"])].reset_index(drop=True)
    result = run_route(rows, "halley", backend)
    np.testing.assert_allclose(result.values, rows["ref_f64"].to_numpy(), rtol=1e-8)


@pytest.mark.parametrize("backend", [b for s, b in ROUTES if s == "jackel"])
def test_subnormal_jackel_calls_are_refused_with_the_signal(fixtures, backend: str) -> None:
    """J1-J3: normalised price below DBL_MIN, outside the Jäckel domain (protocol revision 3)."""
    rows = fixtures[
        fixtures["key"].isin(["fixtures:J1", "fixtures:J2", "fixtures:J3"])
    ].reset_index(drop=True)
    result = run_route(rows, "jackel", backend)
    assert np.isnan(result.values).all() and result.signalled.all()
    halley = run_route(rows, "halley", "numpy")  # the Halley solver solves them
    np.testing.assert_allclose(halley.values, rows["ref_f64"].to_numpy(), rtol=1e-8)
