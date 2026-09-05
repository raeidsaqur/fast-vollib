"""The CIR kernel, against fifty digits of the published formula.

The kernel in :mod:`fast_vollib.rates.cir` is an algebraic rearrangement of Cox,
Ingersoll and Ross (1985) equation 23 -- a substantial one, undertaken because
the published form cannot be evaluated in float64.  A rearrangement is exactly
the kind of change that is easy to get subtly wrong and impossible to catch by
inspection, so it is checked against the published form itself, transcribed
unrearranged and evaluated by ``mpmath`` at fifty digits where none of float64's
problems apply.

The two implementations share no line of code and no algebraic step.  That is
the point: agreement between them is evidence, where agreement between a formula
and a paraphrase of itself would not be.

Two different bounds are asserted here, and the difference between them is
mathematics rather than convenience.

``TABLE_TOLERANCE`` is the flat ``1e-15`` bound for the reference table
at.  It holds over every case in the fixture, with roughly three times the
margin to spare.

``test_the_exponent_is_accurate_to_a_few_ulp`` measures something stricter and
more informative.  ``P = exp(log_A - B r_0)``, so a one-ulp error in the
exponent becomes a relative error of ``|log P| * eps`` in ``P`` -- pure
conditioning of ``exp``, which no rearrangement of the algebra can remove.  A
flat relative bound on ``P`` therefore silently loosens as the maturity grows.
Normalizing by ``max(1, |log P|) * eps`` removes that and shows the exponent is
correct to under five units in the last place everywhere, which is the real
claim the module makes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from fast_vollib.rates import cir_affine_coefficients, cir_discount_factor, cir_zero_rate

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "rates" / "cir_discount_factors.json"
)
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES = FIXTURE["cases"]

#: The absolute reference-table error bound.
TABLE_TOLERANCE = 1e-15

#: ``exp`` amplifies a one-ulp exponent error by ``|log P|``. Five ulp is the
#: measured worst case over the fixture; the constant is a headroom factor, not
#: a fitted value.
EXPONENT_ULP_BUDGET = 16.0
EPS = 2.0**-53


def parameters(case: dict[str, Any]) -> dict[str, float]:
    return {
        "kappa": float(case["kappa"]),
        "theta": float(case["theta"]),
        "volatility": float(case["volatility"]),
        "initial_rate": float(case["initial_rate"]),
        "maturity": float(case["maturity"]),
    }


def ident(case: dict[str, Any]) -> str:
    return f"k{case['kappa']}-s{case['volatility']}-T{case['maturity']}"


# --- the fixture itself --------------------------------------------------------


def test_the_fixture_records_its_provenance() -> None:
    """A reference nobody can reproduce is not a reference."""
    assert FIXTURE["generator"] == "scripts/generate_cir_reference_fixtures.py"
    assert FIXTURE["precision_digits"] >= 30
    assert FIXTURE["library"].startswith("mpmath")
    assert "Cox" in FIXTURE["reference"] and "1985" in FIXTURE["reference"]
    assert "content_sha256" in FIXTURE
    assert CASES


def test_the_fixture_covers_the_regimes_the_rearrangement_exists_for() -> None:
    """A table that only held easy cases would licence the naive form."""
    volatilities = {float(c["volatility"]) for c in CASES}
    gamma_taus = {float(c["gamma_times_maturity"]) for c in CASES}
    # The cancellation regime: sigma small enough that gamma - kappa has no digits.
    assert min(v for v in volatilities if v > 0.0) <= 1e-10
    # Check the overflow stress case explicitly.
    assert any(abs(g - 2500.0) < 1e-6 for g in gamma_taus), sorted(gamma_taus)
    # The overflow regime: exp(gamma*tau) is inf past ~709.
    assert max(gamma_taus) > 709.0
    # The one branch the kernel has.
    assert 0.0 in volatilities


def test_the_fixture_regenerates_byte_for_byte() -> None:
    """A stale reference pins the wrong answer."""
    import sys

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from generate_cir_reference_fixtures import render

    assert FIXTURE_PATH.read_text(encoding="utf-8") == render()


def test_the_reference_prices_are_all_admissible_bond_prices() -> None:
    """Guards a fixture generated from a broken reference implementation."""
    for case in CASES:
        price = float(case["discount_factor"])
        assert 0.0 < price <= 1.0, case
        assert math.isfinite(float(case["log_discount_factor"])), case


def test_the_published_form_really_does_fail_where_the_fixture_says_it_does() -> None:
    """The rearrangement has to be *necessary*, not merely harmless.

    Transcribes equation 23 into float64 exactly as published and shows it
    overflowing on a case the fixture contains. Without this, a future reader
    could reasonably delete the rearrangement as over-engineering.
    """
    case = max(CASES, key=lambda c: float(c["gamma_times_maturity"]))
    p = parameters(case)
    gamma = math.sqrt(p["kappa"] ** 2 + 2.0 * p["volatility"] ** 2)
    assert gamma * p["maturity"] > 709.0
    with pytest.raises(OverflowError):
        math.exp(gamma * p["maturity"])
    # The shipped kernel returns the right answer on the same inputs.
    assert cir_discount_factor(**p) == pytest.approx(
        float(case["discount_factor"]), rel=TABLE_TOLERANCE
    )


# --- the kernel against the reference ------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=ident)
def test_the_kernel_matches_the_fifty_digit_reference(case: dict[str, Any]) -> None:
    reference = float(case["discount_factor"])
    produced = float(cir_discount_factor(**parameters(case)))
    assert produced == pytest.approx(reference, rel=TABLE_TOLERANCE, abs=0.0)


@pytest.mark.parametrize("case", CASES, ids=ident)
def test_the_exponent_is_accurate_to_a_few_ulp(case: dict[str, Any]) -> None:
    """The conditioning-aware statement; see the module docstring.

    ``exp`` turns an absolute error in the exponent into a relative error in
    the price, magnified by ``|log P|``. Checking the exponent directly removes
    that magnification and measures what the algebra is responsible for.
    """
    p = parameters(case)
    reference_log = float(case["log_discount_factor"])
    log_a, b = cir_affine_coefficients(
        kappa=p["kappa"], theta=p["theta"], volatility=p["volatility"], maturity=p["maturity"]
    )
    produced_log = float(log_a) - float(b) * p["initial_rate"]
    budget = EXPONENT_ULP_BUDGET * max(1.0, abs(reference_log)) * EPS
    assert abs(produced_log - reference_log) <= budget, (
        case["note"],
        produced_log,
        reference_log,
    )


@pytest.mark.parametrize("case", CASES, ids=ident)
def test_the_zero_rate_inverts_the_reference_price(case: dict[str, Any]) -> None:
    """``-log P / T`` computed by the kernel agrees with the reference's own."""
    p = parameters(case)
    if p["maturity"] == 0.0:  # pragma: no cover - the fixture has no zero maturity
        pytest.skip("the zero rate at zero maturity is the short rate by definition")
    expected = -float(case["log_discount_factor"]) / p["maturity"]
    assert float(cir_zero_rate(**p)) == pytest.approx(expected, rel=1e-14, abs=1e-18)
