# Author: Raeid Saqur
# Date: 2026-11-19
"""Integrity of the frozen implied-volatility adjudication data.

These tests do not exercise any solver. They pin the tolerance protocol and check
that the independent reference (scripts/reference_inverse.py) reproduces the stored
fixtures bit for bit, so the reference cannot drift silently.
"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import pytest

mpmath = pytest.importorskip("mpmath")
if mpmath.__version__ != "1.3.0":  # pragma: no cover - environment guard
    pytest.skip("reference fixtures are tied to mpmath 1.3.0", allow_module_level=True)

import reference_inverse as ref  # noqa: E402  (scripts/ is on sys.path via conftest)

FIXTURES = Path(__file__).parent / "fixtures" / "iv_adjudication"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_protocol_constants_are_frozen() -> None:
    p = _load("protocol.json")
    assert p["schema"] == "fv-iv-protocol/1"
    t = p["tolerance"]
    assert t["form"] == (
        "tol = max(abs_floor, rel_tol * ref) + M * (h_quote + h_forward + h_discount)"
    )
    assert t["float64"] == {
        "abs_floor": 1e-15,
        "M": 2,
        "rel_tol": {"jackel": 1e-12, "halley": 1e-08},
    }
    assert set(p["scoring"]) >= {
        "strict",
        "open_at_zero",
        "open_at_infinity",
        "inconsistent",
        "invalid_input",
        "expiry",
    }
    assert "zero_quote" in p["scoring"]
    assert p["declared_domain"]["default"].startswith("all strict rows")
    # Revision 3 (owner-approved): the Jaeckel solver refuses beta < DBL_MIN; Halley is unrestricted.
    assert p["revision"] == 3
    assert p["declared_domain"]["current"]["halley"] is None
    assert p["declared_domain"]["current"]["jackel"]["rule"].startswith("beta >= DBL_MIN")
    # float32 constants are deliberately not frozen yet (see protocol status).
    assert t["float32_provisional"]["status"].startswith("NOT FROZEN")


def _contract(rec: dict) -> ref.Contract:
    fh = float.fromhex
    spot = rec.get("S_hex")
    return ref.Contract(
        F=fh(rec["F_hex"]) if rec.get("F_hex") else float("nan"),
        K=fh(rec["K_hex"]),
        T=fh(rec["T_hex"]),
        is_call=rec["flag"] == "c",
        r=fh(rec["r_hex"]),
        spot=fh(spot) if spot else None,
        q=fh(rec["q_hex"]) if rec.get("q_hex") else 0.0,
    )


FIXTURE_IDS = [
    "J1",
    "J2",
    "J3",
    "H1",
    "H1x",
    "NJ-median",
    "B-lb-itm",
    "B-below",
    "B-lowamb",
    "B-upper-1ulp",
    "B-minsub",
    "B-disc",
    "ATM-ctl",
    "B-T0",
    "B-invalid",
    "J1x",
    "B-intr-1ulp",
    "B-upper",
    "W-bs-call",
    "W-bsm-put",
    "W-bsm-atm",
]


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_reference_reproduces_fixture(fixture_id: str) -> None:
    doc = _load("fixtures.json")
    rec = next(f for f in doc["fixtures"] if f["id"] == fixture_id)
    res = ref.adjudicate(_contract(rec), float.fromhex(rec["quote_hex"]))
    stored = rec["reference"]
    assert res.cls == stored["cls"]
    if stored.get("ref_f64_hex"):
        assert res.ref_f64 is not None and float(res.ref_f64).hex() == stored["ref_f64_hex"]
        assert float(res.sigma_int_lo).hex() == stored["sigma_int_lo_hex"]
        assert float(res.sigma_int_hi).hex() == stored["sigma_int_hi_hex"]
        for key in ("h_quote", "h_forward", "h_discount"):
            assert float(getattr(res, key)).hex() == stored[f"{key}_hex"], key


def test_reference_brackets_contain_point_and_lie_in_interval() -> None:
    doc = _load("fixtures.json")
    for rec in doc["fixtures"]:
        r = rec["reference"]
        if not r.get("ref_f64_hex"):
            continue
        lo, mid, hi = (float.fromhex(r[k]) for k in ("ref_lo_hex", "ref_f64_hex", "ref_hi_hex"))
        assert lo <= mid <= hi, rec["id"]
        assert (
            float.fromhex(r["sigma_int_lo_hex"]) <= mid <= float.fromhex(r["sigma_int_hi_hex"])
        ), rec["id"]


def test_canonical_grid_and_heldout_are_complete() -> None:
    def rows(name: str) -> list[str]:
        with gzip.open(FIXTURES / name, "rb") as fh:
            return io.TextIOWrapper(fh, encoding="utf-8").read().splitlines()

    sweep = rows("sweep81_reference.csv.gz")
    heldout = rows("heldout_chains.csv.gz")
    assert len(sweep) == 1 + 2 * 81 * 81
    assert len(heldout) == 1 + 2000
    assert "unresolved" not in "".join(sweep) and "reference_exception" not in "".join(sweep)


def test_rounding_interval_is_one_ulp_wide_and_follows_half_even() -> None:
    import numpy as np

    for q in (0.1, 9.516258202222488, 5e-324, 2.0):
        with mpmath.workdps(60):
            lo, hi, closed = ref.rounding_interval(q, "float64")
            below = ref.exact(float(np.nextafter(q, -np.inf)))
            above = ref.exact(float(np.nextafter(q, np.inf)))
            assert lo == (below + ref.exact(q)) / 2
            assert hi == (ref.exact(q) + above) / 2
        assert closed is (int(np.float64(q).view(np.uint64)) % 2 == 0)
