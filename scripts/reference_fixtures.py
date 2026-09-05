"""Portable comparison and read-only verification of numerical JSON fixtures.

Seeded paths allow 5e-14 relative and 1e-16 absolute rounding error. Fourier
prices allow 5e-10 absolute error (5e-12 of the fixture's reference spot),
covering measured libm/SciPy quadrature differences without approaching the
1e-6 independent quadrature-accuracy tolerance. Metadata remains exact.
Set FV_STRICT_REFERENCE_FIXTURES=1 to require identical stored bits in a
matching numerical environment. Repeated calls in one environment must always
remain identical, independently of this cross-platform allowance.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


def assert_reference_array(actual: Any, expected: Any, *, pricing: bool = False) -> None:
    if os.environ.get("FV_STRICT_REFERENCE_FIXTURES") == "1":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0 if pricing else 5e-14,
            atol=5e-10 if pricing else 1e-16,
            equal_nan=False,
        )


def _verify_digest(body: dict[str, Any]) -> None:
    if "content_sha256" not in body:
        return
    payload = {k: v for k, v in body.items() if k != "content_sha256"}
    rendered = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False)
    assert hashlib.sha256(rendered.encode()).hexdigest() == body["content_sha256"]


def assert_reference_text(actual: str, expected: str) -> None:
    """Compare payloads, validating both hashes before allowing numeric rounding."""
    left, right = json.loads(actual), json.loads(expected)
    _verify_digest(left)
    _verify_digest(right)

    def compare(a: Any, b: Any, key: str = "") -> None:
        if isinstance(a, dict):
            assert isinstance(b, dict) and a.keys() == b.keys()
            for name in a:
                if name != "content_sha256":
                    compare(a[name], b[name], name)
        elif key.endswith("_hex"):
            values_a = a if isinstance(a, list) else [a]
            values_b = b if isinstance(b, list) else [b]
            assert len(values_a) == len(values_b)
            assert_reference_array(
                [float.fromhex(x) for x in values_a],
                [float.fromhex(x) for x in values_b],
                pricing=key == "prices_hex",
            )
        elif isinstance(a, list):
            assert isinstance(b, list) and len(a) == len(b)
            for x, y in zip(a, b):
                compare(x, y, key)
        else:
            assert a == b, (key, a, b)

    compare(left, right)


def write_or_check(path: Path, text: str) -> None:
    """--check never writes; ordinary invocation regenerates the declared file."""
    if "--check" in sys.argv[1:]:
        assert_reference_text(text, path.read_text(encoding="utf-8"))
        print(f"Verified {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        print(f"{path.name} is up to date.")
        return
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path.name}.")
