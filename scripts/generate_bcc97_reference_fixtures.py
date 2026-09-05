"""Generate seeded numerical regression fixtures.

These fixtures record library output, not independent mathematical oracles.
Hex encoding retains exact stored values. Cross-platform checks use the bounded
rounding policy in reference_fixtures.py; repeated local output is deterministic.
Use --check to verify without rewriting artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from reference_fixtures import write_or_check

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fast_vollib.pricing import bcc97_price  # noqa: E402
from fast_vollib.processes import (  # noqa: E402
    BCC97,
    CIR_SCHEMES,
    SCHEMES,
    CIRShortRate,
    ConstantShortRate,
    ConstantVariance,
    HestonVariance,
    LognormalJumps,
    NoJumps,
)

PATHS_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "processes" / "bcc97_paths.json"
PRICES_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "pricing" / "bcc97_prices.json"

VARIANCE = {
    "heston": {"kappa": 2.0, "theta": 0.04, "vol_of_vol": 0.3, "rho": -0.7},
    "constant": None,
}
JUMPS = {
    "lognormal": {"jump_intensity": 1.5, "mean_log_jump": -0.05, "jump_volatility": 0.2},
    "none": None,
}
#: 4*kappa*theta/volatility**2 = 4.44, comfortably above one, so the exact
#: transition takes its ``d > 1`` construction and antithetic pairing is allowed.
RATES = {
    "cir": {"kappa": 0.5, "theta": 0.05, "volatility": 0.15},
    "constant": None,
}
DIVIDEND_YIELD = 0.01
INITIAL_STATE = {"spot": 100.0, "variance": 0.04, "short_rate": 0.03}
TIME_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
N_PATHS = 8
SEED = 20260904


def encode_array(array: np.ndarray) -> list[str]:
    """A C-ordered flat list of hex literals; the shape is recorded separately."""
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    return [float(x).hex() for x in flat]


def rate_schemes_for(name: str) -> tuple[str, ...]:
    """A constant rate draws nothing, so only one scheme is worth recording.

    The argument is still *validated* for it -- that is a test, not a fixture --
    but sweeping three schemes that produce identical paths would record the
    same numbers three times and say nothing.
    """
    return CIR_SCHEMES if name == "cir" else ("quadratic_exponential",)


def build_cases() -> list[dict[str, Any]]:
    grid = np.array(TIME_GRID, dtype=np.float64)
    cases: list[dict[str, Any]] = []
    for variance_name, variance_params in VARIANCE.items():
        variance = (
            ConstantVariance() if variance_params is None else HestonVariance(**variance_params)
        )
        for jump_name, jump_params in JUMPS.items():
            jumps = NoJumps() if jump_params is None else LognormalJumps(**jump_params)
            for rate_name, rate_params in RATES.items():
                rates = ConstantShortRate() if rate_params is None else CIRShortRate(**rate_params)
                process = BCC97(
                    variance=variance,
                    jumps=jumps,
                    rates=rates,
                    dividend_yield=DIVIDEND_YIELD,
                )
                for scheme in SCHEMES:
                    for rate_scheme in rate_schemes_for(rate_name):
                        for antithetic in (False, True):
                            paths = np.asarray(
                                process.sample(
                                    initial_state=INITIAL_STATE,
                                    time_grid=grid,
                                    n_paths=N_PATHS,
                                    rng=SEED,
                                    antithetic=antithetic,
                                    scheme=scheme,
                                    rate_scheme=rate_scheme,
                                )
                            )
                            assert paths.dtype == np.float64, paths.dtype
                            cases.append(
                                {
                                    "variance": variance_name,
                                    "jumps": jump_name,
                                    "rates": rate_name,
                                    "scheme": scheme,
                                    "rate_scheme": rate_scheme,
                                    "antithetic": antithetic,
                                    "shape": list(paths.shape),
                                    "paths_hex": encode_array(paths),
                                }
                            )
    return cases


def _finalize(body: dict[str, Any]) -> str:
    """Serialize, hash the serialization, then re-serialize with the digest."""
    text = json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False)
    body["content_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False) + "\n"


# --- pricer cases ---------------------------------------------------------------

PRICE_HESTON = {
    "moderate": {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "vol_of_vol": 0.3, "rho": -0.7},
    "heavy": {"v0": 0.09, "kappa": 0.5, "theta": 0.06, "vol_of_vol": 0.9, "rho": -0.9},
}
PRICE_JUMPS = {
    "none": {},
    "mild": {"jump_intensity": 0.5, "mean_log_jump": -0.05, "jump_volatility": 0.2},
    "wild": {"jump_intensity": 5.0, "mean_log_jump": -0.02, "jump_volatility": 0.45},
}
PRICE_RATES = {
    "calm": {
        "rate_kappa": 0.3,
        "rate_theta": 0.04,
        "rate_volatility": 0.1,
        "initial_rate": 0.03,
    },
    "volatile": {
        "rate_kappa": 0.15,
        "rate_theta": 0.08,
        "rate_volatility": 0.6,
        "initial_rate": 0.05,
    },
    #: Deterministic and flat: the configuration that must stay bitwise Bates.
    "flat": {
        "rate_kappa": 0.3,
        "rate_theta": 0.03,
        "rate_volatility": 0.0,
        "initial_rate": 0.03,
    },
}
PRICE_SPOT = 100.0
PRICE_STRIKES = (60.0, 80.0, 100.0, 130.0, 180.0)
PRICE_MATURITIES = (0.02, 0.25, 1.0, 10.0)

#: Not zero, so a dividend yield dropped on the floor changes the answer.
PRICE_DIVIDEND_YIELD = 0.015


def build_price_cases() -> list[dict[str, Any]]:
    strikes = np.array(PRICE_STRIKES, dtype=np.float64)
    cases: list[dict[str, Any]] = []
    for heston_name, heston in PRICE_HESTON.items():
        for jump_name, jumps in PRICE_JUMPS.items():
            for rate_name, rates in PRICE_RATES.items():
                for formulation in ("lewis", "gatheral"):
                    for maturity in PRICE_MATURITIES:
                        for is_call in (True, False):
                            priced = np.asarray(
                                bcc97_price(
                                    spot=PRICE_SPOT,
                                    strike=strikes,
                                    maturity=maturity,
                                    is_call=is_call,
                                    dividend_yield=PRICE_DIVIDEND_YIELD,
                                    formulation=formulation,
                                    **heston,
                                    **jumps,
                                    **rates,
                                )
                            )
                            assert priced.dtype == np.float64, priced.dtype
                            cases.append(
                                {
                                    "heston": heston_name,
                                    "jumps": jump_name,
                                    "rates": rate_name,
                                    "formulation": formulation,
                                    "maturity": maturity,
                                    "is_call": is_call,
                                    "shape": list(priced.shape),
                                    "prices_hex": encode_array(priced),
                                }
                            )
    return cases


def render_prices() -> str:
    return _finalize(
        {
            "description": "Library-generated numerical regression reference; not an independent oracle.",
            "generator": "scripts/generate_bcc97_reference_fixtures.py",
            "encoding": "float64 as float.hex(); portable checks use scripts/reference_fixtures.py.",
            "spot": PRICE_SPOT,
            "strikes": list(PRICE_STRIKES),
            "maturities": list(PRICE_MATURITIES),
            "dividend_yield": PRICE_DIVIDEND_YIELD,
            "heston_parameters": PRICE_HESTON,
            "jump_parameters": PRICE_JUMPS,
            "rate_parameters": PRICE_RATES,
            "axes": "prices[strike]",
            "cases": build_price_cases(),
        }
    )


def render() -> str:
    return _finalize(
        {
            "description": "Library-generated numerical regression reference; not an independent oracle.",
            "generator": "scripts/generate_bcc97_reference_fixtures.py",
            "encoding": "float64 as float.hex(); portable checks use scripts/reference_fixtures.py.",
            "namespace": "numpy",
            "dtype": "float64",
            "initial_state": INITIAL_STATE,
            "time_grid": list(TIME_GRID),
            "n_paths": N_PATHS,
            "seed": SEED,
            "dividend_yield": DIVIDEND_YIELD,
            "variance_components": VARIANCE,
            "jump_components": JUMPS,
            "rate_components": RATES,
            "axes": ("paths[n_paths, n_times, state], state = ('spot', 'variance', 'short_rate')"),
            "cases": build_cases(),
        }
    )


def _write(path: Path, text: str) -> None:
    write_or_check(path, text)


def main() -> int:
    _write(PATHS_FIXTURE, render())
    _write(PRICES_FIXTURE, render_prices())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
