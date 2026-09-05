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

from fast_vollib.pricing import bates_price  # noqa: E402
from fast_vollib.processes import (  # noqa: E402
    SCHEMES,
    Bates,
    ConstantVariance,
    HestonVariance,
    LognormalJumps,
    NoJumps,
)

PATHS_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "processes" / "bates_paths.json"
PRICES_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "pricing" / "bates_prices.json"

# --- sampler cases -------------------------------------------------------------

PATH_VARIANCE = {
    "heston": {"kappa": 2.0, "theta": 0.04, "vol_of_vol": 0.3, "rho": -0.7},
    "constant": None,
}
PATH_JUMPS = {
    "lognormal": {"jump_intensity": 1.5, "mean_log_jump": -0.05, "jump_volatility": 0.2},
    "none": None,
}
PATH_DRIFT = 0.03
PATH_INITIAL_STATE = {"spot": 100.0, "variance": 0.04}
PATH_TIME_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
PATH_N_PATHS = 8
PATH_SEED = 20260904

# --- pricer cases --------------------------------------------------------------

PRICE_HESTON = {
    "moderate": {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "vol_of_vol": 0.3, "rho": -0.7},
    "heavy": {"v0": 0.09, "kappa": 0.5, "theta": 0.06, "vol_of_vol": 0.9, "rho": -0.9},
    "fast": {"v0": 0.01, "kappa": 5.0, "theta": 0.02, "vol_of_vol": 0.2, "rho": 0.4},
}
PRICE_JUMPS = {
    "mild": {"jump_intensity": 0.5, "mean_log_jump": -0.05, "jump_volatility": 0.2},
    "crash": {"jump_intensity": 0.3, "mean_log_jump": -0.40, "jump_volatility": 0.05},
    "wild": {"jump_intensity": 5.0, "mean_log_jump": -0.02, "jump_volatility": 0.45},
}
PRICE_FORWARD = 100.0
PRICE_STRIKES = (60.0, 80.0, 100.0, 130.0, 180.0)
PRICE_MATURITIES = (0.02, 0.25, 1.0, 10.0)

#: Not 1.0, so a discount factor dropped on the floor changes the answer.
PRICE_DISCOUNT = 0.97


def encode_array(array: np.ndarray) -> list[str]:
    """A C-ordered flat list of hex literals; the shape is recorded separately."""
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    return [float(x).hex() for x in flat]


def build_path_cases() -> list[dict[str, Any]]:
    grid = np.array(PATH_TIME_GRID, dtype=np.float64)
    cases: list[dict[str, Any]] = []
    for variance_name, variance_params in PATH_VARIANCE.items():
        variance = (
            ConstantVariance() if variance_params is None else HestonVariance(**variance_params)
        )
        for jump_name, jump_params in PATH_JUMPS.items():
            jumps = NoJumps() if jump_params is None else LognormalJumps(**jump_params)
            process = Bates(variance=variance, jumps=jumps, drift=PATH_DRIFT)
            for scheme in SCHEMES:
                for antithetic in (False, True):
                    paths = process.sample(
                        initial_state=PATH_INITIAL_STATE,
                        time_grid=grid,
                        n_paths=PATH_N_PATHS,
                        rng=PATH_SEED,
                        antithetic=antithetic,
                        scheme=scheme,
                    )
                    array = np.asarray(paths)
                    assert array.dtype == np.float64, array.dtype
                    cases.append(
                        {
                            "variance": variance_name,
                            "jumps": jump_name,
                            "scheme": scheme,
                            "antithetic": antithetic,
                            "shape": list(array.shape),
                            "paths_hex": encode_array(array),
                        }
                    )
    return cases


def build_price_cases() -> list[dict[str, Any]]:
    strikes = np.array(PRICE_STRIKES, dtype=np.float64)
    cases: list[dict[str, Any]] = []
    for heston_name, heston in PRICE_HESTON.items():
        for jump_name, jumps in PRICE_JUMPS.items():
            for formulation in ("lewis", "gatheral"):
                for maturity in PRICE_MATURITIES:
                    for is_call in (True, False):
                        priced = bates_price(
                            forward=PRICE_FORWARD,
                            strike=strikes,
                            maturity=maturity,
                            is_call=is_call,
                            discount=PRICE_DISCOUNT,
                            formulation=formulation,
                            **heston,
                            **jumps,
                        )
                        array = np.asarray(priced)
                        assert array.dtype == np.float64, array.dtype
                        cases.append(
                            {
                                "heston": heston_name,
                                "jumps": jump_name,
                                "formulation": formulation,
                                "maturity": maturity,
                                "is_call": is_call,
                                "shape": list(array.shape),
                                "prices_hex": encode_array(array),
                            }
                        )
    return cases


def _finalize(body: dict[str, Any]) -> str:
    """Serialize, hash the serialization, then re-serialize with the digest."""
    text = json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False)
    body["content_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False) + "\n"


def render_paths() -> str:
    return _finalize(
        {
            "description": "Library-generated numerical regression reference; not an independent oracle.",
            "generator": "scripts/generate_bates_reference_fixtures.py",
            "encoding": "float64 as float.hex(); portable checks use scripts/reference_fixtures.py.",
            "namespace": "numpy",
            "dtype": "float64",
            "initial_state": PATH_INITIAL_STATE,
            "time_grid": list(PATH_TIME_GRID),
            "n_paths": PATH_N_PATHS,
            "seed": PATH_SEED,
            "drift": PATH_DRIFT,
            "variance_components": PATH_VARIANCE,
            "jump_components": PATH_JUMPS,
            "axes": "paths[n_paths, n_times, state], state = ('spot', 'variance')",
            "cases": build_path_cases(),
        }
    )


def render_prices() -> str:
    return _finalize(
        {
            "description": "Library-generated numerical regression reference; not an independent oracle.",
            "generator": "scripts/generate_bates_reference_fixtures.py",
            "encoding": "float64 as float.hex(); portable checks use scripts/reference_fixtures.py.",
            "forward": PRICE_FORWARD,
            "strikes": list(PRICE_STRIKES),
            "maturities": list(PRICE_MATURITIES),
            "discount": PRICE_DISCOUNT,
            "heston_parameters": PRICE_HESTON,
            "jump_parameters": PRICE_JUMPS,
            "axes": "prices[strike]",
            "cases": build_price_cases(),
        }
    )


def _write(path: Path, text: str) -> None:
    write_or_check(path, text)


def main() -> int:
    _write(PATHS_FIXTURE, render_paths())
    _write(PRICES_FIXTURE, render_prices())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
