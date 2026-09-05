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

from fast_vollib._simulation_errors import UnsupportedProcessError  # noqa: E402
from fast_vollib.processes import CIR_SCHEMES, CIRShortRate  # noqa: E402

FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "processes" / "cir_paths.json"

#: One set comfortably above one degree of freedom and one below, because the
#: exact transition uses a different construction on each side and the two draw
#: different laws in a different order.  ``d = 4 kappa theta / volatility**2``.
PARAMETERS: dict[str, dict[str, float]] = {
    "feller": {"kappa": 0.3, "theta": 0.04, "volatility": 0.1},
    "few_degrees": {"kappa": 0.3, "theta": 0.04, "volatility": 0.25},
}

INITIAL_STATE = {"short_rate": 0.05}
TIME_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
N_PATHS = 8
SEED = 20260904


def degrees_of_freedom(parameters: dict[str, float]) -> float:
    return 4.0 * parameters["kappa"] * parameters["theta"] / parameters["volatility"] ** 2


def encode_array(array: np.ndarray) -> list[str]:
    """A C-ordered flat list of hex literals; the shape is recorded separately."""
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    return [float(x).hex() for x in flat]


def build_cases() -> list[dict[str, Any]]:
    grid = np.array(TIME_GRID, dtype=np.float64)
    cases: list[dict[str, Any]] = []
    for name, parameters in PARAMETERS.items():
        process = CIRShortRate(**parameters)
        for scheme in CIR_SCHEMES:
            for antithetic in (False, True):
                case: dict[str, Any] = {
                    "parameters": name,
                    "scheme": scheme,
                    "antithetic": antithetic,
                    "degrees_of_freedom": degrees_of_freedom(parameters),
                }
                try:
                    paths = process.sample(
                        initial_state=INITIAL_STATE,
                        time_grid=grid,
                        n_paths=N_PATHS,
                        rng=SEED,
                        antithetic=antithetic,
                        scheme=scheme,
                    )
                except UnsupportedProcessError:
                    # Recorded rather than skipped: a refusal is behaviour, and
                    # removing one should show up as a diff here too.
                    case["refused"] = True
                    cases.append(case)
                    continue
                array = np.asarray(paths)
                assert array.dtype == np.float64, array.dtype
                case["refused"] = False
                case["shape"] = list(array.shape)
                case["paths_hex"] = encode_array(array)
                cases.append(case)
    return cases


def _finalize(body: dict[str, Any]) -> str:
    """Serialize, hash the serialization, then re-serialize with the digest."""
    text = json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False)
    body["content_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False) + "\n"


def render() -> str:
    return _finalize(
        {
            "description": "Library-generated numerical regression reference; not an independent oracle.",
            "generator": "scripts/generate_cir_path_fixtures.py",
            "encoding": "float64 as float.hex(); portable checks use scripts/reference_fixtures.py.",
            "namespace": "numpy",
            "dtype": "float64",
            "initial_state": INITIAL_STATE,
            "time_grid": list(TIME_GRID),
            "n_paths": N_PATHS,
            "seed": SEED,
            "parameter_sets": PARAMETERS,
            "axes": "paths[n_paths, n_times, state], state = ('short_rate',)",
            "cases": build_cases(),
        }
    )


def _write(path: Path, text: str) -> None:
    write_or_check(path, text)


def main() -> int:
    _write(FIXTURE, render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
