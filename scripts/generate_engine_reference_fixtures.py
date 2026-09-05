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

from reference_fixtures import write_or_check

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fast_vollib.instruments import (  # noqa: E402
    AsianOption,
    BarrierOption,
    BinaryOption,
    EuropeanOption,
    Forward,
    LookbackOption,
    VanillaMarketInputs,
    VarianceSwap,
)
from fast_vollib.processes import GBM  # noqa: E402
from fast_vollib.simulation import MonteCarloEngine  # noqa: E402

FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "simulation" / "engine_prices.json"

#: Two processes. The first is the ordinary risk-neutral case; the second is
#: built directly, with a drift that is deliberately not ``MARKET_RATE`` -- the
#: engine is documented never to rewrite a drift, and a fixture in which drift
#: and rate coincide could not tell whether it had.
PROCESSES: dict[str, dict[str, float]] = {
    "risk_neutral": {"drift": 0.03, "volatility": 0.2},
    "physical_drift": {"drift": 0.11, "volatility": 0.45},
}

MARKET_UNDERLYING = 100.0
MARKET_RATE = 0.03

#: Small on purpose, as in the Heston fixtures: a bitwise record gains nothing
#: from more paths, and the estimator is exercised identically at 64 and 64000.
N_PATHS = 64
N_STEPS = 8
SEED = 20260904

UNDERLIER = "ACME"


def instruments() -> dict[str, Any]:
    """One instance of every type the engine supports, plus the branch variants.

    Both payoff requirements are represented: ``EuropeanOption``, ``Forward``
    and ``BinaryOption`` are TERMINAL, the rest are PATH and so reach the
    engine's other dispatch arm.
    """
    return {
        "european_call": EuropeanOption(
            underlier=UNDERLIER, option_type="call", strike=100.0, maturity=1.0
        ),
        "european_put": EuropeanOption(
            underlier=UNDERLIER, option_type="put", strike=110.0, maturity=1.0
        ),
        "forward": Forward(underlier=UNDERLIER, delivery_price=100.0, maturity=1.0),
        "binary_put": BinaryOption(
            underlier=UNDERLIER,
            option_type="put",
            strike=95.0,
            maturity=1.0,
            cash_amount=10.0,
        ),
        "asian_arithmetic_fixed": AsianOption(
            underlier=UNDERLIER,
            option_type="call",
            averaging_method="arithmetic",
            strike_convention="fixed",
            maturity=1.0,
            strike=100.0,
        ),
        "asian_geometric_floating": AsianOption(
            underlier=UNDERLIER,
            option_type="call",
            averaging_method="geometric",
            strike_convention="floating",
            maturity=1.0,
        ),
        "barrier_up_and_out": BarrierOption(
            underlier=UNDERLIER,
            option_type="call",
            strike=100.0,
            barrier=130.0,
            barrier_type="up_and_out",
            maturity=1.0,
        ),
        "barrier_down_and_in": BarrierOption(
            underlier=UNDERLIER,
            option_type="put",
            strike=100.0,
            barrier=85.0,
            barrier_type="down_and_in",
            maturity=1.0,
        ),
        "lookback_floating": LookbackOption(
            underlier=UNDERLIER,
            option_type="call",
            strike_convention="floating",
            maturity=1.0,
        ),
        "lookback_fixed": LookbackOption(
            underlier=UNDERLIER,
            option_type="put",
            strike_convention="fixed",
            maturity=1.0,
            strike=105.0,
        ),
        "variance_swap": VarianceSwap(underlier=UNDERLIER, strike_variance=0.04, maturity=1.0),
    }


def encode(value: Any) -> str:
    """One float64 as a hexadecimal literal that round-trips exactly.

    ``repr`` would also round-trip, but it invites a reader to compare the
    decimal digits by eye and conclude that two nearly equal numbers are the
    same number. The hex form makes the last bit as visible as the first.
    """
    return float(value).hex()


def build_cases() -> list[dict[str, Any]]:
    market = VanillaMarketInputs(underlying=MARKET_UNDERLYING, rate=MARKET_RATE)
    contracts = instruments()
    cases: list[dict[str, Any]] = []
    for process_name, parameters in PROCESSES.items():
        process = GBM(**parameters)
        for antithetic in (False, True):
            engine = MonteCarloEngine(antithetic=antithetic)
            for contract_name, instrument in contracts.items():
                result = engine.price(
                    instrument,
                    market,
                    process=process,
                    n_paths=N_PATHS,
                    n_steps=N_STEPS,
                    rng=SEED,
                )
                assert type(result.price) is float, type(result.price)
                assert type(result.stderr) is float, type(result.stderr)
                cases.append(
                    {
                        "process": process_name,
                        "antithetic": antithetic,
                        "instrument": contract_name,
                        "type": type(instrument).__name__,
                        "price_hex": encode(result.price),
                        "stderr_hex": encode(result.stderr),
                        "n_paths": result.n_paths,
                        "effective_samples": result.effective_samples,
                    }
                )
    return cases


def _finalize(body: dict[str, Any]) -> str:
    """Serialize, hash the serialization, then re-serialize with the digest.

    The same two-pass shape the other generators use, so the digest describes
    the content rather than itself.
    """
    text = json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False)
    body["content_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False) + "\n"


def render() -> str:
    return _finalize(
        {
            "description": "Library-generated numerical regression reference; not an independent oracle.",
            "generator": "scripts/generate_engine_reference_fixtures.py",
            "encoding": "float64 as float.hex(); portable checks use scripts/reference_fixtures.py.",
            "namespace": "numpy",
            "dtype": "float64",
            "market": {"underlying": MARKET_UNDERLYING, "rate": MARKET_RATE},
            "n_paths": N_PATHS,
            "n_steps": N_STEPS,
            "seed": SEED,
            "underlier": UNDERLIER,
            "process_parameters": PROCESSES,
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
