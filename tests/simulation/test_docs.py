"""The simulation page must document what the code actually does."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = REPO_ROOT / "docs" / "simulation.md"
DOCUMENT = DOC_PATH.read_text(encoding="utf-8")


def test_the_page_is_wired_into_the_navigation() -> None:
    nav = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "simulation.md" in nav


@pytest.mark.parametrize(
    "name",
    [
        "simulate",
        "Scenario",
        "MonteCarloEngine",
        "MCResult",
        "GBM",
        "StochasticProcess",
        "SimulationValidationError",
        "ScenarioMismatchError",
        "UnsupportedProcessError",
    ],
)
def test_every_documented_name_exists(name: str) -> None:
    import fast_vollib.processes as processes
    import fast_vollib.simulation as simulation

    assert name in DOCUMENT
    assert hasattr(simulation, name) or hasattr(processes, name)


def test_the_page_states_the_measure_rule() -> None:
    """The most consequential thing a reader can misunderstand."""
    assert "never used to rewrite a drift" in DOCUMENT
    assert "`market.volatility` is not read" in DOCUMENT


def test_the_page_documents_every_payoff_convention() -> None:
    for convention in (
        "Asian fixings are `S_1 … S_n`",
        "includes both endpoints",
        "Monitoring is discrete",
        "a binary pays nothing",
        "no sample-mean subtraction and no factor of 252",
    ):
        assert convention in DOCUMENT, convention


def test_the_page_separates_tape_retention_from_useful_gradients() -> None:
    assert "Retained, and zero almost everywhere" in DOCUMENT
    assert "tape retention and only that" in DOCUMENT


def test_the_documented_estimator_matches_the_implementation() -> None:
    from fast_vollib.simulation import MonteCarloEngine

    samples = np.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
    ordinary = MonteCarloEngine()._estimate(samples, n_paths=6, return_native=False)
    assert ordinary.price == pytest.approx(float(samples.mean()))
    assert ordinary.stderr == pytest.approx(float(np.std(samples, ddof=1) / np.sqrt(6)))

    paired = MonteCarloEngine(antithetic=True)._estimate(samples, n_paths=6, return_native=False)
    halves = 0.5 * (samples[:3] + samples[3:])
    assert paired.price == pytest.approx(float(halves.mean()))
    assert paired.stderr == pytest.approx(float(np.std(halves, ddof=1) / np.sqrt(3)))
    assert paired.effective_samples == 3
    assert "effective_samples = m" in DOCUMENT


def test_the_documented_conventions_table_matches_the_payoffs() -> None:
    """Every contract named in the conventions table has a path or terminal payoff."""
    from fast_vollib.instruments import instrument_types

    documented = {
        "Binary": "binary_option",
        "Barrier": "barrier_option",
        "Asian": "asian_option",
        "lookback": "lookback_option",
        "Variance swap": "variance_swap",
    }
    registry = instrument_types()
    for label, type_id in documented.items():
        assert label in DOCUMENT, label
        assert registry[type_id].payoff_requirement is not None


def test_the_quick_start_example_runs_and_agrees_with_the_analytic_price() -> None:
    """The first thing a reader copies must work, and must be roughly right."""
    from fast_vollib.instruments import EuropeanOption, VanillaMarketInputs, price_instrument
    from fast_vollib.processes import GBM
    from fast_vollib.simulation import MonteCarloEngine

    option = EuropeanOption(underlier="SPX", option_type="call", strike=5000.0, maturity=0.75)
    market = VanillaMarketInputs(underlying=5100.0, rate=0.04)
    result = MonteCarloEngine().price(
        option,
        market,
        process=GBM.risk_neutral(rate=0.04, volatility=0.18),
        n_paths=50_000,
        n_steps=16,
        rng=0,
    )
    exact = float(
        price_instrument(
            option,
            VanillaMarketInputs(underlying=5100.0, rate=0.04, volatility=0.18),
            model="black_scholes",
        )[0]
    )
    assert abs(result.price - exact) <= 5.0 * result.stderr


def test_the_documented_simulate_example_runs() -> None:
    from fast_vollib.processes import GBM
    from fast_vollib.simulation import simulate

    scenario = simulate(
        "SPX",
        GBM.risk_neutral(rate=0.04, volatility=0.18),
        initial_state=5100.0,
        time_grid=np.linspace(0.0, 0.75, 65),
        n_paths=1_000,
        rng=0,
    )
    assert (scenario.n_paths, scenario.n_steps) == (1_000, 64)
    assert scenario.spot.shape == (1_000, 65)
    assert scenario.terminal("spot").shape == (1_000,)


def test_the_documented_path_dependent_example_runs() -> None:
    from fast_vollib.instruments import AsianOption, VanillaMarketInputs
    from fast_vollib.processes import GBM
    from fast_vollib.simulation import MonteCarloEngine

    asian = AsianOption(
        underlier="SPX",
        option_type="call",
        strike=5000.0,
        averaging_method="arithmetic",
        strike_convention="fixed",
        maturity=0.75,
    )
    result = MonteCarloEngine().price(
        asian,
        VanillaMarketInputs(underlying=5100.0, rate=0.04),
        process=GBM.risk_neutral(rate=0.04, volatility=0.18),
        n_paths=20_000,
        n_steps=32,
        rng=0,
    )
    assert result.price > 0.0 and result.stderr > 0.0


def test_the_rng_table_lists_what_each_backend_accepts() -> None:
    table = DOCUMENT.split("## Randomness", 1)[1].split("**Backend selection.**", 1)[0]
    assert "numpy.random.Generator" in table
    assert "torch.Generator" in table
    assert "jax.random.key" in table
    assert "integer seed is refused" in table


def test_the_scope_section_claims_nothing_that_is_implemented() -> None:
    scope = DOCUMENT.split("## Scope", 1)[1]
    for absent in ("American", "calibration", "control variates", "rebates"):
        assert absent in scope, absent
    # Everything the scope disclaims must genuinely be absent from the package.
    import fast_vollib.simulation as simulation

    assert not any(name.lower().startswith("american") for name in dir(simulation))


def test_no_code_fence_is_left_unclosed() -> None:
    assert DOCUMENT.count("```") % 2 == 0


def test_every_internal_link_target_exists() -> None:
    docs = REPO_ROOT / "docs"
    for target in re.findall(r"\]\((?!https?:)([^)#]+)(?:#[^)]*)?\)", DOCUMENT):
        assert (docs / target).exists(), target
