"""Exact parity between instrument adapters and their underlying kernels.

The adapters are a naming layer over the functional API. Bit-for-bit equality
against direct kernel calls verifies that they do not introduce a separate
mathematical implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fast_vollib import backends
from fast_vollib.api import get_all_greeks
from fast_vollib.config import get_backend
from fast_vollib.implied_volatility import fast_implied_volatility, fast_implied_volatility_black
from fast_vollib.instruments import (
    EuropeanOption,
    EuropeanOptionBatch,
    Forward,
    InstrumentValidationError,
    IVSolver,
    MissingMarketInputError,
    PricingModel,
    UnsupportedInstrumentError,
    UnsupportedModelError,
    UnsupportedSolverError,
    VanillaMarketInputs,
    greeks_instrument,
    implied_volatility_instrument,
    price_instrument,
)
from fast_vollib.models import fast_black, fast_black_scholes, fast_black_scholes_merton
from fast_vollib.utils.broadcast import maybe_format_data_and_broadcast, preprocess_flags

MODELS = ["black", "black_scholes", "black_scholes_merton"]
SOLVERS = ["halley", "jackel"]

FLAGS = ["c", "p", "c", "p"]
STRIKES = np.array([90.0, 100.0, 100.0, 110.0])
MATURITIES = np.array([0.25, 0.5, 1.0, 2.0])
UNDERLYING = np.array([100.0, 100.0, 105.0, 95.0])
RATE = np.array([0.02, 0.02, -0.005, 0.04])
VOLATILITY = np.array([0.20, 0.25, 0.18, 0.35])
DIVIDEND = np.array([0.01, 0.0, 0.03, 0.015])


def available_backends() -> list[str]:
    names = ["numpy"]
    for name in ("torch", "jax", "numba"):
        try:
            module = backends.get_module(name)
        except Exception:  # pragma: no cover - optional backend absent
            continue
        if getattr(module, "is_available", lambda: True)():
            names.append(name)
    return names


BACKENDS = available_backends()


def to_host(value: Any) -> np.ndarray:
    """NumPy view of a possibly device-resident native array."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def batch(notional: Any = 1.0) -> EuropeanOptionBatch:
    return EuropeanOptionBatch.from_arrays(
        option_type=FLAGS,
        strike=STRIKES,
        maturity=MATURITIES,
        underlier="ACME",
        notional=notional,
    )


def scalar(index: int = 1, notional: float = 1.0) -> EuropeanOption:
    return EuropeanOption(
        underlier="ACME",
        option_type=FLAGS[index],
        strike=float(STRIKES[index]),
        maturity=float(MATURITIES[index]),
        notional=notional,
    )


def pricing_market() -> VanillaMarketInputs:
    return VanillaMarketInputs(
        underlying=UNDERLYING, rate=RATE, volatility=VOLATILITY, dividend_yield=DIVIDEND
    )


def direct_price(model: str, flag: Any, strike: Any, maturity: Any, backend: str) -> np.ndarray:
    common = dict(return_as="numpy", backend=backend)
    if model == "black":
        return fast_black(flag, UNDERLYING, strike, maturity, RATE, VOLATILITY, **common)
    if model == "black_scholes":
        return fast_black_scholes(flag, UNDERLYING, strike, maturity, RATE, VOLATILITY, **common)
    return fast_black_scholes_merton(
        flag, UNDERLYING, strike, maturity, RATE, VOLATILITY, DIVIDEND, **common
    )


def observed_market(model: str, backend: str = "numpy", notional: float = 1.0):
    """Market inputs whose ``price`` column came from the kernel itself."""
    unit = direct_price(model, FLAGS, STRIKES, MATURITIES, backend)
    return VanillaMarketInputs(
        underlying=UNDERLYING,
        rate=RATE,
        price=unit * notional,
        dividend_yield=DIVIDEND if model == "black_scholes_merton" else None,
    )


# --- price parity -------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_batch_price_equals_the_direct_kernel_call(model: str, backend: str) -> None:
    np.testing.assert_array_equal(
        price_instrument(batch(), pricing_market(), model=model, backend=backend),
        direct_price(model, FLAGS, STRIKES, MATURITIES, backend),
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_scalar_price_equals_the_direct_kernel_call(model: str, backend: str) -> None:
    option = scalar()
    market = VanillaMarketInputs(underlying=100.0, rate=0.02, volatility=0.25, dividend_yield=0.01)
    common = dict(return_as="numpy", backend=backend)
    if model == "black":
        expected = fast_black(
            option.flag, 100.0, option.strike, option.maturity, 0.02, 0.25, **common
        )
    elif model == "black_scholes":
        expected = fast_black_scholes(
            option.flag, 100.0, option.strike, option.maturity, 0.02, 0.25, **common
        )
    else:
        expected = fast_black_scholes_merton(
            option.flag, 100.0, option.strike, option.maturity, 0.02, 0.25, 0.01, **common
        )
    np.testing.assert_array_equal(
        price_instrument(option, market, model=model, backend=backend), expected
    )


@pytest.mark.parametrize("model", MODELS)
def test_notional_scales_the_price(model: str) -> None:
    unit = price_instrument(batch(), pricing_market(), model=model)
    scaled = price_instrument(batch(notional=-250.0), pricing_market(), model=model)
    np.testing.assert_allclose(scaled, -250.0 * unit, rtol=0, atol=0)


@pytest.mark.parametrize("return_as", ["numpy", "dataframe", "series"])
def test_return_as_matches_the_kernel_formatting(return_as: str) -> None:
    got = price_instrument(batch(), pricing_market(), model="black", return_as=return_as)
    expected = fast_black(
        FLAGS, UNDERLYING, STRIKES, MATURITIES, RATE, VOLATILITY, return_as=return_as
    )
    assert type(got) is type(expected)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b in {"torch", "jax"}])
def test_return_native_matches_the_kernel(backend: str) -> None:
    got = price_instrument(
        batch(), pricing_market(), model="black_scholes", backend=backend, return_native=True
    )
    expected = fast_black_scholes(
        FLAGS,
        UNDERLYING,
        STRIKES,
        MATURITIES,
        RATE,
        VOLATILITY,
        return_as="numpy",
        backend=backend,
        return_native=True,
    )
    assert type(got) is type(expected)
    np.testing.assert_array_equal(to_host(got), to_host(expected))


def test_one_kernel_call_per_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch is one fused call, never a Python loop over contracts."""
    backend_name = get_backend("auto")
    module = backends.get_module(backend_name)
    calls = {"n": 0}
    original = module.price_black_scholes

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "price_black_scholes", counting)
    big = EuropeanOptionBatch.from_arrays(
        option_type=np.array(["c"] * 5_000),
        strike=np.linspace(80.0, 120.0, 5_000),
        maturity=0.5,
        underlier="ACME",
    )
    market = VanillaMarketInputs(underlying=100.0, rate=0.02, volatility=0.2)
    result = price_instrument(big, market, model="black_scholes")
    assert len(result) == 5_000
    assert calls["n"] == 1


# --- Greeks parity ------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_greeks_equal_the_direct_kernel_call(model: str, backend: str) -> None:
    got = greeks_instrument(batch(), pricing_market(), model=model, backend=backend)
    expected = get_all_greeks(
        FLAGS,
        UNDERLYING,
        STRIKES,
        MATURITIES,
        RATE,
        VOLATILITY,
        DIVIDEND if model == "black_scholes_merton" else None,
        model=model,
        return_as="numpy",
        backend=backend,
    )
    assert set(got) == set(expected)
    for name in expected:
        np.testing.assert_array_equal(got[name], expected[name], err_msg=name)


@pytest.mark.parametrize("model", MODELS)
def test_notional_scales_every_greek(model: str) -> None:
    unit = greeks_instrument(batch(), pricing_market(), model=model)
    scaled = greeks_instrument(batch(notional=100.0), pricing_market(), model=model)
    for name, value in unit.items():
        np.testing.assert_allclose(scaled[name], 100.0 * value, rtol=0, atol=0)


def test_greeks_return_as_dataframe_matches_the_kernel() -> None:
    got = greeks_instrument(batch(), pricing_market(), model="black", return_as="dataframe")
    expected = get_all_greeks(
        FLAGS,
        UNDERLYING,
        STRIKES,
        MATURITIES,
        RATE,
        VOLATILITY,
        model="black",
        return_as="dataframe",
    )
    np.testing.assert_array_equal(got.to_numpy(), expected.to_numpy())


# --- implied volatility: round trips and solver parity ------------------------


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("solver", SOLVERS)
def test_implied_volatility_round_trips_to_the_input_volatility(model: str, solver: str) -> None:
    market = observed_market(model)
    recovered = implied_volatility_instrument(
        batch(), market, model=model, solver=solver, backend="numpy"
    )
    np.testing.assert_allclose(recovered, VOLATILITY, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("model", MODELS)
def test_halley_equals_the_direct_functional_call(model: str) -> None:
    market = observed_market(model)
    got = implied_volatility_instrument(
        batch(), market, model=model, solver="halley", backend="numpy"
    )
    if model == "black":
        expected = fast_implied_volatility_black(
            market.price,
            UNDERLYING,
            STRIKES,
            RATE,
            MATURITIES,
            FLAGS,
            return_as="numpy",
            backend="numpy",
        )
    else:
        expected = fast_implied_volatility(
            market.price,
            UNDERLYING,
            STRIKES,
            MATURITIES,
            RATE,
            FLAGS,
            q=DIVIDEND if model == "black_scholes_merton" else None,
            model=model,
            return_as="numpy",
            backend="numpy",
        )
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", [b for b in BACKENDS if b in {"numpy", "torch", "jax"}])
def test_jackel_equals_the_full_model_wrapper(model: str, backend: str) -> None:
    """Compared against the same backend's wrapper -- never across backends."""
    import importlib

    market = observed_market(model)
    got = implied_volatility_instrument(
        batch(), market, model=model, solver="jackel", backend=backend
    )

    flag = preprocess_flags(FLAGS)
    if model == "black_scholes_merton":
        price, spot, strike, maturity, rate, yields, flag = maybe_format_data_and_broadcast(
            market.price, UNDERLYING, STRIKES, MATURITIES, RATE, DIVIDEND, flag
        )
    else:
        price, spot, strike, maturity, rate, flag = maybe_format_data_and_broadcast(
            market.price, UNDERLYING, STRIKES, MATURITIES, RATE, flag
        )
        yields = None
    module = importlib.import_module(f"fast_vollib.jackel.{get_backend(backend)}_backend")
    expected = module.implied_volatility(model, price, spot, strike, maturity, rate, flag, q=yields)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("model", MODELS)
def test_the_two_solvers_agree_to_solver_tolerance(model: str) -> None:
    market = observed_market(model)
    common = dict(model=model, backend="numpy")
    halley = implied_volatility_instrument(batch(), market, solver="halley", **common)
    jackel = implied_volatility_instrument(batch(), market, solver="jackel", **common)
    np.testing.assert_allclose(halley, jackel, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("solver", SOLVERS)
def test_observed_prices_are_divided_by_notional_before_inversion(solver: str) -> None:
    """A quote is the price of the position; the kernels invert a unit contract."""
    market = observed_market("black_scholes", notional=250.0)
    recovered = implied_volatility_instrument(
        batch(notional=250.0), market, model="black_scholes", solver=solver, backend="numpy"
    )
    np.testing.assert_allclose(recovered, VOLATILITY, rtol=1e-6, atol=1e-8)


def test_ignoring_the_notional_would_give_a_different_answer() -> None:
    """Guards the previous test against passing for the wrong reason."""
    market = observed_market("black_scholes", notional=250.0)
    unscaled = implied_volatility_instrument(
        batch(notional=1.0), market, model="black_scholes", solver="jackel", backend="numpy"
    )
    assert not np.allclose(unscaled, VOLATILITY, rtol=1e-3)


# --- the Jäckel routing contract ----------------------------------------------


def test_the_adapter_never_reaches_the_raw_undiscounted_solver() -> None:
    """``jackel_iv_black`` takes an undiscounted Black-76 price and a forward.

    Market prices are discounted, so feeding one to the raw solver silently
    produces a wrong volatility. The adapter goes through the full-model
    wrappers instead; here we watch what the raw solver actually receives.
    """
    from fast_vollib.jackel import numpy_backend

    captured: dict[str, np.ndarray] = {}
    original = numpy_backend._jackel_iv_black

    def spy(price: np.ndarray, forward: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        captured["price"] = np.array(price, copy=True)
        captured["forward"] = np.array(forward, copy=True)
        return original(price, forward, *args, **kwargs)

    numpy_backend._jackel_iv_black = spy
    try:
        market = observed_market("black_scholes")
        implied_volatility_instrument(
            batch(), market, model="black_scholes", solver="jackel", backend="numpy"
        )
    finally:
        numpy_backend._jackel_iv_black = original

    discount = np.exp(RATE * MATURITIES)
    np.testing.assert_allclose(captured["price"], np.asarray(market.price) * discount)
    np.testing.assert_allclose(captured["forward"], UNDERLYING * discount)
    # The discounted price itself was never what the raw solver saw.
    assert not np.allclose(captured["price"], np.asarray(market.price))


def test_the_pricing_adapter_does_not_reference_the_raw_solver() -> None:
    """Checked structurally, not textually: the docstrings explain the rule."""
    import ast
    from pathlib import Path

    import fast_vollib.instruments.pricing as pricing

    tree = ast.parse(Path(pricing.__file__).read_text(encoding="utf-8"))
    referenced = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, ast.alias):
            referenced.add(node.name.rsplit(".", 1)[-1])
    assert "jackel_iv_black" not in referenced
    assert "jackel_iv_normalized" not in referenced


# --- the differentiable route -------------------------------------------------


def torch_market(model: str):
    torch = pytest.importorskip("torch", reason="torch not installed")
    unit = direct_price(model, FLAGS, STRIKES, MATURITIES, "numpy")
    return torch, VanillaMarketInputs(
        underlying=torch.tensor(UNDERLYING, dtype=torch.float64, requires_grad=True),
        rate=torch.tensor(RATE, dtype=torch.float64),
        price=torch.tensor(unit, dtype=torch.float64, requires_grad=True),
        dividend_yield=(
            torch.tensor(DIVIDEND, dtype=torch.float64) if model == "black_scholes_merton" else None
        ),
    )


@pytest.mark.parametrize("model", MODELS)
def test_native_torch_jackel_matches_the_autograd_wrapper(model: str) -> None:
    torch, market = torch_market(model)
    from fast_vollib.jackel import implied_volatility_autograd

    got = implied_volatility_instrument(
        batch(), market, model=model, solver="jackel", return_native=True
    )
    expected = implied_volatility_autograd(
        market.price,
        market.underlying,
        STRIKES,
        MATURITIES,
        market.rate,
        FLAGS,
        market.dividend_yield,
        model=model,
    )
    assert torch.is_tensor(got)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("model", MODELS)
def test_native_torch_gradients_match_the_wrapper(model: str) -> None:
    torch, market = torch_market(model)
    from fast_vollib.jackel import implied_volatility_autograd

    got = implied_volatility_instrument(
        batch(), market, model=model, solver="jackel", return_native=True
    )
    assert got.requires_grad
    got.sum().backward()
    adapter_grads = (market.price.grad.clone(), market.underlying.grad.clone())

    _torch, fresh = torch_market(model)
    expected = implied_volatility_autograd(
        fresh.price,
        fresh.underlying,
        STRIKES,
        MATURITIES,
        fresh.rate,
        FLAGS,
        fresh.dividend_yield,
        model=model,
    )
    expected.sum().backward()
    torch.testing.assert_close(adapter_grads[0], fresh.price.grad, rtol=0, atol=0)
    torch.testing.assert_close(adapter_grads[1], fresh.underlying.grad, rtol=0, atol=0)


def test_native_gradient_with_a_notional_scales_the_price_sensitivity() -> None:
    """dsigma/d(quote) picks up the 1/notional from the pre-inversion divide."""
    torch, market = torch_market("black_scholes")
    scaled_market = VanillaMarketInputs(
        underlying=market.underlying,
        rate=market.rate,
        price=(market.price * 250.0).detach().requires_grad_(True),
    )
    iv = implied_volatility_instrument(
        batch(notional=250.0),
        scaled_market,
        model="black_scholes",
        solver="jackel",
        return_native=True,
    )
    iv.sum().backward()

    _torch, unit_market = torch_market("black_scholes")
    unit_iv = implied_volatility_instrument(
        batch(), unit_market, model="black_scholes", solver="jackel", return_native=True
    )
    unit_iv.sum().backward()

    torch.testing.assert_close(
        scaled_market.price.grad, unit_market.price.grad / 250.0, rtol=1e-10, atol=0
    )


def test_native_jax_jackel_matches_the_autograd_wrapper() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jnp = jax.numpy
    from fast_vollib.jackel import implied_volatility_autograd_jax

    unit = direct_price("black_scholes", FLAGS, STRIKES, MATURITIES, "numpy")
    market = VanillaMarketInputs(
        underlying=jnp.asarray(UNDERLYING), rate=jnp.asarray(RATE), price=jnp.asarray(unit)
    )
    got = implied_volatility_instrument(
        batch(), market, model="black_scholes", solver="jackel", return_native=True
    )
    expected = implied_volatility_autograd_jax(
        market.price,
        market.underlying,
        STRIKES,
        MATURITIES,
        market.rate,
        FLAGS,
        None,
        model="black_scholes",
    )
    assert isinstance(got, jax.Array)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_native_jax_gradients_match_the_wrapper() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jnp = jax.numpy
    from fast_vollib.jackel import implied_volatility_autograd_jax

    unit = jnp.asarray(direct_price("black_scholes", FLAGS, STRIKES, MATURITIES, "numpy"))

    def through_adapter(price: Any) -> Any:
        market = VanillaMarketInputs(
            underlying=jnp.asarray(UNDERLYING), rate=jnp.asarray(RATE), price=price
        )
        return implied_volatility_instrument(
            batch(), market, model="black_scholes", solver="jackel", return_native=True
        ).sum()

    def through_wrapper(price: Any) -> Any:
        return implied_volatility_autograd_jax(
            price,
            jnp.asarray(UNDERLYING),
            STRIKES,
            MATURITIES,
            jnp.asarray(RATE),
            FLAGS,
            None,
            model="black_scholes",
        ).sum()

    np.testing.assert_array_equal(
        np.asarray(jax.grad(through_adapter)(unit)),
        np.asarray(jax.grad(through_wrapper)(unit)),
    )


def test_host_output_terminates_the_gradient_path() -> None:
    """Documented boundary: formatting a result ends differentiability."""
    torch, market = torch_market("black_scholes")
    host = implied_volatility_instrument(
        batch(), market, model="black_scholes", solver="jackel", return_native=False
    )
    assert isinstance(host, np.ndarray)
    assert market.price.grad is None


def test_invalid_domain_edge_behaviour_is_preserved() -> None:
    """Below-intrinsic quotes stay NaN; the adapter does not reshape the contract."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    from fast_vollib.jackel import implied_volatility_autograd

    below_intrinsic = np.array([1e-6, 1e-6, 1e-6, 1e-6])
    market = VanillaMarketInputs(
        underlying=torch.tensor(UNDERLYING, dtype=torch.float64),
        rate=torch.tensor(RATE, dtype=torch.float64),
        price=torch.tensor(below_intrinsic, dtype=torch.float64, requires_grad=True),
    )
    got = implied_volatility_instrument(
        batch(), market, model="black_scholes", solver="jackel", return_native=True
    )
    expected = implied_volatility_autograd(
        market.price,
        market.underlying,
        STRIKES,
        MATURITIES,
        market.rate,
        FLAGS,
        None,
        model="black_scholes",
    )
    torch.testing.assert_close(got, expected, equal_nan=True, rtol=0, atol=0)
    assert torch.isnan(got).any()


# --- fail-closed --------------------------------------------------------------


def test_unavailable_native_route_raises_instead_of_substituting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch", reason="torch not installed")
    from fast_vollib import jackel

    monkeypatch.setattr(jackel, "implied_volatility_autograd", None)
    _torch, market = torch_market("black_scholes")
    with pytest.raises(UnsupportedSolverError) as excinfo:
        implied_volatility_instrument(
            batch(), market, model="black_scholes", solver="jackel", return_native=True
        )
    message = str(excinfo.value)
    assert "no gradients" in message
    assert "torch" in message


def test_conflicting_native_namespace_and_backend_raises() -> None:
    pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip("jax", reason="jax not installed")
    _torch, market = torch_market("black_scholes")
    with pytest.raises(UnsupportedSolverError, match="cannot both be honoured"):
        implied_volatility_instrument(
            batch(),
            market,
            model="black_scholes",
            solver="jackel",
            backend="jax",
            return_native=True,
        )


def test_jackel_with_an_unsupported_backend_raises() -> None:
    market = observed_market("black_scholes")
    with pytest.raises(UnsupportedSolverError) as excinfo:
        implied_volatility_instrument(
            batch(), market, model="black_scholes", solver="jackel", backend="numba"
        )
    message = str(excinfo.value)
    assert "numba" in message
    assert "halley" in message


def test_halley_still_serves_the_backend_jackel_cannot() -> None:
    """The refusal names a real alternative, not a dead end."""
    if "numba" not in BACKENDS:
        pytest.skip("numba backend not installed")
    market = observed_market("black_scholes")
    result = implied_volatility_instrument(
        batch(), market, model="black_scholes", solver="halley", backend="numba"
    )
    np.testing.assert_allclose(result, VOLATILITY, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "call",
    [
        lambda m: price_instrument(batch(), m, model="black_scholes_merton"),
        lambda m: greeks_instrument(batch(), m, model="black_scholes_merton"),
        lambda m: implied_volatility_instrument(batch(), m, model="black_scholes_merton"),
    ],
    ids=["price", "greeks", "iv"],
)
def test_missing_dividend_yield_names_the_field(call: Any) -> None:
    market = VanillaMarketInputs(
        underlying=UNDERLYING,
        rate=RATE,
        volatility=VOLATILITY,
        price=direct_price("black_scholes_merton", FLAGS, STRIKES, MATURITIES, "numpy"),
    )
    with pytest.raises(MissingMarketInputError) as excinfo:
        call(market)
    message = str(excinfo.value)
    assert "dividend_yield" in message
    assert "black_scholes_merton" in message


def test_missing_volatility_names_the_field() -> None:
    market = VanillaMarketInputs(underlying=UNDERLYING, rate=RATE)
    with pytest.raises(MissingMarketInputError, match="volatility"):
        price_instrument(batch(), market, model="black")


def test_missing_observed_price_names_the_field() -> None:
    market = VanillaMarketInputs(underlying=UNDERLYING, rate=RATE)
    with pytest.raises(MissingMarketInputError, match="price"):
        implied_volatility_instrument(batch(), market, model="black")


@pytest.mark.parametrize(
    "call",
    [price_instrument, greeks_instrument, implied_volatility_instrument],
    ids=["price", "greeks", "iv"],
)
def test_unknown_model_is_refused_and_lists_the_real_ones(call: Any) -> None:
    with pytest.raises(UnsupportedModelError) as excinfo:
        call(batch(), pricing_market(), model="heston")
    message = str(excinfo.value)
    assert "heston" in message
    assert "black_scholes_merton" in message
    assert "never inferred" in message


def test_unknown_solver_is_refused() -> None:
    with pytest.raises(UnsupportedSolverError, match="brent"):
        implied_volatility_instrument(
            batch(), observed_market("black"), model="black", solver="brent"
        )


@pytest.mark.parametrize(
    "call",
    [price_instrument, greeks_instrument, implied_volatility_instrument],
    ids=["price", "greeks", "iv"],
)
def test_a_forward_is_recognized_but_not_priced_as_an_option(call: Any) -> None:
    forward = Forward(underlier="ACME", delivery_price=100.0, maturity=1.0)
    with pytest.raises(UnsupportedInstrumentError) as excinfo:
        call(forward, pricing_market(), model="black")
    message = str(excinfo.value)
    assert "recognized" in message
    assert "no analytic pricing kernel" in message
    assert "capabilities" in message


@pytest.mark.parametrize(
    "call",
    [price_instrument, greeks_instrument, implied_volatility_instrument],
    ids=["price", "greeks", "iv"],
)
def test_a_foreign_object_is_refused(call: Any) -> None:
    with pytest.raises(UnsupportedInstrumentError, match="not an instrument type known"):
        call(object(), pricing_market(), model="black")


def test_model_is_a_required_keyword() -> None:
    with pytest.raises(TypeError):
        price_instrument(batch(), pricing_market())  # type: ignore[call-arg]


def test_the_asset_class_never_selects_a_model() -> None:
    """Both models run on the same equity option; neither is chosen for you."""
    from fast_vollib.instruments import Asset, AssetClass

    equity = Asset(identifier="ACME", asset_class=AssetClass.EQUITY)
    option = EuropeanOption(underlier=equity.ref(), option_type="c", strike=100.0, maturity=1.0)
    market = VanillaMarketInputs(underlying=100.0, rate=0.05, volatility=0.2, dividend_yield=0.03)
    bs = price_instrument(option, market, model="black_scholes")
    bsm = price_instrument(option, market, model="black_scholes_merton")
    assert not np.allclose(bs, bsm)


def test_capability_set_agrees_with_what_the_adapters_do() -> None:
    from fast_vollib.instruments import capabilities

    caps = capabilities(EuropeanOption)
    market = observed_market("black")
    for model in PricingModel:
        assert caps.supports_price(model)
        for solver in IVSolver:
            assert solver in caps.solvers_for(model)
    for solver in IVSolver:
        implied_volatility_instrument(
            batch(), market, model="black", solver=solver, backend="numpy"
        )


def test_batch_validation_still_applies_through_the_adapter() -> None:
    with pytest.raises(InstrumentValidationError, match="strike"):
        price_instrument(
            EuropeanOptionBatch.from_arrays(
                option_type="c", strike=[100.0, -1.0], maturity=1.0, underlier="ACME"
            ),
            pricing_market(),
            model="black",
        )
