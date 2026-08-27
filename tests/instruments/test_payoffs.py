"""Payoffs: hand values, broadcasting, and native dtype/device/tape preservation."""

from __future__ import annotations

import numpy as np
import pytest

from fast_vollib.instruments import (
    Asset,
    EuropeanOption,
    Forward,
    Future,
    PayoffRequirement,
    UnsupportedInstrumentError,
    payoff,
    payoff_requirement,
)

CALL = EuropeanOption(underlier="ACME", option_type="call", strike=100.0, maturity=1.0)
PUT = EuropeanOption(underlier="ACME", option_type="put", strike=100.0, maturity=1.0)
FORWARD = Forward(underlier="ACME", delivery_price=100.0, maturity=1.0)
FUTURE = Future(underlier="ACME", contract_price=100.0, maturity=1.0)

TERMINAL = np.array([80.0, 100.0, 125.0])

HAND_VALUES = [
    (CALL, [0.0, 0.0, 25.0]),
    (PUT, [20.0, 0.0, 0.0]),
    (FORWARD, [-20.0, 0.0, 25.0]),
    (FUTURE, [-20.0, 0.0, 25.0]),
]
CONTRACTS = [case[0] for case in HAND_VALUES]
IDS = ["call", "put", "forward", "future"]


# --- values ------------------------------------------------------------------


@pytest.mark.parametrize(("instrument", "expected"), HAND_VALUES, ids=IDS)
def test_hand_computed_values(instrument: object, expected: list[float]) -> None:
    np.testing.assert_array_equal(payoff(instrument, TERMINAL), np.array(expected))


@pytest.mark.parametrize(("instrument", "expected"), HAND_VALUES, ids=IDS)
def test_scalar_input(instrument: object, expected: list[float]) -> None:
    for spot, want in zip(TERMINAL, expected):
        assert float(payoff(instrument, float(spot))) == want


@pytest.mark.parametrize("instrument", CONTRACTS, ids=IDS)
def test_notional_scales_the_cashflow_linearly(instrument: object) -> None:
    import dataclasses

    scaled = dataclasses.replace(instrument, notional=-2.5)
    np.testing.assert_allclose(payoff(scaled, TERMINAL), -2.5 * payoff(instrument, TERMINAL))


def test_option_payoffs_are_non_negative_before_notional() -> None:
    assert np.all(payoff(CALL, TERMINAL) >= 0.0)
    assert np.all(payoff(PUT, TERMINAL) >= 0.0)


def test_put_call_parity_holds_at_maturity() -> None:
    """C(S) - P(S) = S - K, pointwise, with no discounting involved."""
    np.testing.assert_allclose(
        payoff(CALL, TERMINAL) - payoff(PUT, TERMINAL), TERMINAL - CALL.strike
    )


def test_forward_and_future_agree_on_the_terminal_payoff() -> None:
    np.testing.assert_array_equal(payoff(FORWARD, TERMINAL), payoff(FUTURE, TERMINAL))


@pytest.mark.parametrize("instrument", CONTRACTS, ids=IDS)
def test_broadcasting_over_a_grid(instrument: object) -> None:
    grid = TERMINAL.reshape(-1, 1) * np.ones((1, 4))
    result = payoff(instrument, grid)
    assert result.shape == grid.shape
    np.testing.assert_array_equal(result[:, 0], payoff(instrument, TERMINAL))


# --- what a payoff must not do -----------------------------------------------


def test_payoff_does_not_discount() -> None:
    """The cashflow is the value at maturity, not at valuation."""
    long_dated = EuropeanOption(underlier="ACME", option_type="call", strike=100.0, maturity=30.0)
    np.testing.assert_array_equal(payoff(long_dated, TERMINAL), payoff(CALL, TERMINAL))


def test_asset_is_recognized_but_has_no_payoff() -> None:
    with pytest.raises(UnsupportedInstrumentError) as excinfo:
        payoff(Asset(identifier="ACME", asset_class="equity"), 100.0)
    message = str(excinfo.value)
    assert "recognized" in message
    assert "no payoff" in message


def test_unknown_type_message_differs_from_recognized_without_payoff() -> None:
    class Barrier:
        pass

    with pytest.raises(UnsupportedInstrumentError) as excinfo:
        payoff(Barrier(), 100.0)  # type: ignore[arg-type]
    assert "not an instrument type known" in str(excinfo.value)


# --- payoff requirements -----------------------------------------------------


@pytest.mark.parametrize("instrument", CONTRACTS, ids=IDS)
def test_terminal_requirement_is_declared(instrument: object) -> None:
    assert payoff_requirement(instrument) is PayoffRequirement.TERMINAL
    assert payoff_requirement(type(instrument)) is PayoffRequirement.TERMINAL


def test_asset_declares_no_payoff_requirement() -> None:
    assert payoff_requirement(Asset) is None


# --- native namespaces --------------------------------------------------------


@pytest.mark.parametrize("instrument", CONTRACTS, ids=IDS)
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_numpy_dtype_is_preserved(instrument: object, dtype_name: str) -> None:
    terminal = TERMINAL.astype(dtype_name)
    assert payoff(instrument, terminal).dtype == terminal.dtype


@pytest.mark.parametrize("instrument", CONTRACTS, ids=IDS)
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_torch_type_dtype_and_device_are_preserved(instrument: object, dtype_name: str) -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    terminal = torch.tensor(TERMINAL.tolist(), dtype=getattr(torch, dtype_name))
    result = payoff(instrument, terminal)
    assert torch.is_tensor(result)
    assert result.dtype == terminal.dtype
    assert result.device == terminal.device
    np.testing.assert_allclose(
        result.detach().cpu().numpy(),
        payoff(instrument, TERMINAL.astype(dtype_name)),
        rtol=1e-6,
    )


@pytest.mark.parametrize("instrument", CONTRACTS, ids=IDS)
def test_torch_cuda_device_is_preserved(instrument: object) -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    terminal = torch.tensor(TERMINAL.tolist(), dtype=torch.float64, device="cuda")
    result = payoff(instrument, terminal)
    assert result.device.type == "cuda"


@pytest.mark.parametrize("instrument", CONTRACTS, ids=IDS)
def test_torch_gradients_flow_away_from_the_kink(instrument: object) -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    terminal = torch.tensor([80.0, 125.0], dtype=torch.float64, requires_grad=True)
    result = payoff(instrument, terminal)
    assert result.requires_grad
    result.sum().backward()
    assert terminal.grad is not None
    assert torch.isfinite(terminal.grad).all()


def test_torch_option_gradient_is_the_exercise_indicator() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    terminal = torch.tensor([80.0, 125.0], dtype=torch.float64, requires_grad=True)
    payoff(CALL, terminal).sum().backward()
    torch.testing.assert_close(terminal.grad, torch.tensor([0.0, 1.0], dtype=torch.float64))


def test_torch_gradcheck_on_the_smooth_region() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")

    terminal = torch.tensor([125.0, 150.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda s: payoff(CALL, s), (terminal,))


@pytest.mark.parametrize("instrument", CONTRACTS, ids=IDS)
def test_jax_type_and_gradients_are_preserved(instrument: object) -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jnp = jax.numpy

    terminal = jnp.asarray(TERMINAL)
    result = payoff(instrument, terminal)
    assert isinstance(result, jax.Array)
    np.testing.assert_allclose(np.asarray(result), payoff(instrument, TERMINAL), rtol=1e-6)

    grad = jax.grad(lambda s: payoff(instrument, s).sum())(jnp.asarray([80.0, 125.0]))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_jax_jit_traces_the_payoff() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jnp = jax.numpy

    jitted = jax.jit(lambda s: payoff(CALL, s))
    np.testing.assert_allclose(np.asarray(jitted(jnp.asarray(TERMINAL))), payoff(CALL, TERMINAL))


@pytest.mark.parametrize("instrument", CONTRACTS, ids=IDS)
def test_no_host_staging_in_the_payoff_path(
    instrument: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staging through host memory would silently drop the tape and the device."""
    torch = pytest.importorskip("torch", reason="torch not installed")

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("payoff evaluation must not stage through host memory")

    for name in ("numpy", "detach", "cpu", "item", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, forbidden, raising=True)

    terminal = torch.tensor(TERMINAL.tolist(), dtype=torch.float64, requires_grad=True)
    result = payoff(instrument, terminal)
    assert result.requires_grad


# --- digital payoffs ----------------------------------------------------------

from fast_vollib.instruments import BinaryOption  # noqa: E402

DIGITAL_CALL = BinaryOption(
    underlier="ACME", option_type="call", strike=100.0, maturity=1.0, cash_amount=10.0
)
DIGITAL_PUT = BinaryOption(
    underlier="ACME", option_type="put", strike=100.0, maturity=1.0, cash_amount=10.0
)


def test_digital_pays_the_cash_amount_on_the_right_side_of_the_strike() -> None:
    np.testing.assert_array_equal(payoff(DIGITAL_CALL, TERMINAL), np.array([0.0, 0.0, 10.0]))
    np.testing.assert_array_equal(payoff(DIGITAL_PUT, TERMINAL), np.array([10.0, 0.0, 0.0]))


def test_at_the_strike_exactly_a_digital_call_and_put_both_pay_nothing() -> None:
    """Strict on both sides: neither pays at a level the market has not resolved."""
    at_strike = np.array([100.0])
    assert float(payoff(DIGITAL_CALL, at_strike)[0]) == 0.0
    assert float(payoff(DIGITAL_PUT, at_strike)[0]) == 0.0


def test_a_digital_call_and_put_never_both_pay() -> None:
    grid = np.linspace(50.0, 150.0, 401)
    both = payoff(DIGITAL_CALL, grid) * payoff(DIGITAL_PUT, grid)
    assert np.all(both == 0.0)


def test_digital_notional_scales_the_cash_amount() -> None:
    short = BinaryOption(
        underlier="ACME",
        option_type="call",
        strike=100.0,
        maturity=1.0,
        cash_amount=10.0,
        notional=-2.5,
    )
    np.testing.assert_array_equal(payoff(short, TERMINAL), np.array([-0.0, -0.0, -25.0]))


def test_digital_declares_a_terminal_requirement() -> None:
    assert payoff_requirement(DIGITAL_CALL) is PayoffRequirement.TERMINAL


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_digital_preserves_the_numpy_dtype(dtype_name: str) -> None:
    """A digital selects between constants; the constants must not promote."""
    terminal = TERMINAL.astype(dtype_name)
    assert payoff(DIGITAL_CALL, terminal).dtype == terminal.dtype


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_digital_preserves_torch_dtype_and_device(dtype_name: str) -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    terminal = torch.tensor(TERMINAL.tolist(), dtype=getattr(torch, dtype_name))
    result = payoff(DIGITAL_CALL, terminal)
    assert result.dtype == terminal.dtype
    assert result.device == terminal.device
    np.testing.assert_allclose(
        result.detach().cpu().numpy(), payoff(DIGITAL_CALL, TERMINAL.astype(dtype_name))
    )


def test_digital_keeps_the_graph_and_differentiates_to_zero() -> None:
    """The pathwise derivative of an indicator is zero, not undefined-and-absent.

    Selecting between two bare constants would detach the result and the
    gradient would come back as ``None``, which reads like "no answer" rather
    than the correct answer of exactly zero.
    """
    torch = pytest.importorskip("torch", reason="torch not installed")
    terminal = torch.tensor([80.0, 125.0], dtype=torch.float64, requires_grad=True)
    result = payoff(DIGITAL_CALL, terminal)
    assert result.requires_grad
    result.sum().backward()
    assert terminal.grad is not None
    torch.testing.assert_close(terminal.grad, torch.zeros(2, dtype=torch.float64))


def test_digital_jax_value_and_zero_gradient() -> None:
    jax = pytest.importorskip("jax", reason="jax not installed")
    jnp = jax.numpy
    terminal = jnp.asarray(TERMINAL)
    np.testing.assert_allclose(
        np.asarray(payoff(DIGITAL_CALL, terminal)), payoff(DIGITAL_CALL, TERMINAL)
    )
    gradient = jax.grad(lambda s: payoff(DIGITAL_CALL, s).sum())(jnp.asarray([80.0, 125.0]))
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros(2))


def test_digital_does_not_stage_through_host_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("payoff evaluation must not stage through host memory")

    for name in ("numpy", "detach", "cpu", "item", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, forbidden, raising=True)
    terminal = torch.tensor(TERMINAL.tolist(), dtype=torch.float64, requires_grad=True)
    assert payoff(DIGITAL_CALL, terminal).requires_grad
