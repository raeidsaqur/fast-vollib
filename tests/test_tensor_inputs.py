"""Tests for torch tensor inputs to to_numpy() and fast_implied_volatility().

Regression tests for the bug where np.asarray(tensor) called Tensor.__array__()
which called .numpy() — illegal for CUDA tensors and CPU tensors with
requires_grad=True.

Note on what this fix does and does not provide:
  - Torch/CUDA tensors are accepted as inputs without crashing.
  - .detach() is called before conversion, so gradients do NOT flow through
    fast_implied_volatility.  The function returns numpy arrays (or a native
    tensor when return_native=True), not an autograd-connected tensor.
  - Compute still round-trips through host numpy: GPU → CPU → numpy → GPU.
    If a fully GPU-resident differentiable IV solver is needed, that is a
    separate feature request.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from fast_vollib.implied_volatility import fast_implied_volatility  # noqa: E402
from fast_vollib.utils.broadcast import to_numpy  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PRICE = 0.10
_S = 30.0
_K = 30.0
_T = 10 / 365.0
_R = 0.05
_FLAG = "c"

_PRICE_T = torch.tensor([_PRICE], dtype=torch.float64)
_S_T = torch.tensor([_S], dtype=torch.float64)
_K_T = torch.tensor([_K], dtype=torch.float64)
_T_T = torch.tensor([_T], dtype=torch.float64)
_R_T = torch.tensor([_R], dtype=torch.float64)


# ---------------------------------------------------------------------------
# to_numpy — unit tests
# ---------------------------------------------------------------------------


def test_to_numpy_cpu_tensor_plain() -> None:
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    result = to_numpy(t)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_to_numpy_cpu_tensor_requires_grad() -> None:
    """Regression: np.asarray(cpu_grad_tensor) raises RuntimeError before fix."""
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    result = to_numpy(t)
    np.testing.assert_array_almost_equal(result, np.array([1.0, 2.0, 3.0]))
    assert result.dtype == np.float64


def test_to_numpy_cpu_tensor_after_grad_op() -> None:
    """Tensor produced by a differentiable op (leaf=False, grad_fn set)."""
    base = torch.tensor([2.0, 4.0], dtype=torch.float64, requires_grad=True)
    t = base * 2  # non-leaf tensor; has grad_fn
    result = to_numpy(t)
    np.testing.assert_array_equal(result, np.array([4.0, 8.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_to_numpy_cuda_tensor() -> None:
    """Regression: np.asarray(cuda_tensor) raises TypeError before fix."""
    t = torch.tensor([1.0, 2.0], dtype=torch.float64).cuda()
    result = to_numpy(t)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_to_numpy_cuda_tensor_requires_grad() -> None:
    t = torch.tensor([3.0, 6.0], dtype=torch.float64, requires_grad=True).cuda()
    result = to_numpy(t)
    np.testing.assert_array_almost_equal(result, np.array([3.0, 6.0]))


# ---------------------------------------------------------------------------
# fast_implied_volatility — end-to-end
# ---------------------------------------------------------------------------


def test_fast_iv_cpu_tensor_requires_grad() -> None:
    """Regression: CUDA-class crash reproduced on CPU via requires_grad=True."""
    price = torch.tensor([_PRICE], dtype=torch.float64, requires_grad=True)
    S = torch.tensor([_S], dtype=torch.float64, requires_grad=True)
    K = torch.tensor([_K], dtype=torch.float64)
    t = torch.tensor([_T], dtype=torch.float64)
    r = torch.tensor([_R], dtype=torch.float64)

    iv = fast_implied_volatility(price=price, S=S, K=K, t=t, r=r, flag=_FLAG, return_as="numpy")
    assert iv.shape == (1,)
    assert np.isfinite(iv[0])
    assert 0.0 < iv[0] < 5.0


def test_fast_iv_cpu_tensor_backend_torch() -> None:
    """Full round-trip: torch tensor inputs, torch backend, numpy output."""
    from fast_vollib.backends import available_backends

    if "torch" not in available_backends():
        pytest.skip("torch backend not available")

    price = torch.tensor([_PRICE], dtype=torch.float64)
    S = torch.tensor([_S], dtype=torch.float64)
    K = torch.tensor([_K], dtype=torch.float64)
    t = torch.tensor([_T], dtype=torch.float64)
    r = torch.tensor([_R], dtype=torch.float64)

    iv = fast_implied_volatility(
        price=price,
        S=S,
        K=K,
        t=t,
        r=r,
        flag=_FLAG,
        backend="torch",
        return_as="numpy",
    )
    assert np.isfinite(iv[0])

    # Result must match the numpy reference within 1 bps
    iv_ref = fast_implied_volatility(
        price=_PRICE, S=_S, K=_K, t=_T, r=_R, flag=_FLAG, return_as="numpy"
    )
    np.testing.assert_allclose(iv, iv_ref, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fast_iv_cuda_tensor() -> None:
    """Regression: the exact crash reported — CUDA tensor inputs."""
    from fast_vollib.backends import available_backends

    if "torch" not in available_backends():
        pytest.skip("torch backend not available")

    price = _PRICE_T.cuda()
    S = _S_T.cuda()
    K = _K_T.cuda()
    t = _T_T.cuda()
    r = _R_T.cuda()

    iv = fast_implied_volatility(
        price=price,
        S=S,
        K=K,
        t=t,
        r=r,
        flag=_FLAG,
        backend="torch",
        return_as="numpy",
    )
    assert np.isfinite(iv[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fast_iv_cuda_tensor_mixed_cpu_cuda() -> None:
    """Mixed CPU/CUDA inputs are each individually brought to host."""
    price = _PRICE_T.cuda()
    S = _S_T  # CPU
    K = _K_T.cuda()
    t = _T_T  # CPU
    r = _R_T  # CPU

    iv = fast_implied_volatility(price=price, S=S, K=K, t=t, r=r, flag=_FLAG, return_as="numpy")
    assert np.isfinite(iv[0])
