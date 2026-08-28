"""The ``surface._xp`` alias must keep resolving to the promoted namespace."""

from __future__ import annotations

import numpy as np

from fast_vollib import _array_api
from fast_vollib.surface import _xp


def test_shim_reexports_are_the_same_objects() -> None:
    assert _xp.ArrayNS is _array_api.ArrayNS
    assert _xp.get_namespace is _array_api.get_namespace
    assert _xp.numpy_namespace is _array_api.numpy_namespace


def test_shim_namespace_is_usable() -> None:
    ns = _xp.get_namespace(np.zeros(3))
    assert ns.name == "numpy"
    assert ns is _array_api.numpy_namespace()


# --- the shared operations the simulation layer is written against ------------
#
# Each is checked against numpy's answer in every installed namespace, because
# their spellings diverge (``ddof`` vs ``correction``, ``axis`` vs ``dim``) and
# a silent disagreement would show up as a wrong standard error rather than an
# exception.

import pytest  # noqa: E402

GRID = np.array([[1.0, 4.0, 2.0, 8.0], [3.0, 1.0, 5.0, 2.0], [7.0, 6.0, 9.0, 4.0]])


def namespaces() -> list[tuple[str, object]]:
    available: list[tuple[str, object]] = [("numpy", GRID)]
    try:
        import torch

        available.append(("torch", torch.tensor(GRID, dtype=torch.float64)))
    except ImportError:  # pragma: no cover - torch is optional
        pass
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        available.append(("jax", jax.numpy.asarray(GRID, dtype=jax.numpy.float64)))
    except ImportError:  # pragma: no cover - jax is optional
        pass
    return available


NAMESPACE_CASES = namespaces()
NAMESPACE_IDS = [name for name, _values in NAMESPACE_CASES]


@pytest.mark.parametrize(("name", "values"), NAMESPACE_CASES, ids=NAMESPACE_IDS)
def test_reductions_match_numpy(name: str, values: object) -> None:
    ns = _array_api.get_namespace(values)
    assert ns.name == name
    for axis in (None, 0, 1):
        np.testing.assert_allclose(np.asarray(ns.mean(values, axis=axis)), GRID.mean(axis=axis))
        np.testing.assert_allclose(np.asarray(ns.amax(values, axis=axis)), GRID.max(axis=axis))
        np.testing.assert_allclose(np.asarray(ns.amin(values, axis=axis)), GRID.min(axis=axis))
        for ddof in (0, 1):
            np.testing.assert_allclose(
                np.asarray(ns.std(values, axis=axis, ddof=ddof)),
                np.std(GRID, axis=axis, ddof=ddof),
            )


@pytest.mark.parametrize(("name", "values"), NAMESPACE_CASES, ids=NAMESPACE_IDS)
def test_cumsum_stack_and_concatenate_match_numpy(name: str, values: object) -> None:
    ns = _array_api.get_namespace(values)
    np.testing.assert_allclose(np.asarray(ns.cumsum(values, axis=1)), np.cumsum(GRID, axis=1))
    np.testing.assert_allclose(np.asarray(ns.cumsum(values, axis=0)), np.cumsum(GRID, axis=0))
    np.testing.assert_allclose(
        np.asarray(ns.concatenate((values, values), axis=1)),
        np.concatenate((GRID, GRID), axis=1),
    )
    np.testing.assert_allclose(
        np.asarray(ns.stack((values, values), axis=0)), np.stack((GRID, GRID), axis=0)
    )


@pytest.mark.parametrize(("name", "values"), NAMESPACE_CASES, ids=NAMESPACE_IDS)
def test_scalar_carries_the_reference_dtype(name: str, values: object) -> None:
    ns = _array_api.get_namespace(values)
    assert ns.scalar(2.5, like=values).dtype == values.dtype  # type: ignore[attr-defined]
    np.testing.assert_allclose(float(np.asarray(ns.scalar(2.5, like=values))), 2.5)


def test_scalar_does_not_promote_a_float32_input() -> None:
    """``asarray`` normalizes to float64; ``scalar`` must not."""
    single = GRID.astype(np.float32)
    ns = _array_api.get_namespace(single)
    assert ns.scalar(1.0, like=single).dtype == np.float32
    assert ns.asarray(1.0, like=single).dtype == np.float64


def test_scalar_without_a_reference_is_double_precision() -> None:
    ns = _array_api.numpy_namespace()
    assert ns.scalar(1.0).dtype == np.float64


def test_torch_scalar_keeps_the_reference_device() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    reference = torch.zeros(3, dtype=torch.float32)
    built = _array_api.get_namespace(reference).scalar(1.0, like=reference)
    assert built.dtype == torch.float32
    assert built.device == reference.device


def test_reductions_preserve_the_torch_tape() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    values = torch.tensor(GRID, dtype=torch.float64, requires_grad=True)
    ns = _array_api.get_namespace(values)
    total = ns.mean(ns.cumsum(values, axis=1), axis=None) + ns.std(values, ddof=1)
    assert total.requires_grad
    total.backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
