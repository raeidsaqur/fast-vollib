"""Backend-neutral array namespace for the surface eval harness.

The arbitrage conditions (§4 of the design) are stated as a handful of
elementwise + reduction operations on a ``(Nk, Nt)`` total-variance / price
grid.  To support both the **offline numpy report path** and the
**differentiable torch/jax penalty path** from a *single* math implementation,
every kernel in :mod:`fast_vollib.surface` is written against the small
:class:`ArrayNS` adapter defined here rather than calling ``numpy`` directly.

Why an adapter rather than ``backends.get_module(...)``?  The library's
inference backends (``torch_backend.price_black`` et al.) intentionally move
data host↔device and return ``np.ndarray`` — they accelerate evaluation but
**break the autograd tape**.  The surface penalty needs gradients to flow back
to the input IV tensor, so it re-derives the normalized-Black price in pure,
tape-preserving ops.  Correctness of that re-derivation is pinned to
``models.fast_black`` in the test-suite oracle.

The adapter normalizes only the handful of operations whose spelling differs
across numpy / torch / jax (``clip`` vs ``clamp``, ``axis`` vs ``dim``,
``nanmax`` availability, the normal CDF).  Everything else is the array
module's own attribute access.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


class ArrayNS:
    """Thin namespace adapter over a single array backend.

    Instances are cheap and stateless; obtain one via :func:`get_namespace`.
    Only the operations used by the surface kernels are exposed.  Methods that
    are spelled identically across numpy/torch/jax are forwarded to the
    underlying module via ``__getattr__``; the divergent few are overridden.
    """

    name: str

    def __init__(self, module: Any, name: str):
        self._m = module
        self.name = name

    # -- generic passthrough (exp, log, sqrt, abs, sign, where, maximum, ...) --
    def __getattr__(self, attr: str) -> Any:
        return getattr(self._m, attr)

    # -- diverging spellings -------------------------------------------------
    def clip(self, x, lo, hi):  # numpy: clip(x, lo, hi); torch: clamp
        raise NotImplementedError  # pragma: no cover - overridden per backend

    def sum(self, x, axis=None):
        raise NotImplementedError  # pragma: no cover

    def nansum(self, x, axis=None):
        raise NotImplementedError  # pragma: no cover

    def nanmax(self, x, axis=None):
        raise NotImplementedError  # pragma: no cover

    def any(self, x, axis=None):
        raise NotImplementedError  # pragma: no cover

    def normcdf(self, x):
        raise NotImplementedError  # pragma: no cover

    def relu(self, x):
        return self._m.maximum(x, self.asarray(0.0, like=x))

    def asarray(self, x, like=None):
        raise NotImplementedError  # pragma: no cover

    def zeros(self, shape, like=None):
        """A zero array of ``shape`` in this backend (dtype/device of ``like``)."""
        raise NotImplementedError  # pragma: no cover

    def to_numpy(self, x) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover

    def is_native(self, x) -> bool:
        raise NotImplementedError  # pragma: no cover


class _NumpyNS(ArrayNS):
    def __init__(self):
        super().__init__(np, "numpy")
        from scipy.special import ndtr

        self._ndtr = ndtr

    def clip(self, x, lo, hi):
        return np.clip(x, lo, hi)

    def sum(self, x, axis=None):
        return np.sum(x, axis=axis)

    def nansum(self, x, axis=None):
        return np.nansum(x, axis=axis)

    def nanmax(self, x, axis=None):
        return np.nanmax(x, axis=axis)

    def any(self, x, axis=None):
        return np.any(x, axis=axis)

    def normcdf(self, x):
        return self._ndtr(x)

    def asarray(self, x, like=None):
        return np.asarray(x, dtype=np.float64)

    def zeros(self, shape, like=None):
        return np.zeros(shape, dtype=np.float64)

    def to_numpy(self, x) -> np.ndarray:
        return np.asarray(x)

    def is_native(self, x) -> bool:
        return isinstance(x, np.ndarray)


class _TorchNS(ArrayNS):
    def __init__(self, torch_module):
        super().__init__(torch_module, "torch")

    def clip(self, x, lo, hi):
        return self._m.clamp(x, lo, hi)

    def sum(self, x, axis=None):
        return self._m.sum(x) if axis is None else self._m.sum(x, dim=axis)

    def nansum(self, x, axis=None):
        return self._m.nansum(x) if axis is None else self._m.nansum(x, dim=axis)

    def nanmax(self, x, axis=None):
        # torch has no nanmax; replace NaN with -inf then reduce.
        neg_inf = self._m.tensor(float("-inf"), dtype=x.dtype, device=x.device)
        filled = self._m.where(self._m.isnan(x), neg_inf, x)
        if axis is None:
            return self._m.max(filled)
        return self._m.max(filled, dim=axis).values

    def any(self, x, axis=None):
        return self._m.any(x) if axis is None else self._m.any(x, dim=axis)

    def normcdf(self, x):
        return 0.5 * self._m.erfc(-x * _INV_SQRT2)

    def asarray(self, x, like=None):
        if like is not None and self._m.is_tensor(like):
            return self._m.as_tensor(x, dtype=like.dtype, device=like.device)
        return self._m.as_tensor(x, dtype=self._m.float64)

    def zeros(self, shape, like=None):
        if like is not None and self._m.is_tensor(like):
            return self._m.zeros(shape, dtype=like.dtype, device=like.device)
        return self._m.zeros(shape, dtype=self._m.float64)

    def to_numpy(self, x) -> np.ndarray:
        if self._m.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def is_native(self, x) -> bool:
        return self._m.is_tensor(x)


class _JaxNS(ArrayNS):
    def __init__(self, jnp_module):
        super().__init__(jnp_module, "jax")
        from jax.scipy.special import ndtr

        self._ndtr = ndtr

    def clip(self, x, lo, hi):
        return self._m.clip(x, lo, hi)

    def sum(self, x, axis=None):
        return self._m.sum(x, axis=axis)

    def nansum(self, x, axis=None):
        return self._m.nansum(x, axis=axis)

    def nanmax(self, x, axis=None):
        return self._m.nanmax(x, axis=axis)

    def any(self, x, axis=None):
        return self._m.any(x, axis=axis)

    def normcdf(self, x):
        return self._ndtr(x)

    def asarray(self, x, like=None):
        return self._m.asarray(x)

    def zeros(self, shape, like=None):
        return self._m.zeros(shape)

    def to_numpy(self, x) -> np.ndarray:
        return np.asarray(x)

    def is_native(self, x) -> bool:
        return hasattr(x, "__jax_array__") or type(x).__module__.startswith("jax")


_NUMPY_NS = _NumpyNS()


def _is_torch_tensor(x: Any) -> bool:
    mod = type(x).__module__
    return mod.startswith("torch")


def _is_jax_array(x: Any) -> bool:
    mod = type(x).__module__
    return mod.startswith("jax") or mod.startswith("jaxlib")


def get_namespace(*arrays: Any) -> ArrayNS:
    """Return the :class:`ArrayNS` matching the type of the input array(s).

    Dispatch is by the runtime type of the first non-None array argument, so
    a torch tensor selects the torch namespace (preserving its autograd tape
    and device), a jax array selects jax, and anything else falls back to
    numpy.  Mixed backends are not supported and should be normalized by the
    caller (e.g. :class:`~fast_vollib.surface.grid.IVSurface`).
    """
    for arr in arrays:
        if arr is None:
            continue
        if _is_torch_tensor(arr):
            import torch

            return _TorchNS(torch)
        if _is_jax_array(arr):
            import jax.numpy as jnp

            return _JaxNS(jnp)
        if isinstance(arr, np.ndarray):
            return _NUMPY_NS
    return _NUMPY_NS


def numpy_namespace() -> ArrayNS:
    """The numpy namespace singleton (used by the report/host path)."""
    return _NUMPY_NS
