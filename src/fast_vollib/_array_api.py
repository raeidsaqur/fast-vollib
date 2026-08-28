"""Backend-neutral array namespace shared across the library.

Several parts of fast-vollib need to run the *same* math on numpy arrays,
torch tensors, and jax arrays while preserving the caller's dtype, device, and
autograd tape: the arbitrage conditions in :mod:`fast_vollib.surface` are
elementwise + reduction operations on a ``(Nk, Nt)`` total-variance / price
grid, and the terminal payoffs in :mod:`fast_vollib.instruments` map a terminal
state to a cashflow.  Both are written against the small :class:`ArrayNS`
adapter defined here rather than calling ``numpy`` directly.

Why an adapter rather than ``backends.get_module(...)``?  The library's
inference backends (``torch_backend.price_black`` et al.) intentionally move
data host<->device and return ``np.ndarray`` -- they accelerate evaluation but
**break the autograd tape**.  The surface penalty needs gradients to flow back
to the input IV tensor, so it re-derives the normalized-Black price in pure,
tape-preserving ops.  Correctness of that re-derivation is pinned to
``models.fast_black`` in the test-suite oracle.  Payoff evaluation has the same
requirement for the same reason.

The adapter normalizes only the handful of operations whose spelling differs
across numpy / torch / jax (``clip`` vs ``clamp``, ``axis`` vs ``dim``,
``nanmax`` availability, the normal CDF).  Everything else is the array
module's own attribute access.

Backend modules (torch, jax) are imported lazily inside
:func:`get_namespace`, so importing this module pulls in nothing beyond numpy
and scipy.
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

    def scalar(self, value, like=None):
        """``value`` as a 0-d array carrying ``like``'s dtype and device.

        Distinct from :meth:`asarray`, which normalizes to the backend's own
        default precision. A payoff selecting between constants needs the
        caller's dtype instead, or a float32 path silently returns float64.
        """
        raise NotImplementedError  # pragma: no cover

    def cumsum(self, x, axis=None):
        raise NotImplementedError  # pragma: no cover

    def concatenate(self, arrays, axis=0):
        raise NotImplementedError  # pragma: no cover

    def stack(self, arrays, axis=0):
        raise NotImplementedError  # pragma: no cover

    def mean(self, x, axis=None):
        raise NotImplementedError  # pragma: no cover

    def std(self, x, axis=None, ddof=0):
        """Standard deviation with an explicit degrees-of-freedom correction.

        Spelled ``ddof`` in numpy and jax and ``correction`` in torch; the
        estimator in :mod:`fast_vollib.simulation` needs ``ddof=1`` and must
        not depend on which backend it happens to be running in.
        """
        raise NotImplementedError  # pragma: no cover

    def amax(self, x, axis=None):
        raise NotImplementedError  # pragma: no cover

    def amin(self, x, axis=None):
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

    def scalar(self, value, like=None):
        dtype = getattr(like, "dtype", None)
        if dtype is not None and np.issubdtype(dtype, np.floating):
            return np.asarray(value, dtype=dtype)
        return np.asarray(value, dtype=np.float64)

    def cumsum(self, x, axis=None):
        return np.cumsum(x, axis=axis)

    def concatenate(self, arrays, axis=0):
        return np.concatenate(tuple(arrays), axis=axis)

    def stack(self, arrays, axis=0):
        return np.stack(tuple(arrays), axis=axis)

    def mean(self, x, axis=None):
        return np.mean(x, axis=axis)

    def std(self, x, axis=None, ddof=0):
        return np.std(x, axis=axis, ddof=ddof)

    def amax(self, x, axis=None):
        return np.amax(x, axis=axis)

    def amin(self, x, axis=None):
        return np.amin(x, axis=axis)

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

    def scalar(self, value, like=None):
        if like is not None and self._m.is_tensor(like):
            return self._m.as_tensor(value, dtype=like.dtype, device=like.device)
        return self._m.as_tensor(value, dtype=self._m.float64)

    def cumsum(self, x, axis=None):
        if axis is None:
            return self._m.cumsum(x.reshape(-1), dim=0)
        return self._m.cumsum(x, dim=axis)

    def concatenate(self, arrays, axis=0):
        return self._m.cat(tuple(arrays), dim=axis)

    def stack(self, arrays, axis=0):
        return self._m.stack(tuple(arrays), dim=axis)

    def mean(self, x, axis=None):
        return self._m.mean(x) if axis is None else self._m.mean(x, dim=axis)

    def std(self, x, axis=None, ddof=0):
        if axis is None:
            return self._m.std(x, correction=ddof)
        return self._m.std(x, dim=axis, correction=ddof)

    def amax(self, x, axis=None):
        return self._m.amax(x) if axis is None else self._m.amax(x, dim=axis)

    def amin(self, x, axis=None):
        return self._m.amin(x) if axis is None else self._m.amin(x, dim=axis)

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

    def scalar(self, value, like=None):
        dtype = getattr(like, "dtype", None)
        if dtype is not None and self._m.issubdtype(dtype, self._m.floating):
            return self._m.asarray(value, dtype=dtype)
        return self._m.asarray(value)

    def cumsum(self, x, axis=None):
        return self._m.cumsum(x, axis=axis)

    def concatenate(self, arrays, axis=0):
        return self._m.concatenate(tuple(arrays), axis=axis)

    def stack(self, arrays, axis=0):
        return self._m.stack(tuple(arrays), axis=axis)

    def mean(self, x, axis=None):
        return self._m.mean(x, axis=axis)

    def std(self, x, axis=None, ddof=0):
        return self._m.std(x, axis=axis, ddof=ddof)

    def amax(self, x, axis=None):
        return self._m.amax(x, axis=axis)

    def amin(self, x, axis=None):
        return self._m.amin(x, axis=axis)

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


def concrete_float(value: Any) -> float | None:
    """``value`` as a Python ``float``, or ``None`` if it cannot be read eagerly.

    Validation has two halves that behave differently under a JAX
    transformation.  Static properties -- shape, rank, dtype -- are known while
    tracing and can always be checked.  A *value* is not: a tracer has no
    number in it, and asking for one raises rather than returning something
    wrong.  Code that validated by calling ``float(x)`` unconditionally would
    therefore break ``jax.grad`` on a function that was otherwise fine.

    So value checks call this first and skip themselves when it returns
    ``None``.  Concrete calls -- which is every ordinary one -- still get the
    full check; a traced call carries the domain precondition instead, because
    a library cannot raise a Python exception from inside a trace.

    torch tensors are read with an explicit synchronization: the number is used
    only to decide whether to raise, and never enters the returned graph.
    """
    if value is None:
        return None
    if _is_jax_array(value):
        import jax

        try:
            return float(value)
        except jax.errors.ConcretizationTypeError:
            return None
        except (TypeError, ValueError):  # pragma: no cover - non-scalar array
            return None
    if _is_torch_tensor(value):
        try:
            return float(value.detach().reshape(()))
        except (RuntimeError, TypeError, ValueError):
            return None
    try:
        as_array = np.asarray(value)
    except (TypeError, ValueError):  # pragma: no cover - not array-like at all
        return None
    numeric = np.issubdtype(as_array.dtype, np.number) or np.issubdtype(as_array.dtype, np.bool_)
    if as_array.size != 1 or not numeric:
        return None
    return float(as_array.reshape(()))


def concrete_bool(value: Any) -> bool | None:
    """``value`` as a Python ``bool``, or ``None`` under a JAX trace.

    The boolean counterpart of :func:`concrete_float`, used for domain checks
    that reduce an array to one flag -- "are all of these states positive?" --
    before a payoff that would otherwise return a silent ``NaN``.
    """
    as_float = concrete_float(value)
    return None if as_float is None else bool(as_float)
