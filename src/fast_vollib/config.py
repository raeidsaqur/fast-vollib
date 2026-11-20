from __future__ import annotations

import importlib.util
import os

from .types import BackendLiteral

_BACKEND_OVERRIDE: BackendLiteral | None = None


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _torch_cuda_available() -> bool:
    if not _has_module("torch"):
        return False
    import torch

    return bool(torch.cuda.is_available())


def _jax_available() -> bool:
    return _has_module("jax")


_VALID_BACKENDS = frozenset({"auto", "numpy", "torch", "jax", "numba"})


def _validated(name: str, source: str) -> BackendLiteral:
    if name not in _VALID_BACKENDS:
        valid = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(f"Unsupported backend {name!r} from {source}. Valid backends: {valid}.")
    return name  # type: ignore[return-value]


def set_backend(name: BackendLiteral) -> None:
    """Set the process-wide backend used wherever ``backend="auto"`` (the default) is passed.

    ``set_backend("auto")`` clears the override.
    """
    global _BACKEND_OVERRIDE
    _BACKEND_OVERRIDE = _validated(name, "set_backend()")


def get_backend(explicit: BackendLiteral | None = None) -> BackendLiteral:
    """Resolve the backend for one call.

    Precedence: an explicit ``backend=`` other than ``"auto"``, then :func:`set_backend`, then
    the ``FAST_VOLLIB_BACKEND`` environment variable, then auto-detection (torch when CUDA is
    available, else JAX when installed, else NumPy). ``"auto"`` at any level defers to the next.
    """
    sources = (
        (explicit, "the backend= argument"),
        (_BACKEND_OVERRIDE, "set_backend()"),
        (os.getenv("FAST_VOLLIB_BACKEND"), "the FAST_VOLLIB_BACKEND environment variable"),
    )
    for choice, source in sources:
        if choice is not None and choice != "":
            resolved = _validated(choice, source)
            if resolved != "auto":
                return resolved
    if _torch_cuda_available():
        return "torch"
    if _jax_available():
        return "jax"
    return "numpy"
