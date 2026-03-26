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


def set_backend(name: BackendLiteral) -> None:
    global _BACKEND_OVERRIDE
    if name not in {"auto", "numpy", "torch", "jax"}:
        raise ValueError(f"Unsupported backend: {name}")
    _BACKEND_OVERRIDE = name


def get_backend(explicit: BackendLiteral | None = None) -> BackendLiteral:
    choice = explicit or _BACKEND_OVERRIDE or os.getenv("FAST_VOLLIB_BACKEND", "auto")
    if choice != "auto":
        return choice
    if _torch_cuda_available():
        return "torch"
    if _jax_available():
        return "jax"
    return "numpy"
