from __future__ import annotations

import importlib

import pytest

from fast_vollib import patch_py_vollib


def test_patch_py_vollib_smoke() -> None:
    pytest.importorskip("py_vollib")
    patch_py_vollib()
    black = importlib.import_module("py_vollib.black")
    black_scholes = importlib.import_module("py_vollib.black_scholes")
    black_scholes_merton = importlib.import_module("py_vollib.black_scholes_merton")

    assert callable(black.black)
    assert callable(black_scholes.black_scholes)
    assert callable(black_scholes_merton.black_scholes_merton)
