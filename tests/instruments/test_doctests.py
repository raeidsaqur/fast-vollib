"""Every doctest in the instruments package is executed, not just written."""

from __future__ import annotations

import doctest
import importlib
import pkgutil
from types import ModuleType

import pytest

import fast_vollib.instruments as instruments


def _package_modules() -> list[ModuleType]:
    modules = [instruments]
    for info in pkgutil.iter_modules(instruments.__path__):
        modules.append(importlib.import_module(f"{instruments.__name__}.{info.name}"))
    return modules


@pytest.mark.parametrize("module", _package_modules(), ids=lambda m: m.__name__)
def test_module_doctests(module: ModuleType) -> None:
    results = doctest.testmod(module, verbose=False, raise_on_error=True)
    del results  # raise_on_error turns any failure into an exception
