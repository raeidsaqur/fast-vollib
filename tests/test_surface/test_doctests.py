"""Every doctest in the surface package is executed."""

from __future__ import annotations

import doctest
import importlib
import pkgutil
from types import ModuleType

import pytest

import fast_vollib.surface as surface


def _package_modules() -> list[ModuleType]:
    modules: list[ModuleType] = [surface]
    for info in pkgutil.walk_packages(surface.__path__, prefix=f"{surface.__name__}."):
        modules.append(importlib.import_module(info.name))
    return modules


@pytest.mark.parametrize("module", _package_modules(), ids=lambda m: m.__name__)
def test_module_doctests(module: ModuleType) -> None:
    results = doctest.testmod(module, verbose=False, raise_on_error=True)
    del results  # raise_on_error turns any failure into an exception
